"""Ticket #154, the Coordinator's contract-fixed half: accepting a generation request, and reporting how it ended. No device is involved."""

try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401

import io
import unittest
import urllib.error

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core import distill_contract

from ekido_coordinator import distillations
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import DistillationJob

SEEDS = [{"text": f"mera parcel {i} nahi aaya", "label": "complaint"} for i in range(30)]
TESTS = [{"text": f"held back ticket {i}", "label": "order"} for i in range(20)]


def make(seeds=SEEDS, tests=TESTS, attempt=0, **changes):
    data = distill_contract.jsonl_bytes(seeds)
    spec = {"template": distill_contract.TEMPLATE, "recipe_id": "ticket-sorter-hinglish",
            "teacher": {"id": distill_contract.TEACHER_ID, "revision": distill_contract.TEACHER_REVISION, "quantization": "4bit"},
            "seeds": {"sha256": __import__("hashlib").sha256(data).hexdigest(), "bytes": len(data), "rows": len(seeds)},
            "test_input_sha256": distill_contract.test_input_hashes("ticket-sorter-hinglish", tests), "per_seed": 20, "generation_seed": 42 + attempt,
            "max_new_tokens": 96, "attempts": 3, "filter_version": 1, "attempt": attempt}
    spec.update(changes)
    return spec, data.decode("utf-8")


class Platform:
    def __init__(self):
        self.calls, self.down = [], False

    def generated(self, distillation_id, rows_jsonl, passed, reasons):
        if self.down:
            raise RuntimeError("the Platform answered 503")
        self.calls.append(("generated", distillation_id, rows_jsonl, passed, reasons))

    def failed(self, distillation_id, reason):
        if self.down:
            raise RuntimeError("down")
        self.calls.append(("failed", distillation_id, reason))


class Base(unittest.TestCase):
    def setUp(self, enabled=True):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None, distillation_enabled=enabled))
        self.client = TestClient(self.app, client=("127.0.0.1", 50000))
        self.engine = self.app.state.engine

    def post(self, spec=None, seeds=None, distillation_id="d-1"):
        s, d = make()
        return self.client.post("/distillations", json={"distillation_id": distillation_id, "spec": spec or s, "seeds_jsonl": d if seeds is None else seeds},
                                headers={"host": "127.0.0.1"})

    def rows(self):
        with Session(self.engine) as s:
            return s.scalars(select(DistillationJob)).all()


class Accepting(Base):
    def test_a_good_request_returns_a_job_id_and_the_same_request_again_returns_the_same_one(self):
        first = self.post()
        self.assertEqual(first.status_code, 201, first.text)
        again = self.post()
        self.assertEqual((again.status_code, again.json()["job_id"]), (201, first.json()["job_id"]))
        (row,) = self.rows()
        self.assertEqual((row.state, row.attempt, row.distillation_id), ("QUEUED", 0, "d-1"))

    def test_the_regeneration_is_its_own_job_and_a_changed_spec_for_the_same_attempt_is_refused(self):
        first = self.post().json()["job_id"]
        spec, data = make(attempt=1)
        second = self.post(spec=spec, seeds=data)
        self.assertEqual(second.status_code, 201)
        self.assertNotEqual(second.json()["job_id"], first)
        changed, data = make(per_seed=3)
        self.assertEqual(self.post(spec=changed, seeds=data).status_code, 409)
        self.assertEqual(len(self.rows()), 2)

    def test_off_by_default_nothing_is_accepted_or_stored(self):
        self.assertFalse(Settings.model_fields["distillation_enabled"].default)
        self.setUp(enabled=False)
        r = self.post()
        self.assertEqual(r.status_code, 409)
        self.assertIn("not available", r.json()["detail"])
        self.assertEqual(self.rows(), [])

    def test_a_spec_that_breaks_the_contract_is_422(self):
        for name, change in {"another teacher": {"teacher": {"id": "x/y", "revision": distill_contract.TEACHER_REVISION, "quantization": "4bit"}},
                             "per_seed too big": {"per_seed": 21}, "unknown field": {"target_rows": 5}, "wrong template": {"template": "llm-lora/v1"},
                             "attempt 2": {"attempt": 2}}.items():
            with self.subTest(name):
                spec, data = make(**change)
                self.assertEqual(self.post(spec=spec, seeds=data).status_code, 422)
        self.assertEqual(self.rows(), [])

    def test_the_seeds_must_be_exactly_what_the_spec_describes(self):
        spec, data = make()
        tampered = data.replace("parcel 1 ", "parcel X ")
        extra = data + data.split("\n")[0] + "\n"
        pretty = data.replace('{"label":', '{ "label":')
        for name, seeds in {"changed bytes": tampered, "one row more": extra, "not the contract's serialisation": pretty, "not json": "{oops\n"}.items():
            with self.subTest(name):
                self.assertEqual(self.post(spec=spec, seeds=seeds).status_code, 422)
        other_columns = [{"text": "x", "wrong": "y"}]
        spec, data = make(seeds=other_columns)
        self.assertEqual(self.post(spec=spec, seeds=data).status_code, 422)
        self.assertEqual(self.rows(), [])

    def test_another_serialisation_is_refused_even_when_the_hash_matches_it(self):
        import hashlib

        spec, data = make()
        pretty = data.replace('{"label":', '{ "label":')
        spec["seeds"] = {"sha256": hashlib.sha256(pretty.encode()).hexdigest(), "bytes": len(pretty.encode()), "rows": 30}
        r = self.post(spec=spec, seeds=pretty)
        self.assertEqual(r.status_code, 422)
        self.assertIn("serialisation", r.json()["detail"])

    def test_a_seed_that_is_also_a_test_input_is_refused(self):
        spec, data = make(seeds=SEEDS[:29] + [dict(TESTS[0])])
        r = self.post(spec=spec, seeds=data)
        self.assertEqual(r.status_code, 422)
        self.assertIn("test inputs", r.json()["detail"])

    def test_an_outsider_cannot_ask(self):
        app = create_app(Settings(database_url="sqlite://", _env_file=None, distillation_enabled=True, api_token="operator-secret"))
        spec, data = make()
        r = TestClient(app).post("/distillations", json={"distillation_id": "d", "spec": spec, "seeds_jsonl": data})
        self.assertIn(r.status_code, (401, 403))

    def test_the_seed_text_is_never_logged(self):
        with self.assertLogs(level="DEBUG") as captured:
            __import__("logging").getLogger("probe").debug("start")
            self.post()
        self.assertNotIn("mera parcel", "\n".join(captured.output))


class Reporting(Base):
    def job(self):
        return self.post().json()["job_id"]

    def test_rows_that_pass_are_reported_and_the_seeds_are_dropped(self):
        job_id, platform = self.job(), Platform()
        with Session(self.engine) as s:
            distillations.finish(s, job_id, rows_jsonl='{"label":"order","text":"new"}\n', passed=True, reasons=[], platform=platform)
        (row,) = self.rows()
        self.assertEqual((row.state, row.seeds_jsonl), ("DONE", None))
        self.assertEqual(platform.calls, [("generated", "d-1", '{"label":"order","text":"new"}\n', True, [])])

    def test_rows_the_verifier_refused_are_not_sent_on_only_the_reasons(self):
        job_id, platform = self.job(), Platform()
        with Session(self.engine) as s:
            distillations.finish(s, job_id, rows_jsonl="poison\n", passed=False, reasons=["BAD_LABEL"], platform=platform)
        self.assertEqual(platform.calls, [("generated", "d-1", "", False, ["BAD_LABEL"])])
        self.assertEqual(self.rows()[0].state, "FAILED")

    def test_if_the_platform_cannot_be_told_the_job_stays_queued_and_a_retry_works_once(self):
        job_id, platform = self.job(), Platform()
        platform.down = True
        with Session(self.engine) as s:
            with self.assertRaises(RuntimeError):
                distillations.fail(s, job_id, "no device", platform=platform)
        self.assertEqual((self.rows()[0].state, self.rows()[0].seeds_jsonl is not None), ("QUEUED", True))
        platform.down = False
        for _ in range(2):
            with Session(self.engine) as s:
                distillations.fail(s, job_id, "no device", platform=platform)
        self.assertEqual(platform.calls, [("failed", "d-1", "no device")])
        self.assertEqual((self.rows()[0].state, self.rows()[0].seeds_jsonl), ("FAILED", None))  # the customer's text is gone

    def test_an_unknown_job_is_an_error(self):
        with Session(self.engine) as s, self.assertRaises(distillations.Refused):
            distillations.fail(s, "nope", "x", platform=Platform())

    def test_the_platform_client_sends_the_token_in_a_header_and_names_only_the_status(self):
        seen = []

        class Response:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, n=-1): return b"{}"

        reports = distillations.PlatformReports("http://127.0.0.1:8010/", "tok-secret", opener=lambda req, t: seen.append(req) or Response())
        reports.generated("a/b", "x\n", True, [])
        reports.failed("a/b", "r" * 500)
        self.assertEqual(seen[0].full_url, "http://127.0.0.1:8010/internal/distillations/a%2Fb/generated")  # the id is quoted
        self.assertEqual(seen[0].get_header("X-internal-token"), "tok-secret")
        self.assertNotIn("tok-secret", seen[0].full_url)
        self.assertLessEqual(len(seen[1].data), 300)

        def refuse(req, t):
            raise urllib.error.HTTPError("u", 500, "x", {}, io.BytesIO(b""))

        with self.assertRaises(RuntimeError) as caught:
            distillations.PlatformReports("http://x", "tok-secret", opener=refuse).failed("d", "r")
        self.assertIn("500", str(caught.exception))
        self.assertNotIn("tok-secret", str(caught.exception))
        with self.assertRaises(ValueError):
            distillations.PlatformReports("http://x", "")


if __name__ == "__main__":
    unittest.main()
