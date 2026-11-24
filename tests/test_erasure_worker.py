"""A14/C17: the worker that destroys job keys for customers who deleted their data."""

try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import io
import json
import logging
import unittest
import urllib.error
from decimal import Decimal

from eth_account import Account
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_coordinator import erasure_worker, jobs
from ekido_coordinator.erasure_worker import ErasureWorker, HttpPlatformClient
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import Job, JobKey, Provider, Wallet

KEY = "0x" + "a" * 64


class FakePlatform:
    def __init__(self, tasks):
        self._tasks = list(tasks)
        self.done, self.failures = [], []

    def tasks(self, limit=100):
        return [t for t in self._tasks if t["id"] not in self.done][:limit]

    def complete(self, task_id):
        self.done.append(task_id)

    def failed(self, task_id):
        self.failures.append(task_id)


class Base(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None))
        self.engine = self.app.state.engine
        with Session(self.engine) as s:
            wallet = Wallet(address=Account.from_key(KEY).address.lower(), role="provider")
            s.add(wallet)
            s.flush()
            provider = Provider(name="A", wallet_id=wallet.id, kind="REAL", price_per_step=Decimal("0.001"))
            s.add(provider)
            s.flush()
            self.provider_id = provider.id
            s.commit()
        self.n = 0

    def job(self, state="SETTLED"):
        self.n += 1
        with Session(self.engine) as s:
            job = jobs.create_job(s, spec={"n": self.n}, policy={}, budget=Decimal("5"), deadline_seconds=600)
            if state != "OPEN":
                jobs.assign(s, job, self.provider_id, {})
            job.state = state
            s.commit()
            return job.id

    def key(self, job_id):
        with Session(self.engine) as s:
            return s.scalar(select(JobKey).where(JobKey.job_id == job_id))

    def state(self, job_id):
        with Session(self.engine) as s:
            return s.get(Job, job_id).state

    def task(self, ref, id_=None, kind="job_data"):
        return {"id": id_ or f"t-{ref}", "kind": kind, "ref": ref, "attempts": 0}


class Passes(Base):
    def test_an_ended_jobs_key_is_destroyed_and_the_task_is_done(self):
        ids = [self.job(state) for state in ("SETTLED", "FAILED", "CANCELLED", "REFUNDED")]
        platform = FakePlatform([self.task(i) for i in ids])
        counts = ErasureWorker(self.engine, platform).run_once()
        self.assertEqual(counts, {"done": 4, "retry": 0, "skipped": 0})
        for job_id in ids:
            row = self.key(job_id)
            self.assertIsNone(row.wrapped_identity)
            self.assertIsNotNone(row.destroyed_at)
        self.assertEqual(sorted(platform.done), sorted(f"t-{i}" for i in ids))
        self.assertEqual(platform.failures, [])

    def test_an_unknown_job_has_nothing_to_destroy_so_its_task_is_done(self):
        platform = FakePlatform([self.task("never-existed")])
        self.assertEqual(ErasureWorker(self.engine, platform).run_once()["done"], 1)

    def test_an_open_job_is_cancelled_and_then_its_key_is_destroyed(self):
        job_id = self.job("OPEN")
        platform = FakePlatform([self.task(job_id)])
        self.assertEqual(ErasureWorker(self.engine, platform).run_once()["done"], 1)
        self.assertEqual(self.state(job_id), "CANCELLED")
        self.assertIsNone(self.key(job_id).wrapped_identity)

    def test_repeating_a_pass_changes_nothing(self):
        job_id = self.job("SETTLED")
        worker = ErasureWorker(self.engine, FakePlatform([self.task(job_id)]))
        worker.run_once()
        stamp = self.key(job_id).destroyed_at
        worker.platform = FakePlatform([self.task(job_id)])  # the Platform asks again (it had not heard back)
        self.assertEqual(worker.run_once()["done"], 1)
        self.assertEqual(self.key(job_id).destroyed_at, stamp)  # the key was not "destroyed again"


class Retries(Base):
    def test_a_job_that_is_still_running_keeps_its_key_and_the_task_is_asked_again_later(self):
        for state in ("ASSIGNED", "RUNNING", "RECOVERING", "VALIDATING", "COMPLETED"):
            with self.subTest(state=state):
                self.setUp()
                job_id = self.job(state)
                platform = FakePlatform([self.task(job_id)])
                counts = ErasureWorker(self.engine, platform).run_once()
                self.assertEqual((counts["done"], counts["retry"]), (0, 1))
                self.assertIsNotNone(self.key(job_id).wrapped_identity)
                self.assertEqual(platform.failures, [f"t-{job_id}"])
                self.assertEqual(self.state(job_id), state)  # it was not touched

    def test_the_job_that_ended_later_is_handled_on_the_next_pass(self):
        job_id = self.job("RUNNING")
        platform = FakePlatform([self.task(job_id)])
        worker = ErasureWorker(self.engine, platform)
        self.assertEqual(worker.run_once()["retry"], 1)
        with Session(self.engine) as s:
            s.get(Job, job_id).state = "SETTLED"
            s.commit()
        self.assertEqual(worker.run_once()["done"], 1)
        self.assertIsNone(self.key(job_id).wrapped_identity)

    def test_one_task_that_blows_up_is_reported_failed_and_the_others_still_run(self):
        good, bad = self.job("SETTLED"), self.job("SETTLED")
        platform = FakePlatform([self.task(bad), self.task(good)])
        real = erasure_worker.job_keys.destroy

        def flaky(session, job_id):
            if job_id == bad:
                raise RuntimeError("database hiccup")
            return real(session, job_id)

        erasure_worker.job_keys.destroy = flaky
        self.addCleanup(setattr, erasure_worker.job_keys, "destroy", real)
        with self.assertLogs("ekido_coordinator.erasure_worker", level="ERROR"):
            counts = ErasureWorker(self.engine, platform).run_once()
        self.assertEqual((counts["done"], counts["retry"]), (1, 1))
        self.assertEqual(platform.failures, [f"t-{bad}"])
        self.assertIsNotNone(self.key(bad).wrapped_identity)
        self.assertIsNone(self.key(good).wrapped_identity)

    def test_a_task_of_an_unknown_kind_is_left_alone(self):
        platform = FakePlatform([self.task("x", kind="storage_purge")])
        self.assertEqual(ErasureWorker(self.engine, platform).run_once(), {"done": 0, "retry": 0, "skipped": 1})
        self.assertEqual((platform.done, platform.failures), ([], []))


class Http(unittest.TestCase):
    def make(self, reply=None, fail=None):
        self.seen = []

        class Response:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, n=-1): return json.dumps(self.body).encode()

        def opener(request, timeout):
            self.seen.append(request)
            if fail:
                raise fail
            return Response(reply)

        return HttpPlatformClient("http://127.0.0.1:8010/", "s3cret-token", opener=opener)

    def test_requests_have_the_right_shape_and_carry_the_token_only_in_the_header(self):
        client = self.make(reply=[{"id": "t1", "kind": "job_data", "ref": "j1", "attempts": 0}])
        self.assertEqual(client.tasks(5)[0]["id"], "t1")
        client.complete("t1")
        client.failed("t2")
        listing, done, failed = self.seen
        self.assertEqual((listing.method, listing.full_url), ("GET", "http://127.0.0.1:8010/internal/erasure/tasks?limit=5"))
        self.assertEqual((done.method, done.full_url), ("POST", "http://127.0.0.1:8010/internal/erasure/tasks/t1/complete"))
        self.assertEqual(failed.full_url, "http://127.0.0.1:8010/internal/erasure/tasks/t2/failed")
        for request in self.seen:
            self.assertEqual(request.get_header("X-internal-token"), "s3cret-token")
            self.assertNotIn("s3cret-token", request.full_url)

    def test_an_error_names_only_the_status_never_the_token(self):
        client = self.make(fail=urllib.error.HTTPError("u", 404, "no", {"X-Internal-Token": "s3cret-token"}, None))
        with self.assertRaises(RuntimeError) as caught:
            client.tasks()
        self.assertIn("404", str(caught.exception))
        self.assertNotIn("s3cret-token", str(caught.exception))

    def test_it_will_not_be_built_without_a_token_and_does_not_follow_redirects(self):
        with self.assertRaises(ValueError):
            HttpPlatformClient("http://x", "")
        handler = urllib.request.HTTPRedirectHandler
        client = HttpPlatformClient("http://127.0.0.1:1", "t")
        self.assertIsNotNone(client)  # constructed; the no-redirect opener is exercised by the real call path below
        import http.server, threading

        class Redirect(http.server.BaseHTTPRequestHandler):
            hits = []

            def do_GET(self):
                Redirect.hits.append(self.path)
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:9/elsewhere")
                self.end_headers()

            def log_message(self, *a): pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        with self.assertRaises(RuntimeError):
            HttpPlatformClient(f"http://127.0.0.1:{server.server_port}", "t").tasks()
        self.assertEqual(len(Redirect.hits), 1)  # it stopped at the redirect; the token was not forwarded anywhere


class Command(unittest.TestCase):
    def test_it_refuses_to_start_without_a_database_or_a_token(self):
        import os

        old = os.environ.pop("INTERNAL_TOKEN", None)
        try:
            with self.assertRaises(SystemExit):
                erasure_worker.main(["--database-url", "sqlite://", "--once"])
        finally:
            if old is not None:
                os.environ["INTERNAL_TOKEN"] = old


if __name__ == "__main__":
    unittest.main()
