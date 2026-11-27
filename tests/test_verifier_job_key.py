"""A12: a verifier gets a job's key only for an evaluation it has claimed, only with a signed session."""

try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import base64
import time
import unittest
from decimal import Decimal

from eth_account import Account
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core import jobkeys
from ekido_core.signing import sign_document
from ekido_coordinator import job_keys, jobs
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import Job, JobKey
from ekido_coordinator.quality_service import EvaluationTask

KEYS = ["0x" + c * 64 for c in "ab"]
ADDRESSES = [Account.from_key(key).address for key in KEYS]
URL = "/verifier/tasks/job-key"


class VerifierKey(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", verifier_addresses=",".join(ADDRESSES), verifier_quorum=2, _env_file=None))
        self.client = TestClient(self.app, base_url="http://192.168.1.2:8000", client=("192.168.1.3", 50000))
        self.quality = self.app.state.gateway.orchestrator.quality
        with Session(self.app.state.engine) as s:
            self.job_id = jobs.create_job(s, spec={}, policy={}, budget=Decimal("5"), deadline_seconds=600).id
            self.other_job = jobs.create_job(s, spec={"n": 2}, policy={}, budget=Decimal("5"), deadline_seconds=600).id
            s.commit()
        self.task = EvaluationTask("task", self.job_id, "a" * 64, None, True)
        self.quality.tasks["task"] = self.task
        self.mine = jobkeys.create_job_key()  # the verifier's one-off key for this request

    def login(self, index=0):
        message = {"kind": "gpuchain.verifier.session.v1", "address": ADDRESSES[index], "ts": int(time.time())}
        _, signature = sign_document(message, KEYS[index])
        return {"Authorization": "Bearer " + self.client.post("/verifier/session", json={"message": message, "signature": signature}).json()["token"]}

    def ask(self, headers, address=0, job=None, recipient=None):
        return self.client.post(URL, json={"address": ADDRESSES[address], "job_id": job or self.job_id, "recipient": recipient or self.mine.recipient}, headers=headers)

    def claim(self, index=0):
        self.task.claimed_by.add(ADDRESSES[index].lower())

    def job_identity(self, job_id=None):
        with Session(self.app.state.engine) as s:
            row = s.scalar(select(JobKey).where(JobKey.job_id == (job_id or self.job_id)))
            return jobkeys.unwrap_job_key(row.wrapped_identity, row.job_id, job_keys.get_wrapper()).identity

    def set_state(self, state):
        with Session(self.app.state.engine) as s:
            s.get(Job, self.job_id).state = state
            s.commit()


class Access(VerifierKey):
    def test_a_verifier_with_a_claimed_evaluation_gets_the_key_encrypted_to_its_own_recipient(self):
        self.claim()
        response = self.ask(self.login())
        self.assertEqual(response.status_code, 200, response.text)
        sealed = base64.b64decode(response.json()["ciphertext_b64"])
        self.assertTrue(jobkeys.is_age_ciphertext(sealed))
        self.assertEqual(jobkeys.decrypt_bytes(sealed, self.mine.identity).decode(), self.job_identity())
        with self.assertRaises(jobkeys.JobKeyError):
            jobkeys.decrypt_bytes(sealed, jobkeys.create_job_key().identity)  # nobody else can read it
        self.assertNotIn("AGE-SECRET-KEY", response.text)

    def test_no_session_a_wrong_session_and_a_provider_token_are_refused(self):
        self.claim()
        self.assertEqual(self.ask({}).status_code, 401)
        self.assertEqual(self.ask({"Authorization": "Bearer nope"}).status_code, 401)
        self.assertEqual(self.ask(self.login(0), address=1).status_code, 403)  # another verifier's identity
        self.app.state.gateway.upload_tokens["provider-token"] = "provider"
        self.assertEqual(self.ask({"Authorization": "Bearer provider-token"}).status_code, 403)

    def test_being_on_the_allowlist_without_a_claimed_evaluation_is_not_enough(self):
        response = self.ask(self.login())
        self.assertEqual(response.status_code, 403)
        self.assertIn("NOT_ASSIGNED", response.text)
        self.assertNotIn("ciphertext", response.text)

    def test_a_claim_on_one_job_gives_no_key_for_another(self):
        self.claim()
        self.assertEqual(self.ask(self.login(), job=self.other_job).status_code, 403)

    def test_another_verifier_that_did_not_claim_gets_nothing(self):
        self.claim(0)
        self.assertEqual(self.ask(self.login(1), address=1).status_code, 403)

    def test_once_it_has_reported_or_the_task_is_decided_the_key_is_no_longer_given(self):
        self.claim()
        headers = self.login()
        self.assertEqual(self.ask(headers).status_code, 200)
        self.task.reports.append({"signer": ADDRESSES[0].lower()})
        self.assertEqual(self.ask(headers).status_code, 403)
        self.task.reports.clear()
        self.assertEqual(self.ask(headers).status_code, 200)
        self.task.done.set()
        self.assertEqual(self.ask(headers).status_code, 403)

    def test_unknown_job_ids_are_refused(self):
        self.claim()
        self.assertEqual(self.ask(self.login(), job="no-such-job").status_code, 403)


class KeyRefusals(VerifierKey):
    def test_an_ended_job_a_destroyed_key_and_a_bad_recipient_are_refused_with_a_reason(self):
        self.claim()
        headers = self.login()
        for bad in ("", "age1nope", "not a key", "AGE-SECRET-KEY-1ABC"):
            with self.subTest(recipient=bad):
                response = self.ask(headers, recipient=bad or "x")
                self.assertEqual((response.status_code, "BAD_RECIPIENT" in response.text), (409, True))
        for state in ("SETTLED", "FAILED", "CANCELLED", "REFUNDED"):
            with self.subTest(state=state):
                self.set_state(state)
                response = self.ask(headers)
                self.assertEqual((response.status_code, "JOB_ENDED" in response.text), (409, True))
        self.set_state("VALIDATING")
        with Session(self.app.state.engine) as s:
            row = s.scalar(select(JobKey).where(JobKey.job_id == self.job_id))
            row.wrapped_identity, row.destroyed_at = None, __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            s.commit()
        response = self.ask(headers)
        self.assertEqual((response.status_code, "NO_KEY" in response.text), (409, True))

    def test_validating_and_completed_jobs_still_have_a_key_to_give(self):
        self.claim()
        for state in ("VALIDATING", "COMPLETED", "RUNNING"):
            with self.subTest(state=state):
                self.set_state(state)
                self.assertEqual(self.ask(self.login()).status_code, 200)

    def test_nothing_unwrapped_reaches_a_log(self):
        self.claim()
        headers = self.login()
        with self.assertLogs(level="DEBUG") as captured:
            import logging

            logging.getLogger("probe").debug("start")
            self.ask(headers)
            self.ask(headers, recipient="age1nope")
            with Session(self.app.state.engine) as s:
                s.scalar(select(JobKey).where(JobKey.job_id == self.job_id)).wrapped_identity = b"junk"
                s.commit()
            self.ask(headers)
        everything = "\n".join(captured.output)
        for secret in (self.mine.identity,):
            self.assertNotIn(secret, everything)


if __name__ == "__main__":
    unittest.main()
