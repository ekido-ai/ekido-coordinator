"""A14: destroying a job's key when the customer deletes their data."""

import unittest
from decimal import Decimal

from eth_account import Account
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core import jobkeys
from ekido_coordinator import job_keys, jobs
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import Job, JobKey, Provider, Wallet

KEY = "0x" + "a" * 64


class Erase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None))
        self.engine = self.app.state.engine
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
        with Session(self.engine) as s:
            wallet = Wallet(address=Account.from_key(KEY).address.lower(), role="provider")
            s.add(wallet)
            s.flush()
            self.provider = Provider(name="A", wallet_id=wallet.id, kind="REAL", price_per_step=Decimal("0.001"))
            s.add(self.provider)
            s.flush()
            self.provider_id = self.provider.id
            self.mine = self.make_job(s)
            self.other = self.make_job(s)
            s.commit()
        self.recipient = self.row(self.mine).recipient

    def make_job(self, session):
        job = jobs.create_job(session, spec={"n": len(session.new)}, policy={}, budget=Decimal("5"), deadline_seconds=1800)
        jobs.assign(session, job, self.provider_id, {})
        return job.id

    def row(self, job_id):
        with Session(self.engine) as s:
            return s.scalar(select(JobKey).where(JobKey.job_id == job_id))

    def set_state(self, job_id, state):
        with Session(self.engine) as s:
            s.get(Job, job_id).state = state
            s.commit()

    def test_destroying_removes_the_identity_and_is_safe_to_repeat(self):
        with Session(self.engine) as s:
            before = self.row(self.mine).wrapped_identity
            self.assertIsNotNone(before)
            self.assertTrue(job_keys.destroy(s, self.mine))
            self.assertFalse(job_keys.destroy(s, self.mine))
            self.assertFalse(job_keys.destroy(s, "no-such-job"))
        after = self.row(self.mine)
        self.assertIsNone(after.wrapped_identity)
        self.assertIsNotNone(after.destroyed_at)
        self.assertEqual(after.recipient, self.recipient)  # the public half is not secret and stays, so the row is not mistaken for missing

    def test_after_destruction_nothing_can_get_the_key_back_through_the_coordinator(self):
        blob = jobkeys.encrypt_bytes(b"customer data", self.recipient)  # something encrypted to this job earlier
        with Session(self.engine) as s:
            job_keys.destroy(s, self.mine)
        device = jobkeys.create_job_key()
        with Session(self.engine) as s, self.assertRaises(job_keys.KeyRefused) as caught:
            job_keys.release_to_device(s, job_id=self.mine, provider_id=self.provider_id, fencing_token=self.fencing(self.mine), agent_recipient=device.recipient)
        self.assertEqual(caught.exception.code, "NO_KEY")
        with self.assertRaises(jobkeys.JobKeyError):
            jobkeys.decrypt_bytes(blob, jobkeys.create_job_key().identity)  # and no other key opens it

    def fencing(self, job_id):
        with Session(self.engine) as s:
            return s.get(Job, job_id).fencing_token

    def test_other_jobs_keys_are_untouched(self):
        with Session(self.engine) as s:
            job_keys.destroy(s, self.mine)
        self.assertIsNotNone(self.row(self.other).wrapped_identity)
        self.assertIsNone(self.row(self.other).destroyed_at)

    def test_the_route_destroys_a_finished_jobs_key_once(self):
        self.set_state(self.mine, "SETTLED")
        first = self.client.post(f"/jobs/{self.mine}/erase-key")
        self.assertEqual((first.status_code, first.json()["destroyed"], first.json()["state"]), (200, True, "SETTLED"))
        again = self.client.post(f"/jobs/{self.mine}/erase-key")
        self.assertEqual((again.status_code, again.json()["destroyed"]), (200, False))
        self.assertIsNone(self.row(self.mine).wrapped_identity)

    def test_a_job_that_has_not_ended_keeps_its_key_and_the_route_says_so(self):
        for state in ("OPEN", "ASSIGNED", "RUNNING", "RECOVERING", "VALIDATING", "COMPLETED"):
            with self.subTest(state=state):
                self.set_state(self.mine, state)
                response = self.client.post(f"/jobs/{self.mine}/erase-key")
                self.assertEqual(response.status_code, 409)
                self.assertIsNotNone(self.row(self.mine).wrapped_identity)

    def test_every_ended_state_is_accepted_and_an_unknown_job_is_404(self):
        for state in ("SETTLED", "FAILED", "CANCELLED", "REFUNDED"):
            with self.subTest(state=state):
                self.setUp()
                self.set_state(self.mine, state)
                self.assertEqual(self.client.post(f"/jobs/{self.mine}/erase-key").status_code, 200)
        self.assertEqual(self.client.post("/jobs/missing/erase-key").status_code, 404)

    def test_the_route_is_not_open_to_outsiders(self):
        outsider = TestClient(self.app, base_url="http://api.example.org", client=("203.0.113.9", 50000))
        self.set_state(self.mine, "SETTLED")
        self.assertIn(outsider.post(f"/jobs/{self.mine}/erase-key").status_code, (401, 403, 404))
        self.assertIsNotNone(self.row(self.mine).wrapped_identity)


if __name__ == "__main__":
    unittest.main()
