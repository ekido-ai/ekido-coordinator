"""Runs against a real Postgres when TEST_DATABASE_URL is set (CI's Postgres job, after `alembic upgrade head`).

SQLite accepts things Postgres refuses: over-long strings, loose types, some constraint orders. These tests push the
main write paths through Postgres. Each test runs inside a transaction that is rolled back, so they leave no rows.
The URL connects as the coordinator's own user, whose search_path is the `coordinator` schema.
"""
import os
import unittest
from decimal import Decimal

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

from ekido_coordinator import jobs
from ekido_coordinator.models import Base, Job, JobTransition, Settlement

URL = os.environ.get("TEST_DATABASE_URL", "")


@unittest.skipUnless(URL.startswith("postgresql"), "set TEST_DATABASE_URL to a migrated Postgres database")
class OnPostgres(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(URL)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.connection = self.engine.connect()
        self.outer = self.connection.begin()
        # Commits inside the code under test become savepoints; the outer rollback removes everything.
        self.session = Session(bind=self.connection, join_transaction_mode="create_savepoint")

    def tearDown(self):
        self.session.close()
        self.outer.rollback()
        self.connection.close()

    def test_migrations_created_every_table_the_models_define(self):
        existing = set(inspect(self.connection).get_table_names())
        self.assertEqual(set(Base.metadata.tables) - existing, set())

    def test_the_account_tables_are_gone(self):
        # Sign-in, teams and keys moved to the Platform API (migration 0002).
        existing = set(inspect(self.connection).get_table_names())
        self.assertFalse({"user_emails", "orgs", "api_keys", "web_sessions"} & existing)

    def test_tables_live_in_the_coordinator_schema(self):
        self.assertEqual(self.connection.execute(text("select current_schema()")).scalar(), "coordinator")

    def test_jobs_keep_json_decimals_and_transitions(self):
        job = jobs.create_job(self.session, spec={"template": "synthetic-linear/v1", "nested": {"a": [1, 2]}}, policy={"q": 0.9}, budget=Decimal("5.123456"), deadline_seconds=600)
        self.session.add(JobTransition(job_id=job.id, seq=99, from_state="OPEN", to_state="CANCELLED"))
        self.session.commit()
        fresh = self.session.get(Job, job.id)
        self.assertEqual(fresh.spec["nested"], {"a": [1, 2]})
        self.assertEqual(fresh.budget, Decimal("5.123456"))

    def test_long_status_values_fit(self):
        # AWAITING_SIGNATURES (19 characters) stopped the live data copy on 16 Nov: it was a String(16) column.
        job = jobs.create_job(self.session, spec={"template": "synthetic-linear/v1"}, policy={}, budget=Decimal("1"), deadline_seconds=600)
        self.session.add(Settlement(job_id=job.id, verdict={"accepted": True}, status="AWAITING_SIGNATURES"))
        self.session.commit()
        widths = {c["name"]: getattr(c["type"], "length", None) for c in inspect(self.connection).get_columns("settlements")}
        self.assertEqual(widths["status"], 32)
        stored = self.session.execute(select(Settlement.status).where(Settlement.job_id == job.id)).scalar_one()
        self.assertEqual(stored, "AWAITING_SIGNATURES")

    def test_job_keys_table_stores_binary_wrapped_keys_once_per_job(self):
        # A12: the wrapped identity is bytea, a job has at most one key, and a release works through Postgres.
        from sqlalchemy.exc import IntegrityError

        from ekido_core import jobkeys
        from ekido_coordinator import job_keys
        from ekido_coordinator.models import JobKey, Provider, Wallet

        columns = {c["name"]: c for c in inspect(self.connection).get_columns("job_keys")}
        self.assertEqual(set(columns), {"id", "created_at", "job_id", "recipient", "wrapped_identity", "destroyed_at"})
        self.assertTrue(columns["wrapped_identity"]["nullable"])
        job = jobs.create_job(self.session, spec={"template": "synthetic-linear/v1"}, policy={}, budget=Decimal("1"), deadline_seconds=600)
        row = self.session.scalar(select(JobKey).where(JobKey.job_id == job.id))
        self.assertIsInstance(bytes(row.wrapped_identity), bytes)
        identity = job_keys.get_wrapper().unwrap(bytes(row.wrapped_identity), job.id)
        self.assertNotIn(identity.encode(), bytes(row.wrapped_identity))
        with self.session.begin_nested():
            self.session.add(JobKey(job_id=job.id, recipient="age1x", wrapped_identity=b"\x00\x01"))
            with self.assertRaises(IntegrityError):
                self.session.flush()
        wallet = Wallet(address="0x" + "b" * 40, role="provider")
        self.session.add(wallet)
        self.session.flush()
        provider = Provider(name="A", wallet_id=wallet.id, kind="REAL")
        other = Provider(name="B", wallet_id=wallet.id, kind="REAL")
        self.session.add_all([provider, other])
        self.session.flush()
        assignment = jobs.assign(self.session, job, provider.id, {})
        device = jobkeys.create_job_key()
        sealed = job_keys.release_to_device(self.session, job_id=job.id, provider_id=provider.id, fencing_token=assignment.fencing_token, agent_recipient=device.recipient)
        self.assertEqual(jobkeys.decrypt_bytes(sealed, device.identity).decode(), identity)
        with self.assertRaises(job_keys.KeyRefused):
            job_keys.release_to_device(self.session, job_id=job.id, provider_id=other.id, fencing_token=assignment.fencing_token, agent_recipient=device.recipient)


if __name__ == "__main__":
    unittest.main()
