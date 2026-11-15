"""Runs against a real Postgres when TEST_DATABASE_URL is set (CI's Postgres job, after `alembic upgrade head`).

SQLite accepts things Postgres refuses: over-long strings, loose types, some constraint orders. These tests push the
main write paths through Postgres. Each test runs inside a transaction that is rolled back, so they leave no rows.
"""
import os
import unittest
from decimal import Decimal

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from ekido_coordinator import accounts, jobs
from ekido_coordinator.accounts import AccountError
from ekido_coordinator.models import Base, Job, JobTransition

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

    def test_accounts_flow(self):
        token = accounts.start_login(self.session, "Asha@Example.com")
        user, session_token = accounts.finish_login(self.session, token)
        self.assertEqual(accounts.user_for_session(self.session, session_token), user.id)
        org = accounts.create_org(self.session, user.id, "Asha's Clinic")
        accounts.add_member(self.session, org.id, "staff@example.com", "member")
        key, secret = accounts.create_api_key(self.session, org.id, user.id, "server", "live")
        self.assertEqual(accounts.principal_for_key(self.session, secret).org_id, org.id)
        self.assertEqual(len(accounts.org_view(self.session, org.id)["members"]), 2)
        with self.assertRaises(AccountError):
            accounts.finish_login(self.session, token)  # used once already

    def test_unique_email_is_enforced(self):
        token = accounts.start_login(self.session, "x@example.com")
        accounts.finish_login(self.session, token)
        again = accounts.start_login(self.session, "X@example.com")
        user, _ = accounts.finish_login(self.session, again)
        rows = self.session.scalars(select(accounts.UserEmail).where(accounts.UserEmail.email == "x@example.com")).all()
        self.assertEqual([r.user_id for r in rows], [user.id])

    def test_jobs_keep_json_decimals_and_transitions(self):
        job = jobs.create_job(self.session, spec={"template": "synthetic-linear/v1", "nested": {"a": [1, 2]}}, policy={"q": 0.9}, budget=Decimal("5.123456"), deadline_seconds=600)
        self.session.add(JobTransition(job_id=job.id, seq=99, from_state="OPEN", to_state="CANCELLED"))
        self.session.commit()
        fresh = self.session.get(Job, job.id)
        self.assertEqual(fresh.spec["nested"], {"a": [1, 2]})
        self.assertEqual(fresh.budget, Decimal("5.123456"))


if __name__ == "__main__":
    unittest.main()
