"""A13: a run's config carries the job's public key only when encrypted checkpoints are switched on."""

try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import unittest
from decimal import Decimal

from eth_account import Account
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_coordinator import jobs
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import JobKey, Provider, Wallet

KEY = "0x" + "a" * 64


class RunConfigRecipient(unittest.TestCase):
    def build(self, **settings):
        app = create_app(Settings(database_url="sqlite://", _env_file=None, **settings))
        self.orchestrator = app.state.gateway.orchestrator
        self.engine = app.state.engine
        with Session(self.engine) as session:
            wallet = Wallet(address=Account.from_key(KEY).address.lower(), role="provider")
            session.add(wallet)
            session.flush()
            provider = Provider(name="A", wallet_id=wallet.id, kind="REAL", price_per_step=Decimal("0.001"))
            session.add(provider)
            session.flush()
            job = jobs.create_job(session, spec={}, policy={}, budget=Decimal("5"), deadline_seconds=1800)
            assignment = jobs.assign(session, job, provider.id, {})
            session.commit()
            self.job_id = job.id
            self.recipient = session.scalar(select(JobKey.recipient).where(JobKey.job_id == job.id))
            self.config = lambda: self.orchestrator._run_config(session.get(type(job), job.id), assignment, provider, None, None)
            self.session = session
            return self.config()

    def test_off_by_default_nothing_is_added(self):
        self.assertNotIn("job_recipient", self.build())

    def test_on_the_config_carries_only_the_public_key(self):
        config = self.build(encrypt_checkpoints=True)
        self.assertEqual(config["job_recipient"], self.recipient)
        self.assertTrue(config["job_recipient"].startswith("age1"))
        self.assertNotIn("AGE-SECRET-KEY", str(config))
        self.assertIn("job_id", config)  # everything else is as before

    def test_a_job_without_a_key_or_with_a_destroyed_key_gets_no_recipient(self):
        self.build(encrypt_checkpoints=True)
        with Session(self.engine) as session:
            row = session.scalar(select(JobKey).where(JobKey.job_id == self.job_id))
            row.wrapped_identity, row.destroyed_at = None, __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            session.commit()
        self.assertNotIn("job_recipient", self.config())
        with Session(self.engine) as session:
            session.delete(session.scalar(select(JobKey).where(JobKey.job_id == self.job_id)))
            session.commit()
        self.assertNotIn("job_recipient", self.config())


if __name__ == "__main__":
    unittest.main()
