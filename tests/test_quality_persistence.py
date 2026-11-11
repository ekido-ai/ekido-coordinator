import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from gpuchain_coordinator.main import Settings, create_app
from gpuchain_coordinator.models import QualityDecision


class QualityPersistence(unittest.TestCase):
    def test_signed_decision_restored_on_coordinator_restart(self):
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        self.addCleanup(engine.dispose)
        settings = Settings(database_url="sqlite://", demo_mode=True, verifier_addresses="0x" + "1" * 40, verifier_quorum=1, _env_file=None)
        app = create_app(settings, engine=engine)
        quality = app.state.gateway.orchestrator.quality
        reports = [{"signer": "0x" + "1" * 40, "signature": "example"}]
        quality.on_decision("job", "a" * 64, reports)
        with Session(engine) as session:
            self.assertEqual(session.scalar(select(QualityDecision)).reports, reports)
        restarted = create_app(settings, engine=engine)
        self.assertEqual(restarted.state.gateway.orchestrator.quality.decided_reports[("job", "a" * 64)], reports)
