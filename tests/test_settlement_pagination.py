"""Settlement cursor paging must not omit ties or repeat rows after a new insert."""
try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import unittest
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import Settlement


class SettlementPagination(unittest.TestCase):
    def test_cursor_ties_and_newer_inserts(self):
        app = create_app(Settings(database_url="sqlite://", _env_file=None))
        client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))
        self.addCleanup(app.state.engine.dispose)
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with Session(app.state.engine) as session:
            for job_id, created in (("a", t), ("b", t), ("c", t), ("d", t - timedelta(seconds=1))):
                session.add(Settlement(job_id=job_id, verdict={}, status="SETTLED", created_at=created))
            session.commit()
        first = client.get("/settlements?limit=2")
        self.assertEqual([r["job_id"] for r in first.json()], ["c", "b"])
        with Session(app.state.engine) as session:
            session.add(Settlement(job_id="new", verdict={}, status="SETTLED", created_at=t + timedelta(seconds=1)))
            session.commit()
        second = client.get("/settlements?limit=2&before_job_id=b")
        self.assertEqual([r["job_id"] for r in second.json()], ["a", "d"])
        self.assertEqual(client.get("/settlements?before_job_id=missing").status_code, 404)
        self.assertEqual(client.get("/settlements?before_job_id=d").json(), [])
