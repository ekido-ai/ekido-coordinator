"""A signed provider message is not authority to control another provider's job."""

try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

from eth_account import Account
from sqlalchemy.orm import Session

from ekido_core.signing import sign_document
from ekido_coordinator import jobs
from ekido_coordinator.agent_gateway import Connection
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import Provider, Wallet

KEY = "0x" + "a" * 64


class AssignmentOwnership(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None))
        self.gateway = self.app.state.gateway
        with Session(self.app.state.engine) as session:
            wallet = Wallet(address=Account.from_key(KEY).address.lower(), role="provider")
            session.add(wallet)
            session.flush()
            a = Provider(name="A", wallet_id=wallet.id, kind="REAL", price_per_step=Decimal("0.001"))
            b = Provider(name="B", wallet_id=wallet.id, kind="REAL", price_per_step=Decimal("0.002"))
            session.add_all([a, b])
            session.flush()
            job = jobs.create_job(session, spec={}, policy={}, budget=Decimal("5"), deadline_seconds=1800)
            jobs.assign(session, job, a.id, {})
            self.a, self.b, self.job, self.token = a.id, b.id, job.id, job.fencing_token
            session.commit()

    def connection(self, provider):
        return Connection(provider, Account.from_key(KEY).address.lower(), AsyncMock(), "token", job_id=self.job, fencing_token=self.token)

    async def deliver(self, connection, kind):
        body = {"seq": 1, "type": kind, "job_id": self.job, "fencing_token": self.token, "payload": {"step": 10, "reason": "OOM"}}
        _, body["sig"] = sign_document(body, KEY)
        await self.gateway._on_message(connection, body)

    async def test_unassigned_provider_cannot_report_progress_or_exit(self):
        for kind in ("run.progress", "run.exited", "run.started", "checkpoint.uploaded", "run.warning"):
            with self.subTest(kind=kind):
                connection = self.connection(self.b)
                self.gateway.orchestrator.emit_live = Mock()
                self.gateway.orchestrator.on_run_started = Mock()
                self.gateway._on_exit = AsyncMock()
                self.gateway._on_checkpoint = AsyncMock()
                await self.deliver(connection, kind)
                self.gateway.orchestrator.emit_live.assert_not_called()
                self.gateway.orchestrator.on_run_started.assert_not_called()
                self.gateway._on_exit.assert_not_awaited()
                self.gateway._on_checkpoint.assert_not_awaited()
                self.assertEqual(connection.websocket.send_json.call_args.args[0]["payload"]["error"], "NOT_ASSIGNED")

    async def test_assigned_provider_can_report_progress(self):
        connection = self.connection(self.a)
        self.gateway.orchestrator.emit_live = Mock()
        await self.deliver(connection, "run.progress")
        self.gateway.orchestrator.emit_live.assert_called_once()
        self.assertEqual(connection.last_step, 10)
