"""Public jobs must not start with an unusably short dispute window."""
import wrapper_setup  # noqa: F401  (job keys need a wrapper; see wrapper_setup.py)
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

from ekido_coordinator.chain import EscrowClient
from ekido_coordinator.settlement_service import NotFunded, SettlementService


class DisputeWindowGuard(unittest.TestCase):
    def setUp(self):
        self.job = SimpleNamespace(
            onchain_id="0x" + "01" * 32,
            spec_hash="spec", policy_hash="policy", budget=Decimal("5"), spec={},
            deadline=datetime.now(timezone.utc) + timedelta(hours=2),
        )
        self.chain = Mock()
        self.chain.job.return_value = {
            "state": "Funded", "spec_hash": "spec", "policy_hash": "policy",
            "budget": 5_000_000, "deadline": self.job.deadline.timestamp(), "dispute_window": 3600,
        }
        self.chain.rpc_chain_id.return_value = 31337

    def service(self, chain_id=84532, local=False):
        return SettlementService(None, None, self.chain, chain_id, "0x" + "02" * 20, set(), 1, None, local_chain=local)

    def test_public_boundary_and_missing_data_fail_closed(self):
        service = self.service()
        service.require_funded(self.job)  # exactly one hour is allowed
        for value in (3599, 1, 0, None, True, "3600"):
            with self.subTest(window=value):
                self.chain.job.return_value["dispute_window"] = value
                with self.assertRaises(NotFunded):
                    service.require_funded(self.job)

    def test_short_window_allowed_only_when_config_and_rpc_are_local(self):
        self.chain.job.return_value["dispute_window"] = 10
        self.service(31337, True).require_funded(self.job)
        for chain_id, local, rpc_id in ((31337, False, 31337), (84532, True, 31337), (31337, True, 84532)):
            with self.subTest(chain_id=chain_id, local=local, rpc_id=rpc_id):
                self.chain.rpc_chain_id.return_value = rpc_id
                with self.assertRaises(NotFunded):
                    self.service(chain_id, local).require_funded(self.job)

    def test_short_window_with_unreachable_rpc_does_not_start(self):
        self.chain.job.return_value["dispute_window"] = 10
        self.chain.rpc_chain_id.side_effect = OSError("RPC unavailable")
        with self.assertRaises(OSError):
            self.service(31337, True).require_funded(self.job)

    def test_chain_reader_uses_dispute_window_field_from_job_struct(self):
        client = EscrowClient.__new__(EscrowClient)
        client.contract = Mock()
        raw = [None] * 17
        raw[0], raw[1], raw[2], raw[4], raw[5], raw[7], raw[8] = "client", 123, 1, 3600, 0, False, 5_000_000
        raw[10], raw[14], raw[15] = 4_000_000, ["0x" + "1" * 40], [1_000_000]
        raw[11], raw[12], raw[13] = b"a" * 32, b"b" * 32, b"c" * 32
        client.contract.functions.getJob.return_value.call.return_value = raw
        self.assertEqual(client.job("0x" + "01" * 32)["dispute_window"], 3600)
