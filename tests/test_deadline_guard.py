"""A job must not start when its on-chain deadline is earlier than the agreed one.

The escrow contract accepts any deadline in the future, and anyone can refund the client an hour after it.
The coordinator is where the lead time is enforced (see contracts/README.md).
"""
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

from gpuchain_coordinator.settlement_service import DEADLINE_TOLERANCE_SECONDS, NotFunded, SettlementService


class DeadlineGuard(unittest.TestCase):
    def setUp(self):
        self.job = SimpleNamespace(
            onchain_id="0x" + "01" * 32,
            spec_hash="spec", policy_hash="policy", budget=Decimal("5"), spec={},
            deadline=datetime.now(timezone.utc) + timedelta(hours=2),
        )
        self.agreed = self.job.deadline.timestamp()
        self.chain = Mock()
        self.chain.job.return_value = {
            "state": "Funded", "spec_hash": "spec", "policy_hash": "policy",
            "budget": 5_000_000, "deadline": self.agreed, "dispute_window": 3600,
        }
        self.service = SettlementService(None, None, self.chain, 84532, "0x" + "02" * 20, set(), 1, None, local_chain=False)

    def check(self, onchain_deadline):
        self.chain.job.return_value["deadline"] = onchain_deadline
        self.service.require_funded(self.job)

    def test_the_agreed_deadline_or_a_later_one_starts(self):
        self.check(self.agreed)
        self.check(self.agreed + 3600)

    def test_a_deadline_inside_the_clock_tolerance_starts(self):
        self.check(self.agreed - DEADLINE_TOLERANCE_SECONDS)

    def test_a_deadline_earlier_than_the_tolerance_does_not_start(self):
        for shortfall in (DEADLINE_TOLERANCE_SECONDS + 1, 3600, 2 * 3600 - 1):
            with self.subTest(shortfall=shortfall):
                with self.assertRaises(NotFunded):
                    self.check(self.agreed - shortfall)


if __name__ == "__main__":
    unittest.main()
