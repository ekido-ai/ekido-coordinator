"""A verifier can retry a verdict after a temporary failure without duplicating signatures."""
try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import unittest
from unittest.mock import patch

from ekido_coordinator.settlement_service import SettlementService, VerdictTask


class VerdictClaims(unittest.TestCase):
    def test_claim_expires_but_submitted_signatures_are_not_reclaimed(self):
        address = "0x" + "1" * 40
        service = SettlementService(None, None, None, 31337, "0x" + "2" * 40, {address}, 1, None)
        task = VerdictTask("job", {"job_id": "job"}, {})
        service.tasks["job"] = task
        with patch("ekido_coordinator.settlement_service.time.monotonic", return_value=100):
            self.assertEqual(service.claim(address), task.inputs)
            self.assertIsNone(service.claim(address))
        with patch("ekido_coordinator.settlement_service.time.monotonic", return_value=130):
            self.assertEqual(service.claim(address), task.inputs)
            task.signatures[address] = "already signed"
        with patch("ekido_coordinator.settlement_service.time.monotonic", return_value=200):
            self.assertIsNone(service.claim(address))
