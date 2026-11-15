"""A failed probe's error text comes from the provider's machine and must not reach the API response."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from gpuchain_coordinator.main import Settings, create_app
from gpuchain_coordinator.orchestrator import Orchestrator
from test_api import BODY

LEAK = "Traceback (most recent call last): File /home/provider/secret.py line 9"


class QualifyErrors(unittest.TestCase):
    def setUp(self):
        app = create_app(Settings(database_url="sqlite://", demo_mode=True, _env_file=None))
        self.client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))

    def qualify(self, results):
        with patch.object(Orchestrator, "qualify_providers", return_value=results):
            return self.client.post("/providers/qualify", json={"workload": BODY["workload"]})

    def test_probe_error_text_is_not_returned(self):
        response = self.qualify([{"provider_id": "p1", "name": "A", "approved": False, "code": "PROBE_ERROR", "reason": LEAK}])
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("secret.py", response.text)
        self.assertEqual(response.json()[0]["code"], "PROBE_ERROR")
        self.assertEqual(response.json()[0]["reason"], "The probe failed on the provider")

    def test_other_reasons_are_unchanged(self):
        response = self.qualify([{"provider_id": "p2", "name": "B", "approved": False, "code": "PROBE_OOM", "reason": "Out of memory at micro-batch 32"}])
        self.assertEqual(response.json()[0]["reason"], "Out of memory at micro-batch 32")


if __name__ == "__main__":
    unittest.main()
