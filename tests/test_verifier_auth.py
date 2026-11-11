"""A public verifier address must not let a stranger consume that verifier's tasks."""

import time
import unittest

from eth_account import Account
from fastapi.testclient import TestClient

from gpuchain_core.signing import sign_document
from gpuchain_coordinator.main import Settings, create_app
from gpuchain_coordinator.quality_service import EvaluationTask

KEYS = ["0x" + c * 64 for c in "ab"]
ADDRESSES = [Account.from_key(key).address for key in KEYS]


class VerifierAuth(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", verifier_addresses=",".join(ADDRESSES), verifier_quorum=2, _env_file=None))
        self.client = TestClient(self.app, base_url="http://192.168.1.2:8000", client=("192.168.1.3", 50000))
        self.quality = self.app.state.gateway.orchestrator.quality
        self.task = EvaluationTask("task", "missing-job", "a" * 64, None, True)
        self.quality.tasks[self.task.task_id] = self.task

    def login(self, index=0):
        message = {"kind": "gpuchain.verifier.session.v1", "address": ADDRESSES[index], "ts": int(time.time())}
        _, signature = sign_document(message, KEYS[index])
        response = self.client.post("/verifier/session", json={"message": message, "signature": signature})
        self.assertEqual(response.status_code, 200)
        return {"Authorization": "Bearer " + response.json()["token"]}

    def test_anonymous_claims_are_refused_without_consuming_a_task(self):
        for path in ("/verifier/tasks/claim", "/verifier/verdicts/claim"):
            with self.subTest(path=path):
                response = self.client.post(path, json={"address": ADDRESSES[0]})
                self.assertEqual(response.status_code, 401)
        self.assertEqual(self.task.claimed_by, set())

    def test_a_verifier_cannot_consume_another_verifiers_tasks(self):
        headers = self.login()
        for path in ("/verifier/tasks/claim", "/verifier/verdicts/claim"):
            self.assertEqual(self.client.post(path, json={"address": ADDRESSES[1]}, headers=headers).status_code, 403)
        self.assertEqual(self.task.claimed_by, set())

    def test_a_provider_artifact_token_is_not_a_verifier_session(self):
        self.app.state.gateway.upload_tokens["provider-token"] = "provider"
        response = self.client.post("/verifier/tasks/claim", json={"address": ADDRESSES[0]}, headers={"Authorization": "Bearer provider-token"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.task.claimed_by, set())

    def test_an_invalid_token_is_refused(self):
        response = self.client.post("/verifier/verdicts/claim", json={"address": ADDRESSES[0]}, headers={"Authorization": "Bearer bad"})
        self.assertEqual(response.status_code, 401)

    def test_a_signed_session_can_claim_its_own_work(self):
        headers = self.login()
        # The missing job is intentional: reaching its lookup proves authentication passed.
        response = self.client.post("/verifier/tasks/claim", json={"address": ADDRESSES[0].lower()}, headers=headers)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.task.claimed_by, {ADDRESSES[0].lower()})
        response = self.client.post("/verifier/verdicts/claim", json={"address": ADDRESSES[0]}, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json())
