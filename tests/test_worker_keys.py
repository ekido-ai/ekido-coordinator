"""A restarted agent generates a new worker key. Work signed with the old one must still be paid."""

import wrapper_setup  # noqa: F401  (job keys need a wrapper; see wrapper_setup.py)
import unittest
from decimal import Decimal
from types import SimpleNamespace

from eth_account import Account
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from ekido_core.signing import sign_document, verify_document
from ekido_coordinator.agent_gateway import enrollment_document
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import Provider, ProviderKeyProof
from sqlalchemy import select
from ekido_coordinator.settlement_service import SettlementService

WALLET_A, WALLET_B = "0x" + "1" * 64, "0x" + "2" * 64
OLD_WORKER, NEW_WORKER = "0x" + "a" * 64, "0x" + "b" * 64
LOCAL = {"base_url": "http://127.0.0.1:8000", "client": ("127.0.0.1", 50000)}


def hello(provider_id, wallet_key, worker_key):
    address = Account.from_key(worker_key).address
    _, enrollment = sign_document(enrollment_document(provider_id, address), wallet_key)
    message = {"v": 1, "seq": 1, "type": "agent.hello", "payload": {"provider_id": provider_id, "worker_key": address, "enrollment_sig": enrollment}}
    _, message["sig"] = sign_document(message, worker_key)
    return message


class WorkerKeyTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", demo_mode=True, _env_file=None))
        self.client = TestClient(self.app, **LOCAL)
        self.gateway = self.app.state.gateway
        self.a = self.register("A", WALLET_A, "0.001")
        self.b = self.register("B", WALLET_B, "0.003")

    def register(self, name, wallet, price):
        return self.client.post("/providers", json={"name": name, "wallet_address": Account.from_key(wallet).address, "kind": "REAL", "price_per_step": price}).json()["provider_id"]

    def enroll(self, provider_id, wallet, worker):
        return self.gateway._authenticate_hello(hello(provider_id, wallet, worker))

    def test_work_signed_before_a_restart_is_still_paid_to_the_same_wallet(self):
        self.assertIsNotNone(self.enroll(self.a, WALLET_A, OLD_WORKER))
        self.assertIsNotNone(self.enroll(self.a, WALLET_A, NEW_WORKER), "a restarted agent must be able to come back")
        old, new = Account.from_key(OLD_WORKER).address.lower(), Account.from_key(NEW_WORKER).address.lower()
        chain = [SimpleNamespace(signer=old, logical_step=100), SimpleNamespace(signer=new, logical_step=250)]
        with Session(self.app.state.engine) as session:
            totals, refund = SettlementService._payouts(None, session, chain, Decimal("5"))
        wallet = Account.from_key(WALLET_A).address.lower()
        self.assertEqual({k.lower(): v for k, v in totals.items()}, {wallet: Decimal("0.25")})
        self.assertEqual(refund, Decimal("4.75"))

    def test_another_provider_cannot_claim_a_key_already_in_use(self):
        """Otherwise a provider could redirect another's accepted work to its own wallet."""
        self.assertIsNotNone(self.enroll(self.a, WALLET_A, OLD_WORKER))
        self.assertIsNone(self.enroll(self.b, WALLET_B, OLD_WORKER))
        with Session(self.app.state.engine) as session:
            self.assertIsNone(session.get(Provider, self.b).enrolled_key)

    def test_payout_check_can_see_every_key_a_provider_has_used(self):
        self.enroll(self.a, WALLET_A, OLD_WORKER)
        self.enroll(self.a, WALLET_A, NEW_WORKER)
        row = next(p for p in self.client.get("/providers").json() if p["provider_id"] == self.a)
        self.assertEqual(set(row["signing_keys"]), {Account.from_key(k).address.lower() for k in (OLD_WORKER, NEW_WORKER)})

    def test_wallet_delegations_survive_an_agent_restart(self):
        self.enroll(self.a, WALLET_A, OLD_WORKER)
        self.enroll(self.a, WALLET_A, NEW_WORKER)
        with Session(self.app.state.engine) as session:
            proofs = list(session.scalars(select(ProviderKeyProof).where(ProviderKeyProof.provider_id == self.a)))
            self.assertEqual({proof.key for proof in proofs}, {Account.from_key(k).address.lower() for k in (OLD_WORKER, NEW_WORKER)})
            for proof in proofs:
                self.assertTrue(verify_document(enrollment_document(self.a, proof.key), proof.signature, Account.from_key(WALLET_A).address))

    def test_payout_evidence_contains_both_signed_delegations(self):
        self.enroll(self.a, WALLET_A, OLD_WORKER)
        self.enroll(self.a, WALLET_A, NEW_WORKER)
        old, new = (Account.from_key(key).address.lower() for key in (OLD_WORKER, NEW_WORKER))
        lineage = [SimpleNamespace(signer=old, job_id="job", manifest_hash="oldhash", manifest={"signer": old, "signature": "old"}), SimpleNamespace(signer=new, job_id="job", manifest_hash="newhash", manifest={"signer": new, "signature": "new"})]
        with Session(self.app.state.engine) as session:
            evidence = SettlementService._payout_evidence(SimpleNamespace(quality=None), session, lineage)
        self.assertEqual(evidence["manifests"], [row.manifest for row in lineage])
        self.assertEqual(set(evidence["delegations"]), {old, new})
        for signer, proof in evidence["delegations"].items():
            self.assertTrue(verify_document(enrollment_document(self.a, signer), proof["signature"], Account.from_key(WALLET_A).address))


if __name__ == "__main__":
    unittest.main()
