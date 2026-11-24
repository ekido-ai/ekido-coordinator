"""API tests against in-memory SQLite. Needs fastapi, httpx, sqlalchemy and pydantic-settings."""

import wrapper_setup  # noqa: F401  (job keys need a wrapper; see wrapper_setup.py)
import copy
import unittest
from eth_account import Account
from sqlalchemy.orm import Session

from fastapi.testclient import TestClient

from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import JobBinding, Provider, Wallet
from ekido_core.payout_terms import job_binding, new_job_ids, provider_quote, workload_hash

BODY = {
    "workload": {
        "template": "synthetic-linear/v1",
        "seed": 1,
        "logical_batch_size": 32,
        "total_logical_steps": 40,
        "optimizer": {"name": "adamw", "lr": 0.01, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8, "max_grad_norm": 1.0},
        "lr_schedule": {"name": "constant", "warmup_steps": 0},
    },
    "policy": {"metric": "accuracy", "min_absolute": 0.88},
    "budget_usdc": "5.00",
    "deadline_seconds": 1800,
}


def local_client(app):
    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client = local_client(create_app(Settings(database_url="sqlite://", demo_mode=True, _env_file=None)))

    def create(self, **headers):
        return self.client.post("/jobs", json=BODY, headers=headers)

    def test_incomplete_spec_is_refused_with_reasons(self):
        short = dict(BODY, workload={"template": "distilbert-lora-agnews/v1", "logical_batch_size": 32, "total_logical_steps": 300, "seed": 1})
        response = self.client.post("/jobs", json=short)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "INVALID_JOB_SPEC")
        self.assertIn("model is missing", response.json()["problems"])
        self.assertEqual(self.client.get("/jobs").json(), [], "a refused spec must not create a job")

    def test_policy_is_only_required_when_verifiers_are_running(self):
        gated = local_client(create_app(Settings(database_url="sqlite://", demo_mode=True, verifier_addresses="0x" + "a" * 40, verifier_quorum=1, _env_file=None)))
        response = gated.post("/jobs", json=BODY)
        self.assertEqual(response.status_code, 422)
        self.assertIn("policy.eval_set_hash is missing", response.json()["problems"])

    def test_health(self):
        self.assertEqual(self.client.get("/healthz").json()["status"], "ok")

    def test_public_chain_misconfiguration_cannot_skip_funding_guard(self):
        app = create_app(Settings(database_url="sqlite://", chain_id=84532, rpc_url="http://127.0.0.1:1", _env_file=None))
        self.addCleanup(app.state.engine.dispose)
        client = local_client(app)
        job_id = client.post("/jobs", json=BODY).json()["job_id"]
        response = client.post(f"/jobs/{job_id}/activate")
        self.assertEqual(response.status_code, 409)
        self.assertIn("settlement verification is not configured", response.json()["detail"])

    def test_secure_creation_retains_signed_binding_and_rejects_tampering(self):
        contract = "0x" + "1" * 40
        app = create_app(Settings(database_url="sqlite://", demo_mode=True, chain_id=31337, escrow_address=contract, _env_file=None))
        client = local_client(app)
        wallet_key, client_key = "0x" + "c" * 64, "0x" + "8" * 64
        wallet_address = Account.from_key(wallet_key).address.lower()
        client_address = Account.from_key(client_key).address.lower()
        with Session(app.state.engine) as session:
            wallet = Wallet(address=wallet_address, role="provider")
            session.add(wallet)
            session.flush()
            provider = Provider(wallet_id=wallet.id, name="provider", price_per_step="0.001")
            session.add(provider)
            session.commit()
            provider_id = provider.id
        job_id, onchain_id = new_job_ids(client_address)
        quote = provider_quote(job_id=job_id, onchain_id=onchain_id, chain_id=31337, contract=contract, provider_id=provider_id, rate="0.001", workload=workload_hash(BODY["workload"]), key=wallet_key)
        body = copy.deepcopy(BODY)
        body["workload"]["provider_terms"] = [{"provider_id": provider_id, "wallet": wallet_address, "price_per_step": "0.001", "quote": quote}]
        body.update(job_id=job_id, client_address=client_address)
        body["job_binding"] = job_binding(job_id=job_id, onchain_id=onchain_id, chain_id=31337, contract=contract, spec=body["workload"], policy=body["policy"], budget=body["budget_usdc"], key=client_key)
        altered = copy.deepcopy(body)
        altered["workload"]["provider_terms"][0]["price_per_step"] = "0.5"
        self.assertEqual(client.post("/jobs", json=altered).status_code, 422)
        response = client.post("/jobs", json=body)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["onchain_id"], onchain_id)
        self.assertEqual(client.post("/jobs", json=body).status_code, 409)
        with Session(app.state.engine) as session:
            self.assertEqual(session.query(JobBinding).filter_by(job_id=job_id).one().document, body["job_binding"])

    def test_create_and_read(self):
        created = self.create()
        self.assertEqual(created.status_code, 201)
        job = self.client.get(f"/jobs/{created.json()['job_id']}").json()
        self.assertEqual(job["state"], "OPEN")
        self.assertEqual(job["budget_usdc"], "5.000000")

    def test_idempotent_retry_returns_same_job(self):
        first = self.create(**{"Idempotency-Key": "k1"}).json()
        second = self.create(**{"Idempotency-Key": "k1"}).json()
        self.assertEqual(first["job_id"], second["job_id"])

    def test_no_key_creates_distinct_jobs(self):
        self.assertNotEqual(self.create().json()["job_id"], self.create().json()["job_id"])

    def test_key_reuse_with_different_body_refused(self):
        self.create(**{"Idempotency-Key": "k1"})
        other = dict(BODY, budget_usdc="9.00")
        self.assertEqual(self.client.post("/jobs", json=other, headers={"Idempotency-Key": "k1"}).status_code, 422)

    def test_invalid_input_refused(self):
        for bad in (dict(BODY, budget_usdc="0"), dict(BODY, budget_usdc="-1"), dict(BODY, deadline_seconds=0), dict(BODY, surprise=1)):
            self.assertEqual(self.client.post("/jobs", json=bad).status_code, 422, msg=str(bad))

    def test_unknown_job_is_404(self):
        self.assertEqual(self.client.get("/jobs/nope").status_code, 404)

    def test_cancel_is_repeatable_and_recorded(self):
        job_id = self.create().json()["job_id"]
        self.assertEqual(self.client.post(f"/jobs/{job_id}/cancel").json()["state"], "CANCELLED")
        self.assertEqual(self.client.post(f"/jobs/{job_id}/cancel").json()["state"], "CANCELLED")
        events = self.client.get(f"/jobs/{job_id}/timeline").json()
        self.assertEqual([e["type"] for e in events], ["job.state_changed"])

    def test_fault_is_labelled_injected(self):
        job_id = self.create().json()["job_id"]
        response = self.client.post("/demo/faults", json={"job_id": job_id, "kind": "KILL_CONTAINER"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["label"], "INJECTED")
        event = self.client.get(f"/jobs/{job_id}/timeline").json()[0]
        self.assertEqual((event["type"], event["data"]["label"], event["provenance"]), ("fault.injected", "INJECTED", "LIVE"))

    def test_unknown_fault_refused(self):
        job_id = self.create().json()["job_id"]
        self.assertEqual(self.client.post("/demo/faults", json={"job_id": job_id, "kind": "NOPE"}).status_code, 422)

    def test_faults_disabled_outside_demo_mode(self):
        client = local_client(create_app(Settings(database_url="sqlite://", demo_mode=False, _env_file=None)))
        job_id = client.post("/jobs", json=BODY).json()["job_id"]
        self.assertEqual(client.post("/demo/faults", json={"job_id": job_id, "kind": "KILL_CONTAINER"}).status_code, 403)

    def test_timeline_catch_up(self):
        job_id = self.create().json()["job_id"]
        for _ in range(3):
            self.client.post("/demo/faults", json={"job_id": job_id, "kind": "THERMAL_WARNING"})
        events = self.client.get(f"/jobs/{job_id}/timeline", params={"after": 1}).json()
        self.assertEqual([e["id"] for e in events], [2, 3])

    def test_concurrent_events_never_collide(self):
        import tempfile
        import threading
        from pathlib import Path

        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from ekido_coordinator.events import events_after, record_event
        from ekido_coordinator.models import Base

        with tempfile.TemporaryDirectory() as tmp:
            engine = create_engine(f"sqlite:///{Path(tmp) / 'race.db'}", connect_args={"timeout": 30})
            Base.metadata.create_all(engine)
            errors = []

            def emit():
                try:
                    for _ in range(15):
                        with Session(engine) as session:
                            record_event(session, None, "telemetry.sample", "race-job", {})
                            session.commit()
                except Exception as exc:  # This test exists to surface exactly these.
                    errors.append(exc)

            threads = [threading.Thread(target=emit) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            with Session(engine) as session:
                self.assertEqual(sorted(e["id"] for e in events_after(session, "race-job", 0)), list(range(1, 91)))
            engine.dispose()

    def test_websocket_replays_backlog_then_streams_live(self):
        job_id = self.create().json()["job_id"]
        self.client.post("/demo/faults", json={"job_id": job_id, "kind": "OOM"})
        with self.client.websocket_connect(f"ws://127.0.0.1:8000/ws/jobs/{job_id}") as socket:
            self.assertEqual(socket.receive_json()["data"]["fault_kind"], "OOM")
            self.client.post("/demo/faults", json={"job_id": job_id, "kind": "DISCONNECT"})
            self.assertEqual(socket.receive_json()["data"]["fault_kind"], "DISCONNECT")


if __name__ == "__main__":
    unittest.main()
