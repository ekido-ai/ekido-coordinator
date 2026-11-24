"""What a public visitor sees: every provider identity replaced, everything else intact (decision 8).

Pure tests of the redactor plus the guard driven as a raw ASGI app, so they need no HTTP client library.
"""
try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import asyncio
import json
import unittest

from ekido_coordinator.api_guard import ApiGuard
from ekido_coordinator.public_view import Redactor

PID = "6f1c2a9e-0000-4000-8000-000000000001"
OTHER = "6f1c2a9e-0000-4000-8000-000000000002"
IDENTITIES = {
    PID: PID, "alice-rtx3060": PID, "0xAbCdEf0000000000000000000000000000000001": PID, "0x00000000000000000000000000000000000000k1": PID,
    OTHER: OTHER, "bob-t4": OTHER,
}


def redactor(identities=IDENTITIES) -> Redactor:
    r = Redactor(lambda: dict(identities), secret="test-secret")
    r.refresh(force=True)
    return r


class RedactorTest(unittest.TestCase):
    def test_ids_names_and_addresses_become_the_same_label(self):
        r = redactor()
        label = r.label(PID)
        self.assertRegex(label, r"^provider-[0-9a-f]{6}$")
        out = r.scrub({"provider_id": PID, "detail": "alice-rtx3060 dropped", "signer": "0xabcdef0000000000000000000000000000000001"})
        self.assertEqual(out["provider_id"], label)
        self.assertEqual(out["detail"], f"{label} dropped")
        self.assertEqual(out["signer"], label)  # matched case-insensitively

    def test_two_providers_get_different_labels_and_text_keeps_its_sense(self):
        r = redactor()
        out = r.scrub({"message": "alice-rtx3060 lost; bob-t4 took over at step 120"})
        self.assertEqual(out["message"], f"{r.label(PID)} lost; {r.label(OTHER)} took over at step 120")

    def test_dict_keys_are_scrubbed_too(self):
        r = redactor()
        out = r.scrub({"payouts": {"alice-rtx3060": "0.12", "bob-t4": "0.03"}})
        self.assertEqual(out["payouts"], {r.label(PID): "0.12", r.label(OTHER): "0.03"})

    def test_unknown_values_under_identity_keys_are_hidden_anyway(self):
        r = redactor({})
        out = r.scrub({"target_provider_id": "gone-provider", "payees": ["0x1111111111111111111111111111111111111111"]})
        self.assertTrue(out["target_provider_id"].startswith("provider-"))
        self.assertNotIn("0x1111", json.dumps(out))

    def test_everything_else_is_left_alone(self):
        r = redactor()
        job = {"job_id": "j-1", "state": "RUNNING", "name": "review-sentiment", "budget_usdc": "5.0", "steps": 400,
               "spec": {"model": "Qwen/Qwen2.5-0.5B", "dataset": {"name": "reviews"}}, "ok": True, "fee": None}
        self.assertEqual(r.scrub(job), job)

    def test_a_short_name_is_matched_only_as_a_whole_value(self):
        r = redactor({"t4": PID})
        self.assertEqual(r.scrub({"gpu": "t4"})["gpu"], r.label(PID))
        self.assertEqual(r.scrub({"detail": "step t4 rest"})["detail"], "step t4 rest")  # too short to match in text

    def test_a_name_inside_a_longer_word_is_not_touched(self):
        r = redactor()
        self.assertEqual(r.scrub("alice-rtx30601"), "alice-rtx30601")

    def test_a_providers_quote_signature_is_hidden(self):
        # Anyone can recover the signing key from a signature, which would undo the labels.
        r = redactor()
        sig = "0x" + "ab" * 65
        out = r.scrub({"spec": {"provider_terms": [{"quote": {"contract": "0x1bf6", "signature": sig}}]}, "client_signature": {"signature": sig}})
        self.assertNotIn(sig, json.dumps(out["spec"]))
        self.assertEqual(out["spec"]["provider_terms"][0]["quote"]["contract"], "0x1bf6")
        self.assertEqual(out["client_signature"]["signature"], sig)  # signatures outside provider_terms are left alone

    def test_labels_cannot_be_reproduced_without_the_secret(self):
        self.assertNotEqual(Redactor(lambda: {}, secret="a").label(PID), Redactor(lambda: {}, secret="b").label(PID))

    def test_non_json_text_is_scrubbed_as_text(self):
        self.assertEqual(redactor().scrub_text("hello alice-rtx3060"), f"hello {redactor().label(PID)}")


def run_asgi(guard: ApiGuard, scope: dict, body_messages: list[dict]) -> list[dict]:
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    async def go():
        await guard(scope, receive, send)

    asyncio.run(go())
    return sent


def http_scope(path: str, method: str = "GET", client=("192.168.1.50", 50000), headers=None) -> dict:
    return {"type": "http", "path": path, "method": method, "client": client,
            "headers": [(b"host", b"api.example")] + (headers or [])}


class GuardRedactionTest(unittest.TestCase):
    def app(self, payload: bytes, content_type: bytes = b"application/json"):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", content_type), (b"content-length", str(len(payload)).encode())]})
            half = len(payload) // 2  # in two chunks, to check the body is joined first
            await send({"type": "http.response.body", "body": payload[:half], "more_body": True})
            await send({"type": "http.response.body", "body": payload[half:]})
        return app

    def guard(self, payload: bytes, **kw) -> ApiGuard:
        return ApiGuard(self.app(payload, **kw), token="op", allowed_origins=[], public_read=True, redactor=redactor())

    def test_a_public_job_page_leaves_without_provider_identities(self):
        payload = json.dumps({"job_id": "j-1", "provider_id": PID, "detail": "alice-rtx3060 joined"}).encode()
        sent = run_asgi(self.guard(payload), http_scope("/jobs/j-1/timeline"), [])
        body = json.loads(sent[-1]["body"])
        self.assertNotIn("alice-rtx3060", sent[-1]["body"].decode())
        self.assertNotIn(PID, sent[-1]["body"].decode())
        self.assertEqual(body["job_id"], "j-1")
        length = dict(sent[0]["headers"])[b"content-length"]
        self.assertEqual(int(length), len(sent[-1]["body"]))  # the length header matches the new body

    def test_the_operator_token_gets_the_full_response(self):
        payload = json.dumps({"provider_id": PID}).encode()
        sent = run_asgi(self.guard(payload), http_scope("/jobs/j-1", headers=[(b"authorization", b"Bearer op")]), [])
        self.assertEqual(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"), payload)

    def test_the_provider_list_stays_locked(self):
        sent = run_asgi(self.guard(b"[]"), http_scope("/providers"), [])
        self.assertEqual(sent[0]["status"], 401)

    def test_a_public_write_stays_locked(self):
        sent = run_asgi(self.guard(b"{}"), http_scope("/jobs", method="POST"), [])
        self.assertEqual(sent[0]["status"], 401)

    def test_a_non_json_public_response_is_hidden(self):
        sent = run_asgi(self.guard(b"alice-rtx3060", content_type=b"text/plain"), http_scope("/jobs/j-1"), [])
        self.assertNotIn(b"alice", sent[-1]["body"])
        self.assertIn(b"HIDDEN", sent[-1]["body"])

    def test_without_a_redactor_job_pages_stay_locked(self):
        guard = ApiGuard(self.app(b"{}"), token="op", allowed_origins=[], public_read=True)
        self.assertEqual(run_asgi(guard, http_scope("/jobs"), [])[0]["status"], 401)

    def test_the_job_stream_is_scrubbed(self):
        async def app(scope, receive, send):
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.send", "text": json.dumps({"event": "provider.lost", "provider_id": PID, "name": "alice-rtx3060"})})
        guard = ApiGuard(app, token="op", allowed_origins=[], public_read=True, redactor=redactor())
        sent: list[dict] = []

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            sent.append(message)

        asyncio.run(guard({"type": "websocket", "path": "/ws/jobs/j-1", "client": ("192.168.1.50", 1), "headers": [(b"host", b"api.example")]}, receive, send))
        text = sent[-1]["text"]
        self.assertNotIn("alice-rtx3060", text)
        self.assertNotIn(PID, text)
        self.assertEqual(json.loads(text)["event"], "provider.lost")


class LoadIdentitiesTest(unittest.TestCase):
    def test_names_wallets_and_keys_come_from_the_database(self):
        from decimal import Decimal

        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from ekido_coordinator.models import Base, Provider, ProviderKey, Wallet
        from ekido_coordinator.public_view import load_identities

        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        with Session(engine) as s:
            wallet = Wallet(address="0x" + "cd" * 20, role="provider")
            s.add(wallet); s.flush()
            provider = Provider(wallet_id=wallet.id, name="carol-a100", kind="REAL", price_per_step=Decimal("0"))
            s.add(provider); s.flush()
            s.add(ProviderKey(provider_id=provider.id, key="0x" + "ef" * 20))
            s.commit()
            pid = provider.id
        found = load_identities(engine)
        for value in (pid, "carol-a100", "0x" + "cd" * 20, "0x" + "ef" * 20):
            self.assertEqual(found.get(value), pid, value)
        r = Redactor(lambda: load_identities(engine), secret="s"); r.refresh(force=True)
        self.assertNotIn("carol-a100", json.dumps(r.scrub({"detail": "carol-a100 joined", "signer": "0x" + "EF" * 20})))


if __name__ == "__main__":
    unittest.main()
