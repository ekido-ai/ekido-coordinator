import time
import unittest

from eth_account import Account
from fastapi.testclient import TestClient

from gpuchain_core.signing import sign_document
from gpuchain_coordinator.main import Settings, create_app

WALLET_KEY = "0x" + "5" * 64
WALLET = Account.from_key(WALLET_KEY).address
REMOTE = {"base_url": "http://192.168.1.20:8000", "client": ("192.168.1.50", 50000)}
LOCAL = {"base_url": "http://127.0.0.1:8000", "client": ("127.0.0.1", 50000)}


def doc(**changes):
    base = {"kind": "gpuchain.provider_join.v1", "name": "Vaibhav RTX 4060", "wallet": WALLET, "price_per_step": "0.002", "device": "cuda", "capacity_mib": 8192, "ts": int(time.time())}
    return {**base, **changes}


def signed(document, key=WALLET_KEY):
    return {"document": document, "signature": sign_document(document, key)[1]}


class ProviderJoinTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", demo_mode=True, _env_file=None))
        self.remote = TestClient(self.app, **REMOTE)

    def test_a_provider_on_another_machine_can_sign_itself_up(self):
        response = self.remote.post("/providers/join", json=signed(doc()))
        self.assertEqual(response.status_code, 201, response.text)
        provider = next(p for p in TestClient(self.app, **LOCAL).get("/providers").json() if p["provider_id"] == response.json()["provider_id"])
        self.assertEqual(provider["wallet_address"], WALLET.lower())
        self.assertEqual(provider["price_per_step"], "0.002000")
        self.assertEqual(provider["capacity_mib"], 8192)

    def test_joining_twice_returns_the_same_provider(self):
        first = self.remote.post("/providers/join", json=signed(doc())).json()
        second = self.remote.post("/providers/join", json=signed(doc()))
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["provider_id"], first["provider_id"])

    def test_someone_else_cannot_sign_up_your_wallet(self):
        response = self.remote.post("/providers/join", json=signed(doc(), key="0x" + "6" * 64))
        self.assertEqual(response.status_code, 401)

    def test_an_old_or_malformed_request_is_refused(self):
        for bad in (doc(ts=int(time.time()) - 3600), doc(price_per_step="5"), doc(price_per_step="0"), doc(device="tpu"), doc(name=""), doc(kind="other")):
            with self.subTest(bad=bad):
                self.assertEqual(self.remote.post("/providers/join", json=signed(bad)).status_code, 422)

    def test_joins_are_rate_limited(self):
        codes = [self.remote.post("/providers/join", json=signed(doc(kind="x"))).status_code for _ in range(12)]
        self.assertIn(429, codes)

    def test_the_operator_api_stays_locked_from_other_machines(self):
        self.assertEqual(self.remote.get("/jobs").status_code, 401)
        self.assertEqual(self.remote.post("/providers", json={}).status_code, 401)


if __name__ == "__main__":
    unittest.main()


class JoinAbuseTest(unittest.TestCase):
    """Loopholes found on 9 Nov: a proxy hid every caller's address, names could impersonate."""

    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", demo_mode=True, _env_file=None))

    def key(self, n):
        return "0x" + format(n + 1000, "x").rjust(64, "0")

    def join_as(self, n, client, name=None, headers=None):
        k = self.key(n)
        d = doc(wallet=Account.from_key(k).address, name=name or f"GPU number {n}")
        return client.post("/providers/join", json=signed(d, k), headers=headers or {})

    def test_behind_the_proxy_each_forwarded_address_has_its_own_limit(self):
        proxy = TestClient(self.app, **LOCAL)
        for n in range(5):
            self.assertEqual(self.join_as(n, proxy, headers={"X-Forwarded-For": "203.0.113.7"}).status_code, 201)
        self.assertEqual(self.join_as(5, proxy, headers={"X-Forwarded-For": "203.0.113.7"}).json()["error"], "TOO_MANY_PROVIDERS")
        self.assertEqual(self.join_as(6, proxy, headers={"X-Forwarded-For": "198.51.100.9"}).status_code, 201)

    def test_a_remote_caller_cannot_forge_its_address(self):
        remote = TestClient(self.app, **REMOTE)
        for n in range(5):
            self.assertEqual(self.join_as(n, remote, headers={"X-Forwarded-For": f"10.0.0.{n}"}).status_code, 201)
        self.assertEqual(self.join_as(9, remote, headers={"X-Forwarded-For": "10.9.9.9"}).status_code, 429)

    def test_names_must_be_unique_and_printable(self):
        remote = TestClient(self.app, **REMOTE)
        self.assertEqual(self.join_as(1, remote, name="Fast 4090").status_code, 201)
        taken = self.join_as(2, remote, name="fast 4090")
        self.assertEqual(taken.status_code, 409)
        self.assertEqual(taken.json()["error"], "NAME_TAKEN")
        spoof = self.join_as(3, remote, name="Provider‮A")
        self.assertEqual(spoof.status_code, 422)
        self.assertIn("printable", " ".join(spoof.json()["problems"]))


class MetadataHashTest(unittest.TestCase):
    def test_providers_publish_the_hash_the_provider_commits_on_chain(self):
        from gpuchain_core.provider_meta import metadata_hash, provider_metadata

        app = create_app(Settings(database_url="sqlite://", demo_mode=True, _env_file=None))
        joined = TestClient(app, **REMOTE).post("/providers/join", json=signed(doc())).json()
        row = next(p for p in TestClient(app, **LOCAL).get("/providers").json() if p["provider_id"] == joined["provider_id"])
        self.assertEqual(row["metadata_hash"], metadata_hash(provider_metadata("Vaibhav RTX 4060", WALLET, "0.002", 8192)))
