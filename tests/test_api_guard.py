import unittest

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from gpuchain_coordinator.main import Settings, create_app

LOCAL = {"base_url": "http://127.0.0.1:8000", "client": ("127.0.0.1", 50000)}
REMOTE = {"base_url": "http://192.168.1.20:8000", "client": ("192.168.1.50", 50000)}
ALLOWED_ORIGIN = "http://127.0.0.1:3100"
WS = "ws://127.0.0.1:8000/ws/jobs/x"


def build(**settings):
    return create_app(Settings(database_url="sqlite://", demo_mode=True, _env_file=None, **settings))


class ApiGuardTest(unittest.TestCase):
    def test_this_machine_can_use_the_api(self):
        self.assertEqual(TestClient(build(), **LOCAL).get("/jobs").status_code, 200)

    def test_ipv6_loopback_counts_as_this_machine(self):
        client = TestClient(build(), base_url="http://[::1]:8000", client=("::1", 50000))
        self.assertEqual(client.get("/jobs").status_code, 200)

    def test_another_machine_is_refused_every_operator_route(self):
        client = TestClient(build(), **REMOTE)
        for method, path in (("GET", "/jobs"), ("POST", "/jobs"), ("POST", "/demo/faults"), ("GET", "/providers"), ("POST", "/providers"), ("POST", "/jobs/x/cancel")):
            with self.subTest(path=path, method=method):
                response = client.request(method, path, json={})
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json()["error"], "API_LOCKED")

    def test_no_token_set_means_an_empty_bearer_opens_nothing(self):
        client = TestClient(build(), **REMOTE)
        self.assertEqual(client.get("/jobs", headers={"Authorization": "Bearer "}).status_code, 401)

    def test_another_machine_needs_the_exact_token(self):
        client = TestClient(build(api_token="right-token"), **REMOTE)
        self.assertEqual(client.get("/jobs", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(client.get("/jobs", headers={"Authorization": "Bearer right-token"}).status_code, 200)

    def test_a_rebinding_page_is_refused_even_from_loopback(self):
        client = TestClient(build(), base_url="http://evil.example:8000", client=("127.0.0.1", 50000))
        self.assertEqual(client.get("/jobs").status_code, 401)

    def test_a_request_through_a_local_proxy_is_not_treated_as_local(self):
        response = TestClient(build(), **LOCAL).get("/jobs", headers={"X-Forwarded-For": "203.0.113.9"})
        self.assertEqual(response.status_code, 401)

    def test_a_tunnel_reaches_only_the_routes_with_their_own_proof(self):
        """Cloudflare Tunnel and ngrok connect from loopback, under their own host name, and add X-Forwarded-For."""
        client = TestClient(build(), base_url="https://demo.example-tunnel.dev", client=("127.0.0.1", 50000))
        via = {"X-Forwarded-For": "198.51.100.7"}
        for method, path in (("GET", "/jobs"), ("POST", "/demo/faults"), ("POST", "/jobs/x/cancel"), ("GET", "/providers")):
            with self.subTest(path=path):
                self.assertEqual(client.request(method, path, json={}, headers=via).status_code, 401)
        self.assertEqual(client.get("/healthz", headers=via).status_code, 200)
        self.assertNotIn("API_LOCKED", client.get("/artifacts/" + "0" * 64, headers=via).text)

    def test_a_foreign_website_is_refused_even_on_this_machine(self):
        client = TestClient(build(), **LOCAL)
        self.assertEqual(client.post("/jobs/x/cancel", headers={"Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(client.get("/jobs", headers={"Origin": ALLOWED_ORIGIN}).status_code, 200)

    def test_the_dashboard_preflight_still_works(self):
        response = TestClient(build(), **LOCAL).options("/jobs", headers={"Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "POST"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], ALLOWED_ORIGIN)

    def test_job_stream_follows_the_same_rule(self):
        app = build()
        with TestClient(app, **LOCAL).websocket_connect(WS) as socket:
            socket.close()
        with self.assertRaises(WebSocketDisconnect):
            with TestClient(app, **REMOTE).websocket_connect(WS):
                pass
        with self.assertRaises(WebSocketDisconnect):
            with TestClient(app, **LOCAL).websocket_connect(WS, headers={"Origin": "https://evil.example"}):
                pass

    def test_routes_with_their_own_proof_stay_reachable_from_other_machines(self):
        client = TestClient(build(), **REMOTE)
        self.assertEqual(client.get("/healthz").status_code, 200)
        # They answer for themselves, so this is their own refusal and not the guard's.
        artifact = client.get("/artifacts/" + "0" * 64)
        self.assertNotIn("API_LOCKED", artifact.text)
        self.assertNotIn("API_LOCKED", client.post("/verifier/session", json={}).text)

    def test_providers_show_what_a_payout_check_needs(self):
        client = TestClient(build(), **LOCAL)
        wallet = "0x" + "AB" * 20
        client.post("/providers", json={"name": "A", "wallet_address": wallet, "kind": "REAL", "capacity_mib": 6144, "price_per_step": "0.001"})
        row = client.get("/providers").json()[0]
        self.assertEqual(row["wallet_address"], wallet.lower())
        self.assertIn("enrolled_key", row)

    def test_settlements_list_is_for_this_machine_only(self):
        self.assertEqual(TestClient(build(), **LOCAL).get("/settlements").json(), [])
        self.assertEqual(TestClient(build(), **REMOTE).get("/settlements").status_code, 401)

    def test_a_public_server_lets_anyone_watch_but_not_change_anything(self):
        client = TestClient(build(public_read=True), **REMOTE)
        self.assertEqual(client.get("/jobs").status_code, 200)
        self.assertEqual(client.get("/providers").status_code, 200)
        for method, path in (("POST", "/jobs"), ("POST", "/demo/faults"), ("POST", "/jobs/x/cancel"), ("POST", "/providers")):
            with self.subTest(path=path):
                self.assertEqual(client.request(method, path, json={}).status_code, 401)
        with TestClient(build(public_read=True), base_url="http://192.168.1.20:8000", client=("192.168.1.50", 50000)).websocket_connect("ws://192.168.1.20:8000/ws/jobs/x") as socket:
            socket.close()

    def test_open_api_is_for_a_port_published_only_on_loopback(self):
        client = TestClient(build(open_api=True), base_url="http://127.0.0.1:8000", client=("172.17.0.1", 50000))
        self.assertEqual(client.get("/jobs").status_code, 200)
        self.assertEqual(client.get("/jobs", headers={"Origin": "https://evil.example"}).status_code, 403)


if __name__ == "__main__":
    unittest.main()
