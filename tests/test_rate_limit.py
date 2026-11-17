import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ekido_coordinator.rate_limit import RateLimiter, TokenBuckets, client_key


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def app_with(clock, reads=3, writes=2):
    app = FastAPI()

    @app.get("/r")
    def r():
        return {"ok": True}

    @app.post("/w")
    def w():
        return {"ok": True}

    app.add_middleware(RateLimiter, reads_per_minute=reads, writes_per_minute=writes, clock=clock)
    return app


class RateLimitTest(unittest.TestCase):
    def test_reads_and_writes_have_separate_budgets(self):
        clock = Clock()
        c = TestClient(app_with(clock), client=("203.0.113.5", 1))
        self.assertEqual([c.get("/r").status_code for _ in range(4)], [200, 200, 200, 429])
        self.assertEqual([c.post("/w").status_code for _ in range(3)], [200, 200, 429])

    def test_refusal_says_when_to_retry_and_budget_refills(self):
        clock = Clock()
        c = TestClient(app_with(clock, writes=2), client=("203.0.113.5", 1))
        c.post("/w"), c.post("/w")
        r = c.post("/w")
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json()["error"], "RATE_LIMITED")
        self.assertGreaterEqual(int(r.headers["retry-after"]), 1)
        clock.t += 31  # 2 per minute: one token back after 30 s.
        self.assertEqual(c.post("/w").status_code, 200)

    def test_each_client_behind_the_proxy_is_limited_on_its_own(self):
        clock = Clock()
        app = app_with(clock, reads=1)
        proxy = TestClient(app, client=("127.0.0.1", 1))
        self.assertEqual(proxy.get("/r", headers={"X-Forwarded-For": "198.51.100.1"}).status_code, 200)
        self.assertEqual(proxy.get("/r", headers={"X-Forwarded-For": "198.51.100.1"}).status_code, 429)
        self.assertEqual(proxy.get("/r", headers={"X-Forwarded-For": "198.51.100.2"}).status_code, 200)

    def test_this_machine_is_not_limited_and_remote_callers_cannot_pretend(self):
        clock = Clock()
        app = app_with(clock, reads=1)
        local = TestClient(app, client=("127.0.0.1", 1))
        self.assertEqual([local.get("/r").status_code for _ in range(5)], [200] * 5)
        remote = TestClient(app, client=("203.0.113.9", 1))
        codes = [remote.get("/r", headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code for i in range(3)]
        self.assertEqual(codes, [200, 429, 429])

    def test_client_key(self):
        self.assertIsNone(client_key({"client": ("127.0.0.1", 1), "headers": []}))
        self.assertEqual(client_key({"client": ("127.0.0.1", 1), "headers": [(b"x-forwarded-for", b"1.2.3.4, 10.0.0.1")]}), "1.2.3.4")
        self.assertEqual(client_key({"client": ("5.6.7.8", 1), "headers": [(b"x-forwarded-for", b"9.9.9.9")]}), "5.6.7.8")

    def test_memory_stays_bounded(self):
        from ekido_coordinator import rate_limit

        old = rate_limit.MAX_TRACKED
        rate_limit.MAX_TRACKED = 8
        try:
            b = TokenBuckets(60, Clock())
            for i in range(20):
                b.take(f"ip{i}")
            self.assertLessEqual(len(b.buckets), 8)
        finally:
            rate_limit.MAX_TRACKED = old


class BrowserReadableRefusalTest(unittest.TestCase):
    """The real app: a refusal must reach the dashboard as a 429 it can read, not as a CORS network error."""

    def app(self, reads):
        from ekido_coordinator.main import Settings, create_app
        return create_app(Settings(database_url="sqlite://", demo_mode=True, _env_file=None,
                                   rate_limit_reads_per_minute=reads, cors_origins=["https://ekido.example"]))

    def test_a_refusal_carries_cors_and_retry_after(self):
        c = TestClient(self.app(reads=2), client=("203.0.113.7", 1))
        origin = {"Origin": "https://ekido.example"}
        statuses = [c.get("/healthz", headers=origin).status_code for _ in range(3)]
        self.assertEqual(statuses[-1], 429)
        refused = c.get("/healthz", headers=origin)
        self.assertEqual(refused.status_code, 429)
        self.assertEqual(refused.headers.get("access-control-allow-origin"), "https://ekido.example")
        self.assertIn("retry-after", refused.headers.get("access-control-expose-headers", "").lower())
        self.assertGreaterEqual(int(refused.headers["retry-after"]), 1)

    def test_preflights_do_not_spend_the_read_budget(self):
        c = TestClient(self.app(reads=2), client=("203.0.113.8", 1))
        preflight = {"Origin": "https://ekido.example", "Access-Control-Request-Method": "GET"}
        for _ in range(5):
            self.assertEqual(c.options("/healthz", headers=preflight).status_code, 200)
        self.assertEqual(c.get("/healthz", headers={"Origin": "https://ekido.example"}).status_code, 200)

    def test_default_leaves_room_for_a_shared_network(self):
        from ekido_coordinator.main import Settings
        self.assertGreaterEqual(Settings(_env_file=None).rate_limit_reads_per_minute, 600)


if __name__ == "__main__":
    unittest.main()
