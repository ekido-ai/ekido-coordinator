"""A job paid in rupees is settled or refunded through the Platform when it ends."""

try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import io
import unittest
import urllib.error
from decimal import Decimal

from eth_account import Account
from sqlalchemy.orm import Session

from ekido_coordinator import jobs
from ekido_coordinator.fiat_settlement import FiatSettlement, HttpPlatformPayments, NoOrder, split_paise
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import Checkpoint, Provider, Wallet


class FakePlatform:
    def __init__(self, budget=10_000, known=True):
        self.budget, self.known = budget, known
        self.settled, self.refunded = [], []

    def order(self, job_id):
        if not self.known:
            raise NoOrder(job_id)
        return {"order_id": "o", "status": "HELD", "provider_budget_paise": self.budget}

    def settle(self, job_id, payouts):
        self.settled.append((job_id, payouts))

    def refund(self, job_id):
        if not self.known:
            raise NoOrder(job_id)
        self.refunded.append(job_id)


class Split(unittest.TestCase):
    def test_always_adds_up_exactly_and_follows_the_earnings(self):
        for budget in (1, 7, 100, 10_001, 999_999):
            for earned in ({"a": Decimal(1), "b": Decimal(1), "c": Decimal(1)}, {"a": Decimal("0.3"), "b": Decimal("0.7")}, {"a": Decimal(5)}):
                with self.subTest(budget=budget, earned=earned):
                    out = split_paise(budget, earned)
                    self.assertEqual(sum(out.values()), budget)
                    self.assertTrue(all(isinstance(v, int) and v > 0 for v in out.values()))
        self.assertEqual(split_paise(100, {"a": Decimal(3), "b": Decimal(1)}), {"a": 75, "b": 25})

    def test_nothing_earned_or_no_budget_gives_no_split(self):
        self.assertEqual(split_paise(100, {}), {})
        self.assertEqual(split_paise(100, {"a": Decimal(0)}), {})
        self.assertEqual(split_paise(0, {"a": Decimal(1)}), {})


class Settle(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None))
        self.engine = self.app.state.engine
        self.platform = FakePlatform()
        self.service = FiatSettlement(self.engine, self.platform)
        self.keys = {}
        with Session(self.engine) as s:
            for name, rate in (("A", "0.001"), ("B", "0.003")):
                key = "0x" + ("1" if name == "A" else "2") * 64
                wallet = Wallet(address=Account.from_key(key).address.lower(), role="provider")
                s.add(wallet)
                s.flush()
                provider = Provider(name=name, wallet_id=wallet.id, kind="REAL", price_per_step=Decimal(rate), enrolled_key="signer-" + name.lower())
                s.add(provider)
                s.flush()
                self.keys[name] = provider.id
            s.commit()

    def job(self, state, accepted=((10, "a"), (40, "b"))):
        with Session(self.engine) as s:
            job = jobs.create_job(s, spec={"n": state + str(len(accepted))}, policy={}, budget=Decimal("5"), deadline_seconds=600)
            job.state = state
            parent = None
            for i, (step, who) in enumerate(accepted):
                digest = f"{job.id[:8]}{i}".ljust(64, "f")
                s.add(Checkpoint(job_id=job.id, manifest_hash=digest, parent_hash=parent, logical_step=step, fencing_token=1, state="ACCEPTED", manifest={}, signer=f"signer-{who}"))
                parent = digest
            s.commit()
            return job.id

    def test_a_completed_job_pays_each_provider_by_what_it_earned_and_exactly_the_budget(self):
        job_id = self.job("COMPLETED")  # A: 10 steps x 0.001 = 0.01; B: 30 steps x 0.003 = 0.09
        self.assertEqual(self.service.on_job_finished(job_id), "settled")
        (called, payouts), = self.platform.settled
        self.assertEqual(called, job_id)
        self.assertEqual(payouts, {self.keys["A"]: 1000, self.keys["B"]: 9000})
        self.assertEqual(self.platform.refunded, [])

    def test_the_whole_provider_budget_is_shared_even_when_the_work_cost_less(self):
        # The product rule written in the module docstring: steps x rate only sets each share, never the total.
        job_id = self.job("COMPLETED", accepted=((1, "a"),))  # one step at 0.001: far less than the 10,000 paise budget
        self.assertEqual(self.service.on_job_finished(job_id), "settled")
        (_, payouts), = self.platform.settled
        self.assertEqual(payouts, {self.keys["A"]: 10_000})
        self.assertEqual(sum(payouts.values()), self.platform.budget)

    def test_a_failed_or_cancelled_job_refunds_the_customer(self):
        for state in ("FAILED", "CANCELLED"):
            job_id = self.job(state, accepted=((5, "a"),))
            self.assertEqual(self.service.on_job_finished(job_id), "refunded")
            self.assertIn(job_id, self.platform.refunded)
        self.assertEqual(self.platform.settled, [])

    def test_a_passed_job_that_earned_nothing_is_refunded_not_settled_empty(self):
        job_id = self.job("COMPLETED", accepted=((5, "a"),))
        with Session(self.engine) as s:
            s.get(Provider, self.keys["A"]).price_per_step = Decimal(0)
            s.commit()
        self.assertEqual(self.service.on_job_finished(job_id), "refunded")
        self.assertEqual(self.platform.settled, [])

    def test_a_job_with_no_paid_order_is_ignored_and_a_running_job_is_refused(self):
        self.platform.known = False
        self.assertEqual(self.service.on_job_finished(self.job("COMPLETED")), "not-fiat")
        self.assertEqual(self.service.on_job_finished(self.job("FAILED", accepted=((5, "a"),))), "not-fiat")
        with self.assertRaises(ValueError):
            self.service.on_job_finished(self.job("RUNNING"))
        with self.assertRaises(ValueError):
            self.service.on_job_finished("missing")

    def test_a_platform_that_cannot_be_reached_raises_and_the_sweep_counts_it(self):
        class Down(FakePlatform):
            def order(self, job_id):
                raise RuntimeError("the Platform answered 503")

        self.service.platform = Down()
        job_id = self.job("COMPLETED")
        with self.assertRaises(RuntimeError):
            self.service.on_job_finished(job_id)
        with self.assertLogs("ekido_coordinator.fiat_settlement", level="ERROR"):
            self.assertEqual(self.service.sweep()["error"], 1)

    def test_the_sweep_finishes_what_was_missed_and_repeating_it_calls_the_platform_again_safely(self):
        self.job("COMPLETED")
        self.job("FAILED", accepted=((5, "a"),))
        self.job("RUNNING")
        self.assertEqual(self.service.sweep(), {"settled": 1, "refunded": 1, "not-fiat": 0, "error": 0})

    def test_an_unknown_signer_is_an_error_not_a_guess(self):
        job_id = self.job("COMPLETED", accepted=((5, "z"),))
        with self.assertRaises(RuntimeError):
            self.service.on_job_finished(job_id)


class Wiring(unittest.TestCase):
    def test_off_by_default_and_on_only_with_both_settings(self):
        self.assertIsNone(create_app(Settings(database_url="sqlite://", _env_file=None)).state.fiat_settlement)
        for extra in ({"platform_url": "http://127.0.0.1:8010"}, {"platform_internal_token": "t"}):
            app = create_app(Settings(database_url="sqlite://", _env_file=None, **extra))
            self.assertIsNone(app.state.fiat_settlement)
        app = create_app(Settings(database_url="sqlite://", _env_file=None, platform_url="http://127.0.0.1:8010", platform_internal_token="t"))
        self.assertIsInstance(app.state.fiat_settlement, FiatSettlement)

    def test_the_orchestrator_calls_it_when_a_job_ends(self):
        app = create_app(Settings(database_url="sqlite://", _env_file=None, platform_url="http://127.0.0.1:8010", platform_internal_token="t"))
        calls = []
        app.state.fiat_settlement.on_job_finished = calls.append
        app.state.gateway.orchestrator._notify_finished("job-1")
        self.assertEqual(calls, ["job-1"])


class Http(unittest.TestCase):
    def make(self, reply=None, code=None):
        self.seen = []

        class Response:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, n=-1): return __import__("json").dumps(self.body).encode()

        def opener(request, timeout):
            self.seen.append(request)
            if code:
                raise urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b""))
            return Response(reply)

        return HttpPlatformPayments("http://127.0.0.1:8010/", "tok-secret", opener=opener)

    def test_requests_carry_the_token_in_the_header_only(self):
        client = self.make(reply={"provider_budget_paise": 5})
        client.order("j1")
        client.settle("j1", {"p": 5})
        client.refund("j1")
        get, settle, refund = self.seen
        self.assertEqual((get.method, get.full_url), ("GET", "http://127.0.0.1:8010/internal/payments/jobs/j1"))
        self.assertEqual(settle.data, b'{"payouts": {"p": 5}}')
        self.assertEqual(refund.full_url, "http://127.0.0.1:8010/internal/payments/jobs/j1/refund")
        for request in self.seen:
            self.assertEqual(request.get_header("X-internal-token"), "tok-secret")
            self.assertNotIn("tok-secret", request.full_url)

    def test_404_means_no_order_and_other_errors_name_only_the_status(self):
        with self.assertRaises(NoOrder):
            self.make(code=404).order("j")
        with self.assertRaises(RuntimeError) as caught:
            self.make(code=500).refund("j")
        self.assertIn("500", str(caught.exception))
        self.assertNotIn("tok-secret", str(caught.exception))
        with self.assertRaises(ValueError):
            HttpPlatformPayments("http://x", "")


if __name__ == "__main__":
    unittest.main()
