import unittest
from decimal import Decimal

from ekido_coordinator.scheduler import Candidate, Requirements, failure_probability, rank


def candidate(pid="a", price="0.01", speed=10.0, peak=1000.0, cap=3000.0, total=10, failed=0, online=True, excluded=False):
    return Candidate(pid, online, excluded, Decimal(price), speed, peak, cap, total, failed)


REQ = Requirements(remaining_steps=100, seconds_to_deadline=600, remaining_budget=Decimal("5"))


class Ranking(unittest.TestCase):
    def test_new_provider_has_no_invented_reputation(self):
        self.assertEqual(failure_probability(0, 0), 0.5)

    def test_cheaper_wins_with_equal_history(self):
        ranked, _ = rank([candidate("pricey", price="0.02"), candidate("cheap", price="0.01")], REQ)
        self.assertEqual([r.provider_id for r in ranked], ["cheap", "pricey"])

    def test_unreliable_can_lose_despite_lower_price(self):
        good = candidate("reliable", price="0.012", total=20, failed=0)
        bad = candidate("flaky", price="0.01", total=20, failed=10)
        ranked, _ = rank([bad, good], REQ)
        self.assertEqual(ranked[0].provider_id, "reliable")

    def test_expected_cost_working_is_exposed(self):
        ranked, _ = rank([candidate(total=0, failed=0)], REQ)
        self.assertEqual(ranked[0].p_fail, 0.5)
        self.assertFalse(ranked[0].has_history)
        self.assertEqual(ranked[0].expected_cost, Decimal("2"))
        self.assertEqual(ranked[0].eta_seconds, 10)

    def test_tie_goes_to_headroom_then_id(self):
        ranked, _ = rank([candidate("b", cap=2000.0), candidate("a", cap=2000.0), candidate("c", cap=4000.0)], REQ)
        self.assertEqual([r.provider_id for r in ranked], ["c", "a", "b"])

    def test_hard_constraints_reject_with_reasons(self):
        ranked, rejected = rank(
            [
                candidate("off", online=False),
                candidate("out", excluded=True),
                candidate("unprobed", speed=None),
                candidate("slow", speed=0.1),
                candidate("expensive", price="1"),
                candidate("ok"),
            ],
            REQ,
        )
        self.assertEqual([r.provider_id for r in ranked], ["ok"])
        self.assertEqual(
            {r.provider_id: r.code for r in rejected},
            {"off": "OFFLINE", "out": "EXCLUDED", "unprobed": "PROBE_ERROR", "slow": "DEADLINE", "expensive": "BUDGET"},
        )

    def test_no_candidates_is_not_an_error(self):
        self.assertEqual(rank([], REQ), ([], []))


if __name__ == "__main__":
    unittest.main()
