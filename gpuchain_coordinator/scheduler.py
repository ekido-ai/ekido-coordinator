"""Provider ranking: hard constraints first, then expected cost to finish.

There are no invented weights. Every input is a measured number or a declared price, and the
result carries its working so the dashboard can show why a provider won.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Candidate:
    provider_id: str
    online: bool
    excluded: bool  # For example, already failed or was rejected on this job.
    price_per_step: Decimal  # USDC per logical step.
    measured_steps_per_s: float | None  # From a real probe. None means it hasn't been probed.
    peak_mem_mib: float | None
    capacity_mib: float | None
    jobs_total: int
    jobs_failed: int


@dataclass(frozen=True)
class Requirements:
    remaining_steps: int
    seconds_to_deadline: float
    remaining_budget: Decimal


@dataclass(frozen=True)
class Ranked:
    provider_id: str
    p_fail: float
    has_history: bool
    eta_seconds: float
    expected_cost: Decimal
    headroom_mib: float


@dataclass(frozen=True)
class Rejected:
    provider_id: str
    code: str
    reason: str


def failure_probability(jobs_total: int, jobs_failed: int) -> float:
    """Laplace's rule, so a brand-new provider gets 0.5 rather than a made-up reputation."""
    return (jobs_failed + 1) / (jobs_total + 2)


def rank(candidates: list[Candidate], req: Requirements) -> tuple[list[Ranked], list[Rejected]]:
    """Lowest expected cost first. Ties go to more memory headroom, then to provider ID, so results are stable."""
    ranked: list[Ranked] = []
    rejected: list[Rejected] = []

    for c in candidates:
        if c.excluded:
            rejected.append(Rejected(c.provider_id, "EXCLUDED", "Excluded from this job"))
            continue
        if not c.online:
            rejected.append(Rejected(c.provider_id, "OFFLINE", "Provider is offline"))
            continue
        if not c.measured_steps_per_s or c.peak_mem_mib is None or c.capacity_mib is None:
            rejected.append(Rejected(c.provider_id, "PROBE_ERROR", "No successful probe, so nothing proves it can run this job"))
            continue

        eta = req.remaining_steps / c.measured_steps_per_s
        if eta > req.seconds_to_deadline:
            rejected.append(Rejected(c.provider_id, "DEADLINE", f"Needs {eta:.0f}s, but only {req.seconds_to_deadline:.0f}s remain"))
            continue

        direct_cost = c.price_per_step * req.remaining_steps
        if direct_cost > req.remaining_budget:
            rejected.append(Rejected(c.provider_id, "BUDGET", f"Needs {direct_cost} USDC, but only {req.remaining_budget} remain"))
            continue

        p_fail = failure_probability(c.jobs_total, c.jobs_failed)
        expected = direct_cost / Decimal(str(1 - p_fail))
        ranked.append(
            Ranked(
                provider_id=c.provider_id,
                p_fail=p_fail,
                has_history=c.jobs_total > 0,
                eta_seconds=eta,
                expected_cost=expected,
                headroom_mib=c.capacity_mib - c.peak_mem_mib,
            )
        )

    ranked.sort(key=lambda r: (r.expected_cost, -r.headroom_mib, r.provider_id))
    return ranked, rejected
