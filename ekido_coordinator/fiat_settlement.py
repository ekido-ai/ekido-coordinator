"""Settles a job paid in rupees through the Platform, when the job ends (tickets C7, A5, A9).

A fiat job has no on-chain escrow: the customer's money is held by the Platform (a paid order linked to the job through
`POST /internal/payments/orders/{id}/job`). When a job ends, `FiatSettlement.on_job_finished` tells the Platform what to do:

- COMPLETED (the final checkpoint was already accepted by the verifier quorum): split the order's provider budget between
  the providers in proportion to what each earned (steps x rate, as in `jobs.compute_payouts`), in whole paise, and call settle;
- FAILED or CANCELLED: call refund (the customer gets everything back, no one is paid);
- a job with no linked order (a crypto job) gets a 404 from the Platform and is ignored here.

The split always adds up to the Platform's provider budget exactly (largest remainder), which is what the Platform demands.
Both Platform calls are idempotent, so a retry is safe. If the Platform cannot be reached the error is raised so the caller
can log it; `sweep` finishes anything missed (run it from the same process on a timer).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from decimal import Decimal
from typing import Any, Callable, Protocol

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ekido_core.job_states import JobState

from . import jobs as job_service
from .models import Job

log = logging.getLogger(__name__)
ENDED = (JobState.COMPLETED.value, JobState.FAILED.value, JobState.CANCELLED.value)


class NoOrder(Exception):
    """The Platform has no paid order for this job."""


class PlatformPayments(Protocol):
    def order(self, job_id: str) -> dict: ...
    def settle(self, job_id: str, payouts: dict[str, int]) -> None: ...
    def refund(self, job_id: str) -> None: ...


class HttpPlatformPayments:
    """The Platform's `/internal/payments` routes. No redirects, replies capped, errors name only the status."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 15.0, opener: Callable[[urllib.request.Request, float], Any] | None = None):
        if not token:
            raise ValueError("an internal token is needed to talk to the Platform")
        self._base, self._token, self._timeout = base_url.rstrip("/"), token, timeout

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        self._open = opener or (lambda request, timeout: urllib.request.build_opener(NoRedirect).open(request, timeout=timeout))

    def _call(self, method: str, path: str, body: dict | None = None) -> Any:
        headers = {"X-Internal-Token": self._token}
        data = None
        if body is not None:
            data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
        request = urllib.request.Request(self._base + path, method=method, headers=headers, data=data)
        try:
            with self._open(request, self._timeout) as response:
                return json.loads(response.read(2_000_000) or b"null")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NoOrder(path) from None
            raise RuntimeError(f"the Platform answered {exc.code} to {method} {path}") from None

    def order(self, job_id: str) -> dict:
        return self._call("GET", f"/internal/payments/jobs/{job_id}")

    def settle(self, job_id: str, payouts: dict[str, int]) -> None:
        self._call("POST", f"/internal/payments/jobs/{job_id}/settle", {"payouts": payouts})

    def refund(self, job_id: str) -> None:
        self._call("POST", f"/internal/payments/jobs/{job_id}/refund")


def split_paise(budget_paise: int, earned: dict[str, Decimal]) -> dict[str, int]:
    """Share `budget_paise` in proportion to `earned`, in whole paise, adding up exactly (largest remainder)."""
    total = sum(earned.values(), Decimal(0))
    if budget_paise <= 0 or total <= 0:
        return {}
    exact = {key: Decimal(budget_paise) * amount / total for key, amount in earned.items() if amount > 0}
    shares = {key: int(value) for key, value in exact.items()}
    leftover = budget_paise - sum(shares.values())
    for key in sorted(exact, key=lambda k: (exact[k] - shares[k], k), reverse=True)[:leftover]:
        shares[key] += 1
    return {key: value for key, value in shares.items() if value > 0}


class FiatSettlement:
    def __init__(self, engine: Engine, platform: PlatformPayments):
        self.engine, self.platform = engine, platform

    def on_job_finished(self, job_id: str) -> str:
        """Returns 'settled', 'refunded' or 'not-fiat'. Raises when the Platform cannot be reached, so the caller logs it."""
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            if job is None or job.state not in ENDED:
                raise ValueError(f"job {job_id} has not ended")
            state = job.state
            payouts = self._earned(session, job) if state == JobState.COMPLETED.value else {}
        try:
            if state != JobState.COMPLETED.value:
                self.platform.refund(job_id)
                return "refunded"
            info = self.platform.order(job_id)
            split = split_paise(int(info["provider_budget_paise"]), payouts)
            if not split:  # a passed job that earned nothing: nobody to pay, so the customer gets it back
                self.platform.refund(job_id)
                return "refunded"
            self.platform.settle(job_id, split)
            return "settled"
        except NoOrder:
            return "not-fiat"

    def _earned(self, session: Session, job: Job) -> dict[str, Decimal]:
        tip = job_service.latest_accepted_checkpoint(session, job)
        chain = job_service.accepted_lineage(session, tip)
        steps = job_service.steps_by_signer(chain)
        by_key = job_service.providers_by_signer(session)
        earned: dict[str, Decimal] = {}
        for signer, count in steps.items():
            provider = by_key.get(signer)
            if provider is None:
                raise RuntimeError(f"no provider has the enrolled key {signer}")
            earned[provider.id] = earned.get(provider.id, Decimal(0)) + count * Decimal(provider.price_per_step)
        return earned

    def sweep(self) -> dict[str, int]:
        """Run on a timer: finish ended jobs whose first attempt failed. Jobs the Platform does not know are skipped quickly."""
        counts = {"settled": 0, "refunded": 0, "not-fiat": 0, "error": 0}
        with Session(self.engine) as session:
            ids = list(session.scalars(select(Job.id).where(Job.state.in_(ENDED))))
        for job_id in ids:
            try:
                counts[self.on_job_finished(job_id)] += 1
            except Exception:
                log.exception("fiat settlement of job %s failed", job_id)
                counts["error"] += 1
        return counts
