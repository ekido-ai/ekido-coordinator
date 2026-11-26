"""The Coordinator's side of example generation: the parts the `distill-gen/v1` contract fixes (ticket #154; docs/api/distill-gen-v1.md).

**What is here:** accepting a generation request from the Platform (`POST /distillations`), checking the spec and the seeds file against the
contract, keeping them, and telling the Platform how it ended (rows with the verifier's verdict, or a failure, which refunds the customer).

**What is deliberately not here:** placing the job on a T4-class device, sending it to the agent, taking back the progress file and the
generated rows, and running the verifier on them. The contract does not define the message between the Coordinator and the device, and the
device side (ekido-workers) is not merged yet, so inventing it here would only have to be thrown away. Whoever builds it calls
`finish` or `fail` below. Until then the setting `DISTILLATION_ENABLED` is **off** and the route refuses with 409, so the Platform releases the
order at once and no customer waits, or pays, for a job that cannot run.

The seeds are the customer's own text. They are untrusted data (they may hold instructions); nothing here reads them as anything but text,
they are never logged, and they are dropped once the job has ended.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core import distill_contract
from ekido_core.recipes import recipe

from .models import DistillationJob

log = logging.getLogger(__name__)

QUEUED, DONE, FAILED = "QUEUED", "DONE", "FAILED"
MAX_SEEDS_BYTES = 4_000_000


class Refused(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def seed_problems(spec: dict, seeds_jsonl: str) -> list[str]:
    """Does the seeds file match the spec? Hash, size, row count, the recipe's two columns, the one serialisation, and no overlap with the tests."""
    data = seeds_jsonl.encode("utf-8")
    seeds = spec["seeds"]
    if len(data) > MAX_SEEDS_BYTES or len(data) != seeds["bytes"] or hashlib.sha256(data).hexdigest() != seeds["sha256"]:
        return ["seeds_jsonl does not match the size and hash in the spec"]
    try:
        rows = [json.loads(line) for line in seeds_jsonl.split("\n") if line]
    except ValueError:
        return ["seeds_jsonl is not JSON lines"]
    r = recipe(spec["recipe_id"])
    columns = set(r.columns)
    if len(rows) != seeds["rows"] or not all(isinstance(x, dict) and set(x) == columns and all(isinstance(x[c], str) and x[c].strip() for c in columns) for x in rows):
        return ["seeds_jsonl must hold the stated number of rows, each with the recipe's two columns"]
    if distill_contract.jsonl_bytes(rows) != data:
        return ["seeds_jsonl is not in the contract's serialisation (compact, key-sorted, one row per line)"]
    tests = set(spec["test_input_sha256"])
    if any(distill_contract.input_sha256(x[r.columns[0]]) in tests for x in rows):
        return ["a seed row is also one of the customer's test inputs"]
    return []


def create(session: Session, *, distillation_id: str, spec: Any, seeds_jsonl: str, enabled: bool) -> DistillationJob:
    """Accept one generation request. The same request again returns the same job; a different one for the same distillation is refused."""
    problems = distill_contract.spec_problems(spec)
    if problems:
        raise Refused(422, "spec: " + "; ".join(problems[:6]))
    problems = seed_problems(spec, seeds_jsonl)
    if problems:
        raise Refused(422, problems[0])
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    existing = session.scalar(select(DistillationJob).where(DistillationJob.distillation_id == distillation_id, DistillationJob.attempt == spec["attempt"]))
    if existing is not None:
        if existing.spec_sha256 != fingerprint:
            raise Refused(409, "this distillation attempt was already requested with a different spec")
        return existing
    if not enabled:
        raise Refused(409, "example generation is not available yet")
    row = DistillationJob(job_id=str(uuid.uuid4()), distillation_id=distillation_id, attempt=spec["attempt"], state=QUEUED, spec=spec,
                          spec_sha256=fingerprint, seeds_jsonl=seeds_jsonl)
    session.add(row)
    session.commit()
    return row


class PlatformReports:
    """The two reports of the contract. No redirects are followed, the token travels only in a header, errors name only the status."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0, opener: Callable | None = None) -> None:
        if not token:
            raise ValueError("an internal token is needed to talk to the Platform")
        self._base, self._token, self._timeout = base_url.rstrip("/"), token, timeout

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        self._open = opener or (lambda request, timeout: urllib.request.build_opener(NoRedirect).open(request, timeout=timeout))

    def _post(self, path: str, body: dict) -> None:
        request = urllib.request.Request(self._base + path, method="POST", data=json.dumps(body).encode("utf-8"),
                                         headers={"X-Internal-Token": self._token, "Content-Type": "application/json"})
        try:
            with self._open(request, self._timeout) as response:
                response.read(100_000)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"the Platform answered {exc.code} to POST {path}") from None

    def generated(self, distillation_id: str, rows_jsonl: str, passed: bool, reasons: list[str]) -> None:
        self._post(f"/internal/distillations/{urllib.request.quote(distillation_id, safe='')}/generated", {"rows_jsonl": rows_jsonl, "passed": passed, "reasons": reasons[:40]})

    def failed(self, distillation_id: str, reason: str) -> None:
        self._post(f"/internal/distillations/{urllib.request.quote(distillation_id, safe='')}/failed", {"reason": reason[:200]})


def finish(session: Session, job_id: str, *, rows_jsonl: str, passed: bool, reasons: list[str], platform) -> DistillationJob:
    """The device finished and the verifier judged the file: tell the Platform, and only then mark the job done and drop the seeds.
    If the Platform cannot be told, the job stays QUEUED so the caller can try again (the Platform's route is safe to repeat)."""
    row = session.get(DistillationJob, job_id)
    if row is None:
        raise Refused(404, "no such generation job")
    if row.state != QUEUED:
        return row
    platform.generated(row.distillation_id, rows_jsonl if passed else "", passed, reasons)
    row.state, row.seeds_jsonl = DONE if passed else FAILED, None
    session.commit()
    return row


def fail(session: Session, job_id: str, reason: str, *, platform) -> DistillationJob:
    """No device, or the job failed: the Platform refunds the order in full. Safe to call again."""
    row = session.get(DistillationJob, job_id)
    if row is None:
        raise Refused(404, "no such generation job")
    if row.state != QUEUED:
        return row
    platform.failed(row.distillation_id, reason)
    row.state, row.seeds_jsonl = FAILED, None
    session.commit()
    return row
