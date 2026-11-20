"""Carries out "delete my data" for the jobs of customers who asked (tickets A14 and C17).

The Platform records what to destroy (an *erasure task* per job of a deleted team). This worker asks the Platform for the pending tasks, destroys
each job's encryption key here, and reports the task done (or failed, so repeated failures are visible). It runs as its own process:

    python -m ekido_coordinator.erasure_worker --database-url ... --platform-url http://127.0.0.1:8010     (INTERNAL_TOKEN in the environment)

For each task, in this order of thought:
- the job is unknown here (never created, or already gone): nothing to destroy, the task is done;
- the job has ended (SETTLED, FAILED, CANCELLED, REFUNDED): its key is destroyed (safe to repeat), the task is done;
- the job is OPEN (nothing has run): it is cancelled, then its key is destroyed, the task is done;
- the job is running or being verified: its key must stay until it ends, so the task is reported failed and asked again on the next pass, which fits
  the 30-day window;
- a task of a kind this worker does not know is left alone (neither done nor failed).

Destroying the key makes everything encrypted to the job unreadable *by Ekido's live systems*. It does not remove the stored objects (the retention job does,
on its schedule) and it does not reach older database backups: see `job_keys.destroy` for that limit. The token is read from the environment, sent only in the
`X-Internal-Token` header and never logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ekido_core.job_states import JobState

from . import job_keys, jobs
from .models import Job

log = logging.getLogger(__name__)


class PlatformClient(Protocol):
    """The three Platform calls the worker uses. Tests pass a fake; production passes `HttpPlatformClient`."""

    def tasks(self, limit: int = 100) -> list[dict]:
        """Pending erasure tasks: `{id, kind, ref, attempts}`."""

    def complete(self, task_id: str) -> None:
        """Report a task done."""

    def failed(self, task_id: str) -> None:
        """Report a failed attempt (counted by the Platform)."""


class HttpPlatformClient:
    """The Platform's `/internal` routes. No redirects are followed and replies are capped, so a misdirected call cannot leak the token."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 15.0, opener: Callable[[urllib.request.Request, float], Any] | None = None):
        if not token:
            raise ValueError("an internal token is needed to talk to the Platform")
        self._base, self._token, self._timeout = base_url.rstrip("/"), token, timeout

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        self._open = opener or (lambda request, timeout: urllib.request.build_opener(NoRedirect).open(request, timeout=timeout))

    def _call(self, method: str, path: str) -> Any:
        request = urllib.request.Request(self._base + path, method=method, headers={"X-Internal-Token": self._token}, data=b"" if method == "POST" else None)
        try:
            with self._open(request, self._timeout) as response:
                return json.loads(response.read(2_000_000) or b"null")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"the Platform answered {exc.code} to {method} {path}") from None  # the status only: never the headers

    def tasks(self, limit: int = 100) -> list[dict]:
        return self._call("GET", f"/internal/erasure/tasks?limit={int(limit)}")

    def complete(self, task_id: str) -> None:
        self._call("POST", f"/internal/erasure/tasks/{task_id}/complete")

    def failed(self, task_id: str) -> None:
        self._call("POST", f"/internal/erasure/tasks/{task_id}/failed")


class ErasureWorker:
    def __init__(self, engine: Engine, platform: PlatformClient):
        self.engine, self.platform = engine, platform

    def run_once(self, limit: int = 100) -> dict[str, int]:
        """One pass over the pending tasks. Returns how many were done, asked to retry, or skipped."""
        counts = {"done": 0, "retry": 0, "skipped": 0}
        for task in self.platform.tasks(limit):
            if task.get("kind") != "job_data":
                counts["skipped"] += 1
                continue
            try:
                finished = self._erase(task["ref"])
            except Exception:  # one bad job must not stop the others; the Platform counts the failed attempt
                log.exception("erasure task %s failed", task.get("id"))
                finished = False
            if finished:
                self.platform.complete(task["id"])
                counts["done"] += 1
            else:
                self.platform.failed(task["id"])
                counts["retry"] += 1
        return counts

    def _erase(self, job_id: str) -> bool:
        """True when the job's key is gone (or there never was one here); False when it must be asked again later."""
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            if job is None:
                return True
            if job.state == JobState.OPEN.value:
                jobs.transition(session, job, JobState.CANCELLED, reason="customer deleted their data", actor="erasure")
                session.commit()
            elif job.state not in job_keys.ENDED_JOB_STATES:
                return False
            job_keys.destroy(session, job_id)
            return True

    def run_forever(self, interval_seconds: float = 300.0) -> None:
        while True:
            try:
                counts = self.run_once()
                if any(counts.values()):
                    log.info("erasure pass: %s", counts)
            except Exception:
                log.exception("erasure pass failed; will try again")  # the Platform may be down: no secret is in this message
            time.sleep(interval_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"), help="the Coordinator's database (or set DATABASE_URL)")
    parser.add_argument("--platform-url", default=os.environ.get("PLATFORM_URL", "http://127.0.0.1:8010"))
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    args = parser.parse_args(argv)
    token = os.environ.get("INTERNAL_TOKEN", "")
    if not args.database_url or not token:
        parser.error("set DATABASE_URL (or --database-url) and INTERNAL_TOKEN in the environment")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    worker = ErasureWorker(create_engine(args.database_url, pool_pre_ping=True), HttpPlatformClient(args.platform_url, token))
    if args.once:
        print(json.dumps(worker.run_once()))
        return 0
    worker.run_forever(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
