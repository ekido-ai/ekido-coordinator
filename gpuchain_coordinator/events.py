"""Dashboard events: stored first, then pushed, so a reconnecting client can catch up without gaps."""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import DashboardEventRow

EVENT_TYPES = frozenset(
    {
        "provider.updated",
        "telemetry.sample",
        "job.state_changed",
        "training.progress",
        "checkpoint.committed",
        "checkpoint.verified",
        "checkpoint.accepted",
        "checkpoint.rejected",
        "failure.warning",
        "failure.detected",
        "recovery.step",
        "compile.decision",
        "quality.result",
        "chain.tx",
        "fault.injected",
    }
)


class EventBus:
    """In-process fan-out. One coordinator process is the MVP's documented design."""

    def __init__(self) -> None:
        self._subscribers: dict[str | None, set[asyncio.Queue]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """The orchestrator publishes from worker threads, and asyncio queues aren't thread-safe."""
        self._loop = loop

    def subscribe(self, job_id: str | None) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.setdefault(job_id, set()).add(queue)
        return queue

    def unsubscribe(self, job_id: str | None, queue: asyncio.Queue) -> None:
        self._subscribers.get(job_id, set()).discard(queue)

    def publish(self, event: dict) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                on_loop_thread = asyncio.get_running_loop() is loop
            except RuntimeError:
                on_loop_thread = False
            if not on_loop_thread:
                loop.call_soon_threadsafe(self._deliver, event)
                return
        self._deliver(event)

    def _deliver(self, event: dict) -> None:
        for queue in list(self._subscribers.get(event["job_id"], ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stuck client must not block training. It recovers through catch-up on reconnect.
                self.unsubscribe(event["job_id"], queue)


_seq_lock = threading.Lock()
_last_seq: dict[tuple[int, str | None], int] = {}


def _next_seq(session: Session, job_id: str | None) -> int:
    """Hands out event numbers under a lock. Several threads emit at once, so reading max+1 from the database collides.

    The one-process coordinator is the MVP's documented design, so a process-level counter is enough.
    It starts from the database, so a restart continues the numbering. A gap after a rollback is harmless.
    """
    key = (id(session.get_bind()), job_id)
    with _seq_lock:
        current = _last_seq.get(key)
        if current is None:
            current = session.scalar(select(func.max(DashboardEventRow.seq)).where(DashboardEventRow.job_id == job_id)) or 0
        _last_seq[key] = current + 1
        return current + 1


def record_event(session: Session, bus: EventBus | None, event_type: str, job_id: str | None, data: dict[str, Any], *, provenance: str | None = None) -> dict:
    """Stores an event in the caller's transaction. The caller publishes after committing."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type: {event_type}")
    # The CPU fallback sets SYNTHETIC, so a screen can't present it as the real model.
    provenance = provenance or os.environ.get("EVENT_PROVENANCE", "LIVE")
    seq = _next_seq(session, job_id)
    event = {
        "v": 1,
        "id": seq,
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "job_id": job_id,
        "provenance": provenance,
        "data": data,
    }
    session.add(DashboardEventRow(job_id=job_id, seq=seq, event=event))
    session.flush()
    return event


def events_after(session: Session, job_id: str | None, after: int) -> list[dict]:
    rows = session.scalars(
        select(DashboardEventRow).where(DashboardEventRow.job_id == job_id, DashboardEventRow.seq > after).order_by(DashboardEventRow.seq)
    )
    return [row.event for row in rows]
