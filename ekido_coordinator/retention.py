"""Which checkpoint blobs the store keeps, and moving the rest to quarantine.

Recovery needs a job's recent checkpoints, and a finished job's client needs its final model. Nothing else has to
stay on disk. Manifests and database rows are never touched: lineage, verdicts and passports need only those.

Rules (decided 15 Nov 2026):
- A job still in progress keeps every checkpoint newer than its newest ACCEPTED one, its last two ACCEPTED ones
  (the rollback target and one more), and every REJECTED one as evidence.
- A job that has ended (settled, refunded, cancelled or failed) keeps everything for 7 days after it ended, then only
  its final checkpoint, until 90 days after it ended.
- A blob no checkpoint references is kept for 24 hours, because uploads land before their manifest commits.
- A blob is kept if any checkpoint needs it. Storage is content-addressed, so this is the reference count.

Removing a blob moves it to quarantine, never deletes it. Quarantine folders older than 30 days are purged.

    python -m ekido_coordinator.retention            # dry run: report what would move
    python -m ekido_coordinator.retention --apply    # move it, and purge old quarantine folders
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ekido_core.checkpoint_files import checkpoint_files
from ekido_core.job_states import TERMINAL, JobState

from .artifacts import ArtifactStore, S3Blobs
from .models import Checkpoint, Job, JobTransition, Settlement

log = logging.getLogger("ekido.retention")

ACCEPTED_TO_KEEP = 2
ENDED_GRACE = timedelta(days=7)
FINAL_MODEL_KEPT_FOR = timedelta(days=90)
ORPHAN_GRACE = timedelta(hours=24)
QUARANTINE_KEPT_FOR = timedelta(days=30)
ENDED_STATES = frozenset({state.value for state in TERMINAL} | {JobState.FAILED.value})


def _utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _by_step(rows: Iterable[Checkpoint]) -> list[Checkpoint]:
    return sorted(rows, key=lambda r: (r.logical_step, _utc(r.created_at)), reverse=True)


def rows_to_keep(rows: list[Checkpoint], *, ended_at: datetime | None, final_hash: str | None, now: datetime) -> list[Checkpoint]:
    """The checkpoints of one job whose blobs must stay. `ended_at` is None while the job is still in progress."""
    if ended_at is None:
        accepted = _by_step(r for r in rows if r.state == "ACCEPTED")
        newest_accepted = accepted[0].logical_step if accepted else -1
        kept_accepted = {r.manifest_hash for r in accepted[:ACCEPTED_TO_KEEP]}
        return [r for r in rows if r.state == "REJECTED" or r.logical_step > newest_accepted or r.manifest_hash in kept_accepted]
    ended_at = _utc(ended_at)
    if now - ended_at < ENDED_GRACE:
        return list(rows)
    if now - ended_at >= FINAL_MODEL_KEPT_FOR:
        return []
    final = [r for r in rows if r.manifest_hash == final_hash] if final_hash else []
    if not final:
        accepted = _by_step(r for r in rows if r.state == "ACCEPTED")
        final = accepted[:1]
    return final


def blob_digests(manifest: dict) -> set[str]:
    """The blobs a manifest points at. A manifest that fails validation still protects whatever hashes it names."""
    try:
        return {entry["sha256"] for entry in checkpoint_files(manifest).values()}
    except ValueError:
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, dict):
            return set()
        return {e["sha256"] for e in files.values() if isinstance(e, dict) and isinstance(e.get("sha256"), str)}


def job_live_bytes(rows: list[Checkpoint]) -> int:
    """Bytes a job in progress keeps on disk under these rules. The storage quota is checked against this."""
    return sum(r.size_bytes or 0 for r in rows_to_keep(rows, ended_at=None, final_hash=None, now=datetime.now(timezone.utc)))


def _final_hash(verdict: dict | None) -> str | None:
    value = (verdict or {}).get("finalCheckpoint")
    if not isinstance(value, str):
        return None
    value = value.lower().removeprefix("0x")
    return None if set(value) == {"0"} else value


def keep_set(session: Session, now: datetime) -> set[str]:
    """Every blob digest some checkpoint still needs."""
    keep: set[str] = set()
    job_ids = session.scalars(select(Checkpoint.job_id).distinct()).all()
    for job_id in job_ids:
        job = session.get(Job, job_id)
        rows = list(session.scalars(select(Checkpoint).where(Checkpoint.job_id == job_id)))
        ended_at = None
        if job is not None and job.state in ENDED_STATES:
            last = session.scalar(select(func.max(JobTransition.created_at)).where(JobTransition.job_id == job_id))
            ended_at = last or job.created_at
        settlement = session.scalar(select(Settlement).where(Settlement.job_id == job_id))
        final_hash = _final_hash(settlement.verdict if settlement else None)
        for row in rows_to_keep(rows, ended_at=ended_at, final_hash=final_hash, now=now):
            keep |= blob_digests(row.manifest)
    return keep


@dataclass
class Plan:
    move: list[str] = field(default_factory=list)
    move_remote: list[str] = field(default_factory=list)  # Blobs in the S3 bucket.
    kept: int = 0
    young_orphans: int = 0
    bytes_to_move: int = 0


def plan(store: ArtifactStore, keep: set[str], now: datetime, blobs: S3Blobs | None = None) -> Plan:
    result = Plan()
    local = ((digest, store.size_of(digest), store.modified_at(digest), result.move) for digest in store.iter_digests())
    remote = ((digest, size, modified, result.move_remote) for digest, size, modified in (blobs.iter_blobs() if blobs else ()))
    for source in (local, remote):
        for digest, size, modified, bucket in source:
            if digest in keep:
                result.kept += 1
            elif now - modified < ORPHAN_GRACE:
                result.young_orphans += 1
            else:
                bucket.append(digest)
                result.bytes_to_move += size
    return result


def run(engine: Engine, store: ArtifactStore, *, apply: bool, now: datetime | None = None, blobs: S3Blobs | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    with Session(engine) as session:
        keep = keep_set(session, now)
    result = plan(store, keep, now, blobs)
    moved, purged = [], []
    if apply:
        for where, digests in ((store, result.move), (blobs, result.move_remote)):
            for digest in digests:
                try:
                    # Uploaded again since the plan was made (a new checkpoint reuses identical content): leave it.
                    if datetime.now(timezone.utc) - where.modified_at(digest) < ORPHAN_GRACE:
                        continue
                    where.quarantine(digest, now)
                    moved.append(digest)
                except FileNotFoundError:
                    continue
        # The bucket's own lifecycle rule empties its quarantine; only the local one is purged here.
        purged = store.purge_quarantine(QUARANTINE_KEPT_FOR, now)
    summary = {
        "applied": apply,
        "kept": result.kept,
        "young_orphans": result.young_orphans,
        "to_quarantine": len(result.move) + len(result.move_remote),
        "bytes_to_quarantine": result.bytes_to_move,
        "moved": len(moved),
        "purged_quarantine_days": purged,
    }
    log.info("retention %s", json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    from .main import Settings, create_engine_for

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="move blobs to quarantine and purge old quarantine folders")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    blobs = S3Blobs(settings.artifact_s3_bucket, region=settings.artifact_s3_region or None) if settings.artifact_s3_bucket else None
    summary = run(create_engine_for(settings.database_url), ArtifactStore(Path(settings.artifact_dir)), apply=args.apply, blobs=blobs)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
