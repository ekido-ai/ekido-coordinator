"""Checkpoint retention, quarantine, the per-job storage cap and the pending-upload budget."""
import wrapper_setup  # noqa: F401  (job keys need a wrapper; see wrapper_setup.py)
import hashlib
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ekido_coordinator import jobs
from ekido_coordinator.artifacts import ArtifactStore, UploadBudget, UploadBudgetExceeded
from ekido_coordinator.models import Base, Checkpoint, Job, JobTransition, Settlement
from ekido_coordinator.retention import ACCEPTED_TO_KEEP, ORPHAN_GRACE, keep_set, plan, rows_to_keep, run

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def manifest(label: str, step: int, size: int = 100) -> dict:
    return {
        "logical_step": step,
        "files": {
            "trainable.safetensors": {"sha256": digest(f"{label}-t"), "bytes": size},
            "optimizer.safetensors": {"sha256": digest(f"{label}-o"), "bytes": size},
        },
    }


def row(job_id: str, label: str, step: int, state: str, size: int = 100) -> Checkpoint:
    return Checkpoint(
        job_id=job_id, manifest_hash=digest(f"manifest-{job_id}-{label}"), logical_step=step, fencing_token=1,
        state=state, size_bytes=2 * size, manifest=manifest(label, step, size), created_at=NOW - timedelta(hours=1),
    )


def blobs(label: str) -> set[str]:
    return {digest(f"{label}-t"), digest(f"{label}-o")}


class Rules(unittest.TestCase):
    def test_a_job_in_progress_keeps_its_tail_the_last_accepted_and_rejected_evidence(self):
        rows = [
            row("j", "old-commit", 5, "COMMITTED"),
            row("j", "a10", 10, "ACCEPTED"),
            row("j", "a20", 20, "ACCEPTED"),
            row("j", "r25", 25, "REJECTED"),
            row("j", "a30", 30, "ACCEPTED"),
            row("j", "v40", 40, "VERIFIED"),
        ]
        kept = {r.logical_step for r in rows_to_keep(rows, ended_at=None, final_hash=None, now=NOW)}
        self.assertEqual(ACCEPTED_TO_KEEP, 2)
        self.assertEqual(kept, {20, 25, 30, 40})

    def test_an_ended_job_keeps_everything_for_a_week(self):
        rows = [row("j", "a10", 10, "ACCEPTED"), row("j", "c5", 5, "COMMITTED")]
        kept = rows_to_keep(rows, ended_at=NOW - timedelta(days=6), final_hash=None, now=NOW)
        self.assertEqual(len(kept), 2)

    def test_after_a_week_only_the_final_model_stays_until_90_days(self):
        final = row("j", "final", 30, "ACCEPTED")
        rows = [row("j", "a10", 10, "ACCEPTED"), final, row("j", "r20", 20, "REJECTED")]
        kept = rows_to_keep(rows, ended_at=NOW - timedelta(days=8), final_hash=final.manifest_hash, now=NOW)
        self.assertEqual([r.manifest_hash for r in kept], [final.manifest_hash])
        # Without a verdict, the newest accepted checkpoint is the final model.
        kept = rows_to_keep(rows, ended_at=NOW - timedelta(days=8), final_hash=None, now=NOW)
        self.assertEqual([r.logical_step for r in kept], [30])
        self.assertEqual(rows_to_keep(rows, ended_at=NOW - timedelta(days=91), final_hash=None, now=NOW), [])


class KeepSetAndQuarantine(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = ArtifactStore(Path(folder.name) / "artifacts")

    def make_job(self, session: Session, state: str, ended_days_ago: float | None = None) -> Job:
        job = jobs.create_job(session, spec={"template": "synthetic-linear/v1"}, policy={}, budget=Decimal("5"), deadline_seconds=600)
        job.state = state
        if ended_days_ago is not None:
            session.add(JobTransition(job_id=job.id, seq=99, from_state="COMPLETED", to_state=state,
                                      created_at=NOW - timedelta(days=ended_days_ago)))
        session.flush()
        return job

    def put_blobs(self, label: str, age: timedelta) -> None:
        for d in blobs(label):
            path = self.store.path_for(d)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 10)
            stamp = (NOW - age).timestamp()
            os.utime(path, (stamp, stamp))

    def test_a_blob_shared_with_a_live_job_is_kept(self):
        with Session(self.engine) as session:
            old = self.make_job(session, "SETTLED", ended_days_ago=100)
            live = self.make_job(session, "RUNNING")
            shared = row(old.id, "shared", 10, "ACCEPTED")
            session.add(shared)
            reuse = row(live.id, "reuse", 10, "VERIFIED")
            reuse.manifest = manifest("shared", 10)  # identical content, so the same blobs
            session.add(reuse)
            session.commit()
            keep = keep_set(session, NOW)
        self.assertEqual(blobs("shared") - keep, set())  # nothing the live job needs is missing

    def test_settled_job_keeps_the_verdicts_final_checkpoint(self):
        with Session(self.engine) as session:
            job = self.make_job(session, "SETTLED", ended_days_ago=10)
            early, final = row(job.id, "early", 10, "ACCEPTED"), row(job.id, "final", 20, "ACCEPTED")
            session.add_all([early, final])
            session.add(Settlement(job_id=job.id, verdict={"finalCheckpoint": "0x" + final.manifest_hash}, status="SETTLED"))
            session.commit()
            keep = keep_set(session, NOW)
        self.assertEqual(keep, blobs("final"))

    def test_plan_and_apply_quarantine_only_old_unneeded_blobs_and_they_can_come_back(self):
        with Session(self.engine) as session:
            job = self.make_job(session, "SETTLED", ended_days_ago=10)
            session.add_all([row(job.id, "early", 10, "ACCEPTED"), row(job.id, "final", 20, "ACCEPTED")])
            session.commit()
        self.put_blobs("early", timedelta(days=12))
        self.put_blobs("final", timedelta(days=12))
        self.put_blobs("fresh-orphan", ORPHAN_GRACE / 2)
        self.put_blobs("old-orphan", timedelta(days=3))
        with Session(self.engine) as session:
            result = plan(self.store, keep_set(session, NOW), NOW)
        self.assertEqual(set(result.move), blobs("early") | blobs("old-orphan"))
        self.assertEqual(result.young_orphans, 2)

        # Apply re-checks each blob's age against the real clock, while the plan uses NOW. Three days before NOW is old
        # enough for both, on any day after NOW. (Stamping relative to the real clock broke this test once the real date
        # passed NOW + 3 days: the blobs then looked newer than NOW and nothing was moved.)
        for label in ("early", "old-orphan"):
            for d in blobs(label):
                stamp = (NOW - timedelta(days=3)).timestamp()
                os.utime(self.store.path_for(d), (stamp, stamp))
        summary = run(self.engine, self.store, apply=True, now=NOW)
        self.assertEqual(summary["moved"], 4)
        for d in blobs("early"):
            self.assertFalse(self.store.exists(d))
            self.assertTrue(self.store.restore(d))
            self.assertTrue(self.store.exists(d))
        self.assertTrue(all(self.store.exists(d) for d in blobs("final") | blobs("fresh-orphan")))

    def test_a_blob_uploaded_again_after_planning_is_not_moved(self):
        self.put_blobs("orphan", timedelta(days=3))
        for d in blobs("orphan"):
            os.utime(self.store.path_for(d))  # what put() does when the same content is uploaded again
        summary = run(self.engine, self.store, apply=True, now=datetime.now(timezone.utc))
        self.assertEqual(summary["moved"], 0)

    def test_old_quarantine_folders_are_purged_and_recent_ones_kept(self):
        for day in ("20261001", "20261119"):
            folder = self.store.quarantine_root / day
            folder.mkdir(parents=True)
            (folder / digest(day)).write_bytes(b"x")
        removed = self.store.purge_quarantine(timedelta(days=30), NOW)
        self.assertEqual(removed, ["20261001"])
        self.assertTrue((self.store.quarantine_root / "20261119").is_dir())


class StorageCap(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)

    def test_a_commit_over_the_jobs_cap_is_refused(self):
        with Session(self.engine) as session:
            job = jobs.create_job(session, spec={"template": "synthetic-linear/v1"}, policy={}, budget=Decimal("5"), deadline_seconds=600)
            job.fencing_token = 1
            session.add(row(job.id, "v10", 10, "VERIFIED", size=400))  # 800 bytes kept
            session.flush()
            new = dict(manifest("v20", 20, size=200), fencing_token=1, immutable_hash=job.spec_hash)
            with self.assertRaises(jobs.StorageQuotaExceeded):
                jobs.commit_checkpoint(session, job, new, fencing_token=1, max_job_bytes=1000)
            committed = jobs.commit_checkpoint(session, job, new, fencing_token=1, max_job_bytes=1200)
            self.assertEqual(committed.size_bytes, 400)


class PendingUploads(unittest.TestCase):
    def test_the_budget_refuses_past_its_limit_and_a_commit_clears_it(self):
        budget = UploadBudget(limit=1000, window_seconds=60)
        budget.check("p", 600, now=0)
        budget.record("p", digest("a"), 600, now=0)
        with self.assertRaises(UploadBudgetExceeded):
            budget.check("p", 500, now=1)
        budget.check("other", 900, now=1)  # budgets are per provider
        budget.release("p")
        budget.check("p", 900, now=2)

    def test_old_pending_uploads_age_out(self):
        budget = UploadBudget(limit=1000, window_seconds=60)
        budget.record("p", digest("a"), 900, now=0)
        self.assertEqual(budget.pending_bytes("p", now=61), 0)


if __name__ == "__main__":
    unittest.main()
