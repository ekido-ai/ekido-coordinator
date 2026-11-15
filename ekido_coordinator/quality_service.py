"""Connects the orchestrator to the verifiers, and turns signed reports into one quorum decision.

Verifiers pull work and return signed reports, so they need no inbound connection. Only allowlisted
signers count. These verifiers are permissioned and, in this prototype, team-operated. That's
disclosed in the product: it is not a decentralized verifier network.
"""

from __future__ import annotations

import threading
import uuid
from time import monotonic
from dataclasses import dataclass, field

from ekido_core.signing import verify_document


class QualityTimeout(Exception):
    pass


EVALUATION_CLAIM_LEASE_SECONDS = 180


def quorum_decision(reports: list[dict], checkpoint_hash: str, quorum: int, allowed: set[str]) -> str | None:
    """PASS or FAIL once enough distinct allowlisted verifiers agree. A failure quorum is checked first."""
    votes: dict[str, set[str]] = {"PASS": set(), "FAIL": set()}
    for report in reports:
        signer = str(report.get("signer", "")).lower()
        if signer not in allowed or report.get("checkpoint_hash") != checkpoint_hash or report.get("verdict") not in votes:
            continue
        if verify_document(report, report.get("signature", ""), signer):
            votes[report["verdict"]].add(signer)
    if len(votes["FAIL"]) >= quorum:
        return "FAIL"
    if len(votes["PASS"]) >= quorum:
        return "PASS"
    return None


@dataclass
class EvaluationTask:
    task_id: str
    job_id: str
    checkpoint_hash: str
    reference_hash: str | None
    is_final: bool
    reports: list[dict] = field(default_factory=list)
    claimed_by: set[str] = field(default_factory=set)
    claimed_at: dict[str, float] = field(default_factory=dict)
    decision: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


class QuorumQuality:
    """Implements QualityLink. evaluate() blocks a worker thread until the verifiers decide."""

    def __init__(self, verifiers: set[str], quorum: int, timeout_seconds: float = 300.0):
        if quorum < 1 or quorum > len(verifiers):
            raise ValueError("quorum must be between 1 and the number of verifiers")
        self.verifiers = {v.lower() for v in verifiers}
        self.quorum = quorum
        self.timeout = timeout_seconds
        self.tasks: dict[str, EvaluationTask] = {}
        self.decided_reports: dict[tuple[str, str], list[dict]] = {}  # (job, checkpoint) -> signed reports.
        self.lock = threading.Lock()
        self.on_report = None  # Set by the app, to show each verifier's result on the dashboard.
        self.on_decision = None  # Persist signed evidence before making a decision visible.

    def evaluate(self, job_id: str, checkpoint_hash: str, reference_hash: str | None, *, is_final: bool) -> str:
        task = EvaluationTask(uuid.uuid4().hex, job_id, checkpoint_hash, reference_hash, is_final)
        with self.lock:
            self.tasks[task.task_id] = task
        try:
            if not task.done.wait(self.timeout):
                # Never treat missing verifiers as a pass. Payment depends on this answer.
                raise QualityTimeout(f"no quorum for {checkpoint_hash} within {self.timeout:.0f}s")
            return task.decision or "FAIL"
        finally:
            with self.lock:
                self.tasks.pop(task.task_id, None)

    def claim(self, verifier: str) -> dict | None:
        """The next task this verifier hasn't taken yet. One verifier can't report twice."""
        verifier = verifier.lower()
        if verifier not in self.verifiers:
            raise PermissionError("not an allowlisted verifier")
        with self.lock:
            for task in self.tasks.values():
                now = monotonic()
                claimed_at = task.claimed_at.get(verifier)
                already_reported = any(str(report.get("signer", "")).lower() == verifier for report in task.reports)
                if not task.done.is_set() and not already_reported and (claimed_at is None or now - claimed_at >= EVALUATION_CLAIM_LEASE_SECONDS):
                    task.claimed_by.add(verifier)
                    task.claimed_at[verifier] = now
                    return {
                        "task_id": task.task_id,
                        "job_id": task.job_id,
                        "checkpoint_hash": task.checkpoint_hash,
                        "reference_hash": task.reference_hash,
                        "is_final": task.is_final,
                    }
        return None

    def submit(self, task_id: str, report: dict) -> str | None:
        """Returns the decision if this report completed the quorum."""
        signer = str(report.get("signer", "")).lower()
        if signer not in self.verifiers:
            raise PermissionError("not an allowlisted verifier")
        if not verify_document(report, report.get("signature", ""), signer):
            raise PermissionError("report signature doesn't verify")
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None or task.done.is_set():
                return None  # Already decided, or abandoned. A late report changes nothing.
            if report.get("checkpoint_hash") not in (task.checkpoint_hash, "0" * 64):
                raise ValueError("report is for a different checkpoint")
            if report.get("job_id") != task.job_id or report.get("reference_checkpoint_hash") != task.reference_hash or report.get("is_final") is not task.is_final:
                raise ValueError("report is for a different evaluation context")
            if any(str(existing.get("signer", "")).lower() == signer for existing in task.reports):
                raise ValueError("verifier has already submitted a report for this task")
            # A checkpoint too broken to load has no hash of its own, so the failure counts for this task.
            normalized = dict(report)
            task.reports.append(normalized)
            decision = quorum_decision_for_task(task, self.quorum, self.verifiers)
            if decision is not None:
                reports = [r for r in task.reports if r.get("verdict") == decision]
                try:
                    if self.on_decision is not None:
                        self.on_decision(task.job_id, task.checkpoint_hash, reports)
                except Exception:
                    task.reports.pop()
                    task.claimed_by.discard(signer)
                    task.claimed_at.pop(signer, None)
                    raise
                task.decision = decision
                self.decided_reports[(task.job_id, task.checkpoint_hash)] = reports
                task.done.set()
        if self.on_report is not None:
            self.on_report(task.job_id, task.checkpoint_hash, report)
        return decision


def quorum_decision_for_task(task: EvaluationTask, quorum: int, allowed: set[str]) -> str | None:
    votes: dict[str, set[str]] = {"PASS": set(), "FAIL": set()}
    for report in task.reports:
        signer = str(report.get("signer", "")).lower()
        if signer not in allowed or report.get("verdict") not in votes:
            continue
        if report.get("checkpoint_hash") == "0" * 64 and report.get("verdict") != "FAIL":
            continue  # "Couldn't load it" can only ever count against a checkpoint.
        votes[report["verdict"]].add(signer)
    if len(votes["FAIL"]) >= quorum:
        return "FAIL"
    if len(votes["PASS"]) >= quorum:
        return "PASS"
    return None
