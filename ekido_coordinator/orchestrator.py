"""Runs a job's life: placement, failure, recovery, validation. Matches plan.md section C.

Synchronous on purpose, and meant to run in a worker thread: probes and evaluations take seconds,
and a plain sequence of steps is far easier to get right than a callback chain. Every stage commits
and emits an event as it finishes, so a coordinator restart and the dashboard both see real progress.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ekido_core.compiler import Decision, ProbeResult, Target, build_adaptation_manifest, compile_for_target, verify_config, workload_key
from ekido_core.job_states import JobState

from . import jobs as job_service
from .events import EventBus, record_event
from .models import AdaptationManifest, Checkpoint, GpuHardware, Job, JobAssignment, Provider, ProviderReputation, RecoveryEvent, TrainingRun, Wallet
from .scheduler import Candidate, Requirements, rank

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
DEFAULT_PROBATION_STEPS = 50


class AgentLink(Protocol):
    """How the orchestrator reaches providers. The gateway implements it, and tests fake it. All calls block."""

    def is_connected(self, provider_id: str) -> bool:
        """True while the provider's agent holds an open connection."""

    def probe(self, provider_id: str, config: dict) -> ProbeResult:
        """Runs one memory probe of this config on the provider and returns what it measured."""

    def start_run(self, provider_id: str, config: dict) -> None:
        """Raises on failure to start."""

    def stop_run(self, provider_id: str, job_id: str, fencing_token: int) -> None:
        """Asks the provider to stop the run that holds this fencing token."""


class QualityLink(Protocol):
    def evaluate(self, job_id: str, checkpoint_hash: str, reference_hash: str | None, *, is_final: bool) -> str:
        """Blocks until a verifier quorum decides. Returns PASS or FAIL."""


@dataclass(frozen=True)
class Placement:
    provider_id: str
    exec_config: dict
    decision: Decision


class Orchestrator:
    def __init__(self, engine: Engine, bus: EventBus, agents: AgentLink, quality: QualityLink | None = None):
        self.engine = engine
        self.bus = bus
        self.agents = agents
        self.quality = quality
        # (provider, workload key) -> what real probes found. In memory: a restart simply measures again.
        self._qualified: dict[tuple[str, str, float | None], Decision] = {}
        self._probation_done: dict[str, int] = {}  # job -> fencing token already checked.
        self.on_job_finished = None  # Set by the settlement service. Called with a job ID.
        self.on_job_started = None  # Called with a job ID once placement succeeds.
        self.on_recovered = None  # Called with (job ID, attempt, adaptation manifest hash).
        self.max_job_storage_bytes: int | None = None  # Per-job checkpoint storage cap. Set from settings.

    # ------------------------------------------------------------- helpers

    def _emit(self, session: Session, event_type: str, job_id: str, data: dict) -> None:
        event = record_event(session, self.bus, event_type, job_id, data)
        session.commit()
        self.bus.publish(event)

    def emit_live(self, event_type: str, job_id: str, data: dict) -> None:
        """For high-rate events from the gateway, in their own short transaction."""
        with Session(self.engine) as session:
            self._emit(session, event_type, job_id, data)

    def _step(self, session: Session, job: Job, stage: str, detail: str, lost_steps: int | None = None, **extra) -> None:
        self._emit(session, "recovery.step", job.id, {"attempt": job.attempt, "stage": stage, "detail": detail, "lost_steps": lost_steps, "fencing_token": job.fencing_token, **extra})

    def _provider_of(self, session: Session, signer: str | None) -> str | None:
        """Which provider signed a checkpoint, for screens that show who did the work."""
        provider = job_service.providers_by_signer(session).get((signer or "").lower())
        return provider.id if provider else None

    def _move(self, session: Session, job: Job, target: JobState, reason: str) -> None:
        previous = job.state
        job_service.transition(session, job, target, reason=reason, actor="orchestrator")
        self._emit(session, "job.state_changed", job.id, {"from": previous, "to": target.value, "reason": reason})

    def _targets(self, session: Session, exclude: set[str]) -> list[tuple[Provider, Target]]:
        result = []
        for provider in session.scalars(select(Provider).order_by(Provider.name)):
            if provider.id in exclude:
                continue
            gpu = session.get(GpuHardware, provider.physical_gpu_id) if provider.physical_gpu_id else None
            capacity = provider.capacity_mib or (gpu.vram_mib if gpu else None)
            result.append(
                (
                    provider,
                    Target(
                        provider_id=provider.id,
                        online=self.agents.is_connected(provider.id),
                        capacity_mib=float(capacity) if capacity else None,
                        free_mib=None,
                        compute_cap=gpu.compute_cap if gpu else None,
                    ),
                )
            )
        return result

    def _excluded(self, session: Session, job: Job) -> set[str]:
        """Providers that already failed or were rejected on this job don't get it again."""
        rows = session.scalars(select(JobAssignment).where(JobAssignment.job_id == job.id, JobAssignment.state.in_(("FAILED", "REJECTED"))))
        return {row.provider_id for row in rows}

    def _seconds_left(self, job: Job) -> float:
        deadline = job.deadline if job.deadline.tzinfo else job.deadline.replace(tzinfo=timezone.utc)
        return (deadline - datetime.now(timezone.utc)).total_seconds()

    def _make_probe(self, spec: dict):
        # Padding is fixed-length, so memory per example doesn't depend on which examples are used.
        # A small subset gives the same peak memory, without tokenizing the whole dataset for each probe.
        probe_spec = dict(spec)
        if "train_examples" in probe_spec:
            probe_spec["train_examples"] = max(256, 8 * spec["logical_batch_size"])

        def probe(target: Target, config: dict) -> ProbeResult:
            return self.agents.probe(target.provider_id, {"spec": probe_spec, "exec": config, "capacity_mib": target.capacity_mib})

        return probe

    @staticmethod
    def _qualification_key(target: Target, spec: dict) -> tuple[str, str, float | None]:
        # Capacity is part of it: a measurement made under a different memory limit says nothing about this one.
        return target.provider_id, workload_key(spec), target.capacity_mib

    def _compile(self, spec: dict, target: Target, batch: int, remaining: int, seconds: float) -> Decision:
        """Reuses earlier measurements to keep recovery fast. An approval still needs one fresh probe now."""
        allow_cpu = spec.get("template") == "synthetic-linear/v1"
        key = self._qualification_key(target, spec)
        known = self._qualified.get(key)
        probe = self._make_probe(spec)
        if known is not None and target.online:
            if not known.approved:
                return Decision(target.provider_id, approved=False, capacity_mib=target.capacity_mib, code=known.code, reason=f"{known.reason} (measured earlier)", tried=known.tried)
            # Good enough to rank on. Whichever provider wins gets a fresh probe before it's used.
            return Decision(target.provider_id, approved=True, exec_config=known.exec_config, probe=known.probe, capacity_mib=target.capacity_mib, reason="Fits, measured earlier", tried=known.tried, needs_fresh_probe=True)
        decision = compile_for_target(target, batch, probe, remaining_steps=remaining, seconds_to_deadline=seconds, allow_cpu=allow_cpu)
        # Only measured outcomes are worth remembering. Being offline or short of time can change at any moment.
        if decision.approved or decision.code in ("PROBE_OOM", "OVER_MEMORY_MARGIN"):
            self._qualified[key] = decision
        return decision

    def qualify_providers(self, spec: dict) -> list[dict]:
        """Run before a job, so the job and any recovery don't wait on the full probe search."""
        results = []
        with Session(self.engine) as session:
            targets = self._targets(session, set())
            names = {provider.id: provider.name for provider, _ in targets}
        for _, target in targets:
            self._qualified.pop(self._qualification_key(target, spec), None)
            decision = self._compile(spec, target, spec["logical_batch_size"], 1, float("inf"))
            results.append(
                {
                    "provider_id": target.provider_id,
                    "name": names[target.provider_id],
                    "approved": decision.approved,
                    "code": decision.code,
                    "reason": decision.reason,
                    "exec": decision.exec_config,
                    "peak_mem_mib": decision.probe.peak_mem_mib if decision.probe else None,
                    "steps_per_s": decision.probe.steps_per_s if decision.probe else None,
                    "probes": len(decision.tried),
                }
            )
        return results

    def forget_provider(self, provider_id: str) -> None:
        """A reconnected provider may have changed, so its old measurements no longer count."""
        for key in [k for k in self._qualified if k[0] == provider_id]:
            self._qualified.pop(key, None)

    def _choose_placement(self, session: Session, job: Job, start_step: int) -> tuple[Placement | None, list[Decision]]:
        """Compile for every candidate, then rank the ones a real probe approved."""
        remaining = job.spec["total_logical_steps"] - start_step
        seconds = self._seconds_left(job)
        batch = job.spec["logical_batch_size"]
        approved: dict[str, Decision] = {}
        rejected: list[Decision] = []
        candidates: list[Candidate] = []
        spent = self._spent(session, job)
        targets_by_id: dict[str, Target] = {}
        previous = session.scalar(select(JobAssignment).where(JobAssignment.job_id == job.id).order_by(JobAssignment.fencing_token.desc()).limit(1))
        from_exec = previous.exec_config if previous else None
        agreed = {term["provider_id"]: term for term in job.spec["provider_terms"]} if "provider_terms" in job.spec else None

        for provider, target in self._targets(session, self._excluded(session, job)):
            if agreed is not None:
                term = agreed.get(provider.id)
                wallet = session.get(Wallet, provider.wallet_id)
                if term is None or wallet is None or wallet.address.lower() != term["wallet"].lower() or Decimal(provider.price_per_step) != Decimal(str(term["price_per_step"])):
                    continue
            targets_by_id[provider.id] = target
            decision = self._compile(job.spec, target, batch, remaining, seconds)
            self._emit(
                session,
                "compile.decision",
                job.id,
                {
                    "provider_id": provider.id,
                    "decision": "ACCEPTED" if decision.approved else "REJECTED",
                    "code": decision.code,
                    "reason": decision.reason or "Fits, confirmed by a real probe",
                    "exec": decision.exec_config,
                    "manifest_hash": None,
                    "attempt": job.attempt,
                    "from_exec": from_exec,
                    "capacity_mib": target.capacity_mib,
                    "peak_mem_mib": decision.probe.peak_mem_mib if decision.probe else None,
                    "steps_per_s": decision.probe.steps_per_s if decision.probe else None,
                },
            )
            if not decision.approved:
                rejected.append(decision)
                continue
            approved[provider.id] = decision
            reputation = session.scalar(select(ProviderReputation).where(ProviderReputation.provider_id == provider.id))
            ok, failed = (reputation.jobs_ok, reputation.jobs_failed) if reputation else (0, 0)
            candidates.append(
                Candidate(
                    provider_id=provider.id,
                    online=True,
                    excluded=False,
                    price_per_step=Decimal(provider.price_per_step),
                    measured_steps_per_s=decision.probe.steps_per_s,
                    peak_mem_mib=decision.probe.peak_mem_mib,
                    capacity_mib=target.capacity_mib,
                    jobs_total=ok + failed,
                    jobs_failed=failed,
                )
            )

        ranked, ranking_rejections = rank(candidates, Requirements(remaining, seconds, Decimal(job.budget) - spent))
        for item in ranking_rejections:
            decision = approved.pop(item.provider_id)
            decision.approved, decision.code, decision.reason = False, item.code, item.reason
            rejected.append(decision)
        probe = self._make_probe(job.spec)
        allow_cpu = job.spec.get("template") == "synthetic-linear/v1"
        for item in ranked:
            decision = approved[item.provider_id]
            if not decision.needs_fresh_probe:
                return Placement(decision.provider_id, decision.exec_config, decision), rejected
            target = targets_by_id[item.provider_id]
            fresh = verify_config(target, decision.exec_config, probe, remaining_steps=remaining, seconds_to_deadline=seconds, allow_cpu=allow_cpu)
            if fresh.approved:
                return Placement(fresh.provider_id, fresh.exec_config, fresh), rejected
            # What was measured earlier no longer holds. Forget it, so the next job searches again.
            self._qualified.pop(self._qualification_key(target, job.spec), None)
            self._emit(session, "compile.decision", job.id, {"provider_id": item.provider_id, "decision": "REJECTED", "code": fresh.code, "reason": fresh.reason, "exec": None, "manifest_hash": None, "attempt": job.attempt, "from_exec": from_exec, "capacity_mib": target.capacity_mib, "peak_mem_mib": None, "steps_per_s": None})
            rejected.append(fresh)
        return None, rejected

    def _spent(self, session: Session, job: Job) -> Decimal:
        """What accepted work already costs, so a replacement can't overrun the budget."""
        tip = job_service.latest_accepted_checkpoint(session, job)
        if tip is None:
            return Decimal(0)
        steps = job_service.steps_by_signer(job_service.accepted_lineage(session, tip))
        owners = job_service.providers_by_signer(session)
        terms = {term["provider_id"]: Decimal(str(term["price_per_step"])) for term in job.spec["provider_terms"]} if "provider_terms" in job.spec else None
        total = Decimal(0)
        for signer, count in steps.items():
            provider = owners.get(signer)
            if provider is None or (terms is not None and provider.id not in terms):
                raise ValueError("accepted checkpoint signer has no agreed provider rate")
            rate = terms[provider.id] if terms is not None else Decimal(provider.price_per_step)
            total += count * rate
        return total

    def _run_config(self, job: Job, assignment: JobAssignment, provider: Provider, checkpoint: Checkpoint | None, adaptation_hash: str | None) -> dict:
        return {
            "job_id": job.id,
            "attempt": assignment.attempt,
            "fencing_token": assignment.fencing_token,
            "spec": job.spec,
            "exec": assignment.exec_config,
            "capacity_mib": provider.capacity_mib,
            "signer": provider.enrolled_key,
            "provider_kind": provider.kind,
            "adaptation_manifest_hash": adaptation_hash,
            "resume_from_hash": checkpoint.manifest_hash if checkpoint else None,
            "resume_manifest": checkpoint.manifest if checkpoint else None,
        }

    def _launch(self, session: Session, job: Job, placement: Placement, checkpoint: Checkpoint | None, adaptation_hash: str | None) -> None:
        provider = session.get(Provider, placement.provider_id)
        assignment = job_service.assign(session, job, provider.id, placement.exec_config, already_fenced=job.state == JobState.RECOVERING.value)
        session.add(TrainingRun(assignment_id=assignment.id, start_step=checkpoint.logical_step if checkpoint else 0))
        session.commit()
        self.agents.start_run(provider.id, self._run_config(job, assignment, provider, checkpoint, adaptation_hash))

    # ------------------------------------------------------- public actions

    def start_job(self, job_id: str) -> str:
        """First placement. Returns the new state."""
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            if job.state != JobState.OPEN.value:
                return job.state  # Already started: a repeated call changes nothing.
            placement, rejected = self._choose_placement(session, job, 0)
            if placement is None:
                reasons = "; ".join(f"{d.provider_id}: {d.code}" for d in rejected) or "no providers"
                self._emit(session, "failure.warning", job.id, {"rule": "NO_PROVIDER", "detail": reasons})
                return job.state
            if self.on_job_started is not None:
                self.on_job_started(job.id)  # Marks it active on-chain before any work starts.
            self._move(session, job, JobState.ASSIGNED, f"Provider chosen on measured throughput and price: {placement.provider_id}")
            self._launch(session, job, placement, None, None)
            return job.state

    def on_run_started(self, job_id: str, fencing_token: int) -> None:
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            job_service.require_current_token(job, fencing_token)
            if job.state in (JobState.ASSIGNED.value, JobState.RECOVERING.value):
                was_recovering = job.state == JobState.RECOVERING.value
                self._move(session, job, JobState.RUNNING, "Training resumed" if was_recovering else "Training started")
                if was_recovering:
                    event = session.scalar(
                        select(RecoveryEvent).where(RecoveryEvent.job_id == job.id).order_by(RecoveryEvent.created_at.desc()).limit(1)
                    )
                    if event is not None:
                        event.resumed_at = datetime.now(timezone.utc)
                    self._step(session, job, "RESUMED", "Training is running on the replacement")

    def on_checkpoint(self, job_id: str, manifest: dict, *, fencing_token: int, signer: str | None, verified: bool = True) -> Checkpoint:
        """Commit first, then mark verified. Fencing and lineage checks live in commit_checkpoint."""
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            checkpoint = job_service.commit_checkpoint(
                session, job, manifest, fencing_token=fencing_token, signer=signer, max_job_bytes=self.max_job_storage_bytes
            )
            data = {"checkpoint_hash": checkpoint.manifest_hash, "step": checkpoint.logical_step, "signer": signer, "reason": None, "provider_id": self._provider_of(session, signer)}
            self._emit(session, "checkpoint.committed", job.id, data)
            if verified and checkpoint.state == "COMMITTED":
                checkpoint.state = "VERIFIED"
                self._emit(session, "checkpoint.verified", job.id, data)
            session.commit()
            # Committing expires loaded fields, so reload them for use after this session closes.
            session.refresh(checkpoint)
            session.expunge(checkpoint)
            return checkpoint

    def check_probation(self, job_id: str, checkpoint: Checkpoint, fencing_token: int) -> threading.Thread | None:
        """A replacement's first checkpoints get evaluated early, while training continues.

        Waiting for the end would pay for a whole run of bad work before finding out. Returns the
        evaluation thread when one starts, so tests can wait on it.
        """
        if self.quality is None:
            return None
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            if job.attempt < 2 or job.state != JobState.RUNNING.value or job.fencing_token != fencing_token:
                return None
            if self._probation_done.get(job_id) == fencing_token or checkpoint.logical_step >= job.spec["total_logical_steps"]:
                return None
            reference = job_service.latest_accepted_checkpoint(session, job)
            reference_step = reference.logical_step if reference else 0
            if checkpoint.logical_step - reference_step < job.policy.get("probation_steps", DEFAULT_PROBATION_STEPS):
                return None
            self._probation_done[job_id] = fencing_token
            reference_hash = reference.manifest_hash if reference else None
        thread = threading.Thread(target=self._run_probation, args=(job_id, checkpoint.manifest_hash, reference_hash, fencing_token), daemon=True, name="probation")
        thread.start()
        return thread

    def _run_probation(self, job_id: str, checkpoint_hash: str, reference_hash: str | None, fencing_token: int) -> None:
        try:
            verdict = self.quality.evaluate(job_id, checkpoint_hash, reference_hash, is_final=False)
        except Exception as exc:  # No quorum: leave the checkpoint unaccepted. The final evaluation still guards payment.
            log.warning("probation evaluation for job %s didn't finish: %s", job_id, exc)
            return
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            if job.fencing_token != fencing_token:
                return  # This worker was already replaced, so the answer no longer matters.
            if verdict == "PASS":
                checkpoint = session.scalar(select(Checkpoint).where(Checkpoint.job_id == job_id, Checkpoint.manifest_hash == checkpoint_hash))
                checkpoint.state = "ACCEPTED"
                self._emit(session, "checkpoint.accepted", job_id, {"checkpoint_hash": checkpoint_hash, "step": checkpoint.logical_step, "signer": checkpoint.signer, "reason": None, "provider_id": self._provider_of(session, checkpoint.signer)})
                return
        self.on_candidate_rejected(job_id, checkpoint_hash, None, ["FAILED_QUALITY_PROBATION"])

    def on_failure(self, job_id: str, *, provider_id: str | None, cause: str, injected: bool = False, last_step: int | None = None, fencing_token: int | None = None) -> str:
        """Entry point for every failure. Returns the job's state afterwards."""
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            if fencing_token is not None and fencing_token != job.fencing_token:
                # A report from an attempt that's already been replaced. Recovering again would kill the replacement.
                log.info("ignoring failure from stale token %s on job %s", fencing_token, job_id)
                return job.state
            if job.state not in (JobState.ASSIGNED.value, JobState.RUNNING.value, JobState.VALIDATING.value):
                return job.state  # Duplicate report, or the job is already finished.

            self._emit(session, "failure.detected", job.id, {"provider_id": provider_id, "cause": cause, "injected": injected, "last_step": last_step})
            self._move(session, job, JobState.RECOVERING, f"{cause}{' (injected)' if injected else ''}")

            # Fence before anything else, so the old worker can't write from this moment on.
            old = session.scalar(select(JobAssignment).where(JobAssignment.job_id == job.id, JobAssignment.fencing_token == job.fencing_token))
            old_exec = old.exec_config if old else None
            old_provider = old.provider_id if old else provider_id
            if old is not None:
                old.state = "FAILED"
            job_service.bump_fencing(session, job)
            session.commit()
            record = RecoveryEvent(job_id=job.id, attempt=job.attempt, kind="FAILURE", cause=cause, injected=injected, detected_at=datetime.now(timezone.utc), fenced_at=datetime.now(timezone.utc))
            session.add(record)
            self._step(session, job, "DETECTED", f"{cause} on provider {old_provider}", provider_id=old_provider)
            self._step(session, job, "FENCED", f"Writes below token {job.fencing_token} are now refused")
            if old is not None:
                try:
                    self.agents.stop_run(old.provider_id, job.id, old.fencing_token)
                except Exception:  # The provider may simply be gone, which is why we're here.
                    log.info("could not reach provider %s to stop its run", old.provider_id)
            return self._recover(session, job, record, old_provider, old_exec, last_step)

    def on_candidate_rejected(self, job_id: str, checkpoint_hash: str, provider_id: str | None, reasons: list[str]) -> str:
        """The Quality Guard refused a checkpoint: no pay for that segment, then roll back."""
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            checkpoint = session.scalar(select(Checkpoint).where(Checkpoint.job_id == job.id, Checkpoint.manifest_hash == checkpoint_hash))
            if checkpoint is None:
                raise ValueError(f"unknown checkpoint {checkpoint_hash}")
            checkpoint.state = "REJECTED"
            # Everything built on the rejected checkpoint goes with it, and so does anything else this
            # attempt wrote that wasn't accepted: that provider's word no longer counts on this job.
            for descendant in self._descendants(session, job, checkpoint):
                descendant.state = "REJECTED"
            for sibling in session.scalars(select(Checkpoint).where(Checkpoint.job_id == job.id, Checkpoint.fencing_token == checkpoint.fencing_token, Checkpoint.state != "ACCEPTED")):
                sibling.state = "REJECTED"
            self._emit(session, "checkpoint.rejected", job.id, {"checkpoint_hash": checkpoint_hash, "step": checkpoint.logical_step, "signer": checkpoint.signer, "reason": ", ".join(reasons), "provider_id": self._provider_of(session, checkpoint.signer)})

            old = session.scalar(select(JobAssignment).where(JobAssignment.job_id == job.id, JobAssignment.fencing_token == job.fencing_token))
            if old is not None:
                old.state = "REJECTED"
            target_provider = provider_id or (old.provider_id if old else None)
            if target_provider:
                reputation = session.scalar(select(ProviderReputation).where(ProviderReputation.provider_id == target_provider))
                if reputation is None:
                    reputation = ProviderReputation(provider_id=target_provider)
                    session.add(reputation)
                    session.flush()
                reputation.rejected_candidates = (reputation.rejected_candidates or 0) + 1
            if job.state in (JobState.RUNNING.value, JobState.VALIDATING.value, JobState.ASSIGNED.value):
                self._move(session, job, JobState.RECOVERING, "Quality Guard rejected a candidate: " + ", ".join(reasons))
            old_exec = old.exec_config if old else None
            job_service.bump_fencing(session, job)
            session.commit()
            record = RecoveryEvent(job_id=job.id, attempt=job.attempt, kind="ROLLBACK", cause="QUALITY_REJECTED", injected=False, detected_at=datetime.now(timezone.utc), fenced_at=datetime.now(timezone.utc))
            session.add(record)
            self._step(session, job, "FENCED", "Rejected worker can no longer write", provider_id=target_provider)
            if old is not None:
                try:
                    self.agents.stop_run(old.provider_id, job.id, old.fencing_token)
                except Exception:
                    log.info("could not reach provider %s to stop its run", old.provider_id)
            return self._recover(session, job, record, target_provider, old_exec, None)

    def on_run_finished(self, job_id: str, fencing_token: int) -> str:
        """Finished training doesn't complete the job. Only the Quality Guard can."""
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            job_service.require_current_token(job, fencing_token)
            if job.state != JobState.RUNNING.value:
                return job.state
            self._move(session, job, JobState.VALIDATING, "Training finished, final evaluation running")
            tip = session.scalar(
                select(Checkpoint).where(Checkpoint.job_id == job.id, Checkpoint.state.in_(("VERIFIED", "ACCEPTED"))).order_by(Checkpoint.logical_step.desc()).limit(1)
            )
            if tip is None or tip.logical_step < job.spec["total_logical_steps"]:
                # A provider saying it's finished without a final checkpoint proves nothing.
                session.commit()
                return self.on_failure(job.id, provider_id=None, cause="NO_FINAL_CHECKPOINT", fencing_token=fencing_token)
            if self.quality is None:
                return job.state
            reference = job_service.latest_accepted_checkpoint(session, job)
            tip_hash = tip.manifest_hash
            reference_hash = reference.manifest_hash if reference else None
            session.commit()
            try:
                verdict = self.quality.evaluate(job.id, tip_hash, reference_hash, is_final=True)
            except Exception as exc:
                # No quorum isn't a pass. The job stays in VALIDATING, and the money stays locked.
                self._emit(session, "failure.warning", job.id, {"rule": "NO_VERIFIER_QUORUM", "detail": str(exc)[:200]})
                return job.state
        return self.on_final_verdict(job_id, tip_hash, verdict)

    def on_final_verdict(self, job_id: str, checkpoint_hash: str, verdict: str, reasons: list[str] | None = None) -> str:
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            if job.state != JobState.VALIDATING.value:
                return job.state
            if verdict != "PASS":
                session.commit()
                return self.on_candidate_rejected(job_id, checkpoint_hash, None, reasons or ["FINAL_EVALUATION_FAILED"])
            checkpoint = session.scalar(select(Checkpoint).where(Checkpoint.job_id == job.id, Checkpoint.manifest_hash == checkpoint_hash))
            checkpoint.state = "ACCEPTED"
            self._emit(session, "checkpoint.accepted", job.id, {"checkpoint_hash": checkpoint_hash, "step": checkpoint.logical_step, "signer": checkpoint.signer, "reason": None})
            self._move(session, job, JobState.COMPLETED, "Final checkpoint accepted by verifier quorum")
            state = job.state
        self._notify_finished(job_id)
        return state

    # ------------------------------------------------------------ internals

    def _descendants(self, session: Session, job: Job, root: Checkpoint) -> list[Checkpoint]:
        found: list[Checkpoint] = []
        frontier = [root.manifest_hash]
        while frontier:
            current = frontier.pop()
            for child in session.scalars(select(Checkpoint).where(Checkpoint.job_id == job.id, Checkpoint.parent_hash == current)):
                found.append(child)
                frontier.append(child.manifest_hash)
        return found

    def _select_checkpoint(self, session: Session, job: Job) -> Checkpoint | None:
        """The newest verified checkpoint. One failing its handover check is rejected, and the next one is tried."""
        unavailable: set[str] = set()
        while True:
            checkpoint = session.scalar(
                select(Checkpoint)
                .where(Checkpoint.job_id == job.id, Checkpoint.state.in_(("VERIFIED", "ACCEPTED")), Checkpoint.manifest_hash.not_in(unavailable))
                .order_by(Checkpoint.logical_step.desc(), Checkpoint.created_at.desc())
                .limit(1)
            )
            if checkpoint is None:
                return None
            if checkpoint.state == "ACCEPTED" or self.quality is None:
                return checkpoint
            self._step(session, job, "HANDOVER_EVAL", f"Evaluating the checkpoint at step {checkpoint.logical_step} before building on it")
            reference = job_service.latest_accepted_checkpoint(session, job)
            session.commit()
            try:
                verdict = self.quality.evaluate(job.id, checkpoint.manifest_hash, reference.manifest_hash if reference else None, is_final=False)
            except Exception as exc:
                # A missing answer is not a failed evaluation. Try an older checkpoint without rejecting this one.
                self._emit(session, "failure.warning", job.id, {"rule": "NO_VERIFIER_QUORUM", "detail": str(exc)[:200]})
                unavailable.add(checkpoint.manifest_hash)
                continue
            data = {"checkpoint_hash": checkpoint.manifest_hash, "step": checkpoint.logical_step, "signer": checkpoint.signer, "reason": None}
            if verdict == "PASS":
                checkpoint.state = "ACCEPTED"
                self._emit(session, "checkpoint.accepted", job.id, data)
                return checkpoint
            checkpoint.state = "REJECTED"
            self._emit(session, "checkpoint.rejected", job.id, dict(data, reason="Failed the handover evaluation"))

    def _abort(self, session: Session, job: Job, reason: str) -> str:
        self._step(session, job, "ABORT", reason)
        self._move(session, job, JobState.FAILED, reason)
        self._notify_finished(job.id)
        return job.state

    def _notify_finished(self, job_id: str) -> None:
        """Settlement can't be allowed to break the job flow, so its failures are only logged here."""
        if self.on_job_finished is None:
            return
        try:
            self.on_job_finished(job_id)
        except Exception:
            log.exception("settlement couldn't start for job %s", job_id)

    def _recover(self, session: Session, job: Job, record: RecoveryEvent, old_provider: str | None, old_exec: dict | None, last_step: int | None) -> str:
        if job.attempt > MAX_ATTEMPTS:
            return self._abort(session, job, f"Gave up after {MAX_ATTEMPTS} attempts")

        checkpoint = self._select_checkpoint(session, job)
        start_step = checkpoint.logical_step if checkpoint else 0
        lost = max(0, last_step - start_step) if last_step is not None else None
        record.from_checkpoint_id = checkpoint.id if checkpoint else None
        record.lost_steps = lost
        detail = f"Resuming from the checkpoint at step {start_step}" if checkpoint else "No valid checkpoint yet, restarting from step 0"
        self._step(session, job, "SELECT_CHECKPOINT", detail, lost, checkpoint_hash=checkpoint.manifest_hash if checkpoint else None, resume_step=start_step)

        self._step(session, job, "FIND_CANDIDATES", "Compiling for each available provider")
        placement, rejected = self._choose_placement(session, job, start_step)
        if placement is None:
            reasons = "; ".join(f"{d.provider_id}: {d.code}" for d in rejected) or "no providers available"
            return self._abort(session, job, f"No eligible replacement: {reasons}")

        adaptation_hash = None
        from_hash = checkpoint.manifest_hash if checkpoint else "0" * 64
        manifest, adaptation_hash = build_adaptation_manifest(
            job_id=job.id,
            attempt=job.attempt,
            from_checkpoint=from_hash,
            from_provider=old_provider or placement.provider_id,
            from_exec=old_exec or placement.exec_config,
            immutable_hash=job.spec_hash,
            logical_batch_size=job.spec["logical_batch_size"],
            approved=placement.decision,
            rejected=rejected,
        )
        session.add(
            AdaptationManifest(job_id=job.id, attempt=job.attempt, target_provider_id=placement.provider_id, manifest=manifest, manifest_hash=adaptation_hash, verdict="ACCEPTED")
        )
        self._step(session, job, "COMPILE", f"Approved micro-batch {placement.exec_config['micro_batch']} x {placement.exec_config['accumulation']}, {placement.exec_config['amp']}")
        self._step(session, job, "ASSIGN", f"Provider {placement.provider_id}", provider_id=placement.provider_id)
        try:
            self._launch(session, job, placement, checkpoint, adaptation_hash)
        except Exception as exc:
            session.rollback()
            log.warning("could not start run on %s: %s", placement.provider_id, exc)
            assignment = session.scalar(select(JobAssignment).where(JobAssignment.job_id == job.id, JobAssignment.fencing_token == job.fencing_token))
            if assignment is not None:
                assignment.state = "FAILED"
            job_service.bump_fencing(session, job)
            session.commit()
            return self._recover(session, job, record, placement.provider_id, placement.exec_config, last_step)
        self._step(session, job, "RESTORE", "Replacement is restoring the checkpoint")
        if self.on_recovered is not None:
            try:
                self.on_recovered(job.id, job.attempt, adaptation_hash)
            except Exception:  # Anchoring is a public log, and recovery doesn't depend on it.
                log.exception("couldn't anchor the recovery of job %s", job.id)
        return job.state
