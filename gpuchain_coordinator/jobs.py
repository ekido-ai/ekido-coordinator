"""Job state changes and fencing. All writes to job state go through here."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from gpuchain_core.canonical import document_hash
from gpuchain_core.job_states import JobState, check_transition

from .models import Checkpoint, Job, JobAssignment, JobTransition, Provider, ProviderKey, ProviderKeyProof, new_id


class StaleVersion(Exception):
    """Someone else changed the job first. Re-read and decide again."""


class StaleFencingToken(Exception):
    """The writer has been replaced. The agent must kill its runtime."""

    def __init__(self, supplied: int, current: int):
        super().__init__(f"token {supplied} is older than {current}")
        self.supplied = supplied
        self.current = current


def create_job(
    session: Session,
    *,
    spec: dict,
    policy: dict,
    budget: Decimal,
    deadline_seconds: int,
    client_wallet_id: str | None = None,
    client_address: str | None = None,
    reserved_id: str | None = None,
) -> Job:
    job_id = reserved_id or new_id()
    job = Job(
        id=job_id,
        client_wallet_id=client_wallet_id,
        # jobId on-chain. The random row ID makes it unguessable, so two identical specs don't collide.
        # Known limit: the contract doesn't tie the ID to the client, so someone watching pending
        # transactions could copy it and block this createJob. Accepted for a testnet demo.
        onchain_id="0x" + document_hash({"job": job_id, "client": client_address.lower() if client_address else None}),
        spec=spec,
        spec_hash=document_hash(spec),
        policy=policy,
        policy_hash=document_hash(policy),
        budget=budget,
        deadline=datetime.now(timezone.utc) + timedelta(seconds=deadline_seconds),
        state=JobState.OPEN.value,
    )
    session.add(job)
    session.flush()
    return job


def transition(session: Session, job: Job, target: JobState, *, reason: str | None = None, actor: str | None = None) -> None:
    """Compare-and-set on state_version, so two concurrent changes can't both win."""
    current = JobState(job.state)
    check_transition(current, target)
    expected = job.state_version
    result = session.execute(
        update(Job)
        .where(Job.id == job.id, Job.state == current.value, Job.state_version == expected)
        .values(state=target.value, state_version=expected + 1)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        session.rollback()
        raise StaleVersion(f"job {job.id} changed while moving {current} to {target}")
    seq = (session.scalar(select(func.count()).select_from(JobTransition).where(JobTransition.job_id == job.id)) or 0) + 1
    session.add(JobTransition(job_id=job.id, seq=seq, from_state=current.value, to_state=target.value, reason=reason, actor=actor))
    session.flush()
    session.refresh(job)


def bump_fencing(session: Session, job: Job) -> int:
    """Called first in any recovery. After this, the previous worker's writes are refused."""
    result = session.execute(
        update(Job)
        .where(Job.id == job.id)
        .values(fencing_token=Job.fencing_token + 1, attempt=Job.attempt + 1)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        raise StaleVersion(f"job {job.id} not found")
    session.flush()
    session.refresh(job)
    return job.fencing_token


def assign(session: Session, job: Job, provider_id: str, exec_config: dict, *, already_fenced: bool = False) -> JobAssignment:
    """Recovery fences the moment a failure is detected, long before a replacement is known."""
    token = job.fencing_token if already_fenced else bump_fencing(session, job)
    assignment = JobAssignment(job_id=job.id, provider_id=provider_id, attempt=job.attempt, fencing_token=token, exec_config=exec_config)
    session.add(assignment)
    session.flush()
    return assignment


def require_current_token(job: Job, supplied: int) -> None:
    if supplied != job.fencing_token:
        raise StaleFencingToken(supplied, job.fencing_token)


class LineageError(ValueError):
    pass


def commit_checkpoint(session: Session, job: Job, manifest: dict, *, fencing_token: int, signer: str | None = None, run_id: str | None = None) -> Checkpoint:
    """The atomic point of the checkpoint lifecycle. Idempotent for the same manifest."""
    require_current_token(job, fencing_token)
    if manifest.get("fencing_token") != fencing_token:
        raise StaleFencingToken(manifest.get("fencing_token", -1), fencing_token)
    if manifest.get("immutable_hash") != job.spec_hash:
        raise LineageError("checkpoint belongs to a different job spec")

    manifest_hash = document_hash(manifest)
    existing = session.scalar(select(Checkpoint).where(Checkpoint.manifest_hash == manifest_hash))
    if existing is not None:
        return existing

    parent = manifest.get("parent_manifest_hash")
    step = manifest["logical_step"]
    if parent is not None:
        parent_row = session.scalar(select(Checkpoint).where(Checkpoint.manifest_hash == parent, Checkpoint.job_id == job.id))
        if parent_row is None:
            raise LineageError("parent checkpoint is unknown")
        if parent_row.state == "REJECTED":
            raise LineageError("parent checkpoint was rejected")
        if step <= parent_row.logical_step:
            raise LineageError("logical step must increase along the lineage")

    files = manifest.get("files", {})
    checkpoint = Checkpoint(
        job_id=job.id,
        run_id=run_id,
        manifest_hash=manifest_hash,
        parent_hash=parent,
        logical_step=step,
        fencing_token=fencing_token,
        size_bytes=sum(f.get("bytes", 0) for f in files.values()),
        signer=signer,
        manifest=manifest,
    )
    session.add(checkpoint)
    session.flush()
    return checkpoint


def latest_resumable_checkpoint(session: Session, job: Job) -> Checkpoint | None:
    """The newest integrity-verified or accepted checkpoint. Never a rejected or unverified one."""
    return session.scalar(
        select(Checkpoint)
        .where(Checkpoint.job_id == job.id, Checkpoint.state.in_(("VERIFIED", "ACCEPTED")))
        .order_by(Checkpoint.logical_step.desc(), Checkpoint.created_at.desc())
        .limit(1)
    )


def latest_accepted_checkpoint(session: Session, job: Job) -> Checkpoint | None:
    """The rollback target and the Quality Guard's reference."""
    return session.scalar(
        select(Checkpoint)
        .where(Checkpoint.job_id == job.id, Checkpoint.state == "ACCEPTED")
        .order_by(Checkpoint.logical_step.desc(), Checkpoint.created_at.desc())
        .limit(1)
    )


def accepted_lineage(session: Session, tip: Checkpoint) -> list[Checkpoint]:
    """Walks parents from the tip to the root. This chain alone decides who gets paid."""
    chain = [tip]
    seen = {tip.manifest_hash}
    current = tip
    while current.parent_hash is not None:
        parent = session.scalar(select(Checkpoint).where(Checkpoint.manifest_hash == current.parent_hash, Checkpoint.job_id == tip.job_id))
        if parent is None or parent.manifest_hash in seen:
            raise LineageError("broken lineage")
        chain.append(parent)
        seen.add(parent.manifest_hash)
        current = parent
    chain.reverse()
    return chain


def enroll_worker_key(session: Session, provider: Provider, key: str, enrollment_signature: str | None = None) -> bool:
    """False if another provider already uses this key, because it could then claim that provider's accepted work."""
    key = key.lower()
    known = session.scalar(select(ProviderKey).where(ProviderKey.key == key))
    if known is not None and known.provider_id != provider.id:
        return False
    if known is None:
        session.add(ProviderKey(provider_id=provider.id, key=key))
    if enrollment_signature is not None:
        proof = session.scalar(select(ProviderKeyProof).where(ProviderKeyProof.key == key))
        if proof is not None and proof.provider_id != provider.id:
            return False
        if proof is None:
            session.add(ProviderKeyProof(provider_id=provider.id, key=key, signature=enrollment_signature))
    provider.enrolled_key = key
    return True


def providers_by_signer(session: Session) -> dict[str, Provider]:
    """Who each signing key belongs to, including keys from before an agent restarted."""
    providers = {p.id: p for p in session.scalars(select(Provider))}
    owners = {p.enrolled_key.lower(): p for p in providers.values() if p.enrolled_key}
    for row in session.scalars(select(ProviderKey)):
        owners[row.key] = providers[row.provider_id]
    return owners


def steps_by_signer(chain: list[Checkpoint]) -> dict[str, int]:
    """Logical steps each signer contributed to the chain. Rolled-back work isn't in the chain, so it earns nothing."""
    totals: dict[str, int] = {}
    previous = 0
    for checkpoint in chain:
        signer = (checkpoint.signer or "").lower()
        totals[signer] = totals.get(signer, 0) + checkpoint.logical_step - previous
        previous = checkpoint.logical_step
    return totals


def compute_payouts(steps: dict[str, int], rates: dict[str, Decimal], budget: Decimal) -> tuple[dict[str, Decimal], Decimal]:
    """payout = steps x rate, stopping at the budget. Whatever remains goes back to the client."""
    payouts: dict[str, Decimal] = {}
    remaining = budget
    for signer in sorted(steps):
        amount = min(remaining, steps[signer] * rates[signer])
        if amount > 0:
            payouts[signer] = amount
            remaining -= amount
    return payouts, remaining
