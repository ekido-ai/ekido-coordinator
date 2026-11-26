"""Database schema, from plan.md section E. Portable across Postgres (real) and SQLite (fast tests)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Timestamped:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Timestamped, Base):
    __tablename__ = "users"
    display_name: Mapped[str | None] = mapped_column(String(120))


class Wallet(Timestamped, Base):
    __tablename__ = "wallets"
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    address: Mapped[str] = mapped_column(String(42), unique=True)  # Stored lowercase.
    role: Mapped[str] = mapped_column(String(32))  # client, provider, verifier or arbiter.


class GpuHardware(Timestamped, Base):
    __tablename__ = "gpu_hardware"
    uuid_hash: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(120))
    vram_mib: Mapped[int] = mapped_column(Integer)
    compute_cap: Mapped[str | None] = mapped_column(String(8))
    driver: Mapped[str | None] = mapped_column(String(32))
    native_bf16: Mapped[bool | None] = mapped_column(Boolean)
    # How far to believe this record. Self-reported data is never presented as verified.
    evidence: Mapped[str] = mapped_column(String(20), default="CLAIMED")


class Provider(Timestamped, Base):
    __tablename__ = "providers"
    wallet_id: Mapped[str] = mapped_column(ForeignKey("wallets.id"))
    name: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(32), default="OFFLINE")
    # REAL, CAPPED_PROFILE or INJECTED_FAULT. Drives the truth badge, so the UI can't hide it.
    kind: Mapped[str] = mapped_column(String(32), default="REAL")
    physical_gpu_id: Mapped[str | None] = mapped_column(ForeignKey("gpu_hardware.id"))
    capacity_mib: Mapped[int | None] = mapped_column(Integer)  # The enforced cap for capped profiles.
    price_per_step: Mapped[Decimal] = mapped_column(Numeric(20, 6), default=Decimal("0"))
    enrolled_key: Mapped[str | None] = mapped_column(String(42))
    enrollment_sig: Mapped[str | None] = mapped_column(String(132))


class ProviderKey(Timestamped, Base):
    """Every worker key a provider has enrolled. A restarted agent gets a new key, and its earlier work still counts."""

    __tablename__ = "provider_keys"
    provider_id: Mapped[str] = mapped_column(ForeignKey("providers.id"))
    key: Mapped[str] = mapped_column(String(42), unique=True)


class ProviderKeyProof(Timestamped, Base):
    """Wallet-signed delegation, retained for each worker key across agent restarts."""

    __tablename__ = "provider_key_proofs"
    provider_id: Mapped[str] = mapped_column(ForeignKey("providers.id"))
    key: Mapped[str] = mapped_column(String(42), unique=True)
    signature: Mapped[str] = mapped_column(String(132))


class Worker(Timestamped, Base):
    __tablename__ = "workers"
    provider_id: Mapped[str] = mapped_column(ForeignKey("providers.id"))
    session_pubkey: Mapped[str | None] = mapped_column(String(42))
    mode: Mapped[str] = mapped_column(String(8), default="docker")
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    state: Mapped[str] = mapped_column(String(32), default="OFFLINE")


class Benchmark(Timestamped, Base):
    __tablename__ = "benchmarks"
    provider_id: Mapped[str] = mapped_column(ForeignKey("providers.id"))
    runtime_hash: Mapped[str | None] = mapped_column(String(64))
    micro_batch: Mapped[int] = mapped_column(Integer)
    amp: Mapped[str] = mapped_column(String(8))
    steps_per_s: Mapped[float] = mapped_column(Float)
    peak_mem_mib: Mapped[float] = mapped_column(Float)
    nonce: Mapped[str | None] = mapped_column(String(66))
    sig: Mapped[str | None] = mapped_column(String(132))


class Job(Timestamped, Base):
    __tablename__ = "jobs"
    client_wallet_id: Mapped[str | None] = mapped_column(ForeignKey("wallets.id"))
    onchain_id: Mapped[str] = mapped_column(String(66), unique=True)
    spec: Mapped[dict] = mapped_column(JSON)
    spec_hash: Mapped[str] = mapped_column(String(64))
    policy: Mapped[dict] = mapped_column(JSON)
    policy_hash: Mapped[str] = mapped_column(String(64))
    budget: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    state: Mapped[str] = mapped_column(String(32), default="OPEN")
    # Optimistic lock: every state change must name the version it read.
    state_version: Mapped[int] = mapped_column(Integer, default=0)
    # Only ever increases. A write carrying an older token comes from a zombie worker and is refused.
    fencing_token: Mapped[int] = mapped_column(BigInteger, default=0)
    attempt: Mapped[int] = mapped_column(Integer, default=0)

    transitions: Mapped[list["JobTransition"]] = relationship(back_populates="job", order_by="JobTransition.seq")


class JobKey(Timestamped, Base):
    """A job's encryption key (ticket A12). The recipient is public; the identity exists here only wrapped by KMS."""

    __tablename__ = "job_keys"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), unique=True)
    recipient: Mapped[str] = mapped_column(String(80))  # age1..., the public half
    wrapped_identity: Mapped[bytes | None] = mapped_column(LargeBinary)  # None once destroyed (delete my data)
    destroyed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

class Proof(Timestamped, Base):
    """A paid job's public proof (ticket C25): hashes only. `payouts` is the payout split in whole paise, kept here so the
    proof can be rebuilt and checked; only its hash goes into the batch."""

    __tablename__ = "proofs"
    job_id: Mapped[str] = mapped_column(String(36), unique=True)
    terms_hash: Mapped[str] = mapped_column(String(64))
    verdict_hash: Mapped[str] = mapped_column(String(64))
    payouts: Mapped[dict] = mapped_column(JSON)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("proof_batches.id"), index=True)
    leaf_index: Mapped[int | None] = mapped_column(Integer)


class ProofBatch(Timestamped, Base):
    """A group of proofs whose Merkle root is written to the ProofAnchor contract. PENDING until the write is confirmed."""

    __tablename__ = "proof_batches"
    root: Mapped[str] = mapped_column(String(64), unique=True)  # hex, no 0x
    count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="PENDING")  # PENDING, ANCHORED or MISMATCH
    tx_hash: Mapped[str | None] = mapped_column(String(66))
    onchain_id: Mapped[int | None] = mapped_column(Integer)  # the batch id inside the contract

class JobBinding(Timestamped, Base):
    """Client-signed binding for a secure settlement; absent for legacy jobs."""

    __tablename__ = "job_bindings"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), unique=True)
    client_address: Mapped[str] = mapped_column(String(42))
    document: Mapped[dict] = mapped_column(JSON)


class QualityDecision(Timestamped, Base):
    """Signed quorum evidence retained across coordinator restarts."""

    __tablename__ = "quality_decisions"
    __table_args__ = (UniqueConstraint("job_id", "checkpoint_hash"),)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    checkpoint_hash: Mapped[str] = mapped_column(String(64))
    reports: Mapped[list] = mapped_column(JSON)


class JobTransition(Timestamped, Base):
    __tablename__ = "job_transitions"
    __table_args__ = (UniqueConstraint("job_id", "seq"),)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    seq: Mapped[int] = mapped_column(Integer)
    from_state: Mapped[str] = mapped_column(String(32))
    to_state: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str | None] = mapped_column(Text)
    actor: Mapped[str | None] = mapped_column(String(64))

    job: Mapped[Job] = relationship(back_populates="transitions")


class IdempotencyRecord(Timestamped, Base):
    """Stores the first response, so a retried request replays it instead of acting twice."""

    __tablename__ = "idempotency_records"
    key: Mapped[str] = mapped_column(String(128), unique=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    status_code: Mapped[int] = mapped_column(Integer)
    response: Mapped[dict] = mapped_column(JSON)


class JobAssignment(Timestamped, Base):
    __tablename__ = "job_assignments"
    __table_args__ = (UniqueConstraint("job_id", "attempt"),)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    provider_id: Mapped[str] = mapped_column(ForeignKey("providers.id"))
    attempt: Mapped[int] = mapped_column(Integer)
    fencing_token: Mapped[int] = mapped_column(BigInteger)
    state: Mapped[str] = mapped_column(String(32), default="ASSIGNED")
    exec_config: Mapped[dict | None] = mapped_column(JSON)


class TrainingRun(Timestamped, Base):
    __tablename__ = "training_runs"
    assignment_id: Mapped[str] = mapped_column(ForeignKey("job_assignments.id"))
    start_step: Mapped[int] = mapped_column(Integer)
    end_step: Mapped[int | None] = mapped_column(Integer)
    accepted_steps: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="RUNNING")


class Checkpoint(Timestamped, Base):
    __tablename__ = "checkpoints"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("training_runs.id"))
    manifest_hash: Mapped[str] = mapped_column(String(64), unique=True)
    parent_hash: Mapped[str | None] = mapped_column(String(64))
    logical_step: Mapped[int] = mapped_column(Integer)
    fencing_token: Mapped[int] = mapped_column(BigInteger)
    # COMMITTED -> VERIFIED -> ACCEPTED or REJECTED.
    state: Mapped[str] = mapped_column(String(32), default="COMMITTED")
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    signer: Mapped[str | None] = mapped_column(String(42))
    manifest: Mapped[dict] = mapped_column(JSON)


class RecoveryEvent(Timestamped, Base):
    __tablename__ = "recovery_events"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    attempt: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32))
    detected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fenced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    from_checkpoint_id: Mapped[str | None] = mapped_column(ForeignKey("checkpoints.id"))
    lost_steps: Mapped[int | None] = mapped_column(Integer)
    cause: Mapped[str | None] = mapped_column(String(32))
    injected: Mapped[bool] = mapped_column(Boolean, default=False)


class AdaptationManifest(Timestamped, Base):
    __tablename__ = "adaptation_manifests"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    attempt: Mapped[int] = mapped_column(Integer)
    target_provider_id: Mapped[str | None] = mapped_column(ForeignKey("providers.id"))
    manifest: Mapped[dict] = mapped_column(JSON)
    manifest_hash: Mapped[str] = mapped_column(String(64))
    verdict: Mapped[str] = mapped_column(String(32))
    reject_reason: Mapped[str | None] = mapped_column(Text)


class QualityEvaluation(Timestamped, Base):
    __tablename__ = "quality_evaluations"
    __table_args__ = (UniqueConstraint("checkpoint_id", "verifier_wallet_id"),)
    checkpoint_id: Mapped[str] = mapped_column(ForeignKey("checkpoints.id"))
    verifier_wallet_id: Mapped[str] = mapped_column(ForeignKey("wallets.id"))
    eval_version_hash: Mapped[str] = mapped_column(String(64))
    reference_checkpoint_id: Mapped[str | None] = mapped_column(ForeignKey("checkpoints.id"))
    metrics: Mapped[dict] = mapped_column(JSON)
    correctness_hash: Mapped[str | None] = mapped_column(String(64))
    ci_lower: Mapped[float] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(String(32))
    report_hash: Mapped[str] = mapped_column(String(64))
    sig: Mapped[str] = mapped_column(String(132))


class VerificationAttestation(Timestamped, Base):
    __tablename__ = "verification_attestations"
    __table_args__ = (UniqueConstraint("job_id", "verdict_hash", "signer"),)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    verdict_hash: Mapped[str] = mapped_column(String(66))
    signer: Mapped[str] = mapped_column(String(42))
    signature: Mapped[str] = mapped_column(String(132))


class EscrowRef(Timestamped, Base):
    __tablename__ = "escrow_refs"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), unique=True)
    chain_id: Mapped[int] = mapped_column(BigInteger)
    contract: Mapped[str] = mapped_column(String(42))
    create_tx: Mapped[str | None] = mapped_column(String(66))
    state: Mapped[str] = mapped_column(String(32), default="Funded")


class Settlement(Timestamped, Base):
    __tablename__ = "settlements"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), unique=True)
    verdict: Mapped[dict] = mapped_column(JSON)
    submit_tx: Mapped[str | None] = mapped_column(String(66))
    finalize_tx: Mapped[str | None] = mapped_column(String(66))
    status: Mapped[str] = mapped_column(String(32), default="PENDING")


class SettlementProposal(Timestamped, Base):
    """Pending verifier inputs, retained so signatures can be re-collected after restart."""

    __tablename__ = "settlement_proposals"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), unique=True)
    inputs: Mapped[dict] = mapped_column(JSON)


class ProviderReputation(Timestamped, Base):
    __tablename__ = "provider_reputation"
    provider_id: Mapped[str] = mapped_column(ForeignKey("providers.id"), unique=True)
    jobs_ok: Mapped[int] = mapped_column(Integer, default=0)
    jobs_failed: Mapped[int] = mapped_column(Integer, default=0)
    recoveries: Mapped[int] = mapped_column(Integer, default=0)
    rejected_candidates: Mapped[int] = mapped_column(Integer, default=0)
    steps_accepted: Mapped[int] = mapped_column(BigInteger, default=0)
    passport_hash: Mapped[str | None] = mapped_column(String(64))


class TelemetrySample(Timestamped, Base):
    __tablename__ = "telemetry_samples"
    worker_id: Mapped[str] = mapped_column(ForeignKey("workers.id"))
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Nullable on purpose: an unknown reading must stay unknown, never turn into 0.
    util: Mapped[float | None] = mapped_column(Float)
    mem_used: Mapped[float | None] = mapped_column(Float)
    temp_c: Mapped[float | None] = mapped_column(Float)
    power_w: Mapped[float | None] = mapped_column(Float)
    step: Mapped[int | None] = mapped_column(Integer)
    loss: Mapped[float | None] = mapped_column(Float)
    steps_per_s: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="HOST_NVML")


class FaultInjection(Timestamped, Base):
    __tablename__ = "fault_injections"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    kind: Mapped[str] = mapped_column(String(32))
    requested_by: Mapped[str | None] = mapped_column(String(64))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[str | None] = mapped_column(Text)


class OutboxItem(Timestamped, Base):
    """Chain writes queued durably. On retry, read chain state first, so nothing applies twice."""

    __tablename__ = "outbox"
    kind: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON)
    key: Mapped[str] = mapped_column(String(128), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    attempts: Mapped[int] = mapped_column(Integer, default=0)


class DashboardEventRow(Base):
    """Stored events, so a reconnecting dashboard can catch up. `seq` matches the event's `id`."""

    __tablename__ = "dashboard_events"
    __table_args__ = (UniqueConstraint("job_id", "seq"),)
    pk: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_id: Mapped[str | None] = mapped_column(String(36))
    seq: Mapped[int] = mapped_column(BigInteger)
    event: Mapped[dict] = mapped_column(JSON)


class ChainLog(Base):
    """A raw contract log copied from the chain by the indexer. Key: "block:logIndex"."""

    __tablename__ = "chain_logs"
    key: Mapped[str] = mapped_column(String(48), primary_key=True)
    block_number: Mapped[int] = mapped_column(BigInteger, index=True)
    log_index: Mapped[int] = mapped_column(Integer)
    tx_hash: Mapped[str] = mapped_column(String(66))
    address: Mapped[str] = mapped_column(String(42))
    topics: Mapped[list] = mapped_column(JSON)
    data: Mapped[str] = mapped_column(Text)


class IndexCursor(Base):
    """The last block an indexer has fully copied."""

    __tablename__ = "index_cursors"
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    block: Mapped[int] = mapped_column(BigInteger)


class DistillationJob(Base):
    """One example-generation request (ticket #154, contract distill-gen/v1). The seeds are customer text and are dropped when the job ends."""

    __tablename__ = "distillation_jobs"
    __table_args__ = (UniqueConstraint("distillation_id", "attempt", name="uq_distillation_attempt"),)
    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    distillation_id: Mapped[str] = mapped_column(String(64), index=True)
    attempt: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(16), index=True)
    spec: Mapped[dict] = mapped_column(JSON)
    spec_sha256: Mapped[str] = mapped_column(String(64))
    seeds_jsonl: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
