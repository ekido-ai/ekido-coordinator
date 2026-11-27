"""Coordinator HTTP and WebSocket API. Agent gateway and recovery attach here as they are built."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import shutil
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import and_, create_engine, func, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ekido_core.canonical import document_hash
from ekido_core.job_spec import policy_problems, spec_problems
from ekido_core.job_states import IllegalTransition, JobState
from ekido_core.payout_terms import validate_terms
from ekido_core.protocol_fee import fee_amount, fee_terms, provider_budget
from ekido_core.provider_meta import metadata_hash, provider_metadata
from ekido_core.signing import verify_document

from . import distillations, job_keys
from . import jobs as job_service
from .agent_gateway import AgentGateway
from .api_guard import ApiGuard
from .public_view import Redactor, load_identities
from .rate_limit import RateLimiter
from .artifacts import ArtifactError, ArtifactStore, S3Blobs, UploadBudget, UploadBudgetExceeded
from .model_bundle import bundle as model_bundle
from .events import EventBus, events_after, record_event
from .models import Base, Checkpoint, FaultInjection, GpuHardware, IdempotencyRecord, Job, JobBinding, Provider, ProviderKey, QualityDecision, RecoveryEvent, Settlement, Wallet
from .orchestrator import Orchestrator, QualityLink
from .quality_service import QuorumQuality

FAULT_KINDS = ("KILL_CONTAINER", "KILL_AGENT", "HEARTBEAT_LOSS", "OOM", "DISCONNECT", "CORRUPT_CHECKPOINT", "CRASH_MID_CHECKPOINT", "POISON", "THERMAL_WARNING")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "sqlite://"
    demo_mode: bool = False
    api_token: str = ""  # Lets another machine use the operator API. Empty means only this machine can.
    # How a job's encryption key is stored at rest (A12): "kms" on staging and live, "local" for tests and a laptop.
    job_key_wrap: str = ""  # no default: startup refuses unless "kms", or "local" with LOCAL_WRAP_IDENTITY (tests and laptops)
    kms_key_id: str = "alias/ekido-data"
    kms_region: str = "ap-south-1"
    local_wrap_identity: str = ""  # an age secret key, for "local" only; empty means a throwaway key
    # A13: send each run the job's public key so its checkpoints are written encrypted. Leave off until the verifier can read them.
    encrypt_checkpoints: bool = False
    open_api: bool = False  # Only for a container whose port is published on 127.0.0.1, where clients never look local.
    # Protocol fee on accepted work, paid by the client on top. When set, every new job's spec must carry exactly these terms.
    protocol_fee_bps: int = 0
    protocol_fee_recipient: str = ""
    public_read: bool = False  # A public server: anyone may read /healthz, /readyz and /fees. Everything else needs API_TOKEN.
    # Per-client limits on every route; this machine's own traffic (operator, verifiers, monitor) is exempt.
    # Chain indexer (chain_index.py): copies escrow/registry logs so the dashboard loads history in one request.
    index_enabled: bool = False
    index_rpc_url: str = "https://base-sepolia-rpc.publicnode.com"
    index_addresses: str = ""  # Comma-separated contract addresses.
    index_from_block: int = 0
    rate_limit_enabled: bool = True
    # Per client IP. Many people can share one IP (a college or office network), and one dashboard tab reads
    # about 35 times a minute, so this leaves room for a classroom without letting one script flood the server.
    rate_limit_reads_per_minute: int = 600
    rate_limit_writes_per_minute: int = 40
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3100", "http://127.0.0.1:3100"]
    artifact_dir: str = "data/artifacts"
    # With a bucket set, workers upload and download checkpoint blobs straight to and from S3 through presigned
    # links. Blobs already on disk, and uploads from older workers, keep using artifact_dir.
    artifact_s3_bucket: str = ""
    artifact_s3_region: str = ""
    # Checkpoint bytes one job may keep on disk at once, counted the way retention keeps them. 0 turns the cap off.
    max_job_storage_bytes: int = 5 * 1024**3
    # Bytes one provider may upload that no committed checkpoint references yet.
    max_pending_upload_bytes: int = 2 * 1024**3
    verifier_addresses: str = ""  # Comma-separated. Empty means no quality gate, so jobs stop at VALIDATING.
    verifier_quorum: int = 2
    final_evaluation_timeout_seconds: float = 1800.0  # how long one asking of the final check waits for a quorum
    # Chain settings. With no RPC_URL or ESCROW_ADDRESS, the coordinator runs with no chain.
    rpc_url: str = ""
    chain_id: int = 84532
    # Jobs paid in rupees through the Platform (tickets C7, A5, A9). Both must be set to turn it on; off by default.
    platform_url: str = ""
    platform_internal_token: str = ""
    # Example generation (ticket #154). Off until a device can run it: while off, POST /distillations refuses, so no order waits on it.
    distillation_enabled: bool = False
    escrow_address: str = ""
    coordinator_private_key: str = ""
    explorer_url: str = ""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DistillationRequest(StrictModel):
    distillation_id: str = Field(min_length=1, max_length=64)
    spec: dict[str, Any]
    seeds_jsonl: str = Field(min_length=1, max_length=4_000_000)


class CreateJobRequest(StrictModel):
    workload: dict[str, Any]
    policy: dict[str, Any]
    budget_usdc: Decimal = Field(gt=0, max_digits=20, decimal_places=6)
    deadline_seconds: int = Field(gt=0, le=7 * 24 * 3600)
    job_id: str | None = None
    client_address: str | None = None
    job_binding: dict[str, Any] | None = None


class FaultRequest(StrictModel):
    job_id: str
    kind: Literal[FAULT_KINDS]  # type: ignore[valid-type]


class QualifyRequest(StrictModel):
    workload: dict[str, Any]


class UploadUrlRequest(StrictModel):
    bytes: int = Field(gt=0)


class VerifierSessionRequest(StrictModel):
    message: dict[str, Any]
    signature: str


class VerifierJobKeyRequest(StrictModel):
    address: str
    job_id: str
    recipient: str  # a one-off age public key the verifier made for this task


class ClaimTaskRequest(StrictModel):
    address: str


class SubmitVerdictRequest(StrictModel):
    job_id: str
    signature: str


class SubmitEvaluationRequest(StrictModel):
    task_id: str
    report: dict[str, Any]


class RegisterProviderRequest(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    wallet_address: str = Field(pattern=r"^0x[0-9a-fA-F]{40}$")
    # REAL, a capped profile of a real GPU, or a deliberately bad provider. Shown as a badge in the UI.
    kind: Literal["REAL", "CAPPED_PROFILE", "INJECTED_FAULT"]
    capacity_mib: int | None = Field(default=None, gt=0)
    price_per_step: Decimal = Field(ge=0, max_digits=20, decimal_places=6)


class JoinProviderRequest(StrictModel):
    """A provider signing itself up with its own wallet. See ekido_agent.join."""

    document: dict[str, Any]
    signature: str = Field(min_length=1, max_length=200)


JOIN_KIND = "gpuchain.provider_join.v1"
JOIN_MAX_AGE_SECONDS = 300
MAX_DATASET_BYTES = 20 * 1024 * 1024  # the Platform refuses larger datasets too (ekido-platform v1_datasets)
JOIN_MAX_PRICE = Decimal("1")  # USDC per step. A typo shouldn't list a provider at a ruinous price.
JOIN_MAX_PROVIDERS = 200
JOIN_PER_MINUTE = 10
JOIN_PER_IP_TOTAL = 5
# /readyz thresholds: below these, monitoring should page someone.
READY_MIN_DISK_BYTES = 2 * 1024**3
READY_MIN_GAS_WEI = 2 * 10**15  # 0.002 ETH: a handful of settlements.
READY_MAX_BLOCK_AGE = 300
READY_MAX_BACKLOG = 20  # New providers one address may create while this process runs.
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def caller_ip(request: Request) -> str:
    """The caller's address. Behind our own Caddy every request arrives from loopback, so only then
    is the proxy's X-Forwarded-For trusted; anyone else could forge it."""
    host = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if host in LOOPBACK and forwarded:
        return forwarded.split(",")[0].strip()[:64] or host
    return host


def name_problem(name: str) -> str | None:
    """Printable names only: no control, format or bidirectional-override characters that could disguise one provider as another."""
    import unicodedata

    if any(unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cs", "Zl", "Zp") for ch in name):
        return "name must use printable characters only"
    return None


def join_problems(doc: dict[str, Any]) -> list[str]:
    problems = []
    if doc.get("kind") != JOIN_KIND:
        problems.append(f"kind must be {JOIN_KIND}")
    name = doc.get("name")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
        problems.append("name must be 1 to 60 characters")
    elif (bad := name_problem(name)) is not None:
        problems.append(bad)
    wallet = doc.get("wallet")
    if not isinstance(wallet, str) or not re.fullmatch(r"0x[0-9a-fA-F]{40}", wallet):
        problems.append("wallet must be an address")
    try:
        price = Decimal(str(doc.get("price_per_step")))
        if not (Decimal(0) < price <= JOIN_MAX_PRICE) or price.as_tuple().exponent < -6:
            problems.append(f"price_per_step must be above 0 and at most {JOIN_MAX_PRICE}, with up to 6 decimals")
    except Exception:
        problems.append("price_per_step must be a number")
    if doc.get("device") not in ("cpu", "cuda"):
        problems.append("device must be cpu or cuda")
    capacity = doc.get("capacity_mib")
    if capacity is not None and (type(capacity) is not int or not 1 <= capacity <= 200_000):
        problems.append("capacity_mib must be a whole number of MiB")
    issued = doc.get("ts")
    if type(issued) is not int or abs(time.time() - issued) > JOIN_MAX_AGE_SECONDS:
        problems.append("ts must be the current Unix time, within 5 minutes")
    return problems


def job_view(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "onchain_id": job.onchain_id,
        "state": job.state,
        "state_version": job.state_version,
        "attempt": job.attempt,
        "fencing_token": job.fencing_token,
        "spec": job.spec,
        "spec_hash": job.spec_hash,
        "policy": job.policy,
        "policy_hash": job.policy_hash,
        "budget_usdc": str(job.budget),
        "deadline": job.deadline.isoformat(),
    }


def create_engine_for(database_url: str) -> Engine:
    kwargs: dict[str, Any] = {}
    if database_url in ("sqlite://", "sqlite:///:memory:"):
        kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
    return create_engine(database_url, **kwargs)


def create_app(settings: Settings | None = None, engine: Engine | None = None, quality: QualityLink | None = None) -> FastAPI:
    settings = settings or Settings()
    if engine is None:
        engine = create_engine_for(settings.database_url)
    # SQLite (tests, the old server): create any missing tables. Postgres: the schema belongs to the Alembic migrations
    # in apps/coordinator/alembic, which the service runs before it starts (ExecStartPre).
    if engine.dialect.name == "sqlite":
        Base.metadata.create_all(engine)

    bus = EventBus()
    store = ArtifactStore(Path(settings.artifact_dir))
    blobs = S3Blobs(settings.artifact_s3_bucket, region=settings.artifact_s3_region or None) if settings.artifact_s3_bucket else None
    if quality is None and settings.verifier_addresses:
        quality = QuorumQuality({a.strip() for a in settings.verifier_addresses.split(",") if a.strip()}, settings.verifier_quorum,
                                final_timeout_seconds=settings.final_evaluation_timeout_seconds)
    # Must come after quality is set: the gateway builds its orchestrator immediately.
    job_keys.configure(job_keys.make_wrapper(settings.job_key_wrap, key_id=settings.kms_key_id, region=settings.kms_region, local_identity=settings.local_wrap_identity))
    gateway = AgentGateway(engine, lambda link: Orchestrator(engine, bus, link, quality))
    gateway.orchestrator.encrypt_checkpoints = settings.encrypt_checkpoints
    gateway.orchestrator.max_job_storage_bytes = settings.max_job_storage_bytes or None
    upload_budget = UploadBudget(settings.max_pending_upload_bytes)
    gateway.upload_budget = upload_budget
    if isinstance(quality, QuorumQuality):
        with Session(engine) as session:
            for row in session.scalars(select(QualityDecision)):
                quality.decided_reports[(row.job_id, row.checkpoint_hash)] = row.reports

        def save_decision(job_id: str, checkpoint_hash: str, reports: list[dict]) -> None:
            with Session(engine) as session:
                row = session.scalar(select(QualityDecision).where(QualityDecision.job_id == job_id, QualityDecision.checkpoint_hash == checkpoint_hash))
                if row is None:
                    session.add(QualityDecision(job_id=job_id, checkpoint_hash=checkpoint_hash, reports=reports))
                else:
                    row.reports = reports
                session.commit()

        quality.on_decision = save_decision

        def show_report(job_id: str, checkpoint_hash: str, report: dict) -> None:
            comparison = report.get("comparison", {})
            gateway.orchestrator.emit_live(
                "quality.result",
                job_id,
                {
                    "checkpoint_hash": checkpoint_hash,
                    "verifier": report.get("signer"),
                    "verdict": report.get("verdict"),
                    "reasons": report.get("reasons"),
                    "accuracy": report.get("metrics", {}).get("accuracy"),
                    "difference": comparison.get("difference"),
                    "ci_lower": comparison.get("ci_lower"),
                    "margin": comparison.get("margin"),
                    "is_final": report.get("is_final"),
                    "against_baseline": report.get("reference_checkpoint_hash") is None,
                },
            )

        quality.on_report = show_report

    settlement = None
    if isinstance(quality, QuorumQuality) and settings.escrow_address and settings.rpc_url:
        # Imported here, so a coordinator with no chain doesn't need web3 installed.
        from .chain import EscrowClient
        from .settlement_service import SettlementService

        settlement = SettlementService(
            engine,
            gateway.orchestrator,
            EscrowClient(settings.rpc_url, settings.escrow_address, settings.coordinator_private_key),
            settings.chain_id,
            settings.escrow_address,
            quality.verifiers,
            quality.quorum,
            quality,
            explorer_url=settings.explorer_url or None,
            local_chain=settings.chain_id == 31337,
        )
        gateway.orchestrator.on_job_finished = settlement.propose
        gateway.orchestrator.on_job_started = settlement.on_job_started
        gateway.orchestrator.on_recovered = settlement.on_recovered
        settlement.restore_pending()

    fiat = None
    if settings.platform_url and settings.platform_internal_token:
        from .fiat_settlement import FiatSettlement, HttpPlatformPayments

        fiat = FiatSettlement(engine, HttpPlatformPayments(settings.platform_url, settings.platform_internal_token))
        previous = gateway.orchestrator.on_job_finished

        def finished(job_id: str) -> None:
            if previous is not None:
                previous(job_id)  # a crypto job still goes through the on-chain settlement
            fiat.on_job_finished(job_id)  # raises when the Platform is down; the orchestrator logs it and the sweep retries

        gateway.orchestrator.on_job_finished = finished

    if settings.index_enabled and settings.index_addresses:
        from .chain_index import ChainIndexer, RpcLogs

        ChainIndexer(engine, RpcLogs(settings.index_rpc_url), [a.strip() for a in settings.index_addresses.split(",") if a.strip()], settings.index_from_block).start()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        bus.bind_loop(asyncio.get_running_loop())
        monitor = asyncio.create_task(gateway.monitor())

        async def fiat_sweeper() -> None:
            while True:  # finishes jobs whose settlement call failed (the Platform was down); both Platform calls are idempotent
                await asyncio.sleep(300)
                try:
                    await asyncio.to_thread(fiat.sweep)
                except Exception:
                    logging.getLogger(__name__).exception("fiat settlement sweep failed")

        sweeper = asyncio.create_task(fiat_sweeper()) if fiat is not None else None

        async def evaluation_sweeper() -> None:
            while True:  # asks again about jobs whose final check got no quorum, or that a restart interrupted
                await asyncio.sleep(60)
                try:
                    for job_id in await asyncio.to_thread(gateway.orchestrator.pending_evaluations):
                        await asyncio.to_thread(gateway.orchestrator.evaluate_final, job_id)
                except Exception:
                    logging.getLogger(__name__).exception("final evaluation sweep failed")

        evaluator = asyncio.create_task(evaluation_sweeper()) if quality is not None else None
        yield
        monitor.cancel()
        if evaluator is not None:
            evaluator.cancel()
        if sweeper is not None:
            sweeper.cancel()

    app = FastAPI(title="Ekido Coordinator", version="0.1.0", lifespan=lifespan)
    # A public server shows job pages to anyone, with provider identities replaced (public_view.py, decision 8).
    redactor = Redactor(lambda: load_identities(engine), secret=settings.api_token) if settings.public_read else None
    app.add_middleware(ApiGuard, token=settings.api_token, allowed_origins=settings.cors_origins, open_api=settings.open_api, public_read=settings.public_read, redactor=redactor)
    # Refused before any other work is done...
    app.add_middleware(RateLimiter, reads_per_minute=settings.rate_limit_reads_per_minute, writes_per_minute=settings.rate_limit_writes_per_minute, enabled=settings.rate_limit_enabled)
    # ...but inside CORS, so a browser can read a 429 and its Retry-After instead of seeing a network error,
    # and CORS preflights never spend a client's read budget.
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"], expose_headers=["Retry-After"])
    app.state.bus = bus
    app.state.engine = engine
    app.state.gateway = gateway
    app.state.fiat_settlement = fiat
    app.state.store = store


    def require_artifact_token(authorization: str | None = Header(default=None)) -> str:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token not in gateway.upload_tokens:
            raise HTTPException(status_code=401, detail="a valid artifact token is required")
        return gateway.upload_tokens[token]

    def get_session():
        with Session(engine) as session:
            yield session

    @app.exception_handler(IllegalTransition)
    async def illegal_transition(_: Request, exc: IllegalTransition):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=409, content={"error": "ILLEGAL_TRANSITION", "detail": str(exc)})

    @app.exception_handler(job_service.StaleFencingToken)
    async def stale_token(_: Request, exc: job_service.StaleFencingToken):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=409, content={"error": "STALE_FENCING_TOKEN", "current": exc.current})

    @app.exception_handler(job_service.LineageError)
    async def lineage(_: Request, exc: job_service.LineageError):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=422, content={"error": "INVALID_LINEAGE", "detail": str(exc)})

    def load_job(session: Session, job_id: str) -> Job:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/healthz")
    def healthz(session: Session = Depends(get_session)) -> dict[str, Any]:
        session.execute(select(1))
        return {
            "status": "ok", "demo_mode": settings.demo_mode,
            "chain_id": settings.chain_id, "escrow_address": settings.escrow_address or None,
            "signed_jobs": settlement is not None and bool(settings.coordinator_private_key),
        }

    @app.get("/chain/logs")
    def chain_logs(from_block: int = 0, limit: int = 5000, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Raw escrow and registry logs from the index. Clients decode them and check them against the chain."""
        from .chain_index import logs_since

        return logs_since(session, "escrow+registry", max(from_block, 0), min(max(limit, 1), 10_000))

    @app.get("/readyz")
    def readyz(session: Session = Depends(get_session)) -> JSONResponse:
        """Deep readiness for monitoring: everything a job and a payout depend on. 503 if any part is unhealthy.

        Reports facts only (ages, counts, balances in ETH); no keys, addresses or job contents.
        """
        checks: dict[str, dict[str, Any]] = {}

        def check(name: str, probe) -> None:
            try:
                ok, detail = probe()
            except Exception as exc:  # A failing dependency is a result here, not a crash.
                ok, detail = False, f"{type(exc).__name__}"
            checks[name] = {"ok": bool(ok), "detail": detail}

        check("database", lambda: (session.execute(select(1)).scalar() == 1, "reachable"))

        def artifacts():
            folder = Path(settings.artifact_dir)
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".ready-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            free = shutil.disk_usage(folder).free
            return free >= READY_MIN_DISK_BYTES, f"writable, {free // 2**20} MiB free"

        check("artifacts", artifacts)
        if settlement is not None:
            chain = settlement.chain

            def rpc():
                lag = time.time() - int(chain.w3.eth.get_block("latest")["timestamp"])
                return lag < READY_MAX_BLOCK_AGE, f"latest block {int(lag)} s old"

            def gas():
                wei = chain.w3.eth.get_balance(chain.account.address)
                return wei >= READY_MIN_GAS_WEI, f"{wei / 1e18:.4f} ETH for settlement gas"

            def backlog():
                with settlement.lock:
                    waiting = sum(1 for task in settlement.tasks.values() if not task.submitted)
                return waiting < READY_MAX_BACKLOG, f"{waiting} settlements waiting for signatures"

            check("chain_rpc", rpc)
            check("coordinator_gas", gas)
            check("settlement_backlog", backlog)
        ready = all(c["ok"] for c in checks.values())
        return JSONResponse(status_code=200 if ready else 503, content={"status": "ready" if ready else "degraded", "checks": checks})

    @app.post("/distillations", status_code=201)
    def create_distillation(body: DistillationRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        """The Platform asks for example generation (contract distill-gen/v1). The operator API: the same token and host rules as /jobs."""
        try:
            row = distillations.create(session, distillation_id=body.distillation_id, spec=body.spec, seeds_jsonl=body.seeds_jsonl,
                                       enabled=settings.distillation_enabled)
        except distillations.Refused as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from None
        return {"job_id": row.job_id}

    @app.post("/jobs", status_code=201)
    def create_job(body: CreateJobRequest, session: Session = Depends(get_session), idempotency_key: str | None = Header(default=None)) -> Any:
        # Refused here, not on a provider after the client has paid. Verifiers only need the policy if they're running.
        problems = spec_problems(body.workload) + (policy_problems(body.policy) if settings.verifier_addresses else [])
        if settings.protocol_fee_bps and body.workload.get("protocol_fee") != {"bps": settings.protocol_fee_bps, "recipient": settings.protocol_fee_recipient}:
            problems.append(f"protocol_fee must be {{\"bps\": {settings.protocol_fee_bps}, \"recipient\": \"{settings.protocol_fee_recipient}\"}}: see GET /fees")
        if problems:
            return JSONResponse(status_code=422, content={"error": "INVALID_JOB_SPEC", "problems": problems})
        request_hash = document_hash(body.model_dump(mode="json"))
        if idempotency_key:
            previous = session.scalar(select(IdempotencyRecord).where(IdempotencyRecord.key == idempotency_key))
            if previous is not None:
                if previous.request_hash != request_hash:
                    raise HTTPException(status_code=422, detail="idempotency key reused with a different request")
                return previous.response

        secure_fields = (body.job_id, body.client_address, body.job_binding)
        if any(item is not None for item in secure_fields):
            if not all(item is not None for item in secure_fields):
                raise HTTPException(status_code=422, detail="job_id, client_address and job_binding must be provided together")
            try:
                validate_terms(job_id=body.job_id, client=body.client_address, binding=body.job_binding, spec=body.workload, policy=body.policy, budget=body.budget_usdc, chain_id=settings.chain_id, contract=settings.escrow_address)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if session.get(Job, body.job_id) is not None:
                raise HTTPException(status_code=409, detail="job ID is already in use")
            for term in body.workload["provider_terms"]:
                provider = session.get(Provider, term["provider_id"])
                wallet = session.get(Wallet, provider.wallet_id) if provider else None
                if not wallet or wallet.address.lower() != term["wallet"].lower() or Decimal(provider.price_per_step) != Decimal(str(term["price_per_step"])):
                    raise HTTPException(status_code=422, detail="provider registration differs from signed terms")
        elif "provider_terms" in body.workload:
            raise HTTPException(status_code=422, detail="signed provider terms require a client job binding")
        elif settlement is not None and settings.chain_id != 31337:
            raise HTTPException(status_code=422, detail="public-chain jobs require signed provider quotes and a client binding")
        job = job_service.create_job(
            session, spec=body.workload, policy=body.policy, budget=body.budget_usdc, deadline_seconds=body.deadline_seconds,
            reserved_id=body.job_id, client_address=body.client_address,
        )
        if body.job_binding is not None:
            session.add(JobBinding(job_id=job.id, client_address=body.client_address.lower(), document=body.job_binding))
        response = {
            "job_id": job.id,
            "onchain_id": job.onchain_id,
            "state": job.state,
            "spec_hash": job.spec_hash,
            "policy_hash": job.policy_hash,
            "next": {"action": "createJob"},
        }
        if idempotency_key:
            session.add(IdempotencyRecord(key=idempotency_key, request_hash=request_hash, status_code=201, response=response))
        session.commit()
        return response

    @app.get("/jobs")
    def list_jobs(limit: int = 20, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
        """Newest first, so the dashboard can follow whichever job was started last."""
        rows = session.scalars(select(Job).order_by(Job.created_at.desc()).limit(min(limit, 100)))
        return [{"job_id": j.id, "state": j.state, "total_steps": j.spec.get("total_logical_steps"), "budget_usdc": str(j.budget), "created_at": j.created_at.isoformat()} for j in rows]

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
        return job_view(load_job(session, job_id))

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
        job = load_job(session, job_id)
        if job.state != JobState.CANCELLED.value:  # Cancelling twice is a no-op, not an error.
            job_service.transition(session, job, JobState.CANCELLED, reason="client cancelled", actor="client")
            event = record_event(session, app.state.bus, "job.state_changed", job.id, {"from": "OPEN", "to": "CANCELLED", "reason": "client cancelled"})
            session.commit()
            app.state.bus.publish(event)
        return job_view(job)

    @app.post("/jobs/{job_id}/erase-key")
    def erase_job_key(job_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Destroy a finished job's encryption key (the customer deleted their data). Operator API, called by the Platform's
        erasure worker. A job that is still running is refused: destroying its key would break the run; ask again once it has ended."""
        job = load_job(session, job_id)
        if job.state not in job_keys.ENDED_JOB_STATES:
            raise HTTPException(status_code=409, detail="this job has not ended; its key can be destroyed after it has")
        return {"job_id": job.id, "state": job.state, "destroyed": job_keys.destroy(session, job.id)}
    @app.get("/jobs/{job_id}/timeline")
    def timeline(job_id: str, after: int = 0, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
        load_job(session, job_id)
        return events_after(session, job_id, after)

    @app.get("/jobs/{job_id}/model")
    def download_model(job_id: str, session: Session = Depends(get_session)) -> Response:
        """The job's final accepted model as a zip: weights, manifest, workload and a script to try it."""
        job = load_job(session, job_id)
        checkpoint = session.scalars(
            select(Checkpoint).where(Checkpoint.job_id == job_id, Checkpoint.state == "ACCEPTED").order_by(Checkpoint.logical_step.desc()).limit(1)
        ).first()
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="No accepted model yet. It appears once the final checkpoint passes the quality check.")
        entry = (checkpoint.manifest.get("files") or {}).get("trainable.safetensors") or {}  # The GPCK/1 weights file. ekido_core.gpck needs torch, so it isn't imported here.
        digest = entry.get("sha256")
        if not isinstance(digest, str):
            raise HTTPException(status_code=422, detail="the accepted checkpoint lists no trained weights")
        try:
            path = store.path_for(digest)
            if path.is_file():
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise HTTPException(status_code=500, detail="stored weights don't match their hash")
            elif blobs is not None and blobs.exists(digest):
                data = blobs.open(digest).read()
            else:
                raise HTTPException(status_code=404, detail="the model's weights are no longer stored")
        except ArtifactError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        body = model_bundle(job.id, job.spec, checkpoint.manifest, checkpoint.manifest_hash, checkpoint.logical_step, data)
        name = f"ekido-model-{job.id[:8]}.zip"
        return Response(body, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/jobs/{job_id}/checkpoints")
    def checkpoints(job_id: str, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
        load_job(session, job_id)
        rows = session.scalars(select(Checkpoint).where(Checkpoint.job_id == job_id).order_by(Checkpoint.logical_step))
        return [
            {"checkpoint_id": c.id, "manifest_hash": c.manifest_hash, "parent_hash": c.parent_hash, "logical_step": c.logical_step, "state": c.state, "size_bytes": c.size_bytes, "signer": c.signer}
            for c in rows
        ]

    @app.get("/jobs/{job_id}/settlement-preview")
    def settlement_preview(job_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Who gets paid what, computed from the accepted lineage alone. Moves no money."""
        job = load_job(session, job_id)
        tip = job_service.latest_accepted_checkpoint(session, job)
        if tip is None:
            return {"state": job.state, "lineage": [], "steps": {}, "payouts": {}, "refund": str(job.budget)}
        chain = job_service.accepted_lineage(session, tip)
        steps = job_service.steps_by_signer(chain)
        providers = job_service.providers_by_signer(session)
        rates = {signer: providers[signer].price_per_step if signer in providers else Decimal(0) for signer in steps}
        terms = fee_terms(job.spec or {})
        budget = Decimal(job.budget)
        payouts, refund = job_service.compute_payouts(steps, rates, provider_budget(budget, terms[0]) if terms else budget)
        fee = Decimal(0)
        if terms:
            fee = fee_amount(sum(payouts.values(), Decimal(0)), terms[0])
            refund = budget - sum(payouts.values(), Decimal(0)) - fee
        return {
            "state": job.state,
            "lineage": [c.logical_step for c in chain],
            "lineage_hashes": [c.manifest_hash for c in chain],
            "steps": {providers[s].name if s in providers else s: n for s, n in steps.items()},
            "payouts": {providers[s].name if s in providers else s: str(a) for s, a in payouts.items()},
            "refund": str(refund),
            "protocol_fee": {"bps": terms[0], "recipient": terms[1], "amount": str(fee)} if terms else None,
        }

    @app.get("/jobs/{job_id}/recoveries")
    def recoveries(job_id: str, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
        load_job(session, job_id)
        rows = session.scalars(select(RecoveryEvent).where(RecoveryEvent.job_id == job_id).order_by(RecoveryEvent.attempt))
        return [{"attempt": r.attempt, "kind": r.kind, "cause": r.cause, "injected": r.injected, "lost_steps": r.lost_steps} for r in rows]

    @app.get("/fees")
    def fees() -> dict[str, Any]:
        """The fee every new job here must carry in its spec. Public, so clients see it before they sign or fund."""
        return {"protocol_fee": {"bps": settings.protocol_fee_bps, "recipient": settings.protocol_fee_recipient} if settings.protocol_fee_bps else None}

    @app.get("/providers")
    def providers(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
        rows = session.scalars(select(Provider).order_by(Provider.name))
        result = []
        for p in rows:
            gpu = session.get(GpuHardware, p.physical_gpu_id) if p.physical_gpu_id else None
            address = session.get(Wallet, p.wallet_id).address
            try:
                # What the provider should have committed in ProviderRegistry; anyone can compare the two.
                meta_hash = metadata_hash(provider_metadata(p.name, address, p.price_per_step, p.capacity_mib))
            except (ValueError, ArithmeticError):
                meta_hash = None
            result.append(
                {
                    "provider_id": p.id,
                    "name": p.name,
                    # Public anyway once paid, and needed to re-check a payout from the steps each key signed.
                    "wallet_address": address,
                    "metadata_hash": meta_hash,
                    "enrolled_key": p.enrolled_key,
                    "signing_keys": sorted(s.key for s in session.scalars(select(ProviderKey).where(ProviderKey.provider_id == p.id))),
                    "kind": p.kind,
                    "status": p.status,
                    "capacity_mib": p.capacity_mib,
                    "price_per_step": str(p.price_per_step),
                    # Read by the provider's own agent: observed, not independently verified.
                    "gpu_model": gpu.model if gpu else None,
                    "gpu_vram_mib": gpu.vram_mib if gpu else None,
                    "compute_cap": gpu.compute_cap if gpu else None,
                    "hardware_evidence": gpu.evidence if gpu else None,
                    # Profiles on one GPU share this, so a dashboard can show they're the same hardware.
                    "physical_gpu_id": p.physical_gpu_id,
                }
            )
        return result

    @app.post("/providers", status_code=201)
    def register_provider(body: RegisterProviderRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        address = body.wallet_address.lower()
        wallet = session.scalar(select(Wallet).where(Wallet.address == address))
        if wallet is None:
            wallet = Wallet(address=address, role="provider")
            session.add(wallet)
            session.flush()
        provider = Provider(wallet_id=wallet.id, name=body.name, kind=body.kind, capacity_mib=body.capacity_mib, price_per_step=body.price_per_step)
        session.add(provider)
        session.commit()
        return {"provider_id": provider.id, "name": provider.name, "kind": provider.kind}

    join_times: dict[str, list[float]] = {}
    joined_by_ip: dict[str, int] = {}

    @app.post("/providers/join")
    def join_provider(body: JoinProviderRequest, request: Request, session: Session = Depends(get_session)) -> JSONResponse:
        """Self-service sign-up. The wallet's own signature is the proof, so this works from any machine.

        The provider is untrusted, like every provider: its work only earns once verifiers accept it.
        """
        client = caller_ip(request)
        now = time.monotonic()
        if len(join_times) > 10_000:  # Bounded memory: forget callers quiet for a minute.
            for ip in [ip for ip, ts in join_times.items() if not ts or now - ts[-1] >= 60]:
                del join_times[ip]
        recent = [t for t in join_times.get(client, []) if now - t < 60]
        if len(recent) >= JOIN_PER_MINUTE:
            return JSONResponse(status_code=429, content={"error": "TOO_MANY_JOINS", "detail": "Try again in a minute."})
        join_times[client] = [*recent, now]
        doc = body.document
        problems = join_problems(doc)
        if problems:
            return JSONResponse(status_code=422, content={"error": "INVALID_JOIN", "problems": problems})
        if not verify_document(doc, body.signature, doc["wallet"]):
            return JSONResponse(status_code=401, content={"error": "BAD_SIGNATURE", "detail": "The signature isn't from this wallet."})
        address = doc["wallet"].lower()
        wallet = session.scalar(select(Wallet).where(Wallet.address == address))
        if wallet is not None:
            existing = session.scalar(select(Provider).where(Provider.wallet_id == wallet.id))
            if existing is not None:
                return JSONResponse(status_code=200, content={"provider_id": existing.id, "name": existing.name, "created": False})
        if session.scalar(select(func.count()).select_from(Provider)) >= JOIN_MAX_PROVIDERS:
            return JSONResponse(status_code=503, content={"error": "NETWORK_FULL", "detail": "No more providers can join right now."})
        if joined_by_ip.get(client, 0) >= JOIN_PER_IP_TOTAL:
            return JSONResponse(status_code=429, content={"error": "TOO_MANY_PROVIDERS", "detail": f"One address can add at most {JOIN_PER_IP_TOTAL} providers."})
        clean = doc["name"].strip()
        if session.scalar(select(Provider).where(func.lower(Provider.name) == clean.lower())) is not None:
            return JSONResponse(status_code=409, content={"error": "NAME_TAKEN", "detail": "Another provider already uses that name. Pick a different one."})
        if wallet is None:
            wallet = Wallet(address=address, role="provider")
            session.add(wallet)
            session.flush()
        joined_by_ip[client] = joined_by_ip.get(client, 0) + 1
        provider = Provider(wallet_id=wallet.id, name=clean, kind="REAL", capacity_mib=doc.get("capacity_mib"), price_per_step=Decimal(str(doc["price_per_step"])))
        session.add(provider)
        session.commit()
        return JSONResponse(status_code=201, content={"provider_id": provider.id, "name": provider.name, "created": True})

    @app.post("/providers/qualify")
    def qualify(body: QualifyRequest) -> Any:
        """Measures each provider against a workload ahead of time, so recovery needs only one fresh probe."""
        problems = spec_problems(body.workload)
        if problems:
            return JSONResponse(status_code=422, content={"error": "INVALID_JOB_SPEC", "problems": problems})
        results = gateway.orchestrator.qualify_providers(body.workload)
        # A failed probe's reason is text from the provider's machine. It stays in the log, not in the response.
        for result in results:
            if result.get("code") == "PROBE_ERROR":
                logging.getLogger(__name__).warning("probe failed on provider %s: %s", result.get("provider_id"), result.get("reason"))
                result["reason"] = "The probe failed on the provider"
        return results

    @app.post("/jobs/{job_id}/activate")
    def activate(job_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Runs placement, including real probes, so it can take a while. Calling it twice changes nothing."""
        job = load_job(session, job_id)
        if job.state == JobState.OPEN.value and settlement is None and settings.chain_id != 31337 and (settings.rpc_url or settings.escrow_address or "provider_terms" in job.spec):
            raise HTTPException(status_code=409, detail="settlement verification is not configured; cannot start a public-chain job")
        if settlement is not None and job.state == JobState.OPEN.value:
            # No funds, no work: a provider must never run a job nobody can pay for.
            try:
                settlement.require_funded(job)
            except Exception as exc:
                raise HTTPException(status_code=409, detail=f"not funded: {exc}") from exc
        state = gateway.orchestrator.start_job(job_id)
        return {"job_id": job_id, "state": state}

    @app.put("/artifacts/{digest}", status_code=201)
    async def put_artifact(digest: str, request: Request, owner: str = Depends(require_artifact_token)) -> dict[str, Any]:
        try:
            already = store.exists(digest)
            if not already:
                # The declared length is checked up front; without one, assume the largest blob the store accepts.
                declared = request.headers.get("content-length")
                upload_budget.check(owner, int(declared) if declared and declared.isdigit() else store.max_bytes)
            size = await store.put(digest, request.stream())
        except UploadBudgetExceeded as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ArtifactError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not already:
            upload_budget.record(owner, digest, size)
        return {"sha256": digest, "bytes": size}

    @app.put("/datasets/{digest}", status_code=201)
    async def put_dataset(digest: str, request: Request) -> dict[str, Any]:
        """A customer's training rows (dataset.jsonl), handed over by the Platform when a paid job starts. Operator API only
        (ApiGuard): agents fetch it by hash through /artifacts like any blob. The store re-checks the hash before keeping it."""
        declared = request.headers.get("content-length")
        if declared is None or not declared.isdigit() or int(declared) > MAX_DATASET_BYTES:
            raise HTTPException(status_code=413, detail=f"a dataset must declare its size and be at most {MAX_DATASET_BYTES} bytes")
        try:
            size = await store.put(digest, request.stream())
        except ArtifactError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"sha256": digest, "bytes": size}

    @app.get("/artifacts/{digest}")
    def get_artifact(digest: str, _: str = Depends(require_artifact_token)) -> Any:
        try:
            path = store.path_for(digest)
            if not path.is_file() and blobs is not None and blobs.exists(digest):
                # An older worker downloading a blob that a newer one uploaded straight to S3.
                body = blobs.open(digest)
                return StreamingResponse(body.iter_chunks(64 * 1024), media_type="application/octet-stream")
        except ArtifactError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(path, media_type="application/octet-stream")

    @app.post("/artifacts/{digest}/upload-url")
    def artifact_upload_url(digest: str, body: UploadUrlRequest, owner: str = Depends(require_artifact_token)) -> dict[str, Any]:
        """A presigned S3 link for one blob. 404 when direct uploads are off: the worker then uploads through PUT."""
        if blobs is None:
            raise HTTPException(status_code=404, detail="direct uploads are not enabled")
        try:
            if store.exists(digest):
                os.utime(store.path_for(digest))  # Keep it safe from retention until its checkpoint commits.
                return {"exists": True}
            if blobs.exists(digest):
                blobs.touch(digest)
                return {"exists": True}
            upload_budget.check(owner, body.bytes)
            offer = blobs.upload_url(digest, body.bytes)
        except UploadBudgetExceeded as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ArtifactError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # Counted when the link is issued: the coordinator never sees the upload itself.
        upload_budget.record(owner, digest, body.bytes)
        return offer

    @app.get("/artifacts/{digest}/download-url")
    def artifact_download_url(digest: str, _: str = Depends(require_artifact_token)) -> dict[str, Any]:
        """A presigned S3 link when the blob is in the bucket. 404 otherwise: the worker then downloads through GET."""
        try:
            if blobs is None or store.exists(digest) or not blobs.exists(digest):
                raise HTTPException(status_code=404, detail="no direct link for this artifact")
            return {"url": blobs.download_url(digest), "expires_in": blobs.url_ttl}
        except ArtifactError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.websocket("/agent/ws")
    async def agent_socket(websocket: WebSocket):
        await gateway.handle(websocket)

    def verifiers_only(address: str) -> None:
        if not isinstance(quality, QuorumQuality) or address.lower() not in quality.verifiers:
            raise HTTPException(status_code=403, detail="not an allowlisted verifier")

    def require_verifier_session(address: str, authorization: str | None) -> None:
        verifiers_only(address)
        scheme, _, token = (authorization or "").partition(" ")
        owner = gateway.upload_tokens.get(token) if scheme.lower() == "bearer" else None
        if owner is None:
            raise HTTPException(status_code=401, detail="a signed verifier session is required")
        if owner != f"verifier:{address.lower()}":
            raise HTTPException(status_code=403, detail="session belongs to a different identity")

    @app.post("/verifier/session")
    def verifier_session(body: VerifierSessionRequest) -> dict[str, Any]:
        """A signed, recent message proves the verifier holds its key. The token only allows reading artifacts."""
        address = str(body.message.get("address", ""))
        verifiers_only(address)
        issued = body.message.get("ts")
        if body.message.get("kind") != "gpuchain.verifier.session.v1" or not isinstance(issued, int) or abs(time.time() - issued) > 300:
            raise HTTPException(status_code=401, detail="stale or malformed session request")
        if not verify_document(body.message, body.signature, address):
            raise HTTPException(status_code=401, detail="signature doesn't verify")
        token = secrets.token_urlsafe(32)
        gateway.upload_tokens[token] = f"verifier:{address.lower()}"
        return {"token": token}

    @app.post("/verifier/tasks/claim")
    def claim_task(body: ClaimTaskRequest, session: Session = Depends(get_session), authorization: str | None = Header(default=None)) -> dict[str, Any] | None:
        require_verifier_session(body.address, authorization)
        task = quality.claim(body.address)
        if task is None:
            return None
        job = load_job(session, task["job_id"])
        candidate = session.scalar(select(Checkpoint).where(Checkpoint.job_id == job.id, Checkpoint.manifest_hash == task["checkpoint_hash"]))
        reference = None
        if task["reference_hash"]:
            reference = session.scalar(select(Checkpoint).where(Checkpoint.job_id == job.id, Checkpoint.manifest_hash == task["reference_hash"]))
        return {
            **task,
            "spec": job.spec,
            "policy": job.policy,
            "manifest": candidate.manifest,
            "reference_manifest": reference.manifest if reference else None,
            "expected_signer": candidate.signer,
        }

    @app.post("/verifier/tasks/job-key")
    def verifier_job_key(body: VerifierJobKeyRequest, session: Session = Depends(get_session), authorization: str | None = Header(default=None)) -> dict[str, Any]:
        """The job's key for a verifier that holds a claimed evaluation of this job, encrypted to the recipient it names. Needs a signed
        verifier session; being on the allowlist is not enough (a verifier gets the key only for work it was handed)."""
        require_verifier_session(body.address, authorization)
        if not isinstance(quality, QuorumQuality) or not quality.is_working_on(body.address, body.job_id):
            raise HTTPException(status_code=403, detail="NOT_ASSIGNED: this verifier has no claimed evaluation of this job")
        try:
            sealed = job_keys.release_to_verifier(session, job_id=body.job_id, verifier_recipient=body.recipient)
        except job_keys.KeyRefused as refusal:
            raise HTTPException(status_code=409, detail=f"KEY_REFUSED: {refusal.code}") from None
        return {"ciphertext_b64": base64.b64encode(sealed).decode("ascii")}
    @app.post("/verifier/verdicts/claim")
    def claim_verdict(body: ClaimTaskRequest, authorization: str | None = Header(default=None)) -> dict[str, Any] | None:
        require_verifier_session(body.address, authorization)
        return settlement.claim(body.address) if settlement is not None else None

    @app.post("/verifier/verdicts", status_code=202)
    def submit_verdict_signature(body: SubmitVerdictRequest) -> dict[str, Any]:
        if settlement is None:
            raise HTTPException(status_code=403, detail="settlement isn't configured")
        try:
            return settlement.submit(body.job_id, body.signature)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/jobs/{job_id}/settlement")
    def settlement_status(job_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
        load_job(session, job_id)
        row = session.scalar(select(Settlement).where(Settlement.job_id == job_id))
        if row is None:
            return {"status": "NOT_STARTED"}
        return {"status": row.status, "verdict": row.verdict, "submit_tx": row.submit_tx, "finalize_tx": row.finalize_tx}

    @app.get("/settlements")
    def list_settlements(limit: int = 50, before_job_id: str | None = None, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
        """One page, newest first. The last job ID can be used as a stable cursor."""
        query = select(Settlement)
        if before_job_id is not None:
            cursor = session.scalar(select(Settlement).where(Settlement.job_id == before_job_id))
            if cursor is None:
                raise HTTPException(status_code=404, detail="settlement cursor not found")
            query = query.where(or_(Settlement.created_at < cursor.created_at, and_(Settlement.created_at == cursor.created_at, Settlement.job_id < cursor.job_id)))
        rows = session.scalars(query.order_by(Settlement.created_at.desc(), Settlement.job_id.desc()).limit(min(max(limit, 1), 200)))
        return [
            {"job_id": r.job_id, "status": r.status, "created_at": r.created_at.isoformat() if r.created_at else None, "verdict": r.verdict, "submit_tx": r.submit_tx, "finalize_tx": r.finalize_tx}
            for r in rows
        ]

    @app.post("/verifier/evaluations", status_code=202)
    def submit_evaluation(body: SubmitEvaluationRequest) -> dict[str, Any]:
        if not isinstance(quality, QuorumQuality):
            raise HTTPException(status_code=403, detail="verifiers aren't configured")
        try:
            decision = quality.submit(body.task_id, body.report)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"decision": decision}

    @app.post("/demo/faults", status_code=202)
    def inject_fault(body: FaultRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Disabled unless DEMO_MODE is set. Every fault is recorded and labelled as injected."""
        if not settings.demo_mode:
            raise HTTPException(status_code=403, detail="fault injection is disabled (DEMO_MODE=0)")
        load_job(session, body.job_id)
        fault = FaultInjection(job_id=body.job_id, kind=body.kind, requested_by="demo")
        session.add(fault)
        session.flush()
        target = gateway.provider_of_job(body.job_id)
        event = record_event(session, app.state.bus, "fault.injected", body.job_id, {"fault_kind": body.kind, "target_provider_id": target, "label": "INJECTED"})
        session.commit()
        app.state.bus.publish(event)
        applied = False
        if target is not None:
            try:
                gateway.inject_fault(target, body.job_id, body.kind)
                applied = True
                fault.applied_at = datetime.now(timezone.utc)
                session.commit()
            except (ConnectionError, TimeoutError) as exc:
                fault.outcome = f"not delivered: {exc}"
                session.commit()
        return {"fault_id": fault.id, "delivered": applied, "label": "INJECTED"}

    @app.websocket("/ws/jobs/{job_id}")
    async def job_stream(websocket: WebSocket, job_id: str, after: int = 0):
        await websocket.accept()
        # Subscribe before catching up, so nothing published in between is missed. Clients de-duplicate by id.
        queue = app.state.bus.subscribe(job_id)
        try:
            with Session(engine) as session:
                backlog = events_after(session, job_id, after)
            for event in backlog:
                await websocket.send_json(event)
            while True:
                try:
                    await websocket.send_json(await asyncio.wait_for(queue.get(), timeout=15))
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
        except WebSocketDisconnect:
            pass
        finally:
            app.state.bus.unsubscribe(job_id, queue)

    return app
