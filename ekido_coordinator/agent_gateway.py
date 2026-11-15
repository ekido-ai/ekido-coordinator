"""The agents' connection to the coordinator. Providers are untrusted, so every message is authenticated.

Identity chain: the provider's wallet signs a delegation to a worker key, and the worker key signs
every message. Sequence numbers stop replays. Agents dial out, so providers behind NAT work.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.engine import Engine
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core.compiler import ProbeResult
from ekido_core.signing import recover_document_signer, verify_document

from . import jobs as job_service
from .jobs import LineageError, StaleFencingToken, StorageQuotaExceeded
from .models import GpuHardware, Job, JobAssignment, Provider, Wallet
from .orchestrator import Orchestrator

log = logging.getLogger(__name__)


def native_bf16(compute_cap: str | None) -> bool | None:
    """BF16 is native from Ampere (8.0). The T4 only emulates it."""
    try:
        return int(str(compute_cap).split(".")[0]) >= 8
    except ValueError:
        return None

HEARTBEAT_INTERVAL = 2.0
SUSPECT_AFTER_MISSES = 3
FAILED_AFTER_MISSES = 5
PROBE_TIMEOUT = 180.0
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


def enrollment_document(provider_id: str, worker_key: str) -> dict:
    return {"kind": "gpuchain.enrollment.v1", "provider_id": provider_id, "worker_key": worker_key.lower()}


@dataclass
class Connection:
    provider_id: str
    worker_key: str
    websocket: WebSocket
    token: str
    last_seen: float = field(default_factory=time.monotonic)
    last_seq: int = -1
    send_seq: int = 0
    suspect: bool = False
    # What's running there, so a lost connection can be turned into a failure for the right job.
    job_id: str | None = None
    fencing_token: int | None = None
    last_step: int | None = None
    attempt: int | None = None


class AgentGateway:
    """Also implements the orchestrator's AgentLink. Those calls block, and run on worker threads."""

    def __init__(self, engine: Engine, orchestrator_factory):
        self.engine = engine
        self.connections: dict[str, Connection] = {}
        self.upload_tokens: dict[str, str] = {}
        self.upload_budget = None  # An artifacts.UploadBudget, set by the app. A commit clears the provider's pending bytes.
        self.loop: asyncio.AbstractEventLoop | None = None
        self.orchestrator: Orchestrator = orchestrator_factory(self)
        self._pending_probes: dict[str, concurrent.futures.Future] = {}

    # ------------------------------------------------- AgentLink (blocking)

    def is_connected(self, provider_id: str) -> bool:
        return provider_id in self.connections

    def _send_blocking(self, provider_id: str, message_type: str, payload: dict, **fields) -> None:
        connection = self.connections.get(provider_id)
        if connection is None or self.loop is None:
            raise ConnectionError(f"provider {provider_id} isn't connected")
        future = asyncio.run_coroutine_threadsafe(self._send(connection, message_type, payload, **fields), self.loop)
        future.result(timeout=15)

    def probe(self, provider_id: str, config: dict) -> ProbeResult:
        request_id = secrets.token_hex(8)
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._pending_probes[request_id] = future
        try:
            self._send_blocking(provider_id, "cmd.probe", {"request_id": request_id, "config": config})
            result = future.result(timeout=PROBE_TIMEOUT)
        except concurrent.futures.TimeoutError:
            return ProbeResult(ok=False, error="Probe didn't complete: timed out")
        except ConnectionError as exc:
            # The exception's text stays in the log. The result, which an API response can carry, gets a fixed message.
            log.warning("probe on provider %s lost its connection: %s", provider_id, exc)
            return ProbeResult(ok=False, error="Probe didn't complete: the provider disconnected")
        finally:
            self._pending_probes.pop(request_id, None)
        return ProbeResult(
            ok=bool(result.get("ok")),
            oom=bool(result.get("oom")),
            peak_mem_mib=result.get("peak_mem_mib"),
            steps_per_s=result.get("steps_per_s"),
            duration_s=result.get("duration_s"),
            error=result.get("error"),
        )

    def start_run(self, provider_id: str, config: dict) -> None:
        connection = self.connections.get(provider_id)
        if connection is None:
            raise ConnectionError(f"provider {provider_id} isn't connected")
        # Only one worker runs a job. Anyone still marked as running it has been replaced.
        for other in self.connections.values():
            if other is not connection and other.job_id == config["job_id"]:
                other.job_id = None
        connection.job_id, connection.fencing_token, connection.last_step, connection.attempt = config["job_id"], config["fencing_token"], None, config.get("attempt")
        self._send_blocking(provider_id, "cmd.start_run", {"config": config}, job_id=config["job_id"], fencing_token=config["fencing_token"])

    def stop_run(self, provider_id: str, job_id: str, fencing_token: int) -> None:
        connection = self.connections.get(provider_id)
        # A stopped worker no longer runs this job, even though it won't report an exit.
        if connection is not None and connection.job_id == job_id and connection.fencing_token == fencing_token:
            connection.job_id = None
        self._send_blocking(provider_id, "cmd.stop_run", {}, job_id=job_id, fencing_token=fencing_token)

    def inject_fault(self, provider_id: str, job_id: str, kind: str) -> None:
        self._send_blocking(provider_id, "cmd.inject_fault", {"kind": kind}, job_id=job_id)

    def provider_of_job(self, job_id: str) -> str | None:
        """The worker running this job now. The newest fencing token wins, should two ever claim it."""
        current = [(c.fencing_token or 0, provider_id) for provider_id, c in self.connections.items() if c.job_id == job_id]
        return max(current)[1] if current else None

    # ----------------------------------------------------------- connection

    async def _send(self, connection: Connection, message_type: str, payload: dict, **fields) -> None:
        connection.send_seq += 1
        await connection.websocket.send_json({"v": 1, "seq": connection.send_seq, "ts": time.time(), "type": message_type, "payload": payload, **fields})

    def _authenticate_hello(self, hello: dict) -> tuple[str, str] | None:
        """Checks the worker's own signature and the provider wallet's delegation to it."""
        payload = hello.get("payload") or {}
        provider_id, worker_key, enrollment = payload.get("provider_id"), payload.get("worker_key"), payload.get("enrollment_sig")
        if not (isinstance(provider_id, str) and isinstance(worker_key, str) and isinstance(enrollment, str)):
            return None
        if not verify_document(hello, hello.get("sig") or "", worker_key):
            return None
        signer = recover_document_signer(enrollment_document(provider_id, worker_key), enrollment)
        if signer is None:
            return None
        with Session(self.engine) as session:
            provider = session.get(Provider, provider_id)
            if provider is None:
                return None
            wallet = session.get(Wallet, provider.wallet_id)
            if wallet is None or wallet.address.lower() != signer.lower():
                return None
            if not job_service.enroll_worker_key(session, provider, worker_key, enrollment):
                return None
            provider.status = "ONLINE"
            hardware = payload.get("hardware")
            if provider.physical_gpu_id is None and isinstance(hardware, dict):
                # What the provider's own agent read. Recorded as observed, never as verified.
                gpu = GpuHardware(
                    uuid_hash=hashlib.sha256(str(hardware.get("uuid")).encode()).hexdigest(),
                    model=str(hardware.get("name"))[:120],
                    vram_mib=int(hardware.get("memory_total_mib") or 0),
                    compute_cap=hardware.get("compute_cap"),
                    driver=hardware.get("driver"),
                    native_bf16=native_bf16(hardware.get("compute_cap")),
                    evidence="OBSERVED",
                )
                session.add(gpu)
                session.flush()
                provider.physical_gpu_id = gpu.id
            session.commit()
        return provider_id, worker_key.lower()

    async def handle(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.loop = asyncio.get_running_loop()
        connection: Connection | None = None
        try:
            hello = await asyncio.wait_for(websocket.receive_json(), timeout=15)
            identity = self._authenticate_hello(hello) if hello.get("type") == "agent.hello" else None
            if identity is None:
                await websocket.send_json({"type": "error", "payload": {"error": "AUTH_FAILED"}})
                await websocket.close(code=4401)
                return
            provider_id, worker_key = identity
            previous = self.connections.get(provider_id)
            token = secrets.token_urlsafe(32)
            connection = Connection(provider_id, worker_key, websocket, token, last_seq=hello.get("seq", 0))
            if previous is not None:
                # A reconnecting agent keeps the record of what it was running.
                connection.job_id, connection.fencing_token, connection.last_step, connection.attempt = previous.job_id, previous.fencing_token, previous.last_step, previous.attempt
                self.upload_tokens.pop(previous.token, None)
            self.connections[provider_id] = connection
            self.upload_tokens[token] = provider_id
            self.orchestrator.forget_provider(provider_id)
            await self._send(connection, "agent.welcome", {"upload_token": token, "heartbeat_interval": HEARTBEAT_INTERVAL})

            while True:
                message = await websocket.receive_json()
                await self._on_message(connection, message)
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError, ValueError):
            pass  # The connection ended or sent unusable data; the finally block below marks the provider offline.
        finally:
            # A newer connection may already have replaced this one. Don't remove that.
            if connection is not None and self.connections.get(connection.provider_id) is connection:
                del self.connections[connection.provider_id]
                self.upload_tokens.pop(connection.token, None)
                with Session(self.engine) as session:
                    provider = session.get(Provider, connection.provider_id)
                    if provider is not None:
                        provider.status = "OFFLINE"
                        session.commit()
                self._report_lost(connection, "DISCONNECT")

    def _report_lost(self, connection: Connection, cause: str) -> None:
        if connection.job_id is None:
            return
        job_id, token, step, provider = connection.job_id, connection.fencing_token, connection.last_step, connection.provider_id
        connection.job_id = None
        asyncio.get_running_loop().run_in_executor(
            None, lambda: self.orchestrator.on_failure(job_id, provider_id=provider, cause=cause, last_step=step, fencing_token=token)
        )

    async def monitor(self) -> None:
        """Turns silence into a failure, even if the socket looks open, as with a stalled or partitioned agent."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL / 2)
            self.check_heartbeats(time.monotonic())

    def check_heartbeats(self, now: float) -> None:
        for connection in list(self.connections.values()):
            silence = now - connection.last_seen
            connection.suspect = silence > HEARTBEAT_INTERVAL * SUSPECT_AFTER_MISSES
            if silence > HEARTBEAT_INTERVAL * FAILED_AFTER_MISSES and connection.job_id is not None:
                self._report_lost(connection, "HEARTBEAT_LOST")

    async def _on_message(self, connection: Connection, message: dict) -> None:
        seq = message.get("seq")
        if not isinstance(seq, int) or seq <= connection.last_seq:
            return  # A duplicate or a replay. Dropped without acting on it.
        if not verify_document(message, message.get("sig") or "", connection.worker_key):
            await self._send(connection, "error", {"error": "BAD_SIGNATURE", "seq": seq})
            return
        connection.last_seq = seq
        connection.last_seen = time.monotonic()
        kind, payload = message.get("type"), message.get("payload") or {}

        if kind == "probe.result":
            future = self._pending_probes.get(payload.get("request_id"))
            if future is not None and not future.done():
                future.set_result(payload.get("result") or {})
            return
        if kind == "agent.heartbeat":
            return
        if kind == "agent.telemetry":
            self._forward_telemetry(connection, payload)
            return

        # Everything below concerns a job, so it runs off the event loop and through fencing.
        job_id, token = message.get("job_id"), message.get("fencing_token")
        if not job_id or token is None:
            return
        loop = asyncio.get_running_loop()
        try:
            if not self._is_current(job_id, token):
                raise StaleFencingToken(token, -1)
            with Session(self.engine) as session:
                assignment = session.scalar(select(JobAssignment).where(JobAssignment.job_id == job_id, JobAssignment.fencing_token == token))
                if assignment is None or assignment.provider_id != connection.provider_id:
                    await self._send(connection, "error", {"error": "NOT_ASSIGNED", "detail": "this job attempt belongs to another provider"})
                    return
            if kind == "run.started":
                await loop.run_in_executor(None, self.orchestrator.on_run_started, job_id, token)
            elif kind == "run.progress":
                self._forward_progress(connection, job_id, token, payload)
            elif kind == "run.warning":
                if not self._is_current(job_id, token):
                    raise StaleFencingToken(token, -1)
                self.orchestrator.emit_live(
                    "failure.warning", job_id,
                    {"provider_id": connection.provider_id, "rule": payload.get("rule"), "detail": payload.get("detail"), "injected": bool(payload.get("injected"))},
                )
            elif kind == "checkpoint.uploaded":
                await self._on_checkpoint(connection, loop, job_id, token, payload)
            elif kind == "run.exited":
                await self._on_exit(connection, loop, job_id, token, payload)
        except StaleFencingToken as exc:
            # This worker has been replaced, and must stop writing and kill its runtime.
            await self._send(connection, "cmd.stop_run", {"reason": "STALE_FENCING_TOKEN", "current": exc.current}, job_id=job_id, fencing_token=token)
        except LineageError as exc:
            await self._send(connection, "error", {"error": "INVALID_LINEAGE", "detail": str(exc)})
        except StorageQuotaExceeded as exc:
            await self._send(connection, "error", {"error": "STORAGE_QUOTA", "detail": str(exc)})

    def _is_current(self, job_id: str, token: int) -> bool:
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            return job is not None and job.fencing_token == token

    def _forward_progress(self, connection: Connection, job_id: str, token: int, payload: dict) -> None:
        if not self._is_current(job_id, token):
            raise StaleFencingToken(token, -1)
        connection.last_step = payload.get("step", connection.last_step)
        self.orchestrator.emit_live(
            "training.progress",
            job_id,
            {
                "provider_id": connection.provider_id,
                "attempt": connection.attempt,
                "step": payload.get("step"),
                "total_steps": payload.get("total_steps"),
                "loss": payload.get("loss"),
                "steps_per_s": payload.get("steps_per_s"),
                "peak_mem_mib": payload.get("peak_mem_mib"),
            },
        )

    def _forward_telemetry(self, connection: Connection, payload: dict) -> None:
        if connection.job_id:
            data = {k: payload.get(k) for k in ("util_pct", "mem_used_mib", "temp_c", "power_w", "source")}
            self.orchestrator.emit_live("telemetry.sample", connection.job_id, dict(data, provider_id=connection.provider_id))

    async def _on_checkpoint(self, connection: Connection, loop, job_id: str, token: int, payload: dict) -> None:
        manifest, signature = payload.get("manifest"), payload.get("signature")
        if not isinstance(manifest, dict) or not signature:
            return
        # The signature must come from this provider's enrolled key, or the payout could go to anyone.
        if manifest.get("signer", "").lower() != connection.worker_key or not verify_document(manifest, signature, connection.worker_key):
            await self._send(connection, "error", {"error": "BAD_MANIFEST_SIGNATURE"})
            return
        signed = dict(manifest, signature=signature)
        checkpoint = await loop.run_in_executor(None, lambda: self.orchestrator.on_checkpoint(job_id, signed, fencing_token=token, signer=connection.worker_key))
        if self.upload_budget is not None:
            self.upload_budget.release(connection.provider_id)
        await loop.run_in_executor(None, self.orchestrator.check_probation, job_id, checkpoint, token)

    async def _on_exit(self, connection: Connection, loop, job_id: str, token: int, payload: dict) -> None:
        reason = payload.get("reason")
        connection.job_id = None
        if reason == "FINISHED":
            await loop.run_in_executor(None, self.orchestrator.on_run_finished, job_id, token)
            return
        cause = {"OOM": "OOM", "LEASE_LOST": "HEARTBEAT_LOST", "DIVERGED": "DIVERGED", "BAD_CHECKPOINT": "BAD_CHECKPOINT", "START_FAILED": "START_FAILED"}.get(reason, "PROCESS_EXIT")
        provider, step, injected = connection.provider_id, payload.get("step") or connection.last_step, bool(payload.get("injected"))
        await loop.run_in_executor(
            None, lambda: self.orchestrator.on_failure(job_id, provider_id=provider, cause=cause, injected=injected, last_step=step, fencing_token=token)
        )
