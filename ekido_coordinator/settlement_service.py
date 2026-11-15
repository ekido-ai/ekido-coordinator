"""Turns a finished job into money movement on-chain.

The coordinator proposes a Verdict from the accepted lineage, but can't move funds alone: verifiers
re-derive it, sign it, and only a quorum of those signatures satisfies the contract.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ekido_core.protocol_fee import add_fee, fee_terms, provider_budget
from ekido_core.job_states import JobState
from ekido_core.settlement import build_verdict
from ekido_core.signing import order_signatures, recover_verdict_signer

from . import jobs as job_service
from .chain import ChainError, EscrowClient
from .models import Job, JobBinding, ProviderKeyProof, Settlement, SettlementProposal, Wallet

DEADLINE_TOLERANCE_SECONDS = 60  # For clock differences between the client, this server and the chain.
MIN_PUBLIC_DISPUTE_WINDOW_SECONDS = 3600

log = logging.getLogger(__name__)

VERDICT_LIFETIME_SECONDS = 3600
# A verdict with less time left than this is rebuilt and re-signed rather than submitted (Arvind, 7 Nov).
MIN_SUBMIT_LEAD_SECONDS = 600
VERDICT_CLAIM_LEASE_SECONDS = 30
RESTORE_RETRY_SECONDS = 30
RECOVERY_ANCHOR_KIND = 1


class NotFunded(Exception):
    """No matching escrow exists on-chain, so no work may start."""


@dataclass
class VerdictTask:
    job_id: str
    inputs: dict
    verdict: dict
    signatures: dict[str, str] = field(default_factory=dict)
    claimed_by: dict[str, float] = field(default_factory=dict)
    submitted: bool = False


class SettlementService:
    def __init__(self, engine: Engine, orchestrator, chain: EscrowClient | None, chain_id: int, contract: str, verifiers: set[str], quorum: int, quality, *, explorer_url: str | None = None, local_chain: bool = False):
        self.engine = engine
        self.orchestrator = orchestrator
        self.chain = chain
        self.chain_id = chain_id
        self.contract = contract
        self.verifiers = {v.lower() for v in verifiers}
        self.quorum = quorum
        self.quality = quality
        self.explorer_url = explorer_url
        self.local_chain = local_chain
        self.tasks: dict[str, VerdictTask] = {}
        self.lock = threading.Lock()
        self._propose_lock = threading.Lock()
        self._restore_lock = threading.Lock()
        self._restore_retry_after = 0.0
        self._finalizing: set[str] = set()  # Jobs whose dispute window a thread is waiting out.

    # ------------------------------------------------------------ on-chain

    def _emit_tx(self, job_id: str, result) -> None:
        link = f"{self.explorer_url.rstrip('/')}/tx/{result.tx_hash}" if self.explorer_url and result.tx_hash else None
        self.orchestrator.emit_live(
            "chain.tx",
            job_id,
            {"action": result.action, "tx_hash": result.tx_hash, "status": result.status, "chain_id": self.chain_id, "explorer_url": link, "local_chain": self.local_chain},
        )

    def require_funded(self, job: Job) -> None:
        """The escrow must exist, and must commit to exactly this job's terms and budget."""
        if self.chain is None:
            return
        onchain = self.chain.job(job.onchain_id)
        if onchain["state"] not in ("Funded", "Active"):
            raise NotFunded(f"escrow is in state {onchain['state']}, so call createJob first")
        if onchain["spec_hash"] != job.spec_hash or onchain["policy_hash"] != job.policy_hash:
            raise NotFunded("the escrow commits to different job terms")
        if onchain["budget"] != int(Decimal(job.budget) * 1_000_000):
            raise NotFunded("the escrow holds a different budget")
        window = onchain.get("dispute_window")
        if type(window) is not int or window <= 0:
            raise NotFunded("the escrow's dispute window is unavailable or invalid")
        if window < MIN_PUBLIC_DISPUTE_WINDOW_SECONDS:
            # Short windows are useful only for the explicitly configured local test chain.
            if not (self.local_chain and self.chain_id == 31337 and self.chain.rpc_chain_id() == 31337):
                raise NotFunded("the escrow's dispute window must be at least one hour")
        if "provider_terms" in job.spec:
            with Session(self.engine) as session:
                binding = session.scalar(select(JobBinding).where(JobBinding.job_id == job.id))
                if binding is None or onchain["client"].lower() != binding.client_address.lower():
                    raise NotFunded("escrow client does not match the signed job binding")
        # The client sets the on-chain deadline, and anyone can refund the client an hour after it. A short one
        # would let them take finished work for free, so no provider starts until it covers the agreed time.
        agreed = job.deadline if job.deadline.tzinfo else job.deadline.replace(tzinfo=timezone.utc)
        if onchain["deadline"] < agreed.timestamp() - DEADLINE_TOLERANCE_SECONDS:
            raise NotFunded("the escrow's deadline is earlier than the job's")

    def on_job_started(self, job_id: str) -> None:
        if self.chain is None:
            return
        with Session(self.engine) as session:
            onchain_id = session.get(Job, job_id).onchain_id
        result = self.chain.ensure_active(onchain_id)
        if result.tx_hash:
            self._emit_tx(job_id, result)

    def on_recovered(self, job_id: str, attempt: int, adaptation_hash: str | None) -> None:
        if self.chain is None or adaptation_hash is None:
            return
        with Session(self.engine) as session:
            onchain_id = session.get(Job, job_id).onchain_id
        threading.Thread(target=self._anchor, args=(job_id, onchain_id, attempt, adaptation_hash), daemon=True).start()

    def _anchor(self, job_id: str, onchain_id: str, attempt: int, adaptation_hash: str) -> None:
        try:
            self._emit_tx(job_id, self.chain.anchor(onchain_id, RECOVERY_ANCHOR_KIND, attempt, "0x" + adaptation_hash))
        except Exception:  # A public log only. Settlement doesn't depend on it.
            log.exception("couldn't anchor the recovery of job %s", job_id)

    # ------------------------------------------------------------ proposal

    def propose(self, job_id: str) -> dict:
        """Called when a job ends. Builds the Verdict inputs for the verifiers to check and sign."""
        with self._propose_lock:
            return self._propose(job_id)

    def _propose(self, job_id: str) -> dict:
        with self.lock:
            if job_id in self.tasks:
                return self.tasks[job_id].verdict
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            budget = Decimal(job.budget)
            expiry = int(time.time()) + VERDICT_LIFETIME_SECONDS
            if job.state == JobState.COMPLETED.value:
                tip = job_service.latest_accepted_checkpoint(session, job)
                chain = job_service.accepted_lineage(session, tip)
                reports = self.quality.decided_reports.get((job.id, tip.manifest_hash), []) if self.quality else []
                if not reports:
                    raise RuntimeError(f"no verifier reports for the final checkpoint of {job.id}")
                payouts, refund = self._payouts(session, chain, budget, job.spec)
                inputs = {
                    "onchain_job_id": job.onchain_id,
                    "passed": True,
                    "final_checkpoint_hash": tip.manifest_hash,
                    "lineage": [c.manifest_hash for c in chain],
                    "reports": reports,
                    "payout_evidence": self._payout_evidence(session, chain),
                    "payouts": {address: str(amount) for address, amount in payouts.items()},
                    "refund": str(refund),
                    "budget": str(budget),
                    "expiry": expiry,
                }
                binding = session.scalar(select(JobBinding).where(JobBinding.job_id == job.id))
                if binding is not None:
                    inputs["job_binding"] = binding.document
            elif job.state == JobState.FAILED.value:
                # Nothing was delivered, so the client gets everything back and no one is paid.
                inputs = {
                    "onchain_job_id": job.onchain_id,
                    "passed": False,
                    "final_checkpoint_hash": None,
                    "lineage": [],
                    "reports": [{"verdict": "FAIL", "job_id": job.id, "reason": "JOB_FAILED"}],
                    "payouts": {},
                    "refund": str(budget),
                    "budget": str(budget),
                    "expiry": expiry,
                }
            else:
                raise ValueError(f"job {job_id} hasn't ended (state {job.state})")
            verdict = verdict_from_inputs(inputs)
            task_inputs = dict(inputs, job_id=job_id, chain_id=self.chain_id, contract=self.contract)
            row = session.scalar(select(Settlement).where(Settlement.job_id == job.id))
            if row is None:
                session.add(Settlement(job_id=job.id, verdict=verdict, status="AWAITING_SIGNATURES"))
            elif row.status == "AWAITING_SIGNATURES":
                row.verdict = verdict
            else:
                raise ValueError("settlement is no longer awaiting signatures")
            proposal = session.scalar(select(SettlementProposal).where(SettlementProposal.job_id == job.id))
            if proposal is None:
                session.add(SettlementProposal(job_id=job.id, inputs=task_inputs))
            else:
                proposal.inputs = task_inputs
            session.commit()
        with self.lock:
            self.tasks.setdefault(job_id, VerdictTask(job_id, task_inputs, verdict))
        return verdict

    def restore_pending(self) -> int:
        """Restore unexpired, still-active proposals; partial signatures are re-collected."""
        with self._propose_lock:
            return self._restore_pending()

    def _restore_pending(self) -> int:
        if self.chain is None:
            return 0
        self._resume_attested()
        with Session(self.engine) as session:
            records = list(session.execute(select(Settlement, SettlementProposal, Job).join(SettlementProposal, SettlementProposal.job_id == Settlement.job_id).join(Job, Job.id == Settlement.job_id).where(Settlement.status == "AWAITING_SIGNATURES")))
            snapshots = [(row.job_id, row.verdict, proposal.inputs, job.onchain_id, job.state) for row, proposal, job in records]
        restored = 0
        for job_id, stored_verdict, inputs, onchain_id, job_state in snapshots:
            with self.lock:
                if job_id in self.tasks:
                    continue
            try:
                if inputs["job_id"] != job_id or inputs["onchain_job_id"].lower() != onchain_id.lower() or inputs["chain_id"] != self.chain_id or inputs["contract"].lower() != self.contract.lower() or type(inputs["expiry"]) is not int:
                    continue
                expired = inputs["expiry"] <= int(time.time()) + MIN_SUBMIT_LEAD_SECONDS
                if type(inputs["passed"]) is not bool or job_state not in (JobState.COMPLETED.value, JobState.FAILED.value) or inputs["passed"] != (job_state == JobState.COMPLETED.value):
                    continue
                if verdict_from_inputs(inputs) != stored_verdict or self.chain.state(onchain_id) != "Active":
                    continue
                if expired:
                    # Signatures on an expired verdict are useless on-chain. Rebuild it from the stored job, with a new expiry,
                    # or nothing would ever propose it again and the providers would wait for a timeout refund.
                    self._propose(job_id)
                    restored += 1
                    continue
            except Exception:
                log.exception("couldn't restore pending settlement for %s", job_id)
                continue
            with self.lock:
                if job_id not in self.tasks:
                    self.tasks[job_id] = VerdictTask(job_id, inputs, stored_verdict)
                    restored += 1
        return restored

    def _payouts(self, session: Session, chain: list, budget: Decimal, spec: dict | None = None) -> tuple[dict[str, Decimal], Decimal]:
        """Workers sign checkpoints, but payment goes to the provider's own wallet. A fee in the spec is added on top."""
        terms = fee_terms(spec or {})
        steps = job_service.steps_by_signer(chain)
        by_key = job_service.providers_by_signer(session)
        rates, wallets = {}, {}
        for signer in steps:
            provider = by_key.get(signer)
            if provider is None:
                raise RuntimeError(f"no provider has the enrolled key {signer}")
            rates[signer] = Decimal(provider.price_per_step)
            wallets[signer] = session.get(Wallet, provider.wallet_id).address
        amounts, refund = job_service.compute_payouts(steps, rates, provider_budget(budget, terms[0]) if terms else budget)
        totals: dict[str, Decimal] = {}
        for signer, amount in amounts.items():
            totals[wallets[signer]] = totals.get(wallets[signer], Decimal(0)) + amount
        if terms:
            return add_fee(totals, budget, terms)
        return totals, refund

    def _payout_evidence(self, session: Session, chain: list) -> dict:
        """Expose signed manifests and known wallet delegations; missing history stays missing."""
        signers = {(checkpoint.signer or "").lower() for checkpoint in chain}
        proofs = session.scalars(select(ProviderKeyProof).where(ProviderKeyProof.key.in_(signers)))
        handovers = {}
        for checkpoint in chain[:-1]:
            reports = self.quality.decided_reports.get((checkpoint.job_id, checkpoint.manifest_hash), []) if self.quality else []
            if reports:
                handovers[checkpoint.manifest_hash] = reports
        return {
            "manifests": [checkpoint.manifest for checkpoint in chain],
            "delegations": {proof.key: {"provider_id": proof.provider_id, "signature": proof.signature} for proof in proofs},
            "handover_reports": handovers,
        }

    # ------------------------------------------------------------ signatures

    def claim(self, verifier: str) -> dict | None:
        verifier = verifier.lower()
        if verifier not in self.verifiers:
            raise PermissionError("not an allowlisted verifier")
        # A transient RPC outage at startup should not strand an otherwise valid proposal.
        now = time.monotonic()
        if self.engine is not None and self.chain is not None and now >= self._restore_retry_after and self._restore_lock.acquire(blocking=False):
            try:
                self._restore_retry_after = now + RESTORE_RETRY_SECONDS
                self.restore_pending()
            finally:
                self._restore_lock.release()
        with self.lock:
            for task in self.tasks.values():
                now = time.monotonic()
                claimed_at = task.claimed_by.get(verifier)
                if not task.submitted and verifier not in task.signatures and (claimed_at is None or now - claimed_at >= VERDICT_CLAIM_LEASE_SECONDS):
                    task.claimed_by[verifier] = now
                    return task.inputs
        return None

    def submit(self, job_id: str, signature: str) -> dict:
        with self.lock:
            task = self.tasks.get(job_id)
            if task is None:
                raise ValueError("no verdict is awaiting signatures for this job")
            try:
                signer = recover_verdict_signer(task.verdict, self.chain_id, self.contract, signature).lower()
            except Exception as exc:
                raise PermissionError("signature is malformed") from exc
            if signer not in self.verifiers:
                raise PermissionError("signature doesn't match the proposed verdict from an allowlisted verifier")
            task.signatures[signer] = signature
            ready = len(task.signatures) >= self.quorum and not task.submitted
            if ready:
                task.submitted = True
        if ready:
            threading.Thread(target=self.settle, args=(job_id,), daemon=True, name="settlement").start()
        return {"signatures": len(task.signatures), "quorum": self.quorum}

    def settle(self, job_id: str) -> None:
        """Submits the attestation, waits out the dispute window, then finalizes. Safe to run again."""
        task = self.tasks[job_id]
        if self.chain is None:
            self._set_status(job_id, "SIGNED_NO_CHAIN")
            return
        try:
            expiry = task.verdict.get("expiry")
            if expiry is not None and self.chain.state(task.verdict["jobId"]) == "Active" and int(expiry) - int(time.time()) < MIN_SUBMIT_LEAD_SECONDS:
                # Too close to expiry to rely on: a slow transaction would revert. Rebuild it and let the verifiers sign again.
                log.warning("verdict for %s is near expiry, so it is rebuilt and re-signed", job_id)
                with self._propose_lock:
                    with self.lock:
                        self.tasks.pop(job_id, None)
                    self._propose(job_id)
                return
            ordered = order_signatures(task.verdict, self.chain_id, self.contract, list(task.signatures.values()))
            result = self.chain.ensure_attested(task.verdict, ordered)
            if result.status == "ALREADY_DONE" and not self._attested_matches(task.verdict):
                raise ChainError("existing on-chain attestation requires reconciliation before finalizing")
            self._emit_tx(job_id, result)
            self._set_status(job_id, "ATTESTED", submit_tx=result.tx_hash)
            self._finalize_attested(job_id, task.verdict)
        except Exception:  # Web3 raises many types. All of them must reach the dashboard.
            self._settlement_failed(job_id)

    def _finalize_attested(self, job_id: str, verdict: dict) -> None:
        """Waits out the dispute window of an attested job, then finalizes it and closes the job."""
        onchain_id = verdict["jobId"]
        self.wait_for_window(onchain_id)
        result = self._finalize_with_retries(onchain_id)
        self._emit_tx(job_id, result)
        self._set_status(job_id, "SETTLED", finalize_tx=result.tx_hash)
        self._finish_job(job_id, passed=verdict["passed"])

    def _settlement_failed(self, job_id: str) -> None:
        log.exception("settlement of job %s stopped", job_id)
        self._set_status(job_id, "ERROR")
        self.orchestrator.emit_live("chain.tx", job_id, {"action": "settlement", "tx_hash": None, "status": "FAILED", "chain_id": self.chain_id, "explorer_url": None, "local_chain": self.local_chain})

    def _resume_attested(self, *, wait: bool = False) -> int:
        """Finalizes jobs attested before a restart. Their waiting thread died with the old process, so nothing else would.

        Only a stored verdict that matches the chain's own record is finalized. A disputed or different attestation is
        left alone for reconciliation.
        """
        with Session(self.engine) as session:
            pending = [(row.job_id, row.verdict) for row in session.scalars(select(Settlement).where(Settlement.status == "ATTESTED"))]
        started = []
        for job_id, verdict in pending:
            with self.lock:
                if job_id in self._finalizing or job_id in self.tasks:
                    continue
            try:
                if not self._attested_matches(verdict):
                    continue
            except Exception:
                log.exception("couldn't check the attestation of job %s", job_id)
                continue
            with self.lock:
                if job_id in self._finalizing:
                    continue
                self._finalizing.add(job_id)
            thread = threading.Thread(target=self._resume_one, args=(job_id, verdict), daemon=True, name="settlement-resume")
            thread.start()
            started.append(thread)
        if wait:
            for thread in started:
                thread.join()
        return len(started)

    def _resume_one(self, job_id: str, verdict: dict) -> None:
        try:
            log.info("resuming settlement of job %s after a restart", job_id)
            self._finalize_attested(job_id, verdict)
        except Exception:
            self._settlement_failed(job_id)
        finally:
            with self.lock:
                self._finalizing.discard(job_id)

    def _attested_matches(self, verdict: dict) -> bool:
        """Check the stored outcome and split before reconciling an already-attested job.

        The contract does not retain every signed verdict field; this checks the stored
        payout, not the identity of the original attestation transaction.
        """
        onchain = self.chain.job(verdict["jobId"])
        if onchain["state"] not in ("Attested", "Settled"):
            return False
        same_checkpoint = onchain["final_checkpoint"].lower() == str(verdict["finalCheckpoint"]).lower().removeprefix("0x")
        try:
            return (same_checkpoint and onchain["passed"] is verdict["passed"]
                    and [address.lower() for address in onchain["payees"]] == [address.lower() for address in verdict["payees"]]
                    and onchain["amounts"] == verdict["amounts"]
                    and onchain["refund"] == verdict["refund"])
        except (KeyError, TypeError, AttributeError):
            return False

    def wait_for_window(self, onchain_id: str) -> None:
        """Waits until the contract says the dispute window has closed, by the chain's own clock."""
        while True:
            remaining = self.chain.seconds_until_dispute_end(onchain_id)
            if remaining <= 0:
                return
            time.sleep(min(remaining + 1, 5))

    def _finalize_with_retries(self, onchain_id: str, attempts: int = 6):
        """A testnet's clock can lag slightly behind ours, so an early refusal just means wait and retry."""
        for attempt in range(attempts):
            try:
                return self.chain.ensure_finalized(onchain_id)
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(3)
        raise ChainError("finalize didn't complete")

    def _set_status(self, job_id: str, status: str, **fields) -> None:
        with Session(self.engine) as session:
            row = session.scalar(select(Settlement).where(Settlement.job_id == job_id))
            if row is not None:
                row.status = status
                for name, value in fields.items():
                    setattr(row, name, value)
                session.commit()

    def _finish_job(self, job_id: str, *, passed: bool) -> None:
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            target = JobState.SETTLED if passed else JobState.REFUNDED
            if job.state != target.value:
                previous = job.state
                job_service.transition(session, job, target, reason="Settled on-chain" if passed else "Refunded on-chain", actor="settlement")
                session.commit()
                self.orchestrator.emit_live("job.state_changed", job_id, {"from": previous, "to": target.value, "reason": "Finalized on-chain"})


def verdict_from_inputs(inputs: dict) -> dict:
    """Coordinator and verifiers both call this, so they derive identical data from the same inputs."""
    return build_verdict(
        onchain_job_id=inputs["onchain_job_id"],
        passed=inputs["passed"],
        final_checkpoint_hash=inputs["final_checkpoint_hash"],
        lineage=inputs["lineage"],
        reports=inputs["reports"],
        payouts={address: Decimal(amount) for address, amount in inputs["payouts"].items()},
        refund=Decimal(inputs["refund"]),
        budget=Decimal(inputs["budget"]),
        expiry=inputs["expiry"],
    )
