"""Talks to GPUChainEscrow. Each action reads chain state first, so a retry can never apply twice."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from eth_account import Account
from web3 import Web3

STATES = ("None", "Funded", "Active", "Attested", "Disputed", "Settled", "Refunded", "Cancelled")
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[3] / "contracts" / "out" / "GPUChainEscrow.sol" / "GPUChainEscrow.json"


class ChainError(RuntimeError):
    pass


@dataclass(frozen=True)
class TxResult:
    action: str
    tx_hash: str | None  # None means chain state was already right, so nothing was sent.
    status: str


def _verdict_tuple(verdict: dict) -> tuple:
    return (
        Web3.to_bytes(hexstr=verdict["jobId"]),
        verdict["passed"],
        Web3.to_bytes(hexstr=verdict["finalCheckpoint"]),
        Web3.to_bytes(hexstr=verdict["lineageRoot"]),
        Web3.to_bytes(hexstr=verdict["qualityReport"]),
        [Web3.to_checksum_address(a) for a in verdict["payees"]],
        list(verdict["amounts"]),
        verdict["refund"],
        verdict["expiry"],
    )


class EscrowClient:
    def __init__(self, rpc_url: str, escrow_address: str, private_key: str, *, artifact: Path = DEFAULT_ARTIFACT, timeout: float = 120.0):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
        abi = json.loads(Path(artifact).read_text(encoding="utf-8"))["abi"]
        self.contract = self.w3.eth.contract(address=Web3.to_checksum_address(escrow_address), abi=abi)
        self.account = Account.from_key(private_key)
        self.timeout = timeout
        self._lock = threading.Lock()  # One sender at a time, so nonces can't collide.

    def _send(self, action: str, call) -> TxResult:
        with self._lock:
            sender = {"from": self.account.address}
            # Estimate against the pending block, whose time is what the transaction will actually see.
            # Against "latest", a time check can fail for ever on a chain that only mines on demand.
            gas = int(call.estimate_gas(sender, "pending") * 1.3)
            tx = call.build_transaction({**sender, "gas": gas, "nonce": self.w3.eth.get_transaction_count(self.account.address, "pending"), "chainId": self.w3.eth.chain_id})
            signed = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.timeout)
        if receipt["status"] != 1:
            raise ChainError(f"{action} reverted in {tx_hash.hex()}")
        return TxResult(action, "0x" + tx_hash.hex().removeprefix("0x"), "CONFIRMED")

    def job(self, job_id: str) -> dict:
        """Field positions follow the Job struct in GPUChainEscrow.sol."""
        raw = self.contract.functions.getJob(Web3.to_bytes(hexstr=job_id)).call()
        return {
            "client": raw[0],
            "deadline": raw[1],
            "state": STATES[raw[2]],
            "dispute_window": int(raw[4]),
            "dispute_end": raw[5],
            "passed": bool(raw[7]),
            "budget": raw[8],
            "refund": raw[10],
            "spec_hash": raw[11].hex().removeprefix("0x"),
            "policy_hash": raw[12].hex().removeprefix("0x"),
            "final_checkpoint": raw[13].hex().removeprefix("0x"),
            "payees": list(raw[14]),
            "amounts": list(raw[15]),
        }

    def state(self, job_id: str) -> str:
        return self.job(job_id)["state"]

    def rpc_chain_id(self) -> int:
        """Read the connected RPC's chain ID, including for local-only exceptions."""
        return int(self.w3.eth.chain_id)

    def seconds_until_dispute_end(self, job_id: str) -> int:
        # Local chains only mine on transactions, so the latest block's time can stand still for ever.
        # The next block will be stamped with at least the current time, so that's what matters.
        now = max(self.w3.eth.get_block("latest")["timestamp"], int(time.time()))
        return self.job(job_id)["dispute_end"] - now

    def timeout_refund_status(self, job_id: str) -> dict:
        """Read eligibility from chain time; this call never sends a transaction."""
        job = self.job(job_id)
        block_time = int(self.w3.eth.get_block("latest")["timestamp"])
        available_at = int(job["deadline"]) + int(self.contract.functions.REFUND_GRACE().call())
        return {"state": job["state"], "client": job["client"], "budget": job["budget"], "chain_time": block_time, "available_at": available_at, "eligible": job["state"] in ("Funded", "Active") and block_time >= available_at}

    def ensure_timeout_refund(self, job_id: str) -> TxResult:
        """Explicit caller action after the contract's deadline and grace period."""
        status = self.timeout_refund_status(job_id)
        if status["state"] == "Refunded":
            return TxResult("claimTimeoutRefund", None, "ALREADY_DONE")
        if not status["eligible"]:
            raise ChainError("timeout refund is not yet eligible, or job is not Funded/Active")
        return self._send("claimTimeoutRefund", self.contract.functions.claimTimeoutRefund(Web3.to_bytes(hexstr=job_id)))

    def verdict_digest(self, verdict: dict) -> str:
        return "0x" + self.contract.functions.hashVerdict(_verdict_tuple(verdict)).call().hex().removeprefix("0x")

    def ensure_active(self, job_id: str) -> TxResult:
        state = self.state(job_id)
        if state != "Funded":
            if state in ("Active", "Attested", "Disputed", "Settled", "Refunded"):
                return TxResult("markActive", None, "ALREADY_DONE")
            raise ChainError(f"job can't become active from state {state}")
        return self._send("markActive", self.contract.functions.markActive(Web3.to_bytes(hexstr=job_id)))

    def anchor(self, job_id: str, kind: int, attempt: int, data_hash: str) -> TxResult:
        """A log only. Skipped once the job has left Active, since the contract would refuse it."""
        if self.state(job_id) != "Active":
            return TxResult("anchorEvent", None, "SKIPPED")
        return self._send("anchorEvent", self.contract.functions.anchorEvent(Web3.to_bytes(hexstr=job_id), kind, attempt, Web3.to_bytes(hexstr=data_hash)))

    def ensure_attested(self, verdict: dict, signatures: list[str]) -> TxResult:
        state = self.state(verdict["jobId"])
        if state in ("Attested", "Disputed", "Settled", "Refunded"):
            return TxResult("submitAttestation", None, "ALREADY_DONE")
        if state != "Active":
            raise ChainError(f"can't attest a job in state {state}")
        return self._send(
            "submitAttestation",
            self.contract.functions.submitAttestation(_verdict_tuple(verdict), [Web3.to_bytes(hexstr=s) for s in signatures]),
        )

    def ensure_finalized(self, job_id: str) -> TxResult:
        state = self.state(job_id)
        if state in ("Settled", "Refunded"):
            return TxResult("finalize", None, "ALREADY_DONE")
        if state != "Attested":
            raise ChainError(f"can't finalize a job in state {state}")
        return self._send("finalize", self.contract.functions.finalize(Web3.to_bytes(hexstr=job_id)))
