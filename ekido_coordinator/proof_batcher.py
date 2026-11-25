"""Every paid job's proof, batched into a Merkle tree and anchored on Base Sepolia (ticket C25).

1. `record_proof` stores a job's proof: its terms hash, its verdict hash and its payout split. Recording the same proof
   again changes nothing; recording a different one for the same job is refused.
2. `Batcher.run_once` takes the proofs not yet in a batch, builds the tree (`ekido_core.proofs`), fixes the batch
   (a PENDING row, with each proof's place in it) and writes the root to the ProofAnchor contract.
3. Crash safety: the batch is fixed *before* the write, so a retry rebuilds nothing and anchors the same root. Before
   sending, the contract is asked whether the root is already there, so a write that succeeded but was never recorded here
   is finished, not repeated.
4. After the write, the root and count are read back from the contract and must match; a difference marks the batch MISMATCH
   and stops, instead of pretending it worked.
5. `proof_for_job` gives a customer their record, place and audit path; `verify_on_chain` checks it against the contract.

The contract holds no money and only this coordinator's address may write to it. It is deployed only after Piyush's "haan".
Nothing here calls `record_proof` from settlement yet: that wiring is one line in the code that finishes a paid job.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Protocol

from eth_account import Account
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from web3 import Web3

from ekido_core import proofs as core_proofs
from .models import Proof, ProofBatch

log = logging.getLogger(__name__)

ANCHORED, PENDING, MISMATCH = "ANCHORED", "PENDING", "MISMATCH"
MAX_BATCH = 1000
# ANCHOR_ARTIFACT points at the compiled ProofAnchor when the contracts aren't three folders up (a split-repo checkout).
DEFAULT_ARTIFACT = Path(os.environ.get("ANCHOR_ARTIFACT") or Path(__file__).resolve().parents[3] / "contracts" / "out" / "ProofAnchor.sol" / "ProofAnchor.json")


class ProofError(RuntimeError):
    pass


class AnchorClient(Protocol):
    """The ProofAnchor contract as we use it. Tests pass a fake; production passes `Web3AnchorClient`."""

    def is_anchored(self, root: bytes) -> tuple[bool, int]:
        """Whether the root is in the contract, and its batch id."""

    def get_batch(self, batch_id: int) -> tuple[bytes, int, int]:
        """(root, count, time) of an anchored batch."""

    def anchor(self, root: bytes, count: int) -> str:
        """Write a batch; returns the transaction hash. Reverts if the root is already there."""


class Web3AnchorClient:
    def __init__(self, rpc_url: str, anchor_address: str, private_key: str, *, artifact: Path = DEFAULT_ARTIFACT, timeout: float = 120.0):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
        abi = json.loads(Path(artifact).read_text(encoding="utf-8"))["abi"]
        self.contract = self.w3.eth.contract(address=Web3.to_checksum_address(anchor_address), abi=abi)
        self.account = Account.from_key(private_key)
        self.timeout = timeout
        self._lock = threading.Lock()  # One sender at a time, so nonces can't collide.

    def is_anchored(self, root: bytes) -> tuple[bool, int]:
        found, batch_id = self.contract.functions.isAnchored(root).call()
        return bool(found), int(batch_id)

    def get_batch(self, batch_id: int) -> tuple[bytes, int, int]:
        root, count, time = self.contract.functions.getBatch(batch_id).call()
        return bytes(root), int(count), int(time)

    def anchor(self, root: bytes, count: int) -> str:
        call = self.contract.functions.anchor(root, count)
        with self._lock:
            sender = {"from": self.account.address}
            gas = int(call.estimate_gas(sender, "pending") * 1.3)
            tx = call.build_transaction({**sender, "gas": gas, "nonce": self.w3.eth.get_transaction_count(self.account.address, "pending"), "chainId": self.w3.eth.chain_id})
            tx_hash = self.w3.eth.send_raw_transaction(self.account.sign_transaction(tx).raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.timeout)
        if receipt["status"] != 1:
            raise ProofError(f"anchor reverted in {tx_hash.hex()}")
        return "0x" + tx_hash.hex().removeprefix("0x")


# ---------------------------------------------------------------------------------------------------- recording


def record_proof(session: Session, *, job_id: str, terms_hash: str, verdict_hash: str, payouts: dict[str, int]) -> Proof:
    """Store a paid job's proof. Idempotent; a different proof for the same job is refused."""
    core_proofs.proof_record(job_id, terms_hash, verdict_hash, payouts)  # validates everything before anything is stored
    existing = session.scalar(select(Proof).where(Proof.job_id == job_id))
    if existing is not None:
        if (existing.terms_hash, existing.verdict_hash, existing.payouts) != (terms_hash, verdict_hash, dict(payouts)):
            raise ProofError(f"job {job_id} already has a different proof")
        return existing
    proof = Proof(job_id=job_id, terms_hash=terms_hash, verdict_hash=verdict_hash, payouts=dict(payouts))
    session.add(proof)
    session.commit()
    return proof


def _record(proof: Proof) -> dict:
    return core_proofs.proof_record(proof.job_id, proof.terms_hash, proof.verdict_hash, proof.payouts)


def _members(session: Session, batch: ProofBatch) -> list[Proof]:
    return list(session.scalars(select(Proof).where(Proof.batch_id == batch.id).order_by(Proof.leaf_index)))


# ---------------------------------------------------------------------------------------------------- batching


class Batcher:
    def __init__(self, engine: Engine, client: AnchorClient, max_batch: int = MAX_BATCH):
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        self.engine, self.client, self.max_batch = engine, client, max_batch

    def run_once(self) -> ProofBatch | None:
        """Finish a pending batch if there is one, else batch and anchor what is waiting. None means nothing to do."""
        with Session(self.engine) as session:
            batch = session.scalar(select(ProofBatch).where(ProofBatch.status == PENDING).order_by(ProofBatch.created_at))
            if batch is None:
                batch = self._form_batch(session)
            if batch is None:
                return None
            return self._anchor(session, batch)

    def _form_batch(self, session: Session) -> ProofBatch | None:
        waiting = list(session.scalars(select(Proof).where(Proof.batch_id.is_(None)).order_by(Proof.created_at, Proof.job_id).limit(self.max_batch)))
        if not waiting:
            return None
        root = core_proofs.batch_root([_record(p) for p in waiting])
        batch = ProofBatch(root=root.hex(), count=len(waiting), status=PENDING)
        session.add(batch)
        session.flush()
        for index, proof in enumerate(waiting):
            proof.batch_id, proof.leaf_index = batch.id, index
        session.commit()  # fixed before anything is sent: a retry anchors this same root
        return batch

    def _anchor(self, session: Session, batch: ProofBatch) -> ProofBatch:
        members = _members(session, batch)
        root = core_proofs.batch_root([_record(p) for p in members])
        if root.hex() != batch.root or len(members) != batch.count:
            batch.status = MISMATCH
            session.commit()
            raise ProofError(f"batch {batch.id} no longer matches its stored root; it was not anchored")
        found, onchain_id = self.client.is_anchored(root)
        if not found:
            batch.tx_hash = self.client.anchor(root, batch.count)
            found, onchain_id = self.client.is_anchored(root)
        if not found:
            raise ProofError(f"batch {batch.id}: the contract does not show the root after the write")
        stored_root, stored_count, _ = self.client.get_batch(onchain_id)  # read it back, don't assume
        if stored_root != root or stored_count != batch.count:
            batch.status = MISMATCH
            session.commit()
            raise ProofError(f"batch {batch.id}: the contract holds a different root or count than was sent")
        batch.status, batch.onchain_id = ANCHORED, onchain_id
        session.commit()
        log.info("anchored batch %s: %s proofs", batch.id, batch.count)
        return batch


# ---------------------------------------------------------------------------------------------------- checking


def proof_for_job(session: Session, job_id: str) -> dict | None:
    """The customer's proof: record, place, batch size and audit path, plus where the batch is. None if not batched yet."""
    proof = session.scalar(select(Proof).where(Proof.job_id == job_id))
    if proof is None or proof.batch_id is None:
        return None
    batch = session.get(ProofBatch, proof.batch_id)
    records = [_record(p) for p in _members(session, batch)]
    out = core_proofs.prove(records, proof.leaf_index)
    out["batch"] = {"root": "0x" + batch.root, "count": batch.count, "status": batch.status, "tx_hash": batch.tx_hash, "onchain_id": batch.onchain_id}
    return out


def verify_on_chain(session: Session, client: AnchorClient, job_id: str) -> bool:
    """Rebuild the job's proof from our records and check it against what the contract actually holds."""
    found = proof_for_job(session, job_id)
    if found is None or found["batch"]["onchain_id"] is None or found["batch"]["status"] != ANCHORED:
        return False
    try:
        root, count, _ = client.get_batch(found["batch"]["onchain_id"])
    except Exception:  # the chain can't be reached, or the id is unknown: not verified
        return False
    return core_proofs.verify_proof({k: found[k] for k in ("record", "index", "tree_size", "path")}, "0x" + root.hex(), count=count)
