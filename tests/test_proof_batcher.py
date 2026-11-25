"""C25: proofs are recorded, batched, anchored and checkable. The contract is faked here (same rules as ProofAnchor.sol);
the last test runs the real contract on a local chain when `anvil` and the compiled contract are available."""

try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import json
import os
import shutil
import subprocess
import time
import unittest
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core import proofs as core_proofs
from ekido_coordinator import proof_batcher
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import Proof, ProofBatch
from ekido_coordinator.proof_batcher import ANCHORED, MISMATCH, PENDING, Batcher, ProofError, proof_for_job, record_proof, verify_on_chain

T, V = "a" * 64, "b" * 64


class FakeAnchor:
    """The ProofAnchor contract's rules: append only, no empty root or count, no repeated root."""

    def __init__(self):
        self.batches, self.sent, self.fail_before_write, self.fail_after_write, self.lie_root = [], 0, False, False, False

    def is_anchored(self, root):
        for i, (r, _, _) in enumerate(self.batches):
            if r == root:
                return True, i
        return False, 0

    def get_batch(self, batch_id):
        if batch_id >= len(self.batches):
            raise IndexError("UnknownBatch")
        root, count, time_ = self.batches[batch_id]
        return (b"\x99" * 32 if self.lie_root else root), count, time_

    def anchor(self, root, count):
        if self.fail_before_write:
            raise ConnectionError("rpc down")
        if root == b"\x00" * 32 or count == 0 or self.is_anchored(root)[0]:
            raise ProofError("reverted")
        self.batches.append((root, count, 1_800_000_000 + len(self.batches)))
        self.sent += 1
        if self.fail_after_write:
            raise ConnectionError("lost the reply after the transaction was mined")
        return "0x" + f"{len(self.batches):064x}"


class Base(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None))
        self.engine = self.app.state.engine
        self.session = Session(self.engine)
        self.addCleanup(self.session.close)
        self.chain = FakeAnchor()
        self.batcher = Batcher(self.engine, self.chain)

    def add(self, n, start=0):
        for i in range(start, start + n):
            record_proof(self.session, job_id=f"job-{i:04d}", terms_hash=T, verdict_hash=f"{i:064x}", payouts={"provider-a": 100 + i, "provider-b": 50})


class RecordTests(Base):
    def test_recording_is_idempotent_and_a_different_proof_for_the_same_job_is_refused(self):
        first = record_proof(self.session, job_id="j1", terms_hash=T, verdict_hash=V, payouts={"a": 5})
        again = record_proof(self.session, job_id="j1", terms_hash=T, verdict_hash=V, payouts={"a": 5})
        self.assertEqual(first.id, again.id)
        for change in (dict(verdict_hash="c" * 64), dict(terms_hash="d" * 64), dict(payouts={"a": 6})):
            with self.subTest(change=change), self.assertRaises(ProofError):
                record_proof(self.session, **{**dict(job_id="j1", terms_hash=T, verdict_hash=V, payouts={"a": 5}), **change})
        self.assertEqual(len(self.session.scalars(select(Proof)).all()), 1)

    def test_bad_proofs_are_refused_before_anything_is_stored(self):
        for kwargs in (dict(job_id="", terms_hash=T, verdict_hash=V, payouts={"a": 1}), dict(job_id="j", terms_hash="xyz", verdict_hash=V, payouts={"a": 1}),
                       dict(job_id="j", terms_hash=T, verdict_hash=V, payouts={}), dict(job_id="j", terms_hash=T, verdict_hash=V, payouts={"a": 0})):
            with self.subTest(kwargs=kwargs), self.assertRaises(MerkleErrorTypes):
                record_proof(self.session, **kwargs)
        self.assertEqual(self.session.scalars(select(Proof)).all(), [])


from ekido_core.merkle import MerkleError as MerkleErrorTypes  # noqa: E402


class BatchTests(Base):
    def test_nothing_waiting_does_nothing(self):
        self.assertIsNone(self.batcher.run_once())
        self.assertEqual(self.chain.sent, 0)

    def test_waiting_proofs_become_one_anchored_batch_and_each_verifies_against_the_contract(self):
        self.add(11)
        batch = self.batcher.run_once()
        self.assertEqual((batch.status, batch.count, batch.onchain_id), (ANCHORED, 11, 0))
        self.assertEqual(self.chain.batches[0][0].hex(), batch.root)
        self.assertEqual(self.chain.batches[0][1], 11)
        for i in range(11):
            proof = proof_for_job(self.session, f"job-{i:04d}")
            self.assertEqual((proof["index"], proof["tree_size"], proof["batch"]["root"]), (i, 11, "0x" + batch.root))
            self.assertTrue(verify_on_chain(self.session, self.chain, f"job-{i:04d}"))
            self.assertTrue(core_proofs.verify_proof(proof, proof["batch"]["root"], count=11))

    def test_a_second_run_with_nothing_new_sends_nothing_and_new_proofs_go_in_a_new_batch(self):
        self.add(3)
        first = self.batcher.run_once()
        self.assertIsNone(self.batcher.run_once())
        self.add(2, start=3)
        second = self.batcher.run_once()
        self.assertNotEqual(first.id, second.id)
        self.assertEqual((second.count, second.onchain_id, self.chain.sent), (2, 1, 2))
        self.assertTrue(verify_on_chain(self.session, self.chain, "job-0000"))  # the first batch is untouched
        self.assertTrue(verify_on_chain(self.session, self.chain, "job-0004"))

    def test_max_batch_splits_the_work_in_order(self):
        self.batcher = Batcher(self.engine, self.chain, max_batch=4)
        self.add(10)
        runs = [self.batcher.run_once().count for _ in range(3)]
        self.assertEqual(runs, [4, 4, 2])
        self.assertIsNone(self.batcher.run_once())
        self.assertEqual(sorted(p.leaf_index for p in self.session.scalars(select(Proof))), [0, 0, 0, 1, 1, 1, 2, 2, 3, 3])

    def test_unknown_and_unbatched_jobs_have_no_proof_yet(self):
        self.assertIsNone(proof_for_job(self.session, "nope"))
        self.add(1)
        self.assertIsNone(proof_for_job(self.session, "job-0000"))
        self.assertFalse(verify_on_chain(self.session, self.chain, "job-0000"))

    def test_the_proof_hides_payees_and_data(self):
        self.add(2)
        self.batcher.run_once()
        text = json.dumps(proof_for_job(self.session, "job-0000"))
        self.assertNotIn("provider", text)
        self.assertNotIn("100", [p for p in json.loads(text)["record"].values() if isinstance(p, int)])


class CrashSafetyTests(Base):
    def test_a_failed_write_leaves_a_pending_batch_that_the_next_run_anchors_with_the_same_root(self):
        self.add(5)
        self.chain.fail_before_write = True
        with self.assertRaises(ConnectionError):
            self.batcher.run_once()
        pending = self.session.scalar(select(ProofBatch))
        self.assertEqual(pending.status, PENDING)
        root = pending.root
        self.add(2, start=5)  # new proofs arrive meanwhile: they must not change the pending batch
        self.chain.fail_before_write = False
        batch = self.batcher.run_once()
        self.assertEqual((batch.id, batch.root, batch.count, batch.status), (pending.id, root, 5, ANCHORED))
        self.assertEqual(self.chain.sent, 1)
        self.assertEqual(self.batcher.run_once().count, 2)  # and then the newcomers

    def test_a_write_that_succeeded_but_was_never_recorded_is_finished_not_repeated(self):
        self.add(4)
        self.chain.fail_after_write = True
        with self.assertRaises(ConnectionError):
            self.batcher.run_once()
        self.assertEqual(self.chain.sent, 1)
        self.assertEqual(self.session.scalar(select(ProofBatch)).status, PENDING)
        self.chain.fail_after_write = False
        batch = self.batcher.run_once()
        self.assertEqual((batch.status, batch.onchain_id, self.chain.sent), (ANCHORED, 0, 1))  # no second transaction

    def test_a_contract_that_stores_something_else_marks_the_batch_mismatch_and_stops(self):
        self.add(3)
        self.chain.lie_root = True
        with self.assertRaises(ProofError):
            self.batcher.run_once()
        self.assertEqual(self.session.scalar(select(ProofBatch)).status, MISMATCH)
        self.assertFalse(verify_on_chain(self.session, self.chain, "job-0000"))

    def test_a_batch_whose_records_changed_is_not_anchored(self):
        self.add(3)
        self.chain.fail_before_write = True
        with self.assertRaises(ConnectionError):
            self.batcher.run_once()
        self.chain.fail_before_write = False
        proof = self.session.scalar(select(Proof).where(Proof.job_id == "job-0001"))
        proof.payouts = {"provider-a": 1}  # tampered between forming the batch and anchoring it
        self.session.commit()
        with self.assertRaises(ProofError):
            self.batcher.run_once()
        self.assertEqual((self.chain.sent, self.session.scalar(select(ProofBatch)).status), (0, MISMATCH))

    def test_tampering_after_anchoring_is_caught_by_the_check(self):
        self.add(6)
        self.batcher.run_once()
        self.assertTrue(verify_on_chain(self.session, self.chain, "job-0002"))
        proof = self.session.scalar(select(Proof).where(Proof.job_id == "job-0002"))
        proof.payouts = {"provider-a": 999_999}
        self.session.commit()
        self.assertFalse(verify_on_chain(self.session, self.chain, "job-0002"))

    def test_an_unreachable_chain_means_not_verified(self):
        self.add(2)
        self.batcher.run_once()
        self.chain.get_batch = lambda batch_id: (_ for _ in ()).throw(ConnectionError("down"))
        self.assertFalse(verify_on_chain(self.session, self.chain, "job-0000"))


ANVIL = shutil.which("anvil") or str(Path.home() / "tools" / "foundry-1.9.0" / "bin" / "anvil.exe")
ARTIFACT = Path(os.environ.get("ANCHOR_ARTIFACT") or Path(__file__).resolve().parents[2] / "ekido-contracts" / "out" / "ProofAnchor.sol" / "ProofAnchor.json")
# Anvil's well-known test accounts (public, worthless).
COORDINATOR_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
COORDINATOR = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
OWNER_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"


@unittest.skipUnless(os.path.exists(ANVIL) and ARTIFACT.exists(), "needs anvil and the compiled ProofAnchor (forge build in ekido-contracts)")
class RealContractTests(Base):
    """The real ProofAnchor on a local chain, through the real web3 client."""

    @classmethod
    def setUpClass(cls):
        cls.node = subprocess.Popen([ANVIL, "--port", "8561", "--silent"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from web3 import Web3

        cls.w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8561"))
        for _ in range(50):
            if cls.w3.is_connected():
                break
            time.sleep(0.2)
        art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        bytecode = art["bytecode"]["object"]
        from eth_account import Account

        owner = Account.from_key(OWNER_KEY)
        contract = cls.w3.eth.contract(abi=art["abi"], bytecode=bytecode)
        tx = contract.constructor(owner.address, COORDINATOR).build_transaction({"from": owner.address, "nonce": cls.w3.eth.get_transaction_count(owner.address), "chainId": cls.w3.eth.chain_id})
        receipt = cls.w3.eth.wait_for_transaction_receipt(cls.w3.eth.send_raw_transaction(owner.sign_transaction(tx).raw_transaction))
        cls.address = receipt["contractAddress"]

    @classmethod
    def tearDownClass(cls):
        cls.node.terminate()
        cls.node.wait(timeout=10)

    def test_a_batch_is_anchored_on_the_real_contract_and_checks_out_and_a_stranger_cannot_write(self):
        client = proof_batcher.Web3AnchorClient("http://127.0.0.1:8561", self.address, COORDINATOR_KEY, artifact=ARTIFACT)
        self.batcher = Batcher(self.engine, client)
        self.add(7)
        batch = self.batcher.run_once()
        self.assertEqual((batch.status, batch.count), (ANCHORED, 7))
        self.assertTrue(batch.tx_hash.startswith("0x"))
        for i in range(7):
            self.assertTrue(verify_on_chain(self.session, client, f"job-{i:04d}"))
        self.assertEqual(client.contract.functions.batchCount().call(), 1)
        # Writing the same root again is refused by the contract; the batcher never needs to try (it asks first).
        with self.assertRaises(Exception):
            client.anchor(bytes.fromhex(batch.root), 7)
        # Someone who is not the coordinator cannot write.
        stranger = proof_batcher.Web3AnchorClient("http://127.0.0.1:8561", self.address, OWNER_KEY, artifact=ARTIFACT)
        with self.assertRaises(Exception):
            stranger.anchor(b"\x01" * 32, 1)
        self.assertEqual(client.contract.functions.batchCount().call(), 1)
        self.assertEqual(self.w3.eth.get_balance(self.address), 0)  # the contract holds no money
        # A second run anchors new proofs as batch 1, and the first batch still verifies.
        self.add(3, start=7)
        self.assertEqual(self.batcher.run_once().onchain_id, 1)
        self.assertTrue(verify_on_chain(self.session, client, "job-0000"))
        self.assertTrue(verify_on_chain(self.session, client, "job-0009"))


if __name__ == "__main__":
    unittest.main()
