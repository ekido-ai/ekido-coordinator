"""State machine, fencing, lineage and payouts, on in-memory SQLite. Needs sqlalchemy."""

import unittest
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from gpuchain_core.canonical import document_hash
from gpuchain_core.job_states import IllegalTransition, JobState
from gpuchain_coordinator import jobs
from gpuchain_coordinator.models import Base, Provider, Wallet

SPEC = {"template": "synthetic-linear/v1", "seed": 1}
POLICY = {"metric": "accuracy"}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.addCleanup(self.session.close)
        wallet = Wallet(address="0x" + "1" * 40, role="provider")
        self.session.add(wallet)
        self.session.flush()
        self.provider = Provider(wallet_id=wallet.id, name="A", kind="REAL")
        self.session.add(self.provider)
        self.session.flush()
        self.job = jobs.create_job(self.session, spec=SPEC, policy=POLICY, budget=Decimal("5"), deadline_seconds=600)

    def manifest(self, step, token=None, parent=None, signer_note="a"):
        return {
            "format": "GPCK/1",
            "logical_step": step,
            "fencing_token": self.job.fencing_token if token is None else token,
            "immutable_hash": self.job.spec_hash,
            "parent_manifest_hash": parent,
            "files": {"trainable.safetensors": {"sha256": "0" * 64, "bytes": 10}},
            "note": signer_note,
        }

    def commit(self, step, parent=None, state="VERIFIED", signer="0xa", note="a"):
        checkpoint = jobs.commit_checkpoint(self.session, self.job, self.manifest(step, parent=parent, signer_note=note), fencing_token=self.job.fencing_token, signer=signer)
        checkpoint.state = state
        self.session.flush()
        return checkpoint


class CreateAndTransition(Fixture):
    def test_hashes_are_committed(self):
        self.assertEqual(self.job.spec_hash, document_hash(SPEC))
        self.assertEqual(self.job.policy_hash, document_hash(POLICY))
        self.assertRegex(self.job.onchain_id, r"^0x[0-9a-f]{64}$")

    def test_identical_specs_get_distinct_onchain_ids(self):
        other = jobs.create_job(self.session, spec=SPEC, policy=POLICY, budget=Decimal("5"), deadline_seconds=600)
        self.assertNotEqual(other.onchain_id, self.job.onchain_id)

    def test_transition_records_history(self):
        jobs.transition(self.session, self.job, JobState.ASSIGNED, reason="scheduled", actor="scheduler")
        jobs.transition(self.session, self.job, JobState.RUNNING)
        self.assertEqual(self.job.state, "RUNNING")
        self.assertEqual(self.job.state_version, 2)
        self.session.expire_all()
        self.assertEqual([(t.seq, t.from_state, t.to_state) for t in self.job.transitions], [(1, "OPEN", "ASSIGNED"), (2, "ASSIGNED", "RUNNING")])

    def test_illegal_transition_refused(self):
        with self.assertRaises(IllegalTransition):
            jobs.transition(self.session, self.job, JobState.SETTLED)
        self.assertEqual(self.job.state, "OPEN")

    def test_concurrent_change_loses(self):
        # A second session moves the job first, so this session's copy is stale.
        self.session.commit()
        with Session(self.engine) as other:
            copy = other.get(type(self.job), self.job.id)
            jobs.transition(other, copy, JobState.ASSIGNED)
            other.commit()
        with self.assertRaises(jobs.StaleVersion):
            jobs.transition(self.session, self.job, JobState.CANCELLED)


class Fencing(Fixture):
    def test_each_assignment_raises_the_token(self):
        first = jobs.assign(self.session, self.job, self.provider.id, {"micro_batch": 32})
        second = jobs.assign(self.session, self.job, self.provider.id, {"micro_batch": 8})
        self.assertEqual((first.fencing_token, second.fencing_token), (1, 2))
        self.assertEqual((first.attempt, second.attempt), (1, 2))

    def test_zombie_write_is_refused(self):
        jobs.assign(self.session, self.job, self.provider.id, {})
        old_manifest = self.manifest(10, token=1)
        jobs.assign(self.session, self.job, self.provider.id, {})  # Recovery replaced the worker.
        with self.assertRaises(jobs.StaleFencingToken) as caught:
            jobs.commit_checkpoint(self.session, self.job, old_manifest, fencing_token=1)
        self.assertEqual(caught.exception.current, 2)

    def test_token_in_manifest_must_match_too(self):
        jobs.assign(self.session, self.job, self.provider.id, {})
        jobs.assign(self.session, self.job, self.provider.id, {})
        with self.assertRaises(jobs.StaleFencingToken):
            jobs.commit_checkpoint(self.session, self.job, self.manifest(10, token=1), fencing_token=2)


class Checkpoints(Fixture):
    def setUp(self):
        super().setUp()
        jobs.assign(self.session, self.job, self.provider.id, {})

    def test_wrong_job_refused(self):
        bad = dict(self.manifest(10), immutable_hash="f" * 64)
        with self.assertRaises(jobs.LineageError):
            jobs.commit_checkpoint(self.session, self.job, bad, fencing_token=1)

    def test_commit_is_idempotent(self):
        manifest = self.manifest(10)
        first = jobs.commit_checkpoint(self.session, self.job, manifest, fencing_token=1)
        second = jobs.commit_checkpoint(self.session, self.job, manifest, fencing_token=1)
        self.assertEqual(first.id, second.id)

    def test_unknown_parent_refused(self):
        with self.assertRaises(jobs.LineageError):
            jobs.commit_checkpoint(self.session, self.job, self.manifest(10, parent="a" * 64), fencing_token=1)

    def test_cannot_build_on_a_rejected_checkpoint(self):
        bad = self.commit(10, state="REJECTED")
        with self.assertRaises(jobs.LineageError):
            jobs.commit_checkpoint(self.session, self.job, self.manifest(20, parent=bad.manifest_hash), fencing_token=1)

    def test_step_must_increase(self):
        parent = self.commit(10)
        with self.assertRaises(jobs.LineageError):
            jobs.commit_checkpoint(self.session, self.job, self.manifest(10, parent=parent.manifest_hash, signer_note="b"), fencing_token=1)

    def test_resume_point_skips_rejected_and_unverified(self):
        accepted = self.commit(10, state="ACCEPTED")
        verified = self.commit(20, parent=accepted.manifest_hash, state="VERIFIED")
        self.commit(30, parent=verified.manifest_hash, state="COMMITTED")
        self.assertEqual(jobs.latest_resumable_checkpoint(self.session, self.job).logical_step, 20)
        verified.state = "REJECTED"
        self.session.flush()
        self.assertEqual(jobs.latest_resumable_checkpoint(self.session, self.job).logical_step, 10)
        self.assertEqual(jobs.latest_accepted_checkpoint(self.session, self.job).logical_step, 10)

    def test_no_checkpoint_yet(self):
        self.assertIsNone(jobs.latest_resumable_checkpoint(self.session, self.job))


class Payouts(Fixture):
    def setUp(self):
        super().setUp()
        jobs.assign(self.session, self.job, self.provider.id, {})

    def test_rolled_back_segment_earns_nothing(self):
        # A does 0-100. Cheater P does 100-150 and is rejected. B redoes it from 100 to 300.
        a = self.commit(100, signer="0xA", state="ACCEPTED")
        self.commit(150, parent=a.manifest_hash, signer="0xP", state="REJECTED", note="p")
        b1 = self.commit(200, parent=a.manifest_hash, signer="0xB", state="ACCEPTED", note="b1")
        tip = self.commit(300, parent=b1.manifest_hash, signer="0xB", state="ACCEPTED", note="b2")

        chain = jobs.accepted_lineage(self.session, tip)
        self.assertEqual([c.logical_step for c in chain], [100, 200, 300])
        steps = jobs.steps_by_signer(chain)
        self.assertEqual(steps, {"0xa": 100, "0xb": 200})
        self.assertNotIn("0xp", steps)

    def test_steps_sum_to_total(self):
        a = self.commit(100, signer="0xa", state="ACCEPTED")
        tip = self.commit(300, parent=a.manifest_hash, signer="0xb", state="ACCEPTED", note="b")
        self.assertEqual(sum(jobs.steps_by_signer(jobs.accepted_lineage(self.session, tip)).values()), 300)

    def test_payouts_and_refund_conserve_budget(self):
        payouts, refund = jobs.compute_payouts({"0xa": 100, "0xb": 200}, {"0xa": Decimal("0.01"), "0xb": Decimal("0.005")}, Decimal("5"))
        self.assertEqual(payouts, {"0xa": Decimal("1.00"), "0xb": Decimal("1.000")})
        self.assertEqual(refund, Decimal("3.000"))

    def test_payouts_stop_at_budget(self):
        payouts, refund = jobs.compute_payouts({"0xa": 1000, "0xb": 1000}, {"0xa": Decimal("0.01"), "0xb": Decimal("0.01")}, Decimal("15"))
        self.assertEqual(sum(payouts.values()) + refund, Decimal("15"))
        self.assertEqual(refund, Decimal("0"))


if __name__ == "__main__":
    unittest.main()
