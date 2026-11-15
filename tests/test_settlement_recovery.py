"""Pending settlement proposals survive restart without trusting stale or changed data."""
import time
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from gpuchain_coordinator import jobs
from gpuchain_coordinator.models import Base, Job, Settlement, SettlementProposal
from gpuchain_coordinator.settlement_service import SettlementService, VerdictTask, verdict_from_inputs


class SettlementRecovery(unittest.TestCase):
    def setUp(self):
        # One shared connection, so the settlement thread sees the same in-memory database.
        self.engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            job = jobs.create_job(session, spec={"template": "synthetic-linear/v1"}, policy={}, budget=Decimal("5"), deadline_seconds=600)
            job.state = "FAILED"
            self.job_id = job.id
            session.commit()
        self.address, self.contract = "0x" + "1" * 40, "0x" + "2" * 40
        self.chain = Mock()
        self.chain.state.return_value = "Active"

    def service(self):
        return SettlementService(self.engine, None, self.chain, 31337, self.contract, {self.address}, 1, None)

    def test_restores_original_proposal_and_recollects_signatures(self):
        original = self.service()
        verdict = original.propose(self.job_id)
        self.assertEqual(original.propose(self.job_id), verdict)
        restarted = self.service()
        self.assertEqual(restarted.restore_pending(), 1)
        task = restarted.claim(self.address)
        self.assertEqual(task["job_id"], self.job_id)
        self.assertEqual(restarted.tasks[self.job_id].verdict, verdict)
        self.assertEqual(restarted.tasks[self.job_id].signatures, {})
        self.assertEqual(restarted.restore_pending(), 0)

    def test_expired_tampered_and_nonactive_proposals_are_not_restored(self):
        service = self.service()
        service.propose(self.job_id)
        with Session(self.engine) as session:
            row = session.scalar(select(SettlementProposal).where(SettlementProposal.job_id == self.job_id))
            original = dict(row.inputs)
            row.inputs = dict(original, expiry=int(time.time()) - 1)
            row_settlement = session.scalar(select(Settlement).where(Settlement.job_id == self.job_id))
            row_settlement.verdict = verdict_from_inputs(row.inputs)
            session.commit()
        # An expired but otherwise valid proposal is rebuilt with a fresh expiry, not dropped.
        rebuilt = self.service()
        self.assertEqual(rebuilt.restore_pending(), 1)
        self.assertGreater(rebuilt.tasks[self.job_id].inputs["expiry"], int(time.time()))
        with Session(self.engine) as session:
            row = session.scalar(select(SettlementProposal).where(SettlementProposal.job_id == self.job_id))
            row.inputs = dict(original, refund="4")
            session.commit()
        self.assertEqual(self.service().restore_pending(), 0)
        with Session(self.engine) as session:
            row = session.scalar(select(SettlementProposal).where(SettlementProposal.job_id == self.job_id))
            row.inputs = original
            session.commit()
        self.chain.state.return_value = "Attested"
        self.assertEqual(self.service().restore_pending(), 0)

    def test_chain_unavailable_does_not_restore(self):
        self.service().propose(self.job_id)
        self.chain.state.side_effect = OSError("offline")
        restarted = self.service()
        self.assertEqual(restarted.restore_pending(), 0)
        self.chain.state.side_effect = None
        self.assertEqual(restarted.claim(self.address)["job_id"], self.job_id)

    def test_passing_proposal_keeps_evidence_across_restart(self):
        with Session(self.engine) as session:
            job = session.get(Job, self.job_id)
            job.state = "COMPLETED"
            inputs = {"job_id": job.id, "onchain_job_id": job.onchain_id, "chain_id": 31337, "contract": self.contract, "passed": True, "expiry": int(time.time()) + 3600, "final_checkpoint_hash": "1" * 64, "lineage": ["1" * 64], "reports": [{"verdict": "PASS"}], "payouts": {self.address: "1"}, "refund": "4", "budget": "5", "payout_evidence": {"manifests": [{"signed": "checkpoint"}], "delegations": {}, "handover_reports": {}}, "job_binding": {"signature": "client-proof"}}
            session.add(Settlement(job_id=job.id, verdict=verdict_from_inputs(inputs), status="AWAITING_SIGNATURES"))
            session.add(SettlementProposal(job_id=job.id, inputs=inputs))
            session.commit()
        restarted = self.service()
        self.assertEqual(restarted.restore_pending(), 1)
        self.assertEqual(restarted.claim(self.address), inputs)

    def test_a_verdict_near_expiry_is_rebuilt_not_submitted(self):
        service = self.service()
        service.propose(self.job_id)
        task = service.tasks[self.job_id]
        task.verdict = dict(task.verdict, expiry=int(time.time()) + 100)
        task.signatures = {self.address: "signature"}
        self.chain.state.return_value = "Active"
        self.chain.state.side_effect = None
        service.settle(self.job_id)
        self.chain.ensure_attested.assert_not_called()
        fresh = service.tasks[self.job_id]
        self.assertGreater(fresh.verdict["expiry"], int(time.time()) + 3000)
        self.assertEqual(fresh.signatures, {})

    def test_our_own_attestation_with_a_lost_receipt_is_finalized(self):
        service = self.service()
        verdict = {"jobId": "0x" + "1" * 64, "finalCheckpoint": "0x" + "ab" * 32, "passed": True,
                   "payees": [self.address], "amounts": [1_000_000], "refund": 4_000_000}
        service.tasks[self.job_id] = VerdictTask(self.job_id, {}, verdict, signatures={self.address: "signature"})
        self.chain.ensure_attested.return_value = Mock(status="ALREADY_DONE", tx_hash=None)
        self.chain.job.return_value = {"state": "Attested", "final_checkpoint": "ab" * 32, "passed": True,
                                       "payees": [self.address.upper()], "amounts": [1_000_000], "refund": 4_000_000}
        service._set_status = Mock()
        service._emit_tx = Mock()
        service._finish_job = Mock()
        service.wait_for_window = Mock()
        service.orchestrator = Mock()
        with patch("gpuchain_coordinator.settlement_service.order_signatures", return_value=["signature"]):
            service.settle(self.job_id)
        self.chain.ensure_finalized.assert_called()
        self.assertNotIn(unittest.mock.call(self.job_id, "ERROR"), service._set_status.call_args_list)

    def _attested_row(self, passed: bool = True) -> dict:
        verdict = {"jobId": "0x" + "1" * 64, "finalCheckpoint": "0x" + "ab" * 32, "passed": passed,
                   "payees": [self.address], "amounts": [1_000_000], "refund": 4_000_000}
        with Session(self.engine) as session:
            session.get(Job, self.job_id).state = "COMPLETED" if passed else "FAILED"
            session.add(Settlement(job_id=self.job_id, verdict=verdict, submit_tx="0x" + "5" * 64, status="ATTESTED"))
            session.commit()
        self.chain.job.return_value = {"state": "Attested", "final_checkpoint": "ab" * 32, "passed": passed,
                                       "payees": [self.address], "amounts": [1_000_000], "refund": 4_000_000}
        return verdict

    def test_an_attestation_from_before_a_restart_is_finalized(self):
        self._attested_row()
        self.chain.seconds_until_dispute_end.return_value = 0
        self.chain.ensure_finalized.return_value = Mock(status="DONE", tx_hash="0x" + "6" * 64)
        restarted = self.service()
        restarted.orchestrator = Mock()
        self.assertEqual(restarted._resume_attested(wait=True), 1)
        self.chain.ensure_finalized.assert_called_once_with("0x" + "1" * 64)
        with Session(self.engine) as session:
            row = session.scalar(select(Settlement).where(Settlement.job_id == self.job_id))
            self.assertEqual((row.status, row.finalize_tx), ("SETTLED", "0x" + "6" * 64))
            self.assertEqual(session.get(Job, self.job_id).state, "SETTLED")
        self.assertEqual(restarted._resume_attested(wait=True), 0)

    def test_a_disputed_or_different_attestation_is_not_resumed(self):
        self._attested_row()
        self.chain.job.return_value = dict(self.chain.job.return_value, state="Disputed")
        self.assertEqual(self.service()._resume_attested(wait=True), 0)
        self.chain.job.return_value = dict(self.chain.job.return_value, state="Attested", amounts=[2_000_000])
        self.assertEqual(self.service()._resume_attested(wait=True), 0)
        self.chain.job.side_effect = OSError("offline")
        self.assertEqual(self.service()._resume_attested(wait=True), 0)
        self.chain.ensure_finalized.assert_not_called()

    def test_a_different_attestation_on_chain_is_not_finalized(self):
        service = self.service()
        verdict = {"jobId": "0x" + "1" * 64, "finalCheckpoint": "0x" + "ab" * 32, "passed": True}
        service.tasks[self.job_id] = VerdictTask(self.job_id, {}, verdict, signatures={self.address: "signature"})
        self.chain.ensure_attested.return_value = Mock(status="ALREADY_DONE", tx_hash=None)
        self.chain.job.return_value = {"state": "Attested", "final_checkpoint": "cd" * 32, "passed": True}
        service._set_status = Mock()
        service.orchestrator = Mock()
        with patch("gpuchain_coordinator.settlement_service.order_signatures", return_value=["signature"]):
            service.settle(self.job_id)
        self.chain.ensure_finalized.assert_not_called()
        service._set_status.assert_called_once_with(self.job_id, "ERROR")

    def test_same_checkpoint_with_different_payout_is_not_finalized(self):
        service = self.service()
        verdict = {"jobId": "0x" + "1" * 64, "finalCheckpoint": "0x" + "ab" * 32, "passed": True,
                   "payees": [self.address], "amounts": [1_000_000], "refund": 4_000_000}
        service.tasks[self.job_id] = VerdictTask(self.job_id, {}, verdict, signatures={self.address: "signature"})
        self.chain.ensure_attested.return_value = Mock(status="ALREADY_DONE", tx_hash=None)
        self.chain.job.return_value = {"state": "Attested", "final_checkpoint": "ab" * 32, "passed": True,
                                       "payees": ["0x" + "3" * 40], "amounts": [1_000_000], "refund": 4_000_000}
        service._set_status = Mock()
        service._emit_tx = Mock()
        service._finish_job = Mock()
        service.wait_for_window = Mock()
        service.orchestrator = Mock()
        with patch("gpuchain_coordinator.settlement_service.order_signatures", return_value=["signature"]):
            service.settle(self.job_id)
        self.chain.ensure_finalized.assert_not_called()
        service._set_status.assert_called_once_with(self.job_id, "ERROR")

    def test_attestation_reconciliation_rejects_missing_or_changed_split(self):
        service = self.service()
        verdict = {"jobId": "0x" + "1" * 64, "finalCheckpoint": "0x" + "ab" * 32, "passed": True,
                   "payees": [self.address], "amounts": [1_000_000], "refund": 4_000_000}
        matching = {"state": "Attested", "final_checkpoint": "ab" * 32, "passed": True,
                    "payees": [self.address], "amounts": [1_000_000], "refund": 4_000_000}
        for change in ({"amounts": [2_000_000]}, {"refund": 3_000_000}, {"payees": []}):
            with self.subTest(change=change):
                self.chain.job.return_value = {**matching, **change}
                self.assertFalse(service._attested_matches(verdict))
        self.chain.job.return_value = {key: value for key, value in matching.items() if key != "refund"}
        self.assertFalse(service._attested_matches(verdict))

    def test_existing_attestation_is_not_blindly_finalized(self):
        service = self.service()
        service.tasks[self.job_id] = VerdictTask(self.job_id, {}, {"jobId": "0x" + "1" * 64}, signatures={self.address: "signature"})
        self.chain.ensure_attested.return_value = Mock(status="ALREADY_DONE")
        service._set_status = Mock()
        service.orchestrator = Mock()
        with patch("gpuchain_coordinator.settlement_service.order_signatures", return_value=["signature"]):
            service.settle(self.job_id)
        self.chain.ensure_finalized.assert_not_called()
        service._set_status.assert_called_once_with(self.job_id, "ERROR")
