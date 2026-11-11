import threading
import unittest
from unittest.mock import patch

from eth_account import Account

from gpuchain_core.signing import sign_document
from gpuchain_coordinator.quality_service import QualityTimeout, QuorumQuality, quorum_decision

KEYS = ["0x" + c * 64 for c in "123"]
ADDRESSES = [Account.from_key(k).address for k in KEYS]
OUTSIDER = "0x" + "9" * 64
CHECKPOINT = "a" * 64


def report(key, verdict="PASS", checkpoint=CHECKPOINT):
    body = {"job_id": "job", "reference_checkpoint_hash": None, "is_final": True, "checkpoint_hash": checkpoint, "verdict": verdict, "signer": Account.from_key(key).address, "device": "cpu"}
    _, body["signature"] = sign_document(body, key)
    return body


def service(quorum=2, timeout=5.0):
    return QuorumQuality(set(ADDRESSES), quorum, timeout)


class Decision(unittest.TestCase):
    allowed = {a.lower() for a in ADDRESSES}

    def test_needs_quorum(self):
        self.assertIsNone(quorum_decision([report(KEYS[0])], CHECKPOINT, 2, self.allowed))
        self.assertEqual(quorum_decision([report(KEYS[0]), report(KEYS[1])], CHECKPOINT, 2, self.allowed), "PASS")

    def test_one_verifier_cannot_count_twice(self):
        self.assertIsNone(quorum_decision([report(KEYS[0]), report(KEYS[0])], CHECKPOINT, 2, self.allowed))

    def test_outsider_does_not_count(self):
        self.assertIsNone(quorum_decision([report(KEYS[0]), report(OUTSIDER)], CHECKPOINT, 2, self.allowed))

    def test_tampered_report_does_not_count(self):
        forged = report(KEYS[1], "FAIL")
        forged["verdict"] = "PASS"
        self.assertIsNone(quorum_decision([report(KEYS[0]), forged], CHECKPOINT, 2, self.allowed))

    def test_other_checkpoint_does_not_count(self):
        self.assertIsNone(quorum_decision([report(KEYS[0]), report(KEYS[1], checkpoint="b" * 64)], CHECKPOINT, 2, self.allowed))


class Flow(unittest.TestCase):
    def run_evaluation(self, svc):
        result = {}

        def target():
            try:
                result["verdict"] = svc.evaluate("job", CHECKPOINT, None, is_final=True)
            except QualityTimeout as exc:
                result["error"] = exc

        thread = threading.Thread(target=target)
        thread.start()
        while not svc.tasks and thread.is_alive():
            pass
        return thread, result

    def test_two_verifiers_decide(self):
        svc = service()
        thread, result = self.run_evaluation(svc)
        for key, address in zip(KEYS[:2], ADDRESSES[:2]):
            task = svc.claim(address)
            self.assertEqual(task["checkpoint_hash"], CHECKPOINT)
            svc.submit(task["task_id"], report(key))
        thread.join(5)
        self.assertEqual(result["verdict"], "PASS")

    def test_verifiers_can_disagree_and_third_breaks_the_tie(self):
        svc = service()
        thread, result = self.run_evaluation(svc)
        for key, address, verdict in zip(KEYS, ADDRESSES, ("PASS", "FAIL", "FAIL")):
            svc.submit(svc.claim(address)["task_id"], report(key, verdict))
        thread.join(5)
        self.assertEqual(result["verdict"], "FAIL")

    def test_missing_verifiers_never_become_a_pass(self):
        svc = service(timeout=0.2)
        thread, result = self.run_evaluation(svc)
        thread.join(5)
        self.assertIsInstance(result.get("error"), QualityTimeout)
        self.assertNotIn("verdict", result)

    def test_unloadable_checkpoint_counts_as_failure(self):
        svc = service()
        thread, result = self.run_evaluation(svc)
        for key, address in zip(KEYS[:2], ADDRESSES[:2]):
            svc.submit(svc.claim(address)["task_id"], report(key, "FAIL", checkpoint="0" * 64))
        thread.join(5)
        self.assertEqual(result["verdict"], "FAIL")

    def test_zero_hash_cannot_vote_pass(self):
        svc = service(timeout=0.3)
        thread, result = self.run_evaluation(svc)
        for key, address in zip(KEYS[:2], ADDRESSES[:2]):
            svc.submit(svc.claim(address)["task_id"], report(key, "PASS", checkpoint="0" * 64))
        thread.join(5)
        self.assertIsInstance(result.get("error"), QualityTimeout)

    def test_outsider_cannot_claim_or_submit(self):
        svc = service()
        thread, _ = self.run_evaluation(svc)
        with self.assertRaises(PermissionError):
            svc.claim(Account.from_key(OUTSIDER).address)
        task_id = next(iter(svc.tasks))
        with self.assertRaises(PermissionError):
            svc.submit(task_id, report(OUTSIDER))
        forged = report(KEYS[0])
        forged["verdict"] = "FAIL"
        with self.assertRaises(PermissionError):
            svc.submit(task_id, forged)
        for key in KEYS[:2]:
            svc.submit(task_id, report(key))
        thread.join(5)

    def test_report_for_wrong_checkpoint_refused(self):
        svc = service()
        thread, _ = self.run_evaluation(svc)
        task_id = next(iter(svc.tasks))
        with self.assertRaises(ValueError):
            svc.submit(task_id, report(KEYS[0], checkpoint="b" * 64))
        svc.submit(task_id, report(KEYS[0]))
        svc.submit(task_id, report(KEYS[1]))
        thread.join(5)

    def test_wrong_context_and_duplicate_vote_are_refused(self):
        svc = service()
        thread, _ = self.run_evaluation(svc)
        task_id = next(iter(svc.tasks))
        wrong = report(KEYS[0])
        wrong["job_id"] = "other"
        _, wrong["signature"] = sign_document(wrong, KEYS[0])
        with self.assertRaises(ValueError):
            svc.submit(task_id, wrong)
        svc.submit(task_id, report(KEYS[0]))
        with self.assertRaises(ValueError):
            svc.submit(task_id, report(KEYS[0]))
        svc.submit(task_id, report(KEYS[1]))
        thread.join(5)

    def test_persistence_failure_never_publishes_decision(self):
        svc = service()
        thread, result = self.run_evaluation(svc)
        task_id = next(iter(svc.tasks))
        svc.submit(task_id, report(KEYS[0]))
        def fail(*_args):
            raise OSError("database unavailable")
        svc.on_decision = fail
        with self.assertRaises(OSError):
            svc.submit(task_id, report(KEYS[1]))
        self.assertFalse(svc.tasks[task_id].done.is_set())
        self.assertNotIn(("job", CHECKPOINT), svc.decided_reports)
        svc.on_decision = None
        svc.submit(task_id, report(KEYS[1]))
        thread.join(5)
        self.assertEqual(result["verdict"], "PASS")

    def test_unfinished_claim_expires_but_submitted_report_is_not_reclaimed(self):
        svc = service()
        thread, _ = self.run_evaluation(svc)
        with patch("gpuchain_coordinator.quality_service.monotonic", return_value=100):
            first = svc.claim(ADDRESSES[0])
            self.assertIsNone(svc.claim(ADDRESSES[0]))
        with patch("gpuchain_coordinator.quality_service.monotonic", return_value=280):
            self.assertEqual(svc.claim(ADDRESSES[0]), first)
            svc.submit(first["task_id"], report(KEYS[0]))
        with patch("gpuchain_coordinator.quality_service.monotonic", return_value=1000):
            self.assertIsNone(svc.claim(ADDRESSES[0]))
        svc.submit(first["task_id"], report(KEYS[1]))
        thread.join(5)

    def test_verifier_gets_each_task_once(self):
        svc = service()
        thread, _ = self.run_evaluation(svc)
        self.assertIsNotNone(svc.claim(ADDRESSES[0]))
        self.assertIsNone(svc.claim(ADDRESSES[0]))
        task_id = next(iter(svc.tasks))
        svc.submit(task_id, report(KEYS[0]))
        svc.submit(task_id, report(KEYS[1]))
        thread.join(5)

    def test_late_report_changes_nothing(self):
        svc = service()
        thread, result = self.run_evaluation(svc)
        task_id = next(iter(svc.tasks))
        svc.submit(task_id, report(KEYS[0]))
        svc.submit(task_id, report(KEYS[1]))
        thread.join(5)
        self.assertIsNone(svc.submit(task_id, report(KEYS[2], "FAIL")))
        self.assertEqual(result["verdict"], "PASS")

    def test_invalid_quorum_refused(self):
        for bad in (0, 4):
            with self.assertRaises(ValueError):
                QuorumQuality(set(ADDRESSES), bad)


if __name__ == "__main__":
    unittest.main()
