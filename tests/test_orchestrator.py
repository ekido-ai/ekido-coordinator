"""The demo's story, against a fake GPU network. Real compiler, scheduler, database and fencing."""

import os
import unittest
from decimal import Decimal
from unittest import mock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ekido_core.compiler import ProbeResult
from ekido_core.canonical import document_hash
from ekido_coordinator import jobs
from ekido_coordinator.events import EventBus, events_after
from ekido_coordinator.models import AdaptationManifest, Base, Checkpoint, GpuHardware, Job, JobAssignment, Provider, ProviderReputation, RecoveryEvent, Wallet
from ekido_coordinator.orchestrator import MAX_ATTEMPTS, Orchestrator

SPEC = {"template": "synthetic-linear/v1", "logical_batch_size": 32, "total_logical_steps": 300, "seed": 1}
POLICY = {"metric": "accuracy"}


class FakeNetwork:
    """Memory need is fixed per example, so a capped provider really does need a smaller micro-batch."""

    def __init__(self):
        self.connected = set()
        self.started = []
        self.stopped = []
        self.fail_start = set()
        self.speed = 10.0

    def is_connected(self, provider_id):
        return provider_id in self.connected

    def probe(self, provider_id, config):
        needed = 300 + 100 * config["exec"]["micro_batch"]
        if needed > config["capacity_mib"]:
            return ProbeResult(ok=False, oom=True)
        return ProbeResult(ok=True, peak_mem_mib=needed, steps_per_s=self.speed, duration_s=1.0)

    def start_run(self, provider_id, config):
        if provider_id in self.fail_start:
            raise ConnectionError("agent unreachable")
        self.started.append((provider_id, config))

    def stop_run(self, provider_id, job_id, token):
        self.stopped.append((provider_id, token))


class FakeQuality:
    def __init__(self):
        self.answers = {}
        self.calls = []

    def evaluate(self, job_id, checkpoint_hash, reference_hash, *, is_final):
        self.calls.append((checkpoint_hash, reference_hash, is_final))
        return self.answers.get(checkpoint_hash, "PASS")


class Fixture(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.net = FakeNetwork()
        self.quality = FakeQuality()
        self.bus = EventBus()
        self.orch = Orchestrator(self.engine, self.bus, self.net, self.quality)
        with Session(self.engine) as s:
            gpu = GpuHardware(model="RTX 3060 Laptop", vram_mib=6144, compute_cap="8.6")
            s.add(gpu)
            s.flush()
            self.a = self._provider(s, gpu, "a-full", "0x" + "a" * 40, 6144, "0.002", "REAL")
            self.b = self._provider(s, gpu, "b-capped", "0x" + "b" * 40, 1200, "0.003", "CAPPED_PROFILE")
            self.c = self._provider(s, gpu, "c-tiny", "0x" + "c" * 40, 350, "0.001", "CAPPED_PROFILE")
            job = jobs.create_job(s, spec=SPEC, policy=POLICY, budget=Decimal("5"), deadline_seconds=3600)
            self.job_id = job.id
            s.commit()
        self.net.connected = {self.a, self.b, self.c}

    def _provider(self, s, gpu, name, key, capacity, price, kind):
        wallet = Wallet(address=key, role="provider")
        s.add(wallet)
        s.flush()
        provider = Provider(wallet_id=wallet.id, name=name, kind=kind, physical_gpu_id=gpu.id, capacity_mib=capacity, price_per_step=Decimal(price), enrolled_key=key)
        s.add(provider)
        s.flush()
        return provider.id

    def job(self):
        with Session(self.engine) as s:
            job = s.get(Job, self.job_id)
            s.expunge(job)
            return job

    def state(self):
        return self.job().state

    def events(self, event_type=None):
        with Session(self.engine) as s:
            found = events_after(s, self.job_id, 0)
        return [e for e in found if event_type is None or e["type"] == event_type]

    def manifest(self, step, parent=None, note=""):
        job = self.job()
        return {"logical_step": step, "fencing_token": job.fencing_token, "immutable_hash": job.spec_hash, "parent_manifest_hash": parent, "files": {"t": {"bytes": 5}}, "note": note}

    def checkpoint(self, step, signer, parent=None, note=""):
        token = self.job().fencing_token
        return self.orch.on_checkpoint(self.job_id, self.manifest(step, parent, note), fencing_token=token, signer=signer)

    def started_job(self):
        self.orch.start_job(self.job_id)
        self.orch.on_run_started(self.job_id, self.job().fencing_token)
        return self.net.started[-1]


class Placement(Fixture):
    def test_accepted_work_uses_agreed_rate_after_provider_changes_price(self):
        with Session(self.engine) as session:
            job = session.get(Job, self.job_id)
            job.spec = dict(job.spec, provider_terms=[{"provider_id": self.a, "wallet": "0x" + "a" * 40, "price_per_step": "0.002"}])
            session.commit()
        self.started_job()
        tip = self.checkpoint(100, "0x" + "a" * 40)
        with Session(self.engine) as session:
            cp = session.scalar(select(Checkpoint).where(Checkpoint.manifest_hash == tip.manifest_hash))
            cp.state = "ACCEPTED"
            session.get(Provider, self.a).price_per_step = Decimal("0.001")
            session.commit()
            self.assertEqual(self.orch._spent(session, session.get(Job, self.job_id)), Decimal("0.2"))
            with mock.patch("ekido_coordinator.orchestrator.job_service.providers_by_signer", return_value={}):
                with self.assertRaisesRegex(ValueError, "no agreed provider rate"):
                    self.orch._spent(session, session.get(Job, self.job_id))

    def test_secure_job_only_places_preagreed_provider(self):
        with Session(self.engine) as session:
            job = session.get(Job, self.job_id)
            job.spec = dict(job.spec, provider_terms=[{"provider_id": self.b, "wallet": "0x" + "b" * 40, "price_per_step": "0.003"}])
            session.commit()
        provider, _ = self.started_job()
        self.assertEqual(provider, self.b)

    def test_secure_job_does_not_place_provider_after_rate_changes(self):
        with Session(self.engine) as session:
            job = session.get(Job, self.job_id)
            job.spec = dict(job.spec, provider_terms=[{"provider_id": self.b, "wallet": "0x" + "b" * 40, "price_per_step": "0.004"}])
            session.commit()
        self.assertEqual(self.orch.start_job(self.job_id), "OPEN")
        self.assertEqual(self.net.started, [])

    def test_start_runs_on_best_eligible_provider(self):
        provider, config = self.started_job()
        self.assertEqual(self.state(), "RUNNING")
        # c is cheapest but can't fit even one example, and a beats b on price.
        self.assertEqual(provider, self.a)
        self.assertEqual(config["exec"]["micro_batch"], 32)  # Needs 3500 MiB, under 85% of 6144.
        self.assertEqual(config["fencing_token"], 1)

    def test_start_is_repeatable(self):
        self.orch.start_job(self.job_id)
        self.orch.start_job(self.job_id)
        self.assertEqual(len(self.net.started), 1)

    def test_no_provider_leaves_job_open_with_a_warning(self):
        self.net.connected = set()
        self.assertEqual(self.orch.start_job(self.job_id), "OPEN")
        self.assertEqual(self.events("failure.warning")[0]["data"]["rule"], "NO_PROVIDER")

    def test_every_decision_is_visible_with_a_reason(self):
        self.orch.start_job(self.job_id)
        decisions = {e["data"]["provider_id"]: e["data"] for e in self.events("compile.decision")}
        self.assertEqual(decisions[self.c]["decision"], "REJECTED")
        self.assertEqual(decisions[self.c]["code"], "PROBE_OOM")
        self.assertEqual(decisions[self.a]["decision"], "ACCEPTED")


class Recovery(Fixture):
    def fail(self, **kw):
        return self.orch.on_failure(self.job_id, provider_id=self.a, cause="PROCESS_EXIT", injected=True, **kw)

    def test_kill_moves_to_adapted_replacement_from_last_checkpoint(self):
        self.started_job()
        tip = self.checkpoint(100, "0x" + "a" * 40)
        self.fail(last_step=110, fencing_token=1)

        provider, config = self.net.started[-1]
        self.assertEqual(provider, self.b)
        self.assertEqual(config["resume_from_hash"], tip.manifest_hash)
        self.assertEqual(config["exec"]["micro_batch"] * config["exec"]["accumulation"], 32)
        self.assertLess(config["exec"]["micro_batch"], 32)
        self.assertEqual(config["fencing_token"], 2)
        self.assertEqual(self.state(), "RECOVERING")

        self.orch.on_run_started(self.job_id, 2)
        self.assertEqual(self.state(), "RUNNING")

    def test_recovery_stages_run_in_order(self):
        self.started_job()
        self.checkpoint(100, "0x" + "a" * 40)
        self.fail(last_step=110, fencing_token=1)
        self.orch.on_run_started(self.job_id, 2)
        stages = [e["data"]["stage"] for e in self.events("recovery.step")]
        self.assertEqual(stages, ["DETECTED", "FENCED", "HANDOVER_EVAL", "SELECT_CHECKPOINT", "FIND_CANDIDATES", "COMPILE", "ASSIGN", "RESTORE", "RESUMED"])

    def test_events_say_who_and_which_attempt(self):
        """A dashboard needs these without guessing from the order events arrive in."""
        self.started_job()
        tip = self.checkpoint(100, "0x" + "a" * 40)
        self.assertEqual(self.events("checkpoint.committed")[0]["data"]["provider_id"], self.a)
        self.fail(last_step=110, fencing_token=1)

        steps = {e["data"]["stage"]: e["data"] for e in self.events("recovery.step")}
        self.assertEqual(steps["DETECTED"]["provider_id"], self.a)
        self.assertEqual(steps["ASSIGN"]["provider_id"], self.b)
        self.assertEqual((steps["SELECT_CHECKPOINT"]["checkpoint_hash"], steps["SELECT_CHECKPOINT"]["resume_step"]), (tip.manifest_hash, 100))
        self.assertEqual(steps["ASSIGN"]["fencing_token"], 2)

        replacement = [e["data"] for e in self.events("compile.decision") if e["data"]["attempt"] == self.job().attempt and e["data"]["provider_id"] == self.b][-1]
        self.assertEqual(replacement["from_exec"]["micro_batch"], 32)
        self.assertEqual(replacement["capacity_mib"], 1200)
        self.assertLessEqual(replacement["peak_mem_mib"], 1200)
        self.assertGreater(replacement["steps_per_s"], 0)
        first = next(e["data"] for e in self.events("compile.decision") if e["data"]["provider_id"] == self.a)
        self.assertIsNone(first["from_exec"], "the first placement has nothing to adapt from")

    def test_rejection_names_the_provider_it_refused(self):
        self.started_job()
        tip = self.checkpoint(100, "0x" + "a" * 40)
        self.orch.on_candidate_rejected(self.job_id, tip.manifest_hash, None, ["FAILED_QUALITY"])
        self.assertEqual(self.events("checkpoint.rejected")[0]["data"]["provider_id"], self.a)
        fenced = next(e["data"] for e in self.events("recovery.step") if e["data"]["stage"] == "FENCED")
        self.assertEqual(fenced["provider_id"], self.a)

    def test_cpu_fallback_events_are_labelled_synthetic(self):
        self.assertEqual(self.orch.start_job(self.job_id), "ASSIGNED")
        self.assertEqual({e["provenance"] for e in self.events()}, {"LIVE"})
        with mock.patch.dict(os.environ, {"EVENT_PROVENANCE": "SYNTHETIC"}):
            self.orch.on_run_started(self.job_id, 1)
        self.assertEqual(self.events("job.state_changed")[-1]["provenance"], "SYNTHETIC")

    def test_failure_event_is_labelled_injected_and_lost_steps_recorded(self):
        self.started_job()
        self.checkpoint(100, "0x" + "a" * 40)
        self.fail(last_step=110, fencing_token=1)
        self.assertTrue(self.events("failure.detected")[0]["data"]["injected"])
        with Session(self.engine) as s:
            record = s.scalar(select(RecoveryEvent))
            self.assertEqual((record.lost_steps, record.injected, record.cause), (10, True, "PROCESS_EXIT"))
            self.assertIsNotNone(record.fenced_at)

    def test_old_worker_is_fenced_and_told_to_stop(self):
        self.started_job()
        old_manifest = self.manifest(100)
        self.fail(fencing_token=1)
        self.assertEqual(self.net.stopped, [(self.a, 1)])
        with self.assertRaises(jobs.StaleFencingToken):
            self.orch.on_checkpoint(self.job_id, old_manifest, fencing_token=1, signer="0x" + "a" * 40)

    def test_unreachable_old_provider_does_not_stop_recovery(self):
        self.started_job()
        self.net.stop_run = lambda *a: (_ for _ in ()).throw(ConnectionError("gone"))
        self.fail(fencing_token=1)
        self.assertEqual(self.net.started[-1][0], self.b)

    def test_failed_provider_is_not_chosen_again(self):
        self.started_job()
        self.fail(fencing_token=1)
        self.assertNotEqual(self.net.started[-1][0], self.a)

    def test_duplicate_failure_report_does_not_recover_twice(self):
        self.started_job()
        self.fail(fencing_token=1)
        count = len(self.net.started)
        self.fail(fencing_token=1)
        self.assertEqual(len(self.net.started), count)
        self.assertEqual(self.job().fencing_token, 2)

    def test_stale_report_cannot_kill_the_replacement(self):
        self.started_job()
        self.fail(fencing_token=1)
        self.orch.on_run_started(self.job_id, 2)
        self.fail(fencing_token=1)  # The old attempt's exit notice, arriving late.
        self.assertEqual(self.state(), "RUNNING")

    def test_failure_before_any_checkpoint_restarts_from_zero(self):
        self.started_job()
        self.fail(fencing_token=1)
        self.assertIsNone(self.net.started[-1][1]["resume_from_hash"])

    def test_no_replacement_fails_the_job_with_reasons(self):
        self.started_job()
        self.net.connected = {self.a}
        self.assertEqual(self.fail(fencing_token=1), "FAILED")
        last = self.events("recovery.step")[-1]["data"]
        self.assertEqual(last["stage"], "ABORT")
        self.assertIn("OFFLINE", last["detail"])

    def test_replacement_that_cannot_start_falls_through_to_next(self):
        self.started_job()
        with Session(self.engine) as s:
            s.get(Provider, self.c).capacity_mib = 1500  # c now fits, and being cheapest it's tried first.
            s.commit()
        self.net.fail_start = {self.c}
        self.fail(fencing_token=1)
        self.assertEqual(self.net.started[-1][0], self.b)
        with Session(self.engine) as s:
            states = {a.provider_id: a.state for a in s.scalars(select(JobAssignment))}
        self.assertEqual(states[self.c], "FAILED")

    def test_too_slow_replacement_is_refused(self):
        self.started_job()
        self.net.speed = 0.0001
        self.assertEqual(self.fail(fencing_token=1), "FAILED")
        self.assertIn("DEADLINE", self.events("recovery.step")[-1]["data"]["detail"])

    def test_adaptation_manifest_is_stored_and_passed_to_the_run(self):
        self.started_job()
        self.checkpoint(100, "0x" + "a" * 40)
        self.fail(fencing_token=1)
        config = self.net.started[-1][1]
        with Session(self.engine) as s:
            row = s.scalar(select(AdaptationManifest))
            self.assertEqual(row.manifest_hash, config["adaptation_manifest_hash"])
            self.assertEqual(row.manifest_hash, document_hash(row.manifest))
            self.assertEqual(row.manifest["to"]["provider_id"], self.b)
            self.assertEqual({r["provider_id"] for r in row.manifest["rejected_candidates"]}, {self.c})
            self.assertTrue(any(c["field"] == "micro_batch" for c in row.manifest["changes"]))

    def test_gives_up_after_too_many_attempts(self):
        self.started_job()
        with Session(self.engine) as s:
            s.get(Job, self.job_id).attempt = MAX_ATTEMPTS
            s.commit()
        self.assertEqual(self.fail(), "FAILED")


class Qualification(Fixture):
    def count_probes(self):
        calls = []
        original = self.net.probe

        def counting(provider_id, config):
            calls.append(provider_id)
            return original(provider_id, config)

        self.net.probe = counting
        return calls

    def test_reports_each_provider(self):
        results = {r["name"]: r for r in self.orch.qualify_providers(SPEC)}
        self.assertTrue(results["a-full"]["approved"])
        self.assertEqual(results["b-capped"]["exec"]["micro_batch"], 4)  # 700 MiB, under 85% of 1200.
        self.assertFalse(results["c-tiny"]["approved"])
        self.assertEqual(results["c-tiny"]["code"], "PROBE_OOM")

    def test_recovery_after_qualifying_needs_only_one_probe(self):
        self.orch.qualify_providers(SPEC)
        self.started_job()
        calls = self.count_probes()
        self.orch.on_failure(self.job_id, provider_id=self.a, cause="PROCESS_EXIT", fencing_token=1)
        self.assertEqual(calls, [self.b], "the chosen provider gets one fresh probe, the refused one none")
        self.assertEqual(self.net.started[-1][0], self.b)

    def test_remembered_refusal_says_it_was_measured_earlier(self):
        self.orch.qualify_providers(SPEC)
        self.started_job()
        reasons = [e["data"]["reason"] for e in self.events("compile.decision") if e["data"]["provider_id"] == self.c]
        self.assertIn("measured earlier", reasons[-1])

    def test_a_fresh_probe_can_overrule_old_results(self):
        self.orch.qualify_providers(SPEC)
        with Session(self.engine) as s:
            s.get(Provider, self.a).capacity_mib = 800  # What was measured no longer holds.
            s.commit()
        self.orch.start_job(self.job_id)
        config = self.net.started[-1][1]
        self.assertLess(config["exec"]["micro_batch"], 32)

    def test_only_the_winner_gets_a_fresh_probe(self):
        self.orch.qualify_providers(SPEC)
        calls = self.count_probes()
        self.orch.start_job(self.job_id)
        self.assertEqual(calls, [self.a])

    def test_winner_failing_its_fresh_probe_falls_to_the_next(self):
        self.orch.qualify_providers(SPEC)
        original = self.net.probe

        def a_is_now_broken(provider_id, config):
            return ProbeResult(ok=False, oom=True) if provider_id == self.a else original(provider_id, config)

        self.net.probe = a_is_now_broken
        self.orch.start_job(self.job_id)
        self.assertEqual(self.net.started[-1][0], self.b)
        decisions = [e["data"] for e in self.events("compile.decision") if e["data"]["provider_id"] == self.a]
        self.assertEqual(decisions[-1]["decision"], "REJECTED")

    def test_reconnecting_clears_old_results(self):
        self.orch.qualify_providers(SPEC)
        self.orch.forget_provider(self.c)
        calls = self.count_probes()
        self.orch.start_job(self.job_id)
        self.assertIn(self.c, calls)

    def test_different_workload_is_measured_separately(self):
        self.orch.qualify_providers(SPEC)
        calls = self.count_probes()
        self.orch.qualify_providers(dict(SPEC, logical_batch_size=16))
        self.assertTrue(calls)

    def test_offline_provider_is_not_remembered_as_refused(self):
        self.net.connected = {self.a, self.b}
        self.orch.qualify_providers(SPEC)
        self.net.connected.add(self.c)
        with Session(self.engine) as s:
            s.get(Provider, self.c).capacity_mib = 6144
            s.commit()
        self.orch.start_job(self.job_id)
        decisions = {e["data"]["provider_id"]: e["data"] for e in self.events("compile.decision")}
        self.assertEqual(decisions[self.c]["decision"], "ACCEPTED")


class Probation(Fixture):
    A, B = "0x" + "a" * 40, "0x" + "b" * 40

    def replacement_running(self):
        """A trained to step 100 and died. B is now running at token 2."""
        self.started_job()
        self.tip = self.checkpoint(100, self.A)
        self.orch.on_failure(self.job_id, provider_id=self.a, cause="PROCESS_EXIT", fencing_token=1)
        self.orch.on_run_started(self.job_id, 2)
        with Session(self.engine) as s:
            s.get(Provider, self.c).capacity_mib = 1500  # So a further replacement exists.
            s.commit()

    def probe_checkpoint(self, step, parent=None):
        checkpoint = self.checkpoint(step, self.B, parent=parent or self.tip.manifest_hash)
        return checkpoint, self.orch.check_probation(self.job_id, checkpoint, 2)

    def test_too_early_is_not_evaluated(self):
        self.replacement_running()
        _, thread = self.probe_checkpoint(125)
        self.assertIsNone(thread)

    def test_first_job_attempt_needs_no_probation(self):
        self.started_job()
        checkpoint = self.checkpoint(100, self.A)
        self.assertIsNone(self.orch.check_probation(self.job_id, checkpoint, 1))

    def test_good_replacement_is_accepted_and_keeps_running(self):
        self.replacement_running()
        checkpoint, thread = self.probe_checkpoint(150)
        thread.join(5)
        self.assertEqual(self.state(), "RUNNING")
        with Session(self.engine) as s:
            self.assertEqual(s.get(Checkpoint, checkpoint.id).state, "ACCEPTED")

    def test_bad_replacement_is_rejected_and_replaced(self):
        self.replacement_running()
        self.quality.answers = {}
        started = len(self.net.started)
        bad = self.checkpoint(150, self.B, parent=self.tip.manifest_hash)
        self.quality.answers[bad.manifest_hash] = "FAIL"
        thread = self.orch.check_probation(self.job_id, bad, 2)
        thread.join(5)
        self.assertEqual(len(self.net.started), started + 1)
        provider, config = self.net.started[-1]
        self.assertNotEqual(provider, self.b)
        self.assertEqual(config["resume_from_hash"], self.tip.manifest_hash)
        with Session(self.engine) as s:
            self.assertEqual(s.get(Checkpoint, bad.id).state, "REJECTED")

    def test_only_one_evaluation_per_attempt(self):
        self.replacement_running()
        first, thread = self.probe_checkpoint(150)
        thread.join(5)
        _, second = self.probe_checkpoint(175, parent=first.manifest_hash)
        self.assertIsNone(second)

    def test_answer_for_a_replaced_worker_changes_nothing(self):
        self.replacement_running()
        checkpoint = self.checkpoint(150, self.B, parent=self.tip.manifest_hash)
        self.quality.answers[checkpoint.manifest_hash] = "FAIL"
        with Session(self.engine) as s:
            jobs.bump_fencing(s, s.get(Job, self.job_id))
            s.commit()
        started = len(self.net.started)
        self.orch._run_probation(self.job_id, checkpoint.manifest_hash, None, 2)
        self.assertEqual(len(self.net.started), started)

    def test_no_quorum_leaves_it_unaccepted_and_running(self):
        self.replacement_running()
        checkpoint = self.checkpoint(150, self.B, parent=self.tip.manifest_hash)
        self.quality.evaluate = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("no quorum"))
        self.orch._run_probation(self.job_id, checkpoint.manifest_hash, None, 2)
        self.assertEqual(self.state(), "RUNNING")
        with Session(self.engine) as s:
            self.assertEqual(s.get(Checkpoint, checkpoint.id).state, "VERIFIED")


class QualityGuard(Fixture):
    A, B, P = "0x" + "a" * 40, "0x" + "b" * 40, "0x" + "c" * 40

    def test_handover_timeout_does_not_reject_checkpoint(self):
        self.started_job()
        accepted = self.checkpoint(50, self.A)
        pending = self.checkpoint(100, self.A, parent=accepted.manifest_hash)
        with Session(self.engine) as session:
            session.get(Checkpoint, accepted.id).state = "ACCEPTED"
            session.commit()
        self.quality.evaluate = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("no quorum"))
        self.orch.on_failure(self.job_id, provider_id=self.a, cause="PROCESS_EXIT", fencing_token=1)
        self.assertEqual(self.net.started[-1][1]["resume_from_hash"], accepted.manifest_hash)
        with Session(self.engine) as session:
            self.assertEqual(session.get(Checkpoint, pending.id).state, "VERIFIED")
        self.assertEqual(self.events("checkpoint.rejected"), [])
        self.assertEqual(self.events("failure.warning")[-1]["data"]["rule"], "NO_VERIFIER_QUORUM")

    def test_handover_rejection_falls_back_to_older_checkpoint(self):
        self.started_job()
        good = self.checkpoint(50, self.A)
        bad = self.checkpoint(100, self.A, parent=good.manifest_hash)
        self.quality.answers[bad.manifest_hash] = "FAIL"
        self.orch.on_failure(self.job_id, provider_id=self.a, cause="PROCESS_EXIT", fencing_token=1)
        self.assertEqual(self.net.started[-1][1]["resume_from_hash"], good.manifest_hash)
        with Session(self.engine) as s:
            self.assertEqual(s.get(Checkpoint, bad.id).state, "REJECTED")

    def test_rejected_candidate_rolls_back_and_excludes_provider(self):
        self.started_job()
        good = self.checkpoint(100, self.A)
        with Session(self.engine) as s:
            s.get(Checkpoint, good.id).state = "ACCEPTED"
            s.commit()
        self.orch.on_failure(self.job_id, provider_id=self.a, cause="PROCESS_EXIT", fencing_token=1)
        cheat, _ = self.net.started[-1]
        self.orch.on_run_started(self.job_id, 2)
        bad = self.checkpoint(150, "0x" + "b" * 40, parent=good.manifest_hash)

        with Session(self.engine) as s:
            s.get(Provider, self.c).capacity_mib = 1500
            s.commit()
        self.orch.on_candidate_rejected(self.job_id, bad.manifest_hash, cheat, ["NON_INFERIORITY_FAILED"])

        provider, config = self.net.started[-1]
        self.assertNotEqual(provider, cheat)
        self.assertEqual(config["resume_from_hash"], good.manifest_hash)
        with Session(self.engine) as s:
            self.assertEqual(s.get(Checkpoint, bad.id).state, "REJECTED")
            reputation = s.scalar(select(ProviderReputation).where(ProviderReputation.provider_id == cheat))
            self.assertEqual(reputation.rejected_candidates, 1)
        self.assertEqual(self.events("checkpoint.rejected")[0]["data"]["reason"], "NON_INFERIORITY_FAILED")

    def test_descendants_of_a_rejected_checkpoint_are_rejected_too(self):
        self.started_job()
        a = self.checkpoint(50, self.A)
        b = self.checkpoint(100, self.A, parent=a.manifest_hash)
        c = self.checkpoint(150, self.A, parent=b.manifest_hash)
        self.orch.on_candidate_rejected(self.job_id, b.manifest_hash, self.a, ["X"])
        with Session(self.engine) as s:
            self.assertEqual([s.get(Checkpoint, x.id).state for x in (b, c)], ["REJECTED", "REJECTED"])

    def test_finished_run_with_passing_evaluation_completes(self):
        self.started_job()
        self.checkpoint(300, self.A)
        self.assertEqual(self.orch.on_run_finished(self.job_id, 1), "COMPLETED")
        self.assertEqual(self.quality.calls[-1][2], True)
        self.assertEqual(self.events("checkpoint.accepted")[-1]["data"]["step"], 300)

    def test_provider_saying_finished_is_not_enough(self):
        self.started_job()
        # No final checkpoint: the claim proves nothing, so the job goes to recovery rather than completing.
        self.orch.on_run_finished(self.job_id, 1)
        self.assertNotEqual(self.state(), "COMPLETED")
        self.assertEqual(self.events("failure.detected")[0]["data"]["cause"], "NO_FINAL_CHECKPOINT")

    def test_failed_final_evaluation_rolls_back(self):
        self.started_job()
        good = self.checkpoint(100, self.A)
        with Session(self.engine) as s:
            s.get(Checkpoint, good.id).state = "ACCEPTED"
            s.commit()
        final = self.checkpoint(300, self.A, parent=good.manifest_hash)
        self.quality.answers[final.manifest_hash] = "FAIL"
        self.orch.on_run_finished(self.job_id, 1)
        self.assertNotEqual(self.state(), "COMPLETED")
        self.assertEqual(self.net.started[-1][1]["resume_from_hash"], good.manifest_hash)

    def test_stale_worker_cannot_finish_the_job(self):
        self.started_job()
        self.orch.on_failure(self.job_id, provider_id=self.a, cause="PROCESS_EXIT", fencing_token=1)
        with self.assertRaises(jobs.StaleFencingToken):
            self.orch.on_run_finished(self.job_id, 1)

    def test_payment_follows_accepted_lineage_after_full_story(self):
        """A trains, dies, B cheats and is rejected, then another provider finishes. The cheater earns nothing."""
        self.started_job()
        a = self.checkpoint(100, self.A)
        with Session(self.engine) as s:
            s.get(Checkpoint, a.id).state = "ACCEPTED"
            s.get(Provider, self.c).capacity_mib = 1500
            s.commit()
        self.orch.on_failure(self.job_id, provider_id=self.a, cause="PROCESS_EXIT", injected=True, fencing_token=1)
        cheat, _ = self.net.started[-1]
        cheat_key = next(k for k, p in ((self.B, self.b), (self.P, self.c)) if p == cheat)
        honest_key = self.B if cheat_key == self.P else self.P
        token = self.job().fencing_token
        self.orch.on_run_started(self.job_id, token)
        bad = self.checkpoint(150, cheat_key, parent=a.manifest_hash)
        self.orch.on_candidate_rejected(self.job_id, bad.manifest_hash, cheat, ["NON_INFERIORITY_FAILED"])

        token = self.job().fencing_token
        self.orch.on_run_started(self.job_id, token)
        final = self.checkpoint(300, honest_key, parent=a.manifest_hash, note="honest")
        self.assertEqual(self.orch.on_run_finished(self.job_id, token), "COMPLETED")

        with Session(self.engine) as s:
            chain = jobs.accepted_lineage(s, s.get(Checkpoint, final.id))
            steps = jobs.steps_by_signer(chain)
        self.assertEqual(steps, {self.A: 100, honest_key: 200})
        self.assertNotIn(cheat_key, steps)


if __name__ == "__main__":
    unittest.main()
