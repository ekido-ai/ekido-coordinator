"""A12: a job's key is made with the job and released only to the device assigned to it, in every case that matters."""

import wrapper_setup  # noqa: F401  (job keys need a wrapper; see wrapper_setup.py)
import base64
import logging
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock

from eth_account import Account
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core import jobkeys
from ekido_core.signing import sign_document
from ekido_coordinator import job_keys, jobs
from ekido_coordinator.agent_gateway import Connection
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import Job, JobAssignment, JobKey, Provider, Wallet

KEY = "0x" + "a" * 64


class Fixture(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None))
        self.engine = self.app.state.engine
        self.wrapper = job_keys.get_wrapper()
        with Session(self.engine) as session:
            wallet = Wallet(address=Account.from_key(KEY).address.lower(), role="provider")
            session.add(wallet)
            session.flush()
            a = Provider(name="A", wallet_id=wallet.id, kind="REAL", price_per_step=Decimal("0.001"))
            b = Provider(name="B", wallet_id=wallet.id, kind="REAL", price_per_step=Decimal("0.002"))
            session.add_all([a, b])
            session.flush()
            job = jobs.create_job(session, spec={}, policy={}, budget=Decimal("5"), deadline_seconds=1800)
            jobs.assign(session, job, a.id, {})
            self.a, self.b, self.job, self.token = a.id, b.id, job.id, job.fencing_token
            session.commit()
        self.device = jobkeys.create_job_key()  # the one-off key a device makes for this assignment

    def release(self, **overrides):
        args = dict(job_id=self.job, provider_id=self.a, fencing_token=self.token, agent_recipient=self.device.recipient)
        args.update(overrides)
        with Session(self.engine) as session:
            return job_keys.release_to_device(session, **args)

    def job_identity(self):
        with Session(self.engine) as session:
            row = session.scalar(select(JobKey).where(JobKey.job_id == self.job))
            return jobkeys.unwrap_job_key(row.wrapped_identity, self.job, self.wrapper).identity

    def refused(self, code, **overrides):
        with self.assertRaises(job_keys.KeyRefused) as caught:
            self.release(**overrides)
        self.assertEqual(caught.exception.code, code)

    def set_job_state(self, state):
        with Session(self.engine) as session:
            session.get(Job, self.job).state = state
            session.commit()


class CreationTests(Fixture):
    def test_a_key_is_created_with_the_job_and_stored_wrapped(self):
        with Session(self.engine) as session:
            rows = session.scalars(select(JobKey).where(JobKey.job_id == self.job)).all()
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertTrue(row.recipient.startswith("age1"))
            self.assertIsNone(row.destroyed_at)
            identity = self.job_identity()
            self.assertEqual(str(jobkeys._identity(identity).to_public()), row.recipient)
            self.assertNotIn(identity.encode(), row.wrapped_identity)  # nothing unwrapped in the database

    def test_every_job_has_its_own_key(self):
        with Session(self.engine) as session:
            other = jobs.create_job(session, spec={"n": 1}, policy={}, budget=Decimal("1"), deadline_seconds=60)
            keys = {r.recipient for r in session.scalars(select(JobKey)).all()}
            self.assertEqual(len(keys), 2)
            self.assertIsNotNone(other)


class ReleaseTests(Fixture):
    def test_the_assigned_device_gets_the_key_encrypted_to_its_own_recipient(self):
        sealed = self.release()
        self.assertTrue(jobkeys.is_age_ciphertext(sealed))
        self.assertEqual(jobkeys.decrypt_bytes(sealed, self.device.identity).decode(), self.job_identity())
        # Anyone else cannot read it, including another device that sees the same message.
        with self.assertRaises(jobkeys.JobKeyError):
            jobkeys.decrypt_bytes(sealed, jobkeys.create_job_key().identity)

    def test_a_device_that_is_not_assigned_is_refused(self):
        self.refused("NOT_ASSIGNED", provider_id=self.b)

    def test_an_unknown_job_or_token_is_refused(self):
        self.refused("NO_JOB", job_id="missing")
        self.refused("NOT_ASSIGNED", fencing_token=self.token + 99)

    def test_a_fenced_assignment_is_refused(self):
        with Session(self.engine) as session:
            jobs.bump_fencing(session, session.get(Job, self.job))  # recovery moved the job on; the old token is stale
            session.commit()
        self.refused("STALE_FENCING")

    def test_a_finished_assignment_is_refused(self):
        for state in ("FAILED", "REJECTED"):
            with self.subTest(state=state):
                with Session(self.engine) as session:
                    session.scalar(select(JobAssignment).where(JobAssignment.job_id == self.job)).state = state
                    session.commit()
                self.refused("ASSIGNMENT_ENDED")

    def test_any_job_that_has_ended_is_refused(self):
        for state in ("SETTLED", "FAILED", "CANCELLED", "REFUNDED"):
            with self.subTest(state=state):
                self.set_job_state(state)
                self.refused("JOB_ENDED")

    def test_a_job_still_running_or_validating_can_get_its_key(self):
        for state in ("ASSIGNED", "RUNNING", "RECOVERING", "VALIDATING"):
            with self.subTest(state=state):
                self.set_job_state(state)
                self.assertTrue(jobkeys.is_age_ciphertext(self.release()))

    def test_a_destroyed_or_missing_key_is_refused(self):
        with Session(self.engine) as session:
            row = session.scalar(select(JobKey).where(JobKey.job_id == self.job))
            row.wrapped_identity = None
            session.commit()
        self.refused("NO_KEY")
        with Session(self.engine) as session:
            session.delete(session.scalar(select(JobKey).where(JobKey.job_id == self.job)))
            session.commit()
        self.refused("NO_KEY")

    def test_a_bad_recipient_is_refused(self):
        for bad in ("", "age1nope", "not a key", "AGE-SECRET-KEY-1ABC"):
            with self.subTest(bad=bad):
                self.refused("BAD_RECIPIENT", agent_recipient=bad)

    def test_a_key_wrapped_for_another_job_cannot_be_released(self):
        with Session(self.engine) as session:
            other = jobs.create_job(session, spec={"n": 2}, policy={}, budget=Decimal("1"), deadline_seconds=60)
            theirs = session.scalar(select(JobKey).where(JobKey.job_id == other.id))
            mine = session.scalar(select(JobKey).where(JobKey.job_id == self.job))
            mine.wrapped_identity = theirs.wrapped_identity  # swapped rows must not unwrap
            session.commit()
        self.refused("UNWRAP_FAILED")


class GatewayTests(Fixture, unittest.IsolatedAsyncioTestCase):
    def connection(self, provider):
        return Connection(provider, Account.from_key(KEY).address.lower(), AsyncMock(), "token", job_id=self.job, fencing_token=self.token)

    async def deliver(self, connection, recipient, token=None, seq=1):
        body = {"seq": seq, "type": "job_key.request", "job_id": self.job, "fencing_token": self.token if token is None else token, "payload": {"recipient": recipient}}
        _, body["sig"] = sign_document(body, KEY)
        await self.app.state.gateway._on_message(connection, body)
        return [call.args[0] for call in connection.websocket.send_json.call_args_list]

    async def test_the_assigned_device_receives_the_key_over_the_gateway(self):
        sent = await self.deliver(self.connection(self.a), self.device.recipient)
        self.assertEqual([m["type"] for m in sent], ["job_key.release"])
        sealed = base64.b64decode(sent[0]["payload"]["ciphertext_b64"])
        self.assertEqual(jobkeys.decrypt_bytes(sealed, self.device.identity).decode(), self.job_identity())

    async def test_another_device_is_told_it_is_not_assigned_and_gets_nothing(self):
        sent = await self.deliver(self.connection(self.b), self.device.recipient)
        self.assertEqual(sent[0]["payload"]["error"], "NOT_ASSIGNED")
        self.assertNotIn("job_key.release", [m["type"] for m in sent])

    async def test_a_stale_token_is_stopped_and_gets_nothing(self):
        with Session(self.engine) as session:
            jobs.bump_fencing(session, session.get(Job, self.job))
            session.commit()
        sent = await self.deliver(self.connection(self.a), self.device.recipient)
        self.assertEqual(sent[0]["type"], "cmd.stop_run")
        self.assertNotIn("job_key.release", [m["type"] for m in sent])

    async def test_an_ended_job_is_refused_with_a_reason(self):
        self.set_job_state("SETTLED")
        sent = await self.deliver(self.connection(self.a), self.device.recipient)
        self.assertEqual((sent[0]["payload"]["error"], sent[0]["payload"]["reason"]), ("KEY_REFUSED", "JOB_ENDED"))

    async def test_a_missing_or_bad_recipient_is_refused(self):
        for recipient in (None, 5, "age1nope"):
            with self.subTest(recipient=recipient):
                connection = self.connection(self.a)
                sent = await self.deliver(connection, recipient)
                self.assertEqual((sent[0]["payload"]["error"], sent[0]["payload"]["reason"]), ("KEY_REFUSED", "BAD_RECIPIENT"))


class NothingUnwrappedInLogsTests(Fixture, unittest.IsolatedAsyncioTestCase):
    async def test_no_unwrapped_key_appears_in_any_log_or_message(self):
        secrets = {self.job_identity(), self.device.identity}
        gateway = self.app.state.gateway
        connection = Connection(self.a, Account.from_key(KEY).address.lower(), AsyncMock(), "token", job_id=self.job, fencing_token=self.token)
        with self.assertLogs(level="DEBUG") as captured:
            logging.getLogger("probe").debug("start")
            seq = 0
            for recipient, state in ((self.device.recipient, "RUNNING"), ("age1nope", "RUNNING"), (self.device.recipient, "SETTLED")):
                self.set_job_state(state)
                seq += 1
                body = {"seq": seq, "type": "job_key.request", "job_id": self.job, "fencing_token": self.token, "payload": {"recipient": recipient}}
                _, body["sig"] = sign_document(body, KEY)
                await gateway._on_message(connection, body)
            # Break the wrapped key so the "could not unwrap" path logs too.
            self.set_job_state("RUNNING")
            with Session(self.engine) as session:
                session.scalar(select(JobKey).where(JobKey.job_id == self.job)).wrapped_identity = b"junk"
                session.commit()
            seq += 1
            body = {"seq": seq, "type": "job_key.request", "job_id": self.job, "fencing_token": self.token, "payload": {"recipient": self.device.recipient}}
            _, body["sig"] = sign_document(body, KEY)
            await gateway._on_message(connection, body)
        everything = "\n".join(captured.output) + repr([c.args for c in connection.websocket.send_json.call_args_list])
        for secret in secrets:
            self.assertNotIn(secret, everything)
            self.assertNotIn(secret.split("-1", 1)[1], everything)


class WrapperConfigTests(unittest.TestCase):
    def test_a_local_wrapper_needs_a_configured_identity_and_other_modes_are_refused(self):
        identity = str(jobkeys.x25519.Identity.generate())
        self.assertIsInstance(job_keys.make_wrapper("local", local_identity=identity), jobkeys.LocalWrapper)
        for mode, identity_arg in (("local", ""), ("", ""), ("", identity), ("plaintext", identity), ("KMS", "")):
            with self.subTest(mode=mode, identity=bool(identity_arg)):
                with self.assertRaises(ValueError):
                    job_keys.make_wrapper(mode, local_identity=identity_arg)

    def test_the_error_says_what_to_set_and_never_shows_a_key(self):
        identity = str(jobkeys.x25519.Identity.generate())
        with self.assertRaises(ValueError) as caught:
            job_keys.make_wrapper("local")
        self.assertIn("LOCAL_WRAP_IDENTITY", str(caught.exception))
        self.assertIn("JOB_KEY_WRAP", str(caught.exception))
        self.assertNotIn("AGE-SECRET-KEY", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            job_keys.make_wrapper(identity)  # a secret pasted into the wrong setting must not be echoed
        self.assertNotIn(identity, str(caught.exception))

    def test_there_is_no_lazy_fallback_wrapper(self):
        saved = job_keys._wrapper
        job_keys._wrapper = None
        try:
            with self.assertRaises(RuntimeError):
                job_keys.get_wrapper()
        finally:
            job_keys._wrapper = saved

    def test_the_app_refuses_to_start_without_a_working_wrapper(self):
        from ekido_coordinator.main import Settings, create_app

        identity = str(jobkeys.x25519.Identity.generate())
        saved = job_keys._wrapper
        try:
            for settings in (dict(job_key_wrap="local", local_wrap_identity=""), dict(job_key_wrap="", local_wrap_identity=""), dict(job_key_wrap="", local_wrap_identity=identity), dict(job_key_wrap="nope")):
                with self.subTest(settings=settings):
                    with self.assertRaises(ValueError):
                        create_app(Settings(database_url="sqlite://", _env_file=None, **settings))
            app = create_app(Settings(database_url="sqlite://", _env_file=None, job_key_wrap="local", local_wrap_identity=identity))
            self.assertIsNotNone(app)
        finally:
            job_keys._wrapper = saved

    def test_the_default_setting_is_not_local(self):
        from ekido_coordinator.main import Settings

        self.assertEqual(Settings.model_fields["job_key_wrap"].default, "")

    def test_local_wrapper_with_a_configured_key_survives_a_restart(self):
        identity = str(jobkeys.x25519.Identity.generate())
        key = jobkeys.create_job_key()
        wrapped = jobkeys.wrap_job_key(key, "j1", job_keys.make_wrapper("local", local_identity=identity))
        again = jobkeys.unwrap_job_key(wrapped, "j1", job_keys.make_wrapper("local", local_identity=identity))
        self.assertEqual(again.identity, key.identity)


if __name__ == "__main__":
    unittest.main()
