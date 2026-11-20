"""Per-job encryption keys: made with the job, released only to the device that is assigned to it (ticket A12).

A job's key is an age X25519 keypair (`ekido_core.jobkeys`). Its public half (the recipient) is stored in the clear.
Its secret half (the identity) is stored only wrapped by KMS, bound to the job id. To let a device decrypt, the
coordinator unwraps the identity in memory, encrypts it to a one-off recipient the device made for this assignment,
and sends that. The unwrapped identity is never logged, never put in an error and is dropped as soon as it is encrypted.

A release is refused (`KeyRefused`) for: a device that is not assigned to the job, a stale fencing token, an assignment
that has ended, a job that has ended, a key that was destroyed, and a recipient that is not a valid age key.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core import jobkeys
from ekido_core.job_states import JobState
from ekido_core.jobkeys import IdentityWrapper, JobKeyError

from .models import Job, JobAssignment, JobKey

log = logging.getLogger(__name__)

LIVE_ASSIGNMENT_STATES = frozenset({"ASSIGNED", "RUNNING"})
ENDED_JOB_STATES = frozenset({JobState.SETTLED.value, JobState.FAILED.value, JobState.CANCELLED.value, JobState.REFUNDED.value})


class KeyRefused(Exception):
    """A key release was refused. `code` is a fixed, safe string; the message never contains key material."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_wrapper: IdentityWrapper | None = None


def make_wrapper(mode: str, *, key_id: str = jobkeys.KMS_KEY_ALIAS, region: str = "ap-south-1", local_identity: str = "") -> IdentityWrapper:
    """`kms` for staging and live; `local` for tests and a laptop. A local wrapper with no configured key is
    ephemeral: wrapped keys cannot be unwrapped after a restart, so never use it where data must outlive the process."""
    if mode == "kms":
        return jobkeys.KmsWrapper.from_boto3(region_name=region, key_id=key_id)
    if mode == "local":
        if not local_identity:
            log.warning("job keys are wrapped with an ephemeral local key; set JOB_KEY_WRAP=kms outside tests")
        return jobkeys.LocalWrapper(local_identity or None)
    raise ValueError(f"unknown job key wrapper {mode!r}; use 'kms' or 'local'")


def configure(wrapper: IdentityWrapper) -> None:
    global _wrapper
    _wrapper = wrapper


def get_wrapper() -> IdentityWrapper:
    global _wrapper
    if _wrapper is None:
        _wrapper = make_wrapper("local")
    return _wrapper


def create_for_job(session: Session, job_id: str, wrapper: IdentityWrapper | None = None) -> JobKey:
    """Make and store a job's key. Called when the job is created."""
    key = jobkeys.create_job_key()
    wrapped = jobkeys.wrap_job_key(key, job_id, wrapper or get_wrapper())
    row = JobKey(job_id=job_id, recipient=key.recipient, wrapped_identity=wrapped)
    session.add(row)
    session.flush()
    return row


def release_to_device(
    session: Session, *, job_id: str, provider_id: str, fencing_token: int, agent_recipient: str, wrapper: IdentityWrapper | None = None
) -> bytes:
    """The job identity, encrypted to the device's own recipient. Raises `KeyRefused` for every case in the module doc."""
    job = session.get(Job, job_id)
    if job is None:
        raise KeyRefused("NO_JOB")
    assignment = session.scalar(select(JobAssignment).where(JobAssignment.job_id == job_id, JobAssignment.fencing_token == fencing_token))
    if assignment is None or assignment.provider_id != provider_id:
        raise KeyRefused("NOT_ASSIGNED")
    if job.fencing_token != fencing_token:
        raise KeyRefused("STALE_FENCING")
    if assignment.state not in LIVE_ASSIGNMENT_STATES:
        raise KeyRefused("ASSIGNMENT_ENDED")
    if job.state in ENDED_JOB_STATES:
        raise KeyRefused("JOB_ENDED")
    row = session.scalar(select(JobKey).where(JobKey.job_id == job_id))
    if row is None or row.destroyed_at is not None or row.wrapped_identity is None:
        raise KeyRefused("NO_KEY")
    try:
        identity = jobkeys.unwrap_job_key(row.wrapped_identity, job_id, wrapper or get_wrapper()).identity
    except JobKeyError:
        log.error("job %s: could not unwrap its key", job_id)  # fixed text: nothing about the key
        raise KeyRefused("UNWRAP_FAILED") from None
    try:
        return jobkeys.encrypt_bytes(identity.encode("ascii"), agent_recipient)
    except JobKeyError:
        raise KeyRefused("BAD_RECIPIENT") from None
    finally:
        del identity


def release_to_verifier(session: Session, *, job_id: str, verifier_recipient: str, wrapper: IdentityWrapper | None = None) -> bytes:
    """The job identity, encrypted to a verifier's one-off recipient, so it can read the checkpoints it judges (ticket A12).

    The caller must already have checked who the verifier is (a signed session) and that it holds a claimed evaluation of this job
    (`QuorumQuality.is_working_on`). Refused like a device: no such job, an ended job, no key or a destroyed one, a bad recipient. A
    job that is VALIDATING or COMPLETED still has a key to give: verification happens before it ends.
    """
    job = session.get(Job, job_id)
    if job is None:
        raise KeyRefused("NO_JOB")
    if job.state in ENDED_JOB_STATES:
        raise KeyRefused("JOB_ENDED")
    row = session.scalar(select(JobKey).where(JobKey.job_id == job_id))
    if row is None or row.destroyed_at is not None or row.wrapped_identity is None:
        raise KeyRefused("NO_KEY")
    try:
        identity = jobkeys.unwrap_job_key(row.wrapped_identity, job_id, wrapper or get_wrapper()).identity
    except JobKeyError:
        log.error("job %s: could not unwrap its key", job_id)
        raise KeyRefused("UNWRAP_FAILED") from None
    try:
        return jobkeys.encrypt_bytes(identity.encode("ascii"), verifier_recipient)
    except JobKeyError:
        raise KeyRefused("BAD_RECIPIENT") from None
    finally:
        del identity
