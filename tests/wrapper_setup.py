"""Test setup, imported first by every test file (so each file works alone, under unittest and pytest).

Job keys need a wrapper, and the app refuses to start without one: there is no throwaway key in the product. Tests get one here,
from LOCAL_WRAP_IDENTITY if the environment gives it (CI makes a fresh one for each run) or from a new one made for this process.
"""
import os

import pyrage

IDENTITY = os.environ.get("LOCAL_WRAP_IDENTITY") or str(pyrage.x25519.Identity.generate())
os.environ.setdefault("JOB_KEY_WRAP", "local")
os.environ["LOCAL_WRAP_IDENTITY"] = IDENTITY

from ekido_coordinator import job_keys  # noqa: E402

job_keys.configure(job_keys.make_wrapper("local", local_identity=IDENTITY))