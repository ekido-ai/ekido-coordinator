"""Direct checkpoint transfers through presigned S3 links, with the old routes kept for older workers."""
try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import asyncio
import base64
import hashlib
import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from ekido_coordinator import main as main_module
from ekido_coordinator.artifacts import ArtifactStore, S3Blobs
from ekido_coordinator.models import Base
from ekido_coordinator.retention import ORPHAN_GRACE, run



class NotFound(Exception):
    response = {"ResponseMetadata": {"HTTPStatusCode": 404}}


class FakeS3:
    """Just enough of the S3 client: objects in a dict, presigned links as strings, S3's SHA-256 check on PUT."""

    def __init__(self):
        self.objects: dict[str, tuple[bytes, datetime]] = {}
        self.calls: list[str] = []

    def put_via_link(self, url: str, body: bytes, checksum: str) -> None:
        key = url.split("key=")[1]
        if base64.b64encode(hashlib.sha256(body).digest()).decode() != checksum:
            raise ValueError("BadDigest")
        self.objects[key] = (body, datetime.now(timezone.utc))

    def generate_presigned_url(self, op, Params, ExpiresIn):
        self.calls.append(op)
        return f"https://s3.test/{op}?key={Params['Key']}"

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise NotFound()
        return {"ContentLength": len(self.objects[Key][0]), "LastModified": self.objects[Key][1]}

    def get_object(self, Bucket, Key):
        body = io.BytesIO(self.objects[Key][0])
        body.iter_chunks = lambda size: iter(lambda: body.read(size), b"")
        return {"Body": body}

    def copy_object(self, Bucket, Key, CopySource, **_):
        self.objects[Key] = (self.objects[CopySource["Key"]][0], datetime.now(timezone.utc))

    def delete_object(self, Bucket, Key):
        del self.objects[Key]

    def get_paginator(self, _):
        fake = self

        class Pages:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": len(v[0]), "LastModified": v[1]} for k, v in fake.objects.items() if k.startswith(Prefix)]}

        return Pages()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DirectUploadRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s3 = FakeS3()
        settings = dict(database_url="sqlite://", demo_mode=True, _env_file=None, artifact_dir=self.tmp.name)
        with patch.object(main_module, "S3Blobs", lambda bucket, region=None: S3Blobs(bucket, client=self.s3)):
            self.app = main_module.create_app(main_module.Settings(artifact_s3_bucket="bucket", **settings))
        self.app.state.gateway.upload_tokens["tok"] = "provider-1"
        self.client = TestClient(self.app)
        self.auth = {"Authorization": "Bearer tok"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_upload_link_then_download_link(self):
        data = b"checkpoint bytes" * 100
        offer = self.client.post(f"/artifacts/{sha(data)}/upload-url", json={"bytes": len(data)}, headers=self.auth).json()
        self.assertIn("url", offer)
        self.s3.put_via_link(offer["url"], data, offer["headers"]["x-amz-checksum-sha256"])
        link = self.client.get(f"/artifacts/{sha(data)}/download-url", headers=self.auth)
        self.assertEqual(link.status_code, 200)
        # An older worker still gets the bytes through the coordinator.
        self.assertEqual(self.client.get(f"/artifacts/{sha(data)}", headers=self.auth).content, data)

    def test_tampered_upload_is_refused_by_the_checksum(self):
        data = b"real"
        offer = self.client.post(f"/artifacts/{sha(data)}/upload-url", json={"bytes": len(data)}, headers=self.auth).json()
        with self.assertRaises(ValueError):
            self.s3.put_via_link(offer["url"], b"fake", offer["headers"]["x-amz-checksum-sha256"])
        self.assertEqual(self.client.get(f"/artifacts/{sha(data)}/download-url", headers=self.auth).status_code, 404)

    def test_existing_blob_needs_no_upload(self):
        data = b"already here"
        asyncio.run(self.app.state.store.put(sha(data), _chunks(data)))
        answer = self.client.post(f"/artifacts/{sha(data)}/upload-url", json={"bytes": len(data)}, headers=self.auth).json()
        self.assertEqual(answer, {"exists": True})
        # Blobs on the coordinator's disk are served from there, not linked.
        self.assertEqual(self.client.get(f"/artifacts/{sha(data)}/download-url", headers=self.auth).status_code, 404)

    def test_links_need_a_token_and_a_valid_digest(self):
        self.assertEqual(self.client.post(f"/artifacts/{'a' * 64}/upload-url", json={"bytes": 1}).status_code, 401)
        self.assertEqual(self.client.post("/artifacts/nothex/upload-url", json={"bytes": 1}, headers=self.auth).status_code, 422)
        self.assertEqual(self.client.post(f"/artifacts/{'a' * 64}/upload-url", json={"bytes": 0}, headers=self.auth).status_code, 422)

    def test_pending_budget_applies_to_links(self):
        self.app.state.gateway.upload_budget.limit = 10
        response = self.client.post(f"/artifacts/{'b' * 64}/upload-url", json={"bytes": 11}, headers=self.auth)
        self.assertEqual(response.status_code, 413)


class DirectUploadsOff(unittest.TestCase):
    def test_workers_fall_back_when_no_bucket_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = main_module.create_app(main_module.Settings(database_url="sqlite://", demo_mode=True, _env_file=None, artifact_dir=tmp))
            app.state.gateway.upload_tokens["tok"] = "p"
            client = TestClient(app)
            auth = {"Authorization": "Bearer tok"}
            self.assertEqual(client.post(f"/artifacts/{'a' * 64}/upload-url", json={"bytes": 1}, headers=auth).status_code, 404)
            self.assertEqual(client.get(f"/artifacts/{'a' * 64}/download-url", headers=auth).status_code, 404)


class RemoteRetention(unittest.TestCase):
    def test_unreferenced_old_blobs_move_to_quarantine_in_the_bucket(self):
        s3 = FakeS3()
        blobs = S3Blobs("bucket", client=s3)
        old, young = b"old orphan", b"young orphan"
        now = datetime.now(timezone.utc)  # The apply step re-checks age against the real clock.
        s3.objects[blobs.key(sha(old))] = (old, now - ORPHAN_GRACE - timedelta(hours=1))
        s3.objects[blobs.key(sha(young))] = (young, now)
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        with tempfile.TemporaryDirectory() as tmp:
            dry = run(engine, ArtifactStore(Path(tmp)), apply=False, now=now, blobs=blobs)
            self.assertEqual((dry["to_quarantine"], dry["young_orphans"]), (1, 1))
            applied = run(engine, ArtifactStore(Path(tmp)), apply=True, now=now, blobs=blobs)
        self.assertEqual(applied["moved"], 1)
        self.assertNotIn(blobs.key(sha(old)), s3.objects)
        self.assertIn(f"quarantine/{now:%Y%m%d}/{sha(old)}", s3.objects)
        self.assertIn(blobs.key(sha(young)), s3.objects)


async def _chunks(data: bytes):
    yield data


if __name__ == "__main__":
    unittest.main()
