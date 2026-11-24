"""PUT /datasets/{sha256}: the Platform hands over a paid job's training rows (ticket C9). Operator API only."""
try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import hashlib
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from ekido_coordinator.main import MAX_DATASET_BYTES, Settings, create_app

LOCAL = {"base_url": "http://127.0.0.1:8000", "client": ("127.0.0.1", 50000)}
REMOTE = {"base_url": "http://192.168.1.20:8000", "client": ("192.168.1.50", 50000)}
ROWS = b'{"label": "refund", "split": "train", "text": "paise wapas chahiye"}\n{"label": "order", "split": "test", "text": "order kab aayega"}\n'
DIGEST = hashlib.sha256(ROWS).hexdigest()


class Datasets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name) / "artifacts"
        self.app = create_app(Settings(database_url="sqlite://", demo_mode=True, _env_file=None, api_token="operator-secret", artifact_dir=str(self.store)))

    def test_the_operator_stores_the_exact_bytes_by_hash(self):
        response = TestClient(self.app, **REMOTE).put(f"/datasets/{DIGEST}", content=ROWS, headers={"Authorization": "Bearer operator-secret"})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json(), {"sha256": DIGEST, "bytes": len(ROWS)})
        self.assertEqual((self.store / DIGEST[:2] / DIGEST).read_bytes(), ROWS)
        again = TestClient(self.app, **LOCAL).put(f"/datasets/{DIGEST}", content=ROWS)  # same bytes again: fine, nothing changes
        self.assertEqual(again.status_code, 201)

    def test_anyone_else_is_refused(self):
        client = TestClient(self.app, **REMOTE)
        self.assertEqual(client.put(f"/datasets/{DIGEST}", content=ROWS).status_code, 401)
        self.assertEqual(client.put(f"/datasets/{DIGEST}", content=ROWS, headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertFalse((self.store / DIGEST[:2] / DIGEST).exists())

    def test_bytes_that_dont_match_the_hash_are_not_kept(self):
        response = TestClient(self.app, **LOCAL).put(f"/datasets/{DIGEST}", content=ROWS + b"tampered\n")
        self.assertEqual(response.status_code, 422)
        self.assertFalse((self.store / DIGEST[:2] / DIGEST).exists())

    def test_a_bad_name_or_an_oversized_dataset_is_refused(self):
        client = TestClient(self.app, **LOCAL)
        self.assertEqual(client.put("/datasets/not-a-hash", content=ROWS).status_code, 422)
        big = client.put(f"/datasets/{DIGEST}", content=ROWS, headers={"Content-Length": str(MAX_DATASET_BYTES + 1)})
        self.assertEqual(big.status_code, 413)


if __name__ == "__main__":
    unittest.main()
