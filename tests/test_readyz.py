import wrapper_setup  # noqa: F401  (job keys need a wrapper; see wrapper_setup.py)
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from ekido_coordinator.main import Settings, create_app

LOCAL = {"base_url": "http://127.0.0.1:8000", "client": ("127.0.0.1", 50000)}


class ReadyzTest(unittest.TestCase):
    def test_ready_when_database_and_storage_are_fine(self):
        with tempfile.TemporaryDirectory() as folder:
            app = create_app(Settings(database_url="sqlite://", artifact_dir=folder, _env_file=None))
            r = TestClient(app, **LOCAL).get("/readyz")
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["status"], "ready")
            self.assertTrue(body["checks"]["database"]["ok"])
            self.assertIn("MiB free", body["checks"]["artifacts"]["detail"])
            self.assertFalse(any(Path(folder).iterdir()), "the write probe must not be left behind")

    def test_degraded_when_storage_is_not_writable(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder) / "artifacts"
            app = create_app(Settings(database_url="sqlite://", artifact_dir=str(store), _env_file=None))
            # The disk goes bad after start-up: the artifact folder is replaced by a plain file.
            store.rmdir()
            store.write_text("x")
            r = TestClient(app, **LOCAL).get("/readyz")
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["status"], "degraded")
            self.assertFalse(r.json()["checks"]["artifacts"]["ok"])

    def test_reveals_no_secrets(self):
        with tempfile.TemporaryDirectory() as folder:
            app = create_app(Settings(database_url="sqlite://", artifact_dir=folder, _env_file=None))
            text = TestClient(app, **LOCAL).get("/readyz").text.lower()
            for word in ("key", "0x", "token", "password"):
                self.assertNotIn(word, text)


if __name__ == "__main__":
    unittest.main()
