try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path

from ekido_coordinator.artifacts import ArtifactError, ArtifactStore


async def chunks(*parts: bytes):
    for part in parts:
        yield part


class ArtifactStoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(Path(self.dir.name))

    def tearDown(self):
        self.dir.cleanup()

    def test_valid_digest_maps_inside_the_store(self):
        digest = "a" * 64
        path = self.store.path_for(digest)
        self.assertEqual(path.name, digest)
        self.assertEqual(path.parent.name, "aa")
        self.assertTrue(str(path).startswith(str(Path(self.dir.name).resolve())))

    def test_trailing_newline_is_refused(self):
        # re.match with "$" used to accept this; fullmatch must not.
        with self.assertRaises(ArtifactError):
            self.store.path_for("a" * 64 + "\n")

    def test_bad_names_are_refused(self):
        for bad in ["", "A" * 64, "a" * 63, "../" + "a" * 61, "a" * 65, "g" * 64]:
            with self.subTest(bad=bad), self.assertRaises(ArtifactError):
                self.store.path_for(bad)

    def test_put_stores_matching_content_and_refuses_a_mismatch(self):
        data = b"checkpoint bytes"
        digest = hashlib.sha256(data).hexdigest()
        self.assertEqual(asyncio.run(self.store.put(digest, chunks(data))), len(data))
        self.assertTrue(self.store.exists(digest))
        with self.assertRaises(ArtifactError):
            asyncio.run(self.store.put("b" * 64, chunks(data)))
        self.assertFalse(self.store.exists("b" * 64))


if __name__ == "__main__":
    unittest.main()
