try:  # job keys need a wrapper; see wrapper_setup.py
    import wrapper_setup  # noqa: F401
except ImportError:  # run as tests.<module> (the Postgres CI step)
    from tests import wrapper_setup  # noqa: F401
import io
import json
import unittest
import zipfile

from ekido_coordinator.model_bundle import bundle

SPEC = {"template": "distilbert-lora-agnews/v1", "model": {"id": "distilbert-base-uncased", "revision": "main"},
        "lora": {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["q_lin", "v_lin"]}, "max_seq_length": 128}


class ModelBundle(unittest.TestCase):
    def open(self, spec):
        data = bundle("job-1", spec, {"format": "GPCK/1"}, "a" * 64, 800, b"weights")
        return zipfile.ZipFile(io.BytesIO(data))

    def test_news_model_bundle_holds_weights_terms_and_a_script(self):
        archive = self.open(SPEC)
        self.assertEqual(set(archive.namelist()), {"trainable.safetensors", "manifest.json", "workload.json", "README.txt", "try_model.py"})
        self.assertEqual(archive.read("trainable.safetensors"), b"weights")
        self.assertEqual(json.loads(archive.read("workload.json")), SPEC)
        self.assertIn("Final step:     800", archive.read("README.txt").decode())
        compile(archive.read("try_model.py").decode(), "try_model.py", "exec")  # the shipped script is valid Python

    def test_synthetic_bundle_says_it_is_a_test_model_and_has_no_script(self):
        archive = self.open({"template": "synthetic-linear/v1"})
        self.assertNotIn("try_model.py", archive.namelist())
        self.assertIn("test model, not a useful one", archive.read("README.txt").decode())


if __name__ == "__main__":
    unittest.main()
