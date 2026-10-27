import base64
import io
from pathlib import Path
import unittest
import zipfile

from scripts.kaggle_runner import read_manifest, source_snapshot, notebook_source


class KaggleRunner(unittest.TestCase):
    def test_manifest_and_generated_notebooks(self):
        manifest = read_manifest()
        payload, digest = source_snapshot()
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
            names = archive.namelist()
            self.assertIn("common/runtime.py", names)
            self.assertIn("experiments/runs.json", names)
            self.assertFalse(any("credentials" in n or "kaggle_accounts" in n or ".venv" in n or "outputs/" in n
                                 or n.startswith(("data/", "course_data/", "remote/")) for n in names))
        for run in manifest["runs"]:
            compile(notebook_source(run, payload, digest), run["id"], "exec")

    def test_declared_modules_exist(self):
        root = Path(__file__).resolve().parents[1]
        for run in read_manifest()["runs"]:
            for command in run.get("setup", []) + [c for lane in run["lanes"] for c in lane]:
                self.assertEqual(command[0], "-m")
                self.assertTrue((root / (command[1].replace(".", "/") + ".py")).exists(), command)
                for flag, value in zip(command, command[1:]):
                    if flag == "--config":
                        self.assertTrue((root / value).exists(), value)


if __name__ == "__main__":
    unittest.main()
