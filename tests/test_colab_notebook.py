"""Contract tests for the thin, reproducible Colab entrypoint."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "00_arranque_colab.ipynb"


class ColabNotebookContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_is_valid_and_points_to_the_controlled_repository(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertIn("https://github.com/AlonsoYess/Knee.git", self.code)
        self.assertIn("codex/fase-0-base-reproducible", self.code)

    def test_notebook_uses_a_clean_identifiable_revision(self):
        self.assertIn("git', 'rev-parse', 'HEAD", self.code)
        self.assertIn("git', 'status', '--porcelain", self.code)
        self.assertIn("Restablece el entorno de ejecución", self.code)
        self.assertIn("record.get('git_commit') != commit", self.code)

    def test_notebook_calls_repository_code_and_governance_checks(self):
        self.assertIn("drive.mount", self.code)
        self.assertIn("KNEE_DATA_ROOT", self.code)
        self.assertIn("userdata.get('KNEE_DATA_ROOT')", self.code)
        self.assertNotIn("/content/drive/MyDrive/", self.code)
        self.assertIn("'knee.change_control'", self.code)
        self.assertIn("'knee.audit'", self.code)
        self.assertIn("'knee.experiment'", self.code)
        self.assertIn("'unittest', 'discover'", self.code)

    def test_notebook_contains_no_model_training_implementation(self):
        forbidden = ("model.fit(", "torch.optim", "tensorflow", "sklearn.model_selection")
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.code)

    def test_notebook_has_no_saved_execution_state(self):
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])


if __name__ == "__main__":
    unittest.main()
