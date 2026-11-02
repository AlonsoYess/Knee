"""Contract tests for the Phase 1 pilot-audit notebook."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_auditoria_piloto_dicoms.ipynb"


class Phase1PilotNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_orchestrates_repository_audit(self):
        self.assertIn("https://github.com/AlonsoYess/Knee.git", self.code)
        self.assertIn("'knee.dicom_audit'", self.code)
        self.assertIn("pilot_audit.example.json", self.code)
        self.assertIn("'unittest', 'discover'", self.code)

    def test_notebook_uses_private_root_without_hard_coding_drive_path(self):
        self.assertIn("userdata.get('KNEE_DATA_ROOT')", self.code)
        self.assertNotIn("/content/drive/MyDrive/", self.code)
        self.assertIn("outputs' / 'auditorias' / 'fase_1'", self.code)

    def test_notebook_does_not_train_or_open_reserved_test(self):
        forbidden = ("model.fit(", "torch.optim", "tensorflow", "test_loader")
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.code)
        self.assertIn("'training_executed': False", self.code)
        self.assertIn("'reserved_test_opened': False", self.code)

    def test_notebook_has_no_saved_execution_state(self):
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])


if __name__ == "__main__":
    unittest.main()
