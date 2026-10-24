"""Contract tests for the Phase 1 bilateral-separation pilot notebook."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "05_validacion_separacion_bilateral.ipynb"


class Phase1BilateralNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_orchestrates_versioned_repository_code(self):
        self.assertIn("https://github.com/AlonsoYess/Knee.git", self.code)
        self.assertIn("'knee.bilateral_separation'", self.code)
        self.assertIn("bilateral_separation.example.json", self.code)
        self.assertIn("'unittest', 'discover'", self.code)

    def test_notebook_reads_only_closed_pilot_audit(self):
        self.assertIn("auditoria_piloto_privada.json", self.code)
        forbidden = ("cohorte_oai", "PROGRESION", "KL_48", "label")
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.code)

    def test_notebook_does_not_mass_process_train_or_open_test(self):
        forbidden = ("cohorte_v00", "model.fit(", "torch.optim", "test_loader")
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.code)
        self.assertIn("'mass_processing_executed': False", self.code)
        self.assertIn("'training_executed': False", self.code)
        self.assertIn("'reserved_test_opened': False", self.code)

    def test_notebook_uses_private_root_without_hard_coding(self):
        self.assertIn("userdata.get('KNEE_DATA_ROOT')", self.code)
        self.assertNotIn("/content/drive/MyDrive/", self.code)

    def test_notebook_has_no_saved_execution_state(self):
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])

    def test_all_code_cells_are_valid_python(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            compile("".join(cell.get("source", [])), f"cell_{index}", "exec")


if __name__ == "__main__":
    unittest.main()
