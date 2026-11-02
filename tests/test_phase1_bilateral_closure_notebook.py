"""Contract tests for the Phase 1 bilateral-review closure notebook."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "06_cierre_revision_separacion_bilateral.ipynb"


class Phase1BilateralClosureNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_runs_repository_closure_command(self):
        self.assertIn("https://github.com/AlonsoYess/Knee.git", self.code)
        self.assertIn("'knee.bilateral_review'", self.code)
        self.assertIn("bilateral_review.example.json", self.code)
        self.assertIn("'--git-commit', commit", self.code)

    def test_notebook_uses_completed_private_review(self):
        self.assertIn("revision_visual_ciega.csv", self.code)
        self.assertIn("cierre_paso_3_publico.json", self.code)
        self.assertIn("parameters_frozen", self.code)

    def test_notebook_preserves_safety_gates(self):
        forbidden = ("model.fit(", "torch.optim", "test_loader", "cohorte_v00")
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
