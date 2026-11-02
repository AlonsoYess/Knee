"""Contract tests for the controlled v0.4 tibiofemoral-localization repeat."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "11_repeticion_localizacion_tibiofemoral_v04.ipynb"


class Phase1JointV04NotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_validates_the_reproducible_v03_rejection(self):
        self.assertIn("registro_cierre_revision.json", self.code)
        self.assertIn("resumen_revision_visual_publico.json", self.code)
        self.assertIn("rejected_after_blinded_review", self.code)
        self.assertIn("acceptable_crops') != 0", self.code)
        self.assertIn("rejected_crops') != 20", self.code)
        self.assertIn("technical_exclusions') != 0", self.code)
        self.assertIn("parameters_frozen", self.code)

    def test_notebook_runs_v04_and_displays_twenty_views(self):
        self.assertIn("https://github.com/AlonsoYess/Knee.git", self.code)
        self.assertIn("knee.joint_localization", self.code)
        self.assertIn("joint_localization.v0.4.example.json", self.code)
        self.assertIn("v0_4_piloto", self.code)
        self.assertIn("len(previews) != 20", self.code)
        self.assertIn("repeat_blinded_tibiofemoral_localization_v04_pilot", self.code)

    def test_notebook_keeps_all_downstream_operations_blocked(self):
        self.assertIn("'mass_processing_executed': False", self.code)
        self.assertIn("'partitions_created': False", self.code)
        self.assertIn("'training_executed': False", self.code)
        self.assertIn("'reserved_test_opened': False", self.code)
        for fragment in ("model.fit(", "torch.optim", "test_loader"):
            self.assertNotIn(fragment, self.code)

    def test_notebook_uses_private_root_without_hard_coding(self):
        self.assertIn("userdata.get('KNEE_DATA_ROOT')", self.code)
        self.assertNotIn("/content/drive/MyDrive/", self.code)

    def test_code_cells_are_clean_and_valid(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])
            compile("".join(cell.get("source", [])), f"cell_{index}", "exec")


if __name__ == "__main__":
    unittest.main()
