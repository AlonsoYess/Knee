"""Contract tests for the controlled v0.3 tibiofemoral-localization repeat."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "09_repeticion_localizacion_tibiofemoral_v03.ipynb"


class Phase1JointV03NotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_validates_the_rejected_v02_review(self):
        self.assertIn("knee.joint_review", self.code)
        self.assertIn("tibiofemoral_crop_v0.2_pilot", self.code)
        self.assertIn("rejected_after_blinded_review", self.code)
        self.assertIn("acceptable_crops') != 8", self.code)
        self.assertIn("rejected_crops') != 12", self.code)

    def test_notebook_runs_v03_and_displays_twenty_views(self):
        self.assertIn("https://github.com/AlonsoYess/Knee.git", self.code)
        self.assertIn("knee.joint_localization", self.code)
        self.assertIn("joint_localization.v0.3.example.json", self.code)
        self.assertIn("v0_3_piloto", self.code)
        self.assertIn("len(previews) != 20", self.code)

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
