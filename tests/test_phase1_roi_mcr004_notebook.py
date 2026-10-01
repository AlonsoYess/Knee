"""Static safety checks for the bounded MCR-2026-004 Colab orchestrator."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/"notebooks"/"13_regresion_historica_roi_mcr004.ipynb"
CONFIG = ROOT/"configs"/"roi_mcr004_historical.example.json"


class RoiMcr004NotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell["source"]) for cell in cls.notebook["cells"]
        )
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_only_historical_10_studies_and_20_knees(self):
        self.assertEqual(self.config["expected_unique_studies"], 10)
        self.assertEqual(self.config["expected_knees"], 20)
        self.assertIn("mcr004_regresion_historica", self.config["output_dir"])
        self.assertIn("v0_4_piloto", self.config["v04_closure_record_json"])
        self.assertIn("v0_4_piloto", self.config["v04_review_summary_json"])
        self.assertIn("test_roi_mcr004*.py", self.source)
        self.assertIn("revision_tecnica_ciega.csv", self.source)

    def test_isolated_environment_and_no_early_downstream_action(self):
        self.assertIn("venv_mcr004", self.source)
        self.assertIn("sys.version_info[:2] != (3, 12)", self.source)
        self.assertIn("virtualenv==21.7.11", self.source)
        self.assertIn("environment['virtualenv_bootstrap'] = '21.7.11'", self.source)
        self.assertIn("--no-periodic-update", self.source)
        self.assertNotIn("'-m', 'venv'", self.source)
        self.assertIn("requirements-mcr004.txt", self.source)
        self.assertIn("registro_preparacion.json", self.source)
        self.assertIn("git_commit", self.source)
        self.assertIn("'mass_processing_executed': False", self.source)
        self.assertIn("'partitions_created': False", self.source)
        self.assertIn("'training_executed': False", self.source)
        self.assertIn("'reserved_test_opened': False", self.source)
        self.assertNotIn("V06", self.source)
        self.assertNotIn("KL", self.source)
        self.assertNotIn("select_new_participants", self.source)

    def test_no_private_fixed_path_or_executed_cells(self):
        self.assertNotIn("/content/drive/MyDrive/", self.source)
        self.assertNotIn("C:\\Users\\", self.source)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
