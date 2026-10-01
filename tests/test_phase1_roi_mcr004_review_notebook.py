"""Boundaries and executability of notebook 14; no real image execution."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Mcr004ReviewNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT/"notebooks"/"14_revision_tecnica_y_cierre_regresion_mcr004.ipynb").read_text(encoding="utf-8"))
        cls.source = "\n".join("".join(cell["source"]) for cell in cls.notebook["cells"])
        cls.code = "\n".join("".join(cell["source"]) for cell in cls.notebook["cells"] if cell["cell_type"]=="code")

    def test_all_cells_compile_without_outputs_or_private_fixed_path(self):
        for index,cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"]=="code":
                compile("".join(cell["source"]),f"notebook14_cell_{index}","exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"],[])
        self.assertNotIn("/content/drive/MyDrive/",self.source)
        self.assertNotIn("C:\\Users\\",self.source)
        self.assertIn("userdata.get('KNEE_DATA_ROOT')",self.code)

    def test_existing_review_only_and_unchanged_environment_bootstrap(self):
        self.assertIn("knee.roi_mcr004_review",self.code)
        self.assertIn("review_call('context')",self.code)
        self.assertIn("review_call('close'",self.code)
        self.assertIn("virtualenv==21.7.11",self.code)
        self.assertIn("--no-periodic-update",self.code)
        self.assertIn("test_roi_mcr004*.py",self.code)
        self.assertNotIn("candidate_for_half",self.code)
        self.assertNotIn("run_historical_regression",self.code)
        self.assertNotIn("'-m', 'venv'",self.code)
        self.assertNotIn("select_new_participants",self.code)
        self.assertIn("Google Colab, libreta 14",self.code)

    def test_human_controls_required_without_automatic_decisions(self):
        self.assertIn("context['pending_knee_aliases']",self.code)
        self.assertIn("confirm_all.value",self.code)
        self.assertIn("blind_confirm.value",self.code)
        self.assertIn("state['inspected'].value",self.code)
        self.assertIn("state['notes'].value.strip()",self.code)
        self.assertIn("if missing:",self.code)
        self.assertIn("step 4",self.source.lower().replace("paso 4","step 4"))
        self.assertNotIn("case_004_LEFT",self.code)
        self.assertNotIn("case_006_LEFT",self.code)
        self.assertNotIn("case_010_LEFT",self.code)


if __name__=="__main__":
    unittest.main()
