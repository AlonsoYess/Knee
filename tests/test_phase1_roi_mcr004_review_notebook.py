"""Boundaries and executability of notebook 14; no real image execution."""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


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
        self.assertIn("completed_review_call('context')",self.code)
        self.assertIn("completed_review_call('assisted-context')",self.code)
        self.assertIn("completed_review_call('close'",self.code)
        self.assertIn("virtualenv==21.7.11",self.code)
        self.assertIn("--no-periodic-update",self.code)
        self.assertIn("test_roi_mcr004*.py",self.code)
        self.assertNotIn("candidate_for_half",self.code)
        self.assertNotIn("run_historical_regression",self.code)
        self.assertNotIn("'-m', 'venv'",self.code)
        self.assertNotIn("select_new_participants",self.code)
        self.assertIn("Google Colab, libreta 14",self.code)

    def test_prepared_data_not_personal_attestations_or_recreated_manual_forms(self):
        self.assertIn("completed_confirm_all.value",self.code)
        self.assertIn("completed_blind_confirm.value",self.code)
        self.assertEqual(self.code.count("widgets.Checkbox(value=False"),2)
        self.assertIn("'assisted_completion'",self.code)
        self.assertNotIn("widgets.Dropdown",self.code)
        self.assertNotIn("forms =",self.code)
        self.assertNotIn("selector =",self.code)
        self.assertIn("step 4",self.source.lower().replace("paso 4","step 4"))
        self.assertNotIn("case_004_LEFT",self.code)
        self.assertNotIn("case_006_LEFT",self.code)
        self.assertNotIn("case_010_LEFT",self.code)

    def test_closure_cell_stops_before_any_write_unless_both_human_confirmations_are_true(self):
        last_code = next(cell for cell in reversed(self.notebook['cells']) if cell['cell_type']=='code')
        for tests, confirmed, blind in ((False,True,True),(True,False,True),(True,True,False)):
            namespace = {'completed_tests_passed':tests,
                'completed_confirm_all':SimpleNamespace(value=confirmed),
                'completed_blind_confirm':SimpleNamespace(value=blind)}
            with self.subTest(tests=tests,confirmed=confirmed,blind=blind), self.assertRaisesRegex(RuntimeError,'confirmaciones'):
                exec(''.join(last_code['source']),namespace)
            self.assertNotIn('completed_response',namespace)

    def test_existing_environment_is_checked_and_original_globals_not_reset(self):
        self.assertIn("globals().get('venv_python')",self.code)
        self.assertIn("completed_environment[key] != value",self.code)
        self.assertIn("completed_tests_passed = False",self.code)
        self.assertNotIn("REPO_DIR =",self.code)
        self.assertNotIn("context =",self.code.replace("completed_original_context =",""))

    def test_confirmed_closure_cell_writes_parseable_json_and_passes_exact_proposals(self):
        last_code = next(cell for cell in reversed(self.notebook['cells']) if cell['cell_type']=='code')
        prepared = {'draft_csv_sha256':'synthetic-draft', 'updates':{'synthetic':{'decision':'no_evaluable'}},
                    'assisted_completion':{'csv_sha256':'synthetic-preparation'}}
        calls = []
        with tempfile.TemporaryDirectory(dir=ROOT) as task:
            def review_call(action, *arguments):
                calls.append((action, arguments))
                response = json.loads(Path(arguments[1]).read_text(encoding='utf-8'))
                self.assertEqual(response, {**prepared, 'confirm_entire_assisted_review':True,
                    'outcome_blinded':True, 'technical_tests_passed':True})
                self.assertTrue(Path(arguments[1]).read_bytes().endswith(b'\n'))
                return {'status':'rejected_after_technical_review', 'step_4_status':'open'}
            namespace = {'completed_tests_passed':True, 'json':json,
                'completed_confirm_all':SimpleNamespace(value=True),
                'completed_blind_confirm':SimpleNamespace(value=True),
                'completed_prepared':prepared, 'completed_task_dir':Path(task),
                'completed_git_commit':'a'*40, 'completed_review_call':review_call}
            with redirect_stdout(io.StringIO()):
                exec(''.join(last_code['source']), namespace)
            self.assertEqual(calls[0][0], 'close')
            self.assertEqual(calls[0][1][2:], ('--git-commit', 'a'*40))
            self.assertEqual(namespace['completed_closed_summary']['step_4_status'], 'open')

    def test_version_label_matches_prepared_notebook(self):
        self.assertEqual(self.notebook['metadata']['colab']['name'],
                         '14_revision_tecnica_y_cierre_regresion_mcr004_v1.1.ipynb')
        self.assertIn('Versión 1.1', self.source)


if __name__=="__main__":
    unittest.main()
