"""Execution boundaries of diagnostic notebook 15 using synthetic output only."""

import ast
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class Mcr004DiagnosticNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(
            (ROOT / "notebooks" / "15_diagnostico_factibilidad_mcr004.ipynb")
            .read_text(encoding="utf-8")
        )
        cls.source = "\n".join("".join(cell["source"]) for cell in cls.notebook["cells"])
        cls.code_cells = [
            "".join(cell["source"])
            for cell in cls.notebook["cells"] if cell["cell_type"] == "code"
        ]
        cls.code = "\n".join(cls.code_cells)
        cls.run_cell = next(
            code for code in cls.code_cells if "diag_raw_summary =" in code
        )

    def test_cells_compile_and_delivery_contains_no_executed_or_private_data(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook15_cell_{index}", "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        self.assertNotIn("/content/drive/MyDrive/", self.source)
        self.assertNotIn("C:\\Users\\", self.source)
        self.assertNotIn("case_00", self.source)
        self.assertIn("userdata.get('KNEE_DATA_ROOT')", self.code)
        self.assertIn("is_relative_to(diag_private_root)", self.code)

    def test_version_and_environment_contract(self):
        self.assertEqual(
            self.notebook["metadata"]["colab"]["name"],
            "15_diagnostico_factibilidad_mcr004_v1.0.ipynb",
        )
        self.assertIn("Versión 1.0", self.source)
        self.assertIn("Colab 2026.07", self.source)
        self.assertIn("sys.version_info[:2] != (3, 12)", self.code)
        self.assertIn("virtualenv==21.7.11", self.code)
        self.assertIn("--no-periodic-update", self.code)
        self.assertNotIn("'-m', 'venv'", self.code)
        self.assertIn("requirements-mcr004.txt", self.code)
        self.assertIn("diag_environment[key] != value", self.code)
        self.assertIn("test_roi_mcr004*.py", self.code)
        self.assertIn("Google Colab, libreta 15", self.code)

    def test_only_diagnostic_entrypoint_and_fresh_code_provenance(self):
        self.assertIn("'knee.roi_mcr004_diagnostic'", self.code)
        self.assertIn("'--git-commit', diag_git_commit", self.code)
        self.assertIn("'git', 'rev-parse', 'HEAD'", self.code)
        self.assertIn("'git', 'status', '--porcelain'", self.code)
        self.assertIn("'src/knee/roi_mcr004_diagnostic.py'", self.code)
        self.assertIn("'tests/test_roi_mcr004_diagnostic.py'", self.code)
        for forbidden in (
            "widgets.", "close_review(", "run_historical_regression(",
            "select_new_participants", "'--action'", "shutil.rmtree",
            ".unlink(", ".write_text(", ".write_bytes(",
        ):
            self.assertNotIn(forbidden, self.code)
        self.assertIn("sin reducción", self.source.lower().replace("a tamaño natural", "sin reducción"))
        self.assertIn("no necesitas asignar nuevas decisiones", self.source.lower())

    def test_run_cell_stops_before_runner_when_tests_missing_or_output_exists(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as task:
            output = Path(task) / "diagnostic"
            cases = ((False, False, RuntimeError), (True, True, FileExistsError))
            for tests_passed, create_output, error in cases:
                if create_output:
                    output.mkdir()
                    (output / "preserved.txt").write_text("original", encoding="utf-8")
                calls = []
                namespace = {
                    "diag_tests_passed": tests_passed,
                    "diag_output_dir": output,
                    "diag_run_checked": lambda *args, **kwargs: calls.append(args),
                }
                with self.subTest(tests=tests_passed, exists=create_output):
                    with self.assertRaises(error):
                        exec(self.run_cell, namespace)
                    self.assertFalse(namespace["diag_completed"])
                    self.assertEqual(calls, [])
            self.assertEqual(
                (output / "preserved.txt").read_text(encoding="utf-8"), "original"
            )

    def execution_namespace(self, task, *, saved_differs=False, incomplete=False):
        output = Path(task) / "diagnostic"
        summary = {"operation": "synthetic_diagnostic", "step_4_status": "open"}
        calls = []

        def run_checked(command, cwd=None):
            calls.append((command, cwd))
            output.mkdir()
            (output / "diagnostico_privado.json").write_text("{}", encoding="utf-8")
            (output / "registro_diagnostico.json").write_text("{}", encoding="utf-8")
            if not incomplete:
                (output / "galeria_diagnostica.html").write_text(
                    '<div style="overflow:auto"><img src="data:image/png;base64,SYNTHETIC"></div>',
                    encoding="utf-8",
                )
            saved = {**summary, "changed": True} if saved_differs else summary
            (output / "resumen_diagnostico_publico.json").write_text(
                json.dumps(saved), encoding="utf-8"
            )
            return json.dumps(summary)

        namespace = {
            "diag_tests_passed": True,
            "diag_output_dir": output,
            "diag_python": Path(task) / "python",
            "diag_config_path": Path(task) / "config.json",
            "diag_output_relative": "outputs/synthetic_diagnostic",
            "diag_git_commit": "a" * 40,
            "diag_repo_dir": Path(task),
            "diag_run_checked": run_checked,
            "json": json,
        }
        return namespace, calls, summary

    def test_successful_diagnostic_passes_exact_paths_and_verifies_saved_summary(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as task:
            namespace, calls, summary = self.execution_namespace(task)
            with redirect_stdout(io.StringIO()):
                exec(self.run_cell, namespace)
            self.assertTrue(namespace["diag_completed"])
            self.assertEqual(namespace["diag_summary"], summary)
            self.assertEqual(namespace["diag_saved_summary"], summary)
            self.assertEqual(calls, [([
                namespace["diag_python"], "-m", "knee.roi_mcr004_diagnostic",
                "--config", namespace["diag_config_path"],
                "--output-dir", "outputs/synthetic_diagnostic",
                "--git-commit", "a" * 40,
            ], Path(task))])

    def test_partial_or_inconsistent_output_is_preserved_and_not_marked_complete(self):
        for options, message in (
            ({"incomplete": True}, "incompleta"),
            ({"saved_differs": True}, "no coincide"),
        ):
            with self.subTest(options=options), tempfile.TemporaryDirectory(dir=ROOT) as task:
                namespace, calls, _ = self.execution_namespace(task, **options)
                with self.assertRaisesRegex(RuntimeError, message):
                    exec(self.run_cell, namespace)
                self.assertFalse(namespace["diag_completed"])
                self.assertEqual(len(calls), 1)
                self.assertTrue(namespace["diag_output_dir"].is_dir())
                self.assertTrue(
                    (namespace["diag_output_dir"] / "registro_diagnostico.json").is_file()
                )

    def test_subprocess_error_prints_internal_error_and_return_code(self):
        tree = ast.parse(self.code_cells[0])
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "diag_run_checked"
        )
        fake_process = SimpleNamespace(
            returncode=7, stdout="synthetic output", stderr="specific internal failure"
        )
        namespace = {"subprocess": SimpleNamespace(run=lambda *args, **kwargs: fake_process)}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "<helper>", "exec"), namespace)
        capture = io.StringIO()
        with redirect_stdout(capture), self.assertRaisesRegex(RuntimeError, "se detuvo"):
            namespace["diag_run_checked"](["synthetic"])
        self.assertIn("7", capture.getvalue())
        self.assertIn("specific internal failure", capture.getvalue())
        self.assertIn("synthetic output", capture.getvalue())

    def test_gallery_displays_verified_html_without_resizing_or_writing(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as task:
            namespace, _, _ = self.execution_namespace(task)
            displayed = []
            namespace.update(display=displayed.append, HTML=lambda value: value)
            with redirect_stdout(io.StringIO()):
                exec(self.run_cell, namespace)
                exec(self.code_cells[-1], namespace)
            self.assertEqual(len(displayed), 1)
            self.assertIn('style="overflow:auto"', displayed[0])
            self.assertNotIn("width=", self.code_cells[-1])
            self.assertNotIn(".resize(", self.code_cells[-1])
            namespace["diag_completed"] = False
            with self.assertRaisesRegex(RuntimeError, "diagnóstico completo"):
                exec(self.code_cells[-1], namespace)
            self.assertEqual(len(displayed), 1)


if __name__ == "__main__":
    unittest.main()
