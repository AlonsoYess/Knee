"""Structural and mocked-cell checks for notebook38; no private data or fits."""
from __future__ import annotations

import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("methodology_notebook38_builder",
    PROJECT / "scripts/build_methodology_inputs_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class MethodologyNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = builder.build("a" * 40)
        self.cells = [cell["source"] for cell in self.notebook["cells"] if cell["cell_type"] == "code"]
        self.source = "\n".join(self.cells)
        self.full = "\n".join(cell["source"] for cell in self.notebook["cells"])

    def test_requires_exact_published_revision(self):
        for value in (None, "main", "a" * 39, "0" * 40, "A" * 40, builder.UNPUBLISHED):
            with self.subTest(revision=value), self.assertRaises(ValueError):
                builder.build(value)
        tree = ast.parse(self.cells[0])
        revision = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "CODE_REVISION" for target in node.targets))
        self.assertEqual(revision, "a" * 40)

    def test_unpublished_draft_is_explicit_and_still_fails_at_runtime(self):
        value = builder.build(builder.UNPUBLISHED, allow_unpublished_placeholder=True)
        self.assertIn(repr(builder.UNPUBLISHED), value["cells"][1]["source"])
        self.assertIn("Esta copia no tiene una revisión publicada", value["cells"][1]["source"])

    def test_unexecuted_valid_python_and_cpu_metadata(self):
        self.assertEqual((self.notebook["nbformat"], self.notebook["nbformat_minor"]), (4, 5))
        self.assertNotIn("accelerator", self.notebook["metadata"])
        self.assertEqual(self.notebook["metadata"]["kernelspec"]["name"], "python3")
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_bootstrap_requires_cpu_colab_and_fresh_session(self):
        for token in ("os.environ.get('COLAB_GPU', '0') == '1'", "COLAB_TPU_ADDR", "TPU_NAME",
                "COLAB_RELEASE_TAG", "'knee' in sys.modules", "knee_methodology38_",
                "from knee import methodology_inputs_colab as orchestration"):
            self.assertIn(token, self.cells[0])

    def test_secure_repository_archive_remains_bound_to_commit(self):
        for token in ("https://codeload.github.com/AlonsoYess/Knee/zip/", "response.geturl() != url",
                "128 * 1024**2", "512 * 1024**2", "len(members) > 20000", "stat.S_ISLNK",
                "parts[0] != 'Knee-' + CODE_REVISION", "destination.open('xb')"):
            self.assertIn(token, self.cells[0])
        self.assertNotIn("extractall", self.cells[0])

    def test_setup_and_tests_use_separate_cpu_interpreter(self):
        self.assertIn("'--operation', 'setup'", self.cells[1])
        self.assertIn("NEURAL_PYTHON = Path(runtime['python_path'])", self.cells[1])
        self.assertIn("RUN_ENV = orchestration.runtime_environment(REPO_DIR)", self.cells[1])
        self.assertIn("str(NEURAL_PYTHON), '-B'", self.cells[2])
        self.assertIn("'--operation', 'tests'", self.cells[2])
        self.assertIn("env=RUN_ENV", self.cells[2])

    def test_only_two_existing_drive_folder_prompts(self):
        inputs = [node for node in ast.walk(ast.parse(self.source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input"]
        self.assertEqual(len(inputs), 2)
        stage = self.cells[3]
        self.assertIn("not selected.is_dir()", stage)
        self.assertIn("selected.is_relative_to(Path('/content/drive'))", stage)
        self.assertIn("orchestration.stage_development(REMOTE_PACKAGE_DIR, LOCAL_PACKAGE_DIR, LOCAL_REFERENCE)", stage)
        self.assertIn("orchestration.stage_baseline(REMOTE_PACKAGE_DIR, LOCAL_BASELINE_DIR)", stage)
        self.assertIn("baseline['projection_path']", stage)
        self.assertIn("baseline['descriptor_path']", stage)
        self.assertNotIn("rglob", stage)

    def test_verify_delegates_to_closed_cpu_operation_and_validates_record(self):
        verify = self.cells[4]
        for token in ("'--operation', 'verify'", "'--package-dir'", "'--reference-path'",
                "'--projection-path'", "'--descriptor-path'", "'--output-dir'",
                "'--execution-environment', 'google_colab_cpu'", "orchestration.CORE_FILENAME",
                "orchestration.validate_result(result, 'google_colab_cpu')"):
            self.assertIn(token, verify)

    def test_no_predictive_training_weight_or_test_option(self):
        for token in ("--train", "--approval-statement", "AUTORIZO", "--weights-root", "--allow-download",
                "reserved_test_package.zip", "--reserved-test", "training_run("):
            self.assertNotIn(token, self.source)
        for node in ast.walk(ast.parse(self.source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"fit", "backward", "step", "execute", "train_neural_fold"})

    def test_previews_are_only_verified_fixed_output_paths(self):
        view = self.cells[5]
        self.assertIn("for relative in orchestration.PREVIEW_FILES", view)
        self.assertIn("LOCAL_OUTPUT / relative", view)
        self.assertIn("not preview.is_relative_to(LOCAL_OUTPUT)", view)
        self.assertIn("not preview.is_file()", view)
        self.assertIn("display(Image(filename=str(preview)))", view)
        self.assertNotIn("glob", view)

    def test_saved_receipts_and_final_summary_are_validated(self):
        save = self.cells[6]
        for token in ("orchestration._save(TEST_RECEIPT, CPU_TESTS)", "orchestration._save(RUNTIME_RECEIPT, runtime)",
                "'--operation', 'save'", "'--records-root', str(PERSISTENT_ROOT)",
                "'--code-revision', CODE_REVISION", "'--test-receipt'", "'--runtime-receipt'",
                "orchestration.validate_summary(FINAL_CHECK)", "json.dumps(FINAL_CHECK"):
            self.assertIn(token, save)

    def test_scope_and_limitations_explained_without_execution_claims(self):
        for token in ("KL inicial", "cirugía previa", "WOMAC", "edad, sexo e IMC", "cinco pliegues",
                "solo dentro de su entrenamiento", "No se entrena ningún modelo", "no se abre la prueba reservada",
                "ni demuestra que el aumento vaya a mejorar", "JSON final"):
            self.assertIn(token, self.full)
        for token in ("D:\\", "C:\\", "@gmail.com", "MyDrive/KneeOAI", "1cdBgbEm"):
            self.assertNotIn(token, self.full)

    def test_builder_has_no_network_or_data_side_effect(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), \
             patch("subprocess.run", side_effect=AssertionError("subprocess forbidden")):
            value = builder.build("b" * 40)
        self.assertEqual(len(value["cells"]), len(self.notebook["cells"]))

    def test_cli_writes_stable_utf8_lf_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "notebook.ipynb"
            with patch.object(sys, "argv", ["builder", "--code-revision", "a" * 40, "--output", str(output)]):
                builder.main()
            raw = output.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(json.loads(raw), self.notebook)

    def test_mocked_post_bootstrap_cells_complete_without_training(self):
        """Execute every preparation cell against a closed synthetic transport."""
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session"
            session.mkdir()
            repository = session / "repository"
            repository.mkdir()
            records = Path(directory) / "records"
            records.mkdir()
            remote = Path(directory) / "remote_development"
            remote.mkdir()
            previews = tuple(f"augmentation_previews/comparison_{i:02d}.png" for i in range(1, 4))
            core_filename = "methodology_inputs_preflight.json"
            core = {"version": "synthetic38", "status": "passed", "training_executed": False}
            final = {**core, "result_files_saved_and_readback_verified": True}
            operations = []
            class DrivePath(type(Path())):
                def is_relative_to(self, *other):
                    if len(other) == 1 and str(other[0]) == str(Path('/content/drive')):
                        return True
                    return super().is_relative_to(*other)
            def save(path, value):
                Path(path).write_text(json.dumps(value), encoding="utf-8")
            def stage_package(incoming, output, reference):
                self.assertEqual(incoming, remote)
                output.mkdir()
                save(reference, {"synthetic_reference": True})
                return {"development_only": True}
            def stage_baseline(incoming, output):
                self.assertEqual(incoming, remote)
                output.mkdir()
                projection, descriptor = output / "projection.json", output / "descriptor.json"
                save(projection, [])
                save(descriptor, {"synthetic_only": True})
                return {"projection_path": str(projection), "descriptor_path": str(descriptor)}
            orchestration = types.SimpleNamespace(checked_path=lambda p: DrivePath(p),
                runtime_environment=Mock(return_value={"CUDA_VISIBLE_DEVICES": ""}),
                stage_development=Mock(side_effect=stage_package), stage_baseline=Mock(side_effect=stage_baseline),
                validate_result=Mock(), validate_summary=Mock(), PREVIEW_FILES=previews, CORE_FILENAME=core_filename,
                _save=save)
            def run(arguments, **kwargs):
                operation = arguments[arguments.index("--operation") + 1]
                operations.append(operation)
                if operation == "setup":
                    return json.dumps({"python_path": str(session / "isolated_python"), "cpu_only": True})
                if operation == "tests":
                    return json.dumps({"passed": True, "tests_run": 9, "skipped": 0})
                if operation == "verify":
                    output = Path(arguments[arguments.index("--output-dir") + 1])
                    output.mkdir()
                    save(output / core_filename, core)
                    for name in previews:
                        target = output / name
                        target.parent.mkdir(exist_ok=True)
                        target.write_bytes(b"SYNTHETIC PREVIEW; NEVER A PARTICIPANT IMAGE")
                    return ""
                self.assertEqual(operation, "save")
                return json.dumps(final)
            display = Mock()
            image = Mock(side_effect=lambda **kwargs: kwargs)
            google, colab, ipython_display = types.ModuleType("google"), types.ModuleType("google.colab"), types.ModuleType("IPython.display")
            colab.drive = types.SimpleNamespace(mount=Mock())
            google.colab = colab
            ipython_display.display, ipython_display.Image = display, image
            namespace = {"Path": Path, "json": json, "sys": sys, "SESSION_ROOT": session,
                "REPO_DIR": repository, "CODE_REVISION": "a" * 40, "orchestration": orchestration, "checked_run": run}
            with patch.dict(sys.modules, {"google": google, "google.colab": colab, "IPython.display": ipython_display}), \
                 patch("builtins.input", side_effect=[str(records), str(remote)]), patch("sys.stdout", io.StringIO()):
                for source in self.cells[1:]:
                    exec(compile(source, "<synthetic_notebook38_cell>", "exec"), namespace)
            self.assertEqual(operations, ["setup", "tests", "verify", "save"])
            orchestration.validate_result.assert_called_once_with(core, "google_colab_cpu")
            orchestration.validate_summary.assert_called_once_with(final)
            self.assertEqual(display.call_count, 3)
            self.assertEqual(namespace["FINAL_CHECK"], final)
            self.assertFalse(final["training_executed"])
            self.assertTrue((session / "synthetic_test_receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
