"""Structural checks for the CPU clinical campaign notebook, without training."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("clinical_notebook37_builder",
    PROJECT / "scripts/build_clinical_fivefold_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ClinicalFivefoldNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = builder.build("a" * 40)
        self.code_cells = [cell["source"] for cell in self.notebook["cells"] if cell["cell_type"] == "code"]
        self.source = "\n".join(self.code_cells)
        self.full = "\n".join(cell["source"] for cell in self.notebook["cells"])

    def test_exact_published_revision_required(self):
        for revision in (None, "main", "a" * 39, "0" * 40, "A" * 40, builder.UNPUBLISHED):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "published_exact"):
                builder.build(revision)
        tree = ast.parse(self.code_cells[0])
        revision = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "CODE_REVISION" for target in node.targets))
        self.assertEqual(revision, "a" * 40)

    def test_unpublished_draft_requires_explicit_flag_and_has_runtime_rejection(self):
        draft = builder.build(builder.UNPUBLISHED, allow_unpublished_placeholder=True)
        self.assertIn(repr(builder.UNPUBLISHED), draft["cells"][1]["source"])
        self.assertIn("Esta copia no tiene una revisión publicada", draft["cells"][1]["source"])

    def test_all_cells_are_unexecuted_and_python_is_valid(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["nbformat_minor"], 5)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_cpu_bootstrap_preserves_host_and_session_guards(self):
        bootstrap = self.code_cells[0]
        self.assertNotIn("accelerator", self.notebook["metadata"])
        self.assertEqual(self.notebook["metadata"]["kernelspec"]["name"], "python3")
        for required in ("os.environ.get('COLAB_GPU', '0') == '1'", "COLAB_TPU_ADDR", "COLAB_RELEASE_TAG",
                "'knee' in sys.modules", "knee_clinical37_", "clinical_fivefold_colab as orchestration"):
            self.assertIn(required, bootstrap)

    def test_secure_repository_archive_is_bound_to_one_commit(self):
        bootstrap = self.code_cells[0]
        for required in ("https://codeload.github.com/AlonsoYess/Knee/zip/", "128 * 1024**2",
                "512 * 1024**2", "len(members) > 20000", "stat.S_ISLNK", "destination.open('xb')",
                "response.geturl() != url", "parts[0] != 'Knee-' + CODE_REVISION"):
            self.assertIn(required, bootstrap)
        self.assertNotIn("extractall", bootstrap)

    def test_only_registered_clinical_scope_and_inputs_are_described(self):
        for required in ("regresión logística", "XGBoost", "MLP", "15 trabajos", "cinco pliegues",
                "2026", "edad, sexo e IMC", "dentro del entrenamiento de cada pliegue",
                "No se abre la prueba reservada", "integridad de sus imágenes", "únicamente las variables clínicas"):
            self.assertIn(required, self.full)

    def test_no_gpu_weights_or_private_source_paths_are_embedded(self):
        for forbidden in ("D:\\", "C:\\", "@gmail.com", "MyDrive/KneeOAI", "reserved_test_package.zip",
                "weights-root", "WEIGHTS_ROOT", "gpu_runtime", "--allow-download", "refs/heads",
                "1256", "1.256", "1721", "1.721"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.full)

    def test_cpu_setup_tests_and_lineage_use_the_new_cli(self):
        self.assertEqual(self.source.count("'--operation', 'setup'"), 1)
        self.assertEqual(self.source.count("'--operation', 'tests'"), 1)
        self.assertEqual(self.source.count("'--operation', 'locate'"), 1)
        self.assertEqual(self.source.count("'--operation', 'train'"), 1)
        for required in ("scripts/train_clinical_fivefold_colab.py", "cpu_runtime", "cpu_setup_receipt.json",
                "CPU_TESTS['tests_run']", "CPU_TESTS['skipped']", "orchestration.runtime_environment(REPO_DIR)"):
            self.assertIn(required, self.source)
        self.assertNotIn("scripts/train_densenet", self.source)

    def test_same_two_drive_folders_and_exact_preparation_lineage(self):
        self.assertEqual(self.source.count("input("), 3)
        for required in ("con los registros privados de la libreta 34", "que contiene transport.json",
                "PREPARATION_DIR = Path(proof['preparation_dir'])", "orchestration.stage_development",
                "model_inputs_loading_reference_v1.json", "is_relative_to(Path('/content/drive'))"):
            self.assertIn(required, self.source)

    def test_one_explicit_confirmation_precedes_all_training(self):
        training = next(cell for cell in self.code_cells if "confirmation = input(" in cell)
        parsed = ast.parse(training)
        self.assertIsInstance(parsed.body[0], ast.Assign)
        self.assertIsInstance(parsed.body[1], ast.If)
        self.assertEqual(parsed.body[1].test.ops[0].__class__, ast.NotEq)
        self.assertIsInstance(parsed.body[1].body[0], ast.Raise)
        self.assertIn("AUTORIZO CLINICOS 5 PLIEGUES", parsed.body[0].value.func.value.args[0].value)
        self.assertLess(training.index("if confirmation !="), training.index("training_log = training_run("))

    def test_scientific_training_is_not_duplicated_inside_notebook(self):
        for forbidden in (".fit(", ".backward(", "torch.optim", "LogisticRegression(", "XGBClassifier(",
                "train_neural_fold(", "fit_clinical_fold(", "median(", "clinical_state.json"):
            self.assertNotIn(forbidden, self.source)

    def test_training_cli_receives_proof_and_private_persistence(self):
        for required in ("'--preparation-dir', str(PREPARATION_DIR)", "'--package-dir', str(LOCAL_PACKAGE_DIR)",
                "'--reference-path', str(LOCAL_REFERENCE)", "'--records-root', str(PERSISTENT_ROOT)",
                "'--code-revision', CODE_REVISION", "'--test-receipt', str(TEST_RECEIPT)",
                "'--runtime-receipt', str(RUNTIME_RECEIPT)", "'--approval-statement', orchestration.APPROVAL_STATEMENT"):
            self.assertIn(required, self.source)
        for forbidden in ("--model-id", "--validation-fold", "--epochs", "--seed", "--maximum-steps"):
            self.assertNotIn(forbidden, self.source)

    def test_streaming_progress_is_real_and_interruption_terminates_subprocess(self):
        for required in ("subprocess.Popen", "stdout=subprocess.PIPE, stderr=subprocess.PIPE",
                "time.monotonic()", "process.poll()", "process.terminate()", "process.wait(timeout=30)",
                "except subprocess.TimeoutExpired", "process.kill()", "No es un porcentaje de avance"):
            self.assertIn(required, self.source)
        self.assertIn("<url>", self.source)
        self.assertIn("<redacted>", self.source)

    def test_reconnection_policy_and_oof_metric_roles_are_explicit(self):
        for required in ("Cada trabajo completo", "no ejecutes dos sesiones simultáneas",
                "un trabajo interrumpido se reinicia", "No se retoma un optimizador parcial",
                "media no ponderada", "no sustituyen la métrica de selección", "JSON final"):
            self.assertIn(required, self.full)

    def test_final_record_is_validated_before_its_display(self):
        self.assertLess(self.source.index("orchestration.validate_result(FINAL_CHECK)"),
            self.source.index("print(json.dumps(FINAL_CHECK"))
        self.assertIn("orchestration.SUMMARY_FILENAME", self.source)

    def test_build_is_deterministic_and_cli_writes_only_requested_output(self):
        self.assertEqual(self.notebook, builder.build("a" * 40))
        with tempfile.TemporaryDirectory(prefix="clinical37_notebook_test_") as temporary:
            output = Path(temporary) / "notebook.ipynb"
            with patch.object(sys, "argv", ["build", "--code-revision", "b" * 40, "--output", str(output)]):
                builder.main()
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), builder.build("b" * 40))
            self.assertNotIn(b"\r\n", output.read_bytes())
            self.assertEqual({path.name for path in Path(temporary).iterdir()}, {"notebook.ipynb"})


if __name__ == "__main__":
    unittest.main()
