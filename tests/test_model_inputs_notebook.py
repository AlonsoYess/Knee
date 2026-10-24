"""Static executable-notebook checks; no private input or runtime install."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest


class ModelInputsNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[1]
        cls.notebook_path = cls.project / "notebooks/32_comprobar_entradas_modelos_sin_entrenar.ipynb"
        cls.notebook = json.loads(cls.notebook_path.read_text(encoding="utf-8"))
        cls.sources = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                       for c in cls.notebook["cells"] if c["cell_type"] == "code"]
        cls.code = "\n".join(cls.sources)
        cls.trees = [ast.parse(source) for source in cls.sources]

    def assignments(self, name):
        return [node.value for tree in self.trees for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]

    def test_every_code_cell_compiles_and_notebook_contains_no_executed_output(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs"), [])
                self.assertIsNone(cell.get("execution_count"))
        for index, source in enumerate(self.sources):
            compile(source, f"notebook_code_{index}", "exec")

    def test_revision_is_an_actual_nonzero_full_commit_pin(self):
        values = self.assignments("CODE_REVISION")
        self.assertEqual(len(values), 1)
        revision = ast.literal_eval(values[0])
        self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertNotEqual(revision, "0" * 40)
        self.assertIn("rev-parse", self.code)
        self.assertIn("--detach", self.code)

    def test_exact_positive_test_count_and_fixed_six_test_modules_are_declared(self):
        count = ast.literal_eval(self.assignments("EXPECTED_TESTS")[0])
        self.assertIs(type(count), int)
        self.assertGreater(count, 0)
        modules = ast.literal_eval(self.assignments("TEST_MODULES")[0])
        self.assertEqual(modules, ["test_clinical_fold_inputs", "test_model_input_adapters",
            "test_model_inputs_preflight", "test_colab_package_transport",
            "test_model_inputs_colab", "test_model_inputs_cpu_setup"])

    def test_effective_unittest_count_regex_matches_real_log_digits(self):
        patterns = [node.args[0].value for tree in self.trees for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "findall" and node.args and isinstance(node.args[0], ast.Constant)]
        self.assertEqual(len(patterns), 1)
        self.assertEqual(re.findall(patterns[0], "Ran 107 tests in 1.2s"), ["107"])

    def test_final_record_write_uses_real_newline_not_literal_backslash_n(self):
        constants = [node.value for tree in self.trees for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertIn("\n", constants)
        self.assertNotIn("\\n", constants)

    def test_cpu_gpu_tpu_guards_and_offline_pretrained_environment_are_present(self):
        for token in ("COLAB_GPU", "COLAB_TPU_ADDR", "TPU_NAME", "COLAB_TPU_TASK_ID",
                      "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONNOUSERSITE"):
            self.assertIn(token, self.code)
        self.assertIn("setup_model_inputs_cpu.py", self.code)
        self.assertIn("NEURAL_PYTHON", self.code)

    def test_reference_is_copied_by_hash_without_editing_transport_sources(self):
        self.assertIn("KNEE_INPUT_REFERENCE_PATH", self.code)
        self.assertIn("stage_development_package", self.code)
        self.assertIn("REFERENCE_HASH", self.code)
        self.assertIn("LOCAL_REFERENCE", self.code)
        self.assertIn("with REFERENCE_SOURCE.open('rb')", self.code)
        self.assertNotIn("REMOTE_PACKAGE_DIR.unlink", self.code)
        self.assertNotIn("REFERENCE_SOURCE.write", self.code)

    def test_local_rehearsal_publishes_under_owned_session_and_is_explicit(self):
        self.assertIn("globals().get('_LOCAL_REHEARSAL') is True", self.code)
        self.assertIn("SESSION_ROOT if EXECUTION_ENVIRONMENT == 'local_rehearsal' else REMOTE_PACKAGE_DIR", self.code)
        self.assertIn("'colab_runtime_executed': loading['colab_runtime_executed']", self.code)

    def test_no_private_ids_local_paths_or_hardcoded_cohort_counts(self):
        self.assertNotRegex(self.code, r"(?<![a-zA-Z0-9_])\d{7}(?![a-zA-Z0-9_])")
        self.assertNotRegex(self.code, r"[CD]:[/\\]")
        count_keys = {"development_participants", "development_knees",
                      "clinical_folds_verified", "loader_row_visits_verified"}
        values = {}
        for tree in self.trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and key.value in count_keys:
                            values[key.value] = value
        self.assertEqual(set(values), count_keys)
        for value in values.values():
            self.assertIsInstance(value, ast.Subscript)

    def test_only_input_check_cli_is_launched_without_training_or_augmentation(self):
        self.assertIn("check_model_inputs_colab.py", self.code)
        for forbidden in ("torch.optim", ".backward(", "model.train(", "RandomResizedCrop(",
                          "RandomHorizontalFlip(", "torch.hub.load(", "hf_hub_download("):
            self.assertNotIn(forbidden, self.code)
        self.assertIn("'pretrained_weights_loaded': False", self.code)
        self.assertIn("'training_executed': False", self.code)

    def test_harness_uses_fresh_isolated_worker_and_preserves_input_snapshots(self):
        harness = self.project / "scripts/rehearse_model_inputs_colab_notebook.py"
        source = harness.read_text(encoding="utf-8")
        compile(source, str(harness), "exec")
        self.assertIn('"-I", "-B"', source)
        self.assertIn('"_LOCAL_REHEARSAL": True', source)
        self.assertIn("snapshot(transport) == original_transport", source)
        self.assertIn("tempfile.TemporaryDirectory", source)
        self.assertNotIn("shutil.rmtree", source)

    def test_harness_snapshot_detects_content_and_name_changes(self):
        harness = self.project / "scripts/rehearse_model_inputs_colab_notebook.py"
        spec = importlib.util.spec_from_file_location("synthetic_notebook_rehearsal_harness", harness)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="knee_rehearsal_snapshot_synthetic_") as name:
            root = Path(name).resolve()
            path = root / "synthetic.txt"
            path.write_bytes(b"first")
            first = module.snapshot(root)
            path.write_bytes(b"second")
            self.assertNotEqual(module.snapshot(root), first)
            path.rename(root / "renamed.txt")
            self.assertNotEqual(set(module.snapshot(root)), set(first))


if __name__ == "__main__":
    unittest.main()
