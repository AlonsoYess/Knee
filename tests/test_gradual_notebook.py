"""Run notebook42 control cells with inert subprocess and Drive fixtures."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gradual_notebook_builder", ROOT / "scripts/build_gradual_training_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
REVISION = "1234567890abcdef1234567890abcdef12345678"


class GradualNotebookTests(unittest.TestCase):
    def test_clean_pinned_scope_and_no_interactive_prompt(self):
        notebook = builder.build(REVISION)
        self.assertEqual(len(notebook["cells"]), 14)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        intro = notebook["cells"][0]["source"]
        self.assertIn("20 trabajos", intro)
        self.assertIn("Revisión 1.1", intro)
        self.assertIn("etapa 5", intro)
        self.assertEqual([intro.index(name) for name in builder.MODEL_NAMES],
            sorted(intro.index(name) for name in builder.MODEL_NAMES))
        bootstrap = notebook["cells"][1]["source"]
        self.assertIn(repr(REVISION), bootstrap)
        self.assertIn("knee_gradual42_", bootstrap)
        self.assertIn("from knee import gradual_campaign_colab as orchestration", bootstrap)
        self.assertIn("CAMPAIGN = 'gradual_four_models'", bootstrap)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(cell["source"])
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "input" for n in ast.walk(tree)))
            self.assertNotIn("scripts/train_image_screening_colab.py", cell["source"])
            self.assertNotIn("/content/drive/MyDrive/", cell["source"])
        self.assertNotIn("'--operation', 'train'", notebook["cells"][9]["source"])
        self.assertIn("'--operation', 'prepare'", notebook["cells"][9]["source"])
        self.assertIn("'--operation', 'train'", notebook["cells"][11]["source"])

    def test_unpublished_or_unpinned_revision_is_rejected(self):
        for revision in ("", "main", "0" * 40, "../other"):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                builder.build(revision)

    def test_only_synthetic_unit_tests_hide_gpu(self):
        notebook = builder.build(REVISION)
        self.assertIn("TEST_ENV.update(CUDA_VISIBLE_DEVICES='', COLAB_GPU='0')", notebook["cells"][5]["source"])
        self.assertIn("env=TEST_ENV", notebook["cells"][5]["source"])
        for index in (9, 11):
            self.assertIn("env=RUN_ENV", notebook["cells"][index]["source"])
            self.assertNotIn("CUDA_VISIBLE_DEVICES='',", notebook["cells"][index]["source"])

    def _control_cell(self, *, train=False, invalid_campaign=False, invalid_receipt=False):
        notebook = builder.build(REVISION)
        tree = ast.parse(notebook["cells"][11 if train else 9]["source"])
        tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]
        with tempfile.TemporaryDirectory(prefix="gradual_cell_") as directory:
            root = Path(directory)
            paths = {name: root / name for name in ("session", "repository", "development", "reference", "records", "preparation")}
            for path in paths.values():
                path.mkdir()
            reference = paths["reference"] / "reference.json"
            reference.write_text("{}", encoding="utf-8")
            def save(path, value):
                with Path(path).open("x", encoding="utf-8") as stream:
                    json.dump(value, stream)
            summary = {"synthetic": "no_training"}
            def subprocess(arguments, *, cwd, env):
                self.assertEqual(cwd, paths["repository"])
                self.assertEqual(env, {"fixture": "gpu_visible_only_to_future_process"})
                arg = lambda key: arguments[arguments.index(key) + 1]
                self.assertEqual(arg("--campaign"), builder.CAMPAIGN)
                self.assertEqual(arg("--code-revision"), REVISION)
                self.assertEqual(arg("--operation"), "train" if train else "prepare")
                self.assertEqual(Path(arg("--records-root")), paths["records"])
                self.assertEqual(Path(arg("--package-dir")), paths["development"])
                output = Path(arg("--output-dir"))
                output.mkdir()
                if train:
                    self.assertTrue(Path(arg("--weights-root")).is_dir())
                    self.assertEqual(Path(arg("--preparation-dir")), paths["preparation"])
                    save(output / "summary.json", summary)
                else:
                    self.assertNotIn("--weights-root", arguments)
                    self.assertNotIn("--approval-statement", arguments)
                return json.dumps(summary)
            operation = Mock(side_effect=subprocess)
            validator = Mock(side_effect=ValueError("invalid receipt") if invalid_receipt else None)
            orchestration = SimpleNamespace(_save=Mock(side_effect=save),
                _json=lambda p: json.loads(Path(p).read_text()), validate_preparation_receipt=validator,
                validate_result=Mock(), SUMMARY_FILENAME="summary.json")
            input_mock = Mock(side_effect=AssertionError("No prompt permitted"))
            namespace = {"CAMPAIGN": "reserved_test" if invalid_campaign else builder.CAMPAIGN,
                "SESSION_ROOT": paths["session"], "CPU_TESTS": {"synthetic": "tests"},
                "runtime": {"synthetic": "runtime"}, "orchestration": orchestration,
                "NEURAL_PYTHON": Path("synthetic_python"), "REPO_DIR": paths["repository"],
                "LOCAL_PACKAGE_DIR": paths["development"], "LOCAL_REFERENCE": reference,
                "PERSISTENT_ROOT": paths["records"], "PREPARATION_DIR": paths["preparation"],
                "CODE_REVISION": REVISION, "RUN_ENV": {"fixture": "gpu_visible_only_to_future_process"},
                "json": json, "checked_run": operation, "training_run": operation, "input": input_mock}
            with redirect_stdout(io.StringIO()):
                if invalid_campaign or (train and invalid_receipt):
                    with self.assertRaises((RuntimeError, ValueError)):
                        exec(compile(tree, "gradual_cell", "exec"), namespace)
                    operation.assert_not_called()
                    self.assertEqual(list(paths["session"].iterdir()), [])
                else:
                    exec(compile(tree, "gradual_cell", "exec"), namespace)
                    operation.assert_called_once()
                    validator.assert_called_once()
                    if train:
                        self.assertEqual(namespace["FINAL_CHECK"], summary)
                        orchestration._save.assert_not_called()
                    else:
                        self.assertEqual(orchestration._save.call_count, 2)
                        self.assertEqual(namespace["PREPARATION_CHECK"], summary)
                input_mock.assert_not_called()

    def test_preparation_cell_never_launches_training(self):
        self._control_cell()

    def test_training_cell_requires_verified_preparation_and_separate_cache(self):
        self._control_cell(train=True)
        self._control_cell(train=True, invalid_receipt=True)

    def test_wrong_campaign_cannot_prepare_or_train(self):
        self._control_cell(invalid_campaign=True)
        self._control_cell(train=True, invalid_campaign=True)

    def test_automatic_paths_stage_only_development_without_legacy_locator(self):
        from knee.colab_data import checked_path
        source = builder.build(REVISION)["cells"][7]["source"]
        self.assertNotIn("'--operation', 'locate'", source)
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "google.colab"
            or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "mount"
            or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id == "MY_DRIVE_ROOT" for t in node.targets))]
        with tempfile.TemporaryDirectory(prefix="gradual_paths_") as directory:
            root = Path(directory)
            drive = root / "drive"
            project = drive / "KneeOAI"
            records, package, session = project / "records", project / "development", root / "session"
            for path in (records, package, session):
                path.mkdir(parents=True)
            (package / "transport.json").write_text("{}")
            settings = {"version": "knee_colab_runtime_paths_v1.0", "records_relative": "records", "development_relative": "development"}
            (project / "colab_runtime_paths.json").write_text(json.dumps(settings))
            stage = Mock(return_value={"synthetic": True})
            namespace = {"Path": Path, "PurePosixPath": PurePosixPath, "MY_DRIVE_ROOT": drive,
                "SESSION_ROOT": session, "os": SimpleNamespace(environ={}),
                "orchestration": SimpleNamespace(checked_path=checked_path, stage_development=stage,
                    _json=lambda p: json.loads(Path(p).read_text()))}
            with redirect_stdout(io.StringIO()):
                exec(compile(tree, "gradual_paths", "exec"), namespace)
            stage.assert_called_once_with(package, session / "development",
                session / "reference/model_inputs_loading_reference_v1.json")
            self.assertEqual(namespace["PERSISTENT_ROOT"], records)

    def test_cli_exposes_only_bounded_prepare_and_training(self):
        source = (ROOT / "scripts/train_gradual_campaign_colab.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('choices=("setup", "tests", "prepare", "train")', source)
        self.assertIn('choices=("gradual_four_models",)', source)
        self.assertNotIn("input(", source)
        self.assertNotIn("reserved_test", source)


if __name__ == "__main__":
    unittest.main()
