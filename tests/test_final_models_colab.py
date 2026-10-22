"""Synthetic-only notebook33 orchestration checks; never install or download."""
from contextlib import redirect_stdout
from copy import deepcopy
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from knee import final_models_colab as subject


PROJECT = Path(__file__).resolve().parents[1]


def imported_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = imported_script("synthetic_notebook33_builder", "build_final_models_notebook.py")
cli = imported_script("synthetic_notebook33_cli", "check_final_models_colab.py")


def synthetic_result(environment="google_colab_cpu"):
    models = {}
    for name in ("image_densenet121", "image_vit_b16", "image_convnextv2_tiny", "image_skelex",
                 "image_biomedclip", "image_mambaout_tiny"):
        models[name] = {"status": "passed_synthetic_cpu_checkpoint_verification", "checkpoint_loaded": True,
            "repeated_features_exact": True, "parameters_and_buffers_unchanged": True,
            "input_and_rng_unchanged": True, "gradients_created": False}
    models["image_dinov3_vits16"] = {"status": "pending_authorized_access", "checkpoint_loaded": False}
    return {"version": subject.CORE_VERSION, "status": subject.CORE_STATUS,
        "execution_environment": environment, "colab_runtime_executed": environment == "google_colab_cpu",
        "registered_image_models": 7, "pretrained_models_loaded": 6, "model_records": models,
        "synthetic_inputs_only": True, **{key: False for key in subject.FALSE_FLAGS},
        "source_files_sha256": {"synthetic.py": "a" * 64}}


class FinalModelsColabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="final_models_colab_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repository = self.root / "repository"
        self.repository.mkdir()

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def setup_repository(self):
        for relative in ("scripts/setup_model_inputs_cpu.py", "configs/model_inputs_colab_runtime_v1.json",
            "requirements-colab-model-inputs.txt", "src/knee/final_models_colab.py"):
            path = self.repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic fixture", encoding="utf-8")
        (self.repository / "requirements-final-models.txt").write_text(
            "timm==1.0.15\ntransformers==4.50.2\n", encoding="utf-8")
        base = Mock(EXPECTED_PACKAGES={"torch": "2.6.0+cpu", "torchvision": "0.21.0+cpu"})
        base.isolated_environment.return_value = {"PATH": "synthetic", "HF_HUB_OFFLINE": "1"}
        def setup(destination, repository):
            exe = destination / "runtime/python/bin/python3.12"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"synthetic isolated executable")
            return {"python_path": str(exe)}
        base.setup.side_effect = setup
        return base

    def test_exact_revision_or_unpublished_is_rejected(self):
        for value in ("main", "HEAD", "a" * 39, "G" * 40, "0" * 40, builder.UNPUBLISHED, None):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                subject._revision(value)
        self.assertEqual(subject._revision("a" * 40), "a" * 40)

    def test_json_save_never_overwrites_and_readback_matches(self):
        path = self.root / "record.json"
        subject._save(path, {"synthetic": True})
        self.assertEqual(subject._json(path), {"synthetic": True})
        with self.assertRaises(FileExistsError):
            subject._save(path, {"synthetic": False})
        self.assertEqual(subject._json(path), {"synthetic": True})

    def test_json_read_rejects_empty_large_and_missing(self):
        empty, large = self.root / "empty.json", self.root / "large.json"
        empty.touch()
        large.write_bytes(b" " * (4 * 1024**2 + 1))
        for path in (empty, large, self.root / "missing.json"):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                subject._json(path)

    def test_requirement_reader_accepts_only_exact_nonconflicting_pins(self):
        self.setup_repository()
        _, pins = subject._requirements(self.repository, {"torch": "2.6.0+cpu"})
        self.assertEqual(pins, {"timm": "1.0.15", "transformers": "4.50.2"})

    def test_requirement_reader_rejects_urls_unpinned_duplicates_and_cpu_changes(self):
        self.setup_repository()
        target = self.repository / "requirements-final-models.txt"
        prefix = "timm==1.0.15\ntransformers==4.50.2\n"
        for extra in ("--extra-index-url https://example.invalid", "numpy>=1", "timm==1.0.15",
                      "torch==2.7.0", "editable @ https://example.invalid/x", "-e ."):
            target.write_text(prefix + extra, encoding="utf-8")
            with self.subTest(extra=extra), self.assertRaises(RuntimeError):
                subject._requirements(self.repository, {"torch": "2.6.0+cpu"})

    def test_requirement_reader_rejects_missing_encoder_dependency(self):
        self.setup_repository()
        (self.repository / "requirements-final-models.txt").write_text("timm==1.0.15\n")
        with self.assertRaisesRegex(RuntimeError, "dependencies_missing"):
            subject._requirements(self.repository, {})

    def test_cpu_guard_rejects_accelerators_and_fake_colab(self):
        for values in ({"COLAB_GPU": "1"}, {"TPU_NAME": "attached"}, {}):
            with self.subTest(values=values), patch.dict(os.environ, values, clear=True), self.assertRaises(RuntimeError):
                subject._cpu_host("google_colab_cpu")
        with patch.dict(os.environ, {}, clear=True):
            subject._cpu_host("local_cpu")
        with self.assertRaises(RuntimeError):
            subject._cpu_host("colab")

    def test_cpu_guard_accepts_actual_linux_colab_evidence(self):
        with patch.dict(os.environ, {"COLAB_RELEASE_TAG": "synthetic-colab"}, clear=True), \
                patch.object(subject.platform, "system", return_value="Linux"):
            subject._cpu_host("google_colab_cpu")

    def test_environment_is_offline_telemetry_disabled_and_code_paths_explicit(self):
        base = self.setup_repository()
        with patch.object(subject, "_bootstrap", return_value=base):
            env = subject.runtime_environment(self.repository)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
            self.assertEqual(env[key], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep),
            [str(self.repository / "src"), str(self.repository / "tests")])

    def test_setup_uses_only_isolated_python_and_reverifies_existing_runtime(self):
        base = self.setup_repository()
        destination = self.root / "cpu"
        with patch.object(subject, "_cpu_host"), patch.object(subject, "_bootstrap", return_value=base), \
                patch.object(subject, "_verify_installed", return_value={}) as verify, \
                patch.object(subject, "_run", return_value="") as execute:
            first = subject.setup_runtime(self.repository, destination)
            second = subject.setup_runtime(self.repository, destination)
        self.assertFalse(first["existing_runtime_reverified"])
        self.assertTrue(second["existing_runtime_reverified"])
        self.assertEqual(base.setup.call_count, 1)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(execute.call_count, 1)
        command = execute.call_args.args[0]
        self.assertTrue(Path(command[0]).is_relative_to(destination))
        self.assertEqual(command[1:4], ["-I", "-m", "pip"])
        self.assertIn("--no-deps", command)
        self.assertFalse(first["host_python_modified"] or first["training_executed"])

    def test_incomplete_or_changed_runtime_not_silently_reused(self):
        base = self.setup_repository()
        destination = self.root / "cpu"
        destination.mkdir()
        with patch.object(subject, "_cpu_host"), patch.object(subject, "_bootstrap", return_value=base), \
                self.assertRaisesRegex(RuntimeError, "incomplete_runtime"):
            subject.setup_runtime(self.repository, destination)
        self.assertEqual(base.setup.call_count, 0)

    def test_changed_requirements_reject_existing_runtime(self):
        base = self.setup_repository()
        destination = self.root / "cpu"
        with patch.object(subject, "_cpu_host"), patch.object(subject, "_bootstrap", return_value=base), \
                patch.object(subject, "_verify_installed", return_value={}), patch.object(subject, "_run", return_value=""):
            subject.setup_runtime(self.repository, destination)
            with (self.repository / "requirements-final-models.txt").open("a") as stream:
                stream.write("# changed approved input\n")
            with self.assertRaisesRegex(RuntimeError, "receipt_mismatch"):
                subject.setup_runtime(self.repository, destination)

    def test_runtime_inside_repository_or_drive_rejected(self):
        base = self.setup_repository()
        drive = self.root / "drive"
        drive.mkdir()
        with patch.object(subject, "_cpu_host"), patch.object(subject, "_bootstrap", return_value=base):
            for target in (self.repository / "runtime", drive / "runtime"):
                with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, "local_temporary_runtime"):
                    subject.setup_runtime(self.repository, target)
        base.setup.assert_not_called()

    def test_installed_runtime_requires_exact_cpu_versions(self):
        expected = {"torch": "2.6.0+cpu"}
        valid = {"python": "3.12.14", "packages": expected, "cuda_available": False, "cuda_build": None}
        with patch.object(subject, "runtime_environment", return_value={}), \
                patch.object(subject, "_run", side_effect=["", json.dumps(valid)]):
            self.assertEqual(subject._verify_installed(Path("synthetic-python"), expected, self.repository), valid)
        for altered in ({**valid, "cuda_build": "12.4"}, {**valid, "python": "3.13.0"},
                        {**valid, "packages": {"torch": "2.6.0"}}):
            with patch.object(subject, "runtime_environment", return_value={}), \
                    patch.object(subject, "_run", side_effect=["", json.dumps(altered)]), self.assertRaises(RuntimeError):
                subject._verify_installed(Path("synthetic-python"), expected, self.repository)

    def test_result_requires_all_six_models_and_explicit_pending_seventh(self):
        subject.validate_result(synthetic_result(), "google_colab_cpu")
        mutations = [("pretrained_models_loaded", 5), ("registered_image_models", 6),
                     ("training_executed", True), ("participant_data_opened", True),
                     ("colab_runtime_executed", False), ("synthetic_inputs_only", False)]
        for key, value in mutations:
            result = synthetic_result()
            result[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                subject.validate_result(result, "google_colab_cpu")

    def test_result_rejects_replacing_gated_or_failed_model_with_another(self):
        result = synthetic_result()
        result["model_records"]["unexpected"] = result["model_records"].pop("image_dinov3_vits16")
        with self.assertRaisesRegex(RuntimeError, "six_loaded_seven"):
            subject.validate_result(result, "google_colab_cpu")
        result = synthetic_result()
        result["model_records"]["image_skelex"]["checkpoint_loaded"] = False
        with self.assertRaises(RuntimeError):
            subject.validate_result(result, "google_colab_cpu")

    def test_result_rejects_changed_weights_or_created_gradients(self):
        for key in ("repeated_features_exact", "parameters_and_buffers_unchanged", "input_and_rng_unchanged", "gradients_created"):
            result = synthetic_result()
            result["model_records"]["image_skelex"][key] = key == "gradients_created"
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                subject.validate_result(result, "google_colab_cpu")

    def test_verify_delegates_only_to_checkout_core_with_synthetic_scope(self):
        import knee
        output = self.root / "result"
        weights = self.root / "weights"
        fake = types.ModuleType("knee.final_models_preflight")
        fake.__file__ = str(PROJECT / "src/knee/final_models_preflight.py")
        result = synthetic_result("local_cpu")
        def core_run(repository, cache, target, **kwargs):
            self.assertEqual(Path(repository), PROJECT)
            self.assertEqual(cache, weights)
            self.assertEqual(kwargs, {"allow_download": False, "execution_environment": "local_cpu"})
            target.mkdir()
            self.write_json(target / "checkpoint_verification.json", result)
            return result
        fake.run = Mock(side_effect=core_run)
        with patch.dict(sys.modules, {"knee.final_models_preflight": fake}), \
                patch.object(knee, "final_models_preflight", fake, create=True), \
                patch.object(subject, "_cpu_host"):
            self.assertEqual(subject.verify(PROJECT, weights, output, execution_environment="local_cpu"), result)
        fake.run.assert_called_once()

    def test_verify_rejects_overlapping_paths_and_drive_weight_cache(self):
        with patch.object(subject, "_cpu_host"):
            for weights, output in ((PROJECT / "weights", self.root / "result"),
                                    (self.root / "weights", self.root / "weights/result"),
                                    (self.root / "drive/cache", self.root / "result")):
                with self.subTest(weights=weights), self.assertRaises(RuntimeError):
                    subject.verify(PROJECT, weights, output)

    def test_verify_will_not_overwrite_previous_result(self):
        output = self.root / "result"
        output.mkdir()
        with patch.object(subject, "_cpu_host"), self.assertRaisesRegex(RuntimeError, "new_verification_output"):
            subject.verify(PROJECT, self.root / "weights", output)

    def publication(self):
        source, persistent = self.root / "local", self.root / "private"
        source.mkdir()
        persistent.mkdir()
        self.write_json(source / "checkpoint_verification.json", synthetic_result())
        (source / "reserved_test_package.zip").write_bytes(b"must never copy or open")
        tests = {"passed": True, "tests_run": 17, "skipped": 0, "scope": list(subject.TEST_MODULES)}
        runtime = {"version": subject.VERSION, "cpu_only": True, "training_executed": False,
            "source_sha256": {"requirements-final-models.txt": "b" * 64}}
        return source, persistent, tests, runtime

    def test_publication_only_copies_two_verified_records_and_never_overwrites(self):
        source, persistent, tests, runtime = self.publication()
        first = subject.publish_result(source, persistent, "a" * 40, tests, runtime)
        original_files = {str(p): p.read_bytes() for p in persistent.rglob("*.json")}
        second = subject.publish_result(source, persistent, "a" * 40, tests, runtime)
        self.assertEqual(len(list(persistent.iterdir())), 2)
        self.assertEqual(len(list(persistent.rglob("*.json"))), 4)
        self.assertFalse(list(persistent.rglob("*.zip")))
        self.assertTrue(all(Path(path).read_bytes() == raw for path, raw in original_files.items()))
        for result in (first, second):
            self.assertTrue(result["result_files_saved_and_readback_verified"])
            self.assertFalse(result["training_executed"] or result["training_authorized"]
                or result["training_smoke_test_passed"] or result["gpu_resource_profile_recorded"])

    def test_publication_rejects_invalid_tests_or_fake_runtime_before_writing(self):
        source, persistent, tests, runtime = self.publication()
        for changed_tests, changed_runtime in (({**tests, "skipped": 1}, runtime),
                ({**tests, "scope": []}, runtime), (tests, {**runtime, "training_executed": True})):
            with self.subTest(tests=changed_tests), self.assertRaises(RuntimeError):
                subject.publish_result(source, persistent, "a" * 40, changed_tests, changed_runtime)
        self.assertFalse(list(persistent.iterdir()))

    def test_publication_rejects_local_rehearsal_as_real_colab(self):
        source, persistent, tests, runtime = self.publication()
        self.write_json(source / "checkpoint_verification.json", synthetic_result("local_cpu"))
        with self.assertRaises(RuntimeError):
            subject.publish_result(source, persistent, "a" * 40, tests, runtime)

    def test_cli_setup_calls_orchestrator_and_emits_one_json(self):
        output = io.StringIO()
        args = ["check", "--operation", "setup", "--repository-root", str(PROJECT), "--destination", str(self.root / "cpu")]
        with patch.object(sys, "argv", args), patch.object(subject, "setup_runtime", return_value={"cpu_only": True}) as setup, redirect_stdout(output):
            cli.main()
        self.assertEqual(json.loads(output.getvalue()), {"cpu_only": True})
        setup.assert_called_once()

    def test_cli_has_no_dataset_training_or_reserved_test_option(self):
        for option in ("--dataset-root", "--train", "--reserved-test", "--package-dir"):
            with patch.object(sys, "argv", ["check", "--operation", "tests", "--repository-root", str(PROJECT), option, "x"]), \
                    patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                cli.main()

    def test_builder_only_accepts_exact_revision_unless_explicit_unpublished_draft(self):
        for value in ("main", "a" * 39, "0" * 40, builder.UNPUBLISHED):
            with self.subTest(value=value), self.assertRaises(ValueError):
                builder.build(value)
        draft = builder.build(builder.UNPUBLISHED, allow_unpublished_placeholder=True)
        self.assertIn(builder.UNPUBLISHED, draft["cells"][1]["source"])

    def test_notebook_has_no_outputs_private_paths_dataset_or_training_calls(self):
        notebook = builder.build("a" * 40)
        self.assertEqual(notebook["nbformat"], 4)
        full = "\n".join(cell["source"] for cell in notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        for forbidden in ("D:\\", "C:\\", "@gmail.com", "reserved_test_package.zip", ".backward(", "torch.optim", ".fit(", "trust_remote_code=True"):
            self.assertNotIn(forbidden, full)
        for required in ("--operation', 'setup'", "--operation', 'tests'", "--operation', 'verify'", "publish_result", "--allow-download"):
            self.assertIn(required, full)

    def test_notebook_bootstrap_has_revision_bounds_and_no_branch_checkout(self):
        code = builder.build("a" * 40)["cells"][1]["source"]
        self.assertIn("https://codeload.github.com/AlonsoYess/Knee/zip/", code)
        self.assertIn("128 * 1024**2", code)
        self.assertIn("512 * 1024**2", code)
        self.assertIn("stat.S_ISLNK", code)
        self.assertIn("destination.open('xb')", code)
        self.assertNotIn("refs/heads", code)
        self.assertNotIn("extractall", code)

    def test_builder_emits_identical_document_for_same_pinned_revision(self):
        self.assertEqual(builder.build("b" * 40), builder.build("b" * 40))

    def test_test_runner_counts_actual_suite_and_keeps_stdout_clean(self):
        suite = Mock()
        suite.countTestCases.return_value = 3
        loader = Mock(errors=[])
        loader.loadTestsFromNames.return_value = suite
        result = Mock(testsRun=3, skipped=[], expectedFailures=[], unexpectedSuccesses=[])
        result.wasSuccessful.return_value = True
        def run(value):
            print("synthetic helper progress")
            return result
        output, errors = io.StringIO(), io.StringIO()
        with patch("unittest.TestLoader", return_value=loader), \
                patch("unittest.TextTestRunner", return_value=Mock(run=Mock(side_effect=run))), \
                patch.object(sys, "stderr", errors), redirect_stdout(output):
            summary = subject.run_tests(PROJECT)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("synthetic helper progress", errors.getvalue())
        self.assertEqual(summary["tests_run"], 3)
        self.assertEqual(summary["scope"], list(subject.TEST_MODULES))

    def test_test_runner_rejects_errors_empty_suites_or_skips(self):
        suite = Mock()
        suite.countTestCases.return_value = 1
        loader = Mock(errors=[])
        loader.loadTestsFromNames.return_value = suite
        result = Mock(testsRun=1, skipped=[("synthetic", "forbidden skip")], expectedFailures=[], unexpectedSuccesses=[])
        result.wasSuccessful.return_value = True
        with patch("unittest.TestLoader", return_value=loader), \
                patch("unittest.TextTestRunner", return_value=Mock(run=Mock(return_value=result))), \
                self.assertRaisesRegex(RuntimeError, "tests_not_complete"):
            subject.run_tests(PROJECT)
        loader.errors = ["import failure"]
        with patch("unittest.TestLoader", return_value=loader), self.assertRaisesRegex(RuntimeError, "inventory_required"):
            subject.run_tests(PROJECT)
        loader.errors = []
        suite.countTestCases.return_value = 0
        with patch("unittest.TestLoader", return_value=loader), self.assertRaisesRegex(RuntimeError, "inventory_required"):
            subject.run_tests(PROJECT)

    def test_checked_in_notebook_matches_builder_and_has_no_encoding_replacement(self):
        path = PROJECT / builder.NOTEBOOK
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = notebook["cells"][1]["source"]
        revision = next(node.value.value for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CODE_REVISION" for t in node.targets))
        self.assertEqual(notebook, builder.build(revision, allow_unpublished_placeholder=revision == builder.UNPUBLISHED))
        self.assertNotIn("\ufffd", json.dumps(notebook, ensure_ascii=False))

    def test_modules_do_not_eagerly_import_neural_libraries(self):
        tree = ast.parse((PROJECT / "src/knee/final_models_colab.py").read_text(encoding="utf-8"))
        modules = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module)
        self.assertFalse(set(modules) & {"torch", "torchvision", "timm", "transformers", "numpy"})


if __name__ == "__main__":
    unittest.main()
