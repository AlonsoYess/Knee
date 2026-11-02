"""Synthetic CPU orchestration tests for notebook34; no fitting or patient data."""
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

from knee import training_preparation_colab as subject


PROJECT = Path(__file__).resolve().parents[1]


def script(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = script("synthetic_notebook34_builder", "build_training_preparation_notebook.py")
cli = script("synthetic_notebook34_cli", "check_training_preparation_colab.py")


def result(environment="google_colab_cpu"):
    return {"version": subject.CORE_VERSION, "status": subject.CORE_STATUS,
        "execution_environment": environment, "colab_runtime_executed": environment == "google_colab_cpu",
        **{name: False for name in subject.FALSE_FLAGS}, "training_runner_implemented": True,
        "synthetic_clinical_eval_forward_verified": True, "image_normalization_only_verified": True,
        "shortlist_search_closed": True, "pending_authorized_access": ["image_dinov3_vits16"],
        "actual_training_loop_executed": False, "training_smoke_test_passed": False,
        "gpu_resource_profile_recorded": False}


class TrainingPreparationColabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="notebook34_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repository = self.root / "repository"
        self.repository.mkdir()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def fake_core(self):
        core = types.ModuleType("knee.training_preparation")
        core.__file__ = str(PROJECT / "src/knee/training_preparation.py")
        core.validate_preflight = Mock(side_effect=lambda path: subject._json(Path(path) / "colab_final_models_check.json"))
        return core

    def patch_core(self, core):
        import knee
        first = patch.dict(sys.modules, {"knee.training_preparation": core})
        second = patch.object(knee, "training_preparation", core, create=True)
        first.start()
        second.start()
        self.addCleanup(first.stop)
        self.addCleanup(second.stop)

    def source(self, suffix, timestamp):
        path = self.root / ("final_model_check_" + suffix)
        self.write(path / "colab_final_models_check.json", {"executed_at_utc": timestamp})
        self.write(path / "checkpoint_verification.json", {"synthetic": True})
        return path

    def test_latest_previous_proof_is_validated_and_only_direct_records_are_opened(self):
        core = self.fake_core()
        self.patch_core(core)
        self.source("old", "2026-10-21T00:00:00+00:00")
        latest = self.source("latest", "2026-10-22T15:48:56+00:00")
        self.write(self.root / "images" / "final_model_check_nested" / "colab_final_models_check.json", {})
        (self.root / "unrelated_file.zip").write_bytes(b"must not open")
        self.assertEqual(subject.find_previous_preflight(self.root), latest)
        self.assertEqual(core.validate_preflight.call_count, 2)
        self.assertTrue(all(call.args[0].parent == self.root for call in core.validate_preflight.call_args_list))

    def completed_actual_schema(self):
        from knee import training_preparation as core
        from knee.final_model_release import PINS
        from test_final_models_colab import synthetic_result
        directory = self.root / "final_model_check_actual_schema"
        checkpoint = synthetic_result()
        checkpoint["source_files_sha256"] = dict(core.SOURCE_33)
        checkpoint["release_component_canonical_sha256"] = PINS
        self.write(directory / "checkpoint_verification.json", checkpoint)
        summary = {"version": checkpoint["version"], "status": checkpoint["status"],
            "code_revision": "b687d64e18d8e34b106c34e4cc28abd683afc000",
            "colab_runtime_executed": True, "local_rehearsal": False, "cpu_only_guard_passed": True,
            "registered_image_models": 7, "pretrained_models_loaded": 6,
            "pending_authorized_access": ["image_dinov3_vits16"],
            "source_files_sha256": dict(core.SOURCE_33),
            "checkpoint_record_sha256": subject.sha_file(directory / "checkpoint_verification.json"),
            "runtime_requirements_sha256": core.SOURCE_33["requirements-final-models.txt"],
            "result_files_saved_and_readback_verified": True, "shortlist_search_closed": True,
            "cpu_tests": {"passed": True, "tests_run": 195, "skipped": 0,
                "scope": list(subject.previous.TEST_MODULES)},
            "executed_at_utc": "2026-10-22T15:48:56.943350+00:00"}
        for flag in (*subject.previous.FALSE_FLAGS, "training_authorized", "training_smoke_test_passed", "gpu_resource_profile_recorded"):
            summary[flag] = False
        self.write(directory / "colab_final_models_check.json", summary)
        return directory

    def test_discovery_integrates_actual_completed_preflight_validator(self):
        directory = self.completed_actual_schema()
        self.assertEqual(subject.find_previous_preflight(self.root), directory)
        summary_path = directory / "colab_final_models_check.json"
        altered = subject._json(summary_path)
        altered["cpu_tests"]["tests_run"] = 194
        self.write(summary_path, altered)
        with self.assertRaisesRegex(ValueError, "completed_colab33_test_scope"):
            subject.find_previous_preflight(self.root)

    def test_prepare_integrates_actual_core_synthetic_forward_and_publication(self):
        from knee import training_preparation as core
        directory = self.completed_actual_schema()
        output = self.root / "actual_preparation"
        persistent = self.root / "private"
        persistent.mkdir()
        original_version = core.importlib.metadata.version
        with patch.object(subject.previous, "_cpu_host"), patch.object(core, "verify_environment", return_value={}), \
                patch.object(core.importlib.metadata, "version", side_effect=lambda name:
                    core.EXTRA_PACKAGES[name] if name in core.EXTRA_PACKAGES else original_version(name)):
            observed = subject.prepare(PROJECT, directory, output, execution_environment="local_cpu")
        self.assertTrue(observed["synthetic_clinical_eval_forward_verified"])
        self.assertFalse(observed["optimizer_created"] or observed["backward_executed"])
        self.assertEqual(subject._json(output / subject.CORE_FILENAME), observed)
        # Real local rehearsals must remain distinguishable from a future Colab run.
        tests = {"passed": True, "tests_run": 200, "skipped": 0, "scope": list(subject.TEST_MODULES)}
        runtime = {"version": subject.VERSION, "cpu_only": True, "training_executed": False}
        with self.assertRaises(RuntimeError):
            subject.publish_result(output, persistent, "a" * 40, tests, runtime)
        self.assertFalse(list(persistent.iterdir()))

    def test_discovery_rejects_ambiguous_latest_timestamps(self):
        self.patch_core(self.fake_core())
        for suffix in ("one", "two"):
            self.source(suffix, "2026-10-22T00:00:00Z")
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            subject.find_previous_preflight(self.root)

    def test_discovery_rejects_missing_proof_and_ignores_incomplete_empty_folder(self):
        self.patch_core(self.fake_core())
        (self.root / "final_model_check_interrupted").mkdir()
        with self.assertRaisesRegex(RuntimeError, "not_found"):
            subject.find_previous_preflight(self.root)

    def test_discovery_does_not_silently_ignore_invalid_completed_proof(self):
        core = self.fake_core()
        core.validate_preflight.side_effect = RuntimeError("invalid sealed proof")
        self.patch_core(core)
        self.source("bad", "2026-10-22T00:00:00+00:00")
        with self.assertRaisesRegex(RuntimeError, "invalid sealed proof"):
            subject.find_previous_preflight(self.root)

    def test_discovery_rejects_timestamps_without_utc_timezone(self):
        for timestamp in ("invalid", "2026-10-22T00:00:00", "2026-10-22T00:00:00+01:00", None):
            with self.subTest(timestamp=timestamp), self.assertRaises(RuntimeError):
                subject._utc_timestamp(timestamp)

    def test_discovery_limits_direct_child_and_proof_count(self):
        self.patch_core(self.fake_core())
        self.source("one", "2026-10-22T00:00:00+00:00")
        (self.root / "unrelated").mkdir()
        with patch.object(subject, "MAX_DIRECT_CHILDREN", 1), self.assertRaisesRegex(RuntimeError, "direct_child_limit"):
            subject.find_previous_preflight(self.root)
        with patch.object(subject, "MAX_PREVIOUS_RECORDS", 0), self.assertRaisesRegex(RuntimeError, "record_limit"):
            subject.find_previous_preflight(self.root)

    def requirements(self, contents="scikit-learn==1.6.1\nxgboost-cpu==3.0.5\n"):
        path = self.repository / "requirements-training-runner.txt"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_requirement_pins_preserve_base_and_require_cpu_xgboost_distribution(self):
        self.requirements()
        _, pins = subject._requirements(self.repository, {"numpy": "1.26.4"})
        self.assertEqual(pins, {"scikit-learn": "1.6.1", "xgboost-cpu": "3.0.5"})
        for extra in ("numpy==2.0.0", "--extra-index-url https://invalid", "scikit-learn==1.6.1", "joblib>=1"):
            self.requirements("scikit-learn==1.6.1\nxgboost-cpu==3.0.5\n" + extra)
            with self.subTest(extra=extra), self.assertRaises(RuntimeError):
                subject._requirements(self.repository, {"numpy": "1.26.4"})
        self.requirements("scikit-learn==1.6.1\nxgboost==3.0.5\n")
        with self.assertRaisesRegex(RuntimeError, "clinical_dependencies"):
            subject._requirements(self.repository, {})

    def setup_files(self):
        self.requirements()
        path = self.repository / "src/knee/training_preparation_colab.py"
        path.parent.mkdir(parents=True)
        path.write_text("synthetic source", encoding="utf-8")
        base = {"version": subject.previous.VERSION, "source_sha256": {"synthetic": "a" * 64},
            "packages": {"torch": "2.6.0+cpu"}}
        def setup(repository, target):
            python = target / "base/runtime/python/bin/python3.12"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"synthetic interpreter")
            return {**base, "python_path": str(python)}
        return setup

    def test_setup_installs_clinical_pins_only_in_isolated_runtime_then_reverifies(self):
        base_setup = self.setup_files()
        destination = self.root / "runtime"
        with patch.object(subject.previous, "_cpu_host"), \
                patch.object(subject.previous, "setup_runtime", side_effect=base_setup) as base, \
                patch.object(subject, "runtime_environment", return_value={}), \
                patch.object(subject.previous, "_run", return_value="") as run, \
                patch.object(subject.previous, "_verify_installed", return_value={}) as verify:
            first = subject.setup_runtime(self.repository, destination)
            second = subject.setup_runtime(self.repository, destination)
        self.assertEqual(base.call_count, 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(verify.call_count, 2)
        self.assertTrue(Path(run.call_args.args[0][0]).is_relative_to(destination))
        self.assertIn("--no-deps", run.call_args.args[0])
        self.assertFalse(first["training_executed"])
        self.assertTrue(second["existing_runtime_reverified"])

    def test_setup_rejects_incomplete_runtime_and_changed_requirements(self):
        self.setup_files()
        destination = self.root / "runtime"
        destination.mkdir()
        with patch.object(subject.previous, "_cpu_host"), self.assertRaisesRegex(RuntimeError, "incomplete_training_runtime"):
            subject.setup_runtime(self.repository, destination)

    def test_setup_rejects_drive_and_repository_destinations(self):
        self.setup_files()
        (self.root / "drive").mkdir()
        with patch.object(subject.previous, "_cpu_host"):
            for path in (self.root / "drive/runtime", self.repository / "runtime"):
                with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "temporary_runtime"):
                    subject.setup_runtime(self.repository, path)

    def test_validation_requires_explicit_environment_and_all_nontraining_flags(self):
        subject.validate_result(result(), "google_colab_cpu")
        for flag in subject.FALSE_FLAGS:
            changed = result()
            changed[flag] = True
            with self.subTest(flag=flag), self.assertRaises(RuntimeError):
                subject.validate_result(changed, "google_colab_cpu")
        for changed in ({**result(), "colab_runtime_executed": False},
                        {**result(), "status": "training_done"}, {**result(), "version": "wrong"},
                        {**result(), "training_smoke_test_passed": True},
                        {**result(), "training_runner_implemented": False}):
            with self.assertRaises(RuntimeError):
                subject.validate_result(changed, "google_colab_cpu")

    def test_prepare_calls_checkout_core_once_without_training_arguments(self):
        core = self.fake_core()
        local = self.root / "local"
        source = self.source("valid", "2026-10-22T00:00:00+00:00")
        def run(repository, previous, output, **kwargs):
            self.assertEqual(Path(repository), PROJECT)
            self.assertEqual(Path(previous), source)
            self.assertEqual(kwargs, {"execution_environment": "local_cpu"})
            output.mkdir()
            self.write(output / subject.CORE_FILENAME, result("local_cpu"))
            return result("local_cpu")
        core.run = Mock(side_effect=run)
        self.patch_core(core)
        with patch.object(subject.previous, "_cpu_host"):
            observed = subject.prepare(PROJECT, source, local, execution_environment="local_cpu")
        self.assertFalse(observed["training_executed"])
        core.run.assert_called_once()

    def test_prepare_rejects_existing_output_and_overlapping_paths(self):
        self.patch_core(self.fake_core())
        source = self.source("valid", "2026-10-22T00:00:00+00:00")
        existing = self.root / "existing"
        existing.mkdir()
        with patch.object(subject.previous, "_cpu_host"):
            for output in (existing, source / "output", PROJECT / "output"):
                with self.subTest(output=output), self.assertRaises(RuntimeError):
                    subject.prepare(PROJECT, source, output)

    def publication(self):
        local, persistent = self.root / "local", self.root / "private"
        local.mkdir()
        persistent.mkdir()
        self.write(local / subject.CORE_FILENAME, result())
        (local / "image.npz").write_bytes(b"must never open or copy")
        tests = {"passed": True, "tests_run": 200, "skipped": 0, "scope": list(subject.TEST_MODULES)}
        runtime = {"version": subject.VERSION, "cpu_only": True, "training_executed": False,
            "source_sha256": {"requirements-training-runner.txt": "a" * 64}}
        return local, persistent, tests, runtime

    def test_publication_copies_only_two_records_and_preserves_previous_runs(self):
        local, persistent, tests, runtime = self.publication()
        first = subject.publish_result(local, persistent, "a" * 40, tests, runtime)
        before = {path: path.read_bytes() for path in persistent.rglob("*.json")}
        subject.publish_result(local, persistent, "a" * 40, tests, runtime)
        self.assertEqual(len(list(persistent.iterdir())), 2)
        self.assertEqual(len(list(persistent.rglob("*.json"))), 4)
        self.assertFalse(list(persistent.rglob("*.npz")))
        self.assertTrue(all(path.read_bytes() == raw for path, raw in before.items()))
        self.assertTrue(first["result_files_saved_and_readback_verified"])
        self.assertFalse(first["training_authorized"] or first["training_executed"])

    def test_publication_rejects_skips_local_rehearsal_and_fake_runtime_before_writing(self):
        local, persistent, tests, runtime = self.publication()
        for bad_tests, bad_runtime in (({**tests, "skipped": 1}, runtime),
                ({**tests, "scope": []}, runtime), (tests, {**runtime, "training_executed": True})):
            with self.subTest(tests=bad_tests), self.assertRaises(RuntimeError):
                subject.publish_result(local, persistent, "a" * 40, bad_tests, bad_runtime)
        self.write(local / subject.CORE_FILENAME, result("local_cpu"))
        with self.assertRaises(RuntimeError):
            subject.publish_result(local, persistent, "a" * 40, tests, runtime)
        self.assertFalse(list(persistent.iterdir()))

    def test_cli_has_no_training_dataset_or_reserved_test_option(self):
        for flag in ("--train", "--dataset-root", "--reserved-test", "--weights-root", "--allow-download"):
            with patch.object(sys, "argv", ["check", "--operation", "tests", "--repository-root", str(PROJECT), flag, "x"]), \
                    patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                cli.main()

    def test_cli_locate_and_prepare_are_exact_orchestrator_calls(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["check", "--operation", "locate", "--repository-root", str(PROJECT), "--records-root", str(self.root)]), \
                patch.object(subject, "find_previous_preflight", return_value=self.root / "proof") as locate, redirect_stdout(output):
            cli.main()
        locate.assert_called_once_with(self.root)
        self.assertEqual(json.loads(output.getvalue()), {"source_preflight_dir": str(self.root / "proof")})

    def test_test_runner_counts_suite_and_rejects_skips_or_import_failures(self):
        suite = Mock()
        suite.countTestCases.return_value = 2
        loader = Mock(errors=[])
        loader.loadTestsFromNames.return_value = suite
        run = Mock(testsRun=2, skipped=[], expectedFailures=[], unexpectedSuccesses=[])
        run.wasSuccessful.return_value = True
        with patch("unittest.TestLoader", return_value=loader), patch("unittest.TextTestRunner", return_value=Mock(run=Mock(return_value=run))):
            observed = subject.run_tests(PROJECT)
        self.assertEqual(observed["tests_run"], 2)
        run.skipped = [("synthetic", "not allowed")]
        with patch("unittest.TestLoader", return_value=loader), patch("unittest.TextTestRunner", return_value=Mock(run=Mock(return_value=run))), self.assertRaises(RuntimeError):
            subject.run_tests(PROJECT)
        loader.errors = ["import failed"]
        with patch("unittest.TestLoader", return_value=loader), self.assertRaises(RuntimeError):
            subject.run_tests(PROJECT)

    def test_builder_requires_published_revision_or_explicit_unpublished_draft(self):
        for revision in ("main", "a" * 39, "0" * 40, builder.UNPUBLISHED):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                builder.build(revision)
        draft = builder.build(builder.UNPUBLISHED, allow_unpublished_placeholder=True)
        self.assertIn(builder.UNPUBLISHED, draft["cells"][1]["source"])

    def test_notebook_contains_no_private_paths_training_commands_or_outputs(self):
        notebook = builder.build("a" * 40)
        full = "\n".join(cell["source"] for cell in notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        for forbidden in ("D:\\", "C:\\", "@gmail.com", ".backward(", "torch.optim", ".fit(",
                          "--allow-download", "--operation', 'train'", "transport.json", "reserved_test_package.zip"):
            self.assertNotIn(forbidden, full)
        for required in ("--operation', 'setup'", "--operation', 'tests'", "--operation', 'locate'",
                         "--operation', 'prepare'", "publish_result", "misma carpeta", "CPU"):
            self.assertIn(required, full)

    def test_notebook_bootstrap_preserves_bounded_exact_revision_extraction(self):
        source = builder.build("a" * 40)["cells"][1]["source"]
        for fragment in ("128 * 1024**2", "512 * 1024**2", "stat.S_ISLNK", "destination.open('xb')",
                         "CODE_REVISION = '" + "a" * 40 + "'", "training_preparation_colab as orchestration"):
            self.assertIn(fragment, source)
        self.assertNotIn("extractall", source)
        self.assertNotIn("refs/heads", source)

    def test_checked_in_notebook_matches_builder(self):
        notebook = json.loads((PROJECT / builder.NOTEBOOK).read_text(encoding="utf-8"))
        tree = ast.parse(notebook["cells"][1]["source"])
        revision = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "CODE_REVISION" for target in node.targets))
        self.assertEqual(notebook, builder.build(revision, allow_unpublished_placeholder=revision == builder.UNPUBLISHED))
        self.assertNotIn("\ufffd", json.dumps(notebook, ensure_ascii=False))

    def test_orchestrator_does_not_eagerly_import_neural_or_clinical_packages(self):
        tree = ast.parse((PROJECT / "src/knee/training_preparation_colab.py").read_text(encoding="utf-8"))
        modules = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module)
        self.assertFalse(modules & {"torch", "numpy", "sklearn", "xgboost", "timm", "transformers"})


if __name__ == "__main__":
    unittest.main()
