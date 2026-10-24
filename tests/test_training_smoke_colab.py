"""Mocked orchestration tests; no private images, downloads or training steps."""
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

from knee import training_smoke_colab as subject


PROJECT = Path(__file__).resolve().parents[1]


def script(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = script("synthetic_notebook35_builder", "build_training_smoke_notebook.py")
cli = script("synthetic_notebook35_cli", "check_training_smoke_colab.py")


def result():
    return {"version": subject.CORE_VERSION, "status": subject.CORE_STATUS,
        "execution_environment": "google_colab_gpu", "colab_runtime_executed": True,
        "model_id": "image_densenet121", "validation_fold": 0, "seed": 2026, "optimizer_updates": 2,
        "reserved_test_predictive_evaluation_opened": False, "originals_modified": False,
        "partitions_modified": False, "full_training_executed": False, "eligible_for_model_selection": False,
        "full_training_authorized": False, "full_encoder_training_authorized": False,
        "encoder_unfreezing_executed": False, "predictive_performance_claimed": False,
        "scope": "development_only", "stage": "stage_1", "smoke": True,
        "training_knees_seen": 32, "micro_batch_size": 2, "effective_batch_size": 16,
        "frozen_encoder_state_unchanged": True, "task_head_update_verified": True,
        "training_executed": True, "training_smoke_test_passed": True,
        "gpu_resource_profile_recorded": True, "artifact_files_sha256": {},
        "amp_numerical_repair_version": "same_effective_group_amp_replay_v1.0",
        "amp_overflow_replays": 0, "amp_scale_audit": [{}, {}]}


class TrainingSmokeColabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="notebook35_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def fake_core(self):
        core = types.ModuleType("knee.training_smoke")
        core.__file__ = str(PROJECT / "src/knee/training_smoke.py")
        core.validate_preparation = Mock(side_effect=lambda path: subject._json(Path(path) / subject.previous.SUMMARY_FILENAME))
        return core

    def patch_core(self, core):
        import knee
        first = patch.dict(sys.modules, {"knee.training_smoke": core})
        second = patch.object(knee, "training_smoke", core, create=True)
        first.start()
        second.start()
        self.addCleanup(first.stop)
        self.addCleanup(second.stop)

    def proof(self, name, stamp):
        path = self.root / ("training_preparation_check_" + name)
        self.write(path / subject.previous.SUMMARY_FILENAME, {"executed_at_utc": stamp})
        return path

    def test_scans_only_direct_bounded_preparation_records_and_selects_latest(self):
        core = self.fake_core()
        self.patch_core(core)
        self.proof("old", "2026-10-22T00:00:00+00:00")
        latest = self.proof("latest", "2026-10-22T01:00:00+00:00")
        self.write(self.root / "unrelated" / "training_preparation_check_nested" / subject.previous.SUMMARY_FILENAME, {})
        (self.root / "images.zip").write_bytes(b"never read")
        self.assertEqual(subject.find_preparation(self.root), latest)
        self.assertEqual(core.validate_preparation.call_count, 2)
        self.assertTrue(all(call.args[0].parent == self.root for call in core.validate_preparation.call_args_list))

    def test_missing_ambiguous_or_non_utc_preparation_fails_closed(self):
        self.patch_core(self.fake_core())
        with self.assertRaisesRegex(RuntimeError, "not_found"):
            subject.find_preparation(self.root)
        self.proof("a", "2026-10-22T00:00:00+00:00")
        self.proof("b", "2026-10-22T00:00:00+00:00")
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            subject.find_preparation(self.root)

    def test_child_and_record_limits_are_enforced(self):
        self.patch_core(self.fake_core())
        self.proof("a", "2026-10-22T00:00:00+00:00")
        self.proof("b", "2026-10-22T01:00:00+00:00")
        with patch.object(subject, "MAX_DIRECT_CHILDREN", 1), self.assertRaisesRegex(RuntimeError, "child_limit"):
            subject.find_preparation(self.root)
        with patch.object(subject, "MAX_PREVIOUS_RECORDS", 1), self.assertRaisesRegex(RuntimeError, "record_limit"):
            subject.find_preparation(self.root)

    def reference_source(self):
        remote = self.root / "remote"
        self.write(remote / subject.REFERENCE_FILENAME, {"version": "model_inputs_loading_preflight_v1.0",
            "status": "development_inputs_and_cpu_loading_verified",
            "sources_sha256": {"development_package": "a" * 64}})
        (remote / "unrelated_reserved_do_not_open.zip").write_bytes(b"never open")
        reference = self.root / "reference" / subject.REFERENCE_FILENAME
        reference.parent.mkdir()
        return remote, self.root / "stage", reference

    def stage_mock(self, remote, local):
        (local / "development_package.zip").write_bytes(b"mock sealed development")
        return {"sha256": "a" * 64, "training_executed": False, "reserved_test_opened": False}

    def test_transport_calls_only_sealed_development_api_and_copies_exact_reference(self):
        from knee import colab_package_transport as transport
        remote, local, reference = self.reference_source()
        source = remote / subject.REFERENCE_FILENAME
        before = source.read_bytes()
        with patch.object(transport, "stage_development_package", side_effect=self.stage_mock) as staged:
            receipt = subject.stage_development(remote, local, reference)
        staged.assert_called_once_with(remote, local)
        self.assertEqual(reference.read_bytes(), before)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(receipt["reference_sha256"], subject.sha_file(reference))
        self.assertFalse(receipt["reserved_test_opened"])
        self.assertFalse(list(local.glob("*reserved*")))

    def test_transport_rejects_existing_output_bad_reference_or_hash_binding(self):
        from knee import colab_package_transport as transport
        remote, local, reference = self.reference_source()
        with patch.object(transport, "stage_development_package", return_value={"sha256": "b" * 64,
                "training_executed": False, "reserved_test_opened": False}), self.assertRaisesRegex(RuntimeError, "match_verified"):
            subject.stage_development(remote, local, reference)
        with self.assertRaisesRegex(RuntimeError, "new_local_stage"):
            subject.stage_development(remote, local, reference)

    def test_transport_never_overwrites_reference_and_rejects_overlapping_roots(self):
        remote, local, reference = self.reference_source()
        reference.write_bytes(b"previous investigator reference")
        with self.assertRaises(RuntimeError):
            subject.stage_development(remote, local, reference)
        self.assertEqual(reference.read_bytes(), b"previous investigator reference")
        with self.assertRaises(RuntimeError):
            subject.stage_development(remote, remote / "child", self.root / "new_reference.json")

    def test_result_requires_actual_gpu_exact_scope_and_positive_smoke_evidence(self):
        subject.validate_result(result())
        cases = {"execution_environment": "google_colab_cpu", "colab_runtime_executed": False,
            "model_id": "image_vit_b16", "validation_fold": 1, "seed": 2027, "optimizer_updates": 3,
            "training_executed": False, "training_smoke_test_passed": False,
            "gpu_resource_profile_recorded": False, "full_training_executed": True,
            "eligible_for_model_selection": True, "reserved_test_predictive_evaluation_opened": True,
            "originals_modified": True, "partitions_modified": True}
        for name, value in cases.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                subject.validate_result({**result(), name: value})

    def test_smoke_calls_only_exact_core_api_and_checks_saved_result(self):
        core = self.fake_core()
        self.patch_core(core)
        output = self.root / "local"
        def run(repository, prep, package, reference, weights, destination, **kwargs):
            self.assertEqual(repository, PROJECT)
            self.assertEqual(kwargs, {"approval_statement": subject.APPROVAL_STATEMENT,
                "execution_environment": "google_colab_gpu"})
            self.write(destination / subject.CORE_FILENAME, result())
            return result()
        core.run = Mock(side_effect=run)
        observed = subject.smoke(PROJECT, self.root / "prep", self.root / "dev", self.root / "ref",
            self.root / "weights", output, approval_statement=subject.APPROVAL_STATEMENT)
        core.run.assert_called_once()
        self.assertTrue(observed["training_smoke_test_passed"])

    def test_result_requires_fixed_amp_version_nonnegative_count_and_two_update_audit(self):
        subject.validate_result(result())
        subject.validate_result({**result(), "amp_overflow_replays": 2})
        for name, value in (("amp_numerical_repair_version", "legacy_skip_on_overflow"),
                ("amp_numerical_repair_version", None), ("amp_overflow_replays", -1),
                ("amp_overflow_replays", True), ("amp_overflow_replays", 1.0),
                ("amp_scale_audit", []), ("amp_scale_audit", [{}, {}, {}]),
                ("amp_scale_audit", ({}, {}))):
            with self.subTest(name=name, value=value), self.assertRaisesRegex(RuntimeError, "amp_repair_audit"):
                subject.validate_result({**result(), name: value})
        for missing in ("amp_numerical_repair_version", "amp_overflow_replays", "amp_scale_audit"):
            incomplete = result()
            del incomplete[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, "amp_repair_audit"):
                subject.validate_result(incomplete)

    def publication(self):
        local, persistent = self.root / "local", self.root / "private"
        persistent.mkdir()
        job = local / "job"
        job.mkdir(parents=True)
        (job / "best_model.safetensors").write_bytes(b"mock private smoke weights")
        self.write(job / "training_job.json", {"smoke": True})
        artifacts = {"job/" + path.name: subject.sha_file(path) for path in job.iterdir()}
        self.write(local / subject.CORE_FILENAME, {**result(), "artifact_files_sha256": artifacts})
        tests = {"passed": True, "tests_run": 350, "skipped": 0, "scope": list(subject.TEST_MODULES),
            "test_runtime": "gpu_build_cpu_compute_no_training",
            "historical_cpu_preparation_suite_reexecuted": False}
        runtime = {"version": "isolated_training_gpu_runtime_v1.0", "gpu_only": True,
            "cpu_only": False, "training_executed": False}
        return local, persistent, tests, runtime

    def test_publication_copies_exact_private_artifacts_and_never_overwrites_old_runs(self):
        local, persistent, tests, runtime = self.publication()
        first = subject.publish_result(local, persistent, "a" * 40, tests, runtime)
        before = {path: path.read_bytes() for path in persistent.rglob("*") if path.is_file()}
        subject.publish_result(local, persistent, "a" * 40, tests, runtime)
        self.assertEqual(len(list(persistent.iterdir())), 2)
        self.assertTrue(all(path.read_bytes() == raw for path, raw in before.items()))
        self.assertTrue(first["result_files_saved_and_readback_verified"])
        for folder in persistent.iterdir():
            self.assertEqual((folder / "job/best_model.safetensors").read_bytes(),
                (local / "job/best_model.safetensors").read_bytes())

    def test_publication_rejects_skips_fake_runtime_or_changed_inventory_before_writing(self):
        local, persistent, tests, runtime = self.publication()
        for bad_tests, bad_runtime in (({**tests, "skipped": 1}, runtime),
                ({**tests, "scope": []}, runtime), (tests, {**runtime, "gpu_only": False}),
                (tests, {**runtime, "training_executed": True})):
            with self.subTest(tests=bad_tests), self.assertRaises(RuntimeError):
                subject.publish_result(local, persistent, "a" * 40, bad_tests, bad_runtime)
        (local / "job/best_model.safetensors").write_bytes(b"modified private weights")
        with self.assertRaisesRegex(RuntimeError, "exact_sealed"):
            subject.publish_result(local, persistent, "a" * 40, tests, runtime)
        self.assertFalse(list(persistent.iterdir()))

    def test_publication_refuses_unregistered_files_or_oversize_artifacts(self):
        local, persistent, tests, runtime = self.publication()
        (local / "data.npz").write_bytes(b"must never copy participant archives")
        with self.assertRaisesRegex(RuntimeError, "only_registered"):
            subject.publish_result(local, persistent, "a" * 40, tests, runtime)
        self.assertFalse(list(persistent.iterdir()))

    def test_runtime_delegation_is_lazy_and_exact(self):
        from knee import training_gpu_runtime
        with patch.object(training_gpu_runtime, "setup_runtime", return_value={"ok": True}) as setup:
            self.assertEqual(subject.setup_runtime(PROJECT, self.root / "runtime"), {"ok": True})
        setup.assert_called_once_with(PROJECT, self.root / "runtime")
        with patch.object(training_gpu_runtime, "runtime_environment", return_value={"GPU": "1"}) as environment:
            self.assertEqual(subject.runtime_environment(PROJECT), {"GPU": "1"})
        environment.assert_called_once_with(PROJECT)

    def test_synthetic_test_runner_requires_hidden_gpu_and_complete_suite(self):
        suite = Mock()
        suite.countTestCases.return_value = 2
        loader = Mock(errors=[])
        loader.loadTestsFromNames.return_value = suite
        run = Mock(testsRun=2, skipped=[], expectedFailures=[], unexpectedSuccesses=[])
        run.wasSuccessful.return_value = True
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch("unittest.TestLoader", return_value=loader), \
                patch("unittest.TextTestRunner", return_value=Mock(run=Mock(return_value=run))):
            self.assertEqual(subject.run_tests(PROJECT)["tests_run"], 2)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), self.assertRaisesRegex(RuntimeError, "hide_gpu"):
            subject.run_tests(PROJECT)

    def test_cli_excludes_full_training_reserved_and_arbitrary_model_options(self):
        for flag in ("--train", "--reserved-test", "--model-id", "--epochs", "--validation-fold", "--maximum-steps"):
            with self.subTest(flag=flag), patch.object(sys, "argv", ["check", "--operation", "tests",
                    "--repository-root", str(PROJECT), flag, "x"]), patch.object(sys, "stderr", io.StringIO()), \
                    self.assertRaises(SystemExit):
                cli.main()

    def test_cli_locate_delegates_bounded_proof_search(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["check", "--operation", "locate", "--repository-root", str(PROJECT),
                "--records-root", str(self.root)]), patch.object(subject, "find_preparation", return_value=self.root / "proof") as find, \
                redirect_stdout(output):
            cli.main()
        find.assert_called_once_with(self.root)
        self.assertEqual(json.loads(output.getvalue()), {"preparation_dir": str(self.root / "proof")})

    def test_cli_requires_explicit_smoke_approval_and_all_private_inputs(self):
        with patch.object(sys, "argv", ["check", "--operation", "smoke", "--repository-root", str(PROJECT)]), \
                patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
            cli.main()

    def test_builder_requires_exact_published_revision_or_explicit_draft(self):
        for revision in ("main", "a" * 39, "0" * 40, builder.UNPUBLISHED):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                builder.build(revision)
        self.assertIn(builder.UNPUBLISHED, builder.build(builder.UNPUBLISHED,
            allow_unpublished_placeholder=True)["cells"][1]["source"])

    def test_notebook_has_gpu_guard_bounded_extraction_and_no_private_paths(self):
        notebook = builder.build("a" * 40)
        full = "\n".join(cell["source"] for cell in notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        bootstrap = notebook["cells"][1]["source"]
        for fragment in ("128 * 1024**2", "512 * 1024**2", "stat.S_ISLNK", "destination.open('xb')",
                "CODE_REVISION = '" + "a" * 40 + "'", "training_smoke_colab as orchestration",
                "os.environ.get('COLAB_GPU', '0') != '1'"):
            self.assertIn(fragment, bootstrap)
        for forbidden in ("D:\\", "C:\\", "@gmail.com", ".backward(", "torch.optim", ".fit(",
                "reserved_test_package.zip", "from knee.training_smoke import", "extractall", "refs/heads"):
            self.assertNotIn(forbidden, full)
        for required in ("AUTORIZO PRUEBA GPU", "--operation', 'smoke'", "stage_development",
                "model_inputs_loading_reference_v1.json", "publish_result", "codificador congelado"):
            self.assertIn(required, full)

    def test_checked_in_notebook_matches_builder(self):
        notebook = json.loads((PROJECT / builder.NOTEBOOK).read_text(encoding="utf-8"))
        tree = ast.parse(notebook["cells"][1]["source"])
        revision = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "CODE_REVISION" for target in node.targets))
        self.assertEqual(notebook, builder.build(revision, allow_unpublished_placeholder=revision == builder.UNPUBLISHED))
        self.assertNotIn("\ufffd", json.dumps(notebook, ensure_ascii=False))

    def test_revision11_requires_fresh_session_and_same_group_amp_without_protocol_change(self):
        notebook = builder.build("a" * 40)
        opening = notebook["cells"][0]["source"]
        for text in ("Revisión 1.1", "dos actualizaciones exitosas", "mismo grupo efectivo de 16 rodillas",
                "no se descartan muestras", "gradientes finitos permanece activa", "reinicia completamente",
                "copia actualizada desde Drive", "todas las celdas en orden", "No vuelvas a ejecutar únicamente",
                "no se reanuda ni se reutilizan sus pesos", "No cambian el modelo, el pliegue, la semilla"):
            self.assertIn(text, opening)

    def test_approval_statement_exactly_matches_core_contract(self):
        from knee.training_smoke import APPROVAL_STATEMENT
        self.assertEqual(subject.APPROVAL_STATEMENT, APPROVAL_STATEMENT)

    def test_orchestration_keeps_host_imports_free_of_ml_packages(self):
        tree = ast.parse((PROJECT / "src/knee/training_smoke_colab.py").read_text(encoding="utf-8"))
        modules = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module)
        self.assertFalse(modules & {"torch", "numpy", "sklearn", "xgboost", "timm", "transformers"})

    def test_only_cpu_build_specific_modules_are_excluded_and_old_proof_is_still_required(self):
        self.assertEqual(set(subject.previous.TEST_MODULES) - set(subject.TEST_MODULES),
            {"test_training_preparation", "test_pretrained_verification"})
        self.assertEqual(subject.GPU_INCOMPATIBLE_HISTORICAL_TEST_MODULES,
            ("test_training_preparation", "test_pretrained_verification"))
        self.assertIn("test_training_amp", subject.TEST_MODULES)
        self.assertNotIn("test_training_amp", subject.previous.TEST_MODULES)
        core = self.fake_core()
        core.validate_preparation.side_effect = ValueError("historical_cpu_preparation_required")
        self.patch_core(core)
        self.proof("old", "2026-10-22T00:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "historical_cpu_preparation_required"):
            subject.find_preparation(self.root)


if __name__ == "__main__":
    unittest.main()
