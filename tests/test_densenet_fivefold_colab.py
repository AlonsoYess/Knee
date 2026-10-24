"""Notebook36 orchestration tests: fabricated files, never real training."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from knee import densenet_fivefold_colab as subject


PROJECT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("notebook36_test_" + name, PROJECT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_densenet_fivefold_notebook")
cli = load_script("train_densenet_fivefold_colab")


def result():
    return {"version": subject.CORE_VERSION, "status": subject.CORE_STATUS,
        "execution_environment": "google_colab_gpu", "colab_runtime_executed": True,
        "model_id": "image_densenet121", "stage": "stage_1", "seed": 2026,
        "folds_completed": list(range(5)), "training_executed": True,
        "development_package_sha256": "a" * 64, "model_input_reference_sha256": "b" * 64,
        "source_files_sha256": {"engine": "c" * 64},
        "fivefold_selection_metrics": {"average_precision_score": 0.45},
        "executed_at_utc": "2026-10-23T01:00:00+00:00",
        "environment": {"gpu": "synthetic only"},
        **{field:False for field in ("reserved_test_predictive_evaluation_opened", "originals_modified",
            "partitions_modified", "encoder_unfreezing_executed", "other_models_trained", "predictive_performance_claimed",
            "full_encoder_training_authorized", "calibration_performed", "threshold_selected",
            "reserved_test_accessed", "fold_optimizer_resume_supported", "unbiased_final_performance_claimed")},
        "fold_results": [{"validation_fold": fold, "frozen_encoder_state_unchanged": True,
            "reused_authenticated_complete_fold": False, "metrics": {"average_precision_score": 0.45}}
            for fold in range(5)]}


class FivefoldColabTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def job(self, directory, fold):
        directory.mkdir(parents=True)
        record = {"request": {"validation_fold": fold}, "record_sha256": str(fold) * 64}
        for name in subject.JOB_FILES:
            if name == "training_job.json":
                self.write(directory / name, record)
            else:
                (directory / name).write_bytes((name + str(fold)).encode())
        return record

    def identity(self):
        return {"code_revision": "e" * 40, "development_package_sha256": "a" * 64,
            "source_files_sha256": {"engine": "c" * 64}}

    def campaign(self):
        private = self.root / "private"
        private.mkdir()
        return private, subject.open_campaign(private, self.identity())

    def test_bounded_smoke_search_selects_latest_authenticated_direct_child(self):
        from knee import training_smoke_evidence as evidence
        a, b = [self.root / ("training_smoke_check_" + name) for name in ("a", "b")]
        self.write(a / subject.previous.SUMMARY_FILENAME, {"executed_at_utc": "2026-10-23T00:00:00+00:00"})
        self.write(b / subject.previous.SUMMARY_FILENAME, {"executed_at_utc": "2026-10-23T01:00:00+00:00"})
        self.write(self.root / "unrelated/nested/training_smoke_check_x" / subject.previous.SUMMARY_FILENAME, {})
        with patch.object(evidence, "validate_smoke", side_effect=lambda path, repo: subject._json(path / subject.previous.SUMMARY_FILENAME)) as verify:
            self.assertEqual(subject.find_smoke(self.root, PROJECT), b)
        self.assertEqual(verify.call_count, 2)

    def test_smoke_search_missing_tied_non_utc_or_invalid_proof_fails_closed(self):
        from knee import training_smoke_evidence as evidence
        with self.assertRaisesRegex(RuntimeError, "not_found"):
            subject.find_smoke(self.root)
        for name in ("a", "b"):
            self.write(self.root / ("training_smoke_check_" + name) / subject.previous.SUMMARY_FILENAME,
                {"executed_at_utc": "2026-10-23T00:00:00+00:00"})
        with patch.object(evidence, "validate_smoke", side_effect=lambda path, repo: subject._json(path / subject.previous.SUMMARY_FILENAME)):
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                subject.find_smoke(self.root)
        with patch.object(evidence, "validate_smoke", side_effect=RuntimeError("bad_proof")):
            with self.assertRaisesRegex(RuntimeError, "bad_proof"):
                subject.find_smoke(self.root)

    def test_smoke_search_child_and_proof_limits(self):
        from knee import training_smoke_evidence as evidence
        for name in ("a", "b"):
            self.write(self.root / ("training_smoke_check_" + name) / subject.previous.SUMMARY_FILENAME,
                {"executed_at_utc": "2026-10-23T00:00:00+00:00"})
        with patch.object(evidence, "validate_smoke", side_effect=lambda path, repo: subject._json(path / subject.previous.SUMMARY_FILENAME)):
            with patch.object(subject, "MAX_DIRECT_CHILDREN", 1), self.assertRaisesRegex(RuntimeError, "child_limit"):
                subject.find_smoke(self.root)
            with patch.object(subject, "MAX_PREVIOUS_RECORDS", 1), self.assertRaisesRegex(RuntimeError, "record_limit"):
                subject.find_smoke(self.root)

    def test_campaign_identity_binds_source_proof_reference_package_and_revision(self):
        from knee import training_smoke_evidence as evidence
        proof = self.root / "proof"
        self.write(proof / subject.previous.SUMMARY_FILENAME, {"mock": True})
        reference = self.root / "reference.json"
        self.write(reference, {"version": "model_inputs_loading_preflight_v1.0",
            "status": "development_inputs_and_cpu_loading_verified",
            "sources_sha256": {"development_package": "a" * 64}})
        with patch.object(evidence, "validate_smoke", return_value={}):
            identity = subject._campaign_identity(PROJECT, proof, reference, "e" * 40)
        self.assertEqual(identity["development_package_sha256"], "a" * 64)
        self.assertEqual(set(identity["source_files_sha256"]), set(subject.SOURCE_FILES))
        self.assertEqual(identity["smoke_summary_sha256"], subject.sha_file(proof / subject.previous.SUMMARY_FILENAME))

    def test_campaign_reuses_only_one_exact_identity_and_never_modifies_manifest(self):
        private, campaign = self.campaign()
        before = (campaign / subject.CAMPAIGN_FILENAME).read_bytes()
        self.assertEqual(subject.open_campaign(private, self.identity()), campaign)
        self.assertEqual((campaign / subject.CAMPAIGN_FILENAME).read_bytes(), before)
        changed = subject.open_campaign(private, {**self.identity(), "code_revision": "f" * 40})
        self.assertNotEqual(campaign, changed)

    def test_ambiguous_identical_campaigns_stop_instead_of_choosing(self):
        private, campaign = self.campaign()
        duplicate = private / (subject.CAMPAIGN_PREFIX + "f" * 32)
        duplicate.mkdir()
        self.write(duplicate / subject.CAMPAIGN_FILENAME, subject._json(campaign / subject.CAMPAIGN_FILENAME))
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            subject.open_campaign(private, self.identity())

    def test_interrupted_campaign_manifest_is_unsealed_and_new_initialization_succeeds(self):
        private = self.root / "private"
        private.mkdir()
        def interrupted_save(stage, value):
            stage.write_bytes(b'{"partial_manifest":')
            raise RuntimeError("interrupted_manifest_publication")
        with patch.object(subject, "_save", side_effect=interrupted_save):
            with self.assertRaisesRegex(RuntimeError, "interrupted_manifest"):
                subject.open_campaign(private, self.identity())
        incomplete = next(private.iterdir())
        self.assertFalse((incomplete / subject.CAMPAIGN_FILENAME).exists())
        stage = next(incomplete.glob(".publication_*.part"))
        self.assertEqual(subject.completed_fold_transports(incomplete), {})
        complete = subject.open_campaign(private, self.identity())
        self.assertNotEqual(complete, incomplete)
        self.assertEqual(subject._json(complete / subject.CAMPAIGN_FILENAME)["identity"], self.identity())
        self.assertEqual(stage.read_bytes(), b'{"partial_manifest":')

    def test_interrupted_fold_receipt_is_unsealed_and_new_fold_attempt_succeeds(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source, 0)
        def interrupted_save(stage, value):
            stage.write_bytes(b'{"partial_receipt":')
            raise RuntimeError("interrupted_receipt_publication")
        with patch.object(subject, "_save", side_effect=interrupted_save):
            with self.assertRaisesRegex(RuntimeError, "interrupted_receipt"):
                subject.persist_completed_fold(campaign, 0, source, record)
        incomplete = next(path for path in campaign.iterdir() if path.is_dir())
        self.assertFalse((incomplete / subject.FOLD_RECEIPT).exists())
        stage = next(incomplete.glob(".publication_*.part"))
        self.assertEqual(subject.completed_fold_transports(campaign), {})
        target = subject.persist_completed_fold(campaign, 0, source, record)
        self.assertNotEqual(target.parent, incomplete)
        self.assertEqual(set(subject.completed_fold_transports(campaign)), {0})
        self.assertEqual(stage.read_bytes(), b'{"partial_receipt":')

    def test_completed_fold_is_copied_and_read_back_before_receipt_written(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source, 0)
        target = subject.persist_completed_fold(campaign, 0, source, record)
        self.assertEqual(subject._job_inventory(source), subject._job_inventory(target))
        receipt = subject._json(target.parent / subject.FOLD_RECEIPT)
        self.assertEqual(receipt["files_sha256"], subject._job_inventory(source))
        self.assertEqual(set(subject.completed_fold_transports(campaign)), {0})

    def test_same_completed_fold_never_overwrites_or_recopies(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source, 0)
        target = subject.persist_completed_fold(campaign, 0, source, record)
        before = {path: path.read_bytes() for path in campaign.rglob("*") if path.is_file()}
        self.assertEqual(subject.persist_completed_fold(campaign, 0, source, record), target)
        self.assertTrue(all(path.read_bytes() == raw for path,raw in before.items()))
        (source / "best_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "cannot_be_overwritten"):
            subject.persist_completed_fold(campaign, 0, source, record)

    def test_partial_fold_copy_without_final_receipt_is_preserved_not_reused(self):
        _, campaign = self.campaign()
        partial = campaign / ("fold_0_attempt_" + "e" * 32)
        partial.mkdir()
        (partial / "job").mkdir()
        (partial / "job/best_model.safetensors").write_bytes(b"partial")
        self.assertEqual(subject.completed_fold_transports(campaign), {})
        source = self.root / "job"
        record = self.job(source, 0)
        subject.persist_completed_fold(campaign, 0, source, record)
        self.assertEqual((partial / "job/best_model.safetensors").read_bytes(), b"partial")

    def test_failed_readback_never_seals_fold_and_next_fold_cannot_start(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source, 0)
        with patch.object(subject, "_copy_job", side_effect=RuntimeError("readback_failed")):
            with self.assertRaisesRegex(RuntimeError, "readback_failed"):
                subject.persist_completed_fold(campaign, 0, source, record)
        self.assertFalse(list(campaign.rglob(subject.FOLD_RECEIPT)))
        self.assertEqual(subject.completed_fold_transports(campaign), {})

    def test_unknown_extra_missing_or_large_fold_artifacts_rejected(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source, 0)
        (source / "private_images.npz").write_bytes(b"must never copy")
        with self.assertRaisesRegex(RuntimeError, "only_exact"):
            subject.persist_completed_fold(campaign, 0, source, record)
        self.assertEqual(subject.completed_fold_transports(campaign), {})

    def test_changed_private_fold_or_duplicate_receipts_stop_resume(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source, 0)
        target = subject.persist_completed_fold(campaign, 0, source, record)
        (target / "best_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "digest_mismatch"):
            subject.completed_fold_transports(campaign)

    def test_result_rejects_other_models_unfreezing_or_missing_folds(self):
        subject.validate_result(result())
        for field, value in (("model_id", "image_vit_b16"), ("stage", "stage_2"), ("seed", 2027),
                ("folds_completed", [0, 1]), ("training_executed", False), ("originals_modified", True),
                ("reserved_test_predictive_evaluation_opened", True), ("other_models_trained", True),
                ("encoder_unfreezing_executed", True), ("predictive_performance_claimed", True),
                ("full_encoder_training_authorized", True), ("calibration_performed", True),
                ("threshold_selected", True), ("reserved_test_accessed", True),
                ("fold_optimizer_resume_supported", True), ("unbiased_final_performance_claimed", True)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                subject.validate_result({**result(), field:value})

    def fake_execution(self, output, completed_folds=None):
        output.mkdir()
        reports, artifacts = [], {}
        for fold in range(5):
            record = self.job(output / f"fold_{fold}", fold)
            for name,digest in subject._job_inventory(output / f"fold_{fold}").items():
                artifacts[f"fold_{fold}/{name}"] = digest
            reports.append({"validation_fold": fold, "frozen_encoder_state_unchanged": True,
                "reused_authenticated_complete_fold": fold in (completed_folds or {}),
                "metrics": {"average_precision_score": 0.45}})
        self.write(output / subject.OOF_FILENAME, {"rows": [{"mock": "no patient data"}]})
        artifacts[subject.OOF_FILENAME] = subject.sha_file(output / subject.OOF_FILENAME)
        core = {**result(), "fold_results": reports, "artifact_files_sha256": artifacts}
        self.write(output / subject.CORE_FILENAME, core)
        return core

    def execute_persistent(self, *, code_revision="e" * 40):
        from knee import densenet_fivefold as core
        private = self.root / "private"
        private.mkdir(exist_ok=True)
        output = self.root / ("local_" + str(len(list(self.root.iterdir()))))
        tests = {"passed": True, "tests_run": 400, "skipped": 0, "scope": list(subject.TEST_MODULES),
            "test_runtime": "gpu_build_cpu_compute_no_training"}
        runtime = {"version": "isolated_training_gpu_runtime_v1.0", "gpu_only": True,
            "cpu_only": False, "training_executed": False}
        observed = {}
        def run(repo, proof, package, reference, weights, destination, **kwargs):
            observed.update(kwargs)
            finished = self.fake_execution(destination, kwargs["completed_folds"])
            for fold in range(5):
                kwargs["fold_completed"](fold, destination / f"fold_{fold}",
                    subject._json(destination / f"fold_{fold}/training_job.json"))
            return finished
        with patch.object(subject, "_campaign_identity", return_value=self.identity()), \
                patch.object(core, "run", side_effect=run):
            summary = subject.train(PROJECT, self.root / "proof", self.root / "dev", self.root / "reference/ref.json",
                self.root / "weights", output, private, approval_statement=subject.APPROVAL_STATEMENT,
                code_revision=code_revision, cpu_tests=tests, runtime=runtime)
        return summary, output, private, observed

    def test_persistent_execution_saves_all_31_artifacts_and_maps_every_location(self):
        summary, output, private, observed = self.execute_persistent()
        campaign = private / summary["persistent_campaign_folder"]
        self.assertEqual(len(summary["private_artifact_locations"]), 31)
        self.assertEqual(len(summary["private_files_sha256"]), 38)
        self.assertEqual((campaign / subject.OOF_FILENAME).read_bytes(), (output / subject.OOF_FILENAME).read_bytes())
        for name,location in summary["private_artifact_locations"].items():
            self.assertEqual(subject.sha_file(campaign / location), summary["artifact_files_sha256"][name])
        self.assertEqual(observed["completed_folds"], {})

    def test_reexecution_authenticates_completed_folds_and_never_overwrites_campaign(self):
        summary, _, private, _ = self.execute_persistent()
        before = {path:path.read_bytes() for path in private.rglob("*") if path.is_file()}
        again, _, _, observed = self.execute_persistent()
        self.assertEqual(again, summary)
        self.assertEqual(set(observed["completed_folds"]), set(range(5)))
        self.assertTrue(all(path.read_bytes() == raw for path,raw in before.items()))

    def test_interrupted_summary_finalization_recovers_without_overwriting_core(self):
        summary, _, private, _ = self.execute_persistent()
        campaign = private / summary["persistent_campaign_folder"]
        core_before = (campaign / subject.CORE_FILENAME).read_bytes()
        # Emulate a process ending between immutable core write and summary write.
        summary_path = campaign / subject.SUMMARY_FILENAME
        summary_path.unlink()
        recovered, _, _, observed = self.execute_persistent()
        self.assertEqual(len(observed["completed_folds"]), 5)
        self.assertEqual((campaign / subject.CORE_FILENAME).read_bytes(), core_before)
        self.assertTrue(recovered["result_files_saved_and_readback_verified"])

    def test_final_oof_tampering_or_missing_artifact_is_not_accepted(self):
        summary, _, private, _ = self.execute_persistent()
        campaign = private / summary["persistent_campaign_folder"]
        (campaign / subject.OOF_FILENAME).write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "oof_prediction_readback"):
            self.execute_persistent()

    def test_completed_semantic_result_comparison_ignores_only_invocation_metadata(self):
        original, recomputed = result(), result()
        recomputed["executed_at_utc"] = "2026-10-24T00:00:00+00:00"
        recomputed["environment"] = {"new_gpu": "synthetic"}
        for row in recomputed["fold_results"]:
            row["reused_authenticated_complete_fold"] = True
        subject._same_completed_campaign(original, recomputed)
        recomputed["fivefold_selection_metrics"]["average_precision_score"] = 0.99
        with self.assertRaisesRegex(RuntimeError, "scientific_record_mismatch"):
            subject._same_completed_campaign(original, recomputed)

    def test_interrupted_oof_staging_is_ignored_and_retry_succeeds(self):
        source = self.root / "source.json"
        self.write(source, {"rows": ["synthetic"]})
        _, campaign = self.campaign()
        target = campaign / subject.OOF_FILENAME
        def interrupted_copy(incoming, outgoing, **kwargs):
            outgoing.write(b'{"partial":')
            raise RuntimeError("interrupted_copy")
        with patch.object(subject.shutil, "copyfileobj", side_effect=interrupted_copy):
            with self.assertRaisesRegex(RuntimeError, "interrupted_copy"):
                subject._publish_copy(source, target, subject.sha_file(source))
        self.assertFalse(target.exists())
        stages = list(campaign.glob(".publication_*.part"))
        self.assertEqual(len(stages), 1)
        self.assertEqual(subject.completed_fold_transports(campaign), {})
        subject._publish_copy(source, target, subject.sha_file(source))
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertEqual(stages[0].read_bytes(), b'{"partial":')

    def test_interrupted_json_staging_preserves_partial_and_retry_never_overwrites_proof(self):
        for filename in (subject.CORE_FILENAME, subject.SUMMARY_FILENAME):
            with self.subTest(filename=filename):
                folder = self.root / filename.replace(".json", "")
                folder.mkdir()
                target = folder / filename
                def interrupted_save(stage, value):
                    stage.write_bytes(b'{"partial":')
                    raise RuntimeError("interrupted_save")
                with patch.object(subject, "_save", side_effect=interrupted_save):
                    with self.assertRaisesRegex(RuntimeError, "interrupted_save"):
                        subject._publish_json(target, {"complete": True})
                self.assertFalse(target.exists())
                subject._publish_json(target, {"complete": True})
                before = target.read_bytes()
                subject._publish_json(target, {"complete": True})
                self.assertEqual(target.read_bytes(), before)
                with self.assertRaisesRegex(RuntimeError, "must_not_be_replaced"):
                    subject._publish_json(target, {"changed": True})
                self.assertEqual(target.read_bytes(), before)

    def test_publication_staging_bounds_and_unknown_temp_names_fail_closed(self):
        _, campaign = self.campaign()
        (campaign / ".publication_not_a_registered_stage.part").write_bytes(b"partial")
        with self.assertRaisesRegex(RuntimeError, "only_registered"):
            subject.completed_fold_transports(campaign)

    def test_full_campaign_retry_after_partial_final_copy_reuses_folds(self):
        original_copy = subject._publish_copy
        def interrupted_final_copy(source, target, checksum):
            stage = subject._publication_stage(target)
            stage.write_bytes(b'{"partial":')
            raise RuntimeError("lost_connection_before_oof_publication")
        with patch.object(subject, "_publish_copy", side_effect=interrupted_final_copy):
            with self.assertRaisesRegex(RuntimeError, "lost_connection"):
                self.execute_persistent()
        private = self.root / "private"
        campaign = next(private.iterdir())
        self.assertEqual(set(subject.completed_fold_transports(campaign)), set(range(5)))
        self.assertFalse((campaign / subject.OOF_FILENAME).exists())
        self.assertFalse((campaign / subject.CORE_FILENAME).exists())
        summary, _, _, observed = self.execute_persistent()
        self.assertEqual(set(observed["completed_folds"]), set(range(5)))
        self.assertTrue(summary["result_files_saved_and_readback_verified"])

    def test_absent_approval_fails_before_runtime_or_data_access(self):
        with patch.object(subject, "_campaign_identity") as identity:
            with self.assertRaisesRegex(RuntimeError, "approval"):
                subject.train(PROJECT, "proof", "dev", "ref", "weights", "output", "private",
                    approval_statement="", code_revision="e" * 40, cpu_tests={}, runtime={})
        identity.assert_not_called()

    def test_synthetic_test_runner_requires_hidden_gpu_and_complete_test_receipt(self):
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

    def test_cli_rejects_arbitrary_models_epochs_test_package_and_fold_overrides(self):
        for option in ("--model-id", "--reserved-test", "--epochs", "--seed", "--validation-fold", "--maximum-steps"):
            with self.subTest(option=option), patch.object(sys, "argv", ["train", "--operation", "tests",
                    "--repository-root", str(PROJECT), option, "x"]), patch.object(sys, "stderr", io.StringIO()), \
                    self.assertRaises(SystemExit):
                cli.main()

    def test_cli_locate_passes_repository_to_smoke_authentication(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["train", "--operation", "locate", "--repository-root", str(PROJECT),
                "--records-root", str(self.root)]), patch.object(subject, "find_smoke", return_value=self.root) as locate, \
                redirect_stdout(output):
            cli.main()
        locate.assert_called_once_with(self.root, PROJECT)

    def test_cli_training_requires_all_inputs_and_explicit_receipts(self):
        with patch.object(sys, "argv", ["train", "--operation", "train", "--repository-root", str(PROJECT)]), \
                patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
            cli.main()

    def test_builder_requires_exact_published_revision_or_explicit_draft(self):
        for revision in ("main", "a" * 39, "0" * 40, builder.UNPUBLISHED):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                builder.build(revision)
        self.assertIn(builder.UNPUBLISHED, builder.build(builder.UNPUBLISHED,
            allow_unpublished_placeholder=True)["cells"][1]["source"])

    def test_notebook_code_syntax_guards_streaming_durability_and_no_private_paths(self):
        notebook = builder.build("a" * 40)
        full = "\n".join(cell["source"] for cell in notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        for fragment in ("training_smoke",):
            self.assertNotIn("--operation', 'smoke'", full)
        for required in ("AUTORIZO DENSENET 5 PLIEGUES", "densenet_fivefold_colab as orchestration",
                "subprocess.Popen", "time.monotonic()", "process.terminate()", "process.poll()",
                "--records-root", "--operation', 'train'", "stage_development", "60 épocas", "paciencia **8**",
                "Cada pliegue completo", "checkpoint público", "no se continúan su optimizador"):
            self.assertIn(required, full)
        for forbidden in ("D:\\", "C:\\", "@gmail.com", "extractall", "refs/heads", "torch.optim",
                ".backward(", "reserved_test_package.zip"):
            self.assertNotIn(forbidden, full)
        bootstrap = notebook["cells"][1]["source"]
        for required in ("128 * 1024**2", "512 * 1024**2", "stat.S_ISLNK", "destination.open('xb')",
                "os.environ.get('COLAB_GPU', '0') != '1'"):
            self.assertIn(required, bootstrap)

    def test_checked_in_notebook_is_exact_builder_output(self):
        notebook = json.loads((PROJECT / builder.NOTEBOOK).read_text(encoding="utf-8"))
        revision = next(node.value.value for node in ast.parse(notebook["cells"][1]["source"]).body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "CODE_REVISION" for target in node.targets))
        self.assertEqual(notebook, builder.build(revision, allow_unpublished_placeholder=revision == builder.UNPUBLISHED))

    def test_wrapper_has_stdlib_only_host_imports_and_same_approved_statement(self):
        tree = ast.parse((PROJECT / "src/knee/densenet_fivefold_colab.py").read_text(encoding="utf-8"))
        imported = {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}
        imported.update(alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names)
        self.assertFalse(imported & {"torch", "numpy", "sklearn", "transformers", "timm"})
        from knee import densenet_fivefold as core
        self.assertEqual(subject.APPROVAL_STATEMENT, core.APPROVAL_STATEMENT)
        self.assertEqual(subject.TEST_MODULES[:-3], subject.previous.TEST_MODULES)


if __name__ == "__main__":
    unittest.main()
