"""Notebook37 transport checks with fabricated files and no estimator fitting."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from knee import clinical_fivefold_colab as subject


ERRORS = (ValueError, RuntimeError)
MODELS = ("clinical_logistic", "clinical_xgboost", "clinical_mlp")
PROJECT = Path(__file__).resolve().parents[1]


def completed_result():
    """Minimal complete orchestration record; contains no participant records."""
    return {"version": subject.CORE_VERSION, "status": subject.CORE_STATUS,
        "execution_environment": "google_colab_cpu", "colab_runtime_executed": True,
        "local_rehearsal": False, "scope": "development_only", "model_ids": list(MODELS),
        "stage": "stage_1", "seed": 2026, "folds_completed": list(range(5)), "job_count": 15,
        "clinical_features": ["AGEYEARS", "SEX", "BMI"], "training_executed": True,
        "cpu_only_guard_passed": True, "participant_disjoint_folds_verified": True,
        "clinical_preprocessing_fitted_inside_training_folds_only": True,
        "every_development_knee_predicted_once_per_model": True,
        "eligible_for_model_selection": True,
        "model_results": {model_id: {"fold_results": [{"validation_fold": fold,
            "reused_authenticated_complete_job": False,
            "metrics": {"average_precision_score": .45, "roc_auc": .6, "brier": .2}}
            for fold in range(5)], "fivefold_selection_metrics": {
                "average_precision_score": .45, "roc_auc": .6, "brier": .2},
            "every_development_knee_predicted_once": True} for model_id in MODELS},
        "executed_at_utc": "2026-10-24T17:00:00+00:00",
        **{field: False for field in ("image_models_trained", "multimodal_models_trained", "image_model_inference_executed",
            "pretrained_weights_loaded", "full_encoder_training_authorized", "encoder_unfreezing_executed",
            "calibration_performed", "threshold_selected", "reserved_test_accessed",
            "reserved_test_predictive_evaluation_opened", "originals_modified", "partitions_modified",
            "fold_optimizer_resume_supported", "predictive_performance_claimed",
            "unbiased_final_performance_claimed")}}


class ClinicalFivefoldColabTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def identity(self):
        return {"code_revision": "e" * 40,
            "development_package_sha256": "a" * 64,
            "source_files_sha256": {"engine": "c" * 64},
            "model_ids": list(MODELS), "validation_folds": list(range(5)), "seed": 2026}

    def campaign(self):
        private = self.root / "private"
        private.mkdir(exist_ok=True)
        return private, subject.open_campaign(private, self.identity())

    def job(self, directory, model_id="clinical_logistic", fold=0):
        """Transport fixture only; scientific authentication belongs to core tests."""
        directory.mkdir(parents=True)
        request = {"model_id": model_id, "validation_fold": fold,
            "operation": "development_fold", "stage": "stage_1",
            "image_encoder_id": None, "seed": 2026}
        record = {"request": request,
            "record_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}
        names = {"training_job.json", "clinical_state.json", "best_validation_predictions.json"}
        if model_id == "clinical_mlp":
            names.update({"best_model.safetensors", "last_model.safetensors", "training_record.json"})
        else:
            names.add("clinical_model.json" if model_id == "clinical_logistic" else "clinical_model.ubj")
        for name in names:
            if name == "training_job.json":
                self.write(directory / name, record)
            else:
                (directory / name).write_bytes((model_id + "_" + str(fold) + "_" + name).encode())
        return record

    def test_inventory_has_exact_registered_four_or_six_files(self):
        for model_id in MODELS:
            with self.subTest(model_id=model_id):
                directory = self.root / model_id
                self.job(directory, model_id)
                inventory = subject._job_inventory(directory)
                self.assertEqual(len(inventory), 6 if model_id == "clinical_mlp" else 4)
                self.assertEqual(inventory, subject._job_inventory(directory, model_id))
                self.assertTrue(all(len(value) == 64 for value in inventory.values()))

    def test_wrong_model_missing_extra_and_empty_files_cannot_enter_transport(self):
        for mutation in ("wrong_model", "missing", "extra", "empty"):
            with self.subTest(mutation=mutation):
                directory = self.root / mutation
                self.job(directory)
                if mutation == "missing":
                    (directory / "clinical_state.json").unlink()
                elif mutation == "extra":
                    (directory / "private_images.npz").write_bytes(b"forbidden")
                elif mutation == "empty":
                    (directory / "clinical_model.json").write_bytes(b"")
                with self.assertRaises(ERRORS):
                    subject._job_inventory(directory, "clinical_xgboost" if mutation == "wrong_model" else None)

    def test_copy_verifies_each_artifact_and_preserves_source(self):
        source = self.root / "source"
        self.job(source, "clinical_mlp", 3)
        before = {p.name: p.read_bytes() for p in source.iterdir()}
        inventory = subject._job_inventory(source)
        target = self.root / "target"
        subject._copy_job(source, target, inventory)
        self.assertEqual(subject._job_inventory(target), inventory)
        self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_copy_rejects_existing_destination_and_changed_expected_hash(self):
        source = self.root / "source"
        self.job(source)
        expected = subject._job_inventory(source)
        target = self.root / "target"
        target.mkdir()
        with self.assertRaises(ERRORS):
            subject._copy_job(source, target, expected)
        expected["clinical_model.json"] = "0" * 64
        with self.assertRaises(ERRORS):
            subject._copy_job(source, self.root / "unused", expected)

    def test_same_campaign_reuses_one_immutable_manifest(self):
        private, campaign = self.campaign()
        before = (campaign / subject.CAMPAIGN_FILENAME).read_bytes()
        self.assertEqual(subject.open_campaign(private, self.identity()), campaign)
        self.assertEqual((campaign / subject.CAMPAIGN_FILENAME).read_bytes(), before)
        changed = subject.open_campaign(private, {**self.identity(), "code_revision": "f" * 40})
        self.assertNotEqual(changed, campaign)

    def test_ambiguous_campaigns_are_not_silently_selected(self):
        private, campaign = self.campaign()
        duplicate = private / (subject.CAMPAIGN_PREFIX + "f" * 32)
        duplicate.mkdir()
        self.write(duplicate / subject.CAMPAIGN_FILENAME,
            subject._json(campaign / subject.CAMPAIGN_FILENAME))
        with self.assertRaises(ERRORS):
            subject.open_campaign(private, self.identity())

    def test_interrupted_campaign_publication_never_creates_completed_manifest(self):
        private = self.root / "private"
        private.mkdir()
        def interrupted(stage, value):
            stage.write_bytes(b'{"incomplete":')
            raise RuntimeError("interrupted manifest")
        with patch.object(subject, "_save", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted manifest"):
                subject.open_campaign(private, self.identity())
        partial = next(private.iterdir())
        self.assertFalse((partial / subject.CAMPAIGN_FILENAME).exists())
        self.assertEqual(subject.completed_job_transports(partial), {})
        complete = subject.open_campaign(private, self.identity())
        self.assertNotEqual(complete, partial)
        self.assertTrue(list(partial.glob(".publication_*.part")))

    def test_model_and_fold_are_both_bound_to_completed_receipt(self):
        _, campaign = self.campaign()
        for model_id in MODELS:
            source = self.root / model_id
            record = self.job(source, model_id, 1)
            target = subject.persist_completed_job(campaign, model_id, 1, source, record)
            receipt = subject._json(target.parent / subject.JOB_RECEIPT)
            self.assertEqual(receipt["model_id"], model_id)
            self.assertEqual(receipt["validation_fold"], 1)
            self.assertEqual(receipt["files_sha256"], subject._job_inventory(source))
        self.assertEqual(set(subject.completed_job_transports(campaign)), {(model, 1) for model in MODELS})

    def test_persistence_rejects_model_or_fold_mismatch_before_copying(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source, "clinical_logistic", 0)
        for model_id, fold in (("clinical_xgboost", 0), ("clinical_logistic", 1), ("image_vit_b16", 0)):
            with self.subTest(model=model_id, fold=fold), self.assertRaises(ERRORS):
                subject.persist_completed_job(campaign, model_id, fold, source, record)
        self.assertEqual(subject.completed_job_transports(campaign), {})

    def test_repeated_completed_job_never_overwrites_prior_bytes(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source)
        target = subject.persist_completed_job(campaign, "clinical_logistic", 0, source, record)
        before = {path: path.read_bytes() for path in campaign.rglob("*") if path.is_file()}
        self.assertEqual(subject.persist_completed_job(campaign, "clinical_logistic", 0, source, record), target)
        self.assertTrue(all(path.read_bytes() == raw for path, raw in before.items()))
        (source / "clinical_model.json").write_bytes(b"changed model")
        with self.assertRaises(ERRORS):
            subject.persist_completed_job(campaign, "clinical_logistic", 0, source, record)
        self.assertTrue(all(path.read_bytes() == raw for path, raw in before.items()))

    def test_partial_job_without_receipt_is_retained_and_not_reused(self):
        _, campaign = self.campaign()
        partial = campaign / ("clinical_mlp_fold_2_attempt_" + "e" * 32)
        (partial / "job").mkdir(parents=True)
        (partial / "job/best_model.safetensors").write_bytes(b"partial")
        self.assertEqual(subject.completed_job_transports(campaign), {})
        source = self.root / "job"
        record = self.job(source, "clinical_mlp", 2)
        target = subject.persist_completed_job(campaign, "clinical_mlp", 2, source, record)
        self.assertNotEqual(target.parent, partial)
        self.assertEqual((partial / "job/best_model.safetensors").read_bytes(), b"partial")

    def test_readback_failure_never_publishes_a_completed_receipt(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source)
        with patch.object(subject, "_copy_job", side_effect=RuntimeError("readback failed")):
            with self.assertRaisesRegex(RuntimeError, "readback failed"):
                subject.persist_completed_job(campaign, "clinical_logistic", 0, source, record)
        self.assertFalse(list(campaign.rglob(subject.JOB_RECEIPT)))
        self.assertEqual(subject.completed_job_transports(campaign), {})

    def test_interrupted_receipt_is_not_reused_and_retry_uses_new_attempt(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source)
        def interrupted(stage, value):
            stage.write_bytes(b'{"incomplete receipt":')
            raise RuntimeError("interrupted receipt")
        with patch.object(subject, "_save", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted receipt"):
                subject.persist_completed_job(campaign, "clinical_logistic", 0, source, record)
        partial = next(p for p in campaign.iterdir() if p.is_dir())
        self.assertFalse((partial / subject.JOB_RECEIPT).exists())
        self.assertEqual(subject.completed_job_transports(campaign), {})
        target = subject.persist_completed_job(campaign, "clinical_logistic", 0, source, record)
        self.assertNotEqual(target.parent, partial)
        self.assertEqual(set(subject.completed_job_transports(campaign)), {("clinical_logistic", 0)})

    def test_tampered_model_or_receipt_fails_closed_during_resume(self):
        for mutation in ("artifact", "receipt"):
            with self.subTest(mutation=mutation):
                private = self.root / mutation
                private.mkdir()
                campaign = subject.open_campaign(private, self.identity())
                source = self.root / (mutation + "_source")
                record = self.job(source)
                target = subject.persist_completed_job(campaign, "clinical_logistic", 0, source, record)
                if mutation == "artifact":
                    (target / "clinical_model.json").write_bytes(b"tampered")
                else:
                    receipt = subject._json(target.parent / subject.JOB_RECEIPT)
                    receipt["model_id"] = "clinical_xgboost"
                    self.write(target.parent / subject.JOB_RECEIPT, receipt)
                with self.assertRaises(ERRORS):
                    subject.completed_job_transports(campaign)

    def test_duplicate_completed_model_fold_receipts_are_rejected(self):
        _, campaign = self.campaign()
        source = self.root / "job"
        record = self.job(source)
        target = subject.persist_completed_job(campaign, "clinical_logistic", 0, source, record)
        duplicate = campaign / ("clinical_logistic_fold_0_attempt_" + "f" * 32)
        shutil.copytree(target.parent, duplicate)
        with self.assertRaises(ERRORS):
            subject.completed_job_transports(campaign)

    def test_atomic_json_publication_is_immutable(self):
        _, campaign = self.campaign()
        target = campaign / subject.CORE_FILENAME
        value = {"synthetic": True}
        subject._publish_json(target, value)
        before = target.read_bytes()
        subject._publish_json(target, value)
        self.assertEqual(target.read_bytes(), before)
        with self.assertRaises(ERRORS):
            subject._publish_json(target, {"synthetic": False})
        self.assertEqual(target.read_bytes(), before)

    def test_interrupted_oof_copy_has_no_final_filename_and_retry_preserves_stage(self):
        _, campaign = self.campaign()
        source = self.root / "source.json"
        self.write(source, {"rows": ["synthetic only"]})
        target = campaign / "clinical_logistic_out_of_fold_predictions.json"
        def interrupted(incoming, outgoing, **kwargs):
            outgoing.write(b'{"incomplete":')
            raise RuntimeError("interrupted copy")
        with patch.object(subject.shutil, "copyfileobj", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted copy"):
                subject._publish_copy(source, target, subject.sha_file(source))
        self.assertFalse(target.exists())
        stages = list(campaign.glob(".publication_*.part"))
        self.assertEqual(len(stages), 1)
        subject._publish_copy(source, target, subject.sha_file(source))
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertEqual(stages[0].read_bytes(), b'{"incomplete":')

    def test_all_fifteen_jobs_and_three_oof_files_have_verified_private_locations(self):
        _, campaign = self.campaign()
        output = self.root / "output"
        output.mkdir()
        artifacts = {}
        for model_id in MODELS:
            for fold in range(5):
                job = output / model_id / f"fold_{fold}"
                record = self.job(job, model_id, fold)
                inventory = subject._job_inventory(job)
                artifacts.update({f"{model_id}/fold_{fold}/{name}": digest for name, digest in inventory.items()})
                subject.persist_completed_job(campaign, model_id, fold, job, record)
            oof = output / model_id / "out_of_fold_predictions.json"
            self.write(oof, {"rows": ["synthetic only"], "model_id": model_id})
            artifacts[f"{model_id}/out_of_fold_predictions.json"] = subject.sha_file(oof)
        self.assertEqual(len(artifacts), 73)
        locations = subject._persist_final_artifacts(output, campaign, {"artifact_files_sha256": artifacts})
        self.assertEqual(set(locations), set(artifacts))
        for name, location in locations.items():
            self.assertEqual(subject.sha_file(campaign / location), artifacts[name])
        self.assertEqual(len(subject.completed_job_transports(campaign)), 15)

    def test_result_requires_all_three_models_five_folds_and_closed_scope(self):
        subject.validate_result(completed_result())
        mutations = [("model_ids", ["clinical_logistic"]), ("stage", "stage_2"),
            ("seed", 2027), ("folds_completed", [0, 1]), ("job_count", 14),
            ("execution_environment", "google_colab_gpu"), ("colab_runtime_executed", False),
            ("training_executed", False), ("cpu_only_guard_passed", False),
            ("participant_disjoint_folds_verified", False),
            ("clinical_preprocessing_fitted_inside_training_folds_only", False)]
        mutations.extend((field, True) for field in ("image_models_trained",
            "image_model_inference_executed", "pretrained_weights_loaded", "calibration_performed",
            "threshold_selected", "reserved_test_accessed", "reserved_test_predictive_evaluation_opened",
            "originals_modified", "partitions_modified", "encoder_unfreezing_executed",
            "predictive_performance_claimed", "unbiased_final_performance_claimed"))
        for field, value in mutations:
            with self.subTest(field=field), self.assertRaises(ERRORS):
                subject.validate_result({**completed_result(), field: value})

    def test_missing_duplicate_or_misattributed_clinical_folds_are_rejected(self):
        for mutation in ("missing_model", "missing_fold", "duplicate_fold"):
            changed = completed_result()
            if mutation == "missing_model":
                changed["model_results"].pop("clinical_xgboost")
            elif mutation == "missing_fold":
                changed["model_results"]["clinical_mlp"]["fold_results"].pop()
            else:
                changed["model_results"]["clinical_logistic"]["fold_results"][4]["validation_fold"] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ERRORS):
                subject.validate_result(changed)

    def test_absent_approval_fails_before_campaign_identity_or_data_access(self):
        with patch.object(subject, "_campaign_identity") as identity:
            with self.assertRaises(ERRORS):
                subject.train(PROJECT, "proof", "dev", "reference", "output", "private",
                    approval_statement="", code_revision="e" * 40, cpu_tests={}, runtime={})
        identity.assert_not_called()

    def fake_execution(self, output, completed_jobs=None):
        output.mkdir()
        artifacts = {}
        result = completed_result()
        for model_id in MODELS:
            for fold in range(5):
                directory = output / model_id / f"fold_{fold}"
                self.job(directory, model_id, fold)
                artifacts.update({f"{model_id}/fold_{fold}/{name}": digest
                    for name, digest in subject._job_inventory(directory).items()})
                result["model_results"][model_id]["fold_results"][fold]["reused_authenticated_complete_job"] = (
                    model_id, fold) in (completed_jobs or {})
            oof = output / model_id / "out_of_fold_predictions.json"
            self.write(oof, {"rows": ["synthetic only"], "model_id": model_id})
            artifacts[f"{model_id}/out_of_fold_predictions.json"] = subject.sha_file(oof)
        result["artifact_files_sha256"] = artifacts
        self.write(output / subject.CORE_FILENAME, result)
        return result

    def receipts(self):
        return ({"passed": True, "tests_run": 123, "skipped": 0,
            "scope": list(subject.TEST_MODULES), "test_runtime": "cpu_synthetic_no_participant_training"},
            {"version": subject.preparation.VERSION, "cpu_only": True,
             "training_executed": False, "pretrained_weights_downloaded": False})

    def execute_persistent(self, *, identity_effect=None):
        from knee import clinical_fivefold as core
        private = self.root / "private"
        private.mkdir(exist_ok=True)
        output = self.root / ("local_" + str(len(list(self.root.iterdir()))))
        tests, runtime = self.receipts()
        observed = {}
        def run(repo, proof, package, reference, destination, **kwargs):
            observed.update(kwargs)
            result = self.fake_execution(destination, kwargs["completed_jobs"])
            for model_id in MODELS:
                for fold in range(5):
                    directory = destination / model_id / f"fold_{fold}"
                    kwargs["job_completed"](model_id, fold, directory,
                        subject._json(directory / "training_job.json"))
            return result
        identity_patch = (patch.object(subject, "_campaign_identity", side_effect=identity_effect)
            if identity_effect is not None else patch.object(subject, "_campaign_identity", return_value=self.identity()))
        with identity_patch, patch.object(core, "run", side_effect=run):
            summary = subject.train(PROJECT, self.root / "proof", self.root / "development",
                self.root / "reference/reference.json", output, private,
                approval_statement=subject.APPROVAL_STATEMENT, code_revision="e" * 40,
                cpu_tests=tests, runtime=runtime)
        return summary, output, private, observed

    def test_persistent_mock_execution_saves_all_fifteen_jobs_and_every_artifact(self):
        summary, output, private, observed = self.execute_persistent()
        campaign = private / summary["persistent_campaign_folder"]
        self.assertEqual(observed["completed_jobs"], {})
        self.assertEqual(len(subject.completed_job_transports(campaign)), 15)
        self.assertEqual(summary["completed_jobs_saved_individually"], 15)
        self.assertEqual(len(summary["private_artifact_locations"]), 73)
        self.assertTrue(summary["result_files_saved_and_readback_verified"])
        for name, location in summary["private_artifact_locations"].items():
            self.assertEqual(subject.sha_file(campaign / location), summary["artifact_files_sha256"][name])
            self.assertEqual((campaign / location).read_bytes(), (output / name).read_bytes())

    def test_second_mock_invocation_reauthenticates_and_reuses_all_completed_jobs(self):
        summary, _, private, _ = self.execute_persistent()
        before = {path: path.read_bytes() for path in private.rglob("*") if path.is_file()}
        again, _, _, observed = self.execute_persistent()
        self.assertEqual(set(observed["completed_jobs"]), {(model, fold) for model in MODELS for fold in range(5)})
        self.assertEqual(again, summary)
        self.assertTrue(all(path.read_bytes() == raw for path, raw in before.items()))

    def test_interrupted_summary_rebuild_preserves_completed_core_and_all_job_bytes(self):
        summary, _, private, _ = self.execute_persistent()
        campaign = private / summary["persistent_campaign_folder"]
        before = (campaign / subject.CORE_FILENAME).read_bytes()
        (campaign / subject.SUMMARY_FILENAME).unlink()
        recovered, _, _, observed = self.execute_persistent()
        self.assertEqual(len(observed["completed_jobs"]), 15)
        self.assertEqual((campaign / subject.CORE_FILENAME).read_bytes(), before)
        self.assertTrue(recovered["result_files_saved_and_readback_verified"])

    def test_changed_source_identity_during_training_prevents_final_publication(self):
        original = self.identity()
        changed = {**original, "source_files_sha256": {"engine": "d" * 64}}
        with self.assertRaises(ERRORS):
            self.execute_persistent(identity_effect=[original, changed])
        campaign = next((self.root / "private").iterdir())
        self.assertEqual(len(subject.completed_job_transports(campaign)), 15)
        self.assertFalse((campaign / subject.CORE_FILENAME).exists())
        self.assertFalse((campaign / subject.SUMMARY_FILENAME).exists())

    def test_nonexact_approval_test_receipts_and_gpu_runtime_fail_before_data_access(self):
        tests, runtime = self.receipts()
        variants = [(subject.APPROVAL_STATEMENT + " otro modelo", tests, runtime)]
        for field, value in (("passed", False), ("skipped", 1), ("scope", []),
                ("test_runtime", "real_participant_training"), ("tests_run", 0)):
            variants.append((subject.APPROVAL_STATEMENT, {**tests, field: value}, runtime))
        for field, value in (("cpu_only", False), ("training_executed", True),
                ("pretrained_weights_downloaded", True), ("version", "gpu_runtime")):
            variants.append((subject.APPROVAL_STATEMENT, tests, {**runtime, field: value}))
        for approval, receipt, setup in variants:
            with self.subTest(approval=approval, receipt=receipt, setup=setup), \
                    patch.object(subject, "_campaign_identity") as identity:
                with self.assertRaises(ERRORS):
                    subject.train(PROJECT, "proof", "dev", "reference", "output", "private",
                        approval_statement=approval, code_revision="e" * 40, cpu_tests=receipt, runtime=setup)
                identity.assert_not_called()

    def test_campaign_identity_binds_preparation_reference_package_and_sources(self):
        from knee import training_smoke as historical
        proof = self.root / "proof"
        self.write(proof / subject.preparation.SUMMARY_FILENAME, {"synthetic proof": True})
        reference = self.root / "reference.json"
        self.write(reference, {"version": "model_inputs_loading_preflight_v1.0",
            "status": "development_inputs_and_cpu_loading_verified",
            "sources_sha256": {"development_package": "a" * 64}})
        with patch.object(historical, "validate_preparation", return_value={}):
            identity = subject._campaign_identity(PROJECT, proof, reference, "e" * 40)
        self.assertEqual(identity["development_package_sha256"], "a" * 64)
        self.assertEqual(identity["model_input_reference_sha256"], subject.sha_file(reference))
        self.assertEqual(identity["preparation_summary_sha256"],
            subject.sha_file(proof / subject.preparation.SUMMARY_FILENAME))
        self.assertEqual(set(identity["source_files_sha256"]), set(subject.SOURCE_FILES))
        self.assertEqual(identity["model_ids"], list(MODELS))
        self.assertEqual(identity["device"], "cpu")

    def test_resume_ignores_execution_metadata_but_preserves_scientific_results(self):
        original = completed_result()
        original["environment"] = {"colab_release_tag": "old", "torch_threads": 2}
        fresh = deepcopy(original)
        fresh["executed_at_utc"] = "2026-10-25T17:00:00+00:00"
        fresh["environment"] = {"colab_release_tag": "new", "torch_threads": 2}
        for model in MODELS:
            for row in fresh["model_results"][model]["fold_results"]:
                row["reused_authenticated_complete_job"] = True
        subject._same_completed_campaign(original, fresh)
        fresh["model_results"]["clinical_mlp"]["fivefold_selection_metrics"]["average_precision_score"] = .99
        with self.assertRaises(ERRORS):
            subject._same_completed_campaign(original, fresh)


if __name__ == "__main__":
    unittest.main()
