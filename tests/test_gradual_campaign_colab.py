"""Synthetic orchestration receipts and phase boundaries; no participant fit."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from knee import gradual_campaign_colab as subject
from knee import training_gpu_runtime as gpu
from knee.colab_data import sha_file

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1234567890abcdef1234567890abcdef12345678"


def runtime_receipt():
    return {"version": gpu.VERSION, "training_executed": False, "host_python_modified": False,
        "pretrained_weights_downloaded": False, "cpu_only": False, "python_version": "3.12.14",
        "source_sha256": {name: sha_file(ROOT / name) for name in gpu.SOURCE_FILES}}


def tests_receipt():
    return {"passed": True, "tests_run": 99, "skipped": 0, "scope": list(subject.TEST_MODULES),
        "test_runtime": "gpu_build_cpu_compute_no_training"}


def complete_result():
    metrics = {"average_precision_score": .5, "roc_auc": .5, "brier": .25}
    return {"version": "development_gradual_image_campaign_v1.0", "campaign": subject.CAMPAIGN,
        "status": "four_model_gradual_development_campaign_complete",
        "scope": "development_only", "stage": "stage_2", "job_count": 20,
        "model_order": list(subject.MODELS), "folds_completed": list(range(5)), "seed": 2026,
        "training_executed_in_campaign": True, "reserved_test_accessed": False,
        "reserved_test_predictive_evaluation_opened": False, "calibration_performed": False,
        "threshold_selected": False, "final_refit_executed": False, "fusion_training_executed": False,
        "originals_modified": False, "partitions_modified": False,
        "completed_jobs_preserved": 0, "jobs_executed_this_invocation": 20,
        "training_executed_this_invocation": True, "development_knees": 30, "development_participants": 20,
        "job_results": [{"key": f"clinical_main__{model}__fold_{fold}", "job_record_sha256": "a" * 64,
            "metrics": deepcopy(metrics), "preserved_complete_job": False}
            for model in subject.MODELS for fold in range(5)],
        "model_results": {"clinical_main__" + model: {"fivefold_selection_metrics": deepcopy(metrics),
            "pooled_oof_metrics_descriptive_only": deepcopy(metrics), "out_of_fold_predictions": 30}
            for model in subject.MODELS},
        "oof_files_sha256": {"clinical_main__" + model + "__oof.json": "b" * 64 for model in subject.MODELS},
        "completed_checkpoint_payloads_copied_on_recovery": False, "checkpoint_payloads_rehashed_on_recovery": False,
        "unbiased_final_performance_claimed": False, "checkpoint_rehash_required_before_weight_reuse": True,
        "new_job_payloads_verified_on_publication": True, "full_encoder_training_authorized": True,
        "encoder_unfreezing_executed": True, "result_files_saved_and_readback_verified": True}


class GradualCampaignColabTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="g42_abi_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.records, self.package, self.reference_dir, self.weights = (
            self.root / name for name in ("records", "package", "reference", "weights"))
        for path in (self.records, self.package, self.reference_dir, self.weights):
            path.mkdir()
        self.reference = self.reference_dir / "reference.json"
        self.reference.write_text("{}", encoding="utf-8")
        self.output = self.root / "preparation"

    def prepare_fixture(self, repository, records, package, reference, output, **kwargs):
        self.assertEqual(repository, ROOT)
        self.assertEqual(records, self.records)
        self.assertEqual(package, self.package)
        self.assertEqual(reference, self.reference)
        self.assertEqual(kwargs["code_revision"], REVISION)
        output.mkdir()
        subject._save(output / subject.SELECTION_FILENAME, {"synthetic_selection": True})
        record = {"status": "gradual_campaign_prepared_no_training", "model_order": list(subject.MODELS),
            "job_count": 20, "training_executed": False,
            "selection_evidence_sha256": sha_file(output / subject.SELECTION_FILENAME),
            "development_package_sha256": "a" * 64}
        subject._save(output / subject.PREPARATION_FILENAME, record)
        return record

    def prepare(self):
        from knee import gradual_campaign as core
        with patch.object(core, "prepare", side_effect=self.prepare_fixture) as prepare, \
             patch.object(core, "run", side_effect=AssertionError("Preparation must not train")):
            result = subject.prepare(ROOT, self.records, self.package, self.reference, self.output,
                code_revision=REVISION, cpu_tests=tests_receipt(), runtime=runtime_receipt())
        prepare.assert_called_once()
        return result

    def test_preparation_binds_selection_code_and_runtime_and_persists_only_small_records(self):
        result = self.prepare()
        self.assertFalse(result["training_executed"])
        self.assertTrue(result["preparation_saved_and_readback_verified"])
        receipt = subject.validate_preparation_receipt(ROOT, self.output, code_revision=REVISION)
        self.assertFalse(receipt["training_executed"])
        self.assertEqual(receipt["source_files_sha256"], subject._source_hashes(ROOT))
        private = self.records / result["persistent_preparation_folder"]
        expected = {subject.PREPARATION_FILENAME, subject.SELECTION_FILENAME, subject.PREPARATION_RECEIPT}
        self.assertEqual({p.name for p in private.iterdir()}, expected)
        for name in expected:
            self.assertEqual((private / name).read_bytes(), (self.output / name).read_bytes())

    def test_invalid_runtime_or_tests_fail_before_core_preparation(self):
        from knee import gradual_campaign as core
        invalid = [("tests", {"passed": False}), ("tests", {"skipped": 1}),
            ("tests", {"scope": []}), ("tests", {"test_runtime": "synthetic_cpu_no_participant_training"}),
            ("runtime", {"cpu_only": True}), ("runtime", {"training_executed": True}),
            ("runtime", {"pretrained_weights_downloaded": True}), ("runtime", {"source_sha256": {}})]
        for which, changes in invalid:
            tests, runtime = tests_receipt(), runtime_receipt()
            (tests if which == "tests" else runtime).update(changes)
            with self.subTest(which=which, changes=changes), patch.object(core, "prepare") as prepare, \
                    self.assertRaises(ValueError):
                subject.prepare(ROOT, self.records, self.package, self.reference, self.output,
                    code_revision=REVISION, cpu_tests=tests, runtime=runtime)
            prepare.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_wrong_preparation_model_set_cannot_be_published(self):
        from knee import gradual_campaign as core
        def wrong(*args, **kwargs):
            record = self.prepare_fixture(*args, **kwargs)
            return {**record, "model_order": list(reversed(subject.MODELS))}
        with patch.object(core, "prepare", side_effect=wrong), self.assertRaisesRegex(ValueError, "exact_untrained"):
            subject.prepare(ROOT, self.records, self.package, self.reference, self.output,
                code_revision=REVISION, cpu_tests=tests_receipt(), runtime=runtime_receipt())
        self.assertEqual(list(self.records.iterdir()), [])

    def test_corrupt_preparation_or_changed_revision_prevents_training(self):
        from knee import gradual_campaign as core
        self.prepare()
        with patch.object(core, "run") as run, self.assertRaisesRegex(ValueError, "unchanged_exact"):
            subject.train(ROOT, self.output, self.package, self.reference, self.root / "new_output",
                weights_root=self.weights, persistent_root=self.records, code_revision="b" * 40)
        run.assert_not_called()
        selection = self.output / subject.SELECTION_FILENAME
        selection.write_text('{"changed":true}', encoding="utf-8")
        with patch.object(core, "run") as run, self.assertRaisesRegex(ValueError, "unchanged_exact"):
            subject.train(ROOT, self.output, self.package, self.reference, self.root / "new_output",
                weights_root=self.weights, persistent_root=self.records, code_revision=REVISION)
        run.assert_not_called()

    def test_training_delegates_exact_cuda_scope_and_publishes_summary_only(self):
        from knee import gradual_campaign as core
        self.prepare()
        destination = self.root / "execution"
        def run(repository, preparation, package, reference, output, **kwargs):
            self.assertEqual((repository, preparation, package, reference, output),
                (ROOT, self.output, self.package, self.reference, destination))
            self.assertEqual(kwargs["device"], "cuda:0")
            self.assertIs(kwargs["allow_download"], True)
            self.assertEqual(kwargs["code_revision"], REVISION)
            self.assertEqual(kwargs["weights_root"], self.weights)
            self.assertEqual(kwargs["persistent_root"], self.records)
            self.assertEqual(kwargs["cpu_tests"], tests_receipt())
            self.assertEqual(kwargs["runtime"], runtime_receipt())
            output.mkdir()
            return complete_result()
        with patch.object(core, "run", side_effect=run) as execute, patch.object(gpu, "_host") as host:
            result = subject.train(ROOT, self.output, self.package, self.reference, destination,
                weights_root=self.weights, persistent_root=self.records, code_revision=REVISION)
        execute.assert_called_once()
        host.assert_called_once_with()
        self.assertEqual(result["colab_orchestration_version"], subject.VERSION)
        self.assertEqual({p.name for p in destination.iterdir()}, {subject.SUMMARY_FILENAME})
        folders = list(self.records.glob("gradual_colab42_*"))
        self.assertEqual(len(folders), 1)
        self.assertEqual((folders[0] / subject.SUMMARY_FILENAME).read_bytes(),
            (destination / subject.SUMMARY_FILENAME).read_bytes())

    def test_non_colab_host_cannot_claim_real_colab_execution(self):
        from knee import gradual_campaign as core
        self.prepare()
        with patch.object(gpu, "_host", side_effect=ValueError("actual Linux Colab required")), \
             patch.object(core, "run") as run, self.assertRaisesRegex(ValueError, "actual Linux Colab"):
            subject.train(ROOT, self.output, self.package, self.reference, self.root / "execution",
                weights_root=self.weights, persistent_root=self.records, code_revision=REVISION)
        run.assert_not_called()

    def test_scope_expansion_is_rejected_before_result_publication(self):
        for change in ({"job_count": 25}, {"stage": "stage_1"}, {"reserved_test_accessed": True},
                {"fusion_training_executed": True}, {"model_order": []}, {"seed": 2027}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                subject.validate_result({**complete_result(), **change})

    def test_incomplete_duplicate_or_inconsistent_metrics_are_not_complete_results(self):
        for change in ("duplicate", "nan", "mean", "coverage", "count", "status"):
            result = complete_result()
            if change == "duplicate":
                result["job_results"][-1] = deepcopy(result["job_results"][0])
            elif change == "nan":
                result["job_results"][0]["metrics"]["roc_auc"] = float("nan")
            elif change == "mean":
                result["model_results"]["clinical_main__image_vit_b16"]["fivefold_selection_metrics"]["roc_auc"] = .6
            elif change == "coverage":
                result["model_results"]["clinical_main__image_vit_b16"]["out_of_fold_predictions"] = 29
            elif change == "count":
                result["completed_jobs_preserved"] = 1
            else:
                result["status"] = "partial"
            with self.subTest(change=change), self.assertRaises(ValueError):
                subject.validate_result(result)

    def test_runtime_and_environment_only_delegate_registered_gpu_setup(self):
        with patch.object(gpu, "setup_runtime", return_value={"synthetic": True}) as setup, \
             patch.object(gpu, "runtime_environment", return_value={"synthetic": "environment"}) as environment:
            self.assertEqual(subject.setup_runtime(ROOT, self.root / "runtime"), {"synthetic": True})
            self.assertEqual(subject.runtime_environment(ROOT), {"synthetic": "environment"})
            setup.assert_called_once_with(ROOT, self.root / "runtime")
            environment.assert_called_once_with(ROOT)
            with self.assertRaises(ValueError):
                subject.setup_runtime(ROOT, self.root / "runtime", "remaining_encoders_frozen")
            with self.assertRaises(ValueError):
                subject.runtime_environment(ROOT, "clinical")


if __name__ == "__main__":
    unittest.main()
