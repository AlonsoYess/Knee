"""Completed-proof and eval-only preparation tests; never fit a predictor."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee import training_preparation as subject
from knee.colab_data import sha_file
from knee.final_models_colab import TEST_MODULES
from knee.final_model_release import PINS
from test_final_models_colab import synthetic_result

ROOT = Path(__file__).resolve().parents[1]


def preflight_fixture(root):
    directory = Path(root) / "final_model_check_synthetic"
    directory.mkdir()
    checkpoint = synthetic_result()
    checkpoint.update(source_files_sha256=subject.SOURCE_33,
        release_component_canonical_sha256=PINS)
    (directory / "checkpoint_verification.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    summary = {key:value for key,value in checkpoint.items() if key != "model_records"}
    summary.update(code_revision="b687d64e18d8e34b106c34e4cc28abd683afc000",
        local_rehearsal=False, cpu_only_guard_passed=True, result_files_saved_and_readback_verified=True,
        checkpoint_record_sha256=sha_file(directory / "checkpoint_verification.json"),
        cpu_tests={"passed":True, "tests_run":195, "skipped":0, "scope":list(TEST_MODULES)},
        runtime_requirements_sha256=subject.SOURCE_33["requirements-final-models.txt"],
        training_authorized=False, training_smoke_test_passed=False, gpu_resource_profile_recorded=False,
        shortlist_search_closed=True, pending_authorized_access=["image_dinov3_vits16"],
        executed_at_utc="2026-10-22T15:48:56.943350+00:00")
    (directory / "colab_final_models_check.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory


class TrainingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="training_preparation_synthetic_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.proof = preflight_fixture(self.root)

    def edit_summary(self, function):
        path = self.proof / "colab_final_models_check.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        function(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_accepts_complete_exact_prior_colab_proof_only(self):
        result = subject.validate_preflight(self.proof)
        self.assertTrue(result["colab_runtime_executed"])
        self.assertFalse(result["training_authorized"])

    def test_checkpoint_tampering_fails_even_when_json_stays_valid(self):
        path = self.proof / "checkpoint_verification.json"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exact_checkpoint"):
            subject.validate_preflight(self.proof)

    def test_wrong_revision_test_count_skips_and_training_flags_are_rejected(self):
        original = (self.proof / "colab_final_models_check.json").read_bytes()
        changes = [lambda v:v.update(code_revision="a"*40), lambda v:v.update(training_authorized=True),
            lambda v:v.update(training_executed=True), lambda v:v["cpu_tests"].update(skipped=1),
            lambda v:v["cpu_tests"].update(tests_run=194), lambda v:v.update(local_rehearsal=True)]
        for change in changes:
            (self.proof / "colab_final_models_check.json").write_bytes(original)
            self.edit_summary(change)
            with self.subTest(change=change), self.assertRaises((ValueError, RuntimeError)):
                subject.validate_preflight(self.proof)

    def test_changed_sources_and_release_cannot_be_resealed_as_previous_proof(self):
        path = self.proof / "checkpoint_verification.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source_files_sha256"]["src/knee/checkpoint_transport.py"] = "0"*64
        path.write_text(json.dumps(value), encoding="utf-8")
        self.edit_summary(lambda v:v.update(source_files_sha256=value["source_files_sha256"],
            checkpoint_record_sha256=sha_file(path)))
        with self.assertRaisesRegex(ValueError, "code_and_release"):
            subject.validate_preflight(self.proof)

    def test_missing_pair_and_duplicate_json_keys_fail_closed(self):
        path = self.root / "duplicate.json"
        path.write_text('{"key":1,"key":2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            subject.read_json(path)
        with self.assertRaises(ValueError):
            subject.validate_preflight(self.root)

    def test_naive_or_invalid_execution_timestamp_is_rejected(self):
        self.edit_summary(lambda v:v.update(executed_at_utc="2026-10-22T15:00:00"))
        with self.assertRaisesRegex(ValueError, "timestamp"):
            subject.validate_preflight(self.proof)

    def test_preparation_runs_only_synthetic_eval_and_never_creates_optimizer(self):
        output = self.root / "prepared"
        with patch("torch.optim.AdamW", side_effect=AssertionError("optimizer forbidden")), \
             patch("torch.Tensor.backward", side_effect=AssertionError("backward forbidden")), \
             patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("fit forbidden")), \
             patch("xgboost.XGBClassifier.fit", side_effect=AssertionError("fit forbidden")):
            result = subject.run(ROOT, self.proof, output)
        self.assertEqual(result["status"], subject.STATUS)
        self.assertTrue(result["training_runner_implemented"])
        for key in ("training_executed", "optimizer_created", "backward_executed", "participant_data_opened",
                    "reserved_test_predictive_evaluation_opened", "training_authorized"):
            self.assertFalse(result[key])
        self.assertEqual(subject.read_json(output / subject.FILENAME), result)
        self.assertEqual(len(result["accessible_image_models_implemented"]), 6)
        self.assertEqual(len(result["conditional_fusion_models_implemented"]), 2)

    def test_prior_evidence_is_unchanged_by_preparation(self):
        before = {p.name:p.read_bytes() for p in self.proof.iterdir()}
        subject.run(ROOT, self.proof, self.root / "prepared")
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.proof.iterdir()})

    def test_existing_or_source_overlapping_output_is_rejected(self):
        for output in (self.proof, ROOT / "unsafe_preparation", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                subject.run(ROOT, self.proof, output)


if __name__ == "__main__":
    unittest.main()
