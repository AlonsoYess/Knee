"""Synthetic records and mocked fits only; no optimizer or estimator is fitted."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from knee import training_runner as r
from knee import training_authority as a
from knee.colab_data import sha_file
from knee.final_model_release import load_release, PINS, RELEASE_PATH
from knee.training_engine import binary_metrics
from knee.training_plan import training_fold_class_weight
from test_training_data import reference_fixture, synthetic_inputs

ROOT = Path(__file__).resolve().parents[1]


def request(package_hash, *, model_id="clinical_logistic", stage="stage_1", fold=0, operation="development_fold"):
    encoder = None if model_id.startswith("clinical_") else model_id
    if model_id.startswith("fusion_"):
        encoder = "image_densenet121"
    return {"version": a.REQUEST_VERSION, "operation": operation, "model_id": model_id,
        "image_encoder_id": encoder, "stage": stage, "validation_fold": fold, "seed": 2026,
        "development_package_sha256": package_hash, "image_micro_batch_size": 4,
        "selection_evidence_sha256": None if stage == "stage_1" else "a" * 64,
        "fold_image_job_sha256": "b" * 64 if stage == "fusion" else None}


def authority(req, registry):
    return {"version": a.VERSION, "training_authorized": True, "reserved_test_access_authorized": False,
        "approval_source": "direct_investigator_authorization", "approval_statement": "SYNTHETIC MOCK ONLY; NO FIT EXECUTED",
        "approved_at_utc": "2026-10-22T00:00:00+00:00", "request_sha256": a.digest(req),
        "modeling_release_sha256": PINS[RELEASE_PATH],
        "eligibility_gates": {key: True for key in registry["eligibility_gates"]},
        "full_encoder_training_authorized": req["stage"] != "stage_1"}


def fake_conventional_fit(model_id, fold, plan, seed, output):
    """Write artificial predictions and model description; never fit anything."""
    targets = list(fold.validation._targets)
    predictions = {"targets": targets, "logits": [2. if target else -2. for target in targets],
        "identities": [list(identity) for identity in fold.validation.identities]}
    r.save_json(output / "best_validation_predictions.json", predictions)
    if model_id == "clinical_logistic":
        r.save_json(output / "clinical_model.json", {"synthetic_mock": True})
    else:
        (output / "clinical_model.ubj").write_bytes(b"SYNTHETIC MOCK; NO MODEL FIT")
    return {"status": "training_fold_complete", "metrics": binary_metrics(targets, predictions["logits"]),
            "optimizer_created": False}


def sealed_image_job(directory, package_hash, rows, registry, *, model_id="image_densenet121", stage="stage_1", fold=0):
    """Artificial proof fixture; model files are never loaded for inference."""
    output = Path(directory)
    output.mkdir()
    req = request(package_hash, model_id=model_id, stage=stage, fold=fold)
    held_out = [row for row in rows if row["cv_fold"] == fold]
    predictions = {"targets": [row["PROGRESION_48M"] for row in held_out],
        "logits": [2. if row["PROGRESION_48M"] else -2. for row in held_out],
        "identities": [[row["src_subject_id"], row["side"]] for row in held_out]}
    r.save_json(output / "best_validation_predictions.json", predictions)
    r.save_json(output / "training_record.json", {"synthetic_mock": True})
    r.save_json(output / "clinical_state.json", {"synthetic_mock": True})
    for name in ("best_model.safetensors", "last_model.safetensors"):
        (output / name).write_bytes(b"ARTIFICIAL HASH FIXTURE; NEVER LOADED AS CHECKPOINT")
    weight = training_fold_class_weight(rows, fold)
    record = {"version": r.VERSION, "scope": "development_only", "request": req,
        "request_sha256": a.digest(req), "plan_sha256": PINS["configs/training_plan_v1_1.json"],
        "eligible_for_model_selection": True, "reserved_test_accessed": False,
        "weights_provenance": {"fit_scope": "development_training_fold", "validation_fold": fold,
            "train_identity_sha256": weight["train_identity_sha256"],
            "trained_participants": sorted({row["src_subject_id"] for row in rows if row["cv_fold"] != fold}),
            "reserved_test_accessed": False},
        "metrics": binary_metrics(predictions["targets"], predictions["logits"]),
        "eligibility_gates": {key: True for key in registry["eligibility_gates"]},
        "selection_evidence": None, "engine_status": "training_fold_complete", "training_executed": True,
        "files_sha256": {path.name: sha_file(path) for path in output.iterdir()}}
    record["record_sha256"] = a.digest(record)
    r.save_json(output / r.JOB_FILENAME, record)
    return record


def overwrite_record(directory, record):
    """Fixture mutation; only artificial temporary JSON is replaced."""
    target = Path(directory) / r.JOB_FILENAME
    record["record_sha256"] = a.digest({key: value for key, value in record.items() if key != "record_sha256"})
    target.unlink()
    r.save_json(target, record)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.plan, cls.registry = load_release(ROOT)

    def fixture(self, directory):
        base = Path(directory)
        source = base / "sources"
        source.mkdir()
        value = reference_fixture(source)
        (base / "outputs").mkdir()
        value["output"] = base / "outputs" / "synthetic_job"
        return value

    def execute_mock(self, fixture, *, operation="development_fold", side_effect=fake_conventional_fit):
        req = request(fixture["package_sha256"], operation=operation)
        with patch.object(r, "_fit_conventional", side_effect=side_effect) as fit, \
             patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("actual fit forbidden")), \
             patch("torch.optim.AdamW", side_effect=AssertionError("actual optimizer forbidden")):
            record = r.execute(req, authority(req, self.registry), repository_root=ROOT,
                package_dir=fixture["package"], reference_path=fixture["reference_path"],
                weights_root=None, output_dir=fixture["output"])
        return record, fit

    def test_no_authority_fails_before_any_source_or_repository_read(self):
        with patch.object(r, "checked_path", side_effect=AssertionError("I/O forbidden")), \
             patch.object(r, "load_release", side_effect=AssertionError("repository read forbidden")), \
             patch.object(r, "_fit_conventional", side_effect=AssertionError("fit forbidden")):
            with self.assertRaisesRegex(ValueError, "separate_training_authority_required_before_execution"):
                r.execute({}, None, repository_root="missing", package_dir="missing", reference_path="missing",
                          weights_root=None, output_dir="missing")

    def test_invalid_authority_cannot_open_development_package(self):
        req = request("a" * 64)
        invalid = authority(req, self.registry)
        invalid["training_authorized"] = False
        with patch("knee.training_data.load_sealed_inputs", side_effect=AssertionError("package open forbidden")):
            with self.assertRaisesRegex(ValueError, "explicit_investigator_training_authority_required"):
                r.execute(req, invalid, repository_root=ROOT, package_dir="missing", reference_path="missing",
                          weights_root=None, output_dir="missing")

    def test_mocked_clinical_lifecycle_seals_artifacts_and_recomputable_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(tmp)
            record, fit = self.execute_mock(fixture)
            self.assertEqual(fit.call_count, 1)
            self.assertTrue(record["eligible_for_model_selection"])
            self.assertEqual(r.completed_job(fixture["output"], fixture["package_sha256"], self.plan, self.registry), record)
            self.assertEqual(set(record["files_sha256"]),
                {"clinical_model.json", "clinical_state.json", "best_validation_predictions.json"})
            self.assertFalse(record["reserved_test_accessed"])
            self.assertFalse(any("reserved" in path.name for path in fixture["output"].iterdir()))

    def test_mocked_smoke_cannot_enter_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(tmp)
            record, _ = self.execute_mock(fixture, operation="single_fold_smoke")
            self.assertFalse(record["eligible_for_model_selection"])
            with self.assertRaisesRegex(ValueError, "authenticated_complete_development_job_required"):
                r.completed_job(fixture["output"], fixture["package_sha256"], self.plan, self.registry)

    def test_changed_artifact_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(tmp)
            self.execute_mock(fixture)
            with (fixture["output"] / "clinical_model.json").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ValueError, "completed_training_artifact_changed"):
                r.completed_job(fixture["output"], fixture["package_sha256"], self.plan, self.registry)

    def test_annotated_metric_must_match_sealed_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(tmp)
            record, _ = self.execute_mock(fixture)
            record["metrics"]["average_precision_score"] = .01
            overwrite_record(fixture["output"], record)
            with self.assertRaisesRegex(ValueError, "selection_metrics_must_match_sealed_predictions"):
                r.completed_job(fixture["output"], fixture["package_sha256"], self.plan, self.registry)

    def test_neural_record_requires_full_checkpoint_manifest_and_complete_status(self):
        rows, _ = synthetic_inputs()
        for mutation in ("missing_best", "smoke_status", "not_trained"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "image"
                record = sealed_image_job(output, "a" * 64, rows, self.registry)
                if mutation == "missing_best":
                    record["files_sha256"].pop("best_model.safetensors")
                elif mutation == "smoke_status":
                    record["engine_status"] = "bounded_training_smoke_complete"
                else:
                    record["training_executed"] = False
                overwrite_record(output, record)
                with self.assertRaises(ValueError):
                    r.completed_job(output, "a" * 64, self.plan, self.registry)

    def test_input_mutation_during_mocked_fit_prevents_completed_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(tmp)
            def mutate(*args):
                result = fake_conventional_fit(*args)
                with fixture["archive"].open("ab") as handle:
                    handle.write(b"synthetic source changed")
                return result
            with self.assertRaisesRegex(ValueError, "sealed_training_source_changed_during_execution"):
                self.execute_mock(fixture, side_effect=mutate)
            self.assertFalse((fixture["output"] / r.JOB_FILENAME).exists())

    def test_stage_two_requires_hash_bound_selection_before_package_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(tmp)
            req = request(fixture["package_sha256"], model_id="image_densenet121", stage="stage_2")
            with patch("knee.training_data.load_sealed_inputs", side_effect=AssertionError("package open forbidden")):
                with self.assertRaisesRegex(ValueError, "exact_approved_selection_evidence_required"):
                    r.execute(req, authority(req, self.registry), repository_root=ROOT,
                        package_dir=fixture["package"], reference_path=fixture["reference_path"], weights_root=None,
                        output_dir=fixture["output"])

    def test_output_must_not_overlap_original_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(tmp)
            fixture["output"] = fixture["package"] / "nested_result"
            with self.assertRaisesRegex(ValueError, "training_output_must_not_overlap_sources"):
                self.execute_mock(fixture)

    def test_mandatory_screening_cannot_be_omitted_from_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = request("a" * 64, model_id="image_densenet121", stage="stage_2")
            evidence = {"version": "development_image_selection_evidence_v1.0", "package_sha256": "a" * 64,
                "screening_jobs": [], "excluded_modern_models": {key: "synthetic technical exclusion"
                    for key in self.plan["radiographic_models"]["conditional_challengers"]},
                "stage_two_jobs": [], "selected_image_encoder_id": None}
            path = Path(tmp) / "selection.json"
            r.save_json(path, evidence)
            req["selection_evidence_sha256"] = sha_file(path)
            with self.assertRaisesRegex(ValueError, "mandatory_baselines_require_complete_five_fold_screening"):
                r.validate_selection(path, req, self.plan, self.registry)

    def test_fusion_selection_strict_point_zero_one_boundary_uses_raw_fold_decimals(self):
        for lower_ap, expected in ((.49, "image_densenet121"), (.4901, "image_vit_b16")):
            with self.subTest(lower_ap=lower_ap), tempfile.TemporaryDirectory() as tmp:
                def folds(ap, auc):
                    return [{"validation_fold": fold, "average_precision_score": ap,
                        "roc_auc": auc, "brier": .2} for fold in range(5)]
                screened = {key: {"folds": folds(.5, .6), "gates": {gate: True for gate in self.registry["eligibility_gates"]}}
                    for key in self.plan["radiographic_models"]["mandatory"]}
                tuned = {"image_densenet121": {"folds": folds(.5, .6)},
                    "image_vit_b16": {"folds": folds(lower_ap, .99)}}
                evidence = {"version": "development_image_selection_evidence_v1.0", "package_sha256": "a" * 64,
                    "screening_jobs": [], "excluded_modern_models": {key: "synthetic technical exclusion"
                        for key in self.plan["radiographic_models"]["conditional_challengers"]},
                    "stage_two_jobs": [], "selected_image_encoder_id": expected}
                path = Path(tmp) / "selection.json"
                r.save_json(path, evidence)
                req = request("a" * 64, model_id="fusion_intermediate_concat", stage="fusion")
                req["image_encoder_id"] = expected
                req["selection_evidence_sha256"] = sha_file(path)
                with patch.object(r, "_fold_reports", side_effect=[screened, tuned]):
                    selected = r.validate_selection(path, req, self.plan, self.registry)
                self.assertEqual(selected["selected_image_encoder_id"], expected)

    def test_excluded_unpromoted_modern_model_cannot_enter_stage_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            screened = {key: {"folds": [{"validation_fold": fold, "average_precision_score": .5,
                "roc_auc": .6, "brier": .2} for fold in range(5)], "gates": {gate: True
                    for gate in self.registry["eligibility_gates"]}} for key in self.plan["radiographic_models"]["mandatory"]}
            evidence = {"version": "development_image_selection_evidence_v1.0", "package_sha256": "a" * 64,
                "screening_jobs": [], "excluded_modern_models": {key: "synthetic technical exclusion"
                    for key in self.plan["radiographic_models"]["conditional_challengers"]},
                "stage_two_jobs": [], "selected_image_encoder_id": None}
            path = Path(tmp) / "selection.json"
            r.save_json(path, evidence)
            req = request("a" * 64, model_id="image_mambaout_tiny", stage="stage_2")
            req["selection_evidence_sha256"] = sha_file(path)
            with patch.object(r, "_fold_reports", return_value=screened):
                with self.assertRaisesRegex(ValueError, "stage_two_requires_mandatory_or_oof_promoted_encoder"):
                    r.validate_selection(path, req, self.plan, self.registry)

    def test_screening_requires_five_distinct_folds_and_excludes_smoke(self):
        rows, _ = synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            entries = []
            for fold in range(5):
                directory = Path(tmp) / str(fold)
                record = sealed_image_job(directory, "a" * 64, rows, self.registry, fold=fold)
                entries.append({"directory": str(directory), "record_sha256": record["record_sha256"]})
            complete = r._fold_reports(entries, "a" * 64, self.plan, self.registry, stage="stage_1")
            self.assertEqual(len(complete["image_densenet121"]["folds"]), 5)
            for invalid in (entries[:-1], entries + entries[:1]):
                with self.assertRaises(ValueError):
                    r._fold_reports(invalid, "a" * 64, self.plan, self.registry, stage="stage_1")

    def test_same_fold_reuse_rejects_wrong_fold_seed_and_modified_checkpoint(self):
        rows, _ = synthetic_inputs()
        for mutation in ("fold", "seed", "checkpoint"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "image"
                record = sealed_image_job(output, "a" * 64, rows, self.registry, stage="stage_2")
                req = request("a" * 64, model_id="fusion_intermediate_concat", stage="fusion")
                req["fold_image_job_sha256"] = record["record_sha256"]
                if mutation == "fold": req["validation_fold"] = 1
                elif mutation == "seed": req["seed"] = 2027
                else:
                    with (output / "best_model.safetensors").open("ab") as handle:
                        handle.write(b"altered")
                with patch("safetensors.torch.load_file", side_effect=AssertionError("checkpoint load forbidden")):
                    with self.assertRaises(ValueError):
                        r._reuse_fold_encoder(Mock(), output, req, rows, self.plan, self.registry)

    def test_same_fold_reuse_rejects_false_training_provenance(self):
        rows, _ = synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "image"
            record = sealed_image_job(output, "a" * 64, rows, self.registry, stage="stage_2")
            record["weights_provenance"]["trained_participants"].append(next(row["src_subject_id"] for row in rows if row["cv_fold"] == 0))
            overwrite_record(output, record)
            req = request("a" * 64, model_id="fusion_intermediate_concat", stage="fusion")
            req["fold_image_job_sha256"] = record["record_sha256"]
            with patch("safetensors.torch.load_file", side_effect=AssertionError("checkpoint load forbidden")):
                with self.assertRaisesRegex(ValueError, "held_out_participant_in_weight_provenance"):
                    r._reuse_fold_encoder(Mock(), output, req, rows, self.plan, self.registry)

    def test_same_fold_reuse_requires_predictions_bound_to_actual_validation_rows(self):
        rows, _ = synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "image"
            record = sealed_image_job(output, "a" * 64, rows, self.registry, stage="stage_2")
            predictions = r.read_json(output / "best_validation_predictions.json")
            predictions["identities"][0][0] = "synthetic_wrong_participant"
            (output / "best_validation_predictions.json").unlink()
            r.save_json(output / "best_validation_predictions.json", predictions)
            record["files_sha256"]["best_validation_predictions.json"] = sha_file(output / "best_validation_predictions.json")
            overwrite_record(output, record)
            req = request("a" * 64, model_id="fusion_intermediate_concat", stage="fusion")
            req["fold_image_job_sha256"] = record["record_sha256"]
            with patch("safetensors.torch.load_file", side_effect=AssertionError("checkpoint load forbidden")):
                with self.assertRaisesRegex(ValueError, "same_fold_weight_predictions_must_match_held_out_rows"):
                    r._reuse_fold_encoder(Mock(), output, req, rows, self.plan, self.registry)

    def test_good_same_fold_reuse_loads_only_encoder_tensors_without_fit(self):
        rows, _ = synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "image"
            record = sealed_image_job(output, "a" * 64, rows, self.registry, stage="stage_2")
            req = request("a" * 64, model_id="fusion_intermediate_concat", stage="fusion")
            req["fold_image_job_sha256"] = record["record_sha256"]
            encoder = Mock()
            encoder.state_dict.return_value = {"weight": torch.ones(1, 1)}
            with patch("safetensors.torch.load_file", return_value={"encoder.weight": torch.ones(1, 1), "task_head.weight": torch.zeros(1, 1)}):
                result = r._reuse_fold_encoder(SimpleNamespace(encoder=encoder), output, req, rows, self.plan, self.registry)
            self.assertEqual(result, record["record_sha256"])
            self.assertEqual(set(encoder.load_state_dict.call_args.args[0]), {"weight"})
            self.assertTrue(encoder.load_state_dict.call_args.kwargs["strict"])


if __name__ == "__main__":
    unittest.main()
