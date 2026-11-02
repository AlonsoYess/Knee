"""Synthetic sealed predictions only; no model fitting, image or test-package I/O."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from knee import late_fusion as subject
from knee.colab_data import sha_file
from knee.aligned_training.authority import digest


def rows_fixture():
    return [{"src_subject_id": f"synthetic_{fold}_{target}", "side": side,
        "preprocessing_role": "routine", "outer_block": fold % 4 + 1, "cv_fold": fold,
        "AGEYEARS": 60., "SEX": 1, "BMI": 25., "PROGRESION_48M": target}
        for fold in range(5) for target in (0, 1) for side in ("LEFT", "RIGHT")]


def prediction_fixture(rows, fold, *, positive=1., negative=-1.):
    chosen = [row for row in rows if row["cv_fold"] == fold]
    return {"identities": [[row["src_subject_id"], row["side"]] for row in chosen],
        "targets": [row["PROGRESION_48M"] for row in chosen],
        "logits": [positive if row["PROGRESION_48M"] else negative for row in chosen]}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def source_fixture(root):
    """Construct integrity-bound synthetic receipts, with dependency validation mocked.

    This fixture does not claim neural training occurred. Source-job authentication
    itself is covered by training_runner's suite; these tests ensure fusion calls
    it with the intended hashes/recipe and rejects contradictory fold provenance.
    """
    rows = rows_fixture()
    plan = {"version": "registered_training_plan_v1.2",
        "clinical_models": {key: {} for key in ("clinical_logistic", "clinical_xgboost", "clinical_mlp")},
        "radiographic_models": {"mandatory": ["image_densenet121"], "conditional_challengers": []},
        "multimodal_models": {"complementary_enabled": [subject.MODEL_ID]},
        "selection": {"screening_seed": 2026, "near_tie_ap_difference_below": .01}}
    registry = {}
    records, candidates, images = {}, {}, []
    for model in [*plan["clinical_models"], "image_densenet121"]:
        refs = []
        for fold in range(5):
            directory = root / f"{model}_{fold}"
            directory.mkdir()
            predictions = prediction_fixture(rows, fold, positive=3. if model.startswith("image") else 1.)
            # Each input may have its own row order. Pairing is by participant+side.
            if model.startswith("image"):
                predictions = {key: list(reversed(value)) for key, value in predictions.items()}
            write_json(directory / "best_validation_predictions.json", predictions)
            training = [row for row in rows if row["cv_fold"] != fold]
            identities = sorted([[row["src_subject_id"], row["side"]] for row in training])
            request = {"model_id": model, "stage": "stage_1" if model in plan["clinical_models"] else "stage_2",
                "validation_fold": fold, "seed": 2026, "clinical_scenario": "clinical_main",
                "baseline_sidecar_sha256": None, "development_package_sha256": "a" * 64}
            record = {"request": request, "weights_provenance": {"fit_scope": "development_training_fold",
                "validation_fold": fold, "train_identity_sha256": digest(identities),
                "trained_participants": sorted({row[0] for row in identities}), "reserved_test_accessed": False},
                "metrics": subject.probability_metrics(predictions["targets"],
                    [subject._probability(z) for z in predictions["logits"]]),
                "files_sha256": {"best_validation_predictions.json": sha_file(directory / "best_validation_predictions.json")}}
            record["record_sha256"] = digest(record)
            write_json(directory / "training_job.json", record)
            records[str(directory)] = record
            refs.append({"directory": str(directory), "record_sha256": record["record_sha256"]})
        if model.startswith("image"):
            images = refs
        else:
            candidates[model] = refs
    selection_path = root / "synthetic_selection.json"
    write_json(selection_path, {"screening_jobs": [], "stage_two_jobs": images})
    arguments = {"package_sha256": "a" * 64, "plan": plan, "registry": registry,
        "selected_image_encoder_id": "image_densenet121", "image_selection_path": selection_path,
        "image_selection_sha256": sha_file(selection_path)}
    return rows, candidates, images, records, arguments


def sealed_fixture(root):
    """Exercise real integrity/request/selection validators with synthetic receipts.

    No source model is fitted here. Artificial checkpoint bytes stand in for
    already sealed artifacts; integrity checks consume their hashes, not models.
    """
    from knee import methodology_release as release
    from knee.aligned_training import runner
    from knee.training_augmentation import VERSION as augmentation_version
    from knee.aligned_training.engine import binary_metrics
    repository = Path(__file__).resolve().parents[1]
    _, _, plan, registry = release.load_release(repository)
    rows = rows_fixture()
    package_sha = "a" * 64
    def create(model, fold, stage):
        directory = root / f"{model}_{stage}_{fold}"
        directory.mkdir()
        clinical = model in plan["clinical_models"]
        request = {"version": release.REQUEST_VERSION, "operation": "development_fold", "model_id": model,
            "image_encoder_id": None if clinical else model, "stage": stage, "validation_fold": fold,
            "seed": 2026, "development_package_sha256": package_sha, "image_micro_batch_size": 4,
            "selection_evidence_sha256": None if stage == "stage_1" else "b" * 64,
            "fold_image_job_sha256": None, "clinical_scenario": "clinical_main", "baseline_sidecar_sha256": None,
            "augmentation_policy": "identity" if clinical else augmentation_version,
            "modeling_release_sha256": release.PINS[release.RELEASE_PATH]}
        confidence = 3. if model == "image_densenet121" else 1.
        predictions = prediction_fixture(rows, fold, positive=confidence, negative=-confidence)
        write_json(directory / "best_validation_predictions.json", predictions)
        write_json(directory / "clinical_state.json", {"synthetic_fixture_no_predictor_fitting": True})
        files = ["best_validation_predictions.json", "clinical_state.json"]
        if model == "clinical_logistic":
            files.append("clinical_model.json")
        elif model == "clinical_xgboost":
            files.append("clinical_model.ubj")
        else:
            files.extend(["best_model.safetensors", "last_model.safetensors", "training_record.json"])
        for name in files[2:]:
            (directory / name).write_bytes(b"SYNTHETIC_ARTIFACT_INTEGRITY_FIXTURE_NO_TRAINING")
        training = sorted([[r["src_subject_id"], r["side"]] for r in rows if r["cv_fold"] != fold])
        record = {"version": runner.VERSION, "scope": "development_only", "eligible_for_model_selection": True,
            "training_executed": True, "engine_status": "training_fold_complete", "reserved_test_accessed": False,
            "request": request, "request_sha256": digest(request), "plan_sha256": release.recipe_sha256(plan),
            "files_sha256": {name: sha_file(directory / name) for name in files},
            "metrics": binary_metrics(predictions["targets"], predictions["logits"]),
            "eligibility_gates": {key: not clinical for key in registry["eligibility_gates"]},
            "weights_provenance": {"fit_scope": "development_training_fold", "validation_fold": fold,
                "train_identity_sha256": digest(training), "trained_participants": sorted({r[0] for r in training}),
                "reserved_test_accessed": False}}
        record["record_sha256"] = digest(record)
        write_json(directory / runner.JOB_FILENAME, record)
        return {"directory": str(directory), "record_sha256": record["record_sha256"]}
    clinical = {model: [create(model, fold, "stage_1") for fold in range(5)] for model in plan["clinical_models"]}
    screening = [create(model, fold, "stage_1") for model in plan["radiographic_models"]["mandatory"] for fold in range(5)]
    tuned = [create(model, fold, "stage_2") for model in plan["radiographic_models"]["mandatory"] for fold in range(5)]
    image_jobs = [reference for reference in tuned if "image_densenet121" in reference["directory"]]
    selection_path = root / "reviewed_image_selection.json"
    selection = {"version": "development_image_selection_evidence_v1.0", "package_sha256": package_sha,
        "screening_jobs": screening, "excluded_modern_models": {model: "Artificial fixture: not trained or evaluated."
            for model in plan["radiographic_models"]["conditional_challengers"]},
        "stage_two_jobs": tuned, "selected_image_encoder_id": "image_densenet121"}
    write_json(selection_path, selection)
    arguments = {"package_sha256": package_sha, "plan": plan, "registry": registry,
        "selected_image_encoder_id": "image_densenet121", "image_selection_path": selection_path,
        "image_selection_sha256": sha_file(selection_path), "output_dir": root / "fusion"}
    return rows, clinical, image_jobs, arguments


class NumericalFusionTests(unittest.TestCase):
    def test_fixed_mean_probabilities_not_mean_logits(self):
        rows = rows_fixture()
        clinical = prediction_fixture(rows, 0, positive=0., negative=-2.)
        image = prediction_fixture(rows, 0, positive=4., negative=-2.)
        fused = subject.fuse_held_out_predictions(clinical, image, rows, 0)
        positive = next(row for row in fused if row["target"])
        self.assertAlmostEqual(positive["probability"], (.5 + 1 / (1 + math.exp(-4))) / 2)
        self.assertNotAlmostEqual(positive["probability"], 1 / (1 + math.exp(-2)))

    def test_independent_row_order_is_keyed(self):
        rows = rows_fixture()
        clinical = prediction_fixture(rows, 2)
        image = prediction_fixture(rows, 2, positive=3.)
        expected = subject.fuse_held_out_predictions(clinical, image, rows, 2)
        reversed_image = {key: list(reversed(value)) for key, value in image.items()}
        self.assertEqual(expected, subject.fuse_held_out_predictions(clinical, reversed_image, rows, 2))

    def test_exact_endpoint_probabilities_are_preserved(self):
        self.assertEqual(subject.fixed_mean_probabilities([0, 1], [0, 1]).tolist(), [0., 1.])
        self.assertEqual(subject.probability_metrics([0, 1], [0., 1.]),
            {"average_precision_score": 1., "roc_auc": 1., "brier": 0.})

    def test_probability_inputs_are_not_mutated(self):
        clinical = np.asarray([.1, .9])
        image = np.asarray([.2, .8])
        clinical.flags.writeable = image.flags.writeable = False
        output = subject.fixed_mean_probabilities(clinical, image)
        self.assertFalse(np.shares_memory(output, clinical))
        np.testing.assert_array_equal(clinical, [.1, .9])

    def test_invalid_probabilities_fail_closed(self):
        for values in ([float("nan"), .1], [float("inf"), .1], [-.1, 1], [.1, 1.1],
                       [True, False], ["0.1", "0.2"], [[.1, .2]], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                subject.fixed_mean_probabilities(values, values)
        with self.assertRaises(ValueError):
            subject.fixed_mean_probabilities([.1], [.1, .2])

    def test_duplicate_or_missing_prediction_rejected(self):
        rows, fold = rows_fixture(), 0
        original = prediction_fixture(rows, fold)
        bad = deepcopy(original)
        for key in bad:
            bad[key][1] = bad[key][0]
        with self.assertRaisesRegex(ValueError, "unique_same_fold"):
            subject.fuse_held_out_predictions(bad, original, rows, fold)
        bad = {key: value[:-1] for key, value in original.items()}
        with self.assertRaisesRegex(ValueError, "complete_held_out"):
            subject.fuse_held_out_predictions(bad, original, rows, fold)

    def test_training_fold_predictions_rejected(self):
        rows = rows_fixture()
        with self.assertRaisesRegex(ValueError, "unique_same_fold"):
            subject.fuse_held_out_predictions(prediction_fixture(rows, 1), prediction_fixture(rows, 0), rows, 0)

    def test_mismatched_targets_rejected(self):
        rows = rows_fixture()
        original = prediction_fixture(rows, 0)
        bad = deepcopy(original)
        bad["targets"][0] = 1 - bad["targets"][0]
        with self.assertRaisesRegex(ValueError, "target_mismatch"):
            subject.fuse_held_out_predictions(bad, original, rows, 0)

    def test_reserved_assignment_rejected(self):
        rows = rows_fixture()
        original = prediction_fixture(rows, 0)
        rows[0]["outer_block"], rows[0]["cv_fold"] = 0, None
        with self.assertRaisesRegex(ValueError, "development_only"):
            subject.fuse_held_out_predictions(original, original, rows, 0)

    def test_participant_cross_fold_rejected(self):
        rows = rows_fixture()
        original = prediction_fixture(rows, 0)
        rows[1]["cv_fold"] = 1
        with self.assertRaisesRegex(ValueError, "participant_crosses"):
            subject.fuse_held_out_predictions(original, original, rows, 0)


class SourceIntegrationTests(unittest.TestCase):
    def invoke(self, root, mutation=None, *, output=True):
        rows, clinical, images, records, arguments = source_fixture(root)
        if mutation is not None:
            mutation(rows, clinical, images, records, arguments)
        def completed(directory, package, plan, registry):
            self.assertEqual(package, "a" * 64)
            return deepcopy(records[str(directory)])
        with patch("knee.methodology_release.validate_plan") as validate_plan, \
                patch("knee.aligned_training.runner.completed_job", side_effect=completed) as auth, \
                patch("knee.aligned_training.runner.validate_selection") as selection:
            result = subject.combine_development_jobs(clinical, images, rows,
                **arguments, output_dir=root / "fusion" if output else None)
            self.assertEqual(auth.call_count, 20)
            validate_plan.assert_called_once()
            selection.assert_called_once()
            self.assertEqual(selection.call_args.args[1]["selection_evidence_sha256"], arguments["image_selection_sha256"])
            return result

    def test_complete_workflow_uses_all_receipts_and_saves_readback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.invoke(root)
            self.assertEqual(result["development_knees"], 20)
            self.assertEqual(result["clinical_model_id"], "clinical_logistic")
            self.assertEqual(result["weights"], [.5, .5])
            self.assertEqual(len(result["prediction_rows"]), 20)
            self.assertFalse(result["training_executed"])
            self.assertFalse(result["unbiased_final_performance_claimed"])
            self.assertEqual(json.loads((root / "fusion" / subject.FILENAME).read_text()), result)

    def test_untrusted_job_receipt_rejected(self):
        def mutate(rows, clinical, images, records, arguments):
            images[0]["record_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "same_release_main"):
            self.invoke(Path(temporary), mutate)

    def test_wrong_selection_hash_rejected_before_jobs(self):
        def mutate(rows, clinical, images, records, arguments):
            arguments["image_selection_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "trusted_image_selection"):
            self.invoke(Path(temporary), mutate)

    def test_complementary_clinical_branch_rejected(self):
        def mutate(rows, clinical, images, records, arguments):
            records[clinical["clinical_logistic"][0]["directory"]]["request"]["clinical_scenario"] = "clinical_plus_baseline_kl"
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "same_release_main"):
            self.invoke(Path(temporary), mutate)

    def test_held_out_person_in_fit_provenance_rejected(self):
        def mutate(rows, clinical, images, records, arguments):
            records[images[0]["directory"]]["weights_provenance"]["trained_participants"].append("synthetic_0_0")
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "held_out_participant"):
            self.invoke(Path(temporary), mutate)

    def test_fusion_source_must_be_same_run_used_for_image_selection(self):
        def mutate(rows, clinical, images, records, arguments):
            write_json(arguments["image_selection_path"], {"screening_jobs": [], "stage_two_jobs": images[1:]})
            arguments["image_selection_sha256"] = sha_file(arguments["image_selection_path"])
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "source_run_differs"):
            self.invoke(Path(temporary), mutate)

    def test_artifact_tampering_detected_before_output(self):
        def mutate(rows, clinical, images, records, arguments):
            path = Path(images[0]["directory"]) / "training_job.json"
            # Simulate a change concurrent with selection verification, after its
            # initial digest is recorded, without changing semantic source rows.
            original = subject.fuse_held_out_predictions
            def tamper(*args, **kwargs):
                path.write_bytes(path.read_bytes() + b" ")
                return original(*args, **kwargs)
            self.tamper = patch.object(subject, "fuse_held_out_predictions", side_effect=tamper)
            self.tamper.start()
        try:
            with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "source_changed"):
                self.invoke(Path(temporary), mutate)
        finally:
            if hasattr(self, "tamper"):
                self.tamper.stop()


class RealArtifactValidatorTests(unittest.TestCase):
    def test_reconciled_release_and_full_image_selection_without_mocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, clinical, images, arguments = sealed_fixture(Path(temporary))
            result = subject.combine_development_jobs(clinical, images, rows, **arguments)
            self.assertEqual(result["clinical_model_id"], "clinical_logistic")
            self.assertEqual(result["image_encoder_id"], "image_densenet121")
            self.assertEqual(result["development_knees"], len(rows))
            self.assertEqual(result["record_sha256"], digest({key: value for key, value in result.items() if key != "record_sha256"}))
            self.assertFalse(result["optimizer_created"])

    def test_historical_request_cannot_mix_with_reconciled_campaign(self):
        from knee import final_model_release as old
        from knee.training_authority import REQUEST_VERSION
        with tempfile.TemporaryDirectory() as temporary:
            rows, clinical, images, arguments = sealed_fixture(Path(temporary))
            reference = images[0]
            path = Path(reference["directory"]) / "training_job.json"
            record = json.loads(path.read_text())
            for field in ("clinical_scenario", "baseline_sidecar_sha256", "augmentation_policy", "modeling_release_sha256"):
                record["request"].pop(field)
            record["request"]["version"] = REQUEST_VERSION
            record["request_sha256"] = digest(record["request"])
            record["plan_sha256"] = old.PINS["configs/training_plan_v1_1.json"]
            record["record_sha256"] = digest({key: value for key, value in record.items() if key != "record_sha256"})
            reference["record_sha256"] = record["record_sha256"]
            write_json(path, record)
            with self.assertRaisesRegex(ValueError, "exact_training_request"):
                subject.combine_development_jobs(clinical, images, rows, **arguments)
            self.assertFalse(arguments["output_dir"].exists())

    def test_modified_prediction_file_fails_before_fusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, clinical, images, arguments = sealed_fixture(Path(temporary))
            path = Path(images[0]["directory"]) / "best_validation_predictions.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "completed_training_artifact_changed"):
                subject.combine_development_jobs(clinical, images, rows, **arguments)
            self.assertFalse(arguments["output_dir"].exists())

    def test_unselected_image_candidate_must_share_training_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, clinical, images, arguments = sealed_fixture(Path(temporary))
            selection_path = arguments["image_selection_path"]
            selection = json.loads(selection_path.read_text())
            reference = next(entry for entry in selection["stage_two_jobs"] if "image_vit_b16" in entry["directory"])
            path = Path(reference["directory"]) / "training_job.json"
            record = json.loads(path.read_text())
            record["weights_provenance"]["trained_participants"].append("synthetic_0_0")
            record["record_sha256"] = digest({key: value for key, value in record.items() if key != "record_sha256"})
            reference["record_sha256"] = record["record_sha256"]
            write_json(path, record)
            write_json(selection_path, selection)
            arguments["image_selection_sha256"] = sha_file(selection_path)
            with self.assertRaisesRegex(ValueError, "held_out_participant"):
                subject.combine_development_jobs(clinical, images, rows, **arguments)


if __name__ == "__main__":
    unittest.main()
