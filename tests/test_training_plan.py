"""Synthetic scope, fold-leakage, and ranking checks; no model is fitted."""
from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path
import tempfile
import unittest

from knee import training_plan as subject


ROOT = Path(__file__).resolve().parents[1]


def synthetic_rows():
    rows = []
    for fold in range(5):
        for side in ("RIGHT", "LEFT"):
            rows.append({"src_subject_id": f"synthetic_{fold}", "side": side,
                         "preprocessing_role": "development" if fold == 0 else "routine",
                         "outer_block": 1 + fold % 4, "cv_fold": fold,
                         "AGEYEARS": 50 + fold, "SEX": "F", "BMI": 25.0,
                         "PROGRESION_48M": int(fold == 1)})
    return rows


class TrainingPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.registry = subject.load_training_plan(ROOT / "configs/training_plan_v1.json",
                                                              ROOT / "configs/model_registry.json")
        self.rows = synthetic_rows()

    def report(self, model_id, ap=0.4, auc=0.6, brier=0.2):
        return {"model_id": model_id, "scope": "development_only",
                "aggregation": "mean_of_five_participant_held_out_fold_metrics",
                "eligibility_gates": dict.fromkeys(self.registry["eligibility_gates"], True),
                "fold_metrics": [{"validation_fold": fold, "average_precision_score": ap,
                                  "roc_auc": auc, "brier": brier} for fold in range(5)]}

    def provenance(self, fold=4):
        weight = subject.training_fold_class_weight(self.rows, fold)
        return {"fit_scope": "development_training_fold", "validation_fold": fold,
                "train_identity_sha256": weight["train_identity_sha256"],
                "trained_participants": sorted({r["src_subject_id"] for r in self.rows if r["cv_fold"] != fold}),
                "reserved_test_accessed": False}

    def test_registered_recipe_is_complete_and_not_authorized_to_train(self):
        result = subject.validate_plan(self.plan, self.registry)
        self.assertEqual(result["registered_candidates"], 12)
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["reserved_test_access_authorized"])
        self.assertEqual(self.plan["neural_training"]["maximum_epochs"], 60)
        self.assertEqual(self.plan["neural_training"]["early_stopping"]["patience_epochs"], 8)
        self.assertEqual(self.plan["selection"]["neural_finalist_seeds"], [2026, 2027, 2028])
        self.assertEqual(self.plan["selection"]["near_tie_ap_difference_below"], .01)
        self.assertIn("unweighted_arithmetic_mean_of_five", self.plan["selection"]["aggregation"])
        self.assertEqual(self.plan["future_calibration_and_threshold"]["neural_calibrator"],
                         "temperature_scaling_on_logits")
        self.assertEqual(self.plan["future_calibration_and_threshold"]["conventional_clinical_calibrator"],
                         "logistic_calibration")
        self.assertEqual(self.plan["future_calibration_and_threshold"]["threshold_tie_break"],
                         "prefer_sensitivity")
        self.assertFalse(self.plan["future_final_refit"]["execution_authorized"])

    def test_authorization_test_access_or_new_features_cannot_be_overridden(self):
        candidates = []
        for key in ("training_authorized", "reserved_test_access_authorized"):
            altered = deepcopy(self.plan)
            altered[key] = True
            candidates.append(altered)
        altered = deepcopy(self.plan)
        altered["inputs"]["clinical_features"].append("KL_V06")
        candidates.append(altered)
        altered = deepcopy(self.plan)
        altered["future_calibration_and_threshold"]["fit_source"] = "reserved_test"
        candidates.append(altered)
        for altered in candidates:
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                subject.validate_plan(altered, self.registry)

    def test_learning_rate_epoch_batch_and_augmentation_changes_require_new_review(self):
        changes = [("maximum_epochs", 600), ("image_effective_batch_size", 8),
                   ("augmentation", {"initial_policy": "horizontal_flip"})]
        for key, value in changes:
            altered = deepcopy(self.plan)
            altered["neural_training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_plan(altered, self.registry)
        altered = deepcopy(self.plan)
        altered["clinical_models"]["clinical_mlp"]["learning_rate"] = 0.002
        with self.assertRaises(ValueError):
            subject.validate_plan(altered, self.registry)

    def test_registry_drift_unknown_architecture_and_extra_override_are_rejected(self):
        registry = deepcopy(self.registry)
        registry["radiographic"]["mandatory_baselines"][0]["family"] = "unregistered_architecture"
        with self.assertRaises(ValueError):
            subject.validate_plan(self.plan, registry)
        self.plan["overrides"] = {"seed": 999}
        with self.assertRaises(ValueError):
            subject.validate_plan(self.plan, self.registry)

    def test_nan_infinity_bool_and_missing_keys_are_rejected(self):
        for value in (float("nan"), float("inf"), True, None):
            altered = deepcopy(self.plan)
            altered["clinical_models"]["clinical_mlp"]["learning_rate"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.validate_plan(altered, self.registry)
        del self.plan["readiness"]
        with self.assertRaises(ValueError):
            subject.validate_plan(self.plan, self.registry)

    def test_json_duplicate_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"training_authorized": false, "training_authorized": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                subject.load_training_plan(path, ROOT / "configs/model_registry.json")

    def test_json_whitespace_does_not_change_frozen_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formatted.json"
            path.write_text(json.dumps(self.plan, indent=4, sort_keys=True), encoding="utf-8")
            loaded, _ = subject.load_training_plan(path, ROOT / "configs/model_registry.json")
            self.assertEqual(loaded, self.plan)

    def test_training_class_weight_uses_only_training_knees(self):
        result = subject.training_fold_class_weight(self.rows, 4)
        self.assertEqual(result["training_knees"], 8)
        self.assertEqual(result["positive_count"], 2)
        self.assertEqual(result["negative_count"], 6)
        self.assertEqual(result["positive_weight"], 3.0)
        for row in self.rows:
            if row["cv_fold"] == 4:
                row["PROGRESION_48M"] = 1
        self.assertEqual(subject.training_fold_class_weight(self.rows, 4), result)

    def test_validation_targets_are_never_accessed_for_class_weights(self):
        class Protected(dict):
            def __getitem__(self, key):
                if key == "PROGRESION_48M":
                    raise AssertionError("validation target was accessed")
                return super().__getitem__(key)
        rows = [Protected(row) if row["cv_fold"] == 4 else row for row in self.rows]
        self.assertEqual(subject.training_fold_class_weight(rows, 4)["positive_weight"], 3.0)

    def test_single_class_training_fold_and_invalid_labels_fail_closed(self):
        for row in self.rows:
            row["PROGRESION_48M"] = 0
        with self.assertRaisesRegex(ValueError, "both_training"):
            subject.training_fold_class_weight(self.rows, 4)
        for value in (True, 0.5, float("nan"), "1"):
            self.rows[0]["PROGRESION_48M"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.training_fold_class_weight(self.rows, 4)

    def test_held_out_fold_is_not_reassigned_and_reserved_rows_are_rejected(self):
        for key, value in (("outer_block", 0), ("cv_fold", None), ("cv_fold", True)):
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                subject.training_fold_class_weight(rows, 4)

    def test_participant_crossing_folds_is_rejected(self):
        self.rows[0]["cv_fold"] = 2
        with self.assertRaisesRegex(ValueError, "participant_crosses"):
            subject.training_fold_class_weight(self.rows, 4)

    def test_duplicate_knees_missing_fold_and_extra_predictors_are_rejected(self):
        cases = [self.rows + [deepcopy(self.rows[0])], [r for r in self.rows if r["cv_fold"] != 3]]
        rows = deepcopy(self.rows)
        rows[0]["WOMAC"] = 10
        cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                subject.training_fold_class_weight(rows, 4)

    def test_fold_hash_is_order_invariant_and_inputs_are_unchanged(self):
        before = deepcopy(self.rows)
        expected = subject.training_fold_class_weight(self.rows, 4)
        self.assertEqual(subject.training_fold_class_weight(list(reversed(self.rows)), 4), expected)
        self.assertEqual(self.rows, before)

    def test_fusion_reuses_only_exact_same_training_fold_weights(self):
        self.assertTrue(subject.validate_fold_weight_provenance(self.rows, 4, self.provenance()))
        with self.assertRaises(ValueError):
            subject.validate_fold_weight_provenance(self.rows, 3, self.provenance())

    def test_all_development_weights_cannot_be_reused_for_a_validation_fold(self):
        provenance = self.provenance()
        provenance["trained_participants"].append("synthetic_4")
        with self.assertRaisesRegex(ValueError, "held_out_participant"):
            subject.validate_fold_weight_provenance(self.rows, 4, provenance)

    def test_wrong_hash_missing_participant_and_test_touched_are_rejected(self):
        for key, value in (("train_identity_sha256", "0" * 64), ("trained_participants", ["synthetic_0"]),
                           ("reserved_test_accessed", True), ("validation_fold", True)):
            provenance = self.provenance()
            provenance[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_fold_weight_provenance(self.rows, 4, provenance)

    def test_at_most_two_challengers_use_mean_ap_then_auc_then_brier(self):
        ids = self.plan["radiographic_models"]["conditional_challengers"]
        reports = [self.report(ids[0], ap=.7, auc=.8, brier=.3),
                   self.report(ids[1], ap=.7, auc=.8, brier=.2),
                   self.report(ids[2], ap=.6, auc=.99, brier=.01)]
        self.assertEqual(subject.select_modern_challengers(reports, self.plan, self.registry), [ids[1], ids[0]])
        for row in reports[0]["fold_metrics"]:
            row["roc_auc"] = .9
        self.assertEqual(subject.select_modern_challengers(reports, self.plan, self.registry), [ids[0], ids[1]])

    def test_mean_uses_all_five_fold_scores_not_best_fold_or_pooled_score(self):
        ids = self.plan["radiographic_models"]["conditional_challengers"]
        first, second = self.report(ids[0]), self.report(ids[1], ap=.45)
        for row, score in zip(first["fold_metrics"], [.95, .25, .25, .25, .25]):
            row["average_precision_score"] = score
        self.assertEqual(subject.select_modern_challengers([first, second], self.plan, self.registry),
                         [ids[1], ids[0]])
        first["pooled_average_precision_score"] = .99
        with self.assertRaisesRegex(ValueError, "exact_oof_report_fields"):
            subject.select_modern_challengers([first], self.plan, self.registry)

    def test_near_tie_strict_boundary_and_no_rounding_expansion(self):
        ids = self.plan["radiographic_models"]["conditional_challengers"]
        best = self.report(ids[0], ap=.7, auc=.6)
        for score, expected in ((.69, [ids[0], ids[1]]), (.6900001, [ids[1], ids[0]])):
            other = self.report(ids[1], ap=score, auc=.9)
            with self.subTest(score=score):
                self.assertEqual(subject.select_modern_challengers([other, best], self.plan, self.registry),
                                 expected)

    def test_near_tie_band_anchors_to_highest_remaining_ap_without_transitive_chains(self):
        ids = self.plan["radiographic_models"]["conditional_challengers"]
        reports = [self.report(ids[0], ap=.7, auc=.6), self.report(ids[1], ap=.695, auc=.7),
                   self.report(ids[2], ap=.688, auc=.99)]
        for ordering in permutations(reports):
            self.assertEqual(subject.select_modern_challengers(list(ordering), self.plan, self.registry),
                             [ids[1], ids[0]])

    def test_exact_ties_use_registry_order_independent_of_input_order(self):
        ids = self.plan["radiographic_models"]["conditional_challengers"]
        reports = [self.report(model_id) for model_id in reversed(ids)]
        self.assertEqual(subject.select_modern_challengers(reports, self.plan, self.registry), ids[:2])

    def test_unmet_or_unknown_eligibility_cannot_promote_candidate(self):
        candidate = self.report("image_skelex")
        gate = self.registry["eligibility_gates"][0]
        candidate["eligibility_gates"][gate] = False
        self.assertEqual(subject.select_modern_challengers([candidate], self.plan, self.registry), [])
        candidate["eligibility_gates"][gate] = "unknown"
        with self.assertRaises(ValueError):
            subject.select_modern_challengers([candidate], self.plan, self.registry)

    def test_pooled_scores_test_scores_and_incomplete_folds_are_rejected(self):
        for key, value in (("aggregation", "pooled_participant_held_out_oof"), ("scope", "reserved_test"),
                           ("fold_metrics", [])):
            candidate = self.report("image_convnextv2_tiny")
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.select_modern_challengers([candidate], self.plan, self.registry)

    def test_invalid_fold_metrics_missing_or_duplicate_fold_are_rejected(self):
        for key, value in (("average_precision_score", float("nan")), ("roc_auc", True),
                           ("brier", -0.1), ("validation_fold", 1), ("validation_fold", True)):
            candidate = self.report("image_convnextv2_tiny")
            candidate["fold_metrics"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                subject.select_modern_challengers([candidate], self.plan, self.registry)

    def test_unregistered_duplicate_and_baseline_promotion_reports_rejected(self):
        for model_id in ("image_other", "image_densenet121"):
            with self.subTest(model_id=model_id), self.assertRaises(ValueError):
                subject.select_modern_challengers([self.report(model_id)], self.plan, self.registry)
        report = self.report("image_skelex")
        with self.assertRaises(ValueError):
            subject.select_modern_challengers([report, deepcopy(report)], self.plan, self.registry)

    def test_final_duration_uses_median_of_exact_five_development_best_epochs(self):
        records = [{"scope": "development_only", "validation_fold": fold, "best_epoch": epoch}
                   for fold, epoch in enumerate([9, 21, 16, 60, 1])]
        before = deepcopy(records)
        self.assertEqual(subject.final_training_epochs(records, self.plan, self.registry), 16)
        self.assertEqual(subject.final_training_epochs(list(reversed(records)), self.plan, self.registry), 16)
        self.assertEqual(records, before)

    def test_final_duration_rejects_test_missing_duplicate_or_out_of_budget_epochs(self):
        records = [{"scope": "development_only", "validation_fold": fold, "best_epoch": 20}
                   for fold in range(5)]
        for key, value in (("scope", "reserved_test"), ("validation_fold", 1),
                           ("best_epoch", 0), ("best_epoch", 61), ("best_epoch", 2.5),
                           ("best_epoch", True)):
            altered = deepcopy(records)
            altered[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                subject.final_training_epochs(altered, self.plan, self.registry)
        with self.assertRaises(ValueError):
            subject.final_training_epochs(records[:4], self.plan, self.registry)


if __name__ == "__main__":
    unittest.main()
