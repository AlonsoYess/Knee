"""Synthetic clinical preprocessing tests; no private data or models are read."""
from copy import deepcopy
import hashlib
import json
import unittest

import numpy as np

from knee import clinical_fold_inputs as subject


def synthetic_rows():
    result = []
    for fold in range(5):
        for side in ("RIGHT", "LEFT"):
            result.append({"src_subject_id": f"synthetic_fold_{fold}", "side": side,
                "preprocessing_role": "development" if fold == 0 else "routine",
                "outer_block": 1 + fold % 4, "cv_fold": fold,
                "AGEYEARS": 50 + 10 * fold, "SEX": "F" if fold % 2 else "M",
                "BMI": float(20 + 2 * fold), "PROGRESION_48M": fold % 2,
                "tensor": {"synthetic_only": True}})
    return result


class ClinicalFoldInputTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()

    def fit(self, validation_fold=4, policy=None):
        return subject.fit_clinical_fold(self.rows, validation_fold, policy=policy)

    def reject_fit(self, validation_fold=4, policy=None):
        with self.assertRaises((ValueError, TypeError, KeyError)):
            self.fit(validation_fold, policy)

    @staticmethod
    def rehash(state):
        raw = json.dumps({k: v for k, v in state.items() if k != "record_sha256"},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        state["record_sha256"] = hashlib.sha256(raw).hexdigest()

    def test_expected_population_statistics_and_even_bmi_median(self):
        state = self.fit()
        self.assertEqual(state["BMI_median"], 23.0)
        self.assertEqual(state["continuous_mean"], {"AGEYEARS": 65.0, "BMI": 23.0})
        self.assertAlmostEqual(state["continuous_scale"]["AGEYEARS"], np.sqrt(125.0))
        self.assertAlmostEqual(state["continuous_scale"]["BMI"], np.sqrt(5.0))
        self.assertEqual(state["train_count"], 8)
        self.assertEqual(state["train_participants"], 4)
        self.assertEqual(state["validation_fold"], 4)
        self.assertEqual(state["policy"], subject.POLICY)

    def test_validation_extremes_do_not_affect_fitted_state(self):
        expected = self.fit()
        for row in self.rows:
            if row["cv_fold"] == 4:
                row.update(AGEYEARS=999, BMI=9999.0, SEX="M")
        self.assertEqual(self.fit(), expected)

    def test_labels_do_not_affect_fitted_state_or_transformed_features(self):
        state = self.fit()
        expected = subject.transform_clinical_rows(self.rows, state)
        for row in self.rows:
            row["PROGRESION_48M"] = 1 - row["PROGRESION_48M"]
        self.assertEqual(self.fit(), state)
        np.testing.assert_array_equal(subject.transform_clinical_rows(self.rows, state), expected)

    def test_target_value_is_never_accessed(self):
        class TargetProtected(dict):
            def __getitem__(self, key):
                if key == "PROGRESION_48M":
                    raise AssertionError("target access is forbidden in feature preparation")
                return super().__getitem__(key)
        protected = [TargetProtected(row) for row in self.rows]
        state = subject.fit_clinical_fold(protected, 4)
        result = subject.transform_clinical_rows(protected, state)
        self.assertEqual(result.shape, (10, 3))

    def test_input_order_does_not_change_fit_but_transform_preserves_order(self):
        state = self.fit()
        expected = subject.transform_clinical_rows(self.rows, state)
        self.rows.reverse()
        self.assertEqual(self.fit(), state)
        np.testing.assert_array_equal(subject.transform_clinical_rows(self.rows, state), expected[::-1])

    def test_repetition_and_fit_transform_leave_rows_policy_state_unchanged(self):
        before_rows, before_policy = deepcopy(self.rows), deepcopy(subject.POLICY)
        state = self.fit()
        before_state = deepcopy(state)
        first = subject.transform_clinical_rows(self.rows, state)
        second = subject.transform_clinical_rows(self.rows, state)
        self.assertEqual(self.fit(), state)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(self.rows, before_rows)
        self.assertEqual(subject.POLICY, before_policy)
        self.assertEqual(state, before_state)

    def test_output_is_readonly_float32_contiguous_exactly_three_features(self):
        state = self.fit()
        result = subject.transform_clinical_rows(self.rows, state)
        self.assertEqual(result.shape, (10, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(result.flags.writeable)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[:, 1], [1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
        with self.assertRaises(ValueError):
            result[0, 0] = 0

    def test_missing_bmi_uses_training_median_before_population_scaling(self):
        for row in self.rows:
            if row["cv_fold"] == 1:
                row["BMI"] = None
        state = self.fit()
        self.assertEqual(state["BMI_median"], 24.0)
        self.assertEqual(state["continuous_mean"]["BMI"], 23.5)
        result = subject.transform_clinical_rows(self.rows, state)
        for index in (2, 3):
            self.assertAlmostEqual(float(result[index, 2]),
                (24.0 - 23.5) / state["continuous_scale"]["BMI"], places=6)
        self.assertIsNone(self.rows[2]["BMI"])

    def test_validation_bmi_missing_does_not_fit_another_median(self):
        state = self.fit()
        for row in self.rows:
            if row["cv_fold"] == 4:
                row["BMI"] = None
        self.assertEqual(self.fit(), state)
        result = subject.transform_clinical_rows(self.rows[-2:], state)
        np.testing.assert_array_equal(result[:, 2], [0.0, 0.0])

    def test_training_all_bmi_missing_fails_despite_observed_validation_bmi(self):
        for row in self.rows:
            if row["cv_fold"] != 4:
                row["BMI"] = None
        self.reject_fit()

    def test_constant_continuous_training_values_have_unit_scale(self):
        for row in self.rows:
            row.update(AGEYEARS=60, BMI=25.0)
        state = self.fit()
        self.assertEqual(state["continuous_scale"], {"AGEYEARS": 1.0, "BMI": 1.0})
        result = subject.transform_clinical_rows(self.rows, state)
        np.testing.assert_array_equal(result[:, [0, 2]], np.zeros((10, 2), dtype=np.float32))

    def test_statistics_weight_knee_rows_without_deduplicating_participants(self):
        self.rows.pop(1)
        state = self.fit()
        self.assertEqual(state["train_count"], 7)
        self.assertEqual(state["train_participants"], 4)
        self.assertAlmostEqual(state["continuous_mean"]["AGEYEARS"], 470.0 / 7)

    def test_all_five_fold_indices_are_required_for_fit(self):
        self.rows = [r for r in self.rows if r["cv_fold"] != 2]
        self.reject_fit()

    def test_validation_fold_is_strict_integer_in_range(self):
        for invalid in (True, False, 4.0, "4", None, -1, 5):
            with self.subTest(value=invalid):
                self.reject_fit(invalid)

    def test_duplicate_identity_is_rejected(self):
        self.rows.append(deepcopy(self.rows[0]))
        self.reject_fit()

    def test_participant_cannot_cross_folds_roles_or_outer_blocks(self):
        for key, value in (("cv_fold", 1), ("outer_block", 2), ("preprocessing_role", "routine")):
            with self.subTest(field=key):
                original = self.rows[1][key]
                self.rows[1][key] = value
                self.reject_fit()
                self.rows[1][key] = original

    def test_baseline_values_must_match_bilateral_rows(self):
        for key, value in (("AGEYEARS", 60), ("BMI", None), ("SEX", "F")):
            with self.subTest(field=key):
                original = self.rows[1][key]
                self.rows[1][key] = value
                self.reject_fit()
                self.rows[1][key] = original

    def test_reserved_test_rows_are_forbidden_in_fit_and_transform(self):
        state = self.fit()
        for row in self.rows[:2]:
            row.update(outer_block=0, cv_fold=None, preprocessing_role="routine")
        self.reject_fit()
        with self.assertRaises(ValueError):
            subject.transform_clinical_rows(self.rows[:2], state)

    def test_clinical_domains_and_types_are_strict(self):
        for key, invalid in (("AGEYEARS", True), ("AGEYEARS", 50.0), ("AGEYEARS", 0),
                             ("SEX", "Unknown"), ("BMI", True), ("BMI", "25"),
                             ("BMI", 0), ("BMI", float("nan")), ("BMI", float("inf"))):
            with self.subTest(field=key, value=invalid):
                original = self.rows[0][key]
                self.rows[0][key] = self.rows[1][key] = invalid
                self.reject_fit()
                self.rows[0][key] = self.rows[1][key] = original

    def test_future_or_additional_clinical_field_is_rejected(self):
        self.rows[0]["KL_48"] = 4
        self.reject_fit()

    def test_unknown_or_altered_policy_is_rejected(self):
        for key, value in (("extra_feature", "KL_BASE"), ("feature_order", ["BMI", "SEX", "AGEYEARS"]),
                           ("missingness_indicator", True), ("constant_scale", True)):
            with self.subTest(field=key):
                policy = deepcopy(subject.POLICY)
                policy[key] = value
                self.reject_fit(policy=policy)

    def test_changed_fitted_state_is_rejected(self):
        state = self.fit()
        state["BMI_median"] += 1
        with self.assertRaisesRegex(ValueError, "fitted_clinical_state_changed"):
            subject.transform_clinical_rows(self.rows, state)

    def test_foreign_state_policy_or_extra_feature_is_rejected_even_if_rehashed(self):
        for field, value in (("version", "foreign_state"), ("extra_column", "KL_BASE"),
                             ("policy", {**subject.POLICY, "missingness_indicator": True})):
            with self.subTest(field=field):
                state = self.fit()
                state[field] = value
                self.rehash(state)
                with self.assertRaises(ValueError):
                    subject.transform_clinical_rows(self.rows, state)

    def test_state_scale_and_mean_must_be_finite_positive(self):
        for key, value in (("continuous_scale", 0.0), ("continuous_scale", True),
                           ("continuous_mean", -1.0)):
            with self.subTest(field=key, value=value):
                state = self.fit()
                state[key]["AGEYEARS"] = value
                self.rehash(state)
                with self.assertRaises(ValueError):
                    subject.transform_clinical_rows(self.rows, state)

    def test_state_training_counts_are_strict_and_consistent(self):
        for key, value in (("train_count", True), ("train_participants", 0), ("train_count", 99)):
            with self.subTest(field=key):
                state = self.fit()
                state[key] = value
                self.rehash(state)
                with self.assertRaises(ValueError):
                    subject.transform_clinical_rows(self.rows, state)

    def test_fit_for_each_of_the_five_folds_uses_only_its_training_rows(self):
        for fold in range(5):
            with self.subTest(fold=fold):
                state = self.fit(fold)
                expected = [r["AGEYEARS"] for r in self.rows if r["cv_fold"] != fold]
                self.assertEqual(state["train_count"], len(expected))
                self.assertAlmostEqual(state["continuous_mean"]["AGEYEARS"], float(np.mean(expected)))
                self.assertAlmostEqual(state["continuous_scale"]["AGEYEARS"], float(np.std(expected, ddof=0)))


if __name__ == "__main__":
    unittest.main()
