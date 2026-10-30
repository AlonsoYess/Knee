"""Synthetic-only tests of baseline complements; no estimator or cohort access."""
from copy import deepcopy
import unittest

import numpy as np

from knee import clinical_fold_inputs as historical
from knee import clinical_scenarios as subject
from test_clinical_fold_inputs import synthetic_rows


def enriched_rows():
    rows = synthetic_rows()
    for i, row in enumerate(rows):
        row.update(KL_BASE=2 + i % 2, CIRUGIA_PREVIA=i % 2, WOMAC_TOTAL=float(i * 8))
    return rows


def projection(rows):
    return [{name: row[name] for name in subject.BASELINE_FIELDS} for row in rows]


class ClinicalScenarioTests(unittest.TestCase):
    def setUp(self):
        self.rows = enriched_rows()
        self.base = synthetic_rows()
        self.package = "a" * 64
        self.source = "b" * 64

    def fit(self, scenario="clinical_plus_baseline_surgery_and_womac", fold=4):
        return subject.fit_scenario_fold(self.rows, fold, scenario)

    def sidecar(self):
        return subject.make_baseline_sidecar(projection(self.rows), self.base, self.package, self.source)

    @staticmethod
    def reseal(record):
        record["record_sha256"] = historical._digest({k: v for k, v in record.items() if k != "record_sha256"})

    def test_main_scenario_is_bitwise_historical_with_or_without_complements(self):
        for rows in (self.rows, self.base):
            state = subject.fit_scenario_fold(rows, 4, "clinical_main")
            expected_state = historical.fit_clinical_fold(self.base, 4)
            self.assertEqual(state["main_state"], expected_state)
            np.testing.assert_array_equal(subject.transform_scenario_rows(rows, state),
                                          historical.transform_clinical_rows(self.base, expected_state))

    def test_all_three_dimensions_and_feature_order(self):
        for scenario, expected in subject.SCENARIOS.items():
            state = self.fit(scenario)
            matrix = subject.transform_scenario_rows(self.rows, state)
            self.assertEqual(matrix.shape, (10, len(expected)))
            self.assertEqual(state["feature_order"], list(expected))
            self.assertEqual(matrix.dtype, np.float32)
            self.assertTrue(matrix.flags.c_contiguous)
            self.assertFalse(matrix.flags.writeable)
            self.assertTrue(np.isfinite(matrix).all())

    def test_kl_standardizes_ordinal_only_no_imputation(self):
        state = self.fit("clinical_plus_baseline_kl")
        self.assertEqual(state["extra_mean"], {"KL_BASE": 2.5})
        self.assertEqual(state["extra_scale"], {"KL_BASE": 0.5})
        self.assertEqual(state["extra_imputation"], {})
        matrix = subject.transform_scenario_rows(self.rows, state)
        np.testing.assert_array_equal(matrix[:, 3], [-1, 1] * 5)

    def test_constant_kl_and_zero_womac_use_unit_scale(self):
        for row in self.rows:
            row.update(KL_BASE=2, WOMAC_TOTAL=0.0)
        for scenario, feature in (("clinical_plus_baseline_kl", "KL_BASE"),
                                  ("clinical_plus_baseline_surgery_and_womac", "WOMAC_TOTAL")):
            state = self.fit(scenario)
            self.assertEqual(state["extra_scale"][feature], 1.0)
            np.testing.assert_array_equal(subject.transform_scenario_rows(self.rows, state)[:, -1], np.zeros(10))

    def test_surgery_mode_ties_choose_zero_and_binary_not_scaled(self):
        state = self.fit()
        self.assertEqual(state["extra_imputation"]["CIRUGIA_PREVIA"], 0)
        np.testing.assert_array_equal(subject.transform_scenario_rows(self.rows, state)[:, 3], [0, 1] * 5)
        self.rows[0]["CIRUGIA_PREVIA"] = 1
        self.assertEqual(self.fit()["extra_imputation"]["CIRUGIA_PREVIA"], 1)

    def test_missing_womac_uses_only_observed_training_median_then_scaling(self):
        self.rows[0]["WOMAC_TOTAL"] = None
        self.rows[9]["WOMAC_TOTAL"] = None
        state = self.fit()
        self.assertEqual(state["extra_imputation"]["WOMAC_TOTAL"], 32.0)
        self.assertAlmostEqual(state["extra_mean"]["WOMAC_TOTAL"], 32.0)
        matrix = subject.transform_scenario_rows(self.rows, state)
        self.assertEqual(float(matrix[0, 4]), 0.0)
        self.assertEqual(float(matrix[9, 4]), 0.0)

    def test_validation_extremes_do_not_change_imputation_or_scaling(self):
        expected = {scenario: self.fit(scenario) for scenario in subject.SCENARIOS}
        for row in self.rows[-2:]:
            row.update(AGEYEARS=999, BMI=9999.0, KL_BASE=3, CIRUGIA_PREVIA=None, WOMAC_TOTAL=96.0)
        for scenario in subject.SCENARIOS:
            self.assertEqual(self.fit(scenario), expected[scenario])

    def test_all_missing_training_complement_fails_despite_observed_validation(self):
        for feature, error in (("CIRUGIA_PREVIA", "observed_surgery"), ("WOMAC_TOTAL", "observed_womac")):
            with self.subTest(feature=feature):
                rows = enriched_rows()
                for row in rows[:-2]:
                    row[feature] = None
                with self.assertRaisesRegex(ValueError, error):
                    subject.fit_scenario_fold(rows, 4, "clinical_plus_baseline_surgery_and_womac")

    def test_target_values_never_accessed_by_preprocessing(self):
        class Protected(dict):
            def __getitem__(self, name):
                if name == "PROGRESION_48M":
                    raise AssertionError("No target access allowed")
                return super().__getitem__(name)
        for scenario in subject.SCENARIOS:
            rows = [Protected(row) for row in self.rows]
            state = subject.fit_scenario_fold(rows, 4, scenario)
            self.assertEqual(subject.transform_scenario_rows(rows, state).shape[0], 10)

    def test_row_permutation_and_inputs_unchanged(self):
        before = deepcopy(self.rows)
        state = self.fit()
        matrix = subject.transform_scenario_rows(self.rows, state)
        self.assertEqual(subject.fit_scenario_fold(self.rows[::-1], 4,
                         "clinical_plus_baseline_surgery_and_womac"), state)
        np.testing.assert_array_equal(subject.transform_scenario_rows(self.rows[::-1], state), matrix[::-1])
        self.assertEqual(self.rows, before)

    def test_knee_specific_values_need_not_match_other_side(self):
        state = self.fit()
        self.assertEqual(state["train_participants"], 4)
        self.assertEqual(state["train_count"], 8)
        self.assertEqual(state["train_identity_sha256"], historical.fit_clinical_fold(self.base, 4)["train_identity_sha256"])

    def test_unknown_predictors_and_future_visits_rejected(self):
        for name in ("KL_48", "WOMAC_V06", "REEMPLAZO_ANTES_V06", "UNREVIEWED"):
            rows = deepcopy(self.rows)
            rows[0][name] = 1
            with self.assertRaisesRegex(ValueError, "closed_development"):
                subject.fit_scenario_fold(rows, 4)

    def test_invalid_values_rejected(self):
        for field, values in (("KL_BASE", (None, True, 1, 4, 2.0, "2")),
                              ("CIRUGIA_PREVIA", (True, 2, -1, 1.0, "0")),
                              ("WOMAC_TOTAL", (True, -1, 96.1, "12", float("nan"), float("inf")))):
            for value in values:
                with self.subTest(field=field, value=value):
                    rows = deepcopy(self.rows)
                    rows[0][field] = value
                    with self.assertRaises(ValueError):
                        subject.fit_scenario_fold(rows, 4)

    def test_reserved_test_assignments_rejected_in_fit_transform_and_sidecar(self):
        state = self.fit()
        self.rows[0]["outer_block"] = 0
        with self.assertRaises(ValueError):
            self.fit()
        with self.assertRaises(ValueError):
            subject.transform_scenario_rows(self.rows, state)
        self.base[0]["outer_block"] = 0
        with self.assertRaises(ValueError):
            self.sidecar()

    def test_duplicate_keys_and_cross_fold_participants_rejected(self):
        self.rows.append(deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.fit()
        self.rows = enriched_rows()
        self.rows[1]["cv_fold"] = 2
        with self.assertRaisesRegex(ValueError, "within_participant"):
            self.fit()

    def test_partial_complements_and_missing_required_complements_rejected(self):
        del self.rows[0]["WOMAC_TOTAL"]
        with self.assertRaisesRegex(ValueError, "complete_baseline"):
            self.fit()
        with self.assertRaisesRegex(ValueError, "complete_baseline"):
            subject.fit_scenario_fold(self.base, 4, "clinical_plus_baseline_kl")

    def test_tampered_state_fails_and_rehashed_wrong_training_statistics_fail_recomputation(self):
        state = self.fit()
        self.assertTrue(subject.verify_scenario_fold_state(self.rows, state))
        state["extra_mean"]["WOMAC_TOTAL"] += 1
        with self.assertRaisesRegex(ValueError, "state_changed"):
            subject.transform_scenario_rows(self.rows, state)
        self.reseal(state)
        with self.assertRaisesRegex(ValueError, "exact_training_fold"):
            subject.verify_scenario_fold_state(self.rows, state)

    def test_invalid_state_imputation_and_scaling_rejected_even_when_resealed(self):
        for path, value in (("CIRUGIA_PREVIA", True), ("CIRUGIA_PREVIA", 2), ("WOMAC_TOTAL", -1)):
            state = self.fit()
            state["extra_imputation"][path] = value
            self.reseal(state)
            with self.assertRaises(ValueError):
                subject.validate_scenario_state(state)
        state = self.fit()
        state["extra_scale"]["WOMAC_TOTAL"] = 0
        self.reseal(state)
        with self.assertRaises(ValueError):
            subject.validate_scenario_state(state)

    def test_sidecar_exact_membership_and_roundtrip_preserves_main_package(self):
        before = deepcopy(self.base)
        sidecar = self.sidecar()
        joined = subject.join_baseline_sidecar(self.base, sidecar, self.package,
                                               expected_sidecar_sha256=sidecar["record_sha256"])
        self.assertEqual(joined, self.rows)
        self.assertEqual(self.base, before)
        self.assertEqual(sidecar["scope"], "development_only")
        self.assertFalse(any("PROGRESION_48M" in row for row in sidecar["rows"]))
        self.assertEqual(subject.join_baseline_sidecar(self.base[::-1], sidecar, self.package), self.rows[::-1])

    def test_sidecar_rejects_extra_participants_missing_sides_duplicates_and_targets(self):
        for alteration in ("extra", "missing", "duplicate", "target", "future"):
            with self.subTest(alteration=alteration):
                rows = projection(self.rows)
                if alteration == "extra":
                    rows[0]["src_subject_id"] = "reserved_or_unknown"
                elif alteration == "missing":
                    rows.pop()
                elif alteration == "duplicate":
                    rows[0] = deepcopy(rows[1])
                elif alteration == "target":
                    rows[0]["PROGRESION_48M"] = 1
                else:
                    rows[0]["KL_48"] = 4
                with self.assertRaises(ValueError):
                    subject.make_baseline_sidecar(rows, self.base, self.package, self.source)

    def test_sidecar_does_not_replace_main_predictors(self):
        for field, value in (("AGEYEARS", 51), ("SEX", "F"), ("BMI", None)):
            rows = projection(self.rows)
            rows[0][field] = value
            with self.assertRaisesRegex(ValueError, "main_predictors_changed"):
                subject.make_baseline_sidecar(rows, self.base, self.package, self.source)

    def test_sidecar_external_package_and_seal_bindings_enforced(self):
        sidecar = self.sidecar()
        with self.assertRaisesRegex(ValueError, "package_binding"):
            subject.join_baseline_sidecar(self.base, sidecar, "c" * 64)
        with self.assertRaisesRegex(ValueError, "execution_binding"):
            subject.join_baseline_sidecar(self.base, sidecar, self.package, expected_sidecar_sha256="c" * 64)
        sidecar["rows"][0]["KL_BASE"] = 5 - sidecar["rows"][0]["KL_BASE"]
        with self.assertRaisesRegex(ValueError, "seal_changed"):
            subject.join_baseline_sidecar(self.base, sidecar, self.package)

    def test_resealed_sidecar_cannot_change_assignments_or_visit(self):
        sidecar = self.sidecar()
        changed = deepcopy(self.base)
        for row in changed[:2]:
            row["outer_block"] = 2
        with self.assertRaisesRegex(ValueError, "context_changed"):
            subject.join_baseline_sidecar(changed, sidecar, self.package)
        sidecar["baseline_visit"] = "V06"
        self.reseal(sidecar)
        with self.assertRaisesRegex(ValueError, "baseline_only"):
            subject.join_baseline_sidecar(self.base, sidecar, self.package)

    def test_existing_complement_not_silently_replaced(self):
        sidecar = self.sidecar()
        self.rows[0]["KL_BASE"] = 3
        with self.assertRaisesRegex(ValueError, "already_present_but_different"):
            subject.join_baseline_sidecar(self.rows, sidecar, self.package)

    def test_baseline_source_and_package_hashes_must_be_sha256(self):
        for value in (None, "", "a" * 63, "A" * 64, True):
            with self.assertRaises(ValueError):
                subject.make_baseline_sidecar(projection(self.rows), self.base, value, self.source)
            with self.assertRaises(ValueError):
                subject.make_baseline_sidecar(projection(self.rows), self.base, self.package, value)


if __name__ == "__main__":
    unittest.main()
