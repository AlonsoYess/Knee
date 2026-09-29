"""Guardrails against silent deviations from the approved thesis scope."""

import json
import unittest
from pathlib import Path


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "governance"
    / "scope_contract.json"
)


class ScopeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_population_time_and_outcome_are_frozen(self):
        study = self.contract["study"]
        self.assertEqual(study["source"], "OAI")
        self.assertEqual(study["analysis_unit"], "knee")
        self.assertEqual(study["group_unit"], "participant")
        self.assertEqual(study["baseline_visit"], "V00")
        self.assertEqual(study["outcome_visit"], "V06")
        self.assertEqual(study["horizon_months"], 48)
        self.assertEqual(study["baseline_kl_included"], [2, 3])
        self.assertEqual(
            self.contract["outcome"]["positive_rule"],
            "KL_V06 - KL_V00 >= 1",
        )

    def test_main_predictors_and_scenarios_cannot_expand_silently(self):
        predictors = self.contract["predictors"]
        self.assertEqual(
            predictors["clinical_main"],
            ["AGEYEARS_V00", "SEX", "BMI_V00"],
        )
        self.assertEqual(
            predictors["kl_baseline_usage"],
            "complementary_reference_only",
        )
        self.assertTrue(
            {"KL_V06", "PROGRESION_48M", "V06_image"}.issubset(
                predictors["forbidden_main"]
            )
        )
        self.assertEqual(
            self.contract["scenarios"]["main"],
            ["clinical", "radiographic", "multimodal_intermediate_fusion"],
        )
        self.assertEqual(
            set(self.contract["scenarios"]["complementary"]),
            {
                "clinical_plus_baseline_kl",
                "clinical_plus_baseline_surgery_and_womac",
                "multimodal_late_fusion",
            },
        )
        self.assertTrue(
            self.contract["scenarios"]["same_cohort_and_partitions_required"]
        )

    def test_grouped_development_and_reserved_test_are_locked(self):
        splitting = self.contract["splitting"]
        self.assertEqual(splitting["seed"], 2026)
        self.assertEqual(splitting["test_block"], 0)
        self.assertEqual(splitting["group_by"], "participant")
        self.assertEqual(splitting["development_cv_folds"], 5)
        self.assertEqual(splitting["development_cv"], "stratified_grouped")
        self.assertFalse(splitting["test_in_development_cv"])
        self.assertEqual(splitting["test_openings_allowed"], 1)

    def test_primary_metric_statistics_and_calibration_are_locked(self):
        self.assertEqual(
            self.contract["selection"]["primary_metric"],
            "average_precision_score",
        )
        self.assertEqual(
            self.contract["selection"]["tie_break_order"],
            ["roc_auc", "brier_score"],
        )
        statistics = self.contract["confirmatory_statistics"]
        self.assertEqual(statistics["bootstrap_resamples"], 2000)
        self.assertEqual(statistics["bootstrap_unit"], "participant")
        self.assertTrue(statistics["paired_model_differences"])
        self.assertEqual(statistics["multiplicity_adjustment"], "holm")
        calibration = self.contract["calibration_and_threshold"]
        self.assertEqual(
            calibration["fit_source"],
            "development_out_of_fold_predictions_only",
        )
        self.assertEqual(calibration["threshold_rule"], "youden_index")

    def test_safety_gates_remain_closed(self):
        self.assertEqual(self.contract["contract_version"], "1.0")
        self.assertEqual(self.contract["status"], "approved_by_researcher")
        self.assertFalse(self.contract["training_authorized"])
        self.assertEqual(
            self.contract["prototype"]["estimated_kl_output"],
            "removed_from_scope_by_researcher_decision",
        )
        self.assertEqual(
            set(self.contract["pending_blockers"]),
            {"PEND-DATA-001", "PEND-PILOT-001"},
        )
        kl_decision = self.contract["resolved_decisions"][0]
        self.assertEqual(kl_decision["id"], "RES-KL-001")
        self.assertEqual(
            kl_decision["decision"],
            "remove_estimated_kl_output_from_chapter_1_and_prototype",
        )
        self.assertTrue(kl_decision["chapter_1_word_update_pending"])
        self.assertEqual(kl_decision["chapter_1_word_update_owner"], "researcher")
        self.assertFalse(self.contract["prototype"]["diagnostic_use"])
        self.assertFalse(
            self.contract["reporting"]["peruvian_population_generalization_allowed"]
        )


if __name__ == "__main__":
    unittest.main()
