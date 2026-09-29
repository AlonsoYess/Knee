"""Guardrails for the pre-registered model comparison."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "model_registry.json"


class ModelRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def test_registry_is_bound_to_the_approved_change(self):
        self.assertEqual(self.registry["governing_change"], "MCR-2026-002")
        self.assertTrue(self.registry["test_block_must_remain_closed"])

    def test_core_modalities_and_outcome_remain_frozen(self):
        contract = self.registry["common_contract"]
        self.assertEqual(
            contract["clinical_predictors"],
            ["AGEYEARS_V00", "SEX", "BMI_V00"],
        )
        self.assertEqual(contract["radiographic_visit"], "V00")
        self.assertEqual(contract["outcome_visit"], "V06")
        self.assertEqual(contract["outcome"], "PROGRESION_48M")
        self.assertEqual(contract["group_unit"], "participant")
        self.assertTrue(contract["same_cohort_and_partitions"])

    def test_mandatory_baselines_and_modern_challengers_are_closed(self):
        image = self.registry["radiographic"]
        self.assertEqual(
            {item["family"] for item in image["mandatory_baselines"]},
            {"densenet121", "vit_b_16"},
        )
        self.assertEqual(
            {item["family"] for item in image["modern_challengers"]},
            {"convnext_v2_tiny", "dinov3_vits16", "skelex_vit_mae"},
        )
        self.assertTrue(
            all(item["license_review"].startswith("required") for item in image["modern_challengers"])
        )

    def test_multimodal_is_primary_and_bilateral_is_complementary(self):
        multimodal = self.registry["multimodal"]
        self.assertEqual(
            {item["family"] for item in multimodal["primary_candidates"]},
            {"intermediate_concatenation", "intermediate_gated_film"},
        )
        bilateral = self.registry["radiographic"]["complementary"][0]
        self.assertFalse(bilateral["confirmatory_role"])
        self.assertIn("contralateral", bilateral["eligibility"])

    def test_screening_never_uses_reserved_test(self):
        screening = self.registry["screening"]
        self.assertFalse(screening["test_in_screening"])
        self.assertEqual(screening["neural_finalists_use_seeds"], [2026, 2027, 2028])
        self.assertIn("test_driven_model_selection", self.registry["forbidden"])
        self.assertIn("automatic_kl_grade_output", self.registry["forbidden"])


if __name__ == "__main__":
    unittest.main()
