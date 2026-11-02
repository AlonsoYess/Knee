"""Additive release governance and unchanged development-only selection rules."""
from copy import deepcopy
from itertools import permutations
from pathlib import Path
import unittest
from unittest.mock import patch

from knee import final_model_release as subject
from knee import training_plan as old

ROOT = Path(__file__).resolve().parents[1]


class FinalModelReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.release, cls.catalog, cls.plan, cls.registry = subject.load_release(ROOT)
        cls.old_plan, cls.old_registry = old.load_training_plan(ROOT / "configs/training_plan_v1.json", ROOT / "configs/model_registry.json")

    def report(self, model_id, ap=.6, auc=.7, brier=.2):
        return {"model_id": model_id, "scope": "development_only",
                "aggregation": "mean_of_five_participant_held_out_fold_metrics",
                "eligibility_gates": {key: True for key in self.registry["eligibility_gates"]},
                "fold_metrics": [{"validation_fold": fold, "average_precision_score": ap,
                                  "roc_auc": auc, "brier": brier} for fold in range(5)]}

    def test_exact_roster_is_fourteen_and_seven_image_models(self):
        result = subject.validate_plan(self.plan, self.registry)
        self.assertEqual(result["registered_candidates"], 14)
        self.assertEqual(len(self.catalog["models"]), 7)
        self.assertEqual(sum(v["verification_enabled"] for v in self.catalog["models"].values()), 6)
        self.assertFalse(result["training_authorized"])

    def test_exact_two_additions_only_and_search_closed(self):
        self.assertEqual(set(subject.registered_model_ids(self.registry)) - set(old.registered_model_ids(self.old_registry)), set(subject.ADDED_IDS))
        self.assertTrue(self.registry["architecture_search_closed"])
        self.assertFalse(self.release["training_authorized"])
        self.assertFalse(self.release["reserved_test_access_authorized"])

    def test_amendment_names_only_the_three_existing_model_scope_fields(self):
        amendment = subject._read_json(ROOT / "configs/governance/final_model_amendment_v1.json")
        self.assertEqual(amendment["supersedes_only"], [
            "model_families.radiographic", "model_families.radiographic_modern_challengers",
            "selection.candidate_registry"])
        self.assertFalse(amendment["training_authorized"])
        self.assertFalse(amendment["reserved_test_access_authorized"])

    def test_core_contract_and_clinical_fusion_entries_are_identical(self):
        for key in ("common_contract", "clinical", "multimodal", "screening", "training_techniques", "eligibility_gates", "forbidden"):
            self.assertEqual(self.registry[key], self.old_registry[key])
        for key in ("inputs", "partitioning", "selection", "clinical_models", "multimodal_models", "neural_training", "future_calibration_and_threshold", "future_final_refit"):
            self.assertEqual(self.plan[key], self.old_plan[key])

    def test_historical_catalog_and_source_pins_still_validate(self):
        from knee.pretrained_verification import load_catalog
        catalog, plan, registry = load_catalog(ROOT)
        self.assertEqual(len(catalog["models"]), 5)
        self.assertEqual(registry["registry_version"], "1.0")
        self.assertEqual(plan, self.old_plan)

    def test_mutations_cannot_authorize_training_or_change_recipe(self):
        for key, value in (("training_authorized", True), ("reserved_test_access_authorized", True), ("version", "registered_training_plan_v1.0")):
            changed = deepcopy(self.plan)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_plan(changed, self.registry)
        changed = deepcopy(self.plan)
        changed["selection"]["maximum_promoted_modern_challengers"] = 3
        with self.assertRaises(ValueError):
            subject.validate_plan(changed, self.registry)

    def test_changed_registry_is_rejected(self):
        changed = deepcopy(self.registry)
        changed["common_contract"]["clinical_predictors"].append("KL")
        with self.assertRaises(ValueError):
            subject.validate_plan(self.plan, changed)

    def test_missing_release_does_not_fallback_to_historical_registry(self):
        with patch.object(subject, "_read_json", side_effect=FileNotFoundError), self.assertRaises(FileNotFoundError):
            subject.load_release(ROOT)

    def test_redirected_descriptor_is_rejected(self):
        original = subject._read_json
        def changed(path):
            value = original(path)
            if Path(path).name == "modeling_release_v1_1.json":
                value["registry"] = "configs/model_registry.json"
            return value
        with patch.object(subject, "_read_json", side_effect=changed), self.assertRaisesRegex(ValueError, "approved_release_component_changed"):
            subject.load_release(ROOT)

    def test_new_candidates_are_ranked_and_only_two_promoted(self):
        reports = [self.report(model_id, ap=.5) for model_id in self.plan["radiographic_models"]["conditional_challengers"]]
        reports[-2] = self.report("image_biomedclip", ap=.8)
        reports[-1] = self.report("image_mambaout_tiny", ap=.9)
        self.assertEqual(subject.select_modern_challengers(reports, self.plan, self.registry), ["image_mambaout_tiny", "image_biomedclip"])

    def test_old_three_candidates_keep_exact_old_ranking(self):
        reports = [self.report("image_convnextv2_tiny", .7, .6), self.report("image_dinov3_vits16", .695, .7), self.report("image_skelex", .688, .99)]
        for order in permutations(reports):
            self.assertEqual(subject.select_modern_challengers(list(order), self.plan, self.registry), old.select_modern_challengers(list(order), self.old_plan, self.old_registry))

    def test_new_models_obey_strict_near_tie_boundary(self):
        first = self.report("image_biomedclip", .7, .6)
        for score, expected in ((.69, ["image_biomedclip", "image_mambaout_tiny"]), (.690001, ["image_mambaout_tiny", "image_biomedclip"])):
            self.assertEqual(subject.select_modern_challengers([first, self.report("image_mambaout_tiny", score, .9)], self.plan, self.registry), expected)

    def test_ineligible_pending_or_test_reports_cannot_promote(self):
        report = self.report("image_biomedclip")
        report["eligibility_gates"][self.registry["eligibility_gates"][0]] = False
        self.assertEqual(subject.select_modern_challengers([report], self.plan, self.registry), [])
        report["scope"] = "reserved_test"
        with self.assertRaises(ValueError):
            subject.select_modern_challengers([report], self.plan, self.registry)

    def test_missing_duplicate_folds_and_pooled_metrics_rejected(self):
        for mutator in (lambda r: r["fold_metrics"].pop(), lambda r: r["fold_metrics"][0].update(validation_fold=1), lambda r: r.update(pooled_average_precision_score=.99)):
            report = self.report("image_mambaout_tiny")
            mutator(report)
            with self.assertRaises(ValueError):
                subject.select_modern_challengers([report], self.plan, self.registry)

    def test_final_epoch_rule_still_identical_no_fit(self):
        rows = [{"scope": "development_only", "validation_fold": f, "best_epoch": e} for f, e in enumerate((3, 12, 24, 17, 60))]
        self.assertEqual(subject.final_training_epochs(rows, self.plan, self.registry), old.final_training_epochs(rows, self.old_plan, self.old_registry))

    def test_dinov3_is_not_auto_enabled(self):
        self.assertFalse(self.catalog["models"]["image_dinov3_vits16"]["verification_enabled"])
        self.assertFalse(self.catalog["models"]["image_dinov3_vits16"]["license_review"]["terms_accepted_on_behalf_of_user"])


if __name__ == "__main__":
    unittest.main()
