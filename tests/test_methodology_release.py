"""Release consistency and authorization boundaries; synthetic metadata only."""
from copy import deepcopy
from pathlib import Path
import unittest

from knee import methodology_release as aligned
from knee import final_model_release as previous
from knee.aligned_training import authority
from knee.clinical_scenarios import SCENARIOS
from knee.training_augmentation import VERSION as AUGMENTATION_VERSION, policy_document
from knee.training_plan import _read_json
from test_training_runner import request as old_request, authority as old_authority

ROOT = Path(__file__).resolve().parents[1]


class MethodologyReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.catalog, cls.plan, cls.registry = aligned.load_release(ROOT)
        cls.old_plan = _read_json(ROOT / "configs/training_plan_v1_1.json")
        cls.old_registry = _read_json(ROOT / "configs/model_registry_v1_1.json")
        previous.validate_plan(cls.old_plan, cls.old_registry)

    def request(self, scenario="clinical_main", image=False):
        value = old_request("a" * 64, model_id="image_densenet121" if image else "clinical_logistic")
        value.update(version=aligned.REQUEST_VERSION, clinical_scenario=scenario,
            baseline_sidecar_sha256=None if scenario == "clinical_main" else "b" * 64,
            augmentation_policy=AUGMENTATION_VERSION if image else "identity",
            modeling_release_sha256=aligned.PINS[aligned.RELEASE_PATH])
        return value

    def test_scope_matches_exact_features_dimensions_and_no_new_architecture(self):
        self.assertEqual(self.registry, self.old_registry)
        self.assertEqual(set(self.plan["clinical_scenarios"]), set(SCENARIOS))
        for key, features in SCENARIOS.items():
            self.assertEqual(self.plan["clinical_scenarios"][key]["features"], list(features))
            self.assertEqual(self.plan["clinical_scenarios"][key]["dimensions"], len(features))
        self.assertEqual(self.plan["neural_training"]["augmentation"], policy_document())
        self.assertEqual(self.plan["multimodal_models"]["complementary_enabled"], ["fusion_late_fixed_mean"])
        self.assertEqual(self.plan["multimodal_models"]["late_fusion"]["probability_weights"], [.5, .5])

    def test_no_training_or_test_authority_created_by_plan(self):
        self.assertFalse(self.plan["training_authorized"])
        self.assertFalse(self.plan["reserved_test_access_authorized"])
        result = aligned.validate_plan(self.plan, self.registry)
        self.assertFalse(result["training_authorized"])
        with self.assertRaises(ValueError):
            authority.validate_authority(None, self.request(), self.plan, self.registry)

    def test_every_recipe_change_requires_new_release(self):
        for mutate in (lambda p:p["clinical_scenarios"]["clinical_main"]["features"].append("KL_BASE"),
                       lambda p:p["neural_training"]["augmentation"].update(probability=.8),
                       lambda p:p["selection"].update(primary_metric="roc_auc"),
                       lambda p:p.update(training_authorized=True)):
            changed = deepcopy(self.plan)
            mutate(changed)
            with self.assertRaisesRegex(ValueError, "methodology_training_plan_changed"):
                aligned.validate_plan(changed, self.registry)

    def test_request_cannot_cross_release_versions(self):
        for req, plan, registry in ((self.request(),self.old_plan,self.old_registry),
                                   (old_request("a"*64),self.plan,self.registry)):
            with self.assertRaises(ValueError):
                authority.validate_request(req, plan, registry)

    def test_scope_sidecar_augmentation_and_release_bound_to_request(self):
        for field,value in (("clinical_scenario","unregistered"),
                            ("baseline_sidecar_sha256","c"*64),
                            ("augmentation_policy",AUGMENTATION_VERSION),
                            ("modeling_release_sha256","c"*64)):
            req=self.request()
            req[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                authority.validate_request(req,self.plan,self.registry)
        req=self.request("clinical_plus_baseline_kl",image=True)
        with self.assertRaisesRegex(ValueError,"registered_clinical_scenario_required"):
            authority.validate_request(req,self.plan,self.registry)

    def test_old_authority_cannot_authorize_new_recipe(self):
        req=self.request()
        record=old_authority(req,self.registry)
        with self.assertRaisesRegex(ValueError,"training_authority_must_match_exact_request_and_release"):
            authority.validate_authority(record,req,self.plan,self.registry)

    def test_historical_recipe_unchanged_and_valid(self):
        previous.validate_plan(self.old_plan,self.old_registry)
        self.assertEqual(self.old_plan["neural_training"]["augmentation"]["initial_policy"],"identity")
        self.assertEqual(self.old_plan["multimodal_models"]["complementary_disabled"],["fusion_late_fixed_mean"])


if __name__ == "__main__":
    unittest.main()
