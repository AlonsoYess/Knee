"""Synthetic v1.2 input/authority integration; no optimization or backward pass."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from knee import final_model_release as historical
from knee import methodology_release as release
from knee import training_augmentation as augmentation
from knee import training_data as historical_data
from knee.aligned_training import authority, data, engine, models
from test_model_inputs_preflight import synthetic_development


ROOT = Path(__file__).resolve().parents[1]


def request_for(*, model_id="image_densenet121", scenario="clinical_main", seed=2026):
    image = model_id if model_id.startswith("image_") else None
    return {"version": release.REQUEST_VERSION, "operation": "development_fold", "model_id": model_id,
        "image_encoder_id": image, "stage": "stage_1", "validation_fold": 0, "seed": seed,
        "development_package_sha256": "a" * 64, "image_micro_batch_size": 4,
        "selection_evidence_sha256": None, "fold_image_job_sha256": None,
        "clinical_scenario": scenario,
        "baseline_sidecar_sha256": None if scenario == "clinical_main" else "b" * 64,
        "augmentation_policy": augmentation.VERSION if image else "identity",
        "modeling_release_sha256": release.PINS[release.RELEASE_PATH]}


def synthetic_authority(request, registry):
    return {"version": authority.VERSION, "training_authorized": True,
        "reserved_test_access_authorized": False, "approval_source": "direct_investigator_authorization",
        "approval_statement": "Synthetic authority fixture only; no execution is performed.",
        "approved_at_utc": "2026-10-25T16:00:00+00:00", "request_sha256": authority.digest(request),
        "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "eligibility_gates": {key: True for key in registry["eligibility_gates"]},
        "full_encoder_training_authorized": False}


class UnusedImageModel(torch.nn.Module):
    """Metadata-only object for failures that must precede model/optimizer use."""
    model_id = "image_densenet121"
    image_encoder_id = "image_densenet121"
    clinical_scenario = "clinical_main"

    def forward(self, *args, **kwargs):
        raise AssertionError("No model forward is allowed in authority-rejection tests")


class MethodologyAugmentationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.catalog, cls.plan, cls.registry = release.load_release(ROOT)
        _, _, cls.old_plan, cls.old_registry = historical.load_release(ROOT)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.rows, self.tensors = synthetic_development()
        self.original_bytes = {key: value.tobytes() for key, value in self.tensors.items()}

    def prepared(self, *, encoder="image_densenet121", seed=2026, policy=True):
        return data.prepare_fold(self.rows, self.tensors, 0, encoder_id=encoder,
            package_sha256="a" * 64, methodology=True, seed=seed,
            augmentation_policy=augmentation.policy_document() if policy and encoder else None)

    def permit(self, request=None):
        request = request_for() if request is None else request
        return authority.validate_authority(synthetic_authority(request, self.registry),
            request, self.plan, self.registry)

    def assert_rejected_before_optimizer(self, prepared, *, pattern, permit=None, plan=None, seed=2026):
        loaders = data.loaders(prepared, seed=seed, micro_batch_size=4)
        permit = self.permit() if permit is None else permit
        plan = self.plan if plan is None else plan
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_be_created"
            with patch.object(engine, "_new_optimizer", side_effect=AssertionError("optimizer forbidden")) as optimizer, \
                 patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")) as backward, \
                 patch.object(engine, "_new_output", side_effect=AssertionError("output forbidden")) as output_gate:
                with self.assertRaisesRegex(ValueError, pattern):
                    engine.train_neural_fold(UnusedImageModel(), *loaders, plan, device="cpu",
                        seed=seed, positive_weight=prepared.class_weight["positive_weight"],
                        output_dir=output, execution_authority=permit)
                optimizer.assert_not_called()
                backward.assert_not_called()
                output_gate.assert_not_called()
            self.assertFalse(output.exists())

    def test_actual_dataset_applies_gamma_before_encoder_normalization(self):
        for encoder in ("image_densenet121", "image_vit_b16", "image_convnextv2_tiny",
                        "image_skelex", "image_biomedclip", "image_mambaout_tiny"):
            prepared = self.prepared(encoder=encoder)
            changed = 0
            with torch.inference_mode():
                for index, (participant, side) in enumerate(prepared.training.identities):
                    source = self.tensors[(participant, side)]
                    expected_gray = augmentation.augment_training_image(source, seed=2026,
                        validation_fold=0, epoch=1, src_subject_id=participant, side=side)
                    expected = models.normalize_for_model(expected_gray, encoder)
                    actual = prepared.training[index]["image"]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(tuple(actual.shape), (3, 224, 224))
                    self.assertTrue(torch.isfinite(actual).all())
                    changed += not torch.equal(actual, models.normalize_for_model(source, encoder))
            self.assertGreater(changed, 0, encoder)
        self.assertEqual(self.original_bytes, {key: value.tobytes() for key, value in self.tensors.items()})

    def test_validation_images_clinical_values_and_targets_identical_to_legacy(self):
        modern = self.prepared()
        legacy = historical_data.prepare_fold(self.rows, self.tensors, 0, encoder_id="image_densenet121",
                                              package_sha256="a" * 64)
        self.assertIsNone(modern.validation.augmentation_policy)
        self.assertEqual(modern.validation.identities, legacy.validation.identities)
        for index in range(len(modern.validation)):
            actual, expected = modern.validation[index], legacy.validation[index]
            for key in ("image", "clinical", "target"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        np.testing.assert_array_equal(modern.training._clinical, legacy.training._clinical)
        self.assertEqual(modern.training._targets, legacy.training._targets)

    def test_dataloader_yields_every_identity_with_correct_epoch_specific_image(self):
        prepared = self.prepared()
        prepared.training.set_epoch(3)
        training_loader, validation_loader = data.loaders(prepared, seed=2026, micro_batch_size=4)
        observed = []
        for batch in training_loader:
            for index, (participant, side) in enumerate(zip(batch["src_subject_id"], batch["side"])):
                identity = participant, side
                observed.append(identity)
                expected_gray = augmentation.augment_training_image(self.tensors[identity], seed=2026,
                    validation_fold=0, epoch=3, src_subject_id=participant, side=side)
                torch.testing.assert_close(batch["image"][index],
                    models.normalize_for_model(expected_gray, "image_densenet121"), rtol=0, atol=0)
        self.assertEqual(set(observed), set(prepared.training.identities))
        self.assertEqual(len(observed), len(prepared.training))
        self.assertEqual(sum(len(batch["target"]) for batch in validation_loader), len(prepared.validation))

    def test_epoch_replay_and_reconstruction_are_exact_and_epoch_changes_variations(self):
        prepared = self.prepared()
        epoch_one = [prepared.training[i]["image"].clone() for i in range(len(prepared.training))]
        prepared.training.set_epoch(2)
        epoch_two = [prepared.training[i]["image"].clone() for i in range(len(prepared.training))]
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(epoch_one, epoch_two)))
        reconstructed = self.prepared()
        reconstructed.training.set_epoch(2)
        for index, expected in enumerate(epoch_two):
            torch.testing.assert_close(reconstructed.training[index]["image"], expected, rtol=0, atol=0)
        prepared.training.set_epoch(1)
        for index, expected in enumerate(epoch_one):
            torch.testing.assert_close(prepared.training[index]["image"], expected, rtol=0, atol=0)

    def test_valid_loader_and_permit_binding_is_accepted_without_training(self):
        prepared = self.prepared()
        loaders = data.loaders(prepared, seed=2026, micro_batch_size=4)
        binding = engine._validate_loader_bindings(*loaders, self.permit(),
                                                  prepared.class_weight["positive_weight"])
        self.assertEqual(binding["training_knees"], len(prepared.training))
        self.assertEqual(binding["validation_knees"], len(prepared.validation))

    def test_legacy_identity_inputs_cannot_enter_new_gamma_campaign(self):
        self.assert_rejected_before_optimizer(self.prepared(policy=False), pattern="augmentation_must_match")

    def test_training_augmentation_seed_mismatch_is_rejected_before_optimizer(self):
        self.assert_rejected_before_optimizer(self.prepared(seed=2027), pattern="augmentation_must_match")

    def test_validation_augmentation_injection_rejected_before_optimizer(self):
        prepared = self.prepared()
        prepared.validation.augmentation_policy = augmentation.policy_document()
        self.assert_rejected_before_optimizer(prepared, pattern="augmentation_must_match")

    def test_mutated_training_policy_rejected_before_optimizer(self):
        prepared = self.prepared()
        prepared.training.augmentation_policy["probability"] = 1.0
        self.assert_rejected_before_optimizer(prepared, pattern="reviewed_training_augmentation_policy")

    def test_historical_plan_cannot_use_new_recipe_permit(self):
        self.assert_rejected_before_optimizer(self.prepared(), plan=self.old_plan,
                                             pattern="recipe_must_match_execution_authority")

    def test_changed_reviewed_plan_cannot_reach_optimizer(self):
        plan = deepcopy(self.plan)
        plan["neural_training"]["augmentation"]["gamma_maximum"] = 1.2
        self.assert_rejected_before_optimizer(self.prepared(), plan=plan,
                                             pattern="approved_training_recipe_required")

    def test_request_cannot_disable_gamma_or_attach_complement_to_image_model(self):
        for change in ({"augmentation_policy": "identity"},
                       {"clinical_scenario": "clinical_plus_baseline_kl", "baseline_sidecar_sha256": "b" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                authority.validate_request({**request_for(), **change}, self.plan, self.registry)

    def test_request_and_approval_bind_new_release_and_sidecar(self):
        request = request_for(model_id="clinical_mlp", scenario="clinical_plus_baseline_kl")
        record = synthetic_authority(request, self.registry)
        authority.validate_authority(record, request, self.plan, self.registry)
        changed = {**request, "baseline_sidecar_sha256": "c" * 64}
        with self.assertRaisesRegex(ValueError, "match_exact_request"):
            authority.validate_authority(record, changed, self.plan, self.registry)
        with self.assertRaisesRegex(ValueError, "release_mismatch"):
            authority.validate_request({**request, "modeling_release_sha256": historical.PINS[historical.RELEASE_PATH]},
                                       self.plan, self.registry)

    def test_conventional_complete_fit_cannot_be_authorized_as_two_update_smoke(self):
        for model_id in ("clinical_logistic", "clinical_xgboost"):
            for scenario in self.plan["clinical_scenarios"]:
                request = request_for(model_id=model_id, scenario=scenario)
                request["operation"] = "single_fold_smoke"
                with self.subTest(model=model_id, scenario=scenario), \
                     self.assertRaisesRegex(ValueError, "bounded_smoke_requires_neural_model"):
                    authority.validate_authority(synthetic_authority(request, self.registry),
                                                 request, self.plan, self.registry)
        request = request_for(model_id="clinical_mlp")
        request["operation"] = "single_fold_smoke"
        self.assertEqual(authority.validate_request(request, self.plan, self.registry), request)

    def test_clinical_mlp_accepts_only_registered_dimensions_without_training(self):
        dimensions = {"clinical_main": 3, "clinical_plus_baseline_kl": 4,
                      "clinical_plus_baseline_surgery_and_womac": 5}
        for scenario, width in dimensions.items():
            with patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer forbidden")), \
                 patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")):
                model, audit = models.build_model("clinical_mlp", self.plan, self.catalog,
                                                  clinical_scenario=scenario)
                before = {key: value.clone() for key, value in model.state_dict().items()}
                with torch.inference_mode():
                    output = model(clinical=torch.zeros((2, width), dtype=torch.float32))
                    with self.assertRaises(ValueError):
                        model(clinical=torch.zeros((2, width + 1), dtype=torch.float32))
                self.assertEqual(tuple(output.shape), (2,))
                self.assertTrue(torch.isfinite(output).all())
                self.assertFalse(audit["training_executed"])
                for key, expected in before.items():
                    torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
