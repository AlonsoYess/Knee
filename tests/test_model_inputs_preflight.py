"""Synthetic input-loading preflight tests; no downloads or large models."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from knee import model_inputs_preflight as subject
from knee.clinical_fold_inputs import fit_clinical_fold
from knee.model_input_adapters import DevelopmentFoldDataset


def synthetic_development():
    rows, tensors = [], {}
    for fold in range(5):
        for person in range(3):
            sid = f"synthetic_{fold}_{person}"
            for side in ("RIGHT", "LEFT"):
                array = np.full((1, 224, 224), (fold * 3 + person + 1) / 20, dtype=np.float32)
                array.setflags(write=False)
                pair = sid, side
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                tensors[pair] = array
                rows.append({"src_subject_id": sid, "side": side,
                    "preprocessing_role": "development" if person == 0 else "routine",
                    "outer_block": 1 + fold % 4, "cv_fold": fold,
                    "AGEYEARS": 50 + fold * 4 + person,
                    "SEX": "F" if person % 2 else "M",
                    "BMI": None if fold == 1 and person == 2 else 23.0 + fold + person,
                    "PROGRESION_48M": int(person == 1 and side == "RIGHT"),
                    "tensor": {"relative_path": f"tensors/{sid}_{side}.npy",
                        "sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
                        "pixel_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                        "shape": [1, 224, 224], "dtype": "float32"}})
    return rows, tensors


class TinyArchitecture(torch.nn.Module):
    """Small stand-in retaining parameters and BN buffers for probe assertions."""
    def __init__(self, width, *, defect=None):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3)
        self.projection = torch.nn.Linear(3, width)
        self.classifier = torch.nn.Identity()
        self.heads = torch.nn.Identity()
        self.defect = defect
        self.forward_checks = []

    def forward(self, image):
        self.forward_checks.append({"eval": not any(m.training for m in self.modules()),
                                    "inference": torch.is_inference_mode_enabled(),
                                    "grad_disabled": not torch.is_grad_enabled()})
        features = self.projection(self.bn(image.mean(dim=(2, 3))))
        if self.defect == "nonfinite":
            return features * float("nan")
        if self.defect == "buffer_mutation":
            self.bn.running_mean.add_(1)
        return features


class ModelInputsPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project = Path(__file__).resolve().parents[1]
        cls.adopted_policy = json.loads((project / "configs/model_inputs_preflight_v1.json").read_text(encoding="utf-8"))
        cls.registry = json.loads((project / "configs/model_registry.json").read_text(encoding="utf-8"))
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.policy = deepcopy(self.adopted_policy)
        self.rows, self.tensors = synthetic_development()

    def dataset(self, *, fold=0, split="validation", encoder="image_densenet121"):
        state = fit_clinical_fold(self.rows, fold)
        return DevelopmentFoldDataset(self.rows, self.tensors, state, encoder, split)

    def test_adopted_closed_policy_and_registry_are_accepted_without_mutation(self):
        before = deepcopy((self.policy, self.registry))
        self.assertIsNone(subject.validate_policy(self.policy, self.registry))
        self.assertEqual((self.policy, self.registry), before)

    def test_unknown_policy_field_is_rejected(self):
        self.policy["unapproved_preprocessing"] = "histogram_equalization"
        with self.assertRaises(ValueError):
            subject.validate_policy(self.policy, self.registry)

    def test_policy_cannot_enable_training_test_access_or_pretrained_download(self):
        for key in ("pretrained_downloads_authorized", "predictive_training_authorized",
                    "reserved_test_access_authorized"):
            with self.subTest(flag=key):
                changed = deepcopy(self.policy)
                changed[key] = True
                with self.assertRaises(ValueError):
                    subject.validate_policy(changed, self.registry)

    def test_policy_cannot_add_predictor_or_unadopted_encoder(self):
        for key, value in (("clinical_features", ["AGEYEARS", "SEX", "BMI", "KL_BASE"]),
                           ("image_encoders", ["image_densenet121", "image_skelex"])):
            with self.subTest(field=key):
                changed = deepcopy(self.policy)
                changed[key] = value
                with self.assertRaises(ValueError):
                    subject.validate_policy(changed, self.registry)

    def test_policy_cannot_change_fold_seed_or_loader_retention(self):
        for key, value in (("seed", 999), ("folds", [0, 1, 2, 3]),
                           ("loader", {"batch_size": 8, "num_workers": 0,
                                       "shuffle": True, "drop_last": True})):
            with self.subTest(field=key):
                changed = deepcopy(self.policy)
                changed[key] = value
                with self.assertRaises(ValueError):
                    subject.validate_policy(changed, self.registry)

    def test_registry_cannot_remove_required_image_baseline(self):
        registry = deepcopy(self.registry)
        registry["radiographic"]["mandatory_baselines"] = registry["radiographic"]["mandatory_baselines"][1:]
        with self.assertRaises(ValueError):
            subject.validate_policy(self.policy, registry)

    def test_prepare_returns_exact_five_fold_states_matching_individual_fit(self):
        states = subject.prepare_fold_states(self.rows, self.policy)
        self.assertEqual(len(states), 5)
        self.assertEqual([state["validation_fold"] for state in states], list(range(5)))
        for fold, state in enumerate(states):
            self.assertEqual(state, fit_clinical_fold(self.rows, fold))
            self.assertEqual(state["train_count"], 24)
            self.assertEqual(state["train_participants"], 12)

    def test_prepare_fold_states_is_repeatable_and_preserves_inputs(self):
        before = deepcopy((self.rows, self.policy))
        first = subject.prepare_fold_states(self.rows, self.policy)
        self.assertEqual(first, subject.prepare_fold_states(self.rows, self.policy))
        self.assertEqual((self.rows, self.policy), before)
        self.rows.reverse()
        self.assertEqual(first, subject.prepare_fold_states(self.rows, self.policy))

    def test_fold_validation_extremes_do_not_leak_into_its_training_statistics(self):
        expected = subject.prepare_fold_states(self.rows, self.policy)
        for row in self.rows:
            if row["cv_fold"] == 2:
                row.update(AGEYEARS=120, BMI=250.0)
        actual = subject.prepare_fold_states(self.rows, self.policy)
        self.assertEqual(actual[2], expected[2])
        self.assertNotEqual(actual[0], expected[0])

    def test_target_flips_do_not_change_five_clinical_states(self):
        expected = subject.prepare_fold_states(self.rows, self.policy)
        for row in self.rows:
            row["PROGRESION_48M"] = 1 - row["PROGRESION_48M"]
        self.assertEqual(subject.prepare_fold_states(self.rows, self.policy), expected)

    def test_loader_processes_every_validation_row_and_partial_last_batch(self):
        result = subject.check_loader(self.dataset(), batch_size=4)
        self.assertEqual(result["rows"], 6)
        self.assertEqual(result["participants"], 3)
        self.assertEqual(result["batches"], 2)
        self.assertEqual(result["image_shape"], [3, 224, 224])
        self.assertEqual(result["clinical_features"], 3)
        self.assertEqual(result["target_shape"], [])
        for key in ("ordered_identity_sha256", "clinical_sha256", "image_sha256", "targets_sha256"):
            self.assertRegex(result[key], r"^[0-9a-f]{64}$")

    def test_loader_full_training_pass_and_final_partial_batch(self):
        result = subject.check_loader(self.dataset(split="training"), batch_size=5)
        self.assertEqual(result["rows"], 24)
        self.assertEqual(result["participants"], 12)
        self.assertEqual(result["batches"], 5)

    def test_loader_hashes_are_independent_of_batch_boundaries(self):
        dataset = self.dataset()
        first = subject.check_loader(dataset, batch_size=4)
        second = subject.check_loader(dataset, batch_size=8)
        self.assertEqual(second["batches"], 1)
        for key in ("ordered_identity_sha256", "clinical_sha256", "image_sha256", "targets_sha256"):
            self.assertEqual(first[key], second[key])

    def test_loader_repeated_run_is_exact_and_does_not_refit_clinical_inputs(self):
        dataset = self.dataset()
        first = subject.check_loader(dataset, batch_size=4)
        with patch("knee.clinical_fold_inputs.fit_clinical_fold",
                   side_effect=AssertionError("loader must never fit")):
            self.assertEqual(subject.check_loader(dataset, batch_size=4), first)

    def test_loader_does_not_modify_rows_or_readonly_source_images(self):
        before_rows = deepcopy(self.rows)
        before_images = {pair: array.tobytes() for pair, array in self.tensors.items()}
        subject.check_loader(self.dataset(), batch_size=4)
        self.assertEqual(self.rows, before_rows)
        for pair, array in self.tensors.items():
            self.assertEqual(array.tobytes(), before_images[pair])
            self.assertFalse(array.flags.writeable)

    def test_loader_targets_remain_aligned_and_change_only_target_hash(self):
        first = subject.check_loader(self.dataset(), batch_size=4)
        for row in self.rows:
            if row["cv_fold"] == 0:
                row["PROGRESION_48M"] = 1 - row["PROGRESION_48M"]
        second = subject.check_loader(self.dataset(), batch_size=4)
        for key in ("ordered_identity_sha256", "clinical_sha256", "image_sha256"):
            self.assertEqual(first[key], second[key])
        self.assertNotEqual(first["targets_sha256"], second["targets_sha256"])

    def test_loader_rejects_nonpositive_or_noninteger_batch_size(self):
        dataset = self.dataset()
        for invalid in (0, -1, True, 2.0, "2"):
            with self.subTest(batch_size=invalid):
                with self.assertRaises((ValueError, TypeError)):
                    subject.check_loader(dataset, batch_size=invalid)

    def test_both_adopted_encoder_adapters_yield_same_geometry_and_alignment(self):
        first = subject.check_loader(self.dataset(encoder="image_densenet121"), batch_size=4)
        second = subject.check_loader(self.dataset(encoder="image_vit_b16"), batch_size=4)
        self.assertEqual(first, second)

    def test_architecture_probes_use_no_weights_eval_inference_and_unchanged_state(self):
        samples = [torch.full((1 + fold % 2, 3, 224, 224), 0.25 + fold / 10,
                              dtype=torch.float32) for fold in range(5)]
        before_samples = [sample.clone() for sample in samples]
        before_rng = torch.random.get_rng_state().clone()
        for encoder, constructor, width in (("image_densenet121", "densenet121", 1024),
                                             ("image_vit_b16", "vit_b_16", 768)):
            with self.subTest(encoder=encoder):
                created = []
                def create(*args, **kwargs):
                    self.assertEqual(args, ())
                    self.assertEqual(kwargs, {"weights": None})
                    model = TinyArchitecture(width)
                    created.append(model)
                    return model
                with patch.object(subject, constructor, side_effect=create) as factory:
                    result = subject.architecture_probe(encoder, samples)
                factory.assert_called_once_with(weights=None)
                self.assertFalse(result["pretrained_checkpoint_loaded"])
                self.assertFalse(result["pretrained_download_attempted"])
                self.assertTrue(result["parameters_and_buffers_unchanged"])
                self.assertTrue(result["all_parameter_gradients_none"])
                self.assertEqual(result["state_sha256_before"], result["state_sha256_after"])
                self.assertEqual(len(result["folds"]), 5)
                self.assertTrue(all(row["feature_width"] == width for row in result["folds"]))
                self.assertEqual(len(created[0].forward_checks), 5)
                self.assertTrue(all(all(check.values()) for check in created[0].forward_checks))
                self.assertTrue(all(parameter.grad is None for parameter in created[0].parameters()))
                self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
                for before, after in zip(before_samples, samples):
                    self.assertTrue(torch.equal(before, after))

    def test_architecture_probe_rejects_nonfinite_features_from_lightweight_model(self):
        samples = [torch.zeros((1, 3, 224, 224), dtype=torch.float32) for _ in range(5)]
        with patch.object(subject, "densenet121", return_value=TinyArchitecture(1024, defect="nonfinite")):
            with self.assertRaisesRegex(ValueError, "finite_expected_architecture_features_required"):
                subject.architecture_probe("image_densenet121", samples)

    def test_architecture_probe_rejects_parameter_or_buffer_mutation(self):
        samples = [torch.zeros((1, 3, 224, 224), dtype=torch.float32) for _ in range(5)]
        with patch.object(subject, "densenet121", return_value=TinyArchitecture(1024, defect="buffer_mutation")):
            with self.assertRaisesRegex(ValueError, "probe_changed_parameters_buffers_or_gradients"):
                subject.architecture_probe("image_densenet121", samples)

    def test_invalid_architecture_probe_inputs_fail_before_model_construction(self):
        valid = [torch.zeros((1, 3, 224, 224), dtype=torch.float32) for _ in range(5)]
        cases = [("foreign_encoder", valid), ("image_densenet121", valid[:4]),
                 ("image_densenet121", [torch.zeros((3, 3, 224, 224))] + valid[1:]),
                 ("image_densenet121", [valid[0].double()] + valid[1:]),
                 ("image_densenet121", [valid[0].clone().requires_grad_()] + valid[1:]),
                 ("image_densenet121", [valid[0] * float("nan")] + valid[1:])]
        with patch.object(subject, "densenet121") as dense, patch.object(subject, "vit_b_16") as vit:
            for encoder, samples in cases:
                with self.subTest(encoder=encoder, sample_shape=tuple(samples[0].shape)):
                    with self.assertRaises(ValueError):
                        subject.architecture_probe(encoder, samples)
            dense.assert_not_called()
            vit.assert_not_called()

    def test_loader_rejects_corrupt_shapes_nonfinite_values_and_nonbinary_targets(self):
        dataset = self.dataset()
        batch = next(iter(subject.make_development_loader(dataset, 6)))
        for field, value in (("clinical", torch.zeros((6, 4))),
                             ("image", torch.zeros((6, 3, 224, 224)) * float("nan")),
                             ("target", torch.full((6,), 2.0))):
            with self.subTest(field=field):
                altered = {**batch, field: value}
                with patch.object(subject, "make_development_loader", return_value=[altered]):
                    with self.assertRaises(ValueError):
                        subject.check_loader(dataset, batch_size=6)

    def test_loader_rejects_changed_identity_order(self):
        dataset = self.dataset()
        batch = next(iter(subject.make_development_loader(dataset, 6)))
        altered = {**batch, "side": list(reversed(batch["side"]))}
        with patch.object(subject, "make_development_loader", return_value=[altered]):
            with self.assertRaisesRegex(ValueError, "complete_ordered_development_loading_required"):
                subject.check_loader(dataset, batch_size=6)


if __name__ == "__main__":
    unittest.main()
