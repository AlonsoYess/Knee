"""Pixel-only checks on synthetic development images, without a model fit."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from knee import methodology_inputs_colab as subject
from knee.methodology_release import load_release
from knee.training_augmentation import augment_training_image
from test_model_inputs_preflight import synthetic_development

ROOT = Path(__file__).resolve().parents[1]


class MethodologyAugmentationPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.plan, _ = load_release(ROOT)
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def check(self, rows, tensors, output):
        with patch("torch.optim.AdamW", side_effect=AssertionError("Optimizer forbidden")), \
             patch("torch.Tensor.backward", side_effect=AssertionError("Backward forbidden")), \
             patch("knee.aligned_training.models.build_model", side_effect=AssertionError("Model creation forbidden")), \
             patch("knee.aligned_training.runner.execute", side_effect=AssertionError("Training forbidden")):
            return subject.verify_augmentation(rows, tensors, self.plan, "a" * 64, output)

    def test_all_five_fold_pixel_visits_and_actual_normalization_pairs(self):
        rows, tensors = synthetic_development()
        before = {key: value.copy() for key, value in tensors.items()}
        with tempfile.TemporaryDirectory() as directory:
            result = self.check(rows, tensors, Path(directory))
            self.assertEqual(result["training_image_visits"], 120)
            self.assertEqual(result["validation_image_visits"], 30)
            self.assertEqual(result["normalization_pairs_verified"], 120)
            self.assertEqual(result["image_encoders_verified"], list(subject.IMAGE_MODELS))
            self.assertEqual(len(result["previews"]), 3)
            self.assertGreater(result["changed_training_images"], 0)
            for name in subject.PREVIEW_FILES:
                self.assertTrue((Path(directory) / name).is_file())
            for key, value in tensors.items():
                np.testing.assert_array_equal(value, before[key])
            self.assertFalse(result["training_executed"])
            self.assertFalse(result["predictive_improvement_claimed"])

    def test_preview_selection_uses_no_target_value(self):
        rows, tensors = synthetic_development()
        with tempfile.TemporaryDirectory() as directory:
            first = self.check(rows, tensors, Path(directory))
        changed = deepcopy(rows)
        for row in changed:
            row["PROGRESION_48M"] = 1 - row["PROGRESION_48M"]
        with tempfile.TemporaryDirectory() as directory:
            second = self.check(changed, tensors, Path(directory))
        self.assertEqual(first["previews"], second["previews"])
        self.assertEqual(first["changed_training_images"], second["changed_training_images"])

    def test_previews_cover_both_applied_draw_directions_and_unapplied_control(self):
        rows, tensors = synthetic_development()
        with tempfile.TemporaryDirectory() as directory:
            result = self.check(rows, tensors, Path(directory))
        lighter, darker, control = result["previews"]
        self.assertTrue(lighter["applied"])
        self.assertLess(lighter["gamma"], 1)
        self.assertNotEqual(lighter["source_pixel_sha256"], lighter["transformed_pixel_sha256"])
        self.assertTrue(darker["applied"])
        self.assertGreater(darker["gamma"], 1)
        self.assertNotEqual(darker["source_pixel_sha256"], darker["transformed_pixel_sha256"])
        self.assertFalse(control["applied"])
        self.assertEqual(control["gamma"], 1)
        self.assertEqual(control["source_pixel_sha256"], control["transformed_pixel_sha256"])

    def test_preview_selection_is_canonical_distinct_and_never_reads_clinical_or_target_fields(self):
        class IdentityOnly(dict):
            def __getitem__(self, key):
                if key not in {"src_subject_id", "side", "cv_fold"}:
                    raise AssertionError("Preview selection cannot read predictors or outcomes")
                return super().__getitem__(key)
        rows, _ = synthetic_development()
        rows = [IdentityOnly(row) for row in rows]
        policy = self.plan["neural_training"]["augmentation"]
        selected = subject._select_preview_keys(rows, policy, 2026)
        self.assertEqual(selected, subject._select_preview_keys(list(reversed(rows)), policy, 2026))
        self.assertEqual(len(selected), len(set(selected)))
        permitted = {(row["src_subject_id"], row["side"]) for row in rows if row["cv_fold"] != 0}
        self.assertLessEqual(set(selected), permitted)

    def test_preview_selection_fills_missing_draw_categories_without_reusing_keys(self):
        rows, _ = synthetic_development()
        policy = self.plan["neural_training"]["augmentation"]
        with patch("knee.training_augmentation.sample_parameters", return_value={"applied": True, "gamma": .95}):
            selected = subject._select_preview_keys(rows, policy, 2026)
        expected = sorted((row["src_subject_id"], row["side"]) for row in rows if row["cv_fold"] != 0)[:3]
        self.assertEqual(selected, expected)
        self.assertEqual(len(selected), len(set(selected)))

    def test_unknown_or_reserved_test_membership_rejected(self):
        rows, tensors = synthetic_development()
        rows[0]["outer_block"] = 0
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            self.check(rows, tensors, Path(directory))

    def test_missing_image_mapping_rejected(self):
        rows, tensors = synthetic_development()
        tensors.pop(next(iter(tensors)))
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "exact_development_tensor_mapping_required"):
            self.check(rows, tensors, Path(directory))

    def test_changed_validation_image_is_detected(self):
        rows, tensors = synthetic_development()
        def changed(array, **kwargs):
            return array.copy()
        with tempfile.TemporaryDirectory() as directory, \
             patch("knee.training_augmentation.apply_image_policy", side_effect=changed), \
             self.assertRaisesRegex(ValueError, "validation_image_policy_must_be_identity"):
            self.check(rows, tensors, Path(directory))

    def test_nondeterministic_training_transform_is_detected(self):
        rows, tensors = synthetic_development()
        count = 0
        def changed(array, **kwargs):
            nonlocal count
            value = augment_training_image(array, **kwargs).copy()
            count += 1
            if count % 2 == 0:
                value[0, 0, 0] = .123456
            value.setflags(write=False)
            return value
        with tempfile.TemporaryDirectory() as directory, \
             patch("knee.training_augmentation.augment_training_image", side_effect=changed), \
             self.assertRaisesRegex(ValueError, "deterministic_anatomy_preserving_augmentation_required"):
            self.check(rows, tensors, Path(directory))


if __name__ == "__main__":
    unittest.main()
