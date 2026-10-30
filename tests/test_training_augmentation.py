"""Synthetic-only augmentation regression tests; no fitting or participant data."""
from copy import deepcopy
import inspect
import unittest

import numpy as np

from knee import training_augmentation as aug


def synthetic_image():
    image = np.zeros((1, 224, 224), dtype=np.float32)
    image[:, 24:200, 40:184] = np.linspace(0.001, 0.999, 176 * 144,
                                         dtype=np.float32).reshape(1, 176, 144)
    image[0, 199, 183] = 1.0
    image.setflags(write=False)
    return image


class TrainingAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.image = synthetic_image()
        self.context = dict(seed=2026, validation_fold=0, epoch=1,
                            src_subject_id="synthetic_participant", side="LEFT")
        # Select an artificial knee with an active transform; no image values
        # or outcomes enter the deterministic draw.
        for index in range(100):
            self.context["src_subject_id"] = f"synthetic_{index}"
            if aug.sample_parameters(**self.context)["applied"]:
                break
        else:
            self.fail("Synthetic fixture did not exercise the gamma operation")

    def test_policy_is_exact_and_a_copy(self):
        policy = aug.policy_document()
        self.assertEqual(aug.validate_policy(policy), aug.POLICY_SHA256)
        policy["gamma_maximum"] = 1.2
        with self.assertRaisesRegex(ValueError, "reviewed_training_augmentation_policy"):
            aug.validate_policy(policy)
        self.assertEqual(aug.policy_document()["gamma_maximum"], 1.1)

    def test_probability_geometry_laterality_and_extra_keys_cannot_change(self):
        for key, value in (("probability", 1.0), ("geometry_modified", True),
                           ("laterality_modified", True), ("extra", "field")):
            policy = aug.policy_document()
            policy[key] = value
            with self.assertRaises(ValueError):
                aug.validate_policy(policy)

    def test_nonfinite_policy_rejected(self):
        policy = aug.policy_document()
        policy["gamma_minimum"] = float("nan")
        with self.assertRaises(ValueError):
            aug.validate_policy(policy)

    def test_real_intensity_variation_without_mutating_source_or_padding(self):
        before = self.image.tobytes()
        result = aug.augment_training_image(self.image, **self.context)
        self.assertEqual(before, self.image.tobytes())
        self.assertFalse(np.shares_memory(result, self.image))
        self.assertEqual(result.shape, (1, 224, 224))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(result.flags.writeable)
        self.assertFalse(np.array_equal(result, self.image))
        np.testing.assert_array_equal(result == 0, self.image == 0)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)
        self.assertEqual(result[0, 199, 183], 1.0)

    def test_no_pixel_relocation_and_monotone_mapping(self):
        parameters = aug.sample_parameters(**self.context)
        result = aug.augment_training_image(self.image, **self.context)
        expected = np.power(self.image.astype(np.float64), parameters["gamma"]).astype(np.float32)
        np.testing.assert_array_equal(result, expected)
        selected = self.image > 0
        self.assertTrue(np.all(np.diff(result[selected]) >= 0))

    def test_identity_draw_still_returns_a_readonly_copy(self):
        context = deepcopy(self.context)
        for index in range(100):
            context["src_subject_id"] = f"inactive_{index}"
            if not aug.sample_parameters(**context)["applied"]:
                break
        else:
            self.fail("Synthetic fixture did not exercise identity draw")
        result = aug.augment_training_image(self.image, **context)
        np.testing.assert_array_equal(result, self.image)
        self.assertFalse(np.shares_memory(result, self.image))
        self.assertFalse(result.flags.writeable)

    def test_exact_replay_and_global_rng_independence(self):
        result = aug.augment_training_image(self.image, **self.context)
        np.random.seed(42)
        np.random.uniform(size=1000)
        repeated = aug.augment_training_image(self.image, **self.context)
        np.testing.assert_array_equal(result, repeated)
        self.assertEqual(aug.sample_parameters(**self.context), aug.sample_parameters(**self.context))

    def test_epoch_seed_fold_participant_and_side_affect_draws(self):
        reference = aug.sample_parameters(**self.context)
        for key, values in (("epoch", range(2, 15)), ("seed", range(2030, 2044)),
                            ("validation_fold", range(1, 5)),
                            ("src_subject_id", [f"other_{i}" for i in range(14)]),
                            ("side", ["RIGHT"])):
            changed = []
            for value in values:
                context = dict(self.context, **{key: value})
                changed.append(aug.sample_parameters(**context))
            self.assertTrue(any(item != reference for item in changed), key)

    def test_half_probability_uses_both_paths_and_reviewed_range(self):
        parameters = [aug.sample_parameters(**dict(self.context, src_subject_id=f"probability_{i}"))
                      for i in range(1000)]
        activated = [p for p in parameters if p["applied"]]
        self.assertGreater(len(activated), 400)
        self.assertLess(len(activated), 600)
        self.assertTrue(all(0.9 <= p["gamma"] <= 1.1 for p in activated))
        self.assertTrue(all(p["gamma"] == 1 for p in parameters if not p["applied"]))

    def test_validation_and_inference_are_exact_identity(self):
        for split in ("validation", "inference"):
            result = aug.apply_image_policy(self.image, split=split)
            self.assertIs(result, self.image)
            self.assertEqual(result.tobytes(), self.image.tobytes())

    def test_validation_cannot_be_augmented_or_receive_training_context(self):
        for split in ("validation", "inference", "test"):
            with self.assertRaisesRegex(ValueError, "training_only"):
                aug.augment_training_image(self.image, split=split, **self.context)
        for split in ("validation", "inference"):
            with self.assertRaisesRegex(ValueError, "evaluation_must_not_receive"):
                aug.apply_image_policy(self.image, split=split, **self.context)
        with self.assertRaisesRegex(ValueError, "known_image_split"):
            aug.apply_image_policy(self.image, split="reserved_test")

    def test_training_dispatch_matches_direct_application(self):
        direct = aug.augment_training_image(self.image, **self.context)
        dispatched = aug.apply_image_policy(self.image, split="training", **self.context)
        np.testing.assert_array_equal(direct, dispatched)

    def test_target_is_not_a_parameter_and_cannot_influence_sampling(self):
        self.assertNotIn("target", inspect.signature(aug.sample_parameters).parameters)
        with self.assertRaises(TypeError):
            aug.sample_parameters(**self.context, target=1)

    def test_invalid_seed_fold_epoch_and_identity_rejected(self):
        cases = (("seed", True), ("seed", -1), ("seed", 2**32),
                 ("validation_fold", 5), ("validation_fold", False),
                 ("epoch", 0), ("epoch", 61), ("epoch", 1.0),
                 ("src_subject_id", ""), ("src_subject_id", " leading"),
                 ("src_subject_id", "line\nbreak"), ("src_subject_id", "x" * 129),
                 ("side", "left"), ("side", None))
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                aug.sample_parameters(**dict(self.context, **{key: value}))

    def test_writable_wrong_dtype_wrong_shape_and_noncontiguous_inputs_rejected(self):
        inputs = [self.image.copy(), self.image.astype(np.float64), self.image[0],
                  self.image[:, :, ::-1], self.image.tolist()]
        for image in inputs:
            with self.subTest(kind=type(image)), self.assertRaises(ValueError):
                aug.augment_training_image(image, **self.context)

    def test_normalized_negative_out_of_range_and_nonfinite_pixels_rejected(self):
        for value in (-0.001, 1.001, float("nan"), float("inf")):
            array = self.image.copy()
            array[0, 100, 100] = value
            array.setflags(write=False)
            with self.subTest(value=value), self.assertRaises(ValueError):
                aug.augment_training_image(array, **self.context)

    def test_zeros_ones_and_subnormal_endpoints_remain_valid(self):
        for value in (0.0, 1.0, np.nextafter(np.float32(0), np.float32(1))):
            image = np.full((1, 224, 224), value, dtype=np.float32)
            image.setflags(write=False)
            transformed = aug.augment_training_image(image, **self.context)
            self.assertTrue(np.isfinite(transformed).all())
            np.testing.assert_array_equal(transformed == 0, image == 0)
            if value in (0, 1):
                np.testing.assert_array_equal(transformed, image)


if __name__ == "__main__":
    unittest.main()
