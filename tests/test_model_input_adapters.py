"""Fixed encoder channels and patient-grouped batches must preserve input data."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.models import DenseNet121_Weights, ViT_B_16_Weights
from torchvision.transforms._presets import ImageClassification

from knee import model_input_adapters as a


def readonly(array):
    array.setflags(write=False)
    return array


class ImageAdapterTests(unittest.TestCase):
    def setUp(self):
        a._verified_spec.cache_clear()
        self.image = readonly(np.arange(224 * 224, dtype=np.float32).reshape(1, 224, 224) / (224 * 224 - 1))

    def test_explicit_checkpoint_enums_and_fixed_metadata(self):
        self.assertIs(a.MODELS["image_densenet121"].weights, DenseNet121_Weights.IMAGENET1K_V1)
        self.assertIs(a.MODELS["image_vit_b16"].weights, ViT_B_16_Weights.IMAGENET1K_V1)
        for identifier in a.MODELS:
            spec = a._verified_spec(identifier)
            self.assertEqual(spec.mean, (0.485, 0.456, 0.406))
            self.assertEqual(spec.std, (0.229, 0.224, 0.225))
            self.assertEqual(spec.crop_size, (224,))

    def test_policy_is_immutable_and_unknown_checkpoints_fail_closed(self):
        with self.assertRaises(TypeError):
            a.MODELS["new"] = "model"
        with self.assertRaises(FrozenInstanceError):
            a.MODELS["image_densenet121"].weights_id = "DEFAULT"
        for identifier in ("modern_convnextv2_tiny", "dinov3", "skelex", "DEFAULT", ""):
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "not_adopted"):
                a.normalize_image(self.image, identifier)

    def test_exact_rgb_normalization_no_geometry_change_and_source_unchanged(self):
        before = hashlib.sha256(self.image.tobytes()).hexdigest()
        for identifier in a.MODELS:
            output = a.normalize_image(self.image, identifier)
            self.assertEqual(tuple(output.shape), (3, 224, 224))
            self.assertEqual(output.dtype, torch.float32)
            self.assertEqual(output.device.type, "cpu")
            self.assertTrue(output.is_contiguous())
            self.assertFalse(output.requires_grad)
            for channel, (mean, scale) in enumerate(zip(a.MEAN, a.STD)):
                expected = (self.image[0] - np.float32(mean)) / np.float32(scale)
                np.testing.assert_array_equal(output[channel].numpy(), expected)
            self.assertEqual(hashlib.sha256(self.image.tobytes()).hexdigest(), before)
            self.assertFalse(self.image.flags.writeable)

    def test_full_transform_and_pretrained_download_never_called(self):
        with patch.object(ImageClassification, "forward", side_effect=AssertionError("no resize/crop")), \
                patch.object(DenseNet121_Weights, "get_state_dict", side_effect=AssertionError("no checkpoint download")), \
                patch.object(ViT_B_16_Weights, "get_state_dict", side_effect=AssertionError("no checkpoint download")):
            for identifier in a.MODELS:
                a.normalize_image(self.image, identifier)

    def test_invalid_shape_dtype_writeability_nonfinite_range(self):
        invalid = [readonly(np.zeros((224, 224), np.float32)), readonly(np.zeros((3, 224, 224), np.float32)),
            readonly(np.zeros((1, 224, 224), np.float64)), np.zeros((1, 224, 224), np.float32),
            readonly(np.full((1, 224, 224), np.nan, np.float32)),
            readonly(np.full((1, 224, 224), -0.01, np.float32)),
            readonly(np.full((1, 224, 224), 1.01, np.float32))]
        for array in invalid:
            with self.subTest(dtype=array.dtype, shape=array.shape), self.assertRaises(ValueError):
                a.normalize_image(array, "image_densenet121")

    def test_runtime_drift_and_metadata_drift_fail(self):
        with patch.object(a.np, "__version__", "2.0.0"):
            with self.assertRaisesRegex(ValueError, "frozen_cpu_adapter_runtime"):
                a.normalize_image(self.image, "image_densenet121")
        with patch.object(ImageClassification, "__init__", return_value=None):
            with self.assertRaises((ValueError, AttributeError)):
                a.normalize_image(self.image, "image_vit_b16")


class DevelopmentDatasetTests(unittest.TestCase):
    def setUp(self):
        from knee.clinical_fold_inputs import fit_clinical_fold
        self.rows, self.tensors = [], {}
        for index in range(5):
            for side in (["LEFT", "RIGHT"] if index == 1 else ["LEFT"]):
                identity = f"synthetic{index}", side
                self.rows.append({"src_subject_id": identity[0], "side": side,
                    "preprocessing_role": "development" if index == 1 else "routine",
                    "outer_block": (index % 4) + 1, "cv_fold": index,
                    "AGEYEARS": 50 + index * 5, "SEX": "F" if index % 2 else "M",
                    "BMI": None if index == 2 else 20.0 + index * 2,
                    "PROGRESION_48M": index % 2,
                    "tensor": {"relative_path": f"tensors/{index}_{side}.npy", "sha256": "a" * 64,
                        "pixel_sha256": "b" * 64, "shape": [1, 224, 224], "dtype": "float32"}})
                self.tensors[identity] = readonly(np.full((1, 224, 224), index / 5, np.float32))
        self.state = fit_clinical_fold(self.rows, 0)

    def dataset(self, split="training", **kwargs):
        return a.DevelopmentFoldDataset(kwargs.get("rows", self.rows), kwargs.get("tensors", self.tensors),
            kwargs.get("state", self.state), "image_densenet121", split)

    def test_group_disjoint_feature_target_identity_alignment_and_immutability(self):
        from knee.clinical_fold_inputs import transform_clinical_rows
        before_rows, before_state = deepcopy(self.rows), deepcopy(self.state)
        hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in self.tensors.items()}
        training, validation = self.dataset(), self.dataset("validation")
        self.assertFalse({key[0] for key in training.identities} & {key[0] for key in validation.identities})
        for split, dataset in (("training", training), ("validation", validation)):
            expected_rows = [r for r in self.rows if (r["cv_fold"] == 0) == (split == "validation")]
            expected_clinical = transform_clinical_rows(expected_rows, self.state)
            for index, row in enumerate(expected_rows):
                item = dataset[index]
                self.assertEqual(set(item), {"image", "clinical", "target", "src_subject_id", "side"})
                self.assertEqual((item["src_subject_id"], item["side"]), (row["src_subject_id"], row["side"]))
                self.assertEqual(item["target"].dtype, torch.float32)
                self.assertEqual(item["target"].ndim, 0)
                self.assertEqual(item["target"].item(), row["PROGRESION_48M"])
                np.testing.assert_array_equal(item["clinical"].numpy(), expected_clinical[index])
        self.assertEqual(before_rows, self.rows)
        self.assertEqual(before_state, self.state)
        self.assertEqual(hashes, {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in self.tensors.items()})

    def test_cpu_loader_keeps_every_row_in_exact_order(self):
        dataset = self.dataset()
        loader = a.make_development_loader(dataset, batch_size=2)
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.drop_last)
        seen = []
        for batch in loader:
            self.assertEqual(batch["image"].shape[1:], (3, 224, 224))
            self.assertEqual(batch["clinical"].shape[1:], (3,))
            self.assertEqual(batch["target"].shape, (len(batch["side"]),))
            seen.extend(zip(batch["src_subject_id"], batch["side"]))
        self.assertEqual(seen, list(dataset.identities))

    def test_test_role_unknowncolumns_badlabel_and_wrong_mapping_fail(self):
        mutations = [dict(outer_block=0, cv_fold=None), dict(PROGRESION_48M=2), dict(KL_48=4)]
        for change in mutations:
            rows = deepcopy(self.rows)
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.dataset(rows=rows)
        with self.assertRaisesRegex(ValueError, "exact_development_image_mapping"):
            self.dataset(tensors={})
        with self.assertRaisesRegex(ValueError, "training_or_validation"):
            self.dataset("reserved_test")

    def test_empty_fold_and_invalid_selected_image_rejected(self):
        rows = [r for r in self.rows if r["cv_fold"] != 0]
        images = {(r["src_subject_id"], r["side"]): self.tensors[(r["src_subject_id"], r["side"])] for r in rows}
        with self.assertRaisesRegex(ValueError, "nonempty_development_fold"):
            self.dataset("validation", rows=rows, tensors=images)
        images = dict(self.tensors)
        images[("synthetic1", "LEFT")] = np.zeros((1, 224, 224), np.float32)
        with self.assertRaisesRegex(ValueError, "readonly_finite_image_mapping"):
            self.dataset(tensors=images)

    def test_state_must_match_the_actual_training_participants(self):
        rows, tensors = deepcopy(self.rows), dict(self.tensors)
        for row in rows:
            if row["src_subject_id"] == "synthetic1":
                row["src_subject_id"] = "other-participant"
                tensors[(row["src_subject_id"], row["side"])] = tensors.pop(("synthetic1", row["side"]))
        with self.assertRaisesRegex(ValueError, "state_does_not_match_development_training_fold"):
            self.dataset(rows=rows, tensors=tensors)


if __name__ == "__main__":
    unittest.main()
