"""FCMAE transfer must preserve geometry and account for every pretrained tensor."""
import hashlib
import unittest

import numpy as np
import torch
from knee import pretrained_convnextv2 as c


def source_state(model):
    """Synthetic published layout with flattened GRN and absent final LayerNorm."""
    source = {}
    for key, tensor in model.state_dict().items():
        if key in c.INITIALIZED_KEYS:
            continue
        value = tensor.detach().clone()
        if ".grn." in key:
            value = value.squeeze(0).squeeze(0)
        source[key] = value
    return {"model": source}


class ConvNeXtTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        cls.model = c.ConvNeXtV2Tiny().eval()
        cls.checkpoint = source_state(cls.model)

    def test_grn_channel_values_preserved_with_only_broadcast_axes_added(self):
        channels = torch.arange(384, dtype=torch.float32).reshape(1, 384)
        key, remapped = c._remap_encoder_tensor("stages.0.0.grn.gamma", channels)
        self.assertEqual(key, "stages.0.0.grn.gamma")
        self.assertEqual(tuple(remapped.shape), (1, 1, 1, 384))
        torch.testing.assert_close(remapped[0, 0, 0], channels[0], rtol=0, atol=0)

    def test_load_exact_encoder_and_explicit_nonpretrained_norm(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        report = c.load_fcmae_state(self.model, self.checkpoint)
        self.assertTrue(report["all_source_tensors_accounted"])
        self.assertEqual(report["initialized_nonpretrained_parameter_keys"], ["norm.bias", "norm.weight"])
        self.assertFalse(report["classification_head_loaded"])
        self.assertEqual(report["feature_dimension"], 768)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_unknown_key_and_missing_encoder_rejected_before_mutation(self):
        for mutation in ("extra", "missing", "legacy", "badshape", "dtype", "nonfinite", "head"):
            checkpoint = {"model": dict(self.checkpoint["model"])}
            first_key = "downsample_layers.0.0.weight"
            if mutation == "extra":
                checkpoint["model"]["decoder.unexpected.weight"] = torch.zeros(1)
            elif mutation == "missing":
                del checkpoint["model"][first_key]
            elif mutation == "legacy":
                checkpoint["model"]["encoder." + first_key] = checkpoint["model"].pop(first_key)
            elif mutation == "head":
                checkpoint["model"]["head.weight"] = torch.zeros(1000, 768)
            elif mutation == "badshape":
                checkpoint["model"][first_key] = torch.zeros(1)
            elif mutation == "dtype":
                checkpoint["model"][first_key] = checkpoint["model"][first_key].double()
            else:
                checkpoint["model"][first_key] = torch.full_like(checkpoint["model"][first_key], float("nan"))
            original = self.model.downsample_layers[0][0].weight.clone()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                c.load_fcmae_state(self.model, checkpoint)
            torch.testing.assert_close(original, self.model.downsample_layers[0][0].weight, rtol=0, atol=0)

    def test_published_grn_layout_must_match_exactly(self):
        for value in (torch.zeros(384), torch.zeros(1, 1, 1, 384), torch.zeros(2, 384)):
            with self.assertRaisesRegex(ValueError, "published_grn_shape"):
                c._remap_encoder_tensor("stages.0.0.grn.gamma", value)

    def test_normalization_is_exact_and_geometry_source_unchanged(self):
        image = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(1, 224, 224)
        image.setflags(write=False)
        before = hashlib.sha256(image.tobytes()).hexdigest()
        result = c.normalize_sealed_image(image)
        self.assertEqual(tuple(result.shape), (3, 224, 224))
        for channel, (mean, std) in enumerate(zip(c.MEAN, c.STD)):
            np.testing.assert_array_equal(result[channel].numpy(),
                (image[0] - np.float32(mean)) / np.float32(std))
        self.assertEqual(before, hashlib.sha256(image.tobytes()).hexdigest())
        self.assertFalse(image.flags.writeable)

    def test_invalid_images_fail_before_conversion(self):
        for image in (np.zeros((1, 224, 224), np.float32),
                np.zeros((3, 224, 224), np.float32),
                np.full((1, 224, 224), np.nan, np.float32)):
            with self.assertRaisesRegex(ValueError, "sealed_readonly"):
                c.normalize_sealed_image(image)

    def test_forward_eval_features_reproducible_without_state_or_gradient_changes(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        image = torch.linspace(-2, 2, 3 * 224 * 224).reshape(1, 3, 224, 224)
        source = image.clone()
        with torch.inference_mode():
            first, second = self.model(image), self.model(image)
        self.assertEqual(tuple(first.shape), (1, 768))
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(image, source, rtol=0, atol=0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_forward_rejects_other_resolution_and_channels(self):
        for image in (torch.zeros(1, 3, 384, 384), torch.zeros(1, 1, 224, 224)):
            with self.assertRaisesRegex(ValueError, "normalized_rgb_224"):
                self.model(image)


if __name__ == "__main__":
    unittest.main()
