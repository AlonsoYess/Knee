"""SKELEX safety/geometry tests; tiny synthetic MAE is not a weight validation."""
from pathlib import Path
import hashlib
import tempfile
import unittest

import torch

from knee import pretrained_skelex as s


def normalized(batch=1):
    gray = torch.linspace(0, 1, 224 * 224).reshape(1, 1, 224, 224)
    mean = torch.tensor(s.MEAN).reshape(1, 3, 1, 1)
    std = torch.tensor(s.STD).reshape(1, 3, 1, 1)
    return ((gray.expand(batch, 3, -1, -1) - mean) / std).contiguous()


class SkelexMetadataTests(unittest.TestCase):
    def test_exact_public_revision_and_metadata_hashes_are_pinned(self):
        self.assertEqual(s.HF_REVISION, "368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e")
        self.assertEqual(s.CHECKPOINT_BYTES, 1318230232)
        for value in (s.CHECKPOINT_SHA256, s.CONFIG_SHA256, s.PROCESSOR_SHA256):
            self.assertRegex(value, r"^[a-f0-9]{64}$")

    def test_metadata_hash_and_exact_bytes_precede_json_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metadata.json"
            raw = b'{"size":224}'
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(s._verified_small_json(path, len(raw), digest), {"size": 224})
            with self.assertRaisesRegex(ValueError, "metadata_hash"):
                s._verified_small_json(path, len(raw), "0" * 64)
            with self.assertRaisesRegex(ValueError, "metadata_file"):
                s._verified_small_json(path, len(raw) + 1, digest)

    def test_missing_or_unapproved_checkpoint_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pytorch_model.bin"
            path.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(ValueError, "official_safetensors"):
                s.load_skelex_encoder(path, "missing.json", "missing_processor.json")

    def test_normalized_gray_accepted_without_modification(self):
        images = normalized(2)
        before = images.clone()
        s._validate_normalized_inputs(images)
        self.assertTrue(torch.equal(before, images))

    def test_invalid_geometry_or_dtype_or_batch_rejected(self):
        for images in (torch.zeros(1, 1, 224, 224), torch.zeros(1, 3, 256, 256),
                       normalized().double(), normalized(3), normalized().requires_grad_()):
            with self.subTest(shape=tuple(images.shape)), self.assertRaises(ValueError):
                s._validate_normalized_inputs(images)

    def test_nonfinite_and_wrong_normalization_rejected(self):
        bad = normalized()
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite_input"):
            s._validate_normalized_inputs(bad)
        for images in (torch.zeros(1, 3, 224, 224), normalized() + 10):
            with self.assertRaisesRegex(ValueError, "normalized_replicated_gray"):
                s._validate_normalized_inputs(images)


class SkelexUnmaskedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import ViTMAEConfig, ViTMAEModel
        cls.config_class = ViTMAEConfig
        cls.model_class = ViTMAEModel

    def tiny_encoder(self, mask_ratio=0):
        config = self.config_class(image_size=224, patch_size=16, num_channels=3,
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, mask_ratio=mask_ratio,
            attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0)
        config._attn_implementation = "eager"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            encoder = self.model_class(config).eval().requires_grad_(False)
        return encoder

    def test_identity_noise_keeps_all_tokens_and_does_not_consume_rng(self):
        encoder = self.tiny_encoder()
        images = normalized()
        state = {key: value.clone() for key, value in encoder.state_dict().items()}
        before_rng = torch.get_rng_state().clone()
        with torch.inference_mode():
            first = s._unmasked_features(encoder, images)
            second = s._unmasked_features(encoder, images)
        self.assertEqual(tuple(first.shape), (1, 32))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, encoder.state_dict()[key]) for key, value in state.items()))
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))

    def test_nonzero_masking_ratio_rejected(self):
        with self.assertRaisesRegex(ValueError, "masking_must_be_disabled"):
            s._unmasked_features(self.tiny_encoder(mask_ratio=0.75), normalized())

    def test_wrapper_rejects_nonofficial_encoder_geometry(self):
        with self.assertRaisesRegex(ValueError, "official_encoder_geometry"):
            s.SkelexCPUFeatures(self.tiny_encoder())

    def test_wrapper_cannot_be_switched_to_training(self):
        wrapper = s.SkelexCPUFeatures.__new__(s.SkelexCPUFeatures)
        torch.nn.Module.__init__(wrapper)
        wrapper.eval()
        self.assertFalse(wrapper.training)
        with self.assertRaisesRegex(ValueError, "training_forbidden"):
            wrapper.train()


if __name__ == "__main__":
    unittest.main()
