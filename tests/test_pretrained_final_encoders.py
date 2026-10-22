"""Synthetic adapter safety tests; not predictive or checkpoint validation."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from knee import pretrained_final_encoders as f


def sealed():
    array = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(1, 224, 224)
    array.setflags(write=False)
    return array


def normalized(model_id=f.BIOMED_ID, batch=1):
    return f.normalize_sealed_image(sealed(), model_id).unsqueeze(0).repeat(batch, 1, 1, 1)


class TinyFeatures(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.bias = nn.Parameter(torch.arange(dimension, dtype=torch.float32))

    def forward(self, images):
        return images.mean((1, 2, 3)).unsqueeze(1) + self.bias.unsqueeze(0)


class FinalNormalizationTests(unittest.TestCase):
    def test_model_specific_normalization_preserves_source_and_geometry(self):
        array = sealed()
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        for model_id in f.NORMALIZATION:
            output = f.normalize_sealed_image(array, model_id)
            self.assertEqual(tuple(output.shape), (3, 224, 224))
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(output.is_contiguous())
            f._validate_normalized_inputs(output.unsqueeze(0), model_id)
            mean, std = f._normalization(model_id)
            recovered = output.unsqueeze(0) * std + mean
            self.assertTrue(torch.allclose(recovered[:, 0], torch.from_numpy(array.copy()), atol=1e-6, rtol=0))
        self.assertEqual(hashlib.sha256(array.tobytes()).hexdigest(), digest)
        self.assertFalse(array.flags.writeable)

    def test_clip_and_imagenet_normalization_are_not_interchangeable(self):
        self.assertFalse(torch.equal(normalized(f.BIOMED_ID), normalized(f.MAMBA_ID)))
        for source, target in ((f.BIOMED_ID, f.MAMBA_ID), (f.MAMBA_ID, f.BIOMED_ID)):
            with self.assertRaisesRegex(ValueError, "model_specific_normalized"):
                f._validate_normalized_inputs(normalized(source), target)

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "registered_visual"):
            f.normalize_sealed_image(sealed(), "unregistered")

    def test_mutable_wrong_type_shape_range_and_nonfinite_sources_rejected(self):
        bad_sources = [sealed().copy(), torch.zeros(1, 224, 224)]
        for value in (np.zeros((3, 224, 224), np.float32), np.zeros((1, 224, 224), np.float64),
                      np.full((1, 224, 224), 2.0, np.float32),
                      np.full((1, 224, 224), np.nan, np.float32)):
            value.setflags(write=False)
            bad_sources.append(value)
        for value in bad_sources:
            with self.subTest(type=type(value)), self.assertRaises(ValueError):
                f.normalize_sealed_image(value, f.BIOMED_ID)

    def test_invalid_batches_and_gradients_rejected(self):
        for value in (torch.zeros(1, 1, 224, 224), torch.zeros(1, 3, 288, 288),
                      normalized().double(), normalized(batch=3), normalized(batch=0),
                      normalized().requires_grad_()):
            with self.subTest(shape=tuple(value.shape)), self.assertRaises(ValueError):
                f._validate_normalized_inputs(value, f.BIOMED_ID)

    def test_nonfinite_input_rejected(self):
        for bad in (float("nan"), float("inf")):
            value = normalized()
            value[0, 0, 0, 0] = bad
            with self.assertRaisesRegex(ValueError, "finite_input"):
                f._validate_normalized_inputs(value, f.BIOMED_ID)


class FinalStateTests(unittest.TestCase):
    def test_exact_public_metadata_pins(self):
        self.assertEqual(f.BIOMED_REVISION, "9f341de24bfb00180f1b847274256e9b65a3a32e")
        self.assertEqual(f.MAMBA_REVISION, "d080941663a60c42d015ecf8d8fad86d12694208")
        self.assertEqual(f.TIMM_VERSION, "1.0.15")
        self.assertEqual(f.FEATURE_DIMENSIONS, {f.BIOMED_ID: 512, f.MAMBA_ID: 2304})
        for digest in (f.BIOMED_WEIGHTS_SHA256, f.BIOMED_CONFIG_SHA256,
                       f.MAMBA_WEIGHTS_SHA256, f.MAMBA_CONFIG_SHA256):
            self.assertRegex(digest, r"^[a-f0-9]{64}$")

    def test_hashed_small_metadata_before_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            raw = b'{"test":224}'
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(f._verified_json(path, len(raw), digest), {"test": 224})
            with self.assertRaisesRegex(ValueError, "sha256"):
                f._verified_json(path, len(raw), "0" * 64)
            with self.assertRaisesRegex(ValueError, "exact_regular"):
                f._verified_json(path, len(raw) + 1, digest)
            with self.assertRaisesRegex(ValueError, "exact_regular"):
                f._verified_file(path, len(raw), digest, ".bin")

    def test_metadata_must_be_json_object(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            raw = b"[]"
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "object_required"):
                f._verified_json(path, len(raw), hashlib.sha256(raw).hexdigest())

    def test_biomed_text_schema_is_complete_and_not_loaded(self):
        schema = f._biomed_unused_schema()
        self.assertEqual(len(schema), 201)
        self.assertEqual(sum(key.startswith("text.") for key in schema), 200)
        self.assertEqual(schema["text.transformer.embeddings.word_embeddings.weight"], (30522, 768))
        self.assertEqual(schema["text.proj.0.weight"], (640, 768))
        self.assertEqual(schema["text.proj.2.weight"], (512, 640))
        self.assertNotIn("text.transformer.pooler.dense.weight", schema)

    def test_unknown_or_missing_complete_checkpoint_keys_fail_before_load(self):
        model = nn.Linear(2, 3)
        for state in ({}, {"unexpected": torch.zeros(1)}, model.state_dict()):
            with self.assertRaisesRegex(ValueError, "exact_complete_published_keys"):
                f._adopt_biomed_state(model, state)
            with self.assertRaisesRegex(ValueError, "exact_complete_published_keys"):
                f._adopt_mamba_state(model, state)

    def test_tensor_shapes_and_dtypes_are_not_silently_converted(self):
        f._validate_tensor(torch.zeros(2, 3), (2, 3))
        for tensor in (torch.zeros(2, 3, dtype=torch.float64), torch.zeros(2, 4),
                       torch.zeros(2, 3, dtype=torch.float16), np.zeros((2, 3))):
            with self.assertRaisesRegex(ValueError, "shape_or_dtype"):
                f._validate_tensor(tensor, (2, 3))

    def test_nonfinite_checkpoint_tensor_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                f._validate_tensor(torch.tensor([bad]), (1,))

    def test_unused_text_positions_are_validated_exactly(self):
        positions = torch.arange(512, dtype=torch.int64).reshape(1, 512)
        f._validate_tensor(positions, (1, 512), integer_positions=True)
        with self.assertRaisesRegex(ValueError, "exact_unused_text_positions"):
            f._validate_tensor(positions.flip(1), (1, 512), integer_positions=True)
        with self.assertRaisesRegex(ValueError, "shape_or_dtype"):
            f._validate_tensor(positions.float(), (1, 512), integer_positions=True)

    def test_bad_model_or_file_list_rejected_before_timm_import(self):
        for model, paths in (("unknown", ["x", "y"]), (f.BIOMED_ID, []),
                             (f.MAMBA_ID, ["x"]), (f.BIOMED_ID, "xy")):
            with self.assertRaisesRegex(ValueError, "model_and_two_local"):
                f.load_final_encoder(model, paths)

    def test_wrong_timm_version_rejected_before_files(self):
        import timm
        with patch.object(timm, "__version__", "unreviewed"), self.assertRaisesRegex(ValueError, "timm_1_0_15"):
            f.load_final_encoder(f.BIOMED_ID, ["missing.bin", "missing.json"])

    def test_missing_local_metadata_never_causes_download(self):
        with tempfile.TemporaryDirectory() as folder:
            for model in f.FEATURE_DIMENSIONS:
                with self.assertRaisesRegex(ValueError, "exact_regular"):
                    f.load_final_encoder(model, [Path(folder) / "missing.bin", Path(folder) / "missing.json"])


class FinalFrozenTests(unittest.TestCase):
    def test_cpu_features_exact_without_mutating_rng_input_or_state(self):
        for model_id, dimension in f.FEATURE_DIMENSIONS.items():
            wrapper = f.FrozenVisualCPU(TinyFeatures(dimension), model_id)
            images = normalized(model_id, 2)
            before = images.clone()
            state = {key: value.clone() for key, value in wrapper.state_dict().items()}
            rng = torch.get_rng_state().clone()
            first, second = wrapper(images), wrapper(images)
            self.assertEqual(tuple(first.shape), (2, dimension))
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(images, before))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(value, wrapper.state_dict()[key]) for key, value in state.items()))
            self.assertFalse(first.requires_grad)
            self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in wrapper.parameters()))

    def test_training_mode_refused(self):
        wrapper = f.FrozenVisualCPU(TinyFeatures(512), f.BIOMED_ID)
        with self.assertRaisesRegex(ValueError, "training_forbidden"):
            wrapper.train()
        self.assertFalse(wrapper.training)

    def test_unfrozen_or_nested_training_or_existing_gradients_refused(self):
        for mutation in (lambda m: m.encoder.train(), lambda m: m.encoder.bias.requires_grad_(),
                         lambda m: setattr(m.encoder.bias, "grad", torch.zeros(512))):
            wrapper = f.FrozenVisualCPU(TinyFeatures(512), f.BIOMED_ID)
            mutation(wrapper)
            with self.assertRaisesRegex(ValueError, "frozen_eval"):
                wrapper(normalized())

    def test_wrong_output_shape_and_nonfinite_refused(self):
        wrong = f.FrozenVisualCPU(TinyFeatures(511), f.BIOMED_ID)
        with self.assertRaisesRegex(ValueError, "finite_expected_visual"):
            wrong(normalized())
        nonfinite = f.FrozenVisualCPU(TinyFeatures(512), f.BIOMED_ID)
        nonfinite.encoder.bias.data[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite_expected_visual"):
            nonfinite(normalized())


class MambaArchitectureTests(unittest.TestCase):
    def test_strict_adoption_preserves_official_2304_pre_logits_not_576_channels(self):
        import timm
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2026)
                complete = timm.create_model("mambaout_tiny", pretrained=False).eval()
            state = complete.state_dict()
            self.assertEqual(len(state), 172)
            self.assertEqual(complete.num_features, 576)
            images = normalized(f.MAMBA_ID)
            with torch.inference_mode():
                expected = complete.forward_head(complete.forward_features(images), pre_logits=True)
            self.assertEqual(tuple(expected.shape), (1, 2304))
            rng = torch.get_rng_state().clone()
            audit = f._adopt_mamba_state(complete, state)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            actual = f.FrozenVisualCPU(complete, f.MAMBA_ID)(images)
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(audit["removed_classifier_keys"], ["head.fc.bias", "head.fc.weight"])
            self.assertEqual(len(complete.state_dict()), 170)
            self.assertTrue(audit["pretrained_pre_logits_mlp_retained"])
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
