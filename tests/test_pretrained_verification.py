"""Synthetic checkpoint and no-training guards; no large model or download."""
from copy import deepcopy
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from knee import pretrained_verification as subject


ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.register_buffer("counter", torch.tensor(0.0))
        self.eval()

    def forward(self, batch):
        return batch.reshape(len(batch), -1)[:, :3] * self.weight


class TinyDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2))
        self.classifier = nn.Linear(2, 3)


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.heads = nn.Linear(3, 2)


class PretrainedVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.prior_threads)

    def setUp(self):
        self.catalog = json.loads((ROOT / "configs/pretrained_weights_v1.json").read_text(encoding="utf-8"))
        self.batch = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)

    def test_catalog_covers_all_registered_models_and_preserves_input_pins(self):
        catalog, plan, registry = subject.load_catalog(ROOT)
        self.assertEqual(len(catalog["models"]), 5)
        self.assertEqual(len(catalog["models_without_external_pretrained_checkpoint"]), 7)
        self.assertFalse(plan["training_authorized"])
        self.assertEqual(catalog["input"]["geometry_operations"], [])
        self.assertFalse(catalog["models"]["image_dinov3_vits16"]["verification_enabled"])

    def test_catalog_mutation_is_rejected_before_acquisition(self):
        altered = deepcopy(self.catalog)
        altered["models"]["image_dinov3_vits16"]["verification_enabled"] = True
        with patch.object(subject, "acquire") as acquire:
            with self.assertRaisesRegex(ValueError, "catalog_changed"):
                subject.checkpoint_files("image_dinov3_vits16", altered, ".", True)
            acquire.assert_not_called()

    def test_dinov3_pending_access_is_rejected_even_when_download_requested(self):
        with patch.object(subject, "acquire") as acquire, patch.object(subject, "load_encoder") as loader:
            with self.assertRaisesRegex(ValueError, "access_or_review_pending"):
                subject.checkpoint_files("image_dinov3_vits16", self.catalog, ".", True)
            acquire.assert_not_called()
            loader.assert_not_called()

    def test_unregistered_encoder_rejected_and_missing_cache_requires_opt_in(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(subject, "acquire") as acquire:
            with self.assertRaisesRegex(ValueError, "registered_image"):
                subject.checkpoint_files("unregistered_encoder", self.catalog, directory, True)
            with self.assertRaisesRegex(ValueError, "not_cached"):
                subject.checkpoint_files("image_densenet121", self.catalog, directory, False)
            acquire.assert_not_called()

    def test_checkpoint_acquisition_receives_full_pinned_hash_and_size(self):
        spec = self.catalog["models"]["image_densenet121"]["files"][0]
        with tempfile.TemporaryDirectory() as directory, patch.object(subject, "acquire") as acquire:
            result = subject.checkpoint_files("image_densenet121", self.catalog, directory, True)
            path = Path(directory) / spec["name"]
            self.assertEqual(result, [path])
            acquire.assert_called_once_with(spec["url"], path, expected_sha256=spec["sha256"],
                                            expected_bytes=spec["bytes"])

    def test_source_pin_drift_rejected(self):
        real = subject.file_digest
        def digest(path):
            return "0" * 64 if Path(path).name == "clinical_fold_inputs.py" else real(path)
        with patch.object(subject, "file_digest", side_effect=digest):
            with self.assertRaisesRegex(ValueError, "previously_verified_input_source_changed"):
                subject.load_catalog(ROOT)

    def test_catalog_file_drift_fails_before_source_or_registry_processing(self):
        altered = deepcopy(self.catalog)
        altered["input"]["geometry_operations"] = ["resize"]
        with patch("knee.training_plan._read_json", return_value=altered):
            with self.assertRaisesRegex(ValueError, "catalog_changed"):
                subject.load_catalog(ROOT)

    def test_legacy_densenet_remaps_only_known_suffixes(self):
        value = torch.tensor([1.0])
        source = {"features.denseblock1.denselayer1.norm.1.weight": value,
                  "features.denseblock1.denselayer1.conv.2.weight": value,
                  "features.norm0.weight": value}
        result = subject._normalize_densenet_keys(source)
        self.assertEqual(set(result), {"features.denseblock1.denselayer1.norm1.weight",
                                      "features.denseblock1.denselayer1.conv2.weight", "features.norm0.weight"})
        self.assertIs(result["features.denseblock1.denselayer1.norm1.weight"], value)
        self.assertIn("features.denseblock1.denselayer1.norm.1.weight", source)

    def test_legacy_densenet_duplicate_remap_fails(self):
        source = {"features.denseblock1.denselayer1.norm.1.weight": torch.tensor(1.0),
                  "features.denseblock1.denselayer1.norm1.weight": torch.tensor(2.0)}
        with self.assertRaisesRegex(ValueError, "duplicate_legacy_densenet"):
            subject._normalize_densenet_keys(source)

    def test_legacy_vit_mlp_remaps_first_and_second_linear(self):
        value = torch.tensor([1.0])
        source = {"encoder.layers.encoder_layer_0.mlp.linear_1.weight": value,
                  "encoder.layers.encoder_layer_11.mlp.linear_2.bias": value,
                  "heads.head.weight": value}
        result = subject._normalize_vit_keys(source)
        self.assertEqual(set(result), {"encoder.layers.encoder_layer_0.mlp.0.weight",
                                      "encoder.layers.encoder_layer_11.mlp.3.bias", "heads.head.weight"})
        self.assertIs(result["encoder.layers.encoder_layer_0.mlp.0.weight"], value)

    def test_legacy_vit_duplicate_remap_fails(self):
        source = {"encoder.layers.encoder_layer_0.mlp.linear_2.weight": torch.tensor(1.0),
                  "encoder.layers.encoder_layer_0.mlp.3.weight": torch.tensor(2.0)}
        with self.assertRaisesRegex(ValueError, "duplicate_legacy_vit"):
            subject._normalize_vit_keys(source)

    def test_legacy_vit_loading_preserves_every_learned_encoder_tensor(self):
        def factory(**_):
            model = nn.Module()
            model.encoder = nn.Module()
            model.encoder.layers = nn.Module()
            block = nn.Module()
            block.mlp = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Dropout(0), nn.Linear(4, 3))
            model.encoder.layers.add_module("encoder_layer_0", block)
            model.heads = nn.Linear(3, 2)
            return model
        expected = factory().state_dict()
        legacy = {key.replace(".mlp.0.", ".mlp.linear_1.").replace(".mlp.3.", ".mlp.linear_2."): value.clone()
                  for key, value in expected.items()}
        with patch("torchvision.models.vit_b_16", side_effect=factory), patch("torch.load", return_value=legacy):
            loaded, audit = subject.load_encoder("image_vit_b16", [Path("synthetic_legacy.pth")])
        self.assertEqual(set(loaded.state_dict()), {key for key in expected if not key.startswith("heads.")})
        for key, value in loaded.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]), key)
        self.assertTrue(audit["all_source_tensors_accounted"])

    def test_legacy_densenet_loading_preserves_every_learned_encoder_tensor(self):
        def factory(**_):
            model = nn.Module()
            model.features = nn.Module()
            model.features.denseblock1 = nn.Module()
            layer = nn.Module()
            layer.norm1 = nn.BatchNorm2d(2)
            layer.conv2 = nn.Conv2d(2, 2, 1)
            model.features.denseblock1.add_module("denselayer1", layer)
            model.classifier = nn.Linear(2, 3)
            return model
        expected = factory().state_dict()
        legacy = {key.replace(".norm1.", ".norm.1.").replace(".conv2.", ".conv.2."): value.clone()
                  for key, value in expected.items() if not key.endswith("num_batches_tracked")}
        with patch("torchvision.models.densenet121", side_effect=factory), patch("torch.load", return_value=legacy):
            loaded, audit = subject.load_encoder("image_densenet121", [Path("synthetic_legacy.pth")])
        for key, value in loaded[0].state_dict().items():
            self.assertTrue(torch.equal(value, expected["features." + key]), key)
        self.assertTrue(audit["all_source_tensors_accounted"])
        self.assertEqual(audit["initialized_nonpretrained_buffer_keys"],
                         ["features.denseblock1.denselayer1.norm1.num_batches_tracked"])

    def test_baseline_loader_safe_deserialization_and_rng_state_preserved(self):
        state = TinyDenseNet().state_dict()
        del state["features.1.num_batches_tracked"]
        before_rng = torch.get_rng_state().clone()
        with patch("torchvision.models.densenet121", side_effect=lambda **_: TinyDenseNet()) as constructor:
            with patch("torch.load", return_value=state) as loader:
                model, audit = subject.load_encoder("image_densenet121", [Path("synthetic.pth")])
        constructor.assert_called_once_with(weights=None)
        loader.assert_called_once_with(Path("synthetic.pth"), map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertFalse(model.training)
        self.assertTrue(all(not param.requires_grad for param in model.parameters()))
        self.assertEqual(audit["initialized_nonpretrained_buffer_keys"], ["features.1.num_batches_tracked"])
        self.assertFalse(audit["classification_head_used_for_prediction"])

    def test_baseline_loader_rejects_missing_learned_or_extra_keys(self):
        for change in ("missing", "extra", "nan"):
            state = TinyDenseNet().state_dict()
            if change == "missing":
                del state["features.0.weight"]
            elif change == "extra":
                state["unrecognized.weight"] = torch.tensor(1.0)
            else:
                state["features.0.weight"][0, 0, 0, 0] = float("nan")
            with self.subTest(change=change), patch("torchvision.models.densenet121", side_effect=lambda **_: TinyDenseNet()):
                with patch("torch.load", return_value=state), self.assertRaises(ValueError):
                    subject.load_encoder("image_densenet121", [Path("synthetic.pth")])

    def test_vit_loader_removes_imagenet_classifier_and_requires_exact_learned_keys(self):
        state = TinyViT().state_dict()
        with patch("torchvision.models.vit_b_16", side_effect=lambda **_: TinyViT()):
            with patch("torch.load", return_value=state):
                model, audit = subject.load_encoder("image_vit_b16", [Path("synthetic.pth")])
        self.assertIsInstance(model.heads, nn.Identity)
        self.assertEqual(audit["initialized_nonpretrained_buffer_keys"], [])
        del state["encoder.weight"]
        with patch("torchvision.models.vit_b_16", side_effect=lambda **_: TinyViT()):
            with patch("torch.load", return_value=state), self.assertRaises(ValueError):
                subject.load_encoder("image_vit_b16", [Path("synthetic.pth")])

    def test_unadopted_loader_does_not_deserialize(self):
        with patch("torch.load") as loader:
            with self.assertRaisesRegex(ValueError, "loader_not_adopted"):
                subject.load_encoder("image_dinov3_vits16", [])
            loader.assert_not_called()

    def test_synthetic_channel_normalization_preserves_known_pixel_geometry(self):
        result = subject.synthetic_inputs()
        self.assertEqual(tuple(result.shape), (2, 3, 224, 224))
        self.assertEqual(result.dtype, torch.float32)
        first = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(1, 224, 224)
        gray = torch.from_numpy(first)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        for channel in range(3):
            self.assertTrue(torch.equal(result[0, channel], (gray[0] - mean[channel]) / std[channel]))
            self.assertTrue(torch.equal(result[1, channel], ((1 - gray[0]) - mean[channel]) / std[channel]))

    def test_changed_checkpoint_normalization_is_detected(self):
        with patch("knee.pretrained_convnextv2.normalize_sealed_image", return_value=torch.zeros(3, 224, 224)):
            with self.assertRaisesRegex(ValueError, "normalization_disagrees"):
                subject.synthetic_inputs()

    def test_probe_is_repeatable_and_preserves_inputs_model_rng_and_gradients(self):
        model = TinyEncoder()
        before = self.batch.clone()
        rng = torch.get_rng_state().clone()
        result = subject.probe_encoder(model, self.batch, 3)
        self.assertEqual(result["feature_shape"], [2, 3])
        self.assertFalse(result["gradients_created"])
        self.assertTrue(torch.equal(self.batch, before))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertIsNone(model.weight.grad)

    def test_probe_rejects_nonfinite_wrong_dimension_and_wrong_dtype(self):
        class Invalid(TinyEncoder):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind
            def forward(self, batch):
                output = super().forward(batch)
                if self.kind == "nan":
                    return output * float("nan")
                if self.kind == "shape":
                    return output[:, :2]
                return output.double()
        for kind in ("nan", "shape", "dtype"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "expected_encoder_features"):
                subject.probe_encoder(Invalid(kind), self.batch, 3)

    def test_probe_rejects_state_input_rng_and_gradient_mutation(self):
        class Mutating(TinyEncoder):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind
            def forward(self, batch):
                if self.kind == "state":
                    self.counter.add_(1)
                elif self.kind == "input":
                    batch.add_(1)
                elif self.kind == "rng":
                    torch.rand(1)
                elif self.kind == "gradient":
                    self.weight.grad = torch.ones_like(self.weight)
                return torch.ones(len(batch), 3)
        for kind in ("state", "input", "rng", "gradient"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "mutated_state_or_created_gradient"):
                subject.probe_encoder(Mutating(kind), self.batch.clone(), 3)

    def test_probe_rejects_training_mode_or_trainable_parameters(self):
        for model in (TinyEncoder().train(), TinyEncoder().requires_grad_(True)):
            with self.assertRaisesRegex(ValueError, "mutated_state_or_created_gradient"):
                subject.probe_encoder(model, self.batch, 3)

    def test_probe_rejects_stochastic_or_changing_features(self):
        class Changing(TinyEncoder):
            def forward(self, batch):
                return torch.rand(len(batch), 3)
        with self.assertRaisesRegex(ValueError, "repeated_cpu_features_differ"):
            subject.probe_encoder(Changing(), self.batch, 3)

    def test_entrypoint_has_no_dataset_or_cohort_argument(self):
        parameters = set(inspect.signature(subject.run).parameters)
        self.assertEqual(parameters, {"repository_root", "weights_root", "output_dir", "allow_download", "execution_environment"})
        with self.assertRaises(TypeError):
            subject.run(ROOT, "cache", "output", dataset="not-permitted")

    def test_existing_output_and_repository_cache_rejected_before_acquisition(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(subject, "acquire") as acquire:
            with self.assertRaisesRegex(ValueError, "new_private_output"):
                subject.run(ROOT, Path(directory) / "cache", directory)
            with self.assertRaisesRegex(ValueError, "cache_must_remain_outside"):
                subject.run(ROOT, ROOT / "weights", Path(directory) / "new")
            acquire.assert_not_called()

    def test_local_run_cannot_claim_colab_without_runtime_marker(self):
        with patch.dict("os.environ", {"COLAB_RELEASE_TAG": ""}):
            with self.assertRaisesRegex(ValueError, "actual_colab_runtime_required"):
                subject.run(ROOT, "cache", "output", execution_environment="google_colab_cpu")

    def test_cli_rejects_cohort_or_training_arguments(self):
        source = ROOT / "scripts/verify_pretrained_models.py"
        spec = importlib.util.spec_from_file_location("checkpoint_cli_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for extra in (["--dataset", "private.zip"], ["--train"], ["--reserved-test", "private.zip"]):
            argv = [str(source), "--weights-root", "cache", "--output-dir", "output"] + extra
            with self.subTest(extra=extra), patch("sys.argv", argv), patch("sys.stderr", new=io.StringIO()):
                with patch.object(module, "run") as run, self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
