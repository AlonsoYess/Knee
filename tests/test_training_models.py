"""Pure construction, phase and input-contract checks; no optimizer or fitting."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from knee import training_models as models

ROOT = Path(__file__).resolve().parents[1]


def _config(name):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def _attach(root, name):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        if part not in parent._modules:
            parent.add_module(part, nn.Module())
        parent = parent._modules[part]
    parent.register_parameter(parts[-1], nn.Parameter(torch.zeros(1)))


class TinyDense(nn.Module):
    """Only topology is stubbed; pretrained file acceptance is tested separately."""

    def __init__(self):
        super().__init__()
        for name in ("0.conv0.weight", "0.norm0.weight", "0.denseblock1.weight",
                     "0.denseblock2.weight", "0.denseblock3.weight", "0.denseblock4.weight",
                     "0.norm5.weight"):
            _attach(self, name)
        self._modules["0"]._modules["norm0"].add_module("batch_norm", nn.BatchNorm1d(1, affine=False))
        self._modules["0"]._modules["norm5"].add_module("batch_norm", nn.BatchNorm1d(1, affine=False))
        self._modules["0"]._modules["denseblock3"].add_module("drop", nn.Dropout(0.1))
        self._modules["0"]._modules["denseblock4"].add_module("drop", nn.Dropout(0.1))

    def forward(self, images):
        return images.mean((1, 2, 3), keepdim=False).unsqueeze(1).expand(-1, 1024)


class TrainingModelsTests(unittest.TestCase):
    def setUp(self):
        self.plan = _config("training_plan_v1_1.json")
        self.catalog = _config("pretrained_weights_v1_1.json")

    def _dense(self, model_id="image_densenet121"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.bin"
            path.write_bytes(b"unit topology only; never accepted as an external checkpoint")
            with patch.object(models, "_verified_paths", return_value=[path]), \
                 patch.object(models, "load_encoder", return_value=(TinyDense(), {"mocked_topology": True})):
                return models.build_model(model_id, self.plan, self.catalog, [path],
                    image_encoder_id="image_densenet121" if model_id in models.FUSION_IDS else None)

    def test_clinical_architecture_and_eval_logits(self):
        model, audit = models.build_model("clinical_mlp", self.plan, self.catalog)
        self.assertEqual([(m.in_features, m.out_features) for m in model.clinical_encoder if isinstance(m, nn.Linear)], [(3, 16), (16, 8)])
        with torch.inference_mode():
            result = model(clinical=torch.ones(2, 3))
        self.assertEqual(tuple(result.shape), (2,))
        self.assertFalse(audit["training_executed"])
        self.assertFalse(audit["optimizer_created"])

    def test_seeded_new_heads_preserve_global_rng(self):
        before = torch.get_rng_state().clone()
        first, _ = models.build_model("clinical_mlp", self.plan, self.catalog, seed=2026)
        second, _ = models.build_model("clinical_mlp", self.plan, self.catalog, seed=2026)
        third, _ = models.build_model("clinical_mlp", self.plan, self.catalog, seed=2027)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(first.task_head.weight, second.task_head.weight))
        self.assertFalse(torch.equal(first.task_head.weight, third.task_head.weight))

    def test_rejects_unregistered_model_seeds_and_extra_modalities(self):
        for model_id in ("image_dinov3_vits16", "image_new", "clinical_logistic"):
            with self.assertRaises(ValueError):
                models.build_model(model_id, self.plan, self.catalog)
        with self.assertRaises(ValueError):
            models.build_model("clinical_mlp", self.plan, self.catalog, seed=True)
        with self.assertRaises(ValueError):
            models.build_model("clinical_mlp", self.plan, self.catalog, image_encoder_id="image_densenet121")
        model, _ = models.build_model("clinical_mlp", self.plan, self.catalog)
        with self.assertRaises(ValueError):
            model(images=torch.zeros(1, 3, 224, 224), clinical=torch.ones(1, 3))

    def test_modified_recipe_and_catalog_rejected_before_loading(self):
        for which in ("plan", "catalog"):
            plan, catalog = deepcopy(self.plan), deepcopy(self.catalog)
            (plan if which == "plan" else catalog)["unapproved"] = True
            with patch.object(models, "load_encoder") as loader, self.assertRaises(ValueError):
                models.build_model("image_densenet121", plan, catalog)
            loader.assert_not_called()

    def test_unverified_fold_weights_and_missing_files_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.catalog["models"]["image_densenet121"]["files"][0]["name"]
            path.write_bytes(b"an arbitrary fitted state is not a pinned external checkpoint")
            with patch.object(models, "load_encoder") as loader, self.assertRaises(ValueError):
                models.build_model("image_densenet121", self.plan, self.catalog, [path])
            loader.assert_not_called()
        with self.assertRaises(ValueError):
            models.build_model("image_densenet121", self.plan, self.catalog, [])

    def test_exact_local_files_checked_by_full_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight.bin"
            path.write_bytes(b"abc")
            import hashlib
            catalog = {"models": {"example": {"verification_enabled": True, "files": [
                {"name": path.name, "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}]}}}
            self.assertEqual(models._verified_paths("example", catalog, [path]), [path])
            path.write_bytes(b"abd")
            with self.assertRaises(ValueError):
                models._verified_paths("example", catalog, [path])

    def test_image_head_is_single_linear_and_eval_forward_works(self):
        model, audit = self._dense()
        self.assertIsInstance(model.task_head, nn.Linear)
        self.assertEqual((model.task_head.in_features, model.task_head.out_features), (1024, 1))
        with torch.inference_mode():
            self.assertEqual(tuple(model(images=torch.ones(2, 3, 224, 224)).shape), (2,))
        self.assertFalse(audit["arbitrary_fold_checkpoint_reuse_accepted"])

    def test_last_block_and_full_preserve_bn_running_statistics_mode(self):
        model, _ = self._dense()
        model.train()
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertFalse(model.encoder.training)
        models.set_phase(model, "last_block")
        self.assertTrue(model.encoder.get_submodule("0.denseblock4").training)
        self.assertFalse(model.encoder.get_submodule("0.denseblock3").training)
        self.assertTrue(model.encoder.get_submodule("0.denseblock4.drop").training)
        self.assertFalse(model.encoder.get_submodule("0.denseblock3.drop").training)
        for name, p in model.encoder.named_parameters():
            self.assertEqual(p.requires_grad, models._layer_for_name("image_densenet121", name) == 4)
        models.set_phase(model, "full")
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(model.encoder.training)
        self.assertTrue(all(not m.training for m in model.encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)))
        model.eval()
        self.assertFalse(any(m.training for m in model.encoder.modules()))

    def test_complete_disjoint_layerwise_groups_and_registered_rates(self):
        model, _ = self._dense()
        groups = models.optimizer_groups(model, self.plan, "head_only")
        self.assertEqual([g["name"] for g in groups], ["task_head"])
        self.assertEqual(groups[0]["lr"], 0.001)
        models.set_phase(model, "full")
        groups = models.optimizer_groups(model, self.plan, "full", stage="stage_2")
        self.assertEqual(len(groups), 6)
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        rates = {g["name"]: g["lr"] for g in groups}
        self.assertEqual(rates["task_head"], 0.0001)
        self.assertAlmostEqual(rates["image_layer_0"], 0.00001 * 0.8 ** 4)
        self.assertEqual(rates["image_layer_4"], 0.00001)
        with self.assertRaises(ValueError):
            models.optimizer_groups(model, self.plan, "full", stage="stage_1")

    def test_fusion_dimensions_and_explicit_film_gamma(self):
        concat, _ = self._dense("fusion_intermediate_concat")
        film, _ = self._dense("fusion_intermediate_gated_film")
        self.assertEqual(concat.task_head[0].in_features, 1032)
        self.assertEqual(film.task_head[0].in_features, 1024)
        images, clinical = torch.ones(2, 3, 224, 224), torch.ones(2, 3)
        with torch.inference_mode():
            cf = film.clinical_encoder(clinical)
            visual = film.encoder(images)
            expected = film.task_head((1 + torch.tanh(film.gamma(cf))) * visual + film.beta(cf)).squeeze(-1)
            self.assertTrue(torch.equal(film(images=images, clinical=clinical), expected))
            self.assertEqual(tuple(concat(images=images, clinical=clinical).shape), (2,))

    def test_modality_shape_and_clinical_finite_requirements(self):
        model, _ = self._dense("fusion_intermediate_concat")
        for clinical in (torch.ones(2, 4), torch.full((2, 3), float("nan"))):
            with self.assertRaises(ValueError):
                model(images=torch.ones(2, 3, 224, 224), clinical=clinical)
        with self.assertRaises(ValueError):
            model(images=torch.ones(2, 3, 223, 224), clinical=torch.ones(2, 3))

    def test_layer_mapping_boundaries_for_all_architectures(self):
        cases = {
            "image_densenet121": [("0.conv0.weight", 0), ("0.denseblock4.denselayer16.conv2.weight", 4)],
            "image_vit_b16": [("conv_proj.weight", 0), ("encoder.layers.encoder_layer_11.mlp.0.weight", 12)],
            "image_convnextv2_tiny": [("downsample_layers.0.0.weight", 0), ("stages.3.2.pwconv2.weight", 18)],
            "image_skelex": [("encoder.embeddings.cls_token", 0), ("encoder.encoder.layer.23.layernorm_after.weight", 24)],
            "image_biomedclip": [("trunk.cls_token", 0), ("trunk.blocks.11.norm1.weight", 12), ("head.proj.weight", 12)],
            "image_mambaout_tiny": [("stem.conv1.weight", 0), ("stages.3.blocks.2.norm.weight", 18), ("head.pre_logits.fc.weight", 18)]}
        for model_id, records in cases.items():
            for name, layer in records:
                self.assertEqual(models._layer_for_name(model_id, name), layer)
            with self.assertRaisesRegex(ValueError, "unmapped_registered"):
                models._layer_for_name(model_id, "unregistered.output.weight")
        with self.assertRaises(ValueError):
            models._layer_for_name("image_mambaout_tiny", "stagesX3XblocksX2.norm.weight")

    def test_skelex_fixed_sinusoidal_positions_never_unfreeze(self):
        encoder = nn.Module()
        for name in ("encoder.embeddings.position_embeddings", "encoder.embeddings.cls_token",
                     "encoder.layernorm.weight", *(f"encoder.encoder.layer.{i}.weight" for i in range(24))):
            _attach(encoder, name)
        model = models.RegisteredNeuralModel("image_skelex", encoder, "image_skelex")
        for phase in ("head_only", "last_block", "full"):
            models.set_phase(model, phase)
            self.assertFalse(encoder.get_parameter("encoder.embeddings.position_embeddings").requires_grad)
        self.assertTrue(encoder.get_parameter("encoder.embeddings.cls_token").requires_grad)
        self.assertTrue(encoder.get_parameter("encoder.encoder.layer.23.weight").requires_grad)

    def test_normalization_preserves_readonly_geometry_and_uses_clip_statistics(self):
        array = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(1, 224, 224)
        before = array.copy()
        array.setflags(write=False)
        dense = models.normalize_for_model(array, "image_densenet121")
        for model_id in models.IMAGE_DIMENSIONS:
            normalized = models.normalize_for_model(array, model_id)
            self.assertEqual(tuple(normalized.shape), (3, 224, 224))
            self.assertEqual(torch.equal(dense, normalized), model_id != "image_biomedclip")
        self.assertTrue(np.array_equal(before, array))
        self.assertFalse(array.flags.writeable)

    def test_skelex_noise_is_device_local_unmasked_spatially_ordered(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=1024, image_size=224, patch_size=16, num_channels=3, mask_ratio=0)
                self.seen_noise = None

            def forward(self, pixel_values, noise, return_dict):
                self.seen_noise = noise
                batch = len(pixel_values)
                return SimpleNamespace(last_hidden_state=torch.zeros(batch, 197, 1024, device=pixel_values.device),
                    mask=torch.zeros(batch, 196, device=pixel_values.device),
                    ids_restore=torch.arange(196, device=pixel_values.device).expand(batch, -1))
        encoder = Encoder()
        adapter = models.SkelexTrainingFeatures(encoder)
        with torch.inference_mode():
            result = adapter(torch.zeros(3, 3, 224, 224))
        self.assertEqual(tuple(result.shape), (3, 1024))
        self.assertEqual(encoder.seen_noise.device, result.device)
        self.assertTrue(torch.equal(encoder.seen_noise, torch.arange(196).float().expand(3, -1)))


if __name__ == "__main__":
    unittest.main()
