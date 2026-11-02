"""Registered neural model construction and explicit unfreezing maps.

Construction accepts only local, full-hash-pinned external checkpoints. This
module contains no downloader, optimizer construction, backward pass or fit.
The inference-only verification wrappers remain unchanged; separate wrappers
here retain their exact adopted encoders and expose one task logit per knee.
"""
from __future__ import annotations

from collections import defaultdict
import re
from pathlib import Path

import torch
from torch import nn

from knee import final_model_release as release
from knee.checkpoint_transport import file_digest
from knee.pretrained_verification import canonical_digest, load_encoder
from knee.pretrained_final_encoders import load_final_encoder, normalize_sealed_image
from knee.pretrained_convnextv2 import normalize_sealed_image as normalize_imagenet
from knee.training_plan import _read_json

VERSION = "registered_neural_model_factory_v1.0"
IMAGE_DIMENSIONS = {
    "image_densenet121": 1024, "image_vit_b16": 768,
    "image_convnextv2_tiny": 768, "image_skelex": 1024,
    "image_biomedclip": 512, "image_mambaout_tiny": 2304,
}
FUSION_IDS = {"fusion_intermediate_concat", "fusion_intermediate_gated_film"}
PHASES = {"head_only", "last_block", "full"}
_DEPTHS = (3, 3, 9, 3)
_LAST_LAYER = {"image_densenet121": 4, "image_vit_b16": 12,
    "image_convnextv2_tiny": 18, "image_skelex": 24,
    "image_biomedclip": 12, "image_mambaout_tiny": 18}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _validate_release(plan, catalog):
    root = Path(__file__).resolve().parents[2]
    registry = _read_json(root / "configs/model_registry_v1_1.json")
    release.validate_plan(plan, registry)
    _require(canonical_digest(catalog) == release.PINS["configs/pretrained_weights_v1_1.json"],
             "approved_training_checkpoint_catalog_required")


def normalize_for_model(readonly_gray, encoder_id):
    """Use the verified normalization without altering sealed image geometry."""
    _require(encoder_id in IMAGE_DIMENSIONS, "accessible_registered_image_encoder_required")
    if encoder_id in {"image_biomedclip", "image_mambaout_tiny"}:
        return normalize_sealed_image(readonly_gray, encoder_id)
    return normalize_imagenet(readonly_gray)


def _verified_paths(model_id, catalog, paths):
    spec = catalog["models"].get(model_id)
    _require(spec is not None and spec["verification_enabled"] is True,
             "authorized_accessible_checkpoint_required")
    _require(isinstance(paths, (list, tuple)) and len(paths) == len(spec["files"]),
             "all_exact_local_checkpoint_files_required")
    verified = []
    for supplied, item in zip(paths, spec["files"]):
        path = Path(supplied)
        _require(path.is_file() and not path.is_symlink()
                 and not any(parent.is_symlink() or (hasattr(parent, "is_junction")
                     and parent.is_junction()) for parent in path.absolute().parents)
                 and path.name == item["name"] and path.stat().st_size == item["bytes"]
                 and file_digest(path) == item["sha256"],
                 "exact_external_checkpoint_hash_and_metadata_required")
        verified.append(path)
    return verified


class SkelexTrainingFeatures(nn.Module):
    """Unmasked CLS readout with deterministic spatial order on either device."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        _require(encoder.config.hidden_size == 1024 and encoder.config.image_size == 224
                 and encoder.config.patch_size == 16 and encoder.config.num_channels == 3
                 and encoder.config.mask_ratio == 0,
                 "verified_unmasked_skelex_encoder_required")

    def forward(self, images):
        patches = (self.encoder.config.image_size // self.encoder.config.patch_size) ** 2
        # mask_ratio=0 alone still randomly sorts tokens in the MAE implementation.
        noise = torch.arange(patches, dtype=torch.float32, device=images.device)
        noise = noise.reshape(1, patches).expand(len(images), -1)
        output = self.encoder(pixel_values=images, noise=noise, return_dict=True)
        expected = torch.arange(patches, device=images.device).expand(len(images), -1)
        _require(tuple(output.last_hidden_state.shape) == (len(images), patches + 1, 1024)
                 and bool(torch.count_nonzero(output.mask) == 0)
                 and torch.equal(output.ids_restore, expected),
                 "skelex_all_tokens_in_spatial_order_required")
        return output.last_hidden_state[:, 0, :].contiguous()


def _layer_for_name(model_id, name):
    """Map every learned tensor explicitly; reject unknown architecture names."""
    if model_id == "image_densenet121":
        if re.fullmatch(r"0\.(conv0|norm0)\.(weight|bias)", name):
            return 0
        match = re.match(r"0\.(denseblock([1-4])|transition([1-3]))\.", name)
        if match:
            return int(match.group(2) or match.group(3))
        if name in {"0.norm5.weight", "0.norm5.bias"}:
            return 4
    elif model_id == "image_vit_b16":
        if name in {"class_token", "encoder.pos_embedding"} or name.startswith("conv_proj."):
            return 0
        match = re.match(r"encoder.layers.encoder_layer_(\d+)\.", name)
        if match and int(match.group(1)) in range(12):
            return int(match.group(1)) + 1
        if name in {"encoder.ln.weight", "encoder.ln.bias"}:
            return 12
    elif model_id in {"image_convnextv2_tiny", "image_mambaout_tiny"}:
        if model_id == "image_convnextv2_tiny":
            if name.startswith("downsample_layers.0."):
                return 0
            if name in {"norm.weight", "norm.bias"}:
                return 18
            block = re.match(r"stages\.([0-3])\.(\d+)\.", name)
            down = re.match(r"downsample_layers\.([1-3])\.", name)
        else:
            if name.startswith("stem."):
                return 0
            if name.startswith(("head.norm.", "head.pre_logits.")):
                return 18
            block = re.match(r"stages\.([0-3])\.blocks\.(\d+)\.", name)
            down = re.match(r"stages\.([1-3])\.downsample\.", name)
        if block:
            stage, index = map(int, block.groups())
            if index in range(_DEPTHS[stage]):
                return sum(_DEPTHS[:stage]) + index + 1
        if down:
            return sum(_DEPTHS[:int(down.group(1))]) + 1
    elif model_id == "image_skelex":
        if name.startswith("encoder.embeddings."):
            return 0
        match = re.match(r"encoder\.encoder\.layer\.(\d+)\.", name)
        if match and int(match.group(1)) in range(24):
            return int(match.group(1)) + 1
        if name in {"encoder.layernorm.weight", "encoder.layernorm.bias"}:
            return 24
    elif model_id == "image_biomedclip":
        if name in {"trunk.cls_token", "trunk.pos_embed"} or name.startswith("trunk.patch_embed."):
            return 0
        match = re.match(r"trunk\.blocks\.(\d+)\.", name)
        if match and int(match.group(1)) in range(12):
            return int(match.group(1)) + 1
        if name.startswith(("trunk.norm.", "head.proj.")):
            return 12
    raise ValueError("unmapped_registered_encoder_parameter:" + model_id + ":" + name)


_LAST_MODULES = {
    "image_densenet121": ("0.denseblock4", "0.norm5"),
    "image_vit_b16": ("encoder.layers.encoder_layer_11", "encoder.ln"),
    "image_convnextv2_tiny": ("stages.3.2", "norm"),
    "image_skelex": ("encoder.encoder.layer.23", "encoder.layernorm"),
    "image_biomedclip": ("trunk.blocks.11", "trunk.norm", "head"),
    "image_mambaout_tiny": ("stages.3.blocks.2", "head"),
}


def _structural_freeze(model_id, name):
    return model_id == "image_skelex" and name == "encoder.embeddings.position_embeddings"


def _clinical_encoder():
    return nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(16, 8), nn.ReLU(), nn.Dropout(0.1))


class RegisteredNeuralModel(nn.Module):
    """One common interface for the registered clinical, image and fusion tasks."""

    def __init__(self, model_id, encoder=None, image_encoder_id=None):
        super().__init__()
        self.model_id = model_id
        self.image_encoder_id = image_encoder_id
        self.encoder = encoder
        self.phase = "head_only"
        self.parameter_layers = {}
        if encoder is not None:
            self.parameter_layers = {name: _layer_for_name(image_encoder_id, name)
                                     for name, _ in encoder.named_parameters()}
            _require(set(self.parameter_layers.values()) == set(range(_LAST_LAYER[image_encoder_id] + 1)),
                     "complete_registered_encoder_layer_coverage_required")
            self.feature_dimension = IMAGE_DIMENSIONS[image_encoder_id]
        if model_id == "clinical_mlp":
            self.clinical_encoder = _clinical_encoder()
            self.task_head = nn.Linear(8, 1)
        elif model_id in IMAGE_DIMENSIONS:
            self.task_head = nn.Linear(self.feature_dimension, 1)
        else:
            _require(model_id in FUSION_IDS, "registered_neural_task_required")
            self.clinical_encoder = _clinical_encoder()
            head_inputs = self.feature_dimension
            if model_id == "fusion_intermediate_concat":
                head_inputs += 8
            else:
                self.gamma = nn.Linear(8, self.feature_dimension)
                self.beta = nn.Linear(8, self.feature_dimension)
            self.task_head = nn.Sequential(nn.Linear(head_inputs, 32), nn.ReLU(),
                                           nn.Dropout(0.1), nn.Linear(32, 1))
        set_phase(self, "head_only")

    def _enforce_encoder_modes(self):
        if self.encoder is None:
            return
        if not self.training or self.phase == "head_only":
            self.encoder.eval()
        elif self.phase == "full":
            self.encoder.train(True)
        else:
            self.encoder.eval()
            for name in _LAST_MODULES[self.image_encoder_id]:
                self.encoder.get_submodule(name).train(True)
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode=True):
        _require(type(mode) is bool, "explicit_neural_training_mode_required")
        super().train(mode)
        self._enforce_encoder_modes()
        return self

    def forward(self, images=None, clinical=None):
        needs_image = self.encoder is not None
        needs_clinical = self.model_id == "clinical_mlp" or self.model_id in FUSION_IDS
        _require((images is not None) == needs_image and (clinical is not None) == needs_clinical,
                 "exact_registered_modalities_required")
        reference = images if needs_image else clinical
        _require(isinstance(reference, torch.Tensor) and reference.ndim >= 2
                 and len(reference) > 0, "nonempty_neural_batch_required")
        if needs_image:
            _require(images.ndim == 4 and tuple(images.shape[1:]) == (3, 224, 224)
                     and images.is_floating_point() and bool(torch.isfinite(images).all()),
                     "normalized_rgb_224_training_batch_required")
            features = self.encoder(images)
            _require(tuple(features.shape) == (len(images), self.feature_dimension),
                     "exact_registered_feature_dimension_required")
        if needs_clinical:
            _require(isinstance(clinical, torch.Tensor) and clinical.ndim == 2
                     and tuple(clinical.shape) == (len(reference), 3)
                     and clinical.is_floating_point() and bool(torch.isfinite(clinical).all())
                     and clinical.device == reference.device,
                     "three_fold_processed_clinical_predictors_required")
            clinical_features = self.clinical_encoder(clinical)
            if not needs_image:
                features = clinical_features
            elif self.model_id == "fusion_intermediate_concat":
                features = torch.cat((features, clinical_features), dim=1)
            else:
                features = (1 + torch.tanh(self.gamma(clinical_features))) * features + self.beta(clinical_features)
        logits = self.task_head(features).squeeze(-1)
        _require(tuple(logits.shape) == (len(reference),) and bool(torch.isfinite(logits).all()),
                 "one_finite_logit_per_knee_required")
        return logits


def set_phase(model, phase):
    """Toggle trainable parameters only; this does not grant training authority."""
    _require(isinstance(model, RegisteredNeuralModel) and phase in PHASES,
             "registered_model_and_unfreezing_phase_required")
    if model.encoder is None:
        _require(phase in {"head_only", "full"}, "clinical_has_no_last_image_block")
    model.phase = phase
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            relative = name[len("encoder."):]
            last = model.parameter_layers[relative] == _LAST_LAYER[model.image_encoder_id]
            parameter.requires_grad_(not _structural_freeze(model.image_encoder_id, relative)
                                     and (phase == "full" or phase == "last_block" and last))
            if not parameter.requires_grad:
                # A parameter frozen after an earlier phase must not retain a
                # stale gradient that AdamW could otherwise still consume.
                parameter.grad = None
        else:
            parameter.requires_grad_(True)
    model._enforce_encoder_modes()
    return model


def optimizer_groups(model, plan, phase, *, stage="stage_1"):
    """Return disjoint complete trainable groups; do not construct an optimizer.

    Stage 2 callers may construct groups once in ``full`` and subsequently
    freeze parameters with ``set_phase``. AdamW skips parameters with no grad,
    preserving state across transitions without decaying frozen weights.
    """
    _require(isinstance(model, RegisteredNeuralModel) and model.phase == phase
             and phase in PHASES and stage in {"stage_1", "stage_2"},
             "explicit_current_phase_and_registered_stage_required")
    registry = _read_json(Path(__file__).resolve().parents[2] / "configs/model_registry_v1_1.json")
    release.validate_plan(plan, registry)
    if model.model_id == "clinical_mlp":
        spec = plan["clinical_models"]["clinical_mlp"]
        head_lr, decay = spec["learning_rate"], spec["weight_decay"]
    elif model.model_id in FUSION_IDS:
        spec = plan["multimodal_models"]
        head_lr, encoder_lr, decay = spec["head_learning_rate"], spec["image_encoder_learning_rate"], spec["weight_decay"]
    else:
        _require(stage == "stage_2" or phase == "head_only", "stage_one_encoder_must_remain_frozen")
        spec = plan["radiographic_models"][stage]
        head_lr = spec["head_learning_rate"]
        decay = spec.get("weight_decay", spec.get("head_weight_decay"))
        encoder_lr = plan["radiographic_models"]["stage_2"]["encoder_learning_rate"]
    grouped = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            layer = model.parameter_layers[name[len("encoder."):]] if name.startswith("encoder.") else None
            grouped[layer].append((name, parameter))
    result = []
    for layer in sorted(grouped, key=lambda value: -1 if value is None else value):
        rate = head_lr if layer is None else encoder_lr * plan["radiographic_models"]["stage_2"]["layerwise_learning_rate_decay"] ** (_LAST_LAYER[model.image_encoder_id] - layer)
        records = grouped[layer]
        result.append({"name": "task_head" if layer is None else f"image_layer_{layer}",
            "params": [p for _, p in records], "param_names": [n for n, _ in records],
            "lr": rate, "base_lr": rate, "initial_lr": rate, "weight_decay": decay})
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    actual = [id(p) for group in result for p in group["params"]]
    _require(len(expected) == len(actual) == len(set(actual)) and set(expected) == set(actual),
             "disjoint_complete_trainable_parameter_groups_required")
    return result


def build_model(model_id, plan, catalog, checkpoint_paths=None, *, image_encoder_id=None, seed=2026):
    """Build from sealed external weights; no fitted fold weights are accepted."""
    _validate_release(plan, catalog)
    _require(type(seed) is int and seed in plan["selection"]["neural_finalist_seeds"],
             "registered_neural_seed_required")
    _require(model_id in IMAGE_DIMENSIONS or model_id in FUSION_IDS or model_id == "clinical_mlp",
             "accessible_registered_neural_model_required")
    if model_id == "clinical_mlp":
        _require(checkpoint_paths is None and image_encoder_id is None,
                 "clinical_mlp_has_no_external_image_checkpoint")
        encoder, adoption, encoder_id = None, {}, None
    else:
        encoder_id = image_encoder_id if model_id in FUSION_IDS else model_id
        _require(encoder_id in IMAGE_DIMENSIONS and (model_id in FUSION_IDS or image_encoder_id is None),
                 "explicit_accessible_selected_fusion_encoder_required")
        paths = _verified_paths(encoder_id, catalog, checkpoint_paths)
        if encoder_id in {"image_biomedclip", "image_mambaout_tiny"}:
            frozen, adoption = load_final_encoder(encoder_id, paths)
            encoder = frozen.encoder
        else:
            frozen, adoption = load_encoder(encoder_id, paths)
            encoder = SkelexTrainingFeatures(frozen.encoder) if encoder_id == "image_skelex" else frozen
        _require(catalog["models"][encoder_id]["feature_dimension"] == IMAGE_DIMENSIONS[encoder_id],
                 "approved_checkpoint_feature_dimension_required")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RegisteredNeuralModel(model_id, encoder, encoder_id)
    model.eval()
    audit = {"version": VERSION, "model_id": model_id, "image_encoder_id": encoder_id,
        "seed": seed, "head_initialization": "seeded_new_head", "initial_phase": "head_only",
        "external_checkpoint_adoption": adoption,
        "external_checkpoint_files": [] if encoder is None else [
            {"name": p.name, "sha256": file_digest(p), "bytes": p.stat().st_size} for p in paths],
        "encoder_feature_dimension": None if encoder is None else IMAGE_DIMENSIONS[encoder_id],
        "layer_mapping": dict(model.parameter_layers),
        "structurally_frozen_parameters": [name for name in model.parameter_layers if _structural_freeze(encoder_id, name)],
        "batch_normalization_running_statistics": "frozen_all_phases",
        "dropout_mode": "frozen_blocks_eval_unfrozen_blocks_follow_training_mode",
        "fusion_head_inputs": None if model_id not in FUSION_IDS else model.task_head[0].in_features,
        "training_executed": False, "optimizer_created": False,
        "arbitrary_fold_checkpoint_reuse_accepted": False,
        "future_fold_weight_reuse_requires_authenticated_training_identity": True}
    return model, audit
