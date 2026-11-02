"""Pinned, visual-only CPU adapters for the two final approved candidates.

No network, patient data, optimizer, tokenizer, or language model is used here.
BiomedCLIP retains its pretrained 768-to-512 visual projection, without L2
normalization (official ``encode_image(normalize=False)`` readout). MambaOut
retains its pretrained MLP and exposes its 2304 pre-logits, not the 576 final
spatial channels. Both operate on the already sealed 224-by-224 geometry.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

VERSION = "final_visual_candidates_cpu_v1.0"
TIMM_VERSION = "1.0.15"
BIOMED_ID = "image_biomedclip"
MAMBA_ID = "image_mambaout_tiny"
FEATURE_DIMENSIONS = {BIOMED_ID: 512, MAMBA_ID: 2304}
BIOMED_REVISION = "9f341de24bfb00180f1b847274256e9b65a3a32e"
MAMBA_REVISION = "d080941663a60c42d015ecf8d8fad86d12694208"
BIOMED_WEIGHTS_BYTES = 783705670
BIOMED_WEIGHTS_SHA256 = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
BIOMED_CONFIG_BYTES = 707
BIOMED_CONFIG_SHA256 = "9a41f334a8c444678772c0ebb9ab854c97ab350bced3a17b803e258d39c23dc0"
MAMBA_WEIGHTS_BYTES = 106197008
MAMBA_WEIGHTS_SHA256 = "ac9535c5b9fe1d325c5bf58dd4ffc3201c1fc04992b37e393977a7526cccafb4"
MAMBA_CONFIG_BYTES = 630
MAMBA_CONFIG_SHA256 = "f65ed317c48459de9fa4399567e4be7a8895f29a395f7c02b778c8290f25fd37"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATION = {
    BIOMED_ID: (CLIP_MEAN, CLIP_STD),
    MAMBA_ID: (IMAGENET_MEAN, IMAGENET_STD),
}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _normalization(model_id):
    _require(model_id in NORMALIZATION, "final_registered_visual_model_required")
    mean, std = NORMALIZATION[model_id]
    return (torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1),
            torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1))


def normalize_sealed_image(array, model_id):
    """Copy and channel-normalize a read-only image; never alter its geometry."""
    mean, std = _normalization(model_id)
    _require(isinstance(array, np.ndarray) and array.shape == (1, 224, 224)
             and array.dtype == np.dtype("float32") and array.flags.c_contiguous
             and not array.flags.writeable and bool(np.isfinite(array).all())
             and array.min() >= 0 and array.max() <= 1,
             "sealed_readonly_zero_one_float32_image_required")
    image = torch.from_numpy(array.copy()).repeat(3, 1, 1).unsqueeze(0)
    return ((image - mean) / std).squeeze(0).contiguous()


def _validate_normalized_inputs(images, model_id):
    mean, std = _normalization(model_id)
    _require(isinstance(images, torch.Tensor) and images.device.type == "cpu"
             and images.dtype == torch.float32 and images.ndim == 4
             and tuple(images.shape[1:]) == (3, 224, 224)
             and 1 <= images.shape[0] <= 2 and not images.requires_grad
             and images.layout == torch.strided,
             "final_small_cpu_normalized_224_batch_required")
    _require(bool(torch.isfinite(images).all()), "final_finite_input_required")
    gray = images * std + mean
    _require(bool(((gray >= -1e-6) & (gray <= 1 + 1e-6)).all())
             and torch.allclose(gray[:, 0], gray[:, 1], atol=1e-6, rtol=0)
             and torch.allclose(gray[:, 0], gray[:, 2], atol=1e-6, rtol=0),
             "final_model_specific_normalized_replicated_gray_required")


def _verified_file(path, expected_bytes, expected_sha256, suffix):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and path.suffix == suffix
             and path.stat().st_size == expected_bytes,
             "final_exact_regular_checkpoint_or_metadata_file_required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    _require(digest.hexdigest() == expected_sha256, "final_file_sha256_mismatch")
    return path


def _verified_json(path, expected_bytes, expected_sha256):
    path = _verified_file(path, expected_bytes, expected_sha256, ".json")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "final_metadata_object_required")
    return value


def _validate_tensor(value, shape, *, integer_positions=False):
    dtype = torch.int64 if integer_positions else torch.float32
    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
             and value.dtype == dtype and value.layout == torch.strided
             and tuple(value.shape) == tuple(shape),
             "final_checkpoint_tensor_shape_or_dtype_invalid")
    _require(bool(torch.isfinite(value).all()), "final_checkpoint_nonfinite_tensor")
    if integer_positions:
        _require(torch.equal(value, torch.arange(512, dtype=torch.int64).reshape(1, 512)),
                 "biomedclip_exact_unused_text_positions_required")


def _biomed_unused_schema():
    """Account for every published text tensor without constructing a BERT."""
    prefix = "text.transformer.embeddings."
    schema = {
        "logit_scale": (),
        prefix + "position_ids": (1, 512),
        prefix + "word_embeddings.weight": (30522, 768),
        prefix + "position_embeddings.weight": (512, 768),
        prefix + "token_type_embeddings.weight": (2, 768),
        prefix + "LayerNorm.weight": (768,),
        prefix + "LayerNorm.bias": (768,),
        "text.proj.0.weight": (640, 768),
        "text.proj.2.weight": (512, 640),
    }
    for index in range(12):
        layer = f"text.transformer.encoder.layer.{index}."
        for name in ("query", "key", "value"):
            schema[layer + f"attention.self.{name}.weight"] = (768, 768)
            schema[layer + f"attention.self.{name}.bias"] = (768,)
        for name in ("attention.output", "output"):
            schema[layer + name + ".dense.weight"] = (768, 768 if name == "attention.output" else 3072)
            schema[layer + name + ".dense.bias"] = (768,)
            schema[layer + name + ".LayerNorm.weight"] = (768,)
            schema[layer + name + ".LayerNorm.bias"] = (768,)
        schema[layer + "intermediate.dense.weight"] = (3072, 768)
        schema[layer + "intermediate.dense.bias"] = (3072,)
    return schema


def _adopt_biomed_state(visual, state):
    expected = visual.state_dict()
    unused = _biomed_unused_schema()
    visual_keys = {"visual." + key for key in expected}
    _require(isinstance(state, Mapping) and set(state) == visual_keys | set(unused)
             and len(visual_keys) == 151 and len(unused) == 201,
             "biomedclip_exact_complete_published_keys_required")
    for key, value in state.items():
        shape = expected[key[7:]].shape if key in visual_keys else unused[key]
        _validate_tensor(value, shape,
                         integer_positions=key == "text.transformer.embeddings.position_ids")
    visual.load_state_dict({key[7:]: state[key] for key in visual_keys}, strict=True)
    return {
        "source_tensors": len(state), "loaded_visual_pretrained_tensors": len(visual_keys),
        "excluded_text_tensor_count": 200, "excluded_logit_scale_tensor_count": 1,
        "excluded_tensor_keys": sorted(unused), "all_source_tensors_accounted": True,
        "text_model_constructed": False, "text_model_executed": False,
        "tokenizer_loaded": False, "visual_projection_retained": True,
        "feature_readout": "official_visual_projected_512_encode_image_normalize_false",
        "feature_l2_normalization_applied": False,
        "feature_dimension": 512, "initialized_nonpretrained_parameter_keys": [],
    }


def _adopt_mamba_state(complete, state):
    expected = complete.state_dict()
    _require(isinstance(state, Mapping) and set(state) == set(expected)
             and len(state) == 172,
             "mambaout_exact_complete_published_keys_required")
    for key, value in state.items():
        _validate_tensor(value, expected[key].shape)
    complete.load_state_dict(state, strict=True)
    _require(isinstance(complete.head.fc, nn.Linear)
             and complete.head.fc.in_features == 2304
             and complete.head.fc.out_features == 1000,
             "mambaout_exact_pretrained_classifier_required")
    complete.reset_classifier(0)
    _require(isinstance(complete.head.fc, nn.Identity), "mambaout_classifier_removal_failed")
    return {
        "source_tensors": len(state), "loaded_pretrained_tensors": len(state),
        "complete_imagenet_state_loaded_strictly_before_head_removal": True,
        "removed_classifier_keys": ["head.fc.bias", "head.fc.weight"],
        "classification_head_used_for_prediction": False,
        "pretrained_pre_logits_mlp_retained": True,
        "last_spatial_channel_dimension": 576, "feature_dimension": 2304,
        "feature_readout": "official_pre_logits_after_pretrained_mlp",
        "all_source_tensors_accounted": True,
        "initialized_nonpretrained_parameter_keys": [],
    }


class FrozenVisualCPU(nn.Module):
    """Deliberately inference-only; this wrapper is not a future trainer."""

    def __init__(self, encoder, model_id):
        super().__init__()
        _require(model_id in FEATURE_DIMENSIONS, "final_registered_visual_model_required")
        self.encoder = encoder.cpu().float()
        self.model_id = model_id
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        _require(mode is False, "final_cpu_verification_training_forbidden")
        return super().train(False)

    @torch.inference_mode()
    def forward(self, images):
        _validate_normalized_inputs(images, self.model_id)
        _require(all(not module.training for module in self.modules())
                 and all(not parameter.requires_grad and parameter.grad is None
                         for parameter in self.parameters()),
                 "final_frozen_eval_state_required")
        features = self.encoder(images)
        _require(isinstance(features, torch.Tensor) and features.device.type == "cpu"
                 and features.dtype == torch.float32
                 and tuple(features.shape) == (len(images), FEATURE_DIMENSIONS[self.model_id])
                 and bool(torch.isfinite(features).all()),
                 "final_finite_expected_visual_features_required")
        return features.contiguous()


def load_final_encoder(model_id, paths):
    """Return (frozen CPU module, audit) from exact local [weights, config]."""
    _require(model_id in FEATURE_DIMENSIONS and isinstance(paths, (list, tuple))
             and len(paths) == 2, "final_model_and_two_local_files_required")
    import timm

    _require(timm.__version__ == TIMM_VERSION, "final_timm_1_0_15_required")
    if model_id == BIOMED_ID:
        config = _verified_json(paths[1], BIOMED_CONFIG_BYTES, BIOMED_CONFIG_SHA256)
        _require(config["model_cfg"]["embed_dim"] == 512
                 and config["model_cfg"]["vision_cfg"] == {
                     "timm_model_name": "vit_base_patch16_224", "timm_model_pretrained": False,
                     "timm_pool": "", "timm_proj": "linear", "image_size": 224}
                 and tuple(config["preprocess_cfg"]["mean"]) == CLIP_MEAN
                 and tuple(config["preprocess_cfg"]["std"]) == CLIP_STD,
                 "biomedclip_exact_visual_metadata_required")
        path = _verified_file(paths[0], BIOMED_WEIGHTS_BYTES, BIOMED_WEIGHTS_SHA256, ".bin")
        # Same visual tower as OpenCLIP TimmModel(pool='', proj='linear',
        # proj_bias=False), but no language model or tokenizer is constructed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            trunk = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
            visual = nn.Sequential(OrderedDict([
                ("trunk", trunk),
                ("head", nn.Sequential(OrderedDict([
                    ("drop", nn.Dropout(0.0)), ("proj", nn.Linear(768, 512, bias=False))]))),
            ]))
        state = torch.load(path, map_location="cpu", weights_only=True)
        audit = _adopt_biomed_state(visual, state)
        del state
        return FrozenVisualCPU(visual, model_id), audit

    config = _verified_json(paths[1], MAMBA_CONFIG_BYTES, MAMBA_CONFIG_SHA256)
    _require(config["architecture"] == "mambaout_tiny" and config["num_classes"] == 1000
             and config["num_features"] == 576
             and config["pretrained_cfg"]["input_size"] == [3, 224, 224]
             and tuple(config["pretrained_cfg"]["mean"]) == IMAGENET_MEAN
             and tuple(config["pretrained_cfg"]["std"]) == IMAGENET_STD,
             "mambaout_exact_input_metadata_required")
    path = _verified_file(paths[0], MAMBA_WEIGHTS_BYTES, MAMBA_WEIGHTS_SHA256, ".safetensors")
    from safetensors.torch import load_file

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        complete = timm.create_model("mambaout_tiny", pretrained=False, num_classes=1000)
    state = load_file(str(path), device="cpu")
    audit = _adopt_mamba_state(complete, state)
    del state
    return FrozenVisualCPU(complete, model_id), audit
