"""Local-only, frozen CPU verification of the registered SKELEX checkpoint.

The complete official MAE state (including decoder) is loaded strictly before
retaining the encoder. Inference exposes its normalized CLS token with all
196 patches in spatial order. It does not reconstruct images, mask patches,
resize, train, or fetch code or weights from a remote repository.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

VERSION = "skelex_pinned_unmasked_cpu_features_v1.0"
MODEL_ID = "image_skelex"
HF_REPOSITORY = "skhoha/SKELEX"
HF_REVISION = "368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e"
TRANSFORMERS_VERSION = "4.50.2"
CHECKPOINT_BYTES = 1318230232
CHECKPOINT_SHA256 = "81cd6e9cf8da0c56d149a2e1a3668fdc6def2742b055f2696f97507332d69ef8"
CONFIG_SHA256 = "b48411f4313c2ee6357586b57d185befac8c7c77cc475bc2188ec4487b1bc6f7"
PROCESSOR_SHA256 = "a250969c94afba52d785a0e08dd36e13aeda97c4dd2b7fd0d24b457288536cea"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_small_json(path, expected_bytes, expected_sha):
    path = Path(path)
    _require(path.is_file() and path.stat().st_size == expected_bytes,
             "skelex_exact_metadata_file_required")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected_sha,
             "skelex_metadata_hash_mismatch")
    value = json.loads(raw)
    _require(isinstance(value, dict), "skelex_metadata_object_required")
    return value


def _validate_normalized_inputs(images):
    _require(isinstance(images, torch.Tensor) and images.device.type == "cpu"
             and images.dtype == torch.float32 and images.ndim == 4
             and tuple(images.shape[1:]) == (3, 224, 224)
             and 1 <= images.shape[0] <= 2 and not images.requires_grad,
             "skelex_small_cpu_normalized_224_batch_required")
    _require(bool(torch.isfinite(images).all()), "skelex_finite_input_required")
    mean = torch.tensor(MEAN, dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).reshape(1, 3, 1, 1)
    gray = images * std + mean
    _require(bool(((gray >= -1e-6) & (gray <= 1 + 1e-6)).all())
             and torch.allclose(gray[:, 0], gray[:, 1], atol=1e-6, rtol=0)
             and torch.allclose(gray[:, 0], gray[:, 2], atol=1e-6, rtol=0),
             "skelex_imagenet_normalized_replicated_gray_required")


def _unmasked_features(encoder, images):
    """MAE mask_ratio=0 still sorts random noise unless noise is supplied."""
    patch_count = (encoder.config.image_size // encoder.config.patch_size) ** 2
    _require(encoder.config.mask_ratio == 0, "skelex_masking_must_be_disabled")
    noise = torch.arange(patch_count, dtype=torch.float32, device="cpu")
    noise = noise.reshape(1, patch_count).expand(images.shape[0], -1)
    output = encoder(pixel_values=images, noise=noise, return_dict=True)
    _require(tuple(output.last_hidden_state.shape) ==
             (images.shape[0], patch_count + 1, encoder.config.hidden_size),
             "skelex_complete_token_sequence_required")
    expected_order = torch.arange(patch_count, device="cpu").expand(images.shape[0], -1)
    _require(bool(torch.count_nonzero(output.mask) == 0)
             and torch.equal(output.ids_restore, expected_order),
             "skelex_mask_or_spatial_permutation_detected")
    return output.last_hidden_state[:, 0, :].contiguous()


class SkelexCPUFeatures(nn.Module):
    """Frozen verification wrapper; an eventual trainer needs its own authority."""

    def __init__(self, encoder):
        super().__init__()
        _require(encoder.config.hidden_size == 1024
                 and encoder.config.image_size == 224
                 and encoder.config.patch_size == 16
                 and encoder.config.num_channels == 3,
                 "skelex_official_encoder_geometry_required")
        self.encoder = encoder.cpu().float()
        self.encoder.config.mask_ratio = 0.0
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        _require(mode is False, "skelex_verification_training_forbidden")
        return super().train(False)

    @torch.inference_mode()
    def forward(self, images):
        _validate_normalized_inputs(images)
        _require(not self.training and not self.encoder.training
                 and all(not parameter.requires_grad and parameter.grad is None
                         for parameter in self.parameters()),
                 "skelex_frozen_eval_state_required")
        features = _unmasked_features(self.encoder, images)
        _require(tuple(features.shape) == (images.shape[0], 1024)
                 and bool(torch.isfinite(features).all()),
                 "skelex_finite_cls_features_required")
        return features


def load_skelex_encoder(checkpoint_path, config_path, processor_path):
    """Verify all local hashes; strictly load safetensors without network access.

    This confirms technical loading only. It does not assert permission to
    redistribute adapted weights or satisfy the later training smoke-test gate.
    """
    checkpoint_path = Path(checkpoint_path)
    _require(checkpoint_path.suffix == ".safetensors" and checkpoint_path.is_file()
             and checkpoint_path.stat().st_size == CHECKPOINT_BYTES,
             "skelex_official_safetensors_file_required")
    config_data = _verified_small_json(config_path, 699, CONFIG_SHA256)
    processor = _verified_small_json(processor_path, 217, PROCESSOR_SHA256)
    _require(processor.get("size") == 224 and processor.get("do_normalize") is True
             and tuple(processor.get("image_mean", [])) == MEAN
             and tuple(processor.get("image_std", [])) == STD,
             "skelex_fixed_input_metadata_required")
    _require(_file_hash(checkpoint_path) == CHECKPOINT_SHA256,
             "skelex_checkpoint_sha256_mismatch")

    import transformers
    from transformers import ViTMAEConfig, ViTMAEForPreTraining
    from safetensors.torch import load_file

    _require(transformers.__version__ == TRANSFORMERS_VERSION,
             "skelex_transformers_4_50_2_required")
    config = ViTMAEConfig.from_dict(config_data)
    config._attn_implementation = "eager"
    # Construction may initialize temporary parameters, but does not consume the
    # caller's RNG stream. The strict state load replaces every parameter.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        complete = ViTMAEForPreTraining(config).cpu().float()
    state = load_file(str(checkpoint_path), device="cpu")
    _require(bool(state) and all(tensor.dtype == torch.float32
             and tensor.device.type == "cpu" and bool(torch.isfinite(tensor).all())
             for tensor in state.values()), "skelex_finite_float32_checkpoint_required")
    complete.load_state_dict(state, strict=True)
    result = SkelexCPUFeatures(complete.vit)
    del state, complete
    return result
