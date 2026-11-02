"""Exact ConvNeXt V2 Tiny FCMAE transfer encoder, without external code loading.

Architecture and GRN remapping adapted from Meta's official release:
https://github.com/facebookresearch/ConvNeXt-V2/tree/2553895753323c6fe0b2bf390683f5ea358a42b9
Copyright (c) Meta Platforms, Inc. and affiliates. Code: MIT; see the adjacent
licenses/convnext_v2_MIT.txt. Released weights separately use CC-BY-NC-4.0.

This module does not download files, fit a classifier, or execute training.
The FCMAE checkpoint has no classification head or final pooling LayerNorm.
The latter is explicitly initialized to weight=1, bias=0, as in official
fine-tuning. Its two parameters are never described as pretrained.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SOURCE_REVISION = "2553895753323c6fe0b2bf390683f5ea358a42b9"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt"
WEIGHTS_SHA256 = "b25eadb89ad634b5c925ea04cdacebafe3e462486baf9482dc2d832bb9af57ef"
WEIGHTS_BYTES = 111520513
FEATURE_DIM = 768
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INITIALIZED_KEYS = frozenset({"norm.weight", "norm.bias"})


def _require(condition, code):
    if not condition:
        raise ValueError(code)


class LayerNorm(nn.Module):
    def __init__(self, channels, channels_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = 1e-6
        self.channels_first = channels_first

    def forward(self, x):
        if not self.channels_first:
            return F.layer_norm(x, (self.weight.numel(),), self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


class GRN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x):
        magnitude = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        response = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * response) + self.beta + x


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.grn = GRN(4 * channels)
        self.pwconv2 = nn.Linear(4 * channels, channels)
        # Official drop-path=0: no extra stochastic augmentation is introduced.
        self.drop_path = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.grn(self.act(self.pwconv1(self.norm(x)))))
        return residual + self.drop_path(x.permute(0, 3, 1, 2))


class ConvNeXtV2Tiny(nn.Module):
    """Official dense Tiny feature encoder; forward returns [batch, 768]."""

    def __init__(self):
        super().__init__()
        self.depths = (3, 3, 9, 3)
        self.dims = (96, 192, 384, 768)
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 96, 4, stride=4), LayerNorm(96, channels_first=True)),
            *[nn.Sequential(LayerNorm(self.dims[index], channels_first=True),
                nn.Conv2d(self.dims[index], self.dims[index + 1], 2, stride=2))
              for index in range(3)],
        ])
        self.stages = nn.ModuleList([
            nn.Sequential(*[Block(channels) for _ in range(depth)])
            for depth, channels in zip(self.depths, self.dims)
        ])
        self.norm = nn.LayerNorm(FEATURE_DIM, eps=1e-6)

    def forward_features(self, image):
        _require(isinstance(image, torch.Tensor) and image.ndim == 4
            and tuple(image.shape[1:]) == (3, 224, 224) and image.shape[0] > 0,
            "convnextv2_requires_normalized_rgb_224_batch")
        for downsample, stage in zip(self.downsample_layers, self.stages):
            image = stage(downsample(image))
        return self.norm(image.mean((-2, -1)))

    def forward(self, image):
        return self.forward_features(image)


def normalize_sealed_image(array):
    """Replicate gray channels and normalize; never resize, crop, or fit pixels."""
    _require(isinstance(array, np.ndarray) and array.shape == (1, 224, 224)
        and array.dtype == np.dtype("float32") and array.flags.c_contiguous
        and not array.flags.writeable and np.isfinite(array).all()
        and array.min() >= 0 and array.max() <= 1, "sealed_readonly_zero_one_float32_image_required")
    image = torch.from_numpy(array.copy()).repeat(3, 1, 1)
    mean = torch.tensor(MEAN, dtype=torch.float32).reshape(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).reshape(3, 1, 1)
    return ((image - mean) / std).contiguous()


def _remap_encoder_tensor(source_key, value):
    """Published FCMAE encoder is dense already; only GRN needs two axes."""
    if ".grn." in source_key:
        _require(value.ndim == 2 and value.shape[0] == 1, "fcmae_published_grn_shape_invalid")
        value = value.unsqueeze(0).unsqueeze(1)
    return source_key, value.contiguous()


def load_fcmae_state(model, checkpoint):
    """Adopt only an exact FCMAE encoder, accounting for every source tensor.

    Caller verifies the downloaded file's full digest before safe deserialization
    with torch.load(..., map_location='cpu', weights_only=True). This function
    receives that mapping; it never invokes pickle, network, or optimizer code.
    """
    _require(isinstance(model, ConvNeXtV2Tiny), "exact_convnextv2_tiny_encoder_required")
    _require(isinstance(checkpoint, Mapping) and set(checkpoint) == {"model"}
        and isinstance(checkpoint.get("model"), Mapping),
        "fcmae_checkpoint_model_mapping_required")
    source = checkpoint["model"]
    _require(bool(source), "fcmae_checkpoint_empty")
    expected = model.state_dict()
    adopted = {}
    _require(set(source) == set(expected) - INITIALIZED_KEYS, "fcmae_exact_published_encoder_keys_required")
    for source_key, value in source.items():
        _require(isinstance(source_key, str) and isinstance(value, torch.Tensor)
            and value.device.type == "cpu" and value.dtype == torch.float32
            and value.layout == torch.strided and bool(torch.isfinite(value).all()),
            "fcmae_tensor_invalid_or_nonfinite")
        target_key, mapped = _remap_encoder_tensor(source_key, value)
        _require(target_key in expected and target_key not in INITIALIZED_KEYS,
            "fcmae_unexpected_encoder_key:" + source_key)
        _require(target_key not in adopted, "fcmae_duplicate_remapped_key:" + target_key)
        _require(mapped.shape == expected[target_key].shape, "fcmae_encoder_shape_mismatch:" + target_key)
        adopted[target_key] = mapped
    _require(set(adopted) == set(expected) - INITIALIZED_KEYS, "fcmae_missing_encoder_tensors")
    # Make the only missing affine transform explicit, deterministic, and auditable.
    adopted["norm.weight"] = torch.ones_like(expected["norm.weight"])
    adopted["norm.bias"] = torch.zeros_like(expected["norm.bias"])
    model.load_state_dict(adopted, strict=True)
    return {"source_revision": SOURCE_REVISION, "source_tensors": len(source),
        "checkpoint_format": "official_released_dense_fcmae_encoder",
        "loaded_pretrained_tensors": len(adopted) - len(INITIALIZED_KEYS),
        "ignored_reconstruction_tensor_keys": [],
        "initialized_nonpretrained_parameter_keys": sorted(INITIALIZED_KEYS),
        "classification_head_loaded": False, "feature_dimension": FEATURE_DIM,
        "all_source_tensors_accounted": len(source) == len(adopted) - len(INITIALIZED_KEYS)}
