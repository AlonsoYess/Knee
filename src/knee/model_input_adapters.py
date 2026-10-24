"""Fixed CPU adapters for sealed development images and clinical fold inputs.

The original single-channel tensor already has its final physical geometry.
Only channel replication and the explicit checkpoint normalization occur here.
No model, checkpoint download, augmentation, or fitted pixel statistic is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import DenseNet121_Weights, ViT_B_16_Weights

from knee.colab_data import validate_rows

VERSION = "fixed_development_input_adapters_v1.0"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EncoderInput:
    weights: object
    weights_id: str
    source_url: str
    mean: tuple = MEAN
    std: tuple = STD
    crop_size: tuple = (224,)


MODELS = MappingProxyType({
    "image_densenet121": EncoderInput(DenseNet121_Weights.IMAGENET1K_V1,
        "DenseNet121_Weights.IMAGENET1K_V1",
        "https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.densenet121.html"),
    "image_vit_b16": EncoderInput(ViT_B_16_Weights.IMAGENET1K_V1,
        "ViT_B_16_Weights.IMAGENET1K_V1",
        "https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.vit_b_16.html"),
})


def _require(condition, code):
    if not condition:
        raise ValueError(code)


@lru_cache(maxsize=2)
def _verified_spec(encoder_id):
    _require(isinstance(encoder_id, str) and encoder_id in MODELS,
        "encoder_checkpoint_or_license_not_adopted")
    _require(np.__version__ == "1.26.4" and torch.__version__ == "2.6.0+cpu"
        and torchvision.__version__ == "0.21.0+cpu", "frozen_cpu_adapter_runtime_required")
    spec = MODELS[encoder_id]
    # Query the explicit weights metadata only; never execute its resize/crop.
    metadata = spec.weights.transforms()
    _require(tuple(metadata.mean) == spec.mean and tuple(metadata.std) == spec.std
        and tuple(metadata.crop_size) == spec.crop_size, "explicit_checkpoint_input_metadata_changed")
    return spec


def normalize_image(array, encoder_id):
    """Return contiguous CPU float32 [3,224,224], preserving the source array."""
    spec = _verified_spec(encoder_id)
    _require(isinstance(array, np.ndarray) and array.shape == (1, 224, 224)
        and array.dtype == np.dtype("float32") and array.flags.c_contiguous
        and not array.flags.writeable, "sealed_readonly_float32_image_required")
    _require(np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1,
        "finite_zero_one_image_range_required")
    copied = np.array(array, dtype=np.float32, order="C", copy=True)
    image = torch.from_numpy(copied).repeat(3, 1, 1)
    mean = torch.tensor(spec.mean, dtype=torch.float32, device="cpu").reshape(3, 1, 1)
    scale = torch.tensor(spec.std, dtype=torch.float32, device="cpu").reshape(3, 1, 1)
    return ((image - mean) / scale).contiguous()


class DevelopmentFoldDataset(Dataset):
    """Expose a fixed development fold with aligned image/clinical/label rows."""

    def __init__(self, rows, tensors, clinical_state, encoder_id, split):
        _verified_spec(encoder_id)
        validate_rows(rows, "development")
        _require(split in {"training", "validation"}, "development_training_or_validation_split_required")
        _require(isinstance(clinical_state, dict) and type(clinical_state.get("validation_fold")) is int
            and 0 <= clinical_state["validation_fold"] <= 4, "valid_clinical_fold_state_required")
        expected = {(r["src_subject_id"], r["side"]) for r in rows}
        _require(isinstance(tensors, dict) and set(tensors) == expected, "exact_development_image_mapping_required")
        fold = clinical_state["validation_fold"]
        selected = [r for r in rows if (r["cv_fold"] == fold) == (split == "validation")]
        _require(bool(selected), "nonempty_development_fold_required")
        training_ids = sorted([[r["src_subject_id"], r["side"]] for r in rows if r["cv_fold"] != fold])
        training_sha = hashlib.sha256(json.dumps(training_ids, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        _require(clinical_state.get("train_identity_sha256") == training_sha
            and clinical_state.get("train_count") == len(training_ids)
            and clinical_state.get("train_participants") == len({identity[0] for identity in training_ids}),
            "clinical_state_does_not_match_development_training_fold")
        # Imported lazily so the image-only adapter cannot trigger clinical fit.
        from knee.clinical_fold_inputs import transform_clinical_rows
        clinical = transform_clinical_rows(selected, clinical_state)
        _require(isinstance(clinical, np.ndarray) and clinical.dtype == np.dtype("float32")
            and clinical.shape == (len(selected), 3) and np.isfinite(clinical).all(),
            "three_finite_aligned_clinical_features_required")
        self.encoder_id, self.split, self.validation_fold = encoder_id, split, fold
        self.identities = tuple((r["src_subject_id"], r["side"]) for r in selected)
        self._targets = tuple(r["PROGRESION_48M"] for r in selected)
        self._clinical = np.array(clinical, dtype=np.float32, order="C", copy=True)
        self._clinical.setflags(write=False)
        self._tensors = {identity: tensors[identity] for identity in self.identities}
        # Verify every selected source before the first batch can be yielded.
        for source in self._tensors.values():
            _require(isinstance(source, np.ndarray) and source.shape == (1, 224, 224)
                and source.dtype == np.dtype("float32") and source.flags.c_contiguous
                and not source.flags.writeable and np.isfinite(source).all()
                and source.min() >= 0 and source.max() <= 1, "sealed_readonly_finite_image_mapping_required")

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, index):
        subject, side = self.identities[index]
        return {"image": normalize_image(self._tensors[(subject, side)], self.encoder_id),
            "clinical": torch.tensor(self._clinical[index].copy(), dtype=torch.float32, device="cpu"),
            "target": torch.tensor(self._targets[index], dtype=torch.float32, device="cpu"),
            "src_subject_id": subject, "side": side}


def make_development_loader(dataset, batch_size=8):
    """Use deterministic ordinary CPU collation; every selected row is retained."""
    _require(isinstance(dataset, DevelopmentFoldDataset) and type(batch_size) is int and batch_size > 0,
        "development_dataset_and_positive_batch_size_required")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
