"""Prespecified, training-only intensity variation on sealed native input copies.

The moderate gamma range addresses small intensity variation without moving,
cropping, mirroring, or interpolating an anatomical pixel. Zero-valued padding
and endpoint intensities are preserved. This starting policy is not a claim of
empirically improved predictive performance. It must run BEFORE checkpoint
normalization, only for the training subset of a development fold.

Parameters depend on seed, fold, one-based epoch and knee identity, never the
target, input order, worker scheduling or global RNG. Replaying an AMP group
or resuming the same epoch therefore yields exactly the same transformed input.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np


VERSION = "training_intensity_gamma_v1.0"
_POLICY = {
    "version": VERSION,
    "operation": "gamma_power_on_zero_one_grayscale",
    "probability": 0.5,
    "gamma_minimum": 0.9,
    "gamma_maximum": 1.1,
    "split": "development_training_only",
    "normalization_order": "before_checkpoint_specific_normalization",
    "randomness": "sha256_seed_fold_epoch_participant_side_without_target",
    "geometry_modified": False,
    "laterality_modified": False,
    "zero_padding_preserved": True,
    "originals_modified": False,
    "validation_inference": "identity",
    "empirical_improvement_claimed": False,
}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("finite_json_augmentation_policy_required") from exc


POLICY_SHA256 = hashlib.sha256(_canonical(_POLICY)).hexdigest()


def policy_document():
    """Return a mutable copy for inclusion in the reviewed release manifest."""
    return deepcopy(_POLICY)


def validate_policy(policy):
    """Reject silent changes to magnitudes, probabilities or split boundaries."""
    _require(isinstance(policy, dict)
             and hashlib.sha256(_canonical(policy)).hexdigest() == POLICY_SHA256,
             "reviewed_training_augmentation_policy_required")
    return POLICY_SHA256


def _validate_image(array):
    _require(isinstance(array, np.ndarray) and array.dtype == np.dtype("float32")
             and array.shape == (1, 224, 224) and array.flags.c_contiguous
             and not array.flags.writeable, "sealed_readonly_grayscale_image_required")
    _require(bool(np.isfinite(array).all()) and array.min() >= 0 and array.max() <= 1,
             "finite_zero_one_grayscale_required_before_augmentation")


def sample_parameters(*, seed, validation_fold, epoch, src_subject_id, side, policy=None):
    """Deterministic independent draws; return the realized power, not a fitted value."""
    validate_policy(_POLICY if policy is None else policy)
    _require(type(seed) is int and 0 <= seed <= 2**32 - 1,
             "bounded_integer_augmentation_seed_required")
    _require(type(validation_fold) is int and validation_fold in range(5),
             "development_validation_fold_required_for_augmentation")
    _require(type(epoch) is int and 1 <= epoch <= 60,
             "one_based_training_epoch_required_for_augmentation")
    _require(isinstance(src_subject_id, str) and 0 < len(src_subject_id) <= 128
             and src_subject_id.strip() == src_subject_id
             and all(ord(character) >= 32 for character in src_subject_id)
             and side in {"LEFT", "RIGHT"}, "valid_knee_identity_required_for_augmentation")
    key = {"policy_sha256": POLICY_SHA256, "seed": seed, "validation_fold": validation_fold,
           "epoch": epoch, "src_subject_id": src_subject_id, "side": side}
    raw = hashlib.sha256(_canonical(key)).digest()
    # Fifty-three bits produce [0, 1) exactly representable in binary64 and do
    # not depend on NumPy/PyTorch generator implementations or process hash salt.
    draw_apply = (int.from_bytes(raw[:8], "big") >> 11) / 2**53
    draw_gamma = (int.from_bytes(raw[8:16], "big") >> 11) / 2**53
    applied = draw_apply < _POLICY["probability"]
    gamma = (_POLICY["gamma_minimum"] + draw_gamma *
             (_POLICY["gamma_maximum"] - _POLICY["gamma_minimum"])) if applied else 1.0
    _require(math.isfinite(gamma) and 0.9 <= gamma <= 1.1, "finite_reviewed_gamma_required")
    return {"policy_version": VERSION, "policy_sha256": POLICY_SHA256,
            "applied": applied, "gamma": gamma}


def augment_training_image(array, *, seed, validation_fold, epoch, src_subject_id,
                           side, policy=None, split="training"):
    """Produce a read-only float32 copy; any non-training request is rejected."""
    _require(split == "training", "augmentation_is_development_training_only")
    _validate_image(array)
    parameters = sample_parameters(seed=seed, validation_fold=validation_fold, epoch=epoch,
        src_subject_id=src_subject_id, side=side, policy=policy)
    output = np.array(array, dtype=np.float32, order="C", copy=True)
    if parameters["applied"]:
        # Use binary64 for the operation, then materialize a float32 input copy.
        # Preserve positive subnormals rather than allowing rounding to create
        # a new zero-padding pixel. This is only a numerical endpoint safeguard.
        transformed = np.power(array.astype(np.float64), parameters["gamma"])
        minimum_positive = np.nextafter(np.float32(0), np.float32(1))
        positive = array > 0
        transformed[positive] = np.maximum(transformed[positive], minimum_positive)
        output[:] = transformed.astype(np.float32)
    _require(bool(np.isfinite(output).all()) and output.min() >= 0 and output.max() <= 1
             and bool(np.array_equal(output == 0, array == 0)),
             "augmentation_output_or_zero_padding_invalid")
    output.setflags(write=False)
    return output


def apply_image_policy(array, *, split, seed=None, validation_fold=None, epoch=None,
                       src_subject_id=None, side=None, policy=None):
    """Keep validation/inference identity; apply the policy only to training.

    Passing training-randomness arguments on validation/inference is rejected,
    so accidentally routing an augmented training call into evaluation fails.
    """
    _require(split in {"training", "validation", "inference"}, "known_image_split_required")
    validate_policy(_POLICY if policy is None else policy)
    if split == "training":
        return augment_training_image(array, seed=seed, validation_fold=validation_fold,
            epoch=epoch, src_subject_id=src_subject_id, side=side, policy=policy)
    _require(all(value is None for value in (seed, validation_fold, epoch, src_subject_id, side)),
             "evaluation_must_not_receive_training_augmentation_context")
    _validate_image(array)
    return array
