"""Deterministic native radiograph to a common grayscale model input.

The original integer ROI is never edited. Geometry uses physical pixel spacing;
the entire ROI is resized once and padded. This is distinct from localization
and from any encoder-specific channel normalization or training augmentation.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image


VERSION = "physical_letterbox_224_p01_p99_v1.0"
CONTRACT = {
    "version": VERSION,
    "output_shape": [1, 224, 224],
    "dtype": "float32",
    "native_input": "two_dimensional_integer_native_roi",
    "intensity": {"percentiles": [1.0, 99.0], "percentile_method": "linear",
        "calculation_dtype": "float64", "clip": [0.0, 1.0],
        "photometric": "invert_MONOCHROME1_after_clipping_before_padding",
        "dicom_voi_window_applied": False, "dicom_padding_mask_applied": False,
        "modality_rescale_policy": "positive_affine_only_cancels_under_per_image_percentile_normalization"},
    "geometry": {"extent": "height_times_row_spacing_width_times_column_spacing",
        "target": 224, "rounding": "floor_scaled_size_plus_half_minimum_one",
        "resampling": "Pillow_F_BILINEAR", "reducing_gap": None,
        "padding": "centered_zero_extra_pixel_bottom_and_right",
        "orientation": "original_acquisition_no_mirroring", "crop": False},
    "runtime": {"numpy": "1.26.4", "Pillow": "11.3.0"},
    "rgb_replication_applied": False, "checkpoint_normalization_applied": False,
    "augmentation_applied": False,
}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def normalize_native(array, metadata):
    """Return a new float32 image and the fixed per-image intensity window."""
    _require(isinstance(array, np.ndarray) and array.ndim == 2 and array.size > 0
             and array.dtype.kind in {"u", "i"}, "native_integer_2d_roi_required")
    _require(metadata.get("photometric") in {"MONOCHROME1", "MONOCHROME2"},
             "known_native_photometric_required")
    values = np.array(array, dtype=np.float64, order="C", copy=True)
    _require(np.isfinite(values).all(), "finite_native_values_required")
    low, high = np.percentile(values, [1.0, 99.0], method="linear")
    _require(np.isfinite(low) and np.isfinite(high) and high > low,
             "nondegenerate_percentile_window_required")
    values -= low
    values /= high - low
    np.clip(values, 0.0, 1.0, out=values)
    if metadata["photometric"] == "MONOCHROME1":
        values = 1.0 - values
    return np.ascontiguousarray(values, dtype=np.float32), [float(low), float(high)]


def transform_native(array, metadata, contract=None):
    """Return float32 [1,224,224] and an auditable geometry/intensity record."""
    _require(contract is None or contract == CONTRACT, "fixed_tensor_contract_required")
    _require(isinstance(metadata, dict), "native_metadata_required")
    _require(metadata.get("side") in {"RIGHT", "LEFT"}, "known_native_side_required")
    _require(metadata.get("native_spatial_orientation") == "original_acquisition_no_mirroring",
             "original_native_orientation_required")
    for name in ("row_spacing_mm", "column_spacing_mm"):
        value = metadata.get(name)
        _require(type(value) in {int, float} and math.isfinite(value) and value > 0,
                 "positive_finite_native_spacing_required")
    normalized, window = normalize_native(array, metadata)
    h, w = array.shape
    physical_h, physical_w = h * metadata["row_spacing_mm"], w * metadata["column_spacing_mm"]
    _require(math.isfinite(physical_h) and math.isfinite(physical_w)
             and physical_h > 0 and physical_w > 0, "finite_native_physical_extent_required")
    longest = max(physical_h, physical_w)
    scaled_h, scaled_w = 224 * (physical_h / longest), 224 * (physical_w / longest)
    new_h, new_w = (max(1, min(224, math.floor(v + 0.5))) for v in (scaled_h, scaled_w))
    top, left = (224 - new_h) // 2, (224 - new_w) // 2
    bottom, right = 224 - new_h - top, 224 - new_w - left
    image = Image.fromarray(normalized)
    resized = np.asarray(image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR,
                                     reducing_gap=None), dtype=np.float32)
    _require(np.isfinite(resized).all(), "finite_interpolation_required")
    # Positive bilinear weights preserve the range; clip only float roundoff.
    _require(resized.min() >= -1e-6 and resized.max() <= 1 + 1e-6,
             "interpolation_range_violation")
    tensor = np.zeros((1, 224, 224), dtype=np.float32)
    tensor[0, top:top+new_h, left:left+new_w] = np.clip(resized, 0.0, 1.0)
    return tensor, {"native_shape": [h, w], "physical_size_mm": [physical_h, physical_w],
        "resized_shape": [new_h, new_w], "padding_ltrb": [left, top, right, bottom],
        "percentile_window": window, "photometric": metadata["photometric"],
        "inversion_applied": metadata["photometric"] == "MONOCHROME1",
        "output_shape": [1, 224, 224], "dtype": "float32", "orientation_preserved": True,
        "spatial_crop_applied": False, "rounding_error_pixels": [new_h-scaled_h, new_w-scaled_w]}
