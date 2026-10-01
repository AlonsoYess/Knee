"""Traceable unilateral ROI candidate for approved MCR-2026-004.

This is an adapter around a pinned, MIT-licensed Emory-HITI core. It does not
assign clinical labels or decide visual acceptability. See docs/25 and
third_party/emory_hiti/LICENSE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor, isfinite
from typing import Any

import cv2
import numpy as np

from knee.third_party.emory_hiti.config import CropConfig
from knee.third_party.emory_hiti.pipeline import (
    find_horizontal_line,
    find_vertical_line,
    preprocess_and_crop,
    process_knee_side,
)


UPSTREAM_COMMIT = "c70a2314bdbeed0d2fba3c2c2c652782ce3a2b93"
ALGORITHM_VERSION = "tibiofemoral_roi_mcr004_v0.1_candidate"


class Abstain(ValueError):
    """A candidate could not be generated without violating the fixed contract."""


@dataclass(frozen=True)
class Trace:
    input_rows: int
    input_columns: int
    valid_x0: int
    valid_x1: int
    valid_y0: int
    valid_y1: int
    resized_rows: int
    resized_columns: int
    trim_top: int
    line_x0: int
    line_x1: int
    line_y0: int
    line_y1: int

    def to_native_bounds(self, bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0, x1, y0, y1 = bounds
        if not (0 <= x0 < x1 <= self.line_x1-self.line_x0):
            raise Abstain("requested_horizontal_bounds_invalid")
        if not (0 <= y0 < y1 <= self.line_y1-self.line_y0):
            raise Abstain("requested_vertical_bounds_truncated")
        sx = (self.valid_x1-self.valid_x0)/self.resized_columns
        sy = (self.valid_y1-self.valid_y0)/self.resized_rows
        native = (
            floor(self.valid_x0 + (self.line_x0+x0)*sx),
            ceil(self.valid_x0 + (self.line_x0+x1)*sx),
            floor(self.valid_y0 + (self.trim_top+self.line_y0+y0)*sy),
            ceil(self.valid_y0 + (self.trim_top+self.line_y0+y1)*sy),
        )
        nx0, nx1, ny0, ny1 = native
        if not (0 <= nx0 < nx1 <= self.input_columns and
                0 <= ny0 < ny1 <= self.input_rows):
            raise Abstain("inverse_transform_out_of_bounds")
        return native


def normalize_half(native_half: np.ndarray, photometric: str) -> np.ndarray:
    """p1/p99 8-bit *working copy*; the returned native crop uses original pixels."""
    raw = np.asarray(native_half)
    if raw.ndim != 2 or min(raw.shape) < 32:
        raise Abstain("unsupported_pixel_geometry")
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise Abstain("unsupported_photometric_interpretation")
    values = raw.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise Abstain("non_finite_pixel_array")
    low, high = np.percentile(values, [1, 99])
    if not isfinite(low) or not isfinite(high) or high <= low:
        raise Abstain("constant_pixel_array")
    result = np.clip((values-low)/(high-low), 0, 1)
    if photometric == "MONOCHROME1":
        result = 1-result
    return np.rint(255*result).astype(np.uint8)


def preprocess_with_trace(image: np.ndarray, cfg: CropConfig) -> tuple[np.ndarray, Trace]:
    """Reproduce upstream pre-crop operations and retain every coordinate transform.

    Unlike the upstream permissive fallback, invalid intermediate geometry
    abstains. A byte equality check against the pinned upstream routine is
    required before using the result.
    """
    if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 32:
        raise Abstain("invalid_working_image")
    rows, cols = image.shape
    col_sums = image.sum(0)
    row_sums = image.sum(1)
    valid_cols = np.where((col_sums > 0) & (col_sums < 254*rows))[0]
    valid_rows = np.where((row_sums > 0) & (row_sums < 254*cols))[0]
    x0, x1, y0, y1 = 0, cols, 0, rows
    if len(valid_rows) and len(valid_cols):
        x0, x1 = int(valid_cols[0]), int(valid_cols[-1])
        y0, y1 = int(valid_rows[0]), int(valid_rows[-1])
    if x1 <= x0 or y1 <= y0:
        raise Abstain("invalid_valid_pixel_crop")
    cropped = image[y0:y1, x0:x1]
    resized_cols = int(cfg.resize_height * cropped.shape[1]/cropped.shape[0])
    if resized_cols <= 2*cfg.intensity_offset or cfg.resize_height <= 2*cfg.vertical_trim:
        raise Abstain("preprocessed_image_too_small")
    resized = cv2.resize(cropped, (resized_cols, cfg.resize_height))
    trimmed = resized[cfg.vertical_trim:-cfg.vertical_trim, :]
    height, width = trimmed.shape
    sobelx = cv2.Sobel(trimmed, cv2.CV_64F, 1, 0, ksize=cfg.sobel_ksize)
    sobely = cv2.Sobel(trimmed, cv2.CV_64F, 0, 1, ksize=cfg.sobel_ksize)
    magnitude = np.uint8(np.sqrt(sobelx**2+sobely**2))
    edges = np.uint8(magnitude > cfg.sobel_threshold)*255
    fraction = cfg.line_search_frac
    left_region = edges[:, :int(width*fraction)]
    right_region = edges[:, int(width*(1-fraction)):]
    top_region = edges[:int(height*fraction), :]
    bottom_region = edges[int(height*(1-fraction)):, :]
    left_line = find_vertical_line(left_region, 0, config=cfg)
    right_line = find_vertical_line(right_region, int(width*(1-fraction)), config=cfg)
    top_line = find_horizontal_line(top_region, 0, config=cfg)
    bottom_line = find_horizontal_line(bottom_region, int(height*(1-fraction)), config=cfg)
    lx0 = max((x for x, _ in left_line), default=0) if left_line else 0
    lx1 = min((x for x, _ in right_line), default=width) if right_line else width
    ly0 = max((y for _, y in top_line), default=0) if top_line else 0
    ly1 = min((y for _, y in bottom_line), default=height) if bottom_line else height
    if not (0 <= lx0 < lx1 <= width and 0 <= ly0 < ly1 <= height):
        raise Abstain("invalid_line_crop")
    result = trimmed[ly0:ly1, lx0:lx1]
    original = preprocess_and_crop(image, config=cfg)
    if not np.array_equal(result, original):
        raise Abstain("upstream_preprocess_mismatch")
    trace = Trace(rows, cols, x0, x1, y0, y1, cfg.resize_height,
                  resized_cols, cfg.vertical_trim, lx0, lx1, ly0, ly1)
    return result, trace


def candidate_for_half(
    native_half: np.ndarray,
    side: str,
    photometric: str,
    half_x0: int,
    row_spacing_mm: float,
    column_spacing_mm: float,
    cfg: CropConfig | None = None,
) -> dict[str, Any]:
    """Return one candidate or a documented per-knee abstention, never an approval."""
    base = {"status": "abstain", "algorithm_version": ALGORITHM_VERSION,
            "upstream_commit": UPSTREAM_COMMIT, "side": side}
    try:
        if side not in {"LEFT", "RIGHT"}:
            raise Abstain("invalid_verified_laterality")
        if half_x0 < 0:
            raise Abstain("invalid_half_offset")
        if not (isfinite(row_spacing_mm) and isfinite(column_spacing_mm) and
                row_spacing_mm > 0 and column_spacing_mm > 0):
            raise Abstain("invalid_pixel_spacing")
        image = normalize_half(native_half, photometric)
        config = cfg or CropConfig()
        if asdict(config) != asdict(CropConfig()):
            raise Abstain("non_pinned_core_parameters")
        working, trace = preprocess_with_trace(image, config)
        if min(working.shape) <= 2*config.intensity_offset or working.shape[0] < config.savgol_window:
            raise Abstain("insufficient_signal_geometry")
        # Side orientation is inherited from the frozen bilateral separation.
        # No mirror or external bilateral separator is applied.
        upstream_crop, requested = process_knee_side(
            working, is_right_knee=(side == "RIGHT"),
            config=config, return_geometry=True,
        )
        x0, x1, y0, y1 = (int(v) for v in requested)
        if upstream_crop.size == 0 or y0 < 0 or y1 > working.shape[0]:
            raise Abstain("requested_crop_truncated")
        if not np.array_equal(upstream_crop, working[y0:y1, x0:x1]):
            raise Abstain("upstream_core_mismatch")
        native_box = trace.to_native_bounds((x0, x1, y0, y1))
        nx0, nx1, ny0, ny1 = native_box
        crop = np.ascontiguousarray(np.asarray(native_half)[ny0:ny1, nx0:nx1])
        if crop.size == 0 or not np.array_equal(crop, np.asarray(native_half)[ny0:ny1, nx0:nx1]):
            raise Abstain("native_pixel_integrity_failure")
        return {
            **base, "status": "candidate", "error_code": "",
            "requested_working_box": [x0, x1, y0, y1],
            "native_half_box": [nx0, nx1, ny0, ny1],
            "native_dicom_box": [nx0+half_x0, nx1+half_x0, ny0, ny1],
            "crop_rows": int(ny1-ny0), "crop_columns": int(nx1-nx0),
            "crop_height_mm": (ny1-ny0)*row_spacing_mm,
            "crop_width_mm": (nx1-nx0)*column_spacing_mm,
            "trace": asdict(trace), "crop": crop,
        }
    except (ValueError, IndexError, cv2.error) as exc:
        return {**base, "error_code": str(exc)}
    except Exception as exc:
        # Preserve the 20-knee denominator even for unexpected per-knee faults.
        # The type is recorded without leaking source paths or DICOM metadata.
        return {**base, "error_code": f"unexpected_{type(exc).__name__}"}
