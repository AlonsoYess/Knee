"""Read-only instrumentation of the fixed MCR004 candidate, not a new localizer.

The instrumented calculations below follow the MIT-licensed Emory-HITI core at
``UPSTREAM_COMMIT``. See ``third_party/emory_hiti/LICENSE`` and ATTRIBUTION.md.
Every invocation checks the instrumented box and pixels against that unmodified
core and the production adapter. A detected dark contour is a detector output,
not a semantic confirmation of an anonymization mask or another artifact.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from typing import Any

import cv2
import numpy as np
from scipy.signal import savgol_filter

from knee import roi_mcr004 as adapter
from knee.third_party.emory_hiti import pipeline as upstream
from knee.third_party.emory_hiti.config import CropConfig


DIAGNOSTIC_VERSION = "mcr004_equivalent_trace_v1.0"


class DiagnosticMismatch(ValueError):
    """The diagnostic replica does not equal the fixed production calculation."""


def _array_hash(array: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _black_box_trace(image: np.ndarray, cfg: CropConfig):
    """Record exactly the contours/rows selected by the existing detector.

    Original external contours stay in memory only so that intersection can be
    measured on their filled footprint, rather than inferred from a bounding box.
    Polygon approximations are the same ones used for upstream angle filtering.
    """
    _, binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.morph_kernel)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rows, records, selected = [], [], []
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w*h <= cfg.black_box_min_area_px:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025*perimeter, True)
        points = approx.reshape(len(approx), 2)
        if len(points) < 3:
            continue
        angles = [
            upstream.calculate_angle(
                points[j], points[(j+1) % len(points)], points[(j+2) % len(points)]
            )
            for j in range(len(points))
        ]
        sharp_count = sum(1 for angle in angles if 85 <= angle <= 95)
        if not 3 <= sharp_count <= 8:
            continue
        contour_rows = [
            int(points[(j+1) % len(points)][1])
            for j, angle in enumerate(angles) if 80 <= angle <= 100
        ]
        rows.extend(contour_rows)
        records.append({
            "detector_contour_index": int(index),
            "lateral_bbox": [int(x), int(x+w), int(y), int(y+h)],
            "lateral_polygon": points.astype(int).tolist(),
            "bounding_box_area_pixels": int(w*h),
            "sharp_angle_count": int(sharp_count),
            "rows": sorted(set(contour_rows)),
        })
        selected.append(contour)
    rows = sorted(set(rows))
    if rows != [int(value) for value in upstream.get_black_box_rows(image, config=cfg)]:
        raise DiagnosticMismatch("diagnostic_black_box_rows_mismatch")
    return rows, records, selected


def _attach_intersections(records, contours, lateral_shape, x_offset, box):
    """Use half-open ROI bounds on the detector's working pixel grid."""
    x0, x1, y0, y1 = box
    lx0, lx1 = max(0, x0-x_offset), min(lateral_shape[1], x1-x_offset)
    ly0, ly1 = max(0, y0), min(lateral_shape[0], y1)
    for record, contour in zip(records, contours):
        footprint = np.zeros(lateral_shape, dtype=np.uint8)
        cv2.drawContours(footprint, [contour], -1, 1, thickness=cv2.FILLED)
        count = (int(np.count_nonzero(footprint[ly0:ly1, lx0:lx1]))
                 if lx0 < lx1 and ly0 < ly1 else 0)
        bx0, bx1, by0, by1 = record["lateral_bbox"]
        record.update({
            "working_bbox": [bx0+x_offset, bx1+x_offset, by0, by1],
            "working_polygon": [[x+x_offset, y] for x, y in record["lateral_polygon"]],
            "final_roi_intersection_pixels": count,
            "intersects_final_roi": count > 0,
        })


def _instrument_core(working: np.ndarray, is_right: bool, cfg: CropConfig):
    """Mirror only the fixed unilateral calculation; never adjust parameters."""
    intensity_sum = working.sum(0)
    peak = (cfg.intensity_offset + np.argmax(
        intensity_sum[cfg.intensity_offset:-cfg.intensity_offset]
    ) if is_right else np.argmax(intensity_sum[:-cfg.intensity_offset]))
    left_min, right_min = upstream.get_mins(working, peak)
    cropped_horizontal = working[:, left_min:right_min]
    lateral = working[:, :peak] if is_right else working[:, peak:]
    x_offset = 0 if is_right else int(peak)
    width = abs(right_min-left_min)
    crop_peak = np.argmax(cropped_horizontal.sum(0))
    portion = (cropped_horizontal[:, :crop_peak] if is_right
               else cropped_horizontal[:, crop_peak:])

    if portion.dtype != np.uint8:
        portion_scaled = ((portion-portion.min())/(portion.max()-portion.min())*255).astype(np.uint8)
    else:
        portion_scaled = portion
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_tile)
    clahe_img = clahe.apply(portion_scaled)
    updated = clahe_img.sum(1).copy()
    height = clahe_img.shape[0]
    start_row, end_row = int(cfg.start_row_frac*height), int(cfg.end_row_frac*height)
    search_before = [start_row, end_row]

    black_rows, records, contours = _black_box_trace(lateral, cfg)
    before = [row for row in black_rows if row < 0.4*height]
    after = [row for row in black_rows if row > 0.6*height]
    if before:
        start_row = max(start_row, max(before)+cfg.black_box_offset)
    if after:
        end_row = min(end_row, min(after)-cfg.black_box_offset)

    blurred = cv2.GaussianBlur(portion, cfg.gaussian_kernel, 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=cfg.sobel_ksize)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=cfg.sobel_ksize)
    magnitude = np.sqrt(grad_x**2+grad_y**2)
    mag_visual = cv2.convertScaleAbs(magnitude)
    row_sums_portion = portion.sum(axis=1)
    max_row = np.argmax(row_sums_portion) if np.any(row_sums_portion > 0) else 0
    edge_threshold = max(
        np.mean(portion[max_row, :])+cfg.edge_threshold_offset, cfg.edge_threshold_min
    )
    _, binary_mask = cv2.threshold(mag_visual, int(edge_threshold), 255, cv2.THRESH_BINARY)
    noisy_rows = [row for row in range(portion.shape[0])
                  if np.sum(binary_mask[row] > 0) >= 2]
    noisy_regions = []
    if noisy_rows:
        region_start = noisy_rows[0]
        for index in range(1, len(noisy_rows)):
            if noisy_rows[index]-noisy_rows[index-1] > 10:
                noisy_regions.append((region_start, noisy_rows[index-1]))
                region_start = noisy_rows[index]
        noisy_regions.append((region_start, noisy_rows[-1]))

    filtered_regions = []
    for region_start, region_end in noisy_regions:
        overlap = max(0, min(region_end, end_row)-max(region_start, start_row))
        if overlap < 0.5*(end_row-start_row):
            filtered_regions.append((region_start, region_end))
    for start, end in filtered_regions:
        expanded_start, expanded_end = max(0, start-2), min(len(updated), end+2)
        if expanded_start >= expanded_end:
            continue
        before_value = updated[max(0, start-2)]
        after_value = updated[min(len(updated)-1, end+2)]
        updated[expanded_start:expanded_end] = np.linspace(
            before_value, after_value, expanded_end-expanded_start
        )

    # Upstream computes a local ``win`` but applies cfg.savgol_window; preserve it.
    smooth = savgol_filter(updated, cfg.savgol_window, cfg.savgol_poly)
    first_derivative = np.gradient(smooth)
    second_derivative = np.gradient(
        savgol_filter(first_derivative, cfg.savgol_window, cfg.savgol_poly)
    )
    notch = start_row+np.argmin(second_derivative[start_row:end_row])
    box = [int(left_min), int(right_min), int(notch-width//2), int(notch+width//2)]
    crop = cropped_horizontal[max(0, box[2]):min(box[3], working.shape[0]), :]
    _attach_intersections(records, contours, lateral.shape, x_offset, box)
    intersects = any(record["intersects_final_roi"] for record in records)
    route = ("none_detected" if not records else
             "detected_but_final_roi_intersects" if intersects else
             "detected_outside_final_roi")
    return crop, {
        "horizontal_peak": int(peak),
        "left_min": int(left_min), "right_min": int(right_min), "width": int(width),
        "search_band_before": search_before,
        "search_band_after": [int(start_row), int(end_row)],
        "black_box_rows": black_rows,
        "black_box_rows_before": before,
        "black_box_rows_after": after,
        "black_box_lateral_offset_x": x_offset,
        "black_box_lateral_region": [x_offset, x_offset+lateral.shape[1], 0, lateral.shape[0]],
        "black_box_contours": records,
        "notch": int(notch), "requested_working_box": box,
        "mask_intersects_final_roi": intersects, "mask_route": route,
        "mask_intersection_basis": "filled_selected_external_contour_on_working_grid",
        "semantic_mask_confirmation": False,
    }


def trace_for_half(native_half: np.ndarray, patient_side: str,
                   photometric: str = "MONOCHROME2") -> dict[str, Any]:
    """Return JSON-compatible diagnostics only when exact equivalence holds.

    Coordinates refer to the supplied frozen unilateral half, not the bilateral
    DICOM. The caller must compare the result to the CLOSED historical records;
    equivalence here verifies the diagnostic implementation, not that provenance.
    Unit spacing and zero offset used for the reference call affect metadata only;
    physical dimensions are deliberately absent from this diagnostic result.
    """
    raw = np.asarray(native_half)
    initial_hash = _array_hash(raw)
    reference = adapter.candidate_for_half(raw, patient_side, photometric, 0, 1.0, 1.0)
    if reference["status"] != "candidate":
        raise ValueError("diagnostic_requires_existing_candidate:"+reference["error_code"])
    cfg = CropConfig()
    working, coordinate_trace = adapter.preprocess_with_trace(
        adapter.normalize_half(raw, photometric), cfg
    )
    replica, details = _instrument_core(working, patient_side == "RIGHT", cfg)
    core_crop, core_box = upstream.process_knee_side(
        working, is_right_knee=patient_side == "RIGHT", config=cfg, return_geometry=True
    )
    box = details["requested_working_box"]
    if box != [int(value) for value in core_box]:
        raise DiagnosticMismatch("diagnostic_upstream_working_box_mismatch")
    if replica.dtype != core_crop.dtype or not np.array_equal(replica, core_crop):
        raise DiagnosticMismatch("diagnostic_upstream_working_pixels_mismatch")
    if box != reference["requested_working_box"]:
        raise DiagnosticMismatch("diagnostic_adapter_working_box_mismatch")
    native_box = list(coordinate_trace.to_native_bounds(tuple(box)))
    if native_box != reference["native_half_box"]:
        raise DiagnosticMismatch("diagnostic_adapter_native_box_mismatch")
    if asdict(coordinate_trace) != reference["trace"]:
        raise DiagnosticMismatch("diagnostic_adapter_coordinate_trace_mismatch")
    x0, x1, y0, y1 = native_box
    native_crop = raw[y0:y1, x0:x1]
    if (native_crop.dtype != reference["crop"].dtype or
            not np.array_equal(native_crop, reference["crop"])):
        raise DiagnosticMismatch("diagnostic_adapter_native_pixels_mismatch")
    if _array_hash(raw) != initial_hash:
        raise DiagnosticMismatch("diagnostic_input_pixels_changed")
    for record in details["black_box_contours"]:
        record["native_half_bbox"] = list(
            coordinate_trace.to_native_bounds(tuple(record["working_bbox"]))
        )
    return {
        "schema_version": 1, "diagnostic_version": DIAGNOSTIC_VERSION,
        "status": "equivalent_candidate", "patient_side": patient_side,
        "photometric": photometric, "algorithm_version": adapter.ALGORITHM_VERSION,
        "upstream_commit": adapter.UPSTREAM_COMMIT,
        "trace": asdict(coordinate_trace),
        "working_shape": [int(value) for value in working.shape],
        "coordinate_convention": "half-open boxes [x0,x1,y0,y1]; polygon vertices are pixel indices",
        **details,
        "native_half_box": native_box,
        "native_crop_shape": [int(value) for value in native_crop.shape],
        "native_crop_dtype": str(native_crop.dtype),
        "native_crop_sha256": _array_hash(native_crop),
        "equivalence_checks": {
            "black_box_rows": True, "upstream_working_box": True,
            "upstream_working_pixels": True, "adapter_working_box": True,
            "adapter_native_box": True, "adapter_coordinate_trace": True,
            "adapter_native_pixels": True, "input_pixels_unchanged": True,
        },
        "historical_record_comparison_required": True,
        "production_crop_written": False, "review_decision_changed": False,
    }
