"""Prepare a blinded pilot for tibiofemoral localization and knee cropping."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from knee.bilateral_separation import (
    _read_single_image,
    _resolve_package,
    _smooth_profile,
    normalize_working_copy,
)
from knee.dicom_audit import sha256_bytes, sha256_file


PRIVATE_RESULT_FIELDS = (
    "case_alias",
    "knee_alias",
    "patient_side",
    "manifest_key",
    "package_relative_path",
    "package_sha256",
    "dicom_sha256",
    "pixel_sha256",
    "native_crop_sha256",
    "spacing_source",
    "row_spacing_mm",
    "column_spacing_mm",
    "half_x0",
    "half_x1",
    "joint_center_x_half",
    "joint_center_y",
    "joint_center_x_fraction",
    "joint_center_y_fraction",
    "crop_x0_half",
    "crop_x1_half",
    "crop_y0",
    "crop_y1",
    "crop_rows",
    "crop_columns",
    "crop_height_mm",
    "crop_width_mm",
    "score_prominence",
    "bone_contrast_score",
    "left_compartment_peak_y",
    "right_compartment_peak_y",
    "compartment_peak_distance_fraction",
    "compartment_agreement_score",
    "inner_edge_clearance_mm",
    "boundary_shift_fraction",
    "background_fraction",
    "saturation_fraction",
    "localization_strategy",
    "left_femoral_edge_y",
    "left_tibial_edge_y",
    "right_femoral_edge_y",
    "right_tibial_edge_y",
    "left_joint_width_mm",
    "right_joint_width_mm",
    "left_pair_score",
    "right_pair_score",
    "directed_edge_strength_score",
    "compartment_width_difference_mm",
    "vertical_edge_gate_passed",
    "compartment_consensus_gate_passed",
    "detected_artifact_components",
    "detected_artifact_extent_mm",
    "effective_inner_margin_mm",
    "artifact_clearance_gate_passed",
    "boundary_gate_passed",
    "background_gate_passed",
    "saturation_gate_passed",
    "mandatory_gates_passed",
    "review_reasons",
    "confidence_score",
    "confidence_level",
    "technical_status",
    "preview_file",
    "native_crop_file",
    "error_code",
)

REVIEW_FIELDS = (
    "knee_alias",
    "case_alias",
    "patient_side",
    "preview_file",
    "automatic_confidence_level",
    "automatic_joint_center_y_fraction",
    "automatic_crop_height_mm",
    "automatic_crop_width_mm",
    "joint_centered_yes_no",
    "tibiofemoral_anatomy_complete_yes_no",
    "text_borders_and_rule_excluded_yes_no",
    "crop_acceptable_yes_no",
    "technical_exclusion_yes_no",
    "exclusion_reason",
    "reviewed_without_outcome_yes_no",
    "reviewer_notes",
)


def _robust_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(values, [10.0, 90.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}_must_be_an_object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def _load_closed_bilateral_pilot(
    bilateral_output_dir: Path,
    expected_unique_studies: int,
    expected_algorithm_version: str,
) -> list[dict[str, str]]:
    closure = _read_json(bilateral_output_dir / "cierre_paso_3_publico.json")
    parameters = _read_json(bilateral_output_dir / "parametros_congelados.json")
    results_path = bilateral_output_dir / "resultados_separacion_privados.csv"
    results = _read_csv(results_path)
    if closure.get("status") != "closed" or not closure.get("parameters_frozen"):
        raise ValueError("bilateral_pilot_is_not_closed")
    if closure.get("algorithm_version") != expected_algorithm_version:
        raise ValueError("unexpected_bilateral_algorithm_version")
    if parameters.get("status") != "frozen_after_blinded_visual_review":
        raise ValueError("bilateral_parameters_are_not_frozen")
    if parameters.get("algorithm_version") != expected_algorithm_version:
        raise ValueError("bilateral_parameter_version_mismatch")
    if parameters.get("proposed_mapping") != {
        "image_left": "RIGHT",
        "image_right": "LEFT",
    }:
        raise ValueError("unexpected_frozen_laterality_mapping")
    expected_hash = str(parameters.get("results_csv_sha256", ""))
    if not expected_hash or sha256_file(results_path) != expected_hash:
        raise ValueError("bilateral_results_integrity_mismatch")
    if len(results) != expected_unique_studies:
        raise ValueError("unexpected_bilateral_result_count")
    aliases = {row.get("case_alias", "") for row in results}
    if len(aliases) != expected_unique_studies or "" in aliases:
        raise ValueError("duplicate_or_missing_case_alias")
    for row in results:
        if row.get("error_code", "").strip():
            raise ValueError("bilateral_result_contains_error")
        if row.get("technical_status") not in {
            "CANDIDATE_OK",
            "REVIEW_REQUIRED_BOUNDARY",
        }:
            raise ValueError("unexpected_bilateral_technical_status")
    return sorted(results, key=lambda row: row["case_alias"])


def _load_verified_pilot(audit_json: Path, expected: int) -> dict[str, dict[str, str]]:
    payload = _read_json(audit_json)
    summary = payload.get("public_summary", {})
    selected = payload.get("private_reconciliation", {}).get("selected_packages", [])
    if summary.get("status") != "ok" or len(selected) != expected:
        raise ValueError("pilot_audit_is_not_closed_or_complete")
    required = {
        "manifest_key",
        "package_relative_path",
        "package_sha256",
        "dicom_sha256",
        "pixel_sha256",
    }
    if any(not required.issubset(record) for record in selected):
        raise ValueError("pilot_audit_lacks_required_integrity_fields")
    mapping = {str(record["manifest_key"]): record for record in selected}
    if len(mapping) != expected:
        raise ValueError("pilot_audit_contains_duplicate_manifest_key")
    return mapping


def _extract_spacing(dataset: Any, accepted_tags: list[str]) -> tuple[float, float, str]:
    for tag in accepted_tags:
        raw = getattr(dataset, tag, None)
        if raw is None:
            continue
        try:
            values = [float(value) for value in raw]
        except (TypeError, ValueError):
            continue
        if len(values) != 2 or any(not 0.05 <= value <= 1.5 for value in values):
            continue
        return values[0], values[1], tag
    raise ValueError("valid_pixel_spacing_not_available")


def _weighted_x_center(normalized: np.ndarray, parameters: dict[str, Any]) -> float:
    rows, columns = normalized.shape
    x_low, x_high = (float(value) for value in parameters["anatomy_x_band"])
    y_low = max(0, int(round(rows * 0.18)))
    y_high = min(rows, int(round(rows * 0.84)))
    first = max(1, int(round(columns * x_low)))
    last = min(columns - 1, int(round(columns * x_high)))
    analysis = normalized[y_low:y_high, first:last]
    if analysis.size == 0 or last - first < 16:
        raise ValueError("empty_anatomical_x_band")
    upper = np.percentile(analysis, 80.0, axis=0)
    spread = upper - np.percentile(analysis, 25.0, axis=0)
    smooth_width = max(3, round(columns * float(parameters["profile_smoothing_fraction"])))
    response = 0.65 * _robust_unit(upper) + 0.35 * _robust_unit(spread)
    response = _smooth_profile(response, smooth_width)
    threshold = float(np.percentile(response, 55.0))
    weights = np.clip(response - threshold, 0.0, None)
    x_values = np.arange(first, last, dtype=np.float64)
    midpoint = (columns - 1) / 2.0
    center_scale = max(1.0, columns * 0.30)
    weights *= np.exp(-0.5 * ((x_values - midpoint) / center_scale) ** 2)
    if float(weights.sum()) <= 1e-9:
        return midpoint
    return float(np.sum(x_values * weights) / np.sum(weights))


def _fit_box(center: float, size: int, limit: int) -> tuple[int, int, int]:
    if size <= 0 or size > limit:
        raise ValueError("physical_crop_does_not_fit_unilateral_field")
    proposed = int(round(center - size / 2.0))
    start = min(max(proposed, 0), limit - size)
    return start, start + size, abs(start - proposed)


def _fit_box_with_inner_guard(
    center: float,
    size: int,
    limit: int,
    inner_edge: str | None,
    margin: int,
) -> tuple[int, int, int, int]:
    """Fit a crop while keeping deterministic clearance from the central ruler."""
    if inner_edge not in {None, "LEFT", "RIGHT"}:
        raise ValueError("invalid_inner_edge")
    if margin < 0:
        raise ValueError("invalid_inner_edge_margin")
    available_start = margin if inner_edge == "LEFT" else 0
    available_end = limit - margin if inner_edge == "RIGHT" else limit
    available = available_end - available_start
    if size <= 0 or size > available:
        raise ValueError("physical_crop_does_not_fit_after_inner_edge_guard")
    proposed = int(round(center - size / 2.0))
    start = min(max(proposed, available_start), available_end - size)
    clearance = start if inner_edge == "LEFT" else limit - (start + size)
    if inner_edge is None:
        clearance = min(start, limit - (start + size))
    return start, start + size, abs(start - proposed), clearance


def _profile_signals(
    profile: np.ndarray, rows: int, parameters: dict[str, Any]
) -> dict[str, np.ndarray]:
    smooth_width = max(
        3, round(rows * float(parameters["profile_smoothing_fraction"]))
    )
    intensity = _smooth_profile(profile, smooth_width)
    raw_signed_gradient = np.diff(intensity, prepend=intensity[:1])
    signed_gradient = _smooth_profile(raw_signed_gradient, smooth_width)
    # Preserve the exact v0.2 absolute-gradient signal for legacy dispatch.
    vertical_gradient = _smooth_profile(np.abs(raw_signed_gradient), smooth_width)
    darkness = 1.0 - _robust_unit(intensity)
    gradient = _robust_unit(vertical_gradient)
    offset = max(2, int(round(rows * float(parameters["bone_offset_fraction"]))))
    before = np.take(intensity, np.clip(np.arange(rows) - offset, 0, rows - 1))
    after = np.take(intensity, np.clip(np.arange(rows) + offset, 0, rows - 1))
    bone_contrast_raw = np.clip((before + after) / 2.0 - intensity, 0.0, None)
    return {
        "intensity": intensity,
        "darkness": darkness,
        "gradient": gradient,
        "signed_gradient": signed_gradient,
        "descending_edge": _robust_unit(np.clip(-signed_gradient, 0.0, None)),
        "ascending_edge": _robust_unit(np.clip(signed_gradient, 0.0, None)),
        "bone_contrast": _robust_unit(bone_contrast_raw),
    }


def _compartment_bounds(
    x_center: float, columns: int, parameters: dict[str, Any]
) -> tuple[tuple[int, int], tuple[int, int]]:
    inner = int(
        round(columns * float(parameters["compartment_inner_offset_fraction"]))
    )
    outer = int(
        round(columns * float(parameters["compartment_outer_offset_fraction"]))
    )
    minimum = int(parameters["minimum_compartment_width_pixels"])
    if not 0 <= inner < outer:
        raise ValueError("invalid_compartment_offsets")
    left = (
        max(0, int(round(x_center)) - outer),
        max(0, int(round(x_center)) - inner),
    )
    right = (
        min(columns, int(round(x_center)) + inner),
        min(columns, int(round(x_center)) + outer),
    )
    if left[1] - left[0] < minimum or right[1] - right[0] < minimum:
        raise ValueError("compartment_band_too_narrow")
    return left, right


def _compartment_score(
    normalized: np.ndarray,
    bounds: tuple[int, int],
    candidates: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x0, x1 = bounds
    profile = np.median(normalized[:, x0:x1], axis=1)
    signals = _profile_signals(profile, normalized.shape[0], parameters)
    score = (
        float(parameters["darkness_weight"]) * signals["darkness"][candidates]
        + float(parameters["bone_contrast_weight"])
        * signals["bone_contrast"][candidates]
        + float(parameters["gradient_weight"]) * signals["gradient"][candidates]
    )
    return score, signals


def _locate_tibiofemoral_joint_v02(
    normalized_half: np.ndarray,
    row_spacing_mm: float,
    column_spacing_mm: float,
    parameters: dict[str, Any],
    inner_edge: str | None = None,
) -> dict[str, Any]:
    """Locate a joint-space candidate and derive a fixed physical field of view."""
    rows, columns = normalized_half.shape
    search_low, search_high = (
        float(value) for value in parameters["joint_line_search_band"]
    )
    if not 0.15 <= search_low < search_high <= 0.85:
        raise ValueError("invalid_joint_line_search_band")

    x_center = _weighted_x_center(normalized_half, parameters)
    offset = max(2, int(round(rows * float(parameters["bone_offset_fraction"]))))
    first = max(offset, int(round(rows * search_low)))
    last = min(rows - offset, int(round(rows * search_high)))
    candidates = np.arange(first, last, dtype=int)
    if candidates.size < 8:
        raise ValueError("joint_search_band_too_small")
    expected_fraction = float(parameters["expected_joint_line_fraction"])
    expected_row = expected_fraction * (rows - 1)
    half_band = max(1.0, (last - first) / 2.0)
    center_distance = np.abs(candidates - expected_row) / half_band
    left_bounds, right_bounds = _compartment_bounds(x_center, columns, parameters)
    left_score, left_signals = _compartment_score(
        normalized_half, left_bounds, candidates, parameters
    )
    right_score, right_signals = _compartment_score(
        normalized_half, right_bounds, candidates, parameters
    )
    left_peak = int(candidates[int(np.argmax(left_score))])
    right_peak = int(candidates[int(np.argmax(right_score))])
    peak_distance_fraction = abs(left_peak - right_peak) / rows
    maximum_peak_distance = float(
        parameters["maximum_compartment_peak_distance_fraction"]
    )
    agreement_score = float(
        np.clip(
            1.0
            - peak_distance_fraction / max(maximum_peak_distance, 1e-6),
            0.0,
            1.0,
        )
    )
    agreement_sigma = max(
        1.0,
        rows * float(parameters["compartment_consensus_sigma_fraction"]),
    )
    consensus = 0.5 * (
        np.exp(-0.5 * ((candidates - left_peak) / agreement_sigma) ** 2)
        + np.exp(-0.5 * ((candidates - right_peak) / agreement_sigma) ** 2)
    )
    score = (
        0.5 * (_robust_unit(left_score) + _robust_unit(right_score))
        + float(parameters["compartment_consensus_weight"]) * consensus
        - float(parameters["vertical_center_penalty"]) * center_distance
    )
    best_index = int(np.argmax(score))
    y_center = int(candidates[best_index])
    score_prominence = float(np.clip(score[best_index] - np.median(score), 0.0, 1.0))
    contrast_at_center = float(
        0.5
        * (
            left_signals["bone_contrast"][y_center]
            + right_signals["bone_contrast"][y_center]
        )
    )

    crop_height_mm = float(parameters["crop_height_mm"])
    crop_width_mm = float(parameters["crop_width_mm"])
    crop_rows = int(round(crop_height_mm / row_spacing_mm))
    crop_columns = int(round(crop_width_mm / column_spacing_mm))
    crop_y0, crop_y1, shift_y = _fit_box(y_center, crop_rows, rows)
    margin_pixels = int(
        round(float(parameters["inner_edge_margin_mm"]) / column_spacing_mm)
    )
    crop_x0, crop_x1, shift_x, inner_clearance_pixels = _fit_box_with_inner_guard(
        x_center, crop_columns, columns, inner_edge, margin_pixels
    )
    boundary_shift_fraction = max(shift_y / rows, shift_x / columns)
    crop = normalized_half[crop_y0:crop_y1, crop_x0:crop_x1]
    background_fraction = float(np.mean(crop <= 0.02))
    saturation_fraction = float(np.mean(crop >= 0.98))

    prominence_component = min(
        1.0,
        score_prominence
        / max(float(parameters["minimum_score_prominence"]), 1e-6),
    )
    vertical_component = max(
        0.0,
        1.0
        - abs(y_center / rows - expected_fraction)
        / max(search_high - search_low, 1e-6),
    )
    horizontal_component = max(0.0, 1.0 - abs(x_center / columns - 0.5) / 0.38)
    artifact_component = max(
        0.0,
        1.0
        - background_fraction
        / max(float(parameters["maximum_background_fraction"]), 1e-6),
    )
    confidence_score = float(
        0.32 * prominence_component
        + 0.20 * contrast_at_center
        + 0.20 * agreement_score
        + 0.10 * vertical_component
        + 0.10 * horizontal_component
        + 0.08 * artifact_component
    )
    high_confidence = (
        score_prominence >= float(parameters["minimum_score_prominence"])
        and boundary_shift_fraction
        <= float(parameters["maximum_boundary_shift_fraction"])
        and background_fraction <= float(parameters["maximum_background_fraction"])
        and peak_distance_fraction <= maximum_peak_distance
        and confidence_score >= float(parameters["high_confidence_threshold"])
    )
    return {
        "joint_center_x_half": int(round(x_center)),
        "joint_center_y": y_center,
        "joint_center_x_fraction": x_center / columns,
        "joint_center_y_fraction": y_center / rows,
        "crop_x0_half": crop_x0,
        "crop_x1_half": crop_x1,
        "crop_y0": crop_y0,
        "crop_y1": crop_y1,
        "crop_rows": crop_rows,
        "crop_columns": crop_columns,
        "crop_height_mm": crop_rows * row_spacing_mm,
        "crop_width_mm": crop_columns * column_spacing_mm,
        "score_prominence": score_prominence,
        "bone_contrast_score": contrast_at_center,
        "left_compartment_peak_y": left_peak,
        "right_compartment_peak_y": right_peak,
        "compartment_peak_distance_fraction": peak_distance_fraction,
        "compartment_agreement_score": agreement_score,
        "inner_edge_clearance_mm": inner_clearance_pixels * column_spacing_mm,
        "boundary_shift_fraction": boundary_shift_fraction,
        "background_fraction": background_fraction,
        "saturation_fraction": saturation_fraction,
        "confidence_score": confidence_score,
        "confidence_level": "HIGH" if high_confidence else "LOW",
        "technical_status": "CANDIDATE_OK" if high_confidence else "REVIEW_REQUIRED",
    }


def _directed_edge_pairs(
    normalized: np.ndarray,
    bounds: tuple[int, int],
    first: int,
    last: int,
    row_spacing_mm: float,
    parameters: dict[str, Any],
) -> list[dict[str, float | int]]:
    """Rank dark gaps bounded by a descending and then ascending edge."""
    x0, x1 = bounds
    profile = np.median(normalized[:, x0:x1], axis=1)
    signals = _profile_signals(profile, normalized.shape[0], parameters)
    rows = normalized.shape[0]
    minimum_gap = max(
        1, int(round(float(parameters["joint_gap_min_mm"]) / row_spacing_mm))
    )
    maximum_gap = max(
        minimum_gap,
        int(round(float(parameters["joint_gap_max_mm"]) / row_spacing_mm)),
    )
    context = max(
        1, int(round(float(parameters["bone_context_mm"]) / row_spacing_mm))
    )
    expected_row = float(parameters["expected_joint_line_fraction"]) * (rows - 1)
    prior_scale = max(1.0, (last - first) / 2.0)
    brightness = _robust_unit(signals["intensity"])
    candidates: list[dict[str, float | int]] = []
    for femoral_edge in range(first, last):
        tibial_first = femoral_edge + minimum_gap
        tibial_last = min(last - 1, femoral_edge + maximum_gap)
        if tibial_first > tibial_last:
            continue
        for tibial_edge in range(tibial_first, tibial_last + 1):
            center = 0.5 * (femoral_edge + tibial_edge)
            edge_strength = float(
                min(
                    signals["descending_edge"][femoral_edge],
                    signals["ascending_edge"][tibial_edge],
                )
            )
            gap_darkness = float(
                np.mean(signals["darkness"][femoral_edge : tibial_edge + 1])
            )
            upper = brightness[max(0, femoral_edge - context) : femoral_edge]
            lower = brightness[
                tibial_edge + 1 : min(rows, tibial_edge + 1 + context)
            ]
            outside_bone = 0.5 * (
                (float(np.mean(upper)) if upper.size else 0.0)
                + (float(np.mean(lower)) if lower.size else 0.0)
            )
            center_prior = float(
                np.exp(-0.5 * ((center - expected_row) / prior_scale) ** 2)
            )
            score = float(
                float(parameters["edge_pair_weight"]) * edge_strength
                + float(parameters["gap_darkness_pair_weight"]) * gap_darkness
                + float(parameters["outside_bone_pair_weight"]) * outside_bone
                + float(parameters["pair_center_prior_weight"]) * center_prior
            )
            candidates.append(
                {
                    "femoral_edge_y": femoral_edge,
                    "tibial_edge_y": tibial_edge,
                    "center_y": center,
                    "width_mm": (tibial_edge - femoral_edge) * row_spacing_mm,
                    "score": score,
                    "minimum_edge_strength": edge_strength,
                    "gap_darkness": gap_darkness,
                    "outside_bone": outside_bone,
                }
            )
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[: int(parameters["top_edge_pairs_per_compartment"])]


def _bright_artifact_components(
    normalized: np.ndarray,
    y0: int,
    y1: int,
    inner_edge: str | None,
    parameters: dict[str, Any],
) -> tuple[int, int]:
    """Return compact bright components and their deepest inner-edge extent."""
    if inner_edge is None:
        return 0, 0
    rows, columns = normalized.shape
    scan_width = max(
        1, int(round(columns * float(parameters["artifact_scan_fraction"])))
    )
    if inner_edge == "LEFT":
        scan_x0, scan_x1 = 0, min(columns, scan_width)
    else:
        scan_x0, scan_x1 = max(0, columns - scan_width), columns
    region = normalized[max(0, y0) : min(rows, y1), scan_x0:scan_x1]
    if region.size == 0:
        return 0, 0
    mask = region >= float(parameters["artifact_bright_threshold"])
    visited = np.zeros(mask.shape, dtype=bool)
    minimum_area = int(parameters["artifact_component_min_area_pixels"])
    maximum_area = max(
        minimum_area,
        int(
            round(
                mask.size
                * float(parameters["artifact_component_max_area_fraction"])
            )
        ),
    )
    maximum_height = max(
        1,
        int(
            round(
                mask.shape[0]
                * float(parameters["artifact_component_max_height_fraction"])
            )
        ),
    )
    maximum_width = max(
        1,
        int(
            round(
                mask.shape[1]
                * float(parameters["artifact_component_max_width_fraction"])
            )
        ),
    )
    minimum_fill = float(parameters["artifact_component_min_fill_ratio"])
    count = 0
    deepest = 0
    height, width = mask.shape
    for start_y, start_x in zip(*np.nonzero(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        area = 0
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        while stack:
            current_y, current_x = stack.pop()
            area += 1
            min_x, max_x = min(min_x, current_x), max(max_x, current_x)
            min_y, max_y = min(min_y, current_y), max(max_y, current_y)
            for delta_y, delta_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor_y = current_y + delta_y
                neighbor_x = current_x + delta_x
                if not (0 <= neighbor_y < height and 0 <= neighbor_x < width):
                    continue
                if visited[neighbor_y, neighbor_x] or not mask[neighbor_y, neighbor_x]:
                    continue
                visited[neighbor_y, neighbor_x] = True
                stack.append((neighbor_y, neighbor_x))
        component_height = max_y - min_y + 1
        component_width = max_x - min_x + 1
        fill = area / (component_height * component_width)
        if not (
            minimum_area <= area <= maximum_area
            and component_height <= maximum_height
            and component_width <= maximum_width
            and fill >= minimum_fill
        ):
            continue
        count += 1
        extent = max_x + 1 if inner_edge == "LEFT" else width - min_x
        deepest = max(deepest, extent)
    return count, deepest


def _locate_tibiofemoral_joint_v03(
    normalized_half: np.ndarray,
    row_spacing_mm: float,
    column_spacing_mm: float,
    parameters: dict[str, Any],
    inner_edge: str | None = None,
) -> dict[str, Any]:
    """Locate the joint with directed edge pairs and mandatory confidence gates."""
    rows, columns = normalized_half.shape
    search_low, search_high = (
        float(value) for value in parameters["joint_line_search_band"]
    )
    if not 0.15 <= search_low < search_high <= 0.85:
        raise ValueError("invalid_joint_line_search_band")
    x_center = _weighted_x_center(normalized_half, parameters)
    first = max(1, int(round(rows * search_low)))
    last = min(rows - 1, int(round(rows * search_high)))
    if last - first < 8:
        raise ValueError("joint_search_band_too_small")
    left_bounds, right_bounds = _compartment_bounds(x_center, columns, parameters)
    left_pairs = _directed_edge_pairs(
        normalized_half, left_bounds, first, last, row_spacing_mm, parameters
    )
    right_pairs = _directed_edge_pairs(
        normalized_half, right_bounds, first, last, row_spacing_mm, parameters
    )
    if not left_pairs or not right_pairs:
        raise ValueError("directed_edge_pair_not_available")

    maximum_center_distance = float(
        parameters["maximum_compartment_peak_distance_fraction"]
    )
    maximum_width_difference = float(
        parameters["maximum_compartment_width_difference_mm"]
    )
    combinations: list[
        tuple[float, dict[str, float | int], dict[str, float | int]]
    ] = []
    for left_pair in left_pairs:
        for right_pair in right_pairs:
            center_distance = abs(
                float(left_pair["center_y"]) - float(right_pair["center_y"])
            ) / rows
            width_difference = abs(
                float(left_pair["width_mm"]) - float(right_pair["width_mm"])
            )
            if (
                center_distance > maximum_center_distance
                or width_difference > maximum_width_difference
            ):
                continue
            center_agreement = max(
                0.0,
                1.0 - center_distance / max(maximum_center_distance, 1e-6),
            )
            width_agreement = max(
                0.0,
                1.0
                - width_difference / max(maximum_width_difference, 1e-6),
            )
            denominator = (
                1.0
                + float(parameters["compartment_consensus_weight"])
                + float(parameters["compartment_width_consensus_weight"])
            )
            combined_score = (
                0.5 * (float(left_pair["score"]) + float(right_pair["score"]))
                + float(parameters["compartment_consensus_weight"])
                * center_agreement
                + float(parameters["compartment_width_consensus_weight"])
                * width_agreement
            ) / denominator
            combinations.append((combined_score, left_pair, right_pair))
    compartment_consensus_gate = bool(combinations)
    if combinations:
        _, left_pair, right_pair = max(combinations, key=lambda item: item[0])
    else:
        left_pair, right_pair = left_pairs[0], right_pairs[0]

    left_center = float(left_pair["center_y"])
    right_center = float(right_pair["center_y"])
    y_center = int(round(0.5 * (left_center + right_center)))
    center_distance_fraction = abs(left_center - right_center) / rows
    agreement_score = float(
        np.clip(
            1.0
            - center_distance_fraction / max(maximum_center_distance, 1e-6),
            0.0,
            1.0,
        )
    )
    width_difference_mm = abs(
        float(left_pair["width_mm"]) - float(right_pair["width_mm"])
    )
    edge_strength = min(
        float(left_pair["minimum_edge_strength"]),
        float(right_pair["minimum_edge_strength"]),
    )
    vertical_edge_gate = (
        edge_strength >= float(parameters["minimum_directed_edge_strength"])
        and float(left_pair["score"]) >= float(parameters["minimum_pair_score"])
        and float(right_pair["score"]) >= float(parameters["minimum_pair_score"])
    )

    crop_height_mm = float(parameters["crop_height_mm"])
    crop_width_mm = float(parameters["crop_width_mm"])
    crop_rows = int(round(crop_height_mm / row_spacing_mm))
    crop_columns = int(round(crop_width_mm / column_spacing_mm))
    crop_y0, crop_y1, shift_y = _fit_box(y_center, crop_rows, rows)
    artifact_count, artifact_extent_pixels = _bright_artifact_components(
        normalized_half, crop_y0, crop_y1, inner_edge, parameters
    )
    static_margin_pixels = int(
        round(float(parameters["inner_edge_margin_mm"]) / column_spacing_mm)
    )
    safety_pixels = int(
        round(float(parameters["artifact_safety_buffer_mm"]) / column_spacing_mm)
    )
    required_margin_pixels = max(
        static_margin_pixels,
        artifact_extent_pixels + safety_pixels if artifact_count else 0,
    )
    maximum_feasible_margin = columns - crop_columns
    if maximum_feasible_margin < 0:
        raise ValueError("physical_crop_does_not_fit_unilateral_field")
    applied_margin_pixels = min(required_margin_pixels, maximum_feasible_margin)
    crop_x0, crop_x1, shift_x, inner_clearance_pixels = _fit_box_with_inner_guard(
        x_center,
        crop_columns,
        columns,
        inner_edge,
        applied_margin_pixels,
    )
    artifact_clearance_gate = required_margin_pixels <= maximum_feasible_margin
    boundary_shift_fraction = max(shift_y / rows, shift_x / columns)
    crop = normalized_half[crop_y0:crop_y1, crop_x0:crop_x1]
    background_fraction = float(np.mean(crop <= 0.02))
    saturation_fraction = float(np.mean(crop >= 0.98))
    boundary_gate = boundary_shift_fraction <= float(
        parameters["maximum_boundary_shift_fraction"]
    )
    background_gate = background_fraction <= float(
        parameters["maximum_background_fraction"]
    )
    saturation_gate = saturation_fraction <= float(
        parameters["maximum_saturation_fraction"]
    )
    mandatory_gates = (
        vertical_edge_gate
        and compartment_consensus_gate
        and boundary_gate
        and background_gate
        and saturation_gate
        and artifact_clearance_gate
    )

    average_pair_score = 0.5 * (
        float(left_pair["score"]) + float(right_pair["score"])
    )
    score_prominence = float(np.clip(average_pair_score, 0.0, 1.0))
    contrast_at_center = 0.5 * (
        float(left_pair["outside_bone"]) + float(right_pair["outside_bone"])
    )
    width_agreement = max(
        0.0,
        1.0 - width_difference_mm / max(maximum_width_difference, 1e-6),
    )
    confidence_score = float(
        0.34 * average_pair_score
        + 0.22 * edge_strength
        + 0.18 * agreement_score
        + 0.10 * width_agreement
        + 0.08 * max(0.0, 1.0 - background_fraction)
        + 0.08 * max(0.0, 1.0 - saturation_fraction)
    )
    high_confidence = mandatory_gates and confidence_score >= float(
        parameters["high_confidence_threshold"]
    )
    reasons: list[str] = []
    if not vertical_edge_gate:
        reasons.append("EDGE_PAIR")
    if not compartment_consensus_gate:
        reasons.append("VERTICAL_DISAGREEMENT")
    if not artifact_clearance_gate:
        reasons.append("ARTIFACT_CLEARANCE")
    if not boundary_gate:
        reasons.append("BOUNDARY_SHIFT")
    if not background_gate:
        reasons.append("BACKGROUND")
    if not saturation_gate:
        reasons.append("SATURATION")
    if mandatory_gates and not high_confidence:
        reasons.append("LOW_COMPOSITE_CONFIDENCE")
    if not vertical_edge_gate:
        technical_status = "REVIEW_REQUIRED_EDGE_PAIR"
    elif not compartment_consensus_gate:
        technical_status = "REVIEW_REQUIRED_VERTICAL_DISAGREEMENT"
    elif not artifact_clearance_gate:
        technical_status = "REVIEW_REQUIRED_ARTIFACT_CLEARANCE"
    elif high_confidence:
        technical_status = "CANDIDATE_OK"
    else:
        technical_status = "REVIEW_REQUIRED"

    return {
        "joint_center_x_half": int(round(x_center)),
        "joint_center_y": y_center,
        "joint_center_x_fraction": x_center / columns,
        "joint_center_y_fraction": y_center / rows,
        "crop_x0_half": crop_x0,
        "crop_x1_half": crop_x1,
        "crop_y0": crop_y0,
        "crop_y1": crop_y1,
        "crop_rows": crop_rows,
        "crop_columns": crop_columns,
        "crop_height_mm": crop_rows * row_spacing_mm,
        "crop_width_mm": crop_columns * column_spacing_mm,
        "score_prominence": score_prominence,
        "bone_contrast_score": contrast_at_center,
        "left_compartment_peak_y": int(round(left_center)),
        "right_compartment_peak_y": int(round(right_center)),
        "compartment_peak_distance_fraction": center_distance_fraction,
        "compartment_agreement_score": agreement_score,
        "inner_edge_clearance_mm": inner_clearance_pixels * column_spacing_mm,
        "boundary_shift_fraction": boundary_shift_fraction,
        "background_fraction": background_fraction,
        "saturation_fraction": saturation_fraction,
        "localization_strategy": "directed_edge_pairs_v0.3",
        "left_femoral_edge_y": int(left_pair["femoral_edge_y"]),
        "left_tibial_edge_y": int(left_pair["tibial_edge_y"]),
        "right_femoral_edge_y": int(right_pair["femoral_edge_y"]),
        "right_tibial_edge_y": int(right_pair["tibial_edge_y"]),
        "left_joint_width_mm": float(left_pair["width_mm"]),
        "right_joint_width_mm": float(right_pair["width_mm"]),
        "left_pair_score": float(left_pair["score"]),
        "right_pair_score": float(right_pair["score"]),
        "directed_edge_strength_score": edge_strength,
        "compartment_width_difference_mm": width_difference_mm,
        "vertical_edge_gate_passed": vertical_edge_gate,
        "compartment_consensus_gate_passed": compartment_consensus_gate,
        "detected_artifact_components": artifact_count,
        "detected_artifact_extent_mm": artifact_extent_pixels * column_spacing_mm,
        "effective_inner_margin_mm": required_margin_pixels * column_spacing_mm,
        "artifact_clearance_gate_passed": artifact_clearance_gate,
        "boundary_gate_passed": boundary_gate,
        "background_gate_passed": background_gate,
        "saturation_gate_passed": saturation_gate,
        "mandatory_gates_passed": mandatory_gates,
        "review_reasons": "|".join(reasons),
        "confidence_score": confidence_score,
        "confidence_level": "HIGH" if high_confidence else "LOW",
        "technical_status": technical_status,
    }


def locate_tibiofemoral_joint(
    normalized_half: np.ndarray,
    row_spacing_mm: float,
    column_spacing_mm: float,
    parameters: dict[str, Any],
    inner_edge: str | None = None,
) -> dict[str, Any]:
    """Dispatch to the explicitly configured localization strategy."""
    if parameters.get("localization_strategy") == "directed_edge_pairs_v0.3":
        return _locate_tibiofemoral_joint_v03(
            normalized_half,
            row_spacing_mm,
            column_spacing_mm,
            parameters,
            inner_edge,
        )
    return _locate_tibiofemoral_joint_v02(
        normalized_half,
        row_spacing_mm,
        column_spacing_mm,
        parameters,
        inner_edge,
    )


def _make_preview(
    normalized_half: np.ndarray,
    localization: dict[str, Any],
    side: str,
    output_path: Path,
    max_width: int,
) -> None:
    base = Image.fromarray(np.round(normalized_half * 255.0).astype(np.uint8), mode="L")
    x0, x1 = int(localization["crop_x0_half"]), int(localization["crop_x1_half"])
    y0, y1 = int(localization["crop_y0"]), int(localization["crop_y1"])
    joint_x = int(localization["joint_center_x_half"])
    joint_y = int(localization["joint_center_y"])
    annotated = base.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, annotated.width // 300)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 255, 255), width=line_width)
    draw.line((x0, joint_y, x1 - 1, joint_y), fill=(255, 50, 50), width=line_width)
    radius = max(4, annotated.width // 120)
    draw.ellipse(
        (joint_x - radius, joint_y - radius, joint_x + radius, joint_y + radius),
        outline=(255, 255, 0),
        width=line_width,
    )
    crop = base.crop((x0, y0, x1, y1)).convert("RGB")
    target_height = annotated.height
    if crop.height != target_height:
        ratio = target_height / crop.height
        crop = crop.resize(
            (max(1, round(crop.width * ratio)), target_height), Image.Resampling.LANCZOS
        )
    canvas = Image.new("RGB", (annotated.width + crop.width, target_height), "black")
    canvas.paste(annotated, (0, 0))
    canvas.paste(crop, (annotated.width, 0))
    label_draw = ImageDraw.Draw(canvas)
    label_draw.rectangle((0, 0, min(canvas.width, 740), 26), fill=(0, 0, 0))
    label_draw.text(
        (6, 6),
        f"PATIENT {side} | CYAN=CROP | RED=JOINT LINE | RIGHT PANEL=CROP",
        fill=(255, 255, 0),
    )
    if canvas.width > max_width:
        ratio = max_width / canvas.width
        canvas = canvas.resize(
            (max_width, max(1, round(canvas.height * ratio))), Image.Resampling.LANCZOS
        )
    canvas.save(output_path, format="PNG", optimize=True)


def _review_has_human_decisions(path: Path) -> bool:
    if not path.is_file():
        return False
    human_fields = REVIEW_FIELDS[8:]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return any(
            any(str(row.get(field, "")).strip() for field in human_fields)
            for row in csv.DictReader(stream, delimiter=";")
        )


def _validate_parameters(parameters: dict[str, Any]) -> None:
    required = {
        "expected_unique_studies",
        "expected_knees",
        "expected_bilateral_algorithm_version",
        "algorithm_version",
        "joint_line_search_band",
        "anatomy_x_band",
        "compartment_inner_offset_fraction",
        "compartment_outer_offset_fraction",
        "minimum_compartment_width_pixels",
        "maximum_compartment_peak_distance_fraction",
        "compartment_consensus_sigma_fraction",
        "compartment_consensus_weight",
        "profile_smoothing_fraction",
        "bone_offset_fraction",
        "expected_joint_line_fraction",
        "darkness_weight",
        "bone_contrast_weight",
        "gradient_weight",
        "vertical_center_penalty",
        "crop_height_mm",
        "crop_width_mm",
        "inner_edge_margin_mm",
        "accepted_spacing_tags",
        "minimum_score_prominence",
        "maximum_boundary_shift_fraction",
        "maximum_background_fraction",
        "high_confidence_threshold",
        "max_preview_width",
    }
    missing = sorted(required - parameters.keys())
    if missing:
        raise ValueError("missing_parameters:" + ",".join(missing))
    weights = sum(
        float(parameters[key])
        for key in ("darkness_weight", "bone_contrast_weight", "gradient_weight")
    )
    if not np.isclose(weights, 1.0):
        raise ValueError("localization_weights_must_sum_to_one")
    if int(parameters["expected_knees"]) != 2 * int(parameters["expected_unique_studies"]):
        raise ValueError("expected_knees_must_equal_two_per_study")
    if float(parameters["crop_height_mm"]) < 100.0:
        raise ValueError("crop_height_mm_is_too_small")
    if float(parameters["crop_width_mm"]) < 100.0:
        raise ValueError("crop_width_mm_is_too_small")
    if float(parameters["inner_edge_margin_mm"]) < 0.0:
        raise ValueError("inner_edge_margin_mm_is_negative")
    if int(parameters["max_preview_width"]) < 640:
        raise ValueError("max_preview_width_is_too_small")
    strategy = parameters.get("localization_strategy")
    if strategy is None:
        return
    if strategy != "directed_edge_pairs_v0.3":
        raise ValueError("unsupported_localization_strategy")
    v03_required = {
        "joint_gap_min_mm",
        "joint_gap_max_mm",
        "bone_context_mm",
        "edge_pair_weight",
        "gap_darkness_pair_weight",
        "outside_bone_pair_weight",
        "pair_center_prior_weight",
        "top_edge_pairs_per_compartment",
        "minimum_directed_edge_strength",
        "minimum_pair_score",
        "maximum_compartment_width_difference_mm",
        "compartment_width_consensus_weight",
        "artifact_scan_fraction",
        "artifact_bright_threshold",
        "artifact_component_min_area_pixels",
        "artifact_component_max_area_fraction",
        "artifact_component_max_height_fraction",
        "artifact_component_max_width_fraction",
        "artifact_component_min_fill_ratio",
        "artifact_safety_buffer_mm",
        "maximum_saturation_fraction",
    }
    missing_v03 = sorted(v03_required - parameters.keys())
    if missing_v03:
        raise ValueError("missing_v03_parameters:" + ",".join(missing_v03))
    pair_weights = sum(
        float(parameters[key])
        for key in (
            "edge_pair_weight",
            "gap_darkness_pair_weight",
            "outside_bone_pair_weight",
            "pair_center_prior_weight",
        )
    )
    if not np.isclose(pair_weights, 1.0):
        raise ValueError("directed_pair_weights_must_sum_to_one")
    if not 0.0 < float(parameters["joint_gap_min_mm"]) < float(
        parameters["joint_gap_max_mm"]
    ):
        raise ValueError("invalid_joint_gap_range")
    if float(parameters["bone_context_mm"]) <= 0.0:
        raise ValueError("bone_context_mm_must_be_positive")
    if int(parameters["top_edge_pairs_per_compartment"]) < 1:
        raise ValueError("top_edge_pairs_per_compartment_must_be_positive")
    for key in (
        "minimum_directed_edge_strength",
        "minimum_pair_score",
        "artifact_scan_fraction",
        "artifact_bright_threshold",
        "artifact_component_max_area_fraction",
        "artifact_component_max_height_fraction",
        "artifact_component_max_width_fraction",
        "artifact_component_min_fill_ratio",
        "maximum_saturation_fraction",
    ):
        if not 0.0 < float(parameters[key]) <= 1.0:
            raise ValueError(f"{key}_must_be_in_zero_one")
    if int(parameters["artifact_component_min_area_pixels"]) < 1:
        raise ValueError("artifact_component_min_area_pixels_must_be_positive")
    if float(parameters["artifact_safety_buffer_mm"]) < 0.0:
        raise ValueError("artifact_safety_buffer_mm_is_negative")
    if float(parameters["maximum_compartment_width_difference_mm"]) <= 0.0:
        raise ValueError("maximum_compartment_width_difference_mm_must_be_positive")


def prepare_joint_localization_pilot(
    source_dir: Path,
    pilot_audit_json: Path,
    bilateral_output_dir: Path,
    output_dir: Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Create 20 pseudonymous crop candidates without reading outcomes."""
    _validate_parameters(parameters)
    expected_studies = int(parameters["expected_unique_studies"])
    bilateral = _load_closed_bilateral_pilot(
        bilateral_output_dir,
        expected_studies,
        str(parameters["expected_bilateral_algorithm_version"]),
    )
    audit = _load_verified_pilot(pilot_audit_json, expected_studies)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews_ciegas"
    crop_dir = output_dir / "recortes_nativos"
    preview_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "revision_visual_ciega.csv"
    if _review_has_human_decisions(review_path):
        raise RuntimeError("existing_human_review_refusing_to_overwrite")
    for path in preview_dir.glob("case_*_localizacion.png"):
        path.unlink()
    for path in crop_dir.glob("case_*_crop.npy"):
        path.unlink()

    result_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    failures = 0
    for split_record in bilateral:
        case_result_start = len(result_rows)
        case_alias = split_record["case_alias"]
        manifest_key = split_record["manifest_key"]
        expected_record = audit.get(manifest_key)
        if expected_record is None:
            raise ValueError("bilateral_result_not_present_in_closed_pilot_audit")
        try:
            package = _resolve_package(source_dir, expected_record["package_relative_path"])
            if sha256_file(package) != expected_record["package_sha256"]:
                raise ValueError("package_hash_mismatch")
            dicom_payload, dataset, pixels = _read_single_image(package)
            if sha256_bytes(dicom_payload) != expected_record["dicom_sha256"]:
                raise ValueError("dicom_hash_mismatch")
            if sha256_bytes(np.ascontiguousarray(pixels).tobytes()) != expected_record["pixel_sha256"]:
                raise ValueError("pixel_hash_mismatch")
            normalized = normalize_working_copy(
                pixels, str(getattr(dataset, "PhotometricInterpretation", ""))
            )
            split_column = int(split_record["split_column"])
            if not 0 < split_column < normalized.shape[1]:
                raise ValueError("invalid_frozen_split_column")
            row_spacing, column_spacing, spacing_source = _extract_spacing(
                dataset, list(parameters["accepted_spacing_tags"])
            )
            halves = (
                ("RIGHT", 0, split_column, normalized[:, :split_column], pixels[:, :split_column]),
                (
                    "LEFT",
                    split_column,
                    normalized.shape[1],
                    normalized[:, split_column:],
                    pixels[:, split_column:],
                ),
            )
            for side, half_x0, half_x1, normalized_half, native_half in halves:
                knee_alias = f"{case_alias}_{side}"
                localization = locate_tibiofemoral_joint(
                    normalized_half,
                    row_spacing,
                    column_spacing,
                    parameters,
                    inner_edge=side,
                )
                x0, x1 = int(localization["crop_x0_half"]), int(localization["crop_x1_half"])
                y0, y1 = int(localization["crop_y0"]), int(localization["crop_y1"])
                native_crop = np.ascontiguousarray(native_half[y0:y1, x0:x1])
                crop_name = f"{knee_alias}_crop.npy"
                crop_path = crop_dir / crop_name
                np.save(crop_path, native_crop, allow_pickle=False)
                preview_name = f"{knee_alias}_localizacion.png"
                _make_preview(
                    normalized_half,
                    localization,
                    side,
                    preview_dir / preview_name,
                    int(parameters["max_preview_width"]),
                )
                result: dict[str, Any] = {field: "" for field in PRIVATE_RESULT_FIELDS}
                result.update(
                    {
                        "case_alias": case_alias,
                        "knee_alias": knee_alias,
                        "patient_side": side,
                        "manifest_key": manifest_key,
                        "package_relative_path": expected_record["package_relative_path"],
                        "package_sha256": expected_record["package_sha256"],
                        "dicom_sha256": expected_record["dicom_sha256"],
                        "pixel_sha256": expected_record["pixel_sha256"],
                        "native_crop_sha256": sha256_bytes(native_crop.tobytes()),
                        "spacing_source": spacing_source,
                        "row_spacing_mm": row_spacing,
                        "column_spacing_mm": column_spacing,
                        "half_x0": half_x0,
                        "half_x1": half_x1,
                        "preview_file": f"previews_ciegas/{preview_name}",
                        "native_crop_file": f"recortes_nativos/{crop_name}",
                        **localization,
                    }
                )
                result_rows.append(result)
                review_rows.append(
                    {
                        "knee_alias": knee_alias,
                        "case_alias": case_alias,
                        "patient_side": side,
                        "preview_file": result["preview_file"],
                        "automatic_confidence_level": result["confidence_level"],
                        "automatic_joint_center_y_fraction": result[
                            "joint_center_y_fraction"
                        ],
                        "automatic_crop_height_mm": result["crop_height_mm"],
                        "automatic_crop_width_mm": result["crop_width_mm"],
                        "joint_centered_yes_no": "",
                        "tibiofemoral_anatomy_complete_yes_no": "",
                        "text_borders_and_rule_excluded_yes_no": "",
                        "crop_acceptable_yes_no": "",
                        "technical_exclusion_yes_no": "",
                        "exclusion_reason": "",
                        "reviewed_without_outcome_yes_no": "",
                        "reviewer_notes": "",
                    }
                )
        except Exception as exc:
            failures += 1
            existing_sides = {
                str(row["patient_side"])
                for row in result_rows[case_result_start:]
            }
            for side in ("RIGHT", "LEFT"):
                if side in existing_sides:
                    continue
                knee_alias = f"{case_alias}_{side}"
                result = {field: "" for field in PRIVATE_RESULT_FIELDS}
                result.update(
                    {
                        "case_alias": case_alias,
                        "knee_alias": knee_alias,
                        "patient_side": side,
                        "manifest_key": manifest_key,
                        "technical_status": "ERROR",
                        "confidence_level": "NONE",
                        "error_code": str(exc),
                    }
                )
                result_rows.append(result)
                review_rows.append(
                    {
                        field: (
                            knee_alias
                            if field == "knee_alias"
                            else case_alias
                            if field == "case_alias"
                            else side
                            if field == "patient_side"
                            else ""
                        )
                        for field in REVIEW_FIELDS
                    }
                )

    with (output_dir / "resultados_localizacion_privados.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=PRIVATE_RESULT_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(result_rows)
    with review_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(review_rows)

    successful = [row for row in result_rows if not row["error_code"]]
    confidence_counts = Counter(str(row["confidence_level"]) for row in successful)
    technical_status_counts = Counter(
        str(row["technical_status"]) for row in successful
    )
    spacing_counts = Counter(str(row["spacing_source"]) for row in successful)
    public_summary = {
        "status": (
            "ready_for_blinded_review"
            if len(successful) == int(parameters["expected_knees"]) and failures == 0
            else "technical_failure"
        ),
        "algorithm_version": parameters["algorithm_version"],
        "localization_strategy": parameters.get(
            "localization_strategy", "legacy_absolute_gradient_v0.2"
        ),
        "source_bilateral_algorithm_version": parameters[
            "expected_bilateral_algorithm_version"
        ],
        "expected_unique_studies": expected_studies,
        "processed_unique_studies": expected_studies - failures,
        "expected_knees": int(parameters["expected_knees"]),
        "processed_knees": len(successful),
        "technical_failures": failures,
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "technical_status_counts": dict(sorted(technical_status_counts.items())),
        "spacing_source_counts": dict(sorted(spacing_counts.items())),
        "crop_height_mm": float(parameters["crop_height_mm"]),
        "crop_width_mm": float(parameters["crop_width_mm"]),
        "inner_edge_margin_mm": float(parameters["inner_edge_margin_mm"]),
        "visual_review_status": "pending",
        "outcome_data_loaded": False,
        "mass_processing_executed": False,
        "training_executed": False,
        "partitions_created": False,
        "reserved_test_opened": False,
    }
    (output_dir / "resumen_localizacion_publico.json").write_text(
        json.dumps(public_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    candidate_parameters = {
        key: parameters[key]
        for key in (
            "algorithm_version",
            "expected_bilateral_algorithm_version",
            "joint_line_search_band",
            "anatomy_x_band",
            "compartment_inner_offset_fraction",
            "compartment_outer_offset_fraction",
            "minimum_compartment_width_pixels",
            "maximum_compartment_peak_distance_fraction",
            "compartment_consensus_sigma_fraction",
            "compartment_consensus_weight",
            "profile_smoothing_fraction",
            "bone_offset_fraction",
            "expected_joint_line_fraction",
            "darkness_weight",
            "bone_contrast_weight",
            "gradient_weight",
            "vertical_center_penalty",
            "crop_height_mm",
            "crop_width_mm",
            "inner_edge_margin_mm",
            "accepted_spacing_tags",
            "minimum_score_prominence",
            "maximum_boundary_shift_fraction",
            "maximum_background_fraction",
            "high_confidence_threshold",
        )
    }
    if parameters.get("localization_strategy") == "directed_edge_pairs_v0.3":
        for key in (
            "localization_strategy",
            "joint_gap_min_mm",
            "joint_gap_max_mm",
            "bone_context_mm",
            "edge_pair_weight",
            "gap_darkness_pair_weight",
            "outside_bone_pair_weight",
            "pair_center_prior_weight",
            "top_edge_pairs_per_compartment",
            "minimum_directed_edge_strength",
            "minimum_pair_score",
            "maximum_compartment_width_difference_mm",
            "compartment_width_consensus_weight",
            "artifact_scan_fraction",
            "artifact_bright_threshold",
            "artifact_component_min_area_pixels",
            "artifact_component_max_area_fraction",
            "artifact_component_max_height_fraction",
            "artifact_component_max_width_fraction",
            "artifact_component_min_fill_ratio",
            "artifact_safety_buffer_mm",
            "maximum_saturation_fraction",
        ):
            candidate_parameters[key] = parameters[key]
    candidate_parameters.update(
        {
            "status": "candidate_not_frozen_until_blinded_visual_review",
            "source_bilateral_closure_sha256": sha256_file(
                bilateral_output_dir / "cierre_paso_3_publico.json"
            ),
            "source_bilateral_parameters_sha256": sha256_file(
                bilateral_output_dir / "parametros_congelados.json"
            ),
        }
    )
    (output_dir / "parametros_candidatos.json").write_text(
        json.dumps(candidate_parameters, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return public_summary


def _resolve_under_root(root: Path, value: str, key: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{key} must be relative to KNEE_DATA_ROOT.")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{key} escapes KNEE_DATA_ROOT.")
    return resolved


def load_config(path: Path) -> dict[str, Any]:
    root_value = os.environ.get("KNEE_DATA_ROOT")
    if not root_value:
        raise ValueError("Set KNEE_DATA_ROOT to the authorized data directory.")
    root = Path(root_value).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("source_dir", "pilot_audit_json", "bilateral_output_dir", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing relative path: {key}.")
        config[key] = _resolve_under_root(root, config[key], key)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = prepare_joint_localization_pilot(
        config["source_dir"],
        config["pilot_audit_json"],
        config["bilateral_output_dir"],
        config["output_dir"],
        config,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "ready_for_blinded_review":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
