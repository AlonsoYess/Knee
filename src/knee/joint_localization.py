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
    "boundary_shift_fraction",
    "background_fraction",
    "saturation_fraction",
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


def locate_tibiofemoral_joint(
    normalized_half: np.ndarray,
    row_spacing_mm: float,
    column_spacing_mm: float,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Locate a joint-space candidate and derive a fixed physical field of view."""
    rows, columns = normalized_half.shape
    search_low, search_high = (
        float(value) for value in parameters["joint_line_search_band"]
    )
    if not 0.15 <= search_low < search_high <= 0.85:
        raise ValueError("invalid_joint_line_search_band")
    if not 0.0 < float(parameters["analysis_half_width_fraction"]) < 0.5:
        raise ValueError("invalid_analysis_half_width_fraction")

    x_center = _weighted_x_center(normalized_half, parameters)
    half_width = max(8, int(round(columns * float(parameters["analysis_half_width_fraction"]))))
    x0 = max(0, int(round(x_center)) - half_width)
    x1 = min(columns, int(round(x_center)) + half_width + 1)
    analysis = normalized_half[:, x0:x1]
    if analysis.shape[1] < 16:
        raise ValueError("joint_profile_band_too_narrow")

    intensity = np.median(analysis, axis=1)
    vertical_gradient = np.median(
        np.abs(np.diff(analysis, axis=0, prepend=analysis[:1])), axis=1
    )
    smooth_width = max(3, round(rows * float(parameters["profile_smoothing_fraction"])))
    intensity = _smooth_profile(intensity, smooth_width)
    vertical_gradient = _smooth_profile(vertical_gradient, smooth_width)
    darkness = 1.0 - _robust_unit(intensity)
    gradient = _robust_unit(vertical_gradient)

    offset = max(2, int(round(rows * float(parameters["bone_offset_fraction"]))))
    before = np.take(intensity, np.clip(np.arange(rows) - offset, 0, rows - 1))
    after = np.take(intensity, np.clip(np.arange(rows) + offset, 0, rows - 1))
    bone_contrast_raw = np.clip((before + after) / 2.0 - intensity, 0.0, None)
    bone_contrast = _robust_unit(bone_contrast_raw)

    first = max(offset, int(round(rows * search_low)))
    last = min(rows - offset, int(round(rows * search_high)))
    candidates = np.arange(first, last, dtype=int)
    if candidates.size < 8:
        raise ValueError("joint_search_band_too_small")
    expected_fraction = float(parameters["expected_joint_line_fraction"])
    expected_row = expected_fraction * (rows - 1)
    half_band = max(1.0, (last - first) / 2.0)
    center_distance = np.abs(candidates - expected_row) / half_band
    score = (
        float(parameters["darkness_weight"]) * darkness[candidates]
        + float(parameters["bone_contrast_weight"]) * bone_contrast[candidates]
        + float(parameters["gradient_weight"]) * gradient[candidates]
        - float(parameters["vertical_center_penalty"]) * center_distance
    )
    best_index = int(np.argmax(score))
    y_center = int(candidates[best_index])
    score_prominence = float(np.clip(score[best_index] - np.median(score), 0.0, 1.0))
    contrast_at_center = float(bone_contrast[y_center])

    crop_size_mm = float(parameters["crop_size_mm"])
    crop_rows = int(round(crop_size_mm / row_spacing_mm))
    crop_columns = int(round(crop_size_mm / column_spacing_mm))
    crop_y0, crop_y1, shift_y = _fit_box(y_center, crop_rows, rows)
    crop_x0, crop_x1, shift_x = _fit_box(x_center, crop_columns, columns)
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
        0.40 * prominence_component
        + 0.25 * contrast_at_center
        + 0.15 * vertical_component
        + 0.10 * horizontal_component
        + 0.10 * artifact_component
    )
    high_confidence = (
        score_prominence >= float(parameters["minimum_score_prominence"])
        and boundary_shift_fraction
        <= float(parameters["maximum_boundary_shift_fraction"])
        and background_fraction <= float(parameters["maximum_background_fraction"])
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
        "boundary_shift_fraction": boundary_shift_fraction,
        "background_fraction": background_fraction,
        "saturation_fraction": saturation_fraction,
        "confidence_score": confidence_score,
        "confidence_level": "HIGH" if high_confidence else "LOW",
        "technical_status": "CANDIDATE_OK" if high_confidence else "REVIEW_REQUIRED",
    }


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
        "analysis_half_width_fraction",
        "profile_smoothing_fraction",
        "bone_offset_fraction",
        "expected_joint_line_fraction",
        "darkness_weight",
        "bone_contrast_weight",
        "gradient_weight",
        "vertical_center_penalty",
        "crop_size_mm",
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
    if float(parameters["crop_size_mm"]) < 100.0:
        raise ValueError("crop_size_mm_is_too_small")
    if int(parameters["max_preview_width"]) < 640:
        raise ValueError("max_preview_width_is_too_small")


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
                    normalized_half, row_spacing, column_spacing, parameters
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
    spacing_counts = Counter(str(row["spacing_source"]) for row in successful)
    public_summary = {
        "status": (
            "ready_for_blinded_review"
            if len(successful) == int(parameters["expected_knees"]) and failures == 0
            else "technical_failure"
        ),
        "algorithm_version": parameters["algorithm_version"],
        "source_bilateral_algorithm_version": parameters[
            "expected_bilateral_algorithm_version"
        ],
        "expected_unique_studies": expected_studies,
        "processed_unique_studies": expected_studies - failures,
        "expected_knees": int(parameters["expected_knees"]),
        "processed_knees": len(successful),
        "technical_failures": failures,
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "spacing_source_counts": dict(sorted(spacing_counts.items())),
        "crop_size_mm": float(parameters["crop_size_mm"]),
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
            "analysis_half_width_fraction",
            "profile_smoothing_fraction",
            "bone_offset_fraction",
            "expected_joint_line_fraction",
            "darkness_weight",
            "bone_contrast_weight",
            "gradient_weight",
            "vertical_center_penalty",
            "crop_size_mm",
            "accepted_spacing_tags",
            "minimum_score_prominence",
            "maximum_boundary_shift_fraction",
            "maximum_background_fraction",
            "high_confidence_threshold",
        )
    }
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
