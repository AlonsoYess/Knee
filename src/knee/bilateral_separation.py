"""Prepare a blinded pilot for bilateral radiograph separation and laterality review."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image, ImageDraw

from knee.dicom_audit import sha256_bytes, sha256_file


PRIVATE_RESULT_FIELDS = (
    "case_alias",
    "manifest_key",
    "package_relative_path",
    "package_sha256",
    "dicom_sha256",
    "pixel_sha256",
    "rows",
    "columns",
    "photometric_interpretation",
    "dicom_laterality",
    "split_column",
    "split_fraction",
    "midpoint_column",
    "midpoint_offset_fraction",
    "image_left_width",
    "image_right_width",
    "half_width_ratio",
    "valley_prominence",
    "confidence_score",
    "confidence_level",
    "technical_status",
    "proposed_image_left_side",
    "proposed_image_right_side",
    "preview_file",
    "error_code",
)

REVIEW_FIELDS = (
    "case_alias",
    "preview_file",
    "automatic_confidence_level",
    "automatic_split_fraction",
    "proposed_image_left_side",
    "proposed_image_right_side",
    "split_acceptable_yes_no",
    "laterality_mapping_supported_yes_no",
    "observed_marker_or_anatomy",
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


def normalize_working_copy(pixels: np.ndarray, photometric: str) -> np.ndarray:
    """Return a float working copy; never alters the native pixel array."""
    image = np.asarray(pixels)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2 or min(image.shape) < 32:
        raise ValueError("unsupported_pixel_geometry")
    image = image.astype(np.float64, copy=False)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("non_finite_pixel_array")
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        raise ValueError("constant_pixel_array")
    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    photometric = str(photometric).upper().strip()
    if photometric == "MONOCHROME1":
        normalized = 1.0 - normalized
    elif photometric != "MONOCHROME2":
        raise ValueError("unsupported_photometric_interpretation")
    return normalized.astype(np.float32)


def _smooth_profile(values: np.ndarray, width: int) -> np.ndarray:
    width = max(3, int(width))
    if width % 2 == 0:
        width += 1
    width = min(width, len(values) - (1 - len(values) % 2))
    if width < 3:
        return np.asarray(values, dtype=np.float64)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def estimate_bilateral_split(
    normalized: np.ndarray, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Estimate a central separation line using intensity and edge profiles."""
    rows, columns = normalized.shape
    search_low, search_high = (float(value) for value in parameters["search_band"])
    vertical_low, vertical_high = (
        float(value) for value in parameters["vertical_analysis_band"]
    )
    if not 0.0 < search_low < 0.5 < search_high < 1.0:
        raise ValueError("invalid_search_band")
    if not 0.0 <= vertical_low < vertical_high <= 1.0:
        raise ValueError("invalid_vertical_analysis_band")

    y0 = int(round(rows * vertical_low))
    y1 = int(round(rows * vertical_high))
    analysis = normalized[y0:y1]
    if analysis.size == 0:
        raise ValueError("empty_analysis_band")

    intensity = np.median(analysis, axis=0)
    horizontal_gradient = np.median(
        np.abs(np.diff(analysis, axis=1, prepend=analysis[:, :1])), axis=0
    )
    smooth_width = max(3, round(columns * float(parameters["smoothing_fraction"])))
    intensity = _smooth_profile(_robust_unit(intensity), smooth_width)
    horizontal_gradient = _smooth_profile(
        _robust_unit(horizontal_gradient), smooth_width
    )
    anatomical_signal = (
        float(parameters["intensity_weight"]) * intensity
        + float(parameters["gradient_weight"]) * horizontal_gradient
    )

    first = max(1, int(round(columns * search_low)))
    last = min(columns - 1, int(round(columns * search_high)))
    candidates = np.arange(first, last, dtype=int)
    midpoint = (columns - 1) / 2.0
    half_band = max(1.0, (last - first) / 2.0)
    center_distance = np.abs(candidates - midpoint) / half_band
    scores = anatomical_signal[candidates] + float(parameters["center_penalty"]) * center_distance
    best_index = int(np.argmin(scores))
    split_column = int(candidates[best_index])

    candidate_signal = anatomical_signal[candidates]
    valley_prominence = float(
        np.clip(np.median(candidate_signal) - anatomical_signal[split_column], 0.0, 1.0)
    )
    left_width = split_column
    right_width = columns - split_column
    half_width_ratio = min(left_width, right_width) / max(left_width, right_width)
    midpoint_offset_fraction = abs(split_column - midpoint) / columns

    minimum_ratio = float(parameters["minimum_half_width_ratio"])
    minimum_prominence = float(parameters["minimum_valley_prominence"])
    geometry_component = min(1.0, half_width_ratio / max(minimum_ratio, 1e-6))
    prominence_component = min(
        1.0, valley_prominence / max(minimum_prominence, 1e-6)
    )
    center_component = max(0.0, 1.0 - midpoint_offset_fraction / (search_high - 0.5))
    confidence_score = float(
        0.35 * geometry_component
        + 0.45 * prominence_component
        + 0.20 * center_component
    )
    high_confidence = (
        half_width_ratio >= minimum_ratio
        and confidence_score >= float(parameters["high_confidence_threshold"])
    )

    return {
        "split_column": split_column,
        "split_fraction": split_column / columns,
        "midpoint_column": int(round(midpoint)),
        "midpoint_offset_fraction": midpoint_offset_fraction,
        "image_left_width": left_width,
        "image_right_width": right_width,
        "half_width_ratio": half_width_ratio,
        "valley_prominence": valley_prominence,
        "confidence_score": confidence_score,
        "confidence_level": "HIGH" if high_confidence else "LOW",
        "technical_status": "CANDIDATE_OK" if high_confidence else "REVIEW_REQUIRED",
    }


def _read_single_image(path: Path) -> tuple[bytes, pydicom.Dataset, np.ndarray]:
    valid: list[tuple[bytes, pydicom.Dataset, np.ndarray]] = []
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            try:
                dataset = pydicom.dcmread(io.BytesIO(payload), force=True)
                if "PixelData" not in dataset:
                    continue
                pixels = dataset.pixel_array
                if pixels.size:
                    valid.append((payload, dataset, pixels))
            except Exception:
                continue
    if len(valid) != 1:
        raise ValueError("package_must_contain_one_readable_dicom")
    return valid[0]


def _resolve_package(source_dir: Path, relative_path: str) -> Path:
    exact = (source_dir / Path(relative_path)).resolve()
    if exact.is_relative_to(source_dir) and exact.is_file():
        return exact
    name = Path(relative_path).name
    matches = [path for path in source_dir.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError("canonical_package_not_found_uniquely")
    return matches[0].resolve()


def _make_preview(
    normalized: np.ndarray,
    split_column: int,
    proposed_mapping: dict[str, str],
    output_path: Path,
    max_width: int,
) -> None:
    grayscale = Image.fromarray(np.round(normalized * 255.0).astype(np.uint8), mode="L")
    if grayscale.width > max_width:
        ratio = max_width / grayscale.width
        grayscale = grayscale.resize(
            (max_width, max(1, round(grayscale.height * ratio))), Image.Resampling.LANCZOS
        )
    scale = grayscale.width / normalized.shape[1]
    split_preview = int(round(split_column * scale))
    preview = grayscale.convert("RGB")
    draw = ImageDraw.Draw(preview)
    line_width = max(2, preview.width // 500)
    draw.line(
        [(split_preview, 0), (split_preview, preview.height)],
        fill=(255, 40, 40),
        width=line_width,
    )
    label = (
        f"IMAGE LEFT -> PATIENT {proposed_mapping['image_left']}"
        f" | IMAGE RIGHT -> PATIENT {proposed_mapping['image_right']}"
    )
    draw.rectangle((0, 0, min(preview.width, 670), 24), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 0))
    preview.save(output_path, format="PNG", optimize=True)


def _load_verified_pilot(audit_json: Path, expected: int) -> list[dict[str, str]]:
    payload = json.loads(audit_json.read_text(encoding="utf-8"))
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
    return sorted(selected, key=lambda record: str(record["manifest_key"]))


def _review_has_human_decisions(path: Path) -> bool:
    if not path.is_file():
        return False
    human_fields = REVIEW_FIELDS[6:]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return any(
            any(str(row.get(field, "")).strip() for field in human_fields)
            for row in csv.DictReader(stream, delimiter=";")
        )


def _validate_parameters(parameters: dict[str, Any]) -> None:
    required = {
        "expected_unique_studies",
        "algorithm_version",
        "search_band",
        "vertical_analysis_band",
        "smoothing_fraction",
        "intensity_weight",
        "gradient_weight",
        "center_penalty",
        "minimum_half_width_ratio",
        "minimum_valley_prominence",
        "high_confidence_threshold",
        "proposed_mapping",
        "max_preview_width",
    }
    missing = sorted(required - parameters.keys())
    if missing:
        raise ValueError("missing_parameters:" + ",".join(missing))
    weight_sum = float(parameters["intensity_weight"]) + float(
        parameters["gradient_weight"]
    )
    if not np.isclose(weight_sum, 1.0):
        raise ValueError("intensity_and_gradient_weights_must_sum_to_one")
    if int(parameters["expected_unique_studies"]) <= 0:
        raise ValueError("expected_unique_studies_must_be_positive")
    if int(parameters["max_preview_width"]) < 320:
        raise ValueError("max_preview_width_is_too_small_for_review")


def prepare_bilateral_pilot(
    source_dir: Path,
    pilot_audit_json: Path,
    output_dir: Path,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Create pseudonymous previews and a review ledger without reading outcomes."""
    _validate_parameters(parameters)
    expected = int(parameters["expected_unique_studies"])
    selected = _load_verified_pilot(pilot_audit_json, expected)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews_ciegas"
    preview_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "revision_visual_ciega.csv"
    if _review_has_human_decisions(review_path):
        raise RuntimeError("existing_human_review_refusing_to_overwrite")
    for stale_preview in preview_dir.glob("case_*_separacion.png"):
        stale_preview.unlink()
    proposed_mapping = dict(parameters["proposed_mapping"])
    if proposed_mapping != {"image_left": "RIGHT", "image_right": "LEFT"}:
        raise ValueError("unexpected_laterality_mapping_requires_methodological_review")

    records: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    case_map: list[dict[str, str]] = []
    failures = 0
    for index, expected_record in enumerate(selected, start=1):
        alias = f"case_{index:03d}"
        record: dict[str, Any] = {field: "" for field in PRIVATE_RESULT_FIELDS}
        record.update(
            {
                "case_alias": alias,
                "manifest_key": expected_record["manifest_key"],
                "package_relative_path": expected_record["package_relative_path"],
                "package_sha256": expected_record["package_sha256"],
                "dicom_sha256": expected_record["dicom_sha256"],
                "pixel_sha256": expected_record["pixel_sha256"],
                "proposed_image_left_side": proposed_mapping["image_left"],
                "proposed_image_right_side": proposed_mapping["image_right"],
            }
        )
        try:
            package = _resolve_package(source_dir, expected_record["package_relative_path"])
            if sha256_file(package) != expected_record["package_sha256"]:
                raise ValueError("package_hash_mismatch")
            dicom_payload, dataset, pixels = _read_single_image(package)
            if sha256_bytes(dicom_payload) != expected_record["dicom_sha256"]:
                raise ValueError("dicom_hash_mismatch")
            pixel_hash = sha256_bytes(np.ascontiguousarray(pixels).tobytes())
            if pixel_hash != expected_record["pixel_sha256"]:
                raise ValueError("pixel_hash_mismatch")
            photometric = str(getattr(dataset, "PhotometricInterpretation", ""))
            normalized = normalize_working_copy(pixels, photometric)
            split = estimate_bilateral_split(normalized, parameters)
            preview_name = f"{alias}_separacion.png"
            _make_preview(
                normalized,
                int(split["split_column"]),
                proposed_mapping,
                preview_dir / preview_name,
                int(parameters["max_preview_width"]),
            )
            record.update(split)
            record.update(
                {
                    "rows": normalized.shape[0],
                    "columns": normalized.shape[1],
                    "photometric_interpretation": photometric,
                    "dicom_laterality": str(getattr(dataset, "Laterality", "") or ""),
                    "preview_file": f"previews_ciegas/{preview_name}",
                }
            )
        except Exception as exc:
            failures += 1
            record["technical_status"] = "ERROR"
            record["confidence_level"] = "NONE"
            record["error_code"] = str(exc)
        records.append(record)
        review_rows.append(
            {
                "case_alias": alias,
                "preview_file": record["preview_file"],
                "automatic_confidence_level": record["confidence_level"],
                "automatic_split_fraction": record["split_fraction"],
                "proposed_image_left_side": proposed_mapping["image_left"],
                "proposed_image_right_side": proposed_mapping["image_right"],
                "split_acceptable_yes_no": "",
                "laterality_mapping_supported_yes_no": "",
                "observed_marker_or_anatomy": "",
                "technical_exclusion_yes_no": "",
                "exclusion_reason": "",
                "reviewed_without_outcome_yes_no": "",
                "reviewer_notes": "",
            }
        )
        case_map.append(
            {
                "case_alias": alias,
                "manifest_key": str(expected_record["manifest_key"]),
                "pixel_sha256": str(expected_record["pixel_sha256"]),
            }
        )

    with (output_dir / "resultados_separacion_privados.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=PRIVATE_RESULT_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(records)
    with review_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(review_rows)
    with (output_dir / "mapa_casos_privado.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("case_alias", "manifest_key", "pixel_sha256"), delimiter=";"
        )
        writer.writeheader()
        writer.writerows(case_map)

    successful = [record for record in records if not record["error_code"]]
    fractions = [float(record["split_fraction"]) for record in successful]
    confidences = Counter(str(record["confidence_level"]) for record in successful)
    public_summary = {
        "status": "ready_for_blinded_review"
        if len(successful) == expected and failures == 0
        else "technical_failure",
        "algorithm_version": parameters["algorithm_version"],
        "expected_unique_studies": expected,
        "processed_unique_studies": len(successful),
        "technical_failures": failures,
        "confidence_counts": dict(sorted(confidences.items())),
        "split_fraction": {
            "minimum": min(fractions) if fractions else None,
            "median": float(np.median(fractions)) if fractions else None,
            "maximum": max(fractions) if fractions else None,
        },
        "laterality_mapping": {
            "image_left": proposed_mapping["image_left"],
            "image_right": proposed_mapping["image_right"],
            "status": "proposed_pending_blinded_visual_confirmation",
            "dicom_laterality_used_as_decision": False,
        },
        "visual_review_status": "pending",
        "outcome_data_loaded": False,
        "mass_processing_executed": False,
        "training_executed": False,
        "reserved_test_opened": False,
    }
    (output_dir / "resumen_separacion_publico.json").write_text(
        json.dumps(public_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    frozen_candidate = {
        key: parameters[key]
        for key in (
            "algorithm_version",
            "search_band",
            "vertical_analysis_band",
            "smoothing_fraction",
            "intensity_weight",
            "gradient_weight",
            "center_penalty",
            "minimum_half_width_ratio",
            "minimum_valley_prominence",
            "high_confidence_threshold",
            "proposed_mapping",
        )
    }
    frozen_candidate["status"] = "candidate_not_frozen_until_visual_review"
    frozen_candidate["source_pilot_audit_sha256"] = sha256_file(pilot_audit_json)
    (output_dir / "parametros_candidatos.json").write_text(
        json.dumps(frozen_candidate, indent=2, ensure_ascii=False) + "\n",
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
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("source_dir", "pilot_audit_json", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing relative path: {key}.")
        config[key] = _resolve_under_root(root, config[key], key)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = prepare_bilateral_pilot(
        config["source_dir"],
        config["pilot_audit_json"],
        config["output_dir"],
        config,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "ready_for_blinded_review":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
