"""Run only the approved MCR-2026-004 historical 10-study/20-knee regression."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import pydicom
import scipy
from PIL import Image, ImageDraw
from PIL import __version__ as pillow_version

from knee.bilateral_separation import _read_single_image
from knee.dicom_audit import sha256_bytes, sha256_file
from knee.joint_localization import (
    _extract_spacing, _load_closed_bilateral_pilot, _load_verified_pilot,
    _resolve_package, _resolve_under_root,
)
from knee.roi_mcr004 import ALGORITHM_VERSION, UPSTREAM_COMMIT, candidate_for_half, normalize_half
from knee.third_party.emory_hiti.config import CropConfig


RESULT_FIELDS = (
    "case_alias", "knee_alias", "patient_side", "manifest_key",
    "package_relative_path", "dicom_sha256", "pixel_sha256",
    "status", "error_code", "half_x0", "half_x1", "native_dicom_box",
    "requested_working_box", "crop_rows", "crop_columns",
    "row_spacing_mm", "column_spacing_mm", "crop_height_mm", "crop_width_mm",
    "native_crop_sha256", "trace_json", "preview_file", "native_crop_file",
)
REVIEW_FIELDS = (
    "knee_alias", "case_alias", "patient_side", "candidate_status",
    "preview_file", "decision", "laterality_ok_yes_no", "coverage_ok_yes_no",
    "frame_ok_yes_no", "visualizable_yes_no",
    "critical_contamination_yes_no", "peripheral_warning_yes_no",
    "outcome_blinded_yes_no", "reviewer_notes",
)


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("expected_json_object")
    return result


def validate_prior_v04_closure(summary_path: Path, record_path: Path) -> dict[str, Any]:
    """Refuse to bypass the Colab closure prerequisite."""
    summary, record = _read_json(summary_path), _read_json(record_path)
    expected = {
        "status": "rejected_after_blinded_review",
        "algorithm_version": "tibiofemoral_crop_v0.4_pilot",
        "reviewed_knees": 20,
        "acceptable_crops": 8,
        "rejected_crops": 12,
        "technical_exclusions": 0,
        "parameters_frozen": False,
        "mass_processing_executed": False,
        "partitions_created": False,
        "training_executed": False,
        "reserved_test_opened": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value or record.get(key) != value:
            raise ValueError(f"v04_closure_mismatch_{key}")
    if record.get("closure_execution_environment") != "Google Colab, libreta 12":
        raise ValueError("v04_colab_closure_not_proven")
    if summary.get("review_csv_sha256") != record.get("review_csv_sha256"):
        raise ValueError("v04_review_csv_hash_mismatch")
    if not summary.get("review_csv_sha256"):
        raise ValueError("v04_review_csv_hash_missing")
    for payload in (summary, record):
        if payload.get("rejection_reason_counts") != {
            "joint_not_centered": 9, "peripheral_artifact_present": 6
        }:
            raise ValueError("v04_rejection_reasons_mismatch")
    return {"summary_sha256": sha256_file(summary_path),
            "record_sha256": sha256_file(record_path),
            "review_csv_sha256": summary["review_csv_sha256"]}


def _preview(
    native_half: np.ndarray, photometric: str,
    box: list[int], target: Path, max_width: int,
) -> None:
    gray = Image.fromarray(normalize_half(native_half, photometric))
    marked = gray.convert("RGB")
    x0, x1, y0, y1 = box
    ImageDraw.Draw(marked).rectangle(
        (x0, y0, x1-1, y1-1), outline=(0, 255, 255),
        width=max(2, marked.width//300),
    )
    crop = gray.crop((x0, y0, x1, y1)).convert("RGB")
    if crop.height != marked.height:
        crop = crop.resize(
            (max(1, round(crop.width*marked.height/crop.height)), marked.height),
            Image.Resampling.LANCZOS,
        )
    canvas = Image.new("RGB", (marked.width+crop.width, marked.height), "black")
    canvas.paste(marked, (0, 0))
    canvas.paste(crop, (marked.width, 0))
    if canvas.width > max_width:
        canvas = canvas.resize(
            (max_width, max(1, round(canvas.height*max_width/canvas.width))),
            Image.Resampling.LANCZOS,
        )
    canvas.save(target, format="PNG")
    if not target.is_file() or target.stat().st_size == 0:
        raise OSError("preview_write_failed")


def _save_crop(path: Path, crop: np.ndarray) -> str:
    with path.open("wb") as stream:
        np.save(stream, crop, allow_pickle=False)
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError("native_crop_write_failed")
    with path.open("rb") as stream:
        reread = np.load(stream, allow_pickle=False)
    if not np.array_equal(reread, crop):
        raise OSError("native_crop_roundtrip_mismatch")
    return sha256_bytes(np.ascontiguousarray(crop).tobytes())


def run_historical_regression(config: dict[str, Any]) -> dict[str, Any]:
    if config["algorithm_version"] != ALGORITHM_VERSION:
        raise ValueError("unexpected_algorithm_version")
    if config["expected_unique_studies"] != 10 or config["expected_knees"] != 20:
        raise ValueError("historical_pilot_size_must_remain_10_studies_20_knees")
    source = config["source_dir"]
    bilateral_dir = config["bilateral_output_dir"]
    output = config["output_dir"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("historical_regression_output_must_be_new")
    closure_hashes = validate_prior_v04_closure(
        config["v04_review_summary_json"], config["v04_closure_record_json"]
    )
    bilateral = _load_closed_bilateral_pilot(
        bilateral_dir, 10, "bilateral_split_v0.2_pilot"
    )
    audit = _load_verified_pilot(config["pilot_audit_json"], 10)
    output.mkdir(parents=True, exist_ok=True)
    previews, crops = output/"previews_ciegas", output/"recortes_nativos"
    previews.mkdir(exist_ok=False)
    crops.mkdir(exist_ok=False)
    core_cfg = CropConfig()
    result_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for study in bilateral:
        case_alias = study["case_alias"]
        manifest_key = study["manifest_key"]
        source_record = audit.get(manifest_key)
        if source_record is None:
            raise ValueError("closed_bilateral_case_missing_from_audit")
        package = _resolve_package(source, source_record["package_relative_path"])
        if sha256_file(package) != source_record["package_sha256"]:
            raise ValueError("source_package_hash_mismatch")
        dicom_payload, dataset, pixels = _read_single_image(package)
        if sha256_bytes(dicom_payload) != source_record["dicom_sha256"]:
            raise ValueError("source_dicom_hash_mismatch")
        if sha256_bytes(np.ascontiguousarray(pixels).tobytes()) != source_record["pixel_sha256"]:
            raise ValueError("source_pixel_hash_mismatch")
        split = int(study["split_column"])
        if not 0 < split < pixels.shape[1]:
            raise ValueError("invalid_frozen_split")
        row_mm, col_mm, _ = _extract_spacing(
            dataset, ["ImagerPixelSpacing", "PixelSpacing"]
        )
        photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper().strip()
        halves = (
            ("RIGHT", 0, split, pixels[:, :split]),
            ("LEFT", split, pixels.shape[1], pixels[:, split:]),
        )
        for side, half_x0, half_x1, native_half in halves:
            alias = f"{case_alias}_{side}"
            candidate = candidate_for_half(
                native_half, side, photometric, half_x0, row_mm, col_mm, core_cfg
            )
            row = {field: "" for field in RESULT_FIELDS}
            row.update({
                "case_alias": case_alias, "knee_alias": alias, "patient_side": side,
                "manifest_key": manifest_key,
                "package_relative_path": source_record["package_relative_path"],
                "dicom_sha256": source_record["dicom_sha256"],
                "pixel_sha256": source_record["pixel_sha256"],
                "status": candidate["status"], "error_code": candidate.get("error_code", ""),
                "half_x0": half_x0, "half_x1": half_x1,
                "row_spacing_mm": row_mm, "column_spacing_mm": col_mm,
            })
            if candidate["status"] == "candidate":
                try:
                    crop_path = crops/f"{alias}_crop.npy"
                    preview_path = previews/f"{alias}_localizacion.png"
                    crop_hash = _save_crop(crop_path, candidate["crop"])
                    _preview(native_half, photometric, candidate["native_half_box"],
                             preview_path, int(config["max_preview_width"]))
                    row.update({
                        "native_dicom_box": json.dumps(candidate["native_dicom_box"]),
                        "requested_working_box": json.dumps(candidate["requested_working_box"]),
                        "crop_rows": candidate["crop_rows"],
                        "crop_columns": candidate["crop_columns"],
                        "crop_height_mm": candidate["crop_height_mm"],
                        "crop_width_mm": candidate["crop_width_mm"],
                        "native_crop_sha256": crop_hash,
                        "trace_json": json.dumps(candidate["trace"], sort_keys=True),
                        "preview_file": f"previews_ciegas/{preview_path.name}",
                        "native_crop_file": f"recortes_nativos/{crop_path.name}",
                    })
                except (OSError, ValueError) as exc:
                    crop_path.unlink(missing_ok=True)
                    preview_path.unlink(missing_ok=True)
                    row.update({
                        "status": "abstain", "error_code": f"output_failure:{type(exc).__name__}",
                    })
            result_rows.append(row)
            review_rows.append({
                **{field: "" for field in REVIEW_FIELDS},
                "knee_alias": alias, "case_alias": case_alias, "patient_side": side,
                "candidate_status": row["status"], "preview_file": row["preview_file"],
            })
    if len(result_rows) != 20 or len(review_rows) != 20:
        raise AssertionError("historical_regression_denominator_changed")
    for name, fields, rows in (
        ("resultados_roi_privados.csv", RESULT_FIELDS, result_rows),
        ("revision_tecnica_ciega.csv", REVIEW_FIELDS, review_rows),
    ):
        with (output/name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
    counts = Counter(row["status"] for row in result_rows)
    summary = {
        "status": "ready_for_technical_review", "algorithm_version": ALGORITHM_VERSION,
        "upstream_commit": UPSTREAM_COMMIT, "prior_v04_closure": closure_hashes,
        "expected_unique_studies": 10, "expected_knees": 20,
        "candidate_knees": counts["candidate"], "abstained_knees": counts["abstain"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "opencv": cv2.__version__,
                        "pydicom": pydicom.__version__, "Pillow": pillow_version},
        "core_config": asdict(core_cfg), "visual_review_status": "pending",
        "outcome_data_loaded": False, "mass_processing_executed": False,
        "partitions_created": False, "training_executed": False,
        "reserved_test_opened": False,
    }
    (output/"resumen_regresion_publico.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"
    )
    return summary


def load_config(path: Path) -> dict[str, Any]:
    root_value = os.environ.get("KNEE_DATA_ROOT")
    if not root_value:
        raise ValueError("Set KNEE_DATA_ROOT to the authorized data directory.")
    root = Path(root_value).expanduser().resolve()
    config = _read_json(path)
    for key in (
        "source_dir", "pilot_audit_json", "bilateral_output_dir", "output_dir",
        "v04_review_summary_json", "v04_closure_record_json",
    ):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing_relative_path_{key}")
        config[key] = _resolve_under_root(root, value, key)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_historical_regression(load_config(args.config)),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
