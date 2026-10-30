"""Review existing MCR004 crops in Colab; never run or tune a localizer."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from knee.bilateral_separation import _read_single_image
from knee.dicom_audit import sha256_bytes, sha256_file
from knee.joint_localization import (
    _extract_spacing, _load_closed_bilateral_pilot, _load_verified_pilot,
    _resolve_package, _resolve_under_root,
)
from knee.roi_mcr004 import ALGORITHM_VERSION, UPSTREAM_COMMIT, Trace
from knee.roi_mcr004_pilot import (
    RESULT_FIELDS, REVIEW_FIELDS, _read_json, load_config, validate_prior_v04_closure,
)
from knee.third_party.emory_hiti.config import CropConfig


ACCEPTED = {"aceptable", "aceptable_con_advertencia_periferica"}
DECISIONS = ACCEPTED | {"rechazado", "no_evaluable"}
FLAGS = REVIEW_FIELDS[6:13]
IMMUTABLE = REVIEW_FIELDS[:5]
BLOCKED = ("mass_processing_executed", "partitions_created",
           "training_executed", "reserved_test_opened")
SUMMARY_FILE = "resumen_revision_tecnica_publico.json"
RECORD_FILE = "registro_cierre_revision_tecnica.json"
COMPLETED_CSV = "revision_tecnica_asistida_completada.csv"
COMPLETED_PROVENANCE = "procedencia_revision_asistida_completada.json"


def _rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"unexpected_csv_schema:{path.name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("malformed_csv_row")
    return rows


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, delimiter=";")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2)+"\n").encode("utf-8")


def _require_false(payload: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if payload.get(field) is not False:
            raise ValueError(f"blocked_stage_not_false:{field}")


def read_context(config: dict[str, Any]) -> dict[str, Any]:
    """Anchor the assisted draft to the successful historical run and its template."""
    output = config["output_dir"]
    if config["algorithm_version"] != ALGORITHM_VERSION or (
            config["expected_unique_studies"], config["expected_knees"]) != (10, 20):
        raise ValueError("historical_contract_changed")
    if (output/RECORD_FILE).exists() or (output/SUMMARY_FILE).exists():
        raise FileExistsError("review_already_closed_or_partial_closure_present")
    prep = _read_json(output/"registro_preparacion.json")
    summary = _read_json(output/"resumen_regresion_publico.json")
    provenance = _read_json(output/"procedencia_revision_asistida.json")
    if (prep.get("operation") != "prepare_mcr004_historical_regression" or
            summary.get("status") != "ready_for_technical_review" or
            summary.get("algorithm_version") != ALGORITHM_VERSION or
            summary.get("upstream_commit") != UPSTREAM_COMMIT):
        raise ValueError("successful_historical_run_not_proven")
    if sha256_file(output/"resumen_regresion_publico.json") != prep.get("summary_sha256"):
        raise ValueError("preparation_summary_hash_mismatch")
    if summary.get("core_config") != json.loads(json.dumps(asdict(CropConfig()))):
        raise ValueError("localizer_parameters_changed")
    _require_false(summary, BLOCKED+("outcome_data_loaded",))
    _require_false(prep, BLOCKED)
    if provenance.get("status") != "assisted_draft_pending_investigator_confirmation":
        raise ValueError("unexpected_assisted_review_status")
    if (provenance.get("outcome_data_consulted") is not False or
            provenance.get("independent_second_reader") is not False or
            provenance.get("investigator_confirmation") is not False or
            provenance.get("closure_executed") is not False or
            provenance.get("version_known_to_assistant") is not True):
        raise ValueError("assisted_provenance_mismatch")
    for name, expected in (
        ("revision_tecnica_ciega.csv", provenance.get("prepared_review_csv_sha256")),
        ("resultados_roi_privados.csv", provenance.get("results_csv_sha256")),
        ("revision_tecnica_ciega_plantilla_original.csv", prep.get("review_template_sha256")),
    ):
        if not expected or sha256_file(output/name) != expected:
            raise ValueError(f"review_input_hash_mismatch:{name}")
    if (provenance.get("original_template_sha256") != prep.get("review_template_sha256") or
            provenance.get("preparation_summary_sha256") != prep.get("summary_sha256") or
            provenance.get("pilot_git_commit") != prep.get("git_commit")):
        raise ValueError("assisted_draft_run_mismatch")
    results = _rows(output/"resultados_roi_privados.csv", RESULT_FIELDS)
    review = _rows(output/"revision_tecnica_ciega.csv", REVIEW_FIELDS)
    template = _rows(output/"revision_tecnica_ciega_plantilla_original.csv", REVIEW_FIELDS)
    if len(results) != 20 or len(review) != 20 or len(template) != 20:
        raise ValueError("historical_denominator_changed")
    keys = [row["knee_alias"] for row in results]
    expected = {f"case_{i:03d}_{side}" for i in range(1, 11) for side in ("RIGHT", "LEFT")}
    if set(keys) != expected or len(set(keys)) != 20:
        raise ValueError("historical_aliases_changed")
    if [r["knee_alias"] for r in review] != keys or [r["knee_alias"] for r in template] != keys:
        raise ValueError("review_identity_or_order_changed")
    counts = Counter(r["status"] for r in results)
    if set(counts) - {"candidate", "abstain"}:
        raise ValueError("invalid_candidate_status")
    for payload in (summary, prep):
        if payload.get("candidate_knees") != counts["candidate"] or (
                payload.get("abstained_knees") != counts["abstain"]):
            raise ValueError("candidate_count_mismatch")
    if summary.get("expected_unique_studies") != 10 or summary.get("expected_knees") != 20:
        raise ValueError("summary_denominator_changed")
    for result, row, original in zip(results, review, template):
        identity = dict(zip(IMMUTABLE, (
            result["knee_alias"], result["case_alias"], result["patient_side"],
            result["status"], result["preview_file"],
        )))
        if any(row[key] != identity[key] or original[key] != identity[key] for key in IMMUTABLE):
            raise ValueError("review_identity_changed")
        if any(original[key] for key in REVIEW_FIELDS[5:]):
            raise ValueError("original_template_was_not_blank")
        if result["knee_alias"] != result["case_alias"]+"_"+result["patient_side"]:
            raise ValueError("result_side_alias_mismatch")
    pending = [r["knee_alias"] for r in review if r["candidate_status"] == "candidate" and not r["decision"]]
    if pending != provenance.get("pending_knee_aliases") or (
            len(review)-len(pending) != provenance.get("proposed_decisions")):
        raise ValueError("assisted_pending_count_mismatch")
    return {"results": results, "review": review, "pending_knee_aliases": pending,
            "review_csv_sha256": sha256_file(output/"revision_tecnica_ciega.csv"),
            "pilot_git_commit": prep["git_commit"], "provenance": provenance,
            "summary": summary}


def _source_studies(config: dict[str, Any]):
    closure = validate_prior_v04_closure(
        config["v04_review_summary_json"], config["v04_closure_record_json"], config["v04_review_csv"])
    prep = _read_json(config["output_dir"]/"registro_preparacion.json")
    if closure != prep.get("prior_v04_closure"):
        raise ValueError("prior_closure_changed_since_historical_run")
    studies = _load_closed_bilateral_pilot(
        config["bilateral_output_dir"], 10, "bilateral_split_v0.2_pilot")
    audit = _load_verified_pilot(config["pilot_audit_json"], 10)
    for study in studies:
        record = audit.get(study["manifest_key"])
        if not record:
            raise ValueError("historical_source_missing_from_audit")
        package = _resolve_package(config["source_dir"], record["package_relative_path"])
        if sha256_file(package) != record["package_sha256"]:
            raise ValueError("source_package_hash_mismatch")
        payload, dataset, pixels = _read_single_image(package)
        if sha256_bytes(payload) != record["dicom_sha256"] or (
                sha256_bytes(np.ascontiguousarray(pixels).tobytes()) != record["pixel_sha256"]):
            raise ValueError("source_dicom_or_pixel_hash_mismatch")
        if pixels.ndim != 2 or not 0 < int(study["split_column"]) < pixels.shape[1]:
            raise ValueError("invalid_frozen_source_geometry")
        yield study, record, dataset, pixels


def _verify_crop(config, row, study, record, dataset, pixels) -> tuple[np.ndarray, list[int]]:
    if row["manifest_key"] != study["manifest_key"] or any(
            row[key] != record[key] for key in ("package_relative_path", "dicom_sha256", "pixel_sha256")):
        raise ValueError("result_source_identity_mismatch")
    split = int(study["split_column"])
    start, end = (0, split) if row["patient_side"] == "RIGHT" else (split, pixels.shape[1])
    if (int(row["half_x0"]), int(row["half_x1"])) != (start, end):
        raise ValueError("frozen_half_changed")
    box = json.loads(row["native_dicom_box"])
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("invalid_native_box")
    x0, x1, y0, y1 = box
    if not (start <= x0 < x1 <= end and 0 <= y0 < y1 <= pixels.shape[0]):
        raise ValueError("native_crop_outside_frozen_half")
    trace = Trace(**json.loads(row["trace_json"]))
    if (trace.input_rows, trace.input_columns) != (pixels.shape[0], end-start):
        raise ValueError("trace_input_geometry_mismatch")
    local = list(trace.to_native_bounds(tuple(json.loads(row["requested_working_box"]))))
    if [local[0]+start, local[1]+start, local[2], local[3]] != box:
        raise ValueError("trace_native_box_mismatch")
    native_path = _resolve_under_root(config["output_dir"], row["native_crop_file"], "native_crop_file")
    saved = np.load(native_path, allow_pickle=False)
    crop = pixels[y0:y1, x0:x1]
    if (saved.dtype != crop.dtype or saved.shape != crop.shape or
            not np.array_equal(saved, crop) or
            sha256_bytes(np.ascontiguousarray(saved).tobytes()) != row["native_crop_sha256"] or
            crop.shape != (int(row["crop_rows"]), int(row["crop_columns"]))):
        raise ValueError("native_crop_pixel_or_shape_mismatch")
    row_mm, col_mm, _ = _extract_spacing(dataset, ["ImagerPixelSpacing", "PixelSpacing"])
    for field, actual in (
        ("row_spacing_mm", row_mm), ("column_spacing_mm", col_mm),
        ("crop_height_mm", crop.shape[0]*row_mm), ("crop_width_mm", crop.shape[1]*col_mm),
    ):
        if not math.isclose(float(row[field]), actual, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("native_crop_spacing_mismatch")
    preview = _resolve_under_root(config["output_dir"], row["preview_file"], "preview_file")
    if not preview.is_file() or preview.stat().st_size == 0:
        raise ValueError("candidate_preview_missing")
    return pixels[:, start:end], local


def verify_native_integrity(config: dict[str, Any], results: list[dict[str, str]]) -> int:
    """Compare the 20 saved ROIs with the 10 approved originals, without inference."""
    seen, checked = set(), 0
    for study, record, dataset, pixels in _source_studies(config):
        matching = [r for r in results if r["case_alias"] == study["case_alias"]]
        if len(matching) != 2 or {r["patient_side"] for r in matching} != {"LEFT", "RIGHT"}:
            raise ValueError("historical_study_mapping_mismatch")
        seen.add(study["case_alias"])
        for row in matching:
            if row["status"] == "candidate":
                _verify_crop(config, row, study, record, dataset, pixels)
                checked += 1
    if seen != {r["case_alias"] for r in results}:
        raise ValueError("historical_source_studies_changed")
    return checked


def window_image(half: np.ndarray, box: list[int], photometric: str,
                 low: float, high: float, max_width: int = 1600) -> Image.Image:
    """Display-only linear window, identical for original half and existing ROI."""
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("invalid_display_window")
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("unsupported_display_photometric")
    gray = np.clip((half.astype(np.float64)-low)/(high-low), 0, 1)
    if photometric == "MONOCHROME1":
        gray = 1-gray
    original = Image.fromarray(np.rint(255*gray).astype(np.uint8))
    marked = original.convert("RGB")
    x0, x1, y0, y1 = box
    ImageDraw.Draw(marked).rectangle((x0, y0, x1-1, y1-1), outline=(0, 255, 255), width=2)
    roi = original.crop((x0, y0, x1, y1)).convert("RGB")
    roi = roi.resize((max(1, round(roi.width*marked.height/roi.height)), marked.height),
                     Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (marked.width+roi.width, marked.height))
    canvas.paste(marked, (0, 0))
    canvas.paste(roi, (marked.width, 0))
    if canvas.width > max_width:
        canvas = canvas.resize((max_width, max(1, round(canvas.height*max_width/canvas.width))),
                               Image.Resampling.LANCZOS)
    return canvas


def make_view(config: dict[str, Any], alias: str, low: float | None, high: float | None):
    context = read_context(config)
    matching = [r for r in context["results"] if r["knee_alias"] == alias and r["status"] == "candidate"]
    if len(matching) != 1:
        raise ValueError("unknown_candidate_alias")
    row = matching[0]
    for study, record, dataset, pixels in _source_studies(config):
        if study["case_alias"] != row["case_alias"]:
            continue
        half, box = _verify_crop(config, row, study, record, dataset, pixels)
        crop = half[box[2]:box[3], box[0]:box[1]]
        baseline = np.percentile(half, [1, 99]).tolist()
        suggested = np.percentile(crop, [.5, 99.5]).tolist()
        if low is None and high is None:
            low, high = baseline
        elif low is None or high is None:
            raise ValueError("both_display_limits_required")
        folder = config["output_dir"]/"revision_visualizacion"
        folder.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        file = folder/f"{alias}_{stamp}.png"
        window_image(half, box, str(dataset.PhotometricInterpretation).upper(),
                     float(low), float(high)).save(file, format="PNG")
        view = {"knee_alias": alias, "display_low": float(low), "display_high": float(high),
                "baseline_window": baseline, "suggested_roi_window": suggested,
                "pixel_min": float(half.min()), "pixel_max": float(half.max()),
                "native_crop_sha256": row["native_crop_sha256"],
                "review_csv_sha256": context["review_csv_sha256"],
                "display_file": file.relative_to(config["output_dir"]).as_posix(),
                "display_sha256": sha256_file(file), "visualization_only": True,
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
        # Independent record prevents edited response JSON from inventing a rendered window.
        with file.with_suffix(".json").open("xb") as stream:
            stream.write(_json_bytes(view))
        return view
    raise ValueError("candidate_original_not_found")


def validate_decisions(rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) != 20 or len({r["knee_alias"] for r in rows}) != 20:
        raise ValueError("historical_denominator_or_aliases_changed")
    for row in rows:
        if row["candidate_status"] == "abstain":
            if any(row[key] for key in REVIEW_FIELDS[5:]):
                raise ValueError("abstention_must_not_be_reclassified")
            continue
        if row["candidate_status"] != "candidate" or row["decision"] not in DECISIONS:
            raise ValueError(f"pending_or_invalid_decision:{row['knee_alias']}")
        if any(row[key] not in {"", "SI", "NO"} for key in FLAGS):
            raise ValueError("invalid_review_flag")
        if row["outcome_blinded_yes_no"] != "SI":
            raise ValueError("outcome_blinding_not_confirmed")
        if not row["reviewer_notes"].strip():
            raise ValueError("review_reason_required")
        if row["decision"] in ACCEPTED:
            expected = {key: "SI" for key in FLAGS[:4]}
            expected["critical_contamination_yes_no"] = "NO"
            expected["peripheral_warning_yes_no"] = (
                "SI" if row["decision"] == "aceptable_con_advertencia_periferica" else "NO")
            if any(row[key] != value for key, value in expected.items()):
                raise ValueError("acceptable_decision_has_failed_or_unknown_criterion")
        elif row["decision"] == "no_evaluable":
            # A visible image may still leave a criterion unresolved. Do not
            # invent a display failure to encode doubt about a mark's nature.
            if row["visualizable_yes_no"] != "NO" and not any(
                    row[key] == "" for key in FLAGS[:6]):
                raise ValueError("no_evaluable_requires_failed_visualization_or_unknown_criterion")
            if any(row[key] == "NO" for key in FLAGS[:3]) or (
                    row["critical_contamination_yes_no"] == "SI"):
                raise ValueError("known_failed_criterion_requires_rejection")
        elif not (any(row[key] == "NO" for key in FLAGS[:4]) or
                  row["critical_contamination_yes_no"] == "SI"):
            raise ValueError("rejection_requires_documented_failed_criterion")
    counts = Counter(r["decision"] for r in rows if r["candidate_status"] == "candidate")
    candidates = sum(r["candidate_status"] == "candidate" for r in rows)
    accepted = sum(counts[d] for d in ACCEPTED)
    warnings = counts["aceptable_con_advertencia_periferica"]
    failures = candidates-accepted
    checks = {"acceptable_fraction_at_least_95pct": accepted >= 19,
              "no_incorrect_candidate": failures == 0,
              "peripheral_warning_fraction_at_most_10pct": warnings <= 2}
    return {"expected_unique_studies": 10, "expected_knees": 20, "reviewed_candidates": candidates,
            "candidate_knees": candidates, "abstained_knees": 20-candidates,
            "acceptable_crops": accepted, "acceptable_without_warning": counts["aceptable"],
            "acceptable_with_peripheral_warning": warnings, "rejected_crops": counts["rechazado"],
            "non_evaluable_crops": counts["no_evaluable"], "incorrect_candidates": failures,
            "acceptable_fraction": accepted/20, "warning_fraction": warnings/20,
            "decision_counts": dict(sorted(counts.items())), "gate_checks": checks,
            "historical_gate_passed": all(checks.values()),
            "both_knees_acceptable_studies": sum(
                all(r["decision"] in ACCEPTED for r in rows if r["case_alias"] == case)
                for case in {r["case_alias"] for r in rows})}


def _verify_saved_view(config, context, row, view):
    if not isinstance(view, dict) or view.get("knee_alias") != row["knee_alias"] or (
            view.get("review_csv_sha256") != context["review_csv_sha256"]):
        raise ValueError("view_identity_mismatch")
    result = next(r for r in context["results"] if r["knee_alias"] == row["knee_alias"])
    if view.get("native_crop_sha256") != result["native_crop_sha256"]:
        raise ValueError("view_native_crop_mismatch")
    file = _resolve_under_root(config["output_dir"], view["display_file"], "display_file")
    if (view.get("visualization_only") is not True or
            not file.relative_to(config["output_dir"].resolve()).as_posix().startswith("revision_visualizacion/") or
            _read_json(file.with_suffix(".json")) != view or
            sha256_file(file) != view.get("display_sha256")):
        raise ValueError("view_integrity_mismatch")


def read_completed_assistance(config: dict[str, Any]) -> dict[str, Any]:
    """Load prepared proposals, not human attestations or a completed closure.

    The original 17 proposals and original draft remain unchanged. Only the
    formerly pending rows are completed, with the exact saved display evidence.
    """
    context = read_context(config)
    output = config["output_dir"]
    provenance = _read_json(output/COMPLETED_PROVENANCE)
    if (provenance.get("schema_version") != 1 or
            provenance.get("status") != "completed_assistance_pending_investigator_confirmation" or
            provenance.get("reviewer") != "Codex" or
            provenance.get("algorithm_version") != ALGORITHM_VERSION or
            provenance.get("pilot_git_commit") != context["pilot_git_commit"] or
            provenance.get("original_assisted_provenance_sha256") != sha256_file(
                output/"procedencia_revision_asistida.json") or
            provenance.get("draft_csv_sha256") != context["review_csv_sha256"]):
        raise ValueError("completed_assistance_provenance_mismatch")
    _require_false(provenance, ("outcome_data_consulted", "independent_second_reader",
                               "investigator_confirmation", "closure_executed")+BLOCKED)
    if provenance.get("version_known_to_assistant") is not True:
        raise ValueError("completed_assistance_version_disclosure_missing")
    if sha256_file(output/COMPLETED_CSV) != provenance.get("completed_csv_sha256"):
        raise ValueError("completed_assistance_csv_hash_mismatch")
    rows = _rows(output/COMPLETED_CSV, REVIEW_FIELDS)
    if [r["knee_alias"] for r in rows] != [r["knee_alias"] for r in context["review"]]:
        raise ValueError("completed_assistance_identity_or_order_changed")
    views = provenance.get("window_evidence")
    pending = set(context["pending_knee_aliases"])
    if not isinstance(views, dict) or set(views) != pending:
        raise ValueError("completed_assistance_window_evidence_incomplete")
    updates = {}
    for original, row in zip(context["review"], rows):
        if any(row[key] != original[key] for key in IMMUTABLE):
            raise ValueError("completed_assistance_identity_changed")
        alias = row["knee_alias"]
        if alias not in pending:
            if row != original:
                raise ValueError("previous_assisted_proposal_changed")
            continue
        if not row["reviewer_notes"].startswith(original["reviewer_notes"]+" | "):
            raise ValueError("prior_pending_observation_not_preserved")
        _verify_saved_view(config, context, row, views[alias])
        updates[alias] = {key: row[key] for key in ("decision", *FLAGS)}
        updates[alias].update(
            reviewer_notes=row["reviewer_notes"][len(original["reviewer_notes"])+3:],
            window_reviewed=True, view=views[alias])
    validate_decisions(rows)
    fingerprint = {"csv_file": COMPLETED_CSV,
                   "csv_sha256": sha256_file(output/COMPLETED_CSV),
                   "provenance_file": COMPLETED_PROVENANCE,
                   "provenance_sha256": sha256_file(output/COMPLETED_PROVENANCE)}
    return {"review": rows, "draft_csv_sha256": context["review_csv_sha256"],
            "updates": updates, "assisted_completion": fingerprint,
            "preparation_status": provenance["status"],
            "review_mode": "codex_prepared_pending_investigator_confirmation"}


def close_review(config: dict[str, Any], response: dict[str, Any], git_commit: str):
    context = read_context(config)
    if (response.get("confirm_entire_assisted_review") is not True or
            response.get("outcome_blinded") is not True or
            response.get("technical_tests_passed") is not True):
        raise ValueError("investigator_confirmation_required")
    if response.get("draft_csv_sha256") != context["review_csv_sha256"]:
        raise ValueError("stale_investigator_response")
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("full_closure_git_commit_required")
    updates = response.get("updates")
    if not isinstance(updates, dict) or not set(updates) <= {r["knee_alias"] for r in context["review"]}:
        raise ValueError("unexpected_review_updates")
    completed = None
    if "assisted_completion" in response:
        completed = read_completed_assistance(config)
        if (response["assisted_completion"] != completed["assisted_completion"] or
                updates != completed["updates"]):
            raise ValueError("prepared_assistance_changed_before_confirmation")
    output = config["output_dir"]
    rows = [dict(r) for r in context["review"]]
    for row in rows:
        update = updates.get(row["knee_alias"])
        if update is None:
            if row["knee_alias"] in context["pending_knee_aliases"]:
                raise ValueError("unresolved_pending_knee")
            continue
        if set(update) != {"decision", *FLAGS, "reviewer_notes", "window_reviewed", "view"}:
            raise ValueError("unexpected_update_fields")
        if update["window_reviewed"] is not True or not update["reviewer_notes"].strip():
            raise ValueError("window_inspection_and_reason_required")
        _verify_saved_view(config, context, row, update["view"])
        for field in ("decision", *FLAGS):
            row[field] = update[field]
        prefix = (" | " if completed else " | Confirmacion/correccion del investigador en Colab: ")
        row["reviewer_notes"] += prefix+update["reviewer_notes"]
    if completed and rows != completed["review"]:
        raise ValueError("prepared_assistance_roundtrip_mismatch")
    metrics = validate_decisions(rows)
    checked = verify_native_integrity(config, context["results"])
    if checked != metrics["candidate_knees"]:
        raise ValueError("incomplete_candidate_integrity_check")
    status = ("historical_regression_passed_pending_confirmation_sample" if metrics["historical_gate_passed"]
              else "rejected_after_technical_review")
    final_bytes = _csv_bytes(rows)
    summary = {
        "phase": 1, "step": 4, "operation": "close_mcr004_historical_technical_review",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
        "algorithm_version": ALGORITHM_VERSION, "upstream_commit": UPSTREAM_COMMIT,
        "pilot_git_commit": context["pilot_git_commit"], "closure_git_commit": git_commit,
        **metrics, "visual_review_status": "complete", "step_4_status": "open",
        "native_candidate_integrity_checked": checked, "parameters_frozen": False,
        "synthetic_test_status": "passed_before_review",
        "synthetic_test_scope": "test_roi_mcr004*.py",
        "closure_environment": {"python": sys.version.split()[0],
            **{name: version(name) for name in ("numpy", "scipy", "opencv-python", "pydicom", "Pillow")}},
        "outcome_data_loaded": False, **{flag: False for flag in BLOCKED},
        "review_csv_sha256": sha256_bytes(final_bytes),
        "assisted_draft_csv_sha256": context["review_csv_sha256"],
        "review_entry_mode": "codex_prepared" if completed else "investigator_form",
        "results_csv_sha256": context["provenance"]["results_csv_sha256"],
        "closure_execution_environment": "Google Colab, libreta 14",
        "review_provenance": "Revision tecnica asistida por Codex y confirmada por el investigador. "
            "Version conocida; sin desenlaces. No es validacion clinica ni segundo lector independiente.",
    }
    backup = output/"revision_tecnica_ciega_borrador_asistido.csv"
    record = {**summary, "summary_sha256": sha256_bytes(_json_bytes(summary)),
              "assisted_provenance_sha256": sha256_file(output/"procedencia_revision_asistida.json"),
              "response": response}
    if completed:
        record["completed_assistance"] = completed["assisted_completion"]
    # Validate everything before writes. Preserve draft and input evidence on any failure.
    if backup.exists() or (output/"respuestas_investigador.json").exists():
        raise FileExistsError("prior_closure_attempt_present_do_not_overwrite")
    with backup.open("xb") as stream:
        stream.write((output/"revision_tecnica_ciega.csv").read_bytes())
    with (output/"respuestas_investigador.json").open("xb") as stream:
        stream.write(_json_bytes(response))
    with (output/SUMMARY_FILE).open("xb") as stream:
        stream.write(_json_bytes(summary))
    with (output/RECORD_FILE).open("xb") as stream:
        stream.write(_json_bytes(record))
    temporary = output/"revision_tecnica_ciega.final.tmp"
    with temporary.open("xb") as stream:
        stream.write(final_bytes)
    os.replace(temporary, output/"revision_tecnica_ciega.csv")
    if sha256_file(output/"revision_tecnica_ciega.csv") != summary["review_csv_sha256"]:
        raise OSError("final_review_roundtrip_mismatch")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--action", required=True, choices=("context", "assisted-context", "view", "close"))
    parser.add_argument("--alias")
    parser.add_argument("--low", type=float)
    parser.add_argument("--high", type=float)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    if os.environ.get("KNEE_REVIEW_EXECUTION_ENVIRONMENT") != "Google Colab, libreta 14":
        raise RuntimeError("This review CLI runs only in investigator Google Colab.")
    config = load_config(args.config)
    if args.action == "context":
        context = read_context(config)
        payload = {key: context[key] for key in ("review", "pending_knee_aliases",
                   "review_csv_sha256", "pilot_git_commit")}
    elif args.action == "assisted-context":
        payload = read_completed_assistance(config)
    elif args.action == "view":
        payload = make_view(config, args.alias, args.low, args.high)
    else:
        if args.response is None or args.git_commit is None:
            raise ValueError("response_and_git_commit_required")
        payload = close_review(config, _read_json(args.response), args.git_commit)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
