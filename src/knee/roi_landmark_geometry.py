"""Exploratory native ROI geometry from frozen KNEEL surface landmarks.

The physical square includes the landmark envelope plus lateral context and
places every predicted point within the central vertical half. These are
geometric conditions, not a certificate of anatomical coverage or clean pixels.
No clinical labels, image-artifact locations or case-specific rules are inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from knee.roi_external_candidate import load_predictions

VERSION = "kneel_landmark_square_v1.1"
RULE = {
    "version": VERSION,
    "lateral_margin_per_side_fraction_of_landmark_span": 0.10,
    "physical_shape": "square",
    "center": "midpoint_of_all_16_landmarks_min_max_envelope",
    "side_mm": "max(1.2 * span_x_mm, 2 * span_y_mm + 2 * row_spacing_mm)",
    "rounding": "floor_min_ceil_max_half_open",
    "out_of_bounds": "abstain_without_clipping_shifting_or_padding",
    "rationale": (
        "Proportional lateral margin is an engineering choice, not a published "
        "clinical tolerance. The square retains adjacent bone context; twice "
        "the vertical point span places points in the central 25-75 percent. "
        "Two row pixels guard that band against outward integer rounding. "
        "No artifact position or acceptability label determines the box."
    ),
    "limitation": "Predicted surface points do not certify full bone contours or image quality.",
}


def landmark_box(landmarks, shape, row_spacing, column_spacing):
    """Return an experimental box or an explicit geometrical abstention."""
    common = {"algorithm_version": VERSION}

    def abstain(reason, **details):
        return dict(common, status="abstain", error_code=reason, **details)

    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(n) not in (int, np.int64, np.int32)
                              or n <= 0 for n in shape)):
        return abstain("invalid_half_shape")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v <= 0 for v in (row_spacing, column_spacing)):
        return abstain("invalid_pixel_spacing")
    try:
        original = np.asarray(landmarks)
        if original.dtype.kind not in "fiu":
            return abstain("invalid_landmarks")
        points = original.astype(np.float64)
    except (TypeError, ValueError):
        return abstain("invalid_landmarks")
    if points.shape != (16, 2) or not np.isfinite(points).all():
        return abstain("invalid_landmarks")
    if (np.any(points < 0) or np.any(points[:, 0] >= shape[1])
            or np.any(points[:, 1] >= shape[0])):
        return abstain("landmarks_outside_frozen_half")
    low, high = points.min(axis=0), points.max(axis=0)
    span = high-low
    if np.any(span <= 0):
        return abstain("degenerate_landmarks")
    span_mm = span*np.array([column_spacing, row_spacing])
    side_mm = max(1.2*span_mm[0], 2*span_mm[1]+2*row_spacing)
    center = (low+high)/2
    radius = side_mm/np.array([column_spacing, row_spacing])/2
    requested = [center[0]-radius[0], center[0]+radius[0],
                 center[1]-radius[1], center[1]+radius[1]]
    common.update(landmark_envelope_half_xyxy=[*low.tolist(), *high.tolist()],
                  requested_native_half_box=[float(v) for v in requested],
                  requested_side_mm=float(side_mm),
                  center_native_half_xy=center.tolist())
    if not np.isfinite(requested).all():
        return abstain("nonfinite_derived_geometry")
    x0, x1, y0, y1 = math.floor(requested[0]), math.ceil(requested[1]), \
        math.floor(requested[2]), math.ceil(requested[3])
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        return abstain("landmark_box_outside_frozen_half")
    band = [(float(low[1])-y0)/(y1-y0), (float(high[1])-y0)/(y1-y0)]
    if band[0] < 0.25 or band[1] > 0.75:
        return abstain("landmark_band_rounding_check_failed")
    return dict(common, status="candidate", error_code="",
                native_half_box=[x0, x1, y0, y1], landmark_vertical_fraction=band)


def build_predictions(kneel_path: Path, verified_dir: Path, output: Path):
    """Reuse the frozen inference and source hashes; do not rerun or train it."""
    kneel_path, verified_dir, output = kneel_path.resolve(), verified_dir.resolve(), output.resolve()
    for source in (kneel_path.parent, verified_dir):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("output_must_be_separate_from_sources")
    if output.exists():
        raise FileExistsError("new_output_directory_required")
    evidence_bytes = (verified_dir/"results.json").read_bytes()
    evidence = json.loads(evidence_bytes)
    aliases = [row["knee_alias"] for row in evidence["rows"]]
    if (len(aliases) != 20 or len(set(aliases)) != 20
            or evidence["summary"]["exact_baseline_replays"] != 20):
        raise ValueError("requires_verified_historical_20_knees")
    frozen, upstream, frozen_sha = load_predictions(kneel_path, aliases)
    if frozen_sha != evidence["summary"]["external_predictions_sha256"]:
        raise ValueError("frozen_landmark_predictions_hash_mismatch")
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True)
    # Written before evaluating any derived box, identifying the single rule.
    config = dict(RULE, recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_code_sha256=code_sha)
    (output/"rule_before_evaluation.json").write_text(
        json.dumps(config, indent=2)+"\n", encoding="utf-8")
    rows = []
    for reference in evidence["rows"]:
        alias = reference["knee_alias"]
        source = frozen[alias]
        half = np.load(verified_dir/"halves"/(alias+".npy"), allow_pickle=False)
        sha = hashlib.sha256(np.ascontiguousarray(half).tobytes()).hexdigest()
        if sha != source["source_half_sha256"]:
            raise ValueError("landmark_source_half_hash_mismatch")
        points = source.get("metadata", {}).get("landmarks_native_half_xy")
        result = landmark_box(points, half.shape, reference["row_spacing_mm"],
                              reference["column_spacing_mm"])
        row = dict(knee_alias=alias, source_half_sha256=sha,
                   status=result["status"], error_code=result["error_code"],
                   metadata=dict(result, landmarks_native_half_xy=points,
                                 upstream_140mm_status=source["status"],
                                 upstream_140mm_box=source.get("native_half_box")))
        if result["status"] == "candidate":
            row["native_half_box"] = result["native_half_box"]
        rows.append(row)
    if (hashlib.sha256(kneel_path.read_bytes()).hexdigest() != frozen_sha
            or (verified_dir/"results.json").read_bytes() != evidence_bytes
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != code_sha):
        raise RuntimeError("input_or_code_changed_during_geometry")
    manifest = dict(candidate=VERSION, rule=config, upstream_manifest=upstream,
                    frozen_landmark_predictions_sha256=frozen_sha,
                    upstream_verified_results_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
                    geometry_source_sha256=code_sha,
                    experimental_only=True, model_retrained=False, outcomes_loaded=False,
                    acceptability_assigned=False, production_method_promoted=False,
                    historical_review_changed=False, step_4_status="open")
    target = output/"predictions.json"
    target.write_text(json.dumps(dict(manifest=manifest, rows=rows), indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(predictions=str(target), knees=len(rows),
                         boxes_in_bounds=sum(r["status"] == "candidate" for r in rows),
                         acceptability_assigned=False)))
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kneel-predictions", type=Path, required=True)
    parser.add_argument("--verified-kneel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_predictions(args.kneel_predictions, args.verified_kneel, args.output)
