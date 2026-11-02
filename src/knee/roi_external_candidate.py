"""Import frozen localizer coordinates; never import its resampled pixels.

Exploratory use on the historical pilot only. A valid box is not a quality
decision. Out-of-bounds predictions abstain without clipping or padding.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def load_predictions(path: Path, expected_aliases):
    content = path.read_bytes()
    data = json.loads(content)
    if not isinstance(data.get("manifest"), dict) or not data["manifest"]:
        raise ValueError("external_manifest_required")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError("external_rows_required")
    aliases = [r.get("knee_alias") for r in rows]
    if (len(aliases) != len(expected_aliases) or len(set(aliases)) != len(aliases)
            or set(aliases) != set(expected_aliases)):
        raise ValueError("external_alias_set_mismatch")
    for row in rows:
        if row.get("status") not in ("candidate", "abstain"):
            raise ValueError("external_unknown_status")
        if not isinstance(row.get("source_half_sha256"), str):
            raise ValueError("external_source_digest_required")
    return {r["knee_alias"]: r for r in rows}, data["manifest"], hashlib.sha256(content).hexdigest()


def candidate_from_prediction(half, prediction, offset, dr, dc):
    native_hash = hashlib.sha256(np.ascontiguousarray(half).tobytes()).hexdigest()
    if prediction["source_half_sha256"] != native_hash:
        raise ValueError("external_source_half_hash_mismatch")
    common = {"algorithm_version": "external_frozen_exploratory_v1",
              "external_metadata": prediction.get("metadata", {})}
    if prediction["status"] == "abstain":
        return dict(common, status="abstain", error_code=prediction.get(
            "error_code", "external_localizer_abstained"))
    box = prediction.get("native_half_box")
    if (not isinstance(box, list) or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
        raise ValueError("external_box_must_have_four_finite_numbers")
    common["requested_native_half_box"] = box
    x0, x1, y0, y1 = box
    if not (0 <= x0 < x1 <= half.shape[1] and 0 <= y0 < y1 <= half.shape[0]):
        return dict(common, status="abstain", error_code="external_box_outside_frozen_half_or_empty")
    x0, x1, y0, y1 = math.floor(x0), math.ceil(x1), math.floor(y0), math.ceil(y1)
    return dict(common, status="candidate", error_code="", native_half_box=[x0, x1, y0, y1],
                native_dicom_box=[x0+offset, x1+offset, y0, y1],
                crop_rows=y1-y0, crop_columns=x1-x0,
                crop_height_mm=(y1-y0)*dr, crop_width_mm=(x1-x0)*dc,
                rounding="outward_floor_min_ceil_max_half_open",
                crop=half[y0:y1, x0:x1].copy())
