"""Single approved vertical-anchor hypothesis over frozen contour evidence.

The four articular points approximate two compartment centers. This is not
an anatomical ground truth or a joint-space measurement. Every original
landmark and surface trace still determines height; manual QC references are
never inputs. Native pixels, horizontal geometry and contour paths are unused
or unchanged. Predicted surfaces are contained, not promised central 25-75%.
"""
from __future__ import annotations

import math
import numpy as np

from knee.roi_landmark_geometry import landmark_box

VERSION = "kneel_guided_gradient_contours_contact_anchor_v1"
RULE = {
    "version": VERSION,
    "zero_based_landmark_pairs": [[1, 10], [7, 14]],
    "center_y_mm": "row_spacing_mm * (y[1]+y[10]+y[7]+y[14]) / 4",
    "height_mm": "max(width_mm/2.4, 2*full_surface_y_span_mm+2*row_spacing_mm)",
    "horizontal_geometry": "unchanged frozen contour bounds",
    "rounding": "floor_min_ceil_max_half_open",
    "out_of_bounds": "abstain_without_clipping_shifting_padding_or_fallback",
    "all_predicted_surfaces_contained": True,
    "all_predicted_surfaces_central_25_to_75pct": False,
    "manual_center_or_quality_inputs": False,
    "clinical_coverage_certified": False,
}


def contact_box(landmarks, source_shape, dr, dc, frozen_geometry, frozen_traces):
    """Return one native box, preserving the verified original width/height rule."""
    auxiliary = landmark_box(landmarks, source_shape, dr, dc)
    if auxiliary["status"] != "candidate":
        raise ValueError(auxiliary["error_code"])
    points = np.asarray(landmarks, dtype=float)
    if not isinstance(frozen_geometry, dict) or frozen_geometry.get("status") != "candidate":
        raise ValueError("frozen_contour_candidate_required")
    try:
        requested = np.asarray(frozen_geometry["requested_native_half_box"], dtype=float)
        paths = frozen_traces["traces_native_half_xy"]
        envelope = np.asarray(frozen_traces["surface_envelope_y_mm"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid_frozen_contour_evidence") from exc
    if (requested.shape != (4,) or not np.isfinite(requested).all()
            or requested[1] <= requested[0] or requested[3] <= requested[2]
            or envelope.shape != (2,) or not np.isfinite(envelope).all()
            or envelope[1] <= envelope[0] or not isinstance(paths, dict)):
        raise ValueError("invalid_frozen_contour_evidence")
    surface_y = [points[:, 1] * dr]
    for name in ("tibia", "femur"):
        try:
            path = np.asarray(paths[name], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid_frozen_surface_path") from exc
        if (path.ndim != 2 or path.shape[1] != 2 or len(path) < 2
                or not np.isfinite(path).all() or np.any(path < 0)
                or np.any(path[:, 0] >= source_shape[1])
                or np.any(path[:, 1] >= source_shape[0])):
            raise ValueError("invalid_frozen_surface_path")
        surface_y.append(path[:, 1] * dr)
    all_y = np.concatenate(surface_y)
    low, high = float(all_y.min()), float(all_y.max())
    if not np.allclose(envelope, [low, high], rtol=0, atol=1e-9):
        raise ValueError("frozen_surface_envelope_mismatch")
    width_mm = float((requested[1] - requested[0]) * dc)
    height_mm = max(width_mm / 2.4, 2 * (high - low) + 2 * dr)
    if (not math.isclose((requested[3] - requested[2]) * dr, height_mm,
                         rel_tol=0, abs_tol=1e-9)
            or not math.isclose((requested[2] + requested[3]) * dr / 2,
                                (low + high) / 2, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("frozen_width_height_rule_mismatch")
    # Both compartment pairs have equal weight. Horizontal mirroring does not
    # change their semantic IDs or this vertical average.
    center_mm = float(points[[1, 10, 7, 14], 1].mean() * dr)
    new_requested = [float(requested[0]), float(requested[1]),
                     (center_mm - height_mm / 2) / dr,
                     (center_mm + height_mm / 2) / dr]
    x0, x1, y0, y1 = (math.floor(new_requested[0]), math.ceil(new_requested[1]),
                       math.floor(new_requested[2]), math.ceil(new_requested[3]))
    inside = 0 <= x0 < x1 <= source_shape[1] and 0 <= y0 < y1 <= source_shape[0]
    return {
        "algorithm_version": VERSION,
        "status": "candidate" if inside else "abstain",
        "error_code": "" if inside else "contact_box_outside_frozen_half",
        "requested_native_half_box": new_requested,
        "native_half_box": [x0, x1, y0, y1],
        "surface_vertical_fraction": [(low / dr - y0) / (y1 - y0),
                                       (high / dr - y0) / (y1 - y0)],
        "crop_width_mm": (x1 - x0) * dc,
        "crop_height_mm": (y1 - y0) * dr,
        "requested_height_mm": height_mm,
        "vertical_anchor_y_mm": center_mm,
        "vertical_anchor_native_y": center_mm / dr,
        "prior_envelope_midpoint_y_mm": (low + high) / 2,
        "surface_envelope_y_mm": [low, high],
        "coverage_certified": False,
        "artifact_absence_certified": False,
    }
