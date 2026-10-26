"""Read pinned accepted-routine DICOM headers, without decoding original pixels.

The same allowlist and intensity compatibility rule used for development apply.
Only deterministic existing caches are considered. No private transport, image
inference, clinical variables, outcomes, exclusions, or source edits occur here.
"""
from __future__ import annotations

from collections import Counter
import re

from knee.development_dicom_headers import HEADER_TAGS, _archive_header, compatibility_reasons
from knee.roi_global_retention_sources import require, safe, sha_file, valid_sha

VERSION = "routine_dicom_intensity_headers_v1.0"
PREFIX_FOLDERS = {
    1: "roi_first_batch_20261004",
    2: "roi_second_batch_20261004",
    3: "roi_third_batch_20261004",
    4: "roi_fourth_batch_20261004",
    5: "roi_fifth_batch_20261004",
}


def _archive_candidates(study, block, native_roots):
    match = re.fullmatch(r"routine_batch_([0-9]{3})", block or "")
    require(match is not None and 1 <= int(match[1]) <= 92,
        "sealed_routine_block_required")
    require(study.get("canonical_source") == "cohorte_v00",
        "routine_canonical_cohort_source_required")
    relative = "dicom/originales/cohorte_v00/" + study["canonical_package_relative_path"]
    candidates = []
    if "private_sources_root" in native_roots:
        candidates.append(("private_sources_root", relative,
            safe(native_roots["private_sources_root"], relative)))
    number = int(match[1])
    if number in PREFIX_FOLDERS:
        prefix = "outputs/" + PREFIX_FOLDERS[number] + "/private_sources/" + relative
        for root_key in ("project_root", "runtime_root"):
            if root_key in native_roots:
                candidates.append((root_key, prefix, safe(native_roots[root_key], prefix)))
    require(bool(candidates), "routine_original_cache_root_required")
    return candidates


def audit_routine_headers(rows, native_roots, inventory_rows, *, progress=None):
    """Verify original package/DICOM pins and allowed headers for accepted routine rows.

    ``private_sources_root`` names the approved private source cache. For the
    first five groups, ``project_root``/``runtime_root`` may contain their existing
    per-group private caches. A complete availability preflight runs before any
    archive read. Missing archives are a stop, never an exclusion or a guessed
    header. The optional callback receives completed and total participants.
    """
    require(isinstance(rows, list) and rows, "nonempty_routine_membership_required")
    require(all(isinstance(r, dict) and r.get("role") == "routine" for r in rows),
        "only_routine_membership_authorized")
    require(len({(r.get("src_subject_id"), r.get("side")) for r in rows}) == len(rows),
        "unique_routine_knees_required")
    require(isinstance(inventory_rows, list), "metadata_only_inventory_rows_required")
    inventory = {r["src_subject_id"]: r for r in inventory_rows}
    require(len(inventory) == len(inventory_rows), "unique_inventory_participants_required")
    require(progress is None or callable(progress), "callable_header_progress_required")
    grouped = {}
    for row in rows:
        require(isinstance(row.get("src_subject_id"), str) and row["src_subject_id"],
            "routine_participant_identity_required")
        require(row.get("side") in {"LEFT", "RIGHT"}, "routine_laterality_required")
        grouped.setdefault(row["src_subject_id"], []).append(row)
    prepared, missing = [], []
    for subject, knees in sorted(grouped.items()):
        require(subject in inventory, "routine_source_missing_from_inventory")
        study = inventory[subject]
        blocks = {r.get("block") for r in knees}
        require(len(blocks) == 1, "one_source_block_per_routine_participant_required")
        block = next(iter(blocks))
        require(valid_sha(study.get("package_sha256")), "sealed_routine_archive_hash_required")
        require(valid_sha(study.get("dicom_sha256")), "sealed_routine_dicom_hash_required")
        for knee in knees:
            require(all(knee.get(k) == study.get(k) for k in
                ("src_subject_id", "acquisition_key", "image_file", "barcode_base")),
                "routine_dicom_membership_identity_changed")
        candidates = _archive_candidates(study, block, native_roots)
        available = [c for c in candidates if c[2].is_file()]
        if not available:
            missing.append((block, len(knees)))
            continue
        root_key, relative, path = available[0]
        prepared.append((subject, knees, study, block, root_key, relative, path))
    if missing:
        blocks = ",".join(f"{b}:{n}" for b, n in sorted(Counter(b for b, _ in missing).items()))
        raise ValueError("routine_original_archives_unavailable:participants=" + str(len(missing))
            + ";knees=" + str(sum(n for _, n in missing)) + ";blocks=" + blocks)
    result, pins = [], {}
    for subject, knees, study, block, root_key, relative, path in prepared:
        package_sha = sha_file(path)
        require(package_sha == study["package_sha256"], "sealed_routine_archive_hash_mismatch")
        member, dicom_sha, header = _archive_header(path, study["dicom_sha256"])
        for knee in knees:
            require(knee.get("photometric") == header["PhotometricInterpretation"],
                "routine_dicom_photometric_changed")
            require(knee.get("source_dtype") == "uint16"
                and knee.get("source_shape", [None])[0] == header["Rows"]
                and type(knee.get("half_x0")) is int and type(knee.get("half_x1")) is int
                and 0 <= knee["half_x0"] < knee["half_x1"] <= header["Columns"],
                "routine_dicom_native_geometry_changed")
        require(sha_file(path) == package_sha, "routine_archive_changed_during_header_audit")
        pins[str(path)] = package_sha
        result.append({"src_subject_id": subject, "block": block,
            "knee_aliases": sorted(k["knee_alias"] for k in knees),
            "source": {"root_key": root_key, "relative_path": relative,
                "package_sha256": package_sha, "dicom_member": member, "dicom_sha256": dicom_sha},
            "header": header, "compatibility_reasons": compatibility_reasons(header)})
        if progress is not None and (len(result) % 25 == 0 or len(result) == len(prepared)):
            progress(len(result), len(prepared))
    counts = {tag: dict(sorted(Counter(
        "ABSENT" if r["header"][tag] is None else str(r["header"][tag]) for r in result).items()))
        for tag in HEADER_TAGS}
    incompatible = [r for r in result if r["compatibility_reasons"]]
    summary = {"routine_knees": len(rows), "routine_participants": len(grouped),
        "original_dicom_headers_audited": len(result), "package_hashes_verified": len(pins),
        "dicom_hashes_verified": len(result), "metadata_counts": counts,
        "source_cache_counts": dict(sorted(Counter(r["source"]["root_key"] for r in result).items())),
        "compatible_with_fixed_quantile_native_recipe": not incompatible,
        "incompatible_participants": len(incompatible),
        "incompatibility_reason_counts": dict(sorted(Counter(
            reason for r in result for reason in r["compatibility_reasons"]).items())),
        "dicom_window_and_voi_applied": False, "native_modality_rescale_applied": False,
        "padding_pixels_masked_or_repaired": False, "pixel_data_decoded": False,
        "native_arrays_loaded": False, "outcomes_opened": False,
        "patient_tags_requested": False, "originals_modified": False}
    return {"version": VERSION, "rows": result, "pins": dict(sorted(pins.items())), "summary": summary}
