"""Read pinned development-only DICOM intensity headers without decoding pixels.

The returned rows contain private acquisition links and a strict header allowlist.
No patient tags, outcomes, model, or native pixel arrays are requested. Archive
and DICOM bytes are streamed for identity hashes; original files are never edited.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import tarfile

import pydicom

from knee.roi_global_retention_sources import require, safe, sha_file, valid_sha

VERSION = "development_dicom_intensity_headers_v1.0"
HEADER_TAGS = (
    "Rows", "Columns", "PhotometricInterpretation", "SamplesPerPixel", "NumberOfFrames",
    "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation", "RescaleSlope",
    "RescaleIntercept", "RescaleType", "ModalityLUTSequence", "VOILUTSequence",
    "WindowCenter", "WindowWidth", "VOILUTFunction", "PresentationLUTShape",
    "PresentationLUTSequence", "PixelIntensityRelationship", "PixelIntensityRelationshipSign",
    "PixelPaddingValue", "PixelPaddingRangeLimit",
)
INTEGER_TAGS = set((
    "Rows Columns SamplesPerPixel NumberOfFrames BitsAllocated BitsStored HighBit "
    "PixelRepresentation PixelIntensityRelationshipSign PixelPaddingValue PixelPaddingRangeLimit"
).split())
NUMBER_TAGS = {"RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth"}
HISTORICAL_BLOCKS = {
    "mcr009": "mcr009_contact_confirmation",
    "mcr012": "mcr012_operational_batch",
}


def _header(dataset):
    """Project allowed tags only; sequence bodies are not returned."""
    require("PixelData" not in dataset, "pixel_data_must_not_be_read")
    result = {}
    for tag in HEADER_TAGS:
        value = getattr(dataset, tag, None)
        if value is None:
            result[tag] = None
        elif tag.endswith("Sequence"):
            result[tag] = {"items": len(value)}
        elif tag in INTEGER_TAGS:
            result[tag] = int(value)
        elif tag in NUMBER_TAGS:
            if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
                result[tag] = [float(v) for v in value]
            else:
                result[tag] = float(value)
        else:
            result[tag] = str(value)
    return result


def compatibility_reasons(header):
    """Compatibility with native-value percentile scaling and one photo inversion.

    A positive affine modality transform cancels in percentile normalization, so
    no such transform is applied twice. VOI windows/LUTs are deliberately replaced
    by the separately specified fixed percentile recipe. Nonlinear modality LUTs,
    negative/zero rescale slopes, and presentation inversion require resolution.
    DICOM padding values remain native pixels; this audit does not create masks.
    """
    reasons = []
    if header.get("PhotometricInterpretation") not in {"MONOCHROME1", "MONOCHROME2"}:
        reasons.append("unsupported_photometric_interpretation")
    if (header.get("SamplesPerPixel") != 1 or header.get("NumberOfFrames") not in {None, 1}
            or not all(type(header.get(k)) is int and header[k] > 0 for k in ("Rows", "Columns"))):
        reasons.append("single_plane_grayscale_required")
    stored = header.get("BitsStored")
    if (header.get("BitsAllocated") != 16 or header.get("PixelRepresentation") != 0
            or type(stored) is not int or not 1 <= stored <= 16
            or header.get("HighBit") != stored - 1):
        reasons.append("native_unsigned_16_bit_storage_required")
    slope, intercept = header.get("RescaleSlope"), header.get("RescaleIntercept")
    if (slope is None) != (intercept is None):
        reasons.append("partial_modality_rescale_metadata")
    elif slope is not None:
        if (type(slope) not in {float, int} or type(intercept) not in {float, int}
                or not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0):
            reasons.append("finite_positive_affine_rescale_required")
    if header.get("ModalityLUTSequence") is not None:
        reasons.append("modality_lut_requires_separate_resolution")
    if (header.get("PresentationLUTSequence") is not None
            or header.get("PresentationLUTShape") not in {None, "IDENTITY"}):
        reasons.append("presentation_lut_requires_separate_resolution")
    return reasons


def _archive_header(path, expected_dicom_sha):
    require(valid_sha(expected_dicom_sha), "sealed_dicom_hash_required")
    found = []
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        require(len(members) <= 512, "bounded_dicom_archive_member_count_required")
        for member in members:
            if not member.isfile():
                continue
            require(0 < member.size <= 2 * 1024**3, "bounded_dicom_member_size_required")
            with archive.extractfile(member) as stream:
                dataset = pydicom.dcmread(stream, force=True, stop_before_pixels=True,
                    specific_tags=list(HEADER_TAGS))
            if "Rows" not in dataset or "Columns" not in dataset:
                continue
            header = _header(dataset)
            with archive.extractfile(member) as stream:
                dicom_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            require(dicom_sha == expected_dicom_sha, "sealed_development_dicom_hash_mismatch")
            found.append((member.name, dicom_sha, header))
    require(len(found) == 1, "exactly_one_development_dicom_image_required")
    return found[0]


def audit_development_headers(rows, native_roots, inventory_rows):
    """Audit only sources referenced by already accepted development membership.

    ``inventory_rows`` is the existing metadata-only acquisition inventory. Roots
    require project_root for historical raw packages and development_root for the
    additive development completion. Other roots are never enumerated or opened.
    """
    require(isinstance(rows, list) and rows, "nonempty_development_membership_required")
    require(all(isinstance(r, dict) and r.get("role") == "development" for r in rows),
        "only_development_membership_authorized")
    require(len({(r.get("src_subject_id"), r.get("side")) for r in rows}) == len(rows),
        "unique_development_knees_required")
    require(isinstance(inventory_rows, list), "metadata_only_inventory_rows_required")
    inventory = {r["src_subject_id"]: r for r in inventory_rows}
    require(len(inventory) == len(inventory_rows), "unique_inventory_participants_required")
    grouped = {}
    for row in rows:
        require(row.get("side") in {"LEFT", "RIGHT"}, "development_laterality_required")
        grouped.setdefault(row["src_subject_id"], []).append(row)
    result, pins = [], {}
    for subject, knees in sorted(grouped.items()):
        require(subject in inventory, "development_source_missing_from_inventory")
        study = inventory[subject]
        blocks = {r.get("block") for r in knees}
        require(len(blocks) == 1, "one_source_block_per_development_participant_required")
        block = next(iter(blocks))
        if block == "development_completion_001":
            root_key = "development_root"
            relative = "sources/raw/" + study["canonical_source"] + "/" + study["canonical_package_relative_path"]
        else:
            require(block in HISTORICAL_BLOCKS, "approved_development_block_required")
            root_key = "project_root"
            relative = "outputs/" + HISTORICAL_BLOCKS[block] + "/raw/" + study["canonical_package_relative_path"]
        require(root_key in native_roots, "development_original_root_required")
        path = safe(native_roots[root_key], relative)
        require(path.is_file(), "development_original_archive_required")
        require(valid_sha(study.get("package_sha256")), "sealed_development_archive_hash_required")
        package_sha = sha_file(path)
        require(package_sha == study["package_sha256"], "sealed_development_archive_hash_mismatch")
        member, dicom_sha, header = _archive_header(path, study["dicom_sha256"])
        for knee in knees:
            require(all(knee.get(k) == study.get(k) for k in
                ("src_subject_id", "acquisition_key", "image_file", "barcode_base")),
                "development_dicom_membership_identity_changed")
            require(knee.get("photometric") == header["PhotometricInterpretation"],
                "development_dicom_photometric_changed")
            require(knee.get("source_dtype") == "uint16"
                and knee.get("source_shape", [None])[0] == header["Rows"]
                and type(knee.get("half_x0")) is int and type(knee.get("half_x1")) is int
                and 0 <= knee["half_x0"] < knee["half_x1"] <= header["Columns"],
                "development_dicom_native_geometry_changed")
        require(sha_file(path) == package_sha, "development_archive_changed_during_header_audit")
        pins[str(path)] = package_sha
        result.append({"src_subject_id": subject, "block": block,
            "knee_aliases": sorted(k["knee_alias"] for k in knees),
            "source": {"root_key": root_key, "relative_path": relative,
                "package_sha256": package_sha, "dicom_member": member, "dicom_sha256": dicom_sha},
            "header": header, "compatibility_reasons": compatibility_reasons(header)})
    counts = {tag: dict(sorted(Counter(
        "ABSENT" if r["header"][tag] is None else str(r["header"][tag]) for r in result).items()))
        for tag in HEADER_TAGS}
    incompatible = [r for r in result if r["compatibility_reasons"]]
    summary = {"development_knees": len(rows), "development_participants": len(grouped),
        "original_dicom_headers_audited": len(result), "package_hashes_verified": len(pins),
        "dicom_hashes_verified": len(result), "metadata_counts": counts,
        "compatible_with_fixed_quantile_native_recipe": not incompatible,
        "incompatible_participants": len(incompatible),
        "incompatibility_reason_counts": dict(sorted(Counter(
            reason for r in result for reason in r["compatibility_reasons"]).items())),
        "dicom_window_and_voi_applied": False, "native_modality_rescale_applied": False,
        "padding_pixels_masked_or_repaired": False, "pixel_data_decoded": False,
        "native_arrays_loaded": False, "outcomes_opened": False,
        "patient_tags_requested": False, "originals_modified": False}
    return {"version": VERSION, "rows": result, "pins": dict(sorted(pins.items())), "summary": summary}
