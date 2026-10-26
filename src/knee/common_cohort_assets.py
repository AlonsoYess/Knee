"""Resolve sealed accepted native exports without decoding or loading images.

NPY headers and streaming byte hashes are checked against the original QC
records.  No arrays are materialized and no anatomical decision is recomputed.
All returned identity rows and paths are private research metadata.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from knee.roi_global_retention_sources import require, safe, sha_file, valid_sha

VERSION = "common_cohort_native_assets_v1.0"
ROOT_KEYS = ("project_root", "runtime_root", "campaign_root", "development_root")


def _read(path, expected, pins):
    require(valid_sha(expected), "sealed_asset_metadata_pin_required")
    actual = sha_file(path)
    require(actual == expected, "sealed_asset_metadata_changed")
    key = str(path)
    require(key not in pins or pins[key] == actual, "asset_changed_during_collection")
    pins[key] = actual
    return json.loads(path.read_text(encoding="utf-8-sig"))


def inspect_npy(path, *, expected_file_sha, expected_pixel_sha, shape, dtype,
                verify_payloads=True):
    """Read a bounded NPY header and, optionally, stream both byte hashes.

    QC crop_sha256 hashes the contiguous native pixels, whereas a file pin also
    includes the NPY header.  Both hashes must agree; neither substitutes for the
    other. Native exports are ordinary two-dimensional C-order integer arrays.
    """
    require(valid_sha(expected_file_sha) and valid_sha(expected_pixel_sha),
            "native_export_pins_required")
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        require(prefix[:6] == b"\x93NUMPY" and prefix[6:] in
                (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "native_npy_format_required")
        length_size = 2 if prefix[6] == 1 else 4
        length_bytes = stream.read(length_size)
        require(len(length_bytes) == length_size, "truncated_native_npy_header")
        length = struct.unpack("<H" if length_size == 2 else "<I", length_bytes)[0]
        require(0 < length <= 65536, "bounded_native_npy_header_required")
        header_bytes = stream.read(length)
        require(len(header_bytes) == length, "truncated_native_npy_header")
        try:
            header = ast.literal_eval(header_bytes.decode("utf-8" if prefix[6] == 3 else "latin1"))
        except (ValueError, SyntaxError, UnicodeError) as exc:
            raise ValueError("invalid_native_npy_header") from exc
        require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"}
                and header["fortran_order"] is False, "native_c_order_required")
        dimensions = header["shape"]
        require(isinstance(dimensions, tuple) and len(dimensions) == 2
                and all(type(v) is int and v > 0 for v in dimensions), "native_2d_shape_required")
        descriptor = header["descr"]
        match = re.fullmatch(r"([<>=|])([ui])([1248])", descriptor) if isinstance(descriptor, str) else None
        require(match is not None, "native_integer_dtype_required")
        byteorder, kind, size_text = match.groups()
        size = int(size_text)
        require(byteorder in {"<", "|", "="} and (byteorder != "|" or size == 1),
                "native_little_endian_required")
        dtype_name = ("uint" if kind == "u" else "int") + str(size * 8)
        require(list(dimensions) == list(shape), "native_crop_shape_changed")
        require(dtype_name == dtype, "native_crop_dtype_changed")
        expected_size = dimensions[0] * dimensions[1] * size
        data_start = stream.tell()
        require(Path(path).stat().st_size == data_start + expected_size,
                "native_npy_payload_length_changed")
        actual_sha = expected_file_sha
        if verify_payloads:
            file_hash, pixel_hash = hashlib.sha256(), hashlib.sha256()
            file_hash.update(prefix + length_bytes + header_bytes)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_hash.update(chunk)
                pixel_hash.update(chunk)
            actual_sha = file_hash.hexdigest()
            require(actual_sha == expected_file_sha, "accepted_native_file_hash_mismatch")
            require(pixel_hash.hexdigest() == expected_pixel_sha, "accepted_native_pixel_hash_mismatch")
    return {"sha256": actual_sha, "pixel_sha256": expected_pixel_sha,
            "shape": list(dimensions), "dtype": dtype_name,
            "bytes": data_start + expected_size}


def _indexed(rows):
    require(isinstance(rows, list) and all(isinstance(r, dict) and isinstance(r.get("knee_alias"), str)
            for r in rows), "asset_alias_rows_required")
    result = {r["knee_alias"]: r for r in rows}
    require(len(result) == len(rows), "unique_asset_aliases_required")
    return result


def _export_set(workspace, aliases):
    folder = safe(workspace, "accepted_native")
    require(folder.is_dir(), "accepted_native_folder_required")
    actual = {p.name for p in folder.iterdir()}
    require(actual == {a + ".npy" for a in aliases}, "excluded_export_or_missing_accepted_export")


def _asset(identity, selected, source, decision, root_key, root, relative,
           expected_file_sha, pins, verify_payloads):
    alias, side = decision["knee_alias"], identity["side"]
    require(source.get("knee_alias") == alias and source.get("side") == side
            and alias == source.get("case_alias", alias.rsplit("_", 1)[0]) + "_" + side,
            "native_asset_laterality_changed")
    shape = source.get("source_shape")
    require(isinstance(shape, list) and len(shape) == 2
            and all(type(v) is int and v > 0 for v in shape), "native_source_shape_required")
    half_x0, half_x1 = source.get("half_x0"), source.get("half_x1")
    require(type(half_x0) is int and type(half_x1) is int and 0 <= half_x0 < half_x1
            and half_x1 - half_x0 == shape[1]
            and (half_x0 == 0 if side == "RIGHT" else half_x0 > 0),
            "native_half_geometry_changed")
    box = decision.get("box")
    require(isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box),
            "native_box_required")
    x0, x1, y0, y1 = box
    require(0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0], "native_box_outside_half")
    if "native_dicom_box" in decision:
        require(decision["native_dicom_box"] == [x0 + half_x0, x1 + half_x0, y0, y1],
                "native_bilateral_box_changed")
    require(source.get("photometric") in {"MONOCHROME1", "MONOCHROME2"},
            "native_photometric_required")
    require(valid_sha(source.get("source_half_sha256")), "sealed_native_half_pixel_pin_required")
    for key in ("row_spacing_mm", "column_spacing_mm"):
        v = source.get(key)
        require(type(v) in (int, float) and math.isfinite(v) and v > 0,
                "positive_native_pixel_spacing_required")
    require(isinstance(source.get("spacing_tag"), str) and source["spacing_tag"],
            "native_spacing_provenance_required")
    relative = str(relative).replace("\\", "/")
    path = safe(root, relative)
    crop = inspect_npy(path, expected_file_sha=expected_file_sha,
        expected_pixel_sha=decision.get("crop_sha256"), shape=[y1-y0, x1-x0],
        dtype=source.get("source_dtype"), verify_payloads=verify_payloads)
    pins[str(path)] = crop["sha256"]
    return {k: identity[k] for k in ("src_subject_id", "side", "role", "block")} | {
        **{k: selected[k] for k in ("acquisition_key", "image_file", "barcode_base")},
        "knee_alias": alias, "crop": {"root_key": root_key, "relative_path": relative, **crop},
        **{k: source[k] for k in ("photometric", "row_spacing_mm", "column_spacing_mm", "spacing_tag")},
        "source_shape": shape, "source_dtype": source["source_dtype"],
        "source_half_sha256": source["source_half_sha256"],
        "half_x0": half_x0, "half_x1": half_x1, "box": box,
        "native_dicom_box": [x0 + half_x0, x1 + half_x0, y0, y1],
        "native_spatial_orientation": "original_acquisition_no_mirroring"}


def _pinned_read(root, relative, pins):
    path = safe(root, relative)
    return _read(path, pins.get(str(path)), pins)


def _routine_specs(block, roots, pins):
    if block == "development_completion_001":
        root_key, base = "development_root", "workspace"
    else:
        number = int(block.removeprefix("routine_batch_"))
        root_key = "project_root" if number < 6 else "campaign_root"
        if number == 1:
            base = "outputs/roi_first_batch_qc_20261004/local_verification_checkpoint/workspace"
        elif number == 2:
            base = "outputs/roi_second_batch_review_20261004/qc_stage_local_v1/workspace"
        elif number < 6:
            folder = {3: "third", 4: "fourth", 5: "fifth"}[number]
            base = f"outputs/roi_{folder}_batch_20261004/persistent/{block}/workspace"
        else:
            base = f"groups/{block}/workspace"
    root, workspace = roots[root_key], safe(roots[root_key], base)
    if block == "routine_batch_001":
        record = _pinned_read(root, "outputs/roi_second_batch_20261004/evidence/colab21_publico.json", pins)
        plan = _pinned_read(root, "outputs/roi_second_batch_20261004/evidence/prior_plan.json", pins)
        checkpoint = _pinned_read(root, "outputs/roi_second_batch_20261004/evidence/prior_checkpoint.json", pins)
        # The original Colab wrapper and local assisted response have distinct
        # hashes. The canonical local payload is pinned explicitly by that record.
        qc = _read(safe(workspace, "qc_review_seal.json"), record["assisted_review_response_sha256"], pins)
    else:
        plan = _pinned_read(root, base + "/plan.json", pins)
        candidates = [p for p in pins if Path(p).parent == safe(workspace, "checkpoints")]
        require(len(candidates) == 1, "one_sealed_terminal_asset_checkpoint_required")
        checkpoint = _read(Path(candidates[0]), pins[candidates[0]], pins)
        qc_path = safe(root, base + "/qc_review_seal.json")
        if str(qc_path) not in pins:
            record_folder = workspace.parent / "execution_records"
            records = [p for p in pins if Path(p).parent == record_folder]
            require(len(records) == 1, "one_sealed_asset_execution_record_required")
            record = _read(Path(records[0]), pins[records[0]], pins)
            qc = _read(qc_path, record.get("qc_review_seal_sha256"), pins)
        else:
            qc = _pinned_read(root, base + "/qc_review_seal.json", pins)
    rows = _indexed(checkpoint["contract_snapshot"]["rows"])
    evidence = qc.get("candidate_evidence_files_sha256")
    require(isinstance(evidence, dict) and evidence, "sealed_candidate_evidence_pins_required")
    accepted = {alias for alias, r in rows.items() if r["state"] == "accepted"}
    _export_set(workspace, accepted)
    return root_key, base, workspace, plan["selected"], rows, evidence


def collect_assets(ledger, sources_config, *, verify_payloads=True):
    """Resolve each accepted ledger knee to its sealed, exact native NPY file."""
    require(type(verify_payloads) is bool, "explicit_native_verification_mode_required")
    roots = {k: Path(sources_config[k]).resolve() for k in ROOT_KEYS}
    pins = dict(ledger["pins"])
    inventory = {r["src_subject_id"]: r for r in ledger["inventory"]}
    keys = [(r["src_subject_id"], r["side"]) for r in ledger["rows"]]
    require(len(keys) == len(set(keys)), "unique_asset_ledger_identity_required")
    expected = {k for k, r in zip(keys, ledger["rows"]) if r["state"] == "accepted"}
    by_block = {}
    for row in ledger["rows"]:
        require(row["state"] in {"accepted", "excluded"}, "terminal_asset_dispositions_required")
        by_block.setdefault(row["block"], []).append(row)
    result = []
    historical = {"mcr009": "mcr009_contact_confirmation", "mcr012": "mcr012_operational_batch"}
    for block, ledger_rows in sorted(by_block.items()):
        if block in historical:
            root_key = "runtime_root"
            root, base = roots[root_key], "outputs/" + historical[block]
            selected = _pinned_read(root, base + "/selection/selection.json", pins)["selected"]
            decisions = _pinned_read(root, base + "/application/results.json", pins)
            manifest = _read(safe(root, "bounded_local_capsule_manifest.json"),
                pins.get(str(safe(root, "bounded_local_capsule_manifest.json"))), pins)
            originals = manifest["workspace_files_sha256"]
            task = _read(safe(root, base + "/sources/task.json"), originals[base + "/sources/task.json"], pins)
            prepared = _read(safe(root, base + "/prepared/manifest.json"),
                decisions["manifest_sha256"], pins)
            require(prepared.get("task_sha256") == pins[str(safe(root, base + "/sources/task.json"))],
                    "historical_native_source_chain_changed")
            rows, sources = _indexed(decisions["rows"]), _indexed(task["rows"])
            accepted = {a for a, r in rows.items() if r["accepted"] is True}
            _export_set(safe(root, base + "/application"), accepted)
            exports = decisions["native_files_sha256"]
            for identity in ledger_rows:
                study = next(s for s in selected if s["src_subject_id"] == identity["src_subject_id"])
                alias = study["case_alias"] + "_" + identity["side"]
                decision = rows[alias]
                require(decision["accepted"] is (identity["state"] == "accepted"),
                        "asset_ledger_disposition_changed")
                if identity["state"] == "accepted":
                    result.append(_asset(identity, inventory[identity["src_subject_id"]], sources[alias], decision,
                        root_key, root, base + "/application/accepted_native/" + alias + ".npy",
                        exports["accepted_native/" + alias + ".npy"], pins, verify_payloads))
            continue
        root_key, base, workspace, selected, rows, evidence = _routine_specs(block, roots, pins)
        selected_by_id = {s["src_subject_id"]: s for s in selected}
        source_cache = {}
        for identity in ledger_rows:
            study = selected_by_id[identity["src_subject_id"]]
            case = study["case_alias"]
            alias = case + "_" + identity["side"]
            decision = rows[alias]
            require(decision["state"] == identity["state"]
                    and decision["src_subject_id"] == identity["src_subject_id"]
                    and decision["side"] == identity["side"], "asset_ledger_disposition_changed")
            if identity["state"] != "accepted":
                continue
            if case not in source_cache:
                relative = f"cases/{case}/sources.json"
                source_cache[case] = _indexed(_read(safe(workspace, relative), evidence[relative], pins)["rows"])
            source = source_cache[case][alias]
            require(decision["source"] == {k: source[k] for k in
                    ("knee_alias", "source_shape", "source_dtype", "source_half_sha256")},
                    "native_source_differs_from_terminal_contract")
            result.append(_asset(identity, inventory[identity["src_subject_id"]], source, decision, root_key,
                roots[root_key], base + "/accepted_native/" + alias + ".npy",
                evidence["candidates/" + alias + ".npy"], pins, verify_payloads))
    result.sort(key=lambda r: (r["src_subject_id"], r["side"]))
    require({(r["src_subject_id"], r["side"]) for r in result} == expected
            and len(result) == len(expected), "exact_accepted_native_asset_set_required")
    return {"version": VERSION, "rows": result, "pins": dict(sorted(pins.items())),
        "summary": {"accepted_native_assets": len(result),
            "accepted_participants": len({r["src_subject_id"] for r in result}),
            "by_role": dict(sorted(Counter(r["role"] for r in result).items())),
            "photometric_counts": dict(sorted(Counter(r["photometric"] for r in result).items())),
            "missing_required_native_metadata": 0,
            "native_file_and_pixel_hashes_verified": verify_payloads,
            "native_arrays_loaded": False, "anatomical_decisions_regraded": False,
            "outcomes_opened": False}}
