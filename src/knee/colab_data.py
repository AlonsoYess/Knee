"""Verify the sealed development package on CPU, without opening reserved test.

No extraction, training, imputation, augmentation, or model-specific normalization
is performed. All patient-level grouping is checked before tensor payload access.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

import numpy as np

VERSION = "colab_data_package_v1.0"
ROW_FIELDS = set("src_subject_id side preprocessing_role outer_block cv_fold AGEYEARS SEX BMI PROGRESION_48M tensor".split())
TENSOR_FIELDS = {"relative_path", "sha256", "pixel_sha256", "shape", "dtype"}
REQUIRED_BINDINGS = {"partition_sha256", "split_policy_sha256", "cohort_closure_sha256", "tensor_contract_sha256"}
MAX_TENSOR_BYTES = 262144


def require(condition, code):
    if not condition:
        raise ValueError(code)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink() and not getattr(part, "is_junction", lambda: False)(), "package_links_forbidden")
    return path


def safe_name(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value
        and "\0" not in value and not value.startswith("/"), "safe_package_relative_path_required")
    require(all(p not in {"", ".", ".."} for p in value.split("/"))
        and str(PurePosixPath(value)) == value, "safe_package_relative_path_required")
    return value


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_manifest_json_key")
        result[key] = value
    return result


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def validate_bindings(bindings):
    require(isinstance(bindings, dict) and REQUIRED_BINDINGS <= set(bindings)
        and all(isinstance(k, str) and re.fullmatch(r"[a-z][a-z0-9_]*_sha256", k)
            and valid_sha(v) for k, v in bindings.items()), "hash_only_package_bindings_required")


def validate_rows(rows, kind=None):
    require(isinstance(rows, list) and rows and len(rows) <= 5000, "bounded_nonempty_package_rows_required")
    seen, files, groups, clinical = set(), set(), {}, {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == ROW_FIELDS, "exact_package_row_allowlist_required")
        subject = row["src_subject_id"]
        require(isinstance(subject, str) and subject.strip() == subject and subject
            and len(subject) <= 128 and row["side"] in {"LEFT", "RIGHT"}, "package_identity_required")
        pair = subject, row["side"]
        require(pair not in seen, "duplicate_package_knee")
        seen.add(pair)
        require(row["preprocessing_role"] in {"development", "routine"}, "known_preprocessing_role_required")
        block, fold = row["outer_block"], row["cv_fold"]
        require(type(block) is int and 0 <= block <= 4, "valid_outer_block_required")
        require((block == 0 and fold is None and row["preprocessing_role"] == "routine")
            or (block != 0 and type(fold) is int and 0 <= fold <= 4), "valid_development_fold_or_reserved_role_required")
        require(kind is None or (kind == "reserved_test") == (block == 0), "package_role_membership_mismatch")
        assignment = row["preprocessing_role"], block, fold
        require(subject not in groups or groups[subject] == assignment, "participant_crosses_package_blocks_or_folds")
        groups[subject] = assignment
        require(type(row["AGEYEARS"]) is int and row["AGEYEARS"] > 0
            and row["SEX"] in {"F", "M"}, "valid_package_age_and_sex_required")
        bmi = row["BMI"]
        require(bmi is None or (type(bmi) in {int, float} and math.isfinite(bmi) and bmi > 0), "valid_or_missing_package_bmi_required")
        participant_values = row["AGEYEARS"], row["SEX"], bmi
        require(subject not in clinical or clinical[subject] == participant_values,
            "participant_baseline_values_inconsistent")
        clinical[subject] = participant_values
        require(type(row["PROGRESION_48M"]) is int and row["PROGRESION_48M"] in {0, 1}, "binary_package_outcome_required")
        tensor = row["tensor"]
        require(isinstance(tensor, dict) and set(tensor) == TENSOR_FIELDS
            and type(tensor["shape"]) is list and all(type(v) is int for v in tensor["shape"])
            and tensor["shape"] == [1, 224, 224] and tensor["dtype"] == "float32"
            and valid_sha(tensor["sha256"]) and valid_sha(tensor["pixel_sha256"]), "exact_frozen_tensor_descriptor_required")
        relative = safe_name(tensor["relative_path"])
        require(relative.endswith(".npy") and relative.casefold() not in files, "unique_package_tensor_paths_required")
        files.add(relative.casefold())
    return groups


def validate_manifest(manifest):
    require(isinstance(manifest, dict) and set(manifest) == {"version", "package_kind", "bindings", "rows"}
        and manifest["version"] == VERSION and manifest["package_kind"] in {"development", "reserved_test"},
        "exact_package_manifest_required")
    validate_bindings(manifest["bindings"])
    validate_rows(manifest["rows"], manifest["package_kind"])


def _inspect_package(package, expected_sha256, *, retain_arrays):
    require(valid_sha(expected_sha256), "expected_package_sha256_required")
    path = checked_path(package)
    require(path.is_file() and sha_file(path) == expected_sha256, "package_sha256_mismatch")
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= 5001, "bounded_package_members_required")
        names = [safe_name(x.filename) for x in entries]
        require(len(set(n.casefold() for n in names)) == len(names), "duplicate_zip_member")
        require("manifest.json" in names, "package_manifest_missing")
        for item in entries:
            mode = item.external_attr >> 16
            require(not item.is_dir() and not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in {0, stat.S_IFREG}
                and not (item.flag_bits & 1) and item.compress_type == zipfile.ZIP_STORED,
                "regular_unencrypted_stored_zip_members_required")
            require(0 < item.file_size <= (20 * 1024**2 if item.filename == "manifest.json" else MAX_TENSOR_BYTES),
                "bounded_package_member_size_required")
        try:
            manifest = json.loads(archive.read("manifest.json"), object_pairs_hook=_json_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_manifest_json")))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid_package_manifest_json") from exc
        validate_manifest(manifest)
        require(manifest["package_kind"] == "development", "reserved_test_package_open_forbidden")
        expected_names = {"manifest.json"} | {r["tensor"]["relative_path"] for r in manifest["rows"]}
        require(set(names) == expected_names, "exact_package_file_set_required")
        for row in manifest["rows"]:
            spec = row["tensor"]
            raw = archive.read(spec["relative_path"])
            require(hashlib.sha256(raw).hexdigest() == spec["sha256"], "package_tensor_file_hash_mismatch")
            try:
                buffer = io.BytesIO(raw)
                tensor = np.load(buffer, allow_pickle=False, max_header_size=4096)
            except (ValueError, OSError, EOFError) as exc:
                raise ValueError("safe_tensor_npy_required") from exc
            require(isinstance(tensor, np.ndarray) and tensor.shape == (1, 224, 224)
                and tensor.dtype == np.dtype("float32") and tensor.flags.c_contiguous
                and np.isfinite(tensor).all() and tensor.min() >= 0 and tensor.max() <= 1,
                "finite_frozen_tensor_shape_dtype_range_required")
            require(hashlib.sha256(tensor.tobytes(order="C")).hexdigest() == spec["pixel_sha256"],
                "package_tensor_pixel_hash_mismatch")
            require(buffer.tell() == len(raw) and 0 < len(raw) - tensor.nbytes <= 4096,
                "bounded_tensor_header_required")
            if retain_arrays:
                tensor.setflags(write=False)
                arrays[(row["src_subject_id"], row["side"])] = tensor
    require(sha_file(path) == expected_sha256, "package_changed_during_verification")
    summary = {"version": VERSION, "status": "development_package_verified_no_training",
        "package_sha256": expected_sha256, "package_kind": "development",
        "participants": len({r["src_subject_id"] for r in manifest["rows"]}), "knees": len(manifest["rows"]),
        "verified_tensors": len(manifest["rows"]), "cv_fold_counts": dict(sorted(Counter(str(r["cv_fold"]) for r in manifest["rows"]).items())),
        "bmi_missing_knees": sum(r["BMI"] is None for r in manifest["rows"]),
        "all_tensor_file_and_pixel_hashes_verified": True, "all_finite_float32_1x224x224": True,
        "participant_assignments_consistent": True, "reserved_test_opened": False,
        "training_executed": False, "imputation_applied": False, "model_normalization_applied": False,
        "files_extracted": False}
    return manifest, arrays, summary


def verify_development_package(package, expected_sha256, record_path=None):
    """Stream-verify development tensors and return aggregate-only results."""
    _, _, summary = _inspect_package(package, expected_sha256, retain_arrays=False)
    if record_path is not None:
        record = checked_path(record_path)
        payload = canonical_json(summary)
        if record.exists():
            require(record.is_file() and record.read_bytes() == payload, "existing_verification_record_changed_no_overwrite")
        else:
            require(record.parent.is_dir(), "existing_record_parent_required")
            with record.open("xb") as stream:
                stream.write(payload)
    return summary


def load_development_package(package, expected_sha256):
    """Explicitly load validated development tensors; arrays are read-only."""
    manifest, arrays, summary = _inspect_package(package, expected_sha256, retain_arrays=True)
    return {"manifest": manifest, "tensors": arrays, "verification": summary}
