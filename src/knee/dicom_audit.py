"""Reconcile a private pilot manifest with DICOM packages and audit integrity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pydicom


DETAIL_FIELDS = (
    "manifest_key",
    "package_relative_path",
    "package_bytes",
    "package_sha256",
    "dicom_member",
    "dicom_sha256",
    "pixel_sha256",
    "dicom_valid",
    "selected_canonical",
    "rows",
    "columns",
    "number_of_frames",
    "bits_allocated",
    "bits_stored",
    "pixel_representation",
    "photometric_interpretation",
    "modality",
    "view_position",
    "laterality",
    "pixel_spacing",
    "transfer_syntax_uid",
    "error_code",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_package_key(value: str) -> str:
    """Return a manifest/package key independent of encoded path and tar suffix."""
    decoded = unquote(str(value)).replace("\\", "/").rstrip("/")
    name = decoded.rsplit("/", 1)[-1]
    lowered = name.lower()
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _read_manifest_keys(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if not reader.fieldnames:
            raise ValueError("The pilot manifest has no header.")
        source_column = next(
            (name for name in ("NOMBRE_ARCHIVO", "IMAGE_FILE") if name in reader.fieldnames),
            None,
        )
        if source_column is None:
            raise ValueError("The pilot manifest lacks NOMBRE_ARCHIVO or IMAGE_FILE.")
        return [
            canonical_package_key(row[source_column])
            for row in reader
            if row.get(source_column, "").strip()
        ]


def _archive_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name.lower().endswith((".tar", ".tar.gz", ".tgz"))
    )


def _string_value(dataset: pydicom.Dataset, name: str) -> str:
    value = getattr(dataset, name, "")
    return "" if value is None else str(value)


def _integer_value(dataset: pydicom.Dataset, name: str, default: int = 0) -> int:
    value = getattr(dataset, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_dicom_member(data: bytes) -> tuple[pydicom.Dataset, np.ndarray]:
    dataset = pydicom.dcmread(io.BytesIO(data), force=True)
    if "PixelData" not in dataset or not getattr(dataset, "Rows", None):
        raise ValueError("not_an_image_dicom")
    pixels = dataset.pixel_array
    if pixels.size == 0:
        raise ValueError("empty_pixel_array")
    return dataset, pixels


def _audit_archive(path: Path, root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {name: "" for name in DETAIL_FIELDS}
    record.update(
        {
            "manifest_key": canonical_package_key(path.name),
            "package_relative_path": path.relative_to(root).as_posix(),
            "package_bytes": path.stat().st_size,
            "package_sha256": sha256_file(path),
            "dicom_valid": False,
            "selected_canonical": False,
        }
    )
    valid_members: list[tuple[str, bytes, pydicom.Dataset, np.ndarray]] = []
    try:
        with tarfile.open(path, "r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read()
                try:
                    dataset, pixels = _read_dicom_member(data)
                except Exception:
                    continue
                valid_members.append((member.name, data, dataset, pixels))
    except (OSError, tarfile.TarError):
        record["error_code"] = "unreadable_archive"
        return record

    if not valid_members:
        record["error_code"] = "dicom_not_found_or_unreadable"
        return record
    if len(valid_members) != 1:
        record["error_code"] = "multiple_dicom_images_in_package"
        return record

    member_name, data, dataset, pixels = valid_members[0]
    file_meta = getattr(dataset, "file_meta", None)
    transfer_syntax = ""
    if file_meta is not None:
        transfer_syntax = str(getattr(file_meta, "TransferSyntaxUID", ""))
    record.update(
        {
            "dicom_member": member_name,
            "dicom_sha256": sha256_bytes(data),
            "pixel_sha256": sha256_bytes(np.ascontiguousarray(pixels).tobytes()),
            "dicom_valid": True,
            "rows": _integer_value(dataset, "Rows"),
            "columns": _integer_value(dataset, "Columns"),
            "number_of_frames": _integer_value(dataset, "NumberOfFrames", 1),
            "bits_allocated": _integer_value(dataset, "BitsAllocated"),
            "bits_stored": _integer_value(dataset, "BitsStored"),
            "pixel_representation": _integer_value(dataset, "PixelRepresentation"),
            "photometric_interpretation": _string_value(
                dataset, "PhotometricInterpretation"
            ),
            "modality": _string_value(dataset, "Modality"),
            "view_position": _string_value(dataset, "ViewPosition"),
            "laterality": _string_value(dataset, "Laterality"),
            "pixel_spacing": _string_value(dataset, "PixelSpacing"),
            "transfer_syntax_uid": transfer_syntax,
            "error_code": "",
        }
    )
    return record


def _selection_rank(record: dict[str, Any]) -> tuple[int, int, int, str]:
    path = str(record["package_relative_path"]).lower()
    compressed = int(not path.endswith((".tar.gz", ".tgz")))
    depth = path.count("/")
    return compressed, depth, int(record["package_bytes"]), path


def _distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = Counter(str(record[field]) or "missing" for record in records)
    return dict(sorted(values.items()))


def audit_pilot(
    source_dir: Path,
    manifest_csv: Path,
    expected_unique_studies: int = 10,
    strict_single_package_per_study: bool = True,
) -> dict[str, Any]:
    """Audit a manifest-defined pilot without using labels for image decisions."""
    source_dir, manifest_csv = Path(source_dir), Path(manifest_csv)
    manifest_keys = _read_manifest_keys(manifest_csv)
    expected_keys = set(manifest_keys)
    records = [_audit_archive(path, source_dir) for path in _archive_paths(source_dir)]
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_key[str(record["manifest_key"])].append(record)

    duplicate_manifest_keys = len(manifest_keys) - len(expected_keys)
    missing_keys: list[str] = []
    conflict_keys: list[str] = []
    selected: list[dict[str, Any]] = []
    duplicate_package_copies = 0
    invalid_expected_packages = 0

    for key in sorted(expected_keys):
        candidates = by_key.get(key, [])
        valid = [record for record in candidates if record["dicom_valid"]]
        invalid_expected_packages += len(candidates) - len(valid)
        if not valid:
            missing_keys.append(key)
            continue
        identities = {str(record["pixel_sha256"]) for record in valid}
        if len(identities) != 1:
            conflict_keys.append(key)
            continue
        duplicate_package_copies += max(0, len(valid) - 1)
        chosen = min(valid, key=_selection_rank)
        chosen["selected_canonical"] = True
        selected.append(chosen)

    identity_to_keys: dict[str, set[str]] = defaultdict(set)
    for record in selected:
        identity_to_keys[str(record["pixel_sha256"])].add(str(record["manifest_key"]))
    cross_key_duplicates = sum(len(keys) - 1 for keys in identity_to_keys.values())
    unexpected_archives = sum(
        1 for record in records if str(record["manifest_key"]) not in expected_keys
    )
    unique_acquisitions = len(identity_to_keys)

    checks = {
        "manifest_row_count_mismatch": int(len(manifest_keys) != expected_unique_studies),
        "manifest_duplicate_keys": duplicate_manifest_keys,
        "missing_expected_studies": len(missing_keys),
        "conflicting_contents_for_manifest_key": len(conflict_keys),
        "cross_key_duplicate_acquisitions": cross_key_duplicates,
        "unexpected_archives": unexpected_archives,
        "invalid_expected_packages": invalid_expected_packages,
        "unique_acquisition_count_mismatch": int(
            unique_acquisitions != expected_unique_studies
        ),
    }
    if strict_single_package_per_study:
        checks["duplicate_package_copies"] = duplicate_package_copies
    failures = {name: count for name, count in checks.items() if count}

    public_summary = {
        "status": "ok" if not failures else "fail",
        "expected_unique_studies": expected_unique_studies,
        "manifest_rows": len(manifest_keys),
        "manifest_unique_studies": len(expected_keys),
        "archives_found": len(records),
        "readable_dicom_packages": sum(bool(record["dicom_valid"]) for record in records),
        "selected_unique_acquisitions": unique_acquisitions,
        "duplicate_package_copies": duplicate_package_copies,
        "checks": failures,
        "technical_profile": {
            "dimensions": _distribution(
                [dict(record, dimensions=f"{record['rows']}x{record['columns']}") for record in selected],
                "dimensions",
            ),
            "frames": _distribution(selected, "number_of_frames"),
            "bits_stored": _distribution(selected, "bits_stored"),
            "photometric_interpretation": _distribution(
                selected, "photometric_interpretation"
            ),
            "modality": _distribution(selected, "modality"),
            "pixel_spacing": _distribution(selected, "pixel_spacing"),
            "view_position": _distribution(selected, "view_position"),
            "laterality": _distribution(selected, "laterality"),
        },
    }
    return {
        "public_summary": public_summary,
        "private_reconciliation": {
            "manifest_sha256": sha256_file(manifest_csv),
            "missing_manifest_keys": missing_keys,
            "conflicting_manifest_keys": conflict_keys,
            "selected_packages": [
                {
                    "manifest_key": record["manifest_key"],
                    "package_relative_path": record["package_relative_path"],
                    "package_sha256": record["package_sha256"],
                    "dicom_sha256": record["dicom_sha256"],
                    "pixel_sha256": record["pixel_sha256"],
                }
                for record in selected
            ],
        },
        "records": records,
    }


def write_audit(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "auditoria_piloto_publica.json").write_text(
        json.dumps(result["public_summary"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    private_payload = {
        "public_summary": result["public_summary"],
        "private_reconciliation": result["private_reconciliation"],
    }
    (output_dir / "auditoria_piloto_privada.json").write_text(
        json.dumps(private_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "auditoria_piloto_detalle.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=DETAIL_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(result["records"])


def _resolve_under_root(root: Path, value: str, key: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{key} must be a relative path below KNEE_DATA_ROOT.")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{key} escapes KNEE_DATA_ROOT.")
    return resolved


def load_audit_config(path: Path) -> dict[str, Any]:
    root_value = os.environ.get("KNEE_DATA_ROOT")
    if not root_value:
        raise ValueError("Set KNEE_DATA_ROOT to the authorized data directory.")
    root = Path(root_value).expanduser().resolve()
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("source_dir", "pilot_manifest_csv", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing relative path: {key}.")
        config[key] = _resolve_under_root(root, config[key], key)
    config.setdefault("expected_unique_studies", 10)
    config.setdefault("strict_single_package_per_study", True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_audit_config(args.config)
    result = audit_pilot(
        config["source_dir"],
        config["pilot_manifest_csv"],
        int(config["expected_unique_studies"]),
        bool(config["strict_single_package_per_study"]),
    )
    write_audit(result, config["output_dir"])
    print(json.dumps(result["public_summary"], indent=2, ensure_ascii=False))
    if result["public_summary"]["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
