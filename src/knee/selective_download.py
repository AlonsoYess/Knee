"""Prepare, validate, and promote one selective NDA download batch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from knee.dicom_audit import (
    _archive_paths,
    _audit_archive,
    _selection_rank,
    canonical_package_key,
    sha256_file,
)


def _read_queue(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        required = {
            "download_batch",
            "inventory_index",
            "acquisition_key",
            "image_file",
            "local_status",
            "action_required",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("The selective-download queue has an invalid schema.")
        return list(reader)


def prepare_batch(queue_csv: Path, batch_number: int, output_txt: Path) -> dict[str, Any]:
    """Write the exact NDA S3 URLs for one deterministic queue batch."""
    if batch_number < 1:
        raise ValueError("batch_number must be positive.")
    rows = [
        row
        for row in _read_queue(Path(queue_csv))
        if row["download_batch"] == str(batch_number)
    ]
    if not rows:
        raise ValueError(f"Download batch {batch_number} is empty or does not exist.")
    if any(row["action_required"] not in {"DESCARGAR", "REEMPLAZAR"} for row in rows):
        raise ValueError("The batch contains an unauthorized action.")
    urls = [row["image_file"].strip() for row in rows]
    if any(not value.startswith("s3://") for value in urls):
        raise ValueError("Every download entry must be an NDA S3 URL.")
    if len(urls) != len(set(urls)):
        raise ValueError("The batch contains duplicated S3 URLs.")
    keys = [canonical_package_key(value) for value in urls]
    if len(keys) != len(set(keys)):
        raise ValueError("The batch contains duplicated acquisition keys.")
    if keys != [row["acquisition_key"] for row in rows]:
        raise ValueError("The queue key does not match its S3 URL.")

    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(urls) + "\n", encoding="utf-8")
    return {
        "status": "ready",
        "batch_number": batch_number,
        "expected_files": len(rows),
        "unique_s3_urls": len(set(urls)),
        "batch_file_sha256": sha256_file(output_txt),
        "training_executed": False,
        "reserved_test_opened": False,
    }


def _read_batch_keys(path: Path) -> list[str]:
    urls = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not urls or any(not value.startswith("s3://") for value in urls):
        raise ValueError("The batch file must contain one S3 URL per line.")
    keys = [canonical_package_key(value) for value in urls]
    if len(keys) != len(set(keys)):
        raise ValueError("The batch file contains duplicated acquisition keys.")
    return keys


def audit_downloaded_batch(download_dir: Path, batch_txt: Path) -> dict[str, Any]:
    """Audit a completed local batch before anything is copied to private Drive."""
    download_dir = Path(download_dir)
    if not download_dir.is_dir():
        raise FileNotFoundError(f"Download directory not found: {download_dir}")
    expected_keys = _read_batch_keys(Path(batch_txt))
    expected_set = set(expected_keys)
    records = [
        _audit_archive(path, download_dir) for path in _archive_paths(download_dir)
    ]
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_key[str(record["manifest_key"])].append(record)

    missing: list[str] = []
    unreadable: list[str] = []
    duplicates: list[str] = []
    conflicts: list[str] = []
    selected: list[dict[str, Any]] = []
    for key in expected_keys:
        candidates = by_key.get(key, [])
        if not candidates:
            missing.append(key)
            continue
        valid = [record for record in candidates if record["dicom_valid"]]
        if not valid:
            unreadable.append(key)
            continue
        identities = {str(record["pixel_sha256"]) for record in valid}
        if len(identities) != 1:
            conflicts.append(key)
            continue
        if len(candidates) != 1:
            duplicates.append(key)
            continue
        selected.append(min(valid, key=_selection_rank))

    unexpected = [
        str(record["manifest_key"])
        for record in records
        if str(record["manifest_key"]) not in expected_set
    ]
    failures = {
        "missing": len(missing),
        "unreadable": len(unreadable),
        "duplicates": len(duplicates),
        "conflicting_content": len(conflicts),
        "unexpected": len(unexpected),
    }
    failures = {name: value for name, value in failures.items() if value}
    return {
        "public_summary": {
            "status": "ok" if not failures and len(selected) == len(expected_keys) else "fail",
            "expected_files": len(expected_keys),
            "archives_found": len(records),
            "readable_archives": sum(bool(record["dicom_valid"]) for record in records),
            "selected_unique_acquisitions": len(selected),
            "checks": failures,
            "training_executed": False,
            "reserved_test_opened": False,
        },
        "private_failures": {
            "missing_keys": missing,
            "unreadable_keys": unreadable,
            "duplicate_keys": duplicates,
            "conflicting_keys": conflicts,
            "unexpected_keys": unexpected,
        },
        "selected": selected,
    }


def promote_downloaded_batch(
    download_dir: Path,
    batch_txt: Path,
    destination_root: Path,
    batch_number: int,
) -> dict[str, Any]:
    """Copy a fully audited batch to its immutable private-Drive lot folder."""
    result = audit_downloaded_batch(download_dir, batch_txt)
    if result["public_summary"]["status"] != "ok":
        raise ValueError("Downloaded batch failed audit; no package was promoted.")
    lot_dir = Path(destination_root) / f"lote_{batch_number:03d}"
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    selected_sources = [
        (
            Path(download_dir) / str(record["package_relative_path"]),
            record,
        )
        for record in result["selected"]
    ]
    target_names = [source.name for source, _ in selected_sources]
    if len(target_names) != len(set(target_names)):
        raise ValueError("Different acquisitions would produce the same destination name.")

    copied = 0
    already_present = 0
    if lot_dir.exists():
        existing = sorted(path.name for path in lot_dir.iterdir() if path.is_file())
        if sorted(target_names) != existing:
            raise FileExistsError(
                "The destination lot exists but is incomplete or contains extra files."
            )
        for source, record in selected_sources:
            if sha256_file(lot_dir / source.name) != record["package_sha256"]:
                raise FileExistsError(
                    "A destination package exists with different content; promotion stopped."
                )
        already_present = len(selected_sources)
    else:
        staging = destination_root / f".lote_{batch_number:03d}.staging"
        if staging.exists():
            raise FileExistsError(
                "A staging folder already exists; inspect it before retrying promotion."
            )
        staging.mkdir()
        try:
            for source, record in selected_sources:
                target = staging / source.name
                shutil.copy2(source, target)
                if sha256_file(target) != record["package_sha256"]:
                    raise IOError("Copied package failed SHA-256 verification.")
            os.replace(staging, lot_dir)
            copied = len(selected_sources)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    promoted: list[dict[str, str]] = []
    for source, record in selected_sources:
        target = lot_dir / source.name
        promoted.append(
            {
                "manifest_key": str(record["manifest_key"]),
                "destination_relative_path": target.relative_to(destination_root).as_posix(),
                "package_sha256": str(record["package_sha256"]),
                "dicom_sha256": str(record["dicom_sha256"]),
                "pixel_sha256": str(record["pixel_sha256"]),
            }
        )
    result["public_summary"].update(
        {
            "batch_number": batch_number,
            "copied_to_drive": copied,
            "already_present_identical": already_present,
        }
    )
    result["promoted"] = promoted
    return result


def write_batch_evidence(result: dict[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = int(result["public_summary"]["batch_number"])
    prefix = f"lote_{batch:03d}"
    (output_dir / f"{prefix}_resumen_publico.json").write_text(
        json.dumps(result["public_summary"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    private = {
        "public_summary": result["public_summary"],
        "private_failures": result["private_failures"],
        "promoted": result["promoted"],
    }
    (output_dir / f"{prefix}_evidencia_privada.json").write_text(
        json.dumps(private, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--queue", required=True, type=Path)
    prepare.add_argument("--batch", required=True, type=int)
    prepare.add_argument("--output", required=True, type=Path)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--download-dir", required=True, type=Path)
    promote.add_argument("--batch-file", required=True, type=Path)
    promote.add_argument("--destination-root", required=True, type=Path)
    promote.add_argument("--batch", required=True, type=int)
    promote.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        summary = prepare_batch(args.queue, args.batch, args.output)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    result = promote_downloaded_batch(
        args.download_dir,
        args.batch_file,
        args.destination_root,
        args.batch,
    )
    write_batch_evidence(result, args.evidence_dir)
    print(json.dumps(result["public_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
