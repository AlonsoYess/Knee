"""Audit the restricted cohort and workbook; emit aggregate findings only."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook

from knee.config import load_paths


COLUMNS = (
    "SRC_SUBJECT_ID", "SIDE", "KL_BASE", "KL_48", "PROGRESION_48M",
    "AGEYEARS", "SEX", "BMI", "BARCODE_BASE", "BARCODE_48",
    "IMAGE_FILE", "IMAGE_THUMBNAIL_FILE", "QC_OUTCOME",
    "CIRUGIA_PREVIA", "WOMAC_TOTAL", "FECHA_V00", "FECHA_V06",
    "REEMPLAZO_AL_INICIO", "FECHA_REEMPLAZO", "DIAS_REEMPLAZO",
    "REEMPLAZO_ANTES_V06",
)
MANIFEST_COLUMNS = (
    "SRC_SUBJECT_ID", "SIDE", "BARCODE_BASE", "IMAGE_FILE", "KL_BASE",
    "KL_48", "PROGRESION_48M", "FECHA_V00", "FECHA_V06",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_cohort(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if reader.fieldnames != list(COLUMNS):
            raise ValueError("Cohort CSV must contain the expected 21 columns in order.")
        return list(reader)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if "Manifiesto rodillas" not in workbook:
            raise ValueError("Workbook lacks the 'Manifiesto rodillas' sheet.")
        rows = workbook["Manifiesto rodillas"].values
        headers = next(rows, None)
        if not headers or not set(MANIFEST_COLUMNS).issubset(headers):
            raise ValueError("Manifest is missing required columns.")
        return [
            {name: "" if value is None else str(value) for name, value in zip(headers, row)}
            for row in rows if any(value is not None for value in row)
        ]
    finally:
        workbook.close()


def _date(value: str):
    return datetime.strptime(value, "%d/%m/%Y").date()


def audit(cohort_csv: Path, manifest_xlsx: Path) -> dict:
    """Return aggregate counts and checks, never individual identifiers or paths."""
    cohort_csv, manifest_xlsx = Path(cohort_csv), Path(manifest_xlsx)
    rows = _read_cohort(cohort_csv)
    manifest = _read_manifest(manifest_xlsx)
    issues = Counter()
    keys = set()
    images_by_subject = defaultdict(set)
    labels = Counter()
    baseline = Counter()
    empty_bmi = 0

    for row in rows:
        key = (row["SRC_SUBJECT_ID"], row["SIDE"])
        if key in keys:
            issues["duplicate_subject_side"] += 1
        keys.add(key)
        if not key[0] or key[1] not in ("1", "2"):
            issues["invalid_key"] += 1
        images_by_subject[key[0]].add((row["IMAGE_FILE"], row["BARCODE_BASE"]))
        if not row["IMAGE_FILE"] or not row["BARCODE_BASE"]:
            issues["missing_image_link"] += 1

        baseline[row["KL_BASE"]] += 1
        labels[row["PROGRESION_48M"]] += 1
        try:
            base, follow, outcome = (
                int(row["KL_BASE"]), int(row["KL_48"]), int(row["PROGRESION_48M"])
            )
            if base not in (2, 3) or follow not in range(5) or outcome not in (0, 1):
                issues["invalid_grade_or_label"] += 1
            elif outcome != int(follow - base >= 1):
                issues["label_mismatch"] += 1
        except ValueError:
            issues["invalid_grade_or_label"] += 1
        try:
            if _date(row["FECHA_V00"]) >= _date(row["FECHA_V06"]):
                issues["invalid_visit_order"] += 1
        except ValueError:
            issues["invalid_visit_date"] += 1
        try:
            age = int(row["AGEYEARS"])
            if age <= 0 or row["SEX"] not in ("F", "M"):
                issues["invalid_clinical_value"] += 1
            if row["BMI"] and float(row["BMI"]) <= 0:
                issues["invalid_clinical_value"] += 1
        except ValueError:
            issues["invalid_clinical_value"] += 1
        empty_bmi += not bool(row["BMI"])

    issues["subject_with_multiple_baseline_images"] += sum(
        len(values) != 1 for values in images_by_subject.values()
    )

    expected = {
        (row["SRC_SUBJECT_ID"], row["SIDE"]): row for row in rows
    }
    manifest_keys = set()
    for entry in manifest:
        key = (entry["SRC_SUBJECT_ID"], entry["SIDE"])
        if key in manifest_keys:
            issues["duplicate_manifest_key"] += 1
        manifest_keys.add(key)
        source = expected.get(key)
        if source is None:
            issues["manifest_key_not_in_cohort"] += 1
            continue
        if any(entry[name] != source[name] for name in MANIFEST_COLUMNS):
            issues["manifest_value_mismatch"] += 1
    issues["cohort_key_not_in_manifest"] = len(keys - manifest_keys)

    failures = {key: value for key, value in sorted(issues.items()) if value}
    return {
        "status": "ok" if not failures else "fail",
        "source_sha256": {"cohort_csv": sha256(cohort_csv), "manifest_xlsx": sha256(manifest_xlsx)},
        "counts": {
            "cohort_rows": len(rows),
            "participants": len(images_by_subject),
            "unique_bilateral_images": len({row["IMAGE_FILE"] for row in rows if row["IMAGE_FILE"]}),
            "manifest_rows": len(manifest),
            "progressors": labels["1"], "non_progressors": labels["0"],
            "baseline_kl2": baseline["2"], "baseline_kl3": baseline["3"],
            "missing_bmi": empty_bmi,
        },
        "checks": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    paths = load_paths(args.config)
    result = audit(paths["cohort_csv"], paths["manifest_xlsx"])
    target = paths["audit_json"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
