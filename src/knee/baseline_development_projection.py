"""Project audited V00 predictors for exact, already-sealed development knees.

The audited CSV contains follow-up columns. CSV tokenization necessarily scans
its bytes, but this module uses identity first and converts only allowlisted
baseline values for selected development keys. Follow-up/outcome fields are
never indexed into a projected row or converted. It does not open a reserved
archive, select new participants, fit an estimator or alter any source file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re

from knee import clinical_scenarios as clinical
from knee.colab_data import checked_path, require, sha_file, valid_sha

VERSION = "audited_development_baseline_projection_v1.0"
PROJECTION_FILENAME = "development_baseline_projection_v1.json"
RECORD_FILENAME = "baseline_projection_record.json"
MAX_SOURCE_BYTES = 64 * 1024**2
MAX_SOURCE_ROWS = 5000

# Exact audited schema from knee.audit.COLUMNS. Do not import that module:
# baseline projection must not require Excel support or read a workbook.
SOURCE_HEADER = tuple((
    "SRC_SUBJECT_ID SIDE KL_BASE KL_48 PROGRESION_48M AGEYEARS SEX BMI BARCODE_BASE "
    "BARCODE_48 IMAGE_FILE IMAGE_THUMBNAIL_FILE QC_OUTCOME CIRUGIA_PREVIA WOMAC_TOTAL "
    "FECHA_V00 FECHA_V06 REEMPLAZO_AL_INICIO FECHA_REEMPLAZO DIAS_REEMPLAZO "
    "REEMPLAZO_ANTES_V06").split())
PROJECTED_SOURCE_COLUMNS = tuple(clinical.SOURCE_COLUMNS.values())
SIDE_MAP = {"1": "RIGHT", "2": "LEFT"}


def _projection_bytes(rows):
    return (json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _integer(raw, *, allowed=None, positive=False):
    require(isinstance(raw, str) and re.fullmatch(r"-?[0-9]+", raw) is not None,
            "integer_baseline_source_value_required")
    value = int(raw)
    require((allowed is None or value in allowed) and (not positive or value > 0),
            "baseline_integer_domain_violation")
    return value


def _number(raw, *, minimum=None, maximum=None, positive=False):
    require(isinstance(raw, str) and raw and raw.strip() == raw,
            "numeric_baseline_source_value_required")
    try:
        value = float(raw)
    except (ValueError, OverflowError):
        raise ValueError("numeric_baseline_source_value_required") from None
    require(math.isfinite(value) and (minimum is None or value >= minimum)
            and (maximum is None or value <= maximum) and (not positive or value > 0),
            "baseline_numeric_domain_violation")
    return value


def _identity(subject, source_side):
    require(isinstance(subject, str) and subject and subject.strip() == subject
            and len(subject) <= 128 and source_side in SIDE_MAP,
            "audited_source_identity_required")
    return subject, SIDE_MAP[source_side]


def project_source(source_csv, expected_source_sha256, development_rows, *, development_package_sha256):
    """Return typed baseline-only rows and aggregate integrity/projection proof.

``development_rows`` must originate from the authenticated development package.
This pure projection validates their development assignment and exact clinical
identity, but authenticating package bytes is the caller's responsibility.
The public CLI performs that authentication before calling this function.
"""
    require(valid_sha(expected_source_sha256) and valid_sha(development_package_sha256),
            "externally_pinned_source_and_development_package_required")
    base = clinical._base_rows(development_rows, all_folds=True)
    expected = {(row["src_subject_id"], row["side"]): row for row in base}
    require(len(expected) == len(base), "unique_development_projection_keys_required")
    source = checked_path(source_csv)
    require(source.is_file() and 0 < source.stat().st_size <= MAX_SOURCE_BYTES,
            "bounded_audited_source_csv_required")
    require(sha_file(source) == expected_source_sha256, "audited_baseline_source_sha256_mismatch")
    indices = {name: SOURCE_HEADER.index(name) for name in PROJECTED_SOURCE_COLUMNS}
    rows, seen, source_count = [], set(), 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        require(next(reader, None) == list(SOURCE_HEADER), "exact_audited_baseline_source_header_required")
        for tokens in reader:
            source_count += 1
            require(source_count <= MAX_SOURCE_ROWS and len(tokens) == len(SOURCE_HEADER),
                    "bounded_well_formed_audited_source_rows_required")
            # Resolve keys before touching any clinical value. Unselected rows
            # can contain arbitrary clinical/follow-up strings without conversion.
            pair = _identity(tokens[0], tokens[1])
            require(pair not in seen, "duplicate_audited_source_knee")
            seen.add(pair)
            if pair not in expected:
                continue
            get = lambda name: tokens[indices[name]]
            age = _integer(get("AGEYEARS"), positive=True)
            sex = get("SEX")
            require(sex in {"F", "M"}, "baseline_source_sex_required")
            bmi = None if get("BMI") == "" else _number(get("BMI"), positive=True)
            original = expected[pair]
            require(age == original["AGEYEARS"] and sex == original["SEX"] and bmi == original["BMI"],
                    "source_main_predictors_differ_from_development_package")
            # Preserve the authenticated package's numeric representation after
            # checking source equality; e.g. an integral BMI may be stored as int.
            row = {"src_subject_id": pair[0], "side": pair[1],
                **{name: original[name] for name in ("AGEYEARS", "SEX", "BMI")},
                "KL_BASE": _integer(get("KL_BASE"), allowed={2, 3}),
                "CIRUGIA_PREVIA": None if get("CIRUGIA_PREVIA") == "" else
                    _integer(get("CIRUGIA_PREVIA"), allowed={0, 1}),
                "WOMAC_TOTAL": None if get("WOMAC_TOTAL") == "" else
                    _number(get("WOMAC_TOTAL"), minimum=0, maximum=96)}
            rows.append(row)
    require({(r["src_subject_id"], r["side"]) for r in rows} == set(expected),
            "audited_source_missing_development_baseline_keys")
    rows.sort(key=lambda row: (row["src_subject_id"], row["side"]))
    require(sha_file(source) == expected_source_sha256, "audited_baseline_source_changed_during_projection")
    # Reuse the exact downstream clinical contract without producing a sidecar
    # or fitting preprocessing; caller will authenticate this projection's SHA.
    clinical.make_baseline_sidecar(rows, development_rows, development_package_sha256,
                                   expected_source_sha256)
    raw = _projection_bytes(rows)
    record = {"version": VERSION, "status": "development_baseline_only_projection_ready",
        "scope": "development_only", "baseline_visit": "V00",
        "source_sha256": expected_source_sha256,
        "development_package_sha256": development_package_sha256,
        "projection_file": PROJECTION_FILENAME, "projection_bytes": len(raw),
        "projection_sha256": hashlib.sha256(raw).hexdigest(),
        "development_participants": len({row["src_subject_id"] for row in rows}),
        "development_knees": len(rows), "source_rows_scanned": source_count,
        "skipped_non_development_rows": source_count - len(rows),
        "source_columns_projected": list(PROJECTED_SOURCE_COLUMNS),
        "missing_counts": {name: sum(row[name] is None for row in rows)
            for name in ("KL_BASE", "CIRUGIA_PREVIA", "WOMAC_TOTAL")},
        "baseline_kl_counts": {str(value): sum(row["KL_BASE"] == value for row in rows) for value in (2, 3)},
        "surgery_counts": {str(value): sum(row["CIRUGIA_PREVIA"] == value for row in rows) for value in (0, 1)},
        "main_predictors_match_authenticated_development_package": True,
        "source_followup_columns_projected": False, "source_outcome_values_converted": False,
        "reserved_test_archive_opened": False, "training_executed": False,
        "originals_modified": False, "partitions_modified": False,
        "source_sha256_unchanged": True,
        "limitations": ["CSV tokenization scans the pinned source file; only selected development baseline fields are projected.",
            "Source/package hashes bind supplied artifacts; clinical validity is not established by projection."]}
    record["record_sha256"] = clinical.main._digest(record)
    return rows, record


def prepare_projection(source_csv, expected_source_sha256, development_rows, output_dir, *, development_package_sha256):
    """Write a new, private baseline projection and aggregate receipt, never overwrite."""
    source, output = map(checked_path, (source_csv, output_dir))
    source, output = source.resolve(), output.resolve()
    repository = Path(__file__).resolve().parents[2]
    require(not output.exists() and output.parent.is_dir()
            and not output.is_relative_to(repository) and not repository.is_relative_to(output)
            and not output.is_relative_to(source.parent) and not source.parent.is_relative_to(output),
            "new_disjoint_private_baseline_projection_output_required")
    rows, record = project_source(source, expected_source_sha256, development_rows,
                                  development_package_sha256=development_package_sha256)
    raw = _projection_bytes(rows)
    output.mkdir()
    target = output / PROJECTION_FILENAME
    with target.open("xb") as stream:
        stream.write(raw)
    require(sha_file(target) == record["projection_sha256"] and target.stat().st_size == record["projection_bytes"]
            and json.loads(target.read_text(encoding="utf-8")) == rows,
            "development_baseline_projection_readback_failed")
    require(sha_file(source) == expected_source_sha256, "audited_baseline_source_changed_during_projection")
    raw_record = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with (output / RECORD_FILENAME).open("xb") as stream:
        stream.write(raw_record)
    require((output / RECORD_FILENAME).read_bytes() == raw_record, "baseline_projection_record_readback_failed")
    return record
