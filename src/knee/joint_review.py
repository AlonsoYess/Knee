"""Validate and summarize a blinded tibiofemoral-crop review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from knee.dicom_audit import sha256_file
from knee.joint_localization import REVIEW_FIELDS


YES_NO_FIELDS = (
    "joint_centered_yes_no",
    "tibiofemoral_anatomy_complete_yes_no",
    "text_borders_and_rule_excluded_yes_no",
    "crop_acceptable_yes_no",
    "technical_exclusion_yes_no",
    "reviewed_without_outcome_yes_no",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}_must_be_an_object")
    return value


def _read_review(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise ValueError("unexpected_joint_review_columns")
        return list(reader)


def summarize_blinded_joint_review(
    output_dir: Path,
    expected_algorithm_version: str,
    expected_knees: int,
) -> dict[str, Any]:
    """Validate every decision and write a public, identifier-free summary."""
    summary_path = output_dir / "resumen_localizacion_publico.json"
    parameters_path = output_dir / "parametros_candidatos.json"
    review_path = output_dir / "revision_visual_ciega.csv"
    summary = _read_json(summary_path)
    parameters = _read_json(parameters_path)
    review = _read_review(review_path)

    if summary.get("status") != "ready_for_blinded_review":
        raise ValueError("joint_pilot_was_not_ready_for_review")
    if summary.get("algorithm_version") != expected_algorithm_version:
        raise ValueError("unexpected_joint_algorithm_version")
    if parameters.get("algorithm_version") != expected_algorithm_version:
        raise ValueError("joint_parameter_version_mismatch")
    if parameters.get("status") != "candidate_not_frozen_until_blinded_visual_review":
        raise ValueError("joint_parameters_were_not_candidate_parameters")
    if len(review) != expected_knees:
        raise ValueError("unexpected_joint_review_count")
    aliases = [row.get("knee_alias", "").strip() for row in review]
    if "" in aliases or len(set(aliases)) != expected_knees:
        raise ValueError("duplicate_or_missing_knee_alias")

    reasons = Counter()
    acceptable = 0
    technical_exclusions = 0
    for row in review:
        for field in YES_NO_FIELDS:
            if row.get(field, "").strip().upper() not in {"SI", "NO"}:
                raise ValueError(f"incomplete_or_invalid_review:{row['knee_alias']}:{field}")
        centered = row["joint_centered_yes_no"].strip().upper() == "SI"
        complete = row["tibiofemoral_anatomy_complete_yes_no"].strip().upper() == "SI"
        clean = row["text_borders_and_rule_excluded_yes_no"].strip().upper() == "SI"
        accepted = row["crop_acceptable_yes_no"].strip().upper() == "SI"
        excluded = row["technical_exclusion_yes_no"].strip().upper() == "SI"
        blinded = row["reviewed_without_outcome_yes_no"].strip().upper() == "SI"
        if not blinded:
            raise ValueError(f"outcome_blinding_not_confirmed:{row['knee_alias']}")
        if accepted != (centered and complete and clean and not excluded):
            raise ValueError(f"inconsistent_crop_decision:{row['knee_alias']}")
        if excluded and not row.get("exclusion_reason", "").strip():
            raise ValueError(f"technical_exclusion_without_reason:{row['knee_alias']}")
        acceptable += int(accepted)
        technical_exclusions += int(excluded)
        if not centered:
            reasons["joint_not_centered"] += 1
        if not complete:
            reasons["anatomy_incomplete"] += 1
        if not clean:
            reasons["peripheral_artifact_present"] += 1

    rejected = expected_knees - acceptable
    status = "accepted_after_blinded_review" if rejected == 0 else "rejected_after_blinded_review"
    public = {
        "status": status,
        "algorithm_version": expected_algorithm_version,
        "reviewed_knees": expected_knees,
        "acceptable_crops": acceptable,
        "rejected_crops": rejected,
        "technical_exclusions": technical_exclusions,
        "rejection_reason_counts": dict(sorted(reasons.items())),
        "reviewed_without_outcome": True,
        "parameters_frozen": False,
        "mass_processing_executed": False,
        "partitions_created": False,
        "training_executed": False,
        "reserved_test_opened": False,
        "review_csv_sha256": sha256_file(review_path),
        "candidate_parameters_sha256": sha256_file(parameters_path),
    }
    (output_dir / "resumen_revision_visual_publico.json").write_text(
        json.dumps(public, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--algorithm-version", required=True)
    parser.add_argument("--expected-knees", required=True, type=int)
    args = parser.parse_args()
    result = summarize_blinded_joint_review(
        args.output_dir.resolve(), args.algorithm_version, args.expected_knees
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
