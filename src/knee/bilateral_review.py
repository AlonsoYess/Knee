"""Validate the blinded bilateral review and freeze the accepted pilot parameters."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knee.dicom_audit import sha256_file


REQUIRED_REVIEW_FIELDS = {
    "case_alias",
    "preview_file",
    "automatic_confidence_level",
    "automatic_split_fraction",
    "proposed_image_left_side",
    "proposed_image_right_side",
    "split_acceptable_yes_no",
    "laterality_mapping_supported_yes_no",
    "observed_marker_or_anatomy",
    "technical_exclusion_yes_no",
    "exclusion_reason",
    "reviewed_without_outcome_yes_no",
    "reviewer_notes",
}

REQUIRED_RESULT_FIELDS = {
    "case_alias",
    "split_fraction",
    "confidence_level",
    "technical_status",
    "proposed_image_left_side",
    "proposed_image_right_side",
    "preview_file",
    "error_code",
}


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path.name)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise ValueError("missing_csv_fields:" + ",".join(missing))
        return list(reader)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path.name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}_must_be_an_object")
    return value


def _decision(value: str, field: str, alias: str) -> str:
    normalized = str(value).strip().upper().replace("Í", "I")
    if normalized not in {"SI", "NO"}:
        raise ValueError(f"invalid_{field}:{alias}")
    return normalized


def finalize_bilateral_review(
    output_dir: Path,
    expected_unique_studies: int,
    expected_algorithm_version: str,
    expected_boundary_review_count: int,
    closure_git_commit: str,
) -> dict[str, Any]:
    """Validate the human ledger and write immutable closure evidence."""
    summary_path = output_dir / "resumen_separacion_publico.json"
    params_path = output_dir / "parametros_candidatos.json"
    preparation_path = output_dir / "registro_preparacion.json"
    review_path = output_dir / "revision_visual_ciega.csv"
    results_path = output_dir / "resultados_separacion_privados.csv"

    summary = _read_json(summary_path)
    parameters = _read_json(params_path)
    preparation = _read_json(preparation_path)
    reviews = _read_csv(review_path, REQUIRED_REVIEW_FIELDS)
    results = _read_csv(results_path, REQUIRED_RESULT_FIELDS)

    if summary.get("status") != "ready_for_blinded_review":
        raise ValueError("pilot_not_ready_for_blinded_review")
    if summary.get("algorithm_version") != expected_algorithm_version:
        raise ValueError("unexpected_algorithm_version")
    if parameters.get("algorithm_version") != expected_algorithm_version:
        raise ValueError("candidate_parameters_version_mismatch")
    if parameters.get("status") != "candidate_not_frozen_until_visual_review":
        raise ValueError("candidate_parameters_have_unexpected_status")
    if preparation.get("status") != "ready_for_blinded_review":
        raise ValueError("preparation_record_not_ready")
    if int(summary.get("processed_unique_studies", -1)) != expected_unique_studies:
        raise ValueError("unexpected_processed_study_count")
    if int(summary.get("technical_failures", -1)) != 0:
        raise ValueError("technical_failures_present")
    if any(
        bool(summary.get(key))
        for key in (
            "outcome_data_loaded",
            "mass_processing_executed",
            "training_executed",
            "reserved_test_opened",
        )
    ):
        raise ValueError("safety_gate_violation")
    if len(reviews) != expected_unique_studies or len(results) != expected_unique_studies:
        raise ValueError("unexpected_review_or_result_count")

    review_by_alias = {row["case_alias"]: row for row in reviews}
    result_by_alias = {row["case_alias"]: row for row in results}
    if len(review_by_alias) != len(reviews) or len(result_by_alias) != len(results):
        raise ValueError("duplicate_case_alias")
    if set(review_by_alias) != set(result_by_alias):
        raise ValueError("review_result_alias_mismatch")

    confidence_counts: Counter[str] = Counter()
    boundary_review_count = 0
    for alias in sorted(result_by_alias):
        result = result_by_alias[alias]
        review = review_by_alias[alias]
        if str(result["error_code"]).strip():
            raise ValueError(f"technical_error_present:{alias}")
        if review["preview_file"] != result["preview_file"]:
            raise ValueError(f"preview_file_mismatch:{alias}")
        if review["automatic_confidence_level"] != result["confidence_level"]:
            raise ValueError(f"automatic_confidence_mismatch:{alias}")
        if abs(
            float(review["automatic_split_fraction"])
            - float(result["split_fraction"])
        ) > 1e-12:
            raise ValueError(f"automatic_split_mismatch:{alias}")
        for field in ("proposed_image_left_side", "proposed_image_right_side"):
            if review[field] != result[field]:
                raise ValueError(f"laterality_proposal_mismatch:{alias}")
        if _decision(review["split_acceptable_yes_no"], "split_decision", alias) != "SI":
            raise ValueError(f"split_not_accepted:{alias}")
        if (
            _decision(
                review["laterality_mapping_supported_yes_no"],
                "laterality_decision",
                alias,
            )
            != "SI"
        ):
            raise ValueError(f"laterality_not_supported:{alias}")
        if _decision(review["technical_exclusion_yes_no"], "exclusion", alias) != "NO":
            raise ValueError(f"technical_exclusion_present:{alias}")
        if _decision(review["reviewed_without_outcome_yes_no"], "blinding", alias) != "SI":
            raise ValueError(f"review_not_blinded:{alias}")
        if not str(review["observed_marker_or_anatomy"]).strip():
            raise ValueError(f"visual_evidence_missing:{alias}")
        if not str(review["reviewer_notes"]).strip():
            raise ValueError(f"reviewer_notes_missing:{alias}")

        confidence_counts[result["confidence_level"]] += 1
        if result["technical_status"] == "REVIEW_REQUIRED_BOUNDARY":
            boundary_review_count += 1
        elif result["technical_status"] != "CANDIDATE_OK":
            raise ValueError(f"unexpected_technical_status:{alias}")

    if boundary_review_count != expected_boundary_review_count:
        raise ValueError("unexpected_boundary_review_count")

    frozen_at = datetime.now(timezone.utc).isoformat()
    frozen_parameters = dict(parameters)
    frozen_parameters.update(
        {
            "status": "frozen_after_blinded_visual_review",
            "frozen_at_utc": frozen_at,
            "preparation_git_commit": preparation.get("git_commit"),
            "closure_git_commit": closure_git_commit,
            "reviewed_unique_studies": expected_unique_studies,
            "boundary_cases_reviewed_and_accepted": boundary_review_count,
            "review_csv_sha256": sha256_file(review_path),
            "results_csv_sha256": sha256_file(results_path),
            "preparation_record_sha256": sha256_file(preparation_path),
        }
    )
    (output_dir / "parametros_congelados.json").write_text(
        json.dumps(frozen_parameters, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    closure = {
        "phase": 1,
        "step": 3,
        "status": "closed",
        "closed_at_utc": frozen_at,
        "algorithm_version": expected_algorithm_version,
        "reviewed_unique_studies": expected_unique_studies,
        "split_accepted": expected_unique_studies,
        "laterality_mapping_supported": expected_unique_studies,
        "technical_exclusions": 0,
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "boundary_cases_reviewed_and_accepted": boundary_review_count,
        "laterality_mapping": {
            "image_left": "RIGHT",
            "image_right": "LEFT",
            "status": "confirmed_by_blinded_visual_review",
            "dicom_laterality_used_as_decision": False,
        },
        "reviewed_without_outcome": True,
        "parameters_frozen": True,
        "mass_processing_executed": False,
        "training_executed": False,
        "reserved_test_opened": False,
    }
    (output_dir / "cierre_paso_3_publico.json").write_text(
        json.dumps(closure, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return closure


def _resolve_under_root(root: Path, value: str, key: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{key} must be relative to KNEE_DATA_ROOT.")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{key} escapes KNEE_DATA_ROOT.")
    return resolved


def load_config(path: Path) -> dict[str, Any]:
    root_value = os.environ.get("KNEE_DATA_ROOT")
    if not root_value:
        raise ValueError("Set KNEE_DATA_ROOT to the authorized data directory.")
    root = Path(root_value).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config["output_dir"] = _resolve_under_root(root, config["output_dir"], "output_dir")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    closure = finalize_bilateral_review(
        config["output_dir"],
        int(config["expected_unique_studies"]),
        str(config["expected_algorithm_version"]),
        int(config["expected_boundary_review_count"]),
        args.git_commit,
    )
    print(json.dumps(closure, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
