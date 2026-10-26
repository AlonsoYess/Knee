"""Bounded, baseline-only global retention monitoring after sealed image QC.

Run in the separate analysis interpreter, never the frozen neural environment.
Only aggregate JSON is written. This does not authorize the next study stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys


VERSION = "global_retention_monitor_v1.0"
COLUMNS = ["SRC_SUBJECT_ID", "SIDE", "KL_BASE", "AGEYEARS", "SEX", "BMI"]
COMPONENTS = {
    "src/knee/roi_feasibility_baseline.py",
    "src/knee/roi_feasibility_accounting.py", "src/knee/audit.py",
    "src/knee/roi_global_retention_sources.py",
    "src/knee/roi_global_retention_summary.py",
    "src/knee/roi_global_retention_monitor.py",
}
FORBIDDEN_ACTIONS = (
    "outcome_access", "new_image_processing", "historical_regrading",
    "partitioning", "training", "reserved_test_access", "drive_upload",
    "data_deletion", "algorithm_change", "common_cohort_freeze",
)


def require(condition, code):
    if not condition:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(value, *, directory=False):
    path = Path(value)
    require(path.is_absolute() and path != Path(path.anchor), "absolute_nonroot_path_required")
    for part in (path, *path.parents):
        require(not part.is_symlink() and not getattr(part, "is_junction", lambda: False)(),
                "linked_paths_forbidden")
    require(path.is_dir() if directory else path.is_file(), "required_path_missing")
    return path.resolve()


def read_pinned(spec):
    require(isinstance(spec, dict) and set(spec) == {"path", "sha256"}, "exact_pinned_source_required")
    path = checked_path(spec["path"])
    require(sha_file(path) == spec["sha256"], "pinned_source_hash_changed")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_config(config):
    require(isinstance(config, dict) and set(config) == {
        "version", "sources", "baseline", "policy", "authority", "component_pins", "output_root"},
        "exact_monitor_config_required")
    require(config["version"] == VERSION, "monitor_version_changed")
    project = checked_path(config["sources"]["project_root"], directory=True)
    require(set(config["component_pins"]) == COMPONENTS, "exact_monitor_components_required")
    for relative, expected in config["component_pins"].items():
        path = checked_path(project / relative)
        require(path.is_relative_to(project) and sha_file(path) == expected, "monitor_component_hash_changed")
    require(checked_path(__file__) == project / "src/knee/roi_global_retention_monitor.py",
            "adopted_monitor_module_required")
    authority = read_pinned(config["authority"])
    require(set(authority) == {"version", "approved_by", "user_statement", "scope", "baseline_columns",
                              "allowed_actions", "forbidden_actions", "executed_at_utc"},
            "exact_bounded_authority_required")
    require(authority["version"] == VERSION and authority["approved_by"] == "researcher"
            and isinstance(authority["user_statement"], str) and authority["user_statement"].strip()
            and authority["scope"] == "sealed_qc_metadata_and_baseline_retention_only"
            and authority["baseline_columns"] == COLUMNS
            and authority["allowed_actions"] == ["metadata_reconciliation", "baseline_projection",
                                                 "descriptive_aggregation", "private_aggregate_json"]
            and authority["forbidden_actions"] == list(FORBIDDEN_ACTIONS), "bounded_authority_changed")
    policy = read_pinned(config["policy"])
    amendment = policy["prospective_amendment"]
    require(policy["mcr_id"] == "MCR-2026-013"
            and amendment["baseline_projection_only"] is True
            and amendment["baseline_columns"] == COLUMNS
            and amendment["exclusion_monitoring_required"] is True
            and amendment["prior_development_assignment_permanent"] is True
            and amendment["recover_prior_development_cases_for_test_allowed"] is False
            and amendment["smd_role"] == "descriptive_only_not_a_pass_fail_threshold_or_proof_of_no_bias",
            "approved_descriptive_policy_changed")
    baseline_spec = config["baseline"]
    require(isinstance(baseline_spec, dict) and set(baseline_spec) == {"path", "sha256", "bytes", "columns"}
            and baseline_spec["columns"] == COLUMNS and type(baseline_spec["bytes"]) is int,
            "exact_baseline_projection_contract_required")
    baseline_path = checked_path(baseline_spec["path"])
    require(baseline_path.is_relative_to(project), "baseline_outside_project")
    policy_anchor = policy["evidence_anchors"]["baseline_mixed_csv"]
    require(baseline_path == project / policy_anchor["path"]
            and baseline_spec["sha256"] == policy_anchor["sha256"], "policy_baseline_anchor_changed")
    output = Path(config["output_root"])
    require(output.is_absolute() and output.name == "global_retention_monitor_001"
            and output != Path(output.anchor), "dedicated_private_output_required")
    checked_path(output, directory=True)
    require(all(not output.is_relative_to(checked_path(config["sources"][key], directory=True))
                for key in ("project_root", "runtime_root", "campaign_root", "development_root")),
            "output_must_not_modify_existing_source_trees")
    require(checked_path(config["authority"]["path"]).is_relative_to(output / "governance"),
            "private_authority_must_be_adopted_under_output")
    return project, baseline_path, output


def validate_baseline_codes(values):
    """Reuse original source-domain rules without imputation or new QC cutoffs."""
    for value in values.values():
        age, bmi = value["AGEYEARS"], value["BMI"]
        require(age is None or (math.isfinite(age) and age > 0 and float(age).is_integer()),
                "observed_age_source_contract_violation")
        require(bmi is None or (math.isfinite(bmi) and bmi > 0), "observed_bmi_source_contract_violation")
        require(value["SEX"] in {"", "F", "M"}, "observed_sex_source_contract_violation")
        require(value["KL_BASE"] in {"2", "3"}, "observed_eligible_kl_source_contract_violation")


def check_aggregate_privacy(value, identities, _pattern=None):
    if _pattern is None:
        require(bool(identities), "nonempty_private_identity_set_required")
        _pattern = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(
            re.escape(sid) for sid in sorted(identities)) + r")(?![A-Za-z0-9])")
    def contains_identity(text):
        return _pattern.search(text) is not None
    if isinstance(value, dict):
        require(not any(contains_identity(str(key)) for key in value), "individual_identity_key_in_aggregate")
        for item in value.values():
            check_aggregate_privacy(item, identities, _pattern)
    elif isinstance(value, (tuple, list)):
        for item in value:
            check_aggregate_privacy(item, identities, _pattern)
    elif isinstance(value, str):
        require(not contains_identity(value), "individual_identity_value_in_aggregate")


def load_component(project, name):
    expected = project / "src/knee" / (name + ".py")
    qualified = "knee." + name
    spec = importlib.util.find_spec(qualified)
    require(spec is not None and checked_path(spec.origin) == expected, "component_import_path_changed")
    module = importlib.import_module(qualified)
    require(checked_path(module.__file__) == expected, "loaded_component_path_changed")
    return module


def write_once(path, record):
    """Exclusive immutable write; a rerun verifies and reuses the same analysis."""
    require(not path.is_symlink() and not getattr(path, "is_junction", lambda: False)(),
            "linked_result_file_forbidden")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(existing, dict) and "record_sha256" in existing
                and digest({k: v for k, v in existing.items() if k != "record_sha256"})
                == existing["record_sha256"], "existing_monitor_envelope_changed")
        require(existing.get("analysis_sha256") == record["analysis_sha256"]
                and digest(existing["analysis"]) == record["analysis_sha256"], "existing_monitor_result_changed")
        require(set(existing) == set(record) and existing["version"] == VERSION,
                "existing_monitor_record_contract_changed")
        return existing, True
    data = (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    require(path.read_bytes() == data, "aggregate_output_exact_readback_failed")
    return record, False


def run(config_path):
    config_path = checked_path(config_path)
    config_hash = sha_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project, baseline_path, output = validate_config(config)
    # Validate dependency locations before any scientific/analysis import.
    for name in ("roi_global_retention_sources", "roi_global_retention_summary",
                 "roi_feasibility_baseline", "roi_feasibility_accounting"):
        spec = importlib.util.find_spec("knee." + name)
        require(spec is not None and checked_path(spec.origin) == project / "src/knee" / (name + ".py"),
                "component_import_path_changed")
    sources = load_component(project, "roi_global_retention_sources")
    baseline = load_component(project, "roi_feasibility_baseline")
    load_component(project, "roi_feasibility_accounting")
    summary = load_component(project, "roi_global_retention_summary")
    ledger = sources.collect(config["sources"])
    spec = config["baseline"]
    frame = baseline.read_projection(baseline_path, spec["sha256"], spec["bytes"])
    values, people = baseline.reconcile(frame, ledger["inventory"])
    validate_baseline_codes(values)
    aggregated = summary.summarize(ledger["rows"], values, people)
    check_aggregate_privacy(aggregated, set(people))
    pins = dict(ledger["pins"])
    pins.update({str(baseline_path): spec["sha256"], str(config_path): config_hash,
                 str(checked_path(config["policy"]["path"])): config["policy"]["sha256"],
                 str(checked_path(config["authority"]["path"])): config["authority"]["sha256"]})
    pins.update({str(project / p): sha for p, sha in config["component_pins"].items()})
    require(all(sha_file(checked_path(path)) == sha for path, sha in pins.items()),
            "source_changed_during_monitoring")
    analysis = {
        "status": "global_descriptive_retention_and_baseline_monitor_complete",
        "phase": 1, "step": 5, "algorithm_version": sources.CANDIDATE,
        "technical_accounting": ledger["summary"], "baseline_retention": aggregated,
        "source_files_sha256": dict(sorted(pins.items())),
        "sealed_private_join_rows_sha256": digest(ledger["rows"]),
        "authorized_projection_columns": COLUMNS, "baseline_row_count": len(frame),
        "step_4_status": "closed_technical", "step_5_status": "global_monitor_complete_pending_common_cohort",
        "development_participants_permanently_excluded_from_final_test": len(ledger["roles"]["development_ids"]),
        "outcome_columns_materialized": False, "individual_records_emitted": False,
        "new_images_processed": False, "native_pixel_audit_repeated_in_this_run": False,
        "sealed_prior_native_integrity_evidence_reused": True,
        "qc_decisions_regraded": False, "algorithm_changed": False, "imputation_performed": False,
        "partitions_created": False, "training_executed": False, "training_ready": False,
        "reserved_test_opened": False, "drive_upload_performed": False, "data_deleted": False,
        "cohort_feasibility_certified": False, "scientific_or_clinical_validation_claimed": False,
        "external_pretraining_disjointness_verified": False,
        "next_stage_requires_separate_authority": True,
        "pending_before_training": ["authorized_common_cohort_and_outcome_definition",
                                    "development_only_native_to_tensor_contract",
                                    "authorized_participant_splits_and_predictive_evaluability",
                                    "verified_private_drive_training_package"],
    }
    check_aggregate_privacy(analysis, set(people))
    import pandas
    import numpy
    record = {"version": VERSION, "executed_at_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_environment": {"python": platform.python_version(), "pandas": pandas.__version__,
                                       "numpy": numpy.__version__, "separate_from_frozen_neural_runtime": True},
              "analysis_sha256": digest(analysis), "analysis": analysis}
    record["record_sha256"] = digest(record)
    result_path = output / "global_retention_baseline.json"
    record, resumed = write_once(result_path, record)
    return {"version": VERSION, "status": analysis["status"],
            "technical_accounting": ledger["summary"], "analysis_sha256": record["analysis_sha256"],
            "result_sha256": sha_file(result_path), "result_path": str(result_path),
            "existing_result_reused": resumed, "training_ready": False,
            "next_stage_requires_separate_authority": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config), ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        print(json.dumps({"version": VERSION, "status": "global_monitor_failed_no_stage_advancement"}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
