"""Freeze the common analytic membership after sealed image QC, before splitting.

Private inputs and outputs are configured externally. Only accepted knees have
their outcome projected, after immutable outcome-free membership is written.
This command neither transforms pixels nor creates partitions or fits models.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform

from knee import roi_global_retention_sources as sources
from knee.roi_global_retention_monitor import checked_path, check_aggregate_privacy


VERSION = "common_effective_cohort_v1.0"
BASELINE_COLUMNS = ["SRC_SUBJECT_ID", "SIDE", "KL_BASE", "AGEYEARS", "SEX", "BMI",
                    "BARCODE_BASE", "IMAGE_FILE"]
LABEL_COLUMNS = ["SRC_SUBJECT_ID", "SIDE", "KL_BASE", "KL_48", "PROGRESION_48M",
                 "FECHA_V00", "FECHA_V06"]
SOURCE_HEADER = tuple((
    "SRC_SUBJECT_ID SIDE KL_BASE KL_48 PROGRESION_48M AGEYEARS SEX BMI BARCODE_BASE "
    "BARCODE_48 IMAGE_FILE IMAGE_THUMBNAIL_FILE QC_OUTCOME CIRUGIA_PREVIA WOMAC_TOTAL "
    "FECHA_V00 FECHA_V06 REEMPLAZO_AL_INICIO FECHA_REEMPLAZO DIAS_REEMPLAZO "
    "REEMPLAZO_ANTES_V06").split())
SIDE_MAP = {"1": "RIGHT", "2": "LEFT"}
ASSET_FIELDS = set(("src_subject_id side role block acquisition_key image_file barcode_base knee_alias crop "
    "photometric row_spacing_mm column_spacing_mm spacing_tag source_shape source_dtype "
    "source_half_sha256 half_x0 half_x1 box native_dicom_box native_spatial_orientation").split())
require, digest, sha_file = sources.require, sources.digest, sources.sha_file


def key(row, *, source=False):
    sid = row.get("SRC_SUBJECT_ID" if source else "src_subject_id")
    side = row.get("SIDE" if source else "side")
    if source:
        side = SIDE_MAP.get(side)
    require(isinstance(sid, str) and sid.strip() == sid and bool(sid)
            and side in {"RIGHT", "LEFT"}, "invalid_cohort_identity")
    return sid, side


def unique_rows(rows, *, source=False):
    result = {}
    for row in rows:
        pair = key(row, source=source)
        require(pair not in result, "duplicate_participant_side")
        result[pair] = row
    return result


def read_projection(path, spec_columns, accepted_keys=None):
    """Project only allowlisted columns; skip excluded outcome records by key.

    CSV tokenization necessarily scans the pinned file; excluded outcomes and
    unrelated columns are never converted into projected records or emitted.
    """
    require(spec_columns in (BASELINE_COLUMNS, LABEL_COLUMNS), "unapproved_projection_columns")
    require(accepted_keys is not None if spec_columns == LABEL_COLUMNS else accepted_keys is None,
            "outcome_projection_requires_sealed_membership")
    result, seen = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        header = next(reader, None)
        require(header == list(SOURCE_HEADER), "exact_source_header_required")
        indices = {name: header.index(name) for name in spec_columns}
        for tokens in reader:
            require(len(tokens) == len(header), "malformed_source_record")
            pair = key({"SRC_SUBJECT_ID": tokens[0], "SIDE": tokens[1]}, source=True)
            require(pair not in seen, "duplicate_participant_side")
            seen.add(pair)
            if accepted_keys is not None and pair not in accepted_keys:
                continue
            result.append({name: tokens[index] for name, index in indices.items()})
    if accepted_keys is not None:
        require({key(r, source=True) for r in result} == accepted_keys,
                "accepted_outcome_source_keys_missing")
    return result


def _integer(value, allowed=None):
    require(isinstance(value, str) and value.strip() == value and value.lstrip("-").isdigit(),
            "integer_source_value_required")
    number = int(value)
    require(allowed is None or number in allowed, "integer_source_domain_violation")
    return number


def _clinical(row):
    age = _integer(row["AGEYEARS"])
    require(age > 0 and row["SEX"] in {"F", "M"}, "invalid_age_or_sex")
    bmi = None
    if row["BMI"] != "":
        try:
            bmi = float(row["BMI"])
        except (TypeError, ValueError):
            raise ValueError("invalid_bmi") from None
        require(math.isfinite(bmi) and bmi > 0, "invalid_bmi")
    return {"AGEYEARS": age, "SEX": row["SEX"], "BMI": bmi}


def build_membership(ledger, baseline_rows, assets):
    """Join baseline identity and native assets using QC decisions, never labels."""
    qc = unique_rows(ledger["rows"])
    baseline = unique_rows(baseline_rows, source=True)
    inventory = {r["src_subject_id"]: r for r in ledger["inventory"]}
    require(len(inventory) == len(ledger["inventory"]), "duplicate_inventory_identity")
    prior = set(ledger["roles"]["development_ids"])
    routine = set(ledger["roles"]["routine_ids"])
    require(not prior & routine and prior | routine == set(inventory), "permanent_role_union_changed")
    expected = {(sid, side) for sid, r in inventory.items() for side in sources.eligible_sides(r)}
    require(set(qc) == set(baseline) == expected, "complete_baseline_inventory_qc_join_required")
    asset_rows = assets["rows"] if isinstance(assets, dict) else assets
    native = unique_rows(asset_rows)
    accepted = {pair for pair, r in qc.items() if r["state"] == "accepted"}
    require(set(native) == accepted, "exact_accepted_asset_set_required")
    people, rows, exclusions = {}, [], []
    for pair in sorted(qc):
        sid, side = pair
        decision, base, original = qc[pair], baseline[pair], inventory[sid]
        role = "development" if sid in prior else "routine"
        require(decision["role"] == role, "permanent_development_role_changed")
        require(decision["state"] in {"accepted", "excluded"}, "terminal_qc_required")
        require((not decision["reasons"]) if decision["state"] == "accepted" else bool(decision["reasons"]),
                "terminal_qc_reasons_changed")
        require(base["IMAGE_FILE"] == original["image_file"]
                and base["BARCODE_BASE"] == original["barcode_base"], "baseline_acquisition_link_mismatch")
        clinical = _clinical(base)
        signature = (clinical, base["IMAGE_FILE"], base["BARCODE_BASE"])
        require(sid not in people or people[sid] == signature, "inconsistent_bilateral_baseline")
        people[sid] = signature
        kl = _integer(base["KL_BASE"], {2, 3})
        if pair not in accepted:
            exclusions.append(dict(decision, acquisition_key=original["acquisition_key"]))
            continue
        asset = native[pair]
        require(all(asset.get(name) == value for name, value in {
            "role": role, "block": decision["block"], "acquisition_key": original["acquisition_key"],
            "image_file": original["image_file"], "barcode_base": original["barcode_base"]}.items()),
            "native_asset_identity_mismatch")
        require(set(asset) <= ASSET_FIELDS, "unapproved_native_membership_fields")
        rows.append(dict(asset, clinical=clinical, KL_BASE=kl))
    missing = [r for r in rows if r["clinical"]["BMI"] is None]
    return {"version": VERSION, "membership_rule": "sealed_accepted_eligible_native_crop_and_baseline_identity",
        "outcomes_used_for_membership": False, "rows": rows, "exclusions": exclusions,
        "development_reservations": sorted(prior),
        "summary": {"eligible_knees": len(qc), "accepted": len(rows), "excluded": len(exclusions),
            "pending": 0, "participants": len({r["src_subject_id"] for r in rows}),
            "permanent_development_participants": len(prior),
            "routine_knees": sum(r["role"] == "routine" for r in rows),
            "routine_participants": len({r["src_subject_id"] for r in rows if r["role"] == "routine"}),
            "development_knees": sum(r["role"] == "development" for r in rows),
            "development_participants": len({r["src_subject_id"] for r in rows if r["role"] == "development"}),
            "missing_bmi_knees": len(missing),
            "missing_bmi_participants": len({r["src_subject_id"] for r in missing})},
        "imputation": {"performed": False, "BMI": "median_fit_on_training_fold_only",
                       "missingness_indicator_added": False},
        "same_membership_for": ["clinical", "radiographic", "multimodal"],
        "partitions_created": False}


def build_labels(membership, label_rows):
    members = unique_rows(membership["rows"])
    labels = unique_rows(label_rows, source=True)
    require(set(members) == set(labels), "exact_accepted_outcome_join_required")
    rows, counts, participant_classes = [], Counter(), {}
    for pair in sorted(members):
        member, raw = members[pair], labels[pair]
        base = _integer(raw["KL_BASE"], {2, 3})
        follow = _integer(raw["KL_48"], set(range(5)))
        y = _integer(raw["PROGRESION_48M"], {0, 1})
        require(base == member["KL_BASE"] and y == int(follow - base >= 1), "predefined_label_mismatch")
        try:
            dates = [datetime.strptime(raw[name], "%d/%m/%Y").date()
                     for name in ("FECHA_V00", "FECHA_V06")]
        except (TypeError, ValueError):
            raise ValueError("invalid_visit_date") from None
        require(dates[0] < dates[1], "invalid_visit_order")
        rows.append({"src_subject_id": pair[0], "side": pair[1], "KL_BASE": base,
            "KL_48": follow, "PROGRESION_48M": y,
            "FECHA_V00": dates[0].isoformat(), "FECHA_V06": dates[1].isoformat()})
        counts[(member["role"], y)] += 1
        participant_classes.setdefault(pair[0], set()).add(y)
    return {"version": VERSION, "membership_sha256": digest(membership),
        "definition": "int(KL_V06 - KL_V00 >= 1)", "rows": rows,
        "summary": {"knees": len(rows), "positive_knees": sum(r["PROGRESION_48M"] for r in rows),
            "negative_knees": sum(r["PROGRESION_48M"] == 0 for r in rows),
            "participants_with_any_positive_knee": sum(1 in v for v in participant_classes.values()),
            "participants_without_positive_knee": sum(1 not in v for v in participant_classes.values()),
            "role_class_counts": {role: {"positive": counts[(role, 1)], "negative": counts[(role, 0)]}
                                  for role in ("routine", "development")}},
        "used_for_qc_or_membership_selection": False, "partitions_created": False}


def write_once(path, payload):
    path = Path(path)
    require(not path.is_symlink() and not getattr(path, "is_junction", lambda: False)(), "linked_output_forbidden")
    if path.exists():
        require(json.loads(path.read_text(encoding="utf-8")) == payload, "immutable_output_changed")
        return True
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == data, "output_readback_failed")
    return False


def read_pinned(spec):
    require(set(spec) == {"path", "sha256"} and sources.valid_sha(spec["sha256"]), "pinned_source_required")
    path = checked_path(spec["path"])
    require(sha_file(path) == spec["sha256"], "source_pin_changed")
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def run(config_path):
    config_path = checked_path(config_path)
    config_hash = sha_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(set(config) == {"version", "monitor", "monitor_config", "authority", "component_pins", "output_root"}
            and config["version"] == VERSION, "exact_cohort_config_required")
    monitor, monitor_config, authority = (read_pinned(config[name]) for name in ("monitor", "monitor_config", "authority"))
    require(monitor["analysis_sha256"] == digest(monitor["analysis"])
            and monitor["record_sha256"] == digest({k: v for k, v in monitor.items() if k != "record_sha256"}),
            "prior_monitor_envelope_changed")
    require(authority.get("version") == VERSION and authority.get("approved_by") == "researcher"
            and authority.get("scope") == "freeze_common_membership_and_verify_predefined_outcomes"
            and authority.get("user_statement")
            and authority.get("allowed_actions") == ["sealed_metadata_join", "accepted_native_integrity_check",
                "baseline_identity_projection", "outcome_free_membership_seal", "accepted_outcome_verification",
                "private_common_cohort_seal"]
            and authority.get("forbidden_actions") == ["training", "partitioning", "reserved_test_access",
                "qc_regrading", "algorithm_change", "native_crop_edit", "new_selection", "drive_upload", "data_deletion"],
            "cohort_authority_scope_changed")
    project = checked_path(monitor_config["sources"]["project_root"], directory=True)
    require(checked_path(__file__) == project / "src/knee/common_cohort.py",
            "adopted_cohort_module_required")
    components = {"src/knee/common_cohort.py", "src/knee/common_cohort_assets.py",
                  "src/knee/roi_global_retention_sources.py", "src/knee/roi_global_retention_monitor.py"}
    require(set(config["component_pins"]) == components, "exact_cohort_components_required")
    pins = {str(config_path): config_hash}
    for relative, sha in config["component_pins"].items():
        path = checked_path(project / relative)
        require(path.is_relative_to(project) and sha_file(path) == sha, "cohort_component_changed")
        pins[str(path)] = sha
    for name in ("monitor", "monitor_config", "authority"):
        pins[str(checked_path(config[name]["path"]))] = config[name]["sha256"]
    # The new authority governs this stage; earlier monitor-only authority stays unchanged.
    require(all(sha_file(checked_path(path)) == sha for path, sha in monitor["analysis"]["source_files_sha256"].items()),
            "prior_monitor_sources_changed")
    ledger = sources.collect(monitor_config["sources"])
    require(digest(ledger["rows"]) == monitor["analysis"]["sealed_private_join_rows_sha256"],
            "qc_ledger_differs_from_reviewed_monitor")
    output = Path(config["output_root"])
    require(output.is_absolute() and output.name == "common_cohort_001", "dedicated_cohort_output_required")
    checked_path(output, directory=True)
    source_roots = [project] + [checked_path(monitor_config["sources"][k], directory=True)
        for k in ("runtime_root", "campaign_root", "development_root")]
    require(all(not output.resolve().is_relative_to(root) for root in source_roots), "output_overlaps_source_tree")
    require(checked_path(config["authority"]["path"]).is_relative_to(output.resolve() / "governance"),
            "authority_must_belong_to_new_cohort")
    baseline_spec = monitor_config["baseline"]
    baseline_path = checked_path(baseline_spec["path"])
    require(sha_file(baseline_path) == baseline_spec["sha256"]
            and baseline_path.stat().st_size == baseline_spec["bytes"], "baseline_source_changed")
    from knee.common_cohort_assets import collect_assets
    require(checked_path(collect_assets.__code__.co_filename) == project / "src/knee/common_cohort_assets.py",
            "adopted_asset_module_required")
    assets = collect_assets(ledger, monitor_config["sources"], verify_payloads=True)
    baseline = read_projection(baseline_path, BASELINE_COLUMNS)
    membership = build_membership(ledger, baseline, assets)
    require(membership["summary"]["accepted"] == 2161 and membership["summary"]["participants"] == 1571
            and membership["summary"]["excluded"] == 617
            and membership["summary"]["permanent_development_participants"] == 90,
            "reviewed_common_cohort_counts_changed")
    membership_path = output / "membership_without_outcomes.json"
    membership_reused = write_once(membership_path, membership)
    membership_file_sha = sha_file(membership_path)
    # This is deliberately after immutable outcome-free membership readback.
    keys = {key(row) for row in membership["rows"]}
    label_rows = read_projection(baseline_path, LABEL_COLUMNS, keys)
    labels = build_labels(membership, label_rows)
    require(sha_file(membership_path) == membership_file_sha, "membership_changed_during_outcome_verification")
    labels_path = output / "outcomes.json"
    labels_reused = write_once(labels_path, labels)
    for more_pins in (ledger["pins"], assets["pins"]):
        require(all(path not in pins or pins[path] == sha for path, sha in more_pins.items()),
                "conflicting_source_pin")
        pins.update(more_pins)
    pins[str(baseline_path)] = baseline_spec["sha256"]
    require(all(sha_file(checked_path(path)) == sha for path, sha in pins.items()), "source_changed_during_cohort_closure")
    provenance = {"version": VERSION, "source_files_sha256": dict(sorted(pins.items())),
        "execution_environment": {"python": platform.python_version(), "separate_from_neural_runtime": True},
        "native_integrity": assets["summary"], "baseline_projection_columns": BASELINE_COLUMNS,
        "accepted_only_outcome_projection_columns": LABEL_COLUMNS,
        "membership_verified_and_written_before_outcome_projection": True,
        "native_roots": {k: monitor_config["sources"][k] for k in
                         ("project_root", "runtime_root", "campaign_root", "development_root")}}
    provenance_path = output / "provenance.json"
    write_once(provenance_path, provenance)
    summary = {"version": VERSION, "phase": 1, "step": 6,
        "operation": "freeze_common_effective_cohort_from_sealed_qc_and_verified_labels",
        "status": "common_cohort_frozen_input_tensor_and_splits_pending",
        "algorithm_version": sources.CANDIDATE, "membership": membership["summary"],
        "outcomes": labels["summary"], "native_integrity": assets["summary"],
        "same_cohort_for_three_main_models": True,
        "membership_sealed_before_outcome_projection": True,
        "outcome_data_loaded": True, "outcomes_used_for_qc_or_membership": False,
        "imputation_performed": False, "qc_regraded": False, "algorithm_changed": False,
        "native_crop_edits": False, "partitions_created": False, "training_executed": False,
        "reserved_test_opened": False, "drive_upload_performed": False, "data_deleted": False,
        "training_ready": False, "cohort_predictive_evaluability_certified": False,
        "external_pretraining_disjointness_verified": False,
        "files_sha256": {p.name: sha_file(p) for p in (membership_path, labels_path, provenance_path)},
        "pending_before_training": ["development_only_native_to_224_tensor_contract",
            "participant_level_splits_and_class_evaluability", "verified_private_drive_training_package"],
        "limitations": ["Assisted technical QC is not independent clinical validation.",
            "Unequal retention by sex and baseline KL limits generalization.",
            "External KNEEL/OAI pretraining overlap remains unresolved."],
        "next_step": "verify_and_freeze_model_input_transformation_using_development_only"}
    check_aggregate_privacy(summary, {r["src_subject_id"] for r in ledger["rows"]})
    summary_path = output / "closure_summary.json"
    reused = write_once(summary_path, summary)
    return dict(summary, existing_outputs_reused=reused and membership_reused and labels_reused,
                closure_summary_path=str(summary_path), closure_summary_sha256=sha_file(summary_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.config)
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        print(json.dumps({"version": VERSION, "status": "cohort_closure_failed_no_training_or_partition"}))
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
