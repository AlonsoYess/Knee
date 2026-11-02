"""Read only, metadata-only reconciliation of all sealed eligible QC decisions.

No pixel, model, clinical table, future label or outcome is opened. This module
does not regrade a decision. The returned private ledger is for internal joins;
callers must publish aggregates only and retain the role distinction.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


VERSION = "global_sealed_qc_metadata_collection_v1.0"
CANDIDATE = "kneel_guided_gradient_contours_contact_anchor_v1"
INVENTORY_REL = "outputs/mcr006_confirmation/inputs/inventario_adquisiciones_privado.csv"
MONITOR_REL = "outputs/roi_fifth_batch_20261004/retention_five_batches_publico.json"
INVENTORY_FIELDS = tuple((
    "inventory_index acquisition_key src_subject_id barcode_base image_file image_thumbnail_file "
    "associated_knees lateralities local_status action_required download_batch archive_count "
    "readable_archive_count distinct_pixel_contents canonical_source canonical_package_relative_path "
    "package_bytes package_sha256 dicom_sha256 pixel_sha256").split())
SIDES = {"Derecha": ["RIGHT"], "Izquierda": ["LEFT"],
         "Derecha + Izquierda": ["RIGHT", "LEFT"]}
COMPONENTS = {
    "src/knee/roi_contact_geometry.py": "bbf42ba23acc38fa4b7157d1898bcc75bd392a6650bcea19fdb839deeae897ed",
    "src/knee/roi_contour_geometry.py": "39e488d79ae25253925c4b9f1a3e7998a15fa39b2af8739c49b5e5059b752607",
    "src/knee/roi_landmark_geometry.py": "d613f8816d6d110d32d1c5923ffe1048a43e2115ddd2737906c4eb12352bdf16",
    "src/knee/roi_task_qc.py": "e143855d41ef997964b4fe6fc8a33f700ebee1c7b54b6e7f34f7db90acf9ef94",
    "src/knee/roi_confirmation_v2.py": "4e8532cebdd54b1999dc104592d7f5091a5aa21ea57584259475107e78d6c547",
}
ROW_FIELDS = set("src_subject_id case_alias side knee_alias state source_status source center box observation crop_sha256 reasons".split())
PREFIX = {
    1: {"record": "outputs/roi_second_batch_20261004/evidence/colab21_publico.json",
        "plan": "outputs/roi_second_batch_20261004/evidence/prior_plan.json",
        "checkpoint": "outputs/roi_second_batch_20261004/evidence/prior_checkpoint.json"},
}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(root, relative):
    root = Path(root).resolve()
    rel = str(relative).replace("\\", "/")
    require(not Path(rel).is_absolute() and ":" not in rel
            and all(x not in {"", ".", ".."} for x in rel.split("/")), "relative_metadata_path_required")
    target = root
    for part in rel.split("/"):
        target /= part
        require(not target.is_symlink() and not getattr(target, "is_junction", lambda: False)(), "metadata_links_forbidden")
    require(target.resolve().is_relative_to(root), "metadata_path_escape")
    return target


def _unique(values, code):
    require(isinstance(values, list) and all(isinstance(x, str) and x.strip() for x in values)
            and len(values) == len(set(values)), code)
    return set(values)


def eligible_sides(row):
    result = SIDES.get(row.get("lateralities"))
    require(result is not None and row.get("associated_knees") == str(len(result)), "eligible_sides_changed")
    return list(result)


def read_inventory(path):
    require(Path(path).suffix.lower() == ".csv", "inventory_csv_required")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        # Validate the exact metadata header before the first data row is read.
        require(reader.fieldnames == list(INVENTORY_FIELDS), "metadata_only_inventory_header_required")
        rows = list(reader)
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows), "malformed_inventory")
    ids = _unique([r.get("src_subject_id") for r in rows], "unique_inventory_ids_required")
    _unique([r.get("acquisition_key") for r in rows], "unique_inventory_acquisitions_required")
    require(len(ids) == 1916 and sum(len(eligible_sides(r)) for r in rows) == 2778, "complete_inventory_required")
    return rows


class MetadataReader:
    def __init__(self):
        self.pins = {}

    def read(self, root, rel, expected=None):
        path = safe(root, rel)
        require(path.suffix.lower() == ".json", "json_metadata_only")
        sha = sha_file(path)
        if expected is not None:
            require(valid_sha(expected) and sha == expected, "sealed_metadata_sha_mismatch")
        key = str(path)
        require(key not in self.pins or self.pins[key] == sha, "metadata_changed_during_collection")
        self.pins[key] = sha
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def match_record(self, root, rel, expected):
        folder = safe(root, rel)
        matches = [p for p in folder.glob("*.json") if sha_file(p) == expected]
        require(len(matches) == 1, "one_sealed_record_required")
        return self.read(root, matches[0].relative_to(Path(root)).as_posix(), expected)


def validate_selection(selected, inventory, allowed_ids=None):
    require(isinstance(selected, list) and selected, "nonempty_sealed_selection_required")
    ids = _unique([s.get("src_subject_id") for s in selected], "unique_selected_ids_required")
    _unique([s.get("case_alias") for s in selected], "unique_selected_cases_required")
    require(ids <= set(inventory) and (allowed_ids is None or ids <= set(allowed_ids)), "selected_role_or_inventory_mismatch")
    identities = {}
    for study in selected:
        require(set(study) <= set(INVENTORY_FIELDS) | {"case_alias", "eligible_sides", "development_reserved"},
                "metadata_only_selected_study_fields_required")
        if "development_reserved" in study:
            require(type(study["development_reserved"]) is bool
                    and (allowed_ids is None or study["development_reserved"] is True),
                    "historical_development_reservation_changed")
        original = inventory[study["src_subject_id"]]
        require(all(study.get(k) == original[k] for k in INVENTORY_FIELDS), "selected_inventory_metadata_changed")
        case = study["case_alias"]
        require(case not in {".", ".."} and not any(c in case for c in "/\\:"), "invalid_selected_alias")
        sides = eligible_sides(original)
        if "eligible_sides" in study:
            require(study["eligible_sides"] == sides, "selected_eligible_sides_changed")
        for side in sides:
            alias = case + "_" + side
            require(alias not in identities, "duplicate_selected_knee_alias")
            identities[alias] = {"src_subject_id": study["src_subject_id"], "case_alias": case,
                "side": side, "knee_alias": alias}
    return identities


def validate_disposition(row, historical=False):
    reasons = row.get("reasons")
    require(isinstance(reasons, list) and all(isinstance(x, str) and x.strip() for x in reasons)
            and len(reasons) == len(set(reasons)), "explicit_unique_reason_list_required")
    if historical:
        require(type(row.get("accepted")) is bool, "historical_acceptance_boolean_required")
        state = "accepted" if row["accepted"] else "excluded"
        require(row.get("state") == ("accepted_candidate" if row["accepted"] else "excluded_technical"),
                "historical_terminal_state_changed")
    else:
        state = row.get("state")
        require(state in {"accepted", "excluded"}, "all_dispositions_terminal_required")
    require(not reasons if state == "accepted" else bool(reasons), "state_reason_inconsistency")
    return state, list(reasons)


def collect_contract(plan, checked, inventory, prior, role, block, record=None):
    require(plan.get("mode") == "routine_data" and plan.get("purpose") == role
            and plan.get("known_ids") == [] and plan.get("component_hashes") == COMPONENTS,
            "frozen_role_and_science_plan_required")
    require(_unique(plan.get("prior_development_ids"), "prior_roles_required") == prior, "prior_role_union_changed")
    identities = validate_selection(plan.get("selected"), inventory)
    selected_ids = {x["src_subject_id"] for x in identities.values()}
    require((not selected_ids & prior) if role == "routine" else selected_ids <= prior, "development_routine_overlap")
    payload = checked.get("contract_snapshot")
    expected = {"version": "frozen_routine_roi_contract_v1.0", "purpose": role,
        "candidate": CANDIDATE, "qc_version": "mcr008_confirmation_v1.0",
        "frozen_components_sha256": COMPONENTS, "source_centers_sealed_before_roi_display": True,
        "outcome_blinded": True, "outcomes_loaded": False, "parameters_retuned": False,
        "sample_replacements": 0, "adjustment_subject_ids": []}
    require(isinstance(payload, dict) and set(payload) == set(expected) | {"rows"}
            and all(payload.get(k) == v and type(payload.get(k)) is type(v) for k, v in expected.items()),
            "frozen_blinded_terminal_contract_required")
    contract_sha = digest(payload)
    require(checked.get("contract_sha256") == contract_sha, "terminal_contract_digest_changed")
    rows = payload["rows"]
    require(isinstance(rows, list) and len(rows) == len(identities)
            and all(isinstance(r, dict) and set(r) == ROW_FIELDS for r in rows)
            and {r["knee_alias"] for r in rows} == set(identities), "all_unique_eligible_terminal_rows_required")
    require(_unique(checked.get("prior_development_subject_ids"), "checkpoint_prior_roles_required") == prior
            and _unique(checked.get("development_subject_ids"), "checkpoint_development_roles_required") == prior
            and _unique(checked.get("routine_only_subject_ids"), "checkpoint_routine_roles_required")
            == (selected_ids if role == "routine" else set()), "checkpoint_roles_changed")
    output = []
    for row in rows:
        require(all(row[k] == v for k, v in identities[row["knee_alias"]].items()), "terminal_identity_or_side_changed")
        state, reasons = validate_disposition(row)
        output.append({"src_subject_id": row["src_subject_id"], "side": row["side"], "role": role,
            "block": block, "state": state, "reasons": reasons})
    counts = summarize(output)
    decisions = checked.get("decisions")
    require(isinstance(decisions, list) and len(decisions) == len(rows)
            and all(isinstance(x, dict) for x in decisions)
            and {x.get("knee_alias") for x in decisions} == set(identities),
            "all_derived_terminal_decisions_required")
    by_alias = {x["knee_alias"]: x for x in rows}
    for decision in decisions:
        row = by_alias[decision["knee_alias"]]
        require(decision.get("state") == row["state"]
                and decision.get("accepted") is (row["state"] == "accepted")
                and decision.get("reasons") == row["reasons"],
                "derived_terminal_decision_disagrees")
    summary = checked.get("summary", {})
    require(all(summary.get(k) == counts[k] for k in ("participants", "eligible_knees", "accepted", "excluded"))
            and summary.get("pending") == 0 and summary.get("all_dispositions_terminal") is True,
            "checkpoint_terminal_counts_changed")
    if record is not None:
        require(record.get("contract_sha256") == contract_sha
                and record.get("selected_participants") == counts["participants"]
                and record.get("selected_eligible_knees") == counts["eligible_knees"]
                and record.get("accepted_knees") == counts["accepted"]
                and record.get("excluded_knees") == counts["excluded"]
                and record.get("pending_knees") == 0 and record.get("all_dispositions_terminal") is True,
                "sealed_execution_record_counts_changed")
    return output


def summarize(rows):
    return {"participants": len({r["src_subject_id"] for r in rows}), "eligible_knees": len(rows),
        "accepted": sum(r["state"] == "accepted" for r in rows),
        "excluded": sum(r["state"] == "excluded" for r in rows), "pending": 0}


def _prefix_history(reader, project, monitor, number):
    batch = f"routine_batch_{number:03d}"
    pin = monitor["sealed_evidence"][batch]
    if number == 1:
        paths = PREFIX[number]
        return {k: reader.read(project, rel, pin[k + "_sha256"]) for k, rel in paths.items()}
    if number == 2:
        base = "outputs/roi_second_batch_review_20261004/qc_stage_local_v1"
        return {"plan": reader.read(project, base + "/workspace/plan.json", pin["plan_sha256"]),
            "checkpoint": reader.read(project, base + f"/workspace/checkpoints/{pin['contract_sha256']}.json", pin["checkpoint_sha256"]),
            "record": reader.match_record(project, base + "/execution_records", pin["record_sha256"])}
    folder = {3: "third", 4: "fourth", 5: "fifth"}[number]
    base = f"outputs/roi_{folder}_batch_20261004/persistent/{batch}"
    result = {"plan": reader.read(project, base + "/workspace/plan.json", pin["plan_sha256"]),
        "checkpoint": reader.read(project, base + f"/workspace/checkpoints/{pin['contract_sha256']}.json", pin["checkpoint_sha256"]),
        "record": reader.match_record(project, base + "/execution_records", pin["record_sha256"])}
    reader.read(project, base + "/workspace/reference_seal.json", pin["source_reference_sha256"])
    reader.read(project, base + "/workspace/qc_review_seal.json", pin["qc_review_seal_sha256"])
    return result


def _historical_development(reader, runtime, inventory, allowed, selection_pins, metadata_pins):
    output, selected_ids = [], set()
    policy_rel = "configs/roi_mcr010_workflow_policy.json"
    registry_rel = "configs/governance/methodology_change_log.json"
    policy = reader.read(runtime, policy_rel, metadata_pins[policy_rel])
    registry = reader.read(runtime, registry_rel, metadata_pins[registry_rel])
    entries = [x for x in registry["proposals"] if x.get("id") == "MCR-2026-012"]
    require(len(entries) == 1, "unique_historical_result_registry_anchor_required")
    specs = [("mcr009", "mcr009_contact_confirmation", policy["historical_evidence"]),
             ("mcr012", "mcr012_operational_batch", entries[0]["implementation"])]
    extra = 0
    for block, folder, anchor in specs:
        result_rel = f"outputs/{folder}/application/results.json"
        selection_rel = f"outputs/{folder}/selection/selection.json"
        require(anchor.get("results_path", anchor.get("result_path")) == result_rel, "historical_result_path_changed")
        require(anchor.get("results_sha256", anchor.get("result_sha256")) == metadata_pins[result_rel],
                "historical_result_manifest_anchor_disagrees")
        result = reader.read(runtime, result_rel, metadata_pins[result_rel])
        selection = reader.read(runtime, selection_rel, selection_pins[selection_rel])
        require(result.get("candidate") == CANDIDATE and result.get("outcomes_loaded") is False
                and result.get("parameters_retuned") is False and result.get("partitions_created") is False
                and result.get("training_executed") is False and result.get("reserved_test_opened") is False,
                "frozen_historical_development_result_required")
        identities = validate_selection(selection["selected"], inventory, allowed)
        current_ids = {x["src_subject_id"] for x in selection["selected"]}
        require(not current_ids & selected_ids, "historical_development_overlap")
        selected_ids |= current_ids
        technical = {s["case_alias"] + "_" + side for s in selection["selected"]
                     for side in (["RIGHT", "LEFT"] if block == "mcr009" else eligible_sides(s))}
        rows = result["rows"]
        require(len(rows) == len(technical) and {x.get("knee_alias") for x in rows} == technical,
                "historical_complete_technical_denominator_required")
        by_alias = {x["knee_alias"]: x for x in rows}
        for row in rows:
            validate_disposition(row, historical=True)
        extra += len(technical) - len(identities)
        for alias, identity in identities.items():
            row = by_alias[alias]
            if "case_alias" in row:
                require(row["case_alias"] == identity["case_alias"], "historical_case_changed")
            if "side" in row:
                require(row["side"] == identity["side"], "historical_side_changed")
            state, reasons = validate_disposition(row, historical=True)
            output.append({"src_subject_id": identity["src_subject_id"], "side": identity["side"],
                "role": "development", "block": block, "state": state, "reasons": reasons})
    require(selected_ids == allowed and summarize(output) == {"participants": 40, "eligible_knees": 55,
        "accepted": 47, "excluded": 8, "pending": 0}, "historical_eligible_development_denominator_changed")
    require(extra == 9, "historical_ineligible_rows_accounting_changed")
    return output, extra


def collect(config):
    """Return private rows, all metadata pins, inventory and permanent role sets.

    Required config keys: project_root, runtime_root, campaign_root,
    development_root, prior_consolidation:{path,sha256},
    development_completion:{path,sha256}, inventory:{path,sha256}.
    This function writes nothing and prints nothing.
    """
    expected_keys = {"project_root", "runtime_root", "campaign_root", "development_root",
                     "prior_consolidation", "development_completion", "inventory"}
    require(isinstance(config, dict) and set(config) == expected_keys, "exact_sources_config_required")
    roots = {k: Path(config[k]).resolve() for k in ("project_root", "runtime_root", "campaign_root", "development_root")}
    require(all(p.is_dir() and p != Path(p.anchor) for p in roots.values()) and len(set(roots.values())) == 4,
            "separate_nonroot_metadata_roots_required")
    project, runtime, campaign, development = (roots[k] for k in
        ("project_root", "runtime_root", "campaign_root", "development_root"))
    reader = MetadataReader()
    fixed = (("prior_consolidation", campaign),
             ("development_completion", development), ("inventory", runtime))
    adopted = {}
    for key, root in fixed:
        spec = config[key]
        require(isinstance(spec, dict) and set(spec) == {"path", "sha256"}
                and valid_sha(spec["sha256"]), "fixed_sealed_source_pin_required")
        expected_sha = spec["sha256"]
        path = Path(spec["path"]).resolve()
        require(path.is_relative_to(root), "adopted_metadata_source_outside_role_root")
        relative = path.relative_to(root).as_posix()
        if key == "inventory":
            require(relative == INVENTORY_REL and sha_file(safe(root, relative)) == expected_sha,
                    "canonical_inventory_hash_changed")
            reader.pins[str(path)] = expected_sha
            adopted[key] = read_inventory(path)
        else:
            adopted[key] = reader.read(root, relative, expected_sha)
    completion = adopted["development_completion"]
    prior_consolidation = adopted["prior_consolidation"]
    devterminal_sha = completion.get("terminal_sha256")
    require(completion.get("all_inventory_technical_dispositions_terminal") is True
            and valid_sha(devterminal_sha)
            and completion.get("training_executed") is False and completion.get("partitions_created") is False
            and completion.get("reserved_test_opened") is False, "sealed_development_completion_scope_changed")
    inventory_rows = adopted["inventory"]
    inventory = {x["src_subject_id"]: x for x in inventory_rows}
    devterminal = reader.read(development, "terminal.json", devterminal_sha)
    devfiles = devterminal["files_sha256"]
    devplan = reader.read(development, "workspace/plan.json", devfiles["plan.json"])
    cfg = reader.read(development, "governance/config.json", devplan["policy"]["development_config_sha256"])
    roles = reader.read(development, "governance/roles.json", cfg["roles"]["sha256"])
    require(set(roles) == {"prior_development_ids", "historical_processed_ids", "routine_processed_ids"}, "exact_role_map_required")
    prior = _unique(roles["prior_development_ids"], "unique_permanent_development_ids_required")
    historical = _unique(roles["historical_processed_ids"], "unique_historical_ids_required")
    routine = _unique(roles["routine_processed_ids"], "unique_routine_ids_required")
    require(len(prior) == 90 and len(historical) == 40 and historical <= prior and len(routine) == 1826
            and not prior & routine and prior | routine == set(inventory), "exact_complete_disjoint_roles_required")
    for name in ("authority", "selection", "roles"):
        spec = cfg[name]
        path = Path(spec["path"]).resolve()
        require(path == safe(development, f"governance/{name}.json"), "development_governance_path_changed")
        reader.read(development, f"governance/{name}.json", spec["sha256"])
    devcheckpoint_rel = f"checkpoints/{devterminal['contract_sha256']}.json"
    devchecked = reader.read(development, "workspace/" + devcheckpoint_rel, devfiles[devcheckpoint_rel])
    reader.read(development, "workspace/reference_seal.json", devterminal["source_reference_sha256"])
    reader.read(development, "workspace/qc_review_seal.json", devterminal["qc_review_sha256"])
    devselection = reader.read(development, "governance/selection.json", cfg["selection"]["sha256"])
    require(devselection.get("selected") == devplan.get("selected")
            and digest(devplan["selected"]) == devterminal["selection_sha256"]
            and digest(sorted(prior)) == devterminal["development_ids_sha256"]
            and {s["src_subject_id"] for s in devplan["selected"]} == prior - historical,
            "exact_fifty_remaining_development_selection_required")
    development_rows = collect_contract(devplan, devchecked, inventory, prior, "development", "development_completion_001")
    require(summarize(development_rows) == {"participants": 50, "eligible_knees": 77,
        "accepted": 50, "excluded": 27, "pending": 0}, "new_development_counts_changed")
    manifest = reader.read(runtime, "bounded_local_capsule_manifest.json", cfg["runtime_manifest_sha256"])
    metadata_pins = manifest["workspace_files_sha256"]
    require(metadata_pins[INVENTORY_REL] == config["inventory"]["sha256"], "runtime_manifest_inventory_pin_disagrees")
    historical_rows, extra = _historical_development(reader, runtime, inventory, historical,
        devplan["input_anchors"], metadata_pins)
    prior_reconciliation = prior_consolidation["routine_reconciliation"]
    monitor = reader.read(project, MONITOR_REL, prior_reconciliation["historical_monitoring_sha256"])
    require(set(monitor.get("sealed_evidence", {})) == {f"routine_batch_{i:03d}" for i in range(1, 6)},
            "all_five_prefix_seals_required")
    campaign_plan_sha = prior_reconciliation["campaign_plan_sha256"]
    campaign_plan = reader.read(campaign, "campaign_plan.json", campaign_plan_sha)
    anchor = reader.read(campaign, "campaign_plan_anchor.json")
    require(anchor == {"sha256": digest(campaign_plan)}, "campaign_plan_anchor_changed")
    metadata = campaign_plan["metadata"]
    queue = metadata["full_queue_ids"]
    require(_unique(queue, "unique_routine_queue_required") == routine and digest(queue) == metadata["queue_sha256"]
            and metadata["inventory_metadata_sha256"] == digest(inventory_rows)
            and set(metadata["prior_development_ids"]) == prior and metadata["group_count"] == 87
            and metadata["group_participants"] == 20 and metadata["queue_offset"] == 90,
            "complete_sealed_campaign_queue_required")
    routine_rows = []
    prefix_slices = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 90)]
    for number, (start, end) in enumerate(prefix_slices, 1):
        batch = f"routine_batch_{number:03d}"
        history = _prefix_history(reader, project, monitor, number)
        pin = monitor["sealed_evidence"][batch]
        require([x["src_subject_id"] for x in history["plan"]["selected"]] == queue[start:end]
                and digest(history["plan"]["selected"]) == pin["selection_sha256"]
                and history["checkpoint"]["contract_sha256"] == pin["contract_sha256"],
                "prefix_queue_or_contract_changed")
        routine_rows.extend(collect_contract(history["plan"], history["checkpoint"], inventory, prior,
            "routine", batch, history["record"]))
    require(summarize(routine_rows) == {"participants": 90, "eligible_knees": 128,
        "accepted": 123, "excluded": 5, "pending": 0}, "prefix_denominators_changed")
    expected_terminal_files = {f"routine_batch_{n:03d}.json" for n in range(6, 93)}
    require({p.name for p in safe(campaign, "terminal").glob("*.json")} == expected_terminal_files,
            "exact_contiguous_87_campaign_terminals_required")
    previous_sha = None
    for ordinal, number in enumerate(range(6, 93)):
        batch = f"routine_batch_{number:03d}"
        terminal_rel = f"terminal/{batch}.json"
        terminal = reader.read(campaign, terminal_rel,
            prior_reconciliation["final_terminal92_sha256"] if number == 92 else None)
        require(terminal.get("version") == "frozen_routine_cohort_campaign_v1.0"
                and terminal.get("campaign_id") == "routine_cohort_campaign_001"
                and terminal.get("batch_id") == batch and terminal.get("ordinal") == ordinal
                and terminal.get("previous_terminal_sha256") == previous_sha
                and terminal.get("all_dispositions_terminal") is True
                and terminal.get("scientific_components_unchanged") is True, "sealed_campaign_terminal_chain_changed")
        base = f"groups/{batch}"
        files = terminal["files_sha256"]
        plan = reader.read(campaign, base + "/workspace/plan.json", files["workspace/plan.json"])
        checkpoint_rel = f"workspace/checkpoints/{terminal['contract_sha256']}.json"
        checked = reader.read(campaign, base + "/" + checkpoint_rel, files[checkpoint_rel])
        record_rel = terminal["execution_record_relative"]
        record = reader.read(campaign, base + "/" + record_rel, files[record_rel])
        reader.read(campaign, base + "/workspace/reference_seal.json", files["workspace/reference_seal.json"])
        reader.read(campaign, base + "/workspace/qc_review_seal.json", files["workspace/qc_review_seal.json"])
        journal = reader.read(campaign, f"selections/{batch}.json", terminal["selection_journal_sha256"])
        start = 90 + ordinal * 20
        expected_ids = queue[start:start + 20]
        require([x["src_subject_id"] for x in plan["selected"]] == expected_ids
                and digest(plan["selected"]) == terminal["selection_sha256"]
                and terminal["queue_start"] == start and terminal["queue_end"] == start + len(expected_ids)
                and journal.get("selected_metadata_sha256") == terminal["selection_sha256"]
                and journal.get("batch_id") == batch and journal.get("ordinal") == ordinal
                and journal.get("campaign_id") == "routine_cohort_campaign_001"
                and journal.get("plan_sha256") == files["workspace/plan.json"]
                and journal.get("previous_terminal_sha256") == previous_sha,
                "sealed_campaign_selection_journal_or_queue_changed")
        rows = collect_contract(plan, checked, inventory, prior, "routine", batch, record)
        require(all(terminal["counts"].get(k) == v for k, v in summarize(rows).items() if k != "pending"),
                "campaign_terminal_counts_changed")
        routine_rows.extend(rows)
        previous_sha = reader.pins[str(safe(campaign, terminal_rel))]
    require(summarize(routine_rows) == {"participants": 1826, "eligible_knees": 2646,
        "accepted": 2064, "excluded": 582, "pending": 0}, "complete_routine_counts_changed")
    rows = routine_rows + historical_rows + development_rows
    expected_pairs = {(sid, side) for sid, item in inventory.items() for side in eligible_sides(item)}
    pairs = [(r["src_subject_id"], r["side"]) for r in rows]
    require(len(pairs) == len(set(pairs)) and set(pairs) == expected_pairs, "complete_unique_2778_eligible_dispositions_required")
    require(all(r["role"] == ("development" if r["src_subject_id"] in prior else "routine") for r in rows),
            "permanent_development_role_changed")
    rows.sort(key=lambda x: (x["src_subject_id"], x["side"]))
    summary = {"all_inventory": summarize(rows), "routine": summarize(routine_rows),
        "development": summarize(historical_rows + development_rows),
        "historical_ineligible_technical_rows_excluded_from_analytic_denominator": extra,
        "sealed_routine_groups": 92, "native_payloads_opened": False,
        "outcomes_consulted": False, "decisions_regraded": False}
    require(summary["all_inventory"] == {"participants": 1916, "eligible_knees": 2778,
        "accepted": 2161, "excluded": 617, "pending": 0}, "global_terminal_counts_changed")
    return {"version": VERSION, "rows": rows, "pins": dict(sorted(reader.pins.items())),
        "inventory": inventory_rows, "roles": {"development_ids": sorted(prior), "routine_ids": sorted(routine)},
        "summary": summary, "private_rows_for_internal_join_only": True}
