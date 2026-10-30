"""One reproducible participant-level split, with permanent development exclusions.

Only fixed outcome stratification and class-presence checks are performed here.
There is no fitting, imputation, model selection, or test-set prediction.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math


VERSION = "participant_partitions_v1.0"
POLICY = {
    "version": VERSION,
    "seed": 2026,
    "outer_blocks": 5,
    "reserved_test_block": 0,
    "test_participant_fraction_approx": 0.2,
    "development_cv_folds": 5,
    "group_unit": "participant",
    "stratum": "any_accepted_knee_with_PROGRESION_48M_1_per_participant",
    "algorithm": "seeded_sha256_rank_with_continuous_round_robin_class_quotas",
    "permanent_preprocessing_development_reservations": "never_test_even_if_no_accepted_knee",
    "outer_non_test_blocks": [1, 2, 3, 4],
    "cv_assignment": "all_non_test_participants_including_preprocessing_development",
    "class_presence_gate": "both_classes_in_test_each_cv_validation_and_each_cv_training",
    "retry_other_seeds_or_assignments": False,
    "fit_preprocessing": False,
    "model_specific_normalization": "pending_encoder_integration",
    "same_assignment_for": ["clinical", "radiographic", "multimodal"],
    "historical_processing_block_reused_as_partition": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(row):
    sid, side = row.get("src_subject_id"), row.get("side")
    require(isinstance(sid, str) and bool(sid) and sid == sid.strip()
            and side in {"RIGHT", "LEFT"}, "invalid_participant_side")
    return sid, side


def _indexed(rows):
    require(isinstance(rows, list), "rows_list_required")
    result = {}
    for row in rows:
        require(isinstance(row, dict), "row_object_required")
        pair = identity(row)
        require(pair not in result, "duplicate_participant_side")
        result[pair] = row
    return result


def _rank(stage, stratum, sid):
    return hashlib.sha256(canonical([POLICY["seed"], stage, stratum, sid])).hexdigest(), sid


def _quotas(groups, k):
    """Stratified class quotas whose combined group counts differ by at most one."""
    result, offset = {}, 0
    for label in (0, 1):
        counts = [0] * k
        for index in range(len(groups[label])):
            counts[(offset + index) % k] += 1
        result[label] = counts
        offset = (offset + len(groups[label])) % k
    return result


def _summary(sids, members, labels):
    keys = [pair for pair in members if pair[0] in sids]
    positives = sum(labels[pair]["PROGRESION_48M"] for pair in keys)
    participant_positive = {pair[0] for pair in keys if labels[pair]["PROGRESION_48M"] == 1}
    return {"participants": len(sids), "knees": len(keys),
            "positive_knees": positives, "negative_knees": len(keys) - positives,
            "participants_with_positive_knee": len(participant_positive)}


def build_partitions(membership, labels, policy=POLICY):
    require(canonical(policy) == canonical(POLICY), "fixed_partition_policy_required")
    members, outcomes = _indexed(membership["rows"]), _indexed(labels["rows"])
    require(bool(members) and set(members) == set(outcomes), "exact_cohort_label_join_required")
    reservations = membership["development_reservations"]
    require(isinstance(reservations, list) and all(isinstance(v, str) and v and v.strip() == v
            for v in reservations) and len(set(reservations)) == len(reservations),
            "unique_permanent_development_reservations_required")
    reserved = set(reservations)
    people, person_labels = {}, {}
    for pair in sorted(members):
        member, outcome = members[pair], outcomes[pair]
        y = outcome.get("PROGRESION_48M")
        require(type(y) is int and y in {0, 1}, "binary_integer_outcome_required")
        sid = pair[0]
        expected_role = "development" if sid in reserved else "routine"
        require(member.get("role") == expected_role, "permanent_preprocessing_role_changed")
        clinical = member.get("clinical")
        require(isinstance(clinical, dict) and set(clinical) == {"AGEYEARS", "SEX", "BMI"},
                "exact_main_clinical_fields_required")
        require(type(clinical["AGEYEARS"]) is int and clinical["AGEYEARS"] > 0
                and clinical["SEX"] in {"F", "M"}
                and (clinical["BMI"] is None or (type(clinical["BMI"]) in {int, float}
                     and math.isfinite(clinical["BMI"]) and clinical["BMI"] > 0)),
                "invalid_main_clinical_value")
        if sid in people:
            require(people[sid] == (expected_role, clinical), "within_participant_baseline_changed")
        else:
            people[sid] = (expected_role, clinical)
        person_labels[sid] = max(y, person_labels.get(sid, 0))
    groups = {label: sorted(sid for sid in people if person_labels[sid] == label) for label in (0, 1)}
    require(all(len(groups[label]) >= 5 for label in (0, 1)), "insufficient_stratum_groups_no_resampling")
    quotas, assignment = _quotas(groups, 5), {}
    for label in (0, 1):
        eligible = sorted((sid for sid in groups[label] if sid not in reserved),
                          key=lambda sid: _rank("outer_test", label, sid))
        count = quotas[label][0]
        require(len(eligible) >= count, "test_quota_conflicts_with_permanent_development_no_resampling")
        test = set(eligible[:count])
        for sid in test:
            assignment[sid] = {"outer_block": 0, "cv_fold": None}
        remaining = sorted((sid for sid in groups[label] if sid not in test),
                           key=lambda sid: _rank("outer_development", label, sid))
        offset = 0
        for block in (1, 2, 3, 4):
            for sid in remaining[offset:offset + quotas[label][block]]:
                assignment[sid] = {"outer_block": block, "cv_fold": None}
            offset += quotas[label][block]
        require(offset == len(remaining), "outer_quota_accounting_failed")
    dev_groups = {label: [sid for sid in groups[label] if assignment[sid]["outer_block"] != 0]
                  for label in (0, 1)}
    require(all(len(dev_groups[label]) >= 5 for label in (0, 1)), "insufficient_cv_stratum_groups_no_resampling")
    cv_quotas = _quotas(dev_groups, 5)
    for label in (0, 1):
        ranked = sorted(dev_groups[label], key=lambda sid: _rank("inner_cv", label, sid))
        offset = 0
        for fold in range(5):
            for sid in ranked[offset:offset + cv_quotas[label][fold]]:
                assignment[sid]["cv_fold"] = fold
            offset += cv_quotas[label][fold]
        require(offset == len(ranked), "cv_quota_accounting_failed")
    require(set(assignment) == set(people)
            and not {sid for sid in reserved if sid in assignment and assignment[sid]["outer_block"] == 0},
            "participant_coverage_or_reservation_failed")
    outer = {str(b): _summary({sid for sid in people if assignment[sid]["outer_block"] == b}, members, outcomes)
             for b in range(5)}
    development = {sid for sid in people if assignment[sid]["outer_block"] != 0}
    folds = {}
    for fold in range(5):
        valid = {sid for sid in development if assignment[sid]["cv_fold"] == fold}
        folds[str(fold)] = {"training": _summary(development - valid, members, outcomes),
                            "validation": _summary(valid, members, outcomes)}
    require(all(v["positive_knees"] > 0 and v["negative_knees"] > 0
                for v in [outer["0"]] + [item[role] for item in folds.values()
                                         for role in ("training", "validation")]),
            "class_presence_failed_no_resampling")
    rows = [{"src_subject_id": sid, "preprocessing_role": people[sid][0], **assignment[sid]}
            for sid in sorted(people)]
    return {"version": VERSION, "policy": json.loads(canonical(POLICY)),
        "participants": rows, "development_reservations": sorted(reserved),
        "summary": {"total": _summary(set(people), members, outcomes), "outer_blocks": outer,
            "development": _summary(development, members, outcomes), "development_cv": folds,
            "permanent_development_reservations": len(reserved),
            "accepted_preprocessing_development_participants": len(set(people) & reserved),
            "reservations_without_accepted_knees": len(reserved - set(people)),
            "test_participant_fraction": outer["0"]["participants"] / len(people),
            "both_classes_present_in_all_required_splits": True,
            "no_participant_crosses_partitions": True,
            "all_three_main_models_share_assignments": True},
        "outcomes_access_scope": "fixed_stratification_and_class_presence_only",
        "imputation_performed": False, "training_executed": False,
        "reserved_test_used_for_model_development": False,
        "reserved_test_predictive_evaluation_opened": False,
        "external_pretraining_disjointness_verified": False,
        "statistical_precision_or_power_certified": False}
