"""Validate a prespecified training recipe without creating or fitting a model.

This module is a configuration and provenance boundary. It does not import Torch,
construct an optimizer, download a checkpoint, open an archive, or authorize
training. Fold checks operate on supplied development rows only. Provenance
checks require a separately authenticated artifact record; metadata alone cannot
prove that a checkpoint was trained on the stated participants.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path


VERSION = "registered_training_plan_v1.0"
PLAN_CANONICAL_SHA256 = "55d5c733b71502e971f1b58a66f22f9a8601c3c633e184499770f48f5b4c3b51"
REGISTRY_CANONICAL_SHA256 = "de26988f0d9296e902c0c5c18346c940a03a54c60193b8f882f08b67935a4e48"
_ROW_REQUIRED = {"src_subject_id", "side", "preprocessing_role", "outer_block", "cv_fold",
                 "AGEYEARS", "SEX", "BMI"}
_ROW_ALLOWED = _ROW_REQUIRED | {"PROGRESION_48M", "tensor"}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("finite_json_training_configuration_required") from exc


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_configuration_json_key")
        result[key] = value
    return result


def _read_json(path):
    source = Path(path)
    _require(source.is_file() and 0 < source.stat().st_size <= 128 * 1024,
             "bounded_configuration_file_required")
    try:
        return json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("valid_training_configuration_json_required") from exc


def registered_model_ids(registry):
    """Return all candidate IDs in the governing registry, preserving order."""
    groups = (registry["clinical"]["candidates"], registry["radiographic"]["mandatory_baselines"],
              registry["radiographic"]["modern_challengers"], registry["radiographic"]["complementary"],
              registry["multimodal"]["primary_candidates"], registry["multimodal"]["complementary"])
    return tuple(item["id"] for group in groups for item in group)


def validate_plan(plan, registry):
    """Reject silent overrides; a changed recipe requires a reviewed new version.

Pins cover canonical JSON rather than whitespace or line endings. These pins
freeze the proposed recipe, not authorization to run it. Eligibility evidence
belongs in a separate run record and must never be inserted as automatic approval.
"""
    _require(isinstance(plan, dict) and isinstance(registry, dict), "configuration_objects_required")
    _require(_digest(registry) == REGISTRY_CANONICAL_SHA256, "governing_model_registry_changed")
    _require(_digest(plan) == PLAN_CANONICAL_SHA256, "prespecified_training_recipe_changed")
    _require(plan["version"] == VERSION and plan["training_authorized"] is False
             and plan["reserved_test_access_authorized"] is False, "preparation_only_recipe_required")
    configured = (list(plan["clinical_models"]) + plan["radiographic_models"]["mandatory"]
                  + plan["radiographic_models"]["conditional_challengers"]
                  + plan["radiographic_models"]["complementary_disabled"]
                  + plan["multimodal_models"]["primary"]
                  + plan["multimodal_models"]["complementary_disabled"])
    registered = registered_model_ids(registry)
    _require(len(configured) == len(set(configured)) == len(registered)
             and set(configured) == set(registered), "exact_registered_model_coverage_required")
    return {"version": VERSION, "status": "configuration_verified_training_not_authorized",
            "registered_candidates": len(registered), "training_authorized": False,
            "reserved_test_access_authorized": False, "plan_canonical_sha256": _digest(plan),
            "registry_canonical_sha256": _digest(registry)}


def load_training_plan(plan_path, registry_path):
    plan, registry = _read_json(plan_path), _read_json(registry_path)
    validate_plan(plan, registry)
    return deepcopy(plan), deepcopy(registry)


def _fold_context(rows, validation_fold):
    _require(type(validation_fold) is int and validation_fold in range(5), "integer_validation_fold_required")
    _require(isinstance(rows, list) and 0 < len(rows) <= 5000, "bounded_development_rows_required")
    seen, people, folds = set(), {}, set()
    for row in rows:
        _require(isinstance(row, dict) and _ROW_REQUIRED <= set(row) <= _ROW_ALLOWED,
                 "exact_development_row_fields_required")
        sid, side = row["src_subject_id"], row["side"]
        _require(isinstance(sid, str) and sid and sid.strip() == sid and side in {"LEFT", "RIGHT"},
                 "valid_participant_and_side_required")
        _require((sid, side) not in seen, "duplicate_development_knee")
        seen.add((sid, side))
        block, fold, role = row["outer_block"], row["cv_fold"], row["preprocessing_role"]
        _require(type(block) is int and block in {1, 2, 3, 4}
                 and type(fold) is int and fold in range(5) and role in {"routine", "development"},
                 "development_only_assignments_required")
        _require(sid not in people or people[sid] == (block, fold, role),
                 "participant_crosses_blocks_folds_or_roles")
        people[sid] = block, fold, role
        folds.add(fold)
    _require(folds == set(range(5)), "all_five_development_folds_required")
    training = sorted((r for r in rows if r["cv_fold"] != validation_fold),
                      key=lambda r: (r["src_subject_id"], r["side"]))
    validation = sorted((r for r in rows if r["cv_fold"] == validation_fold),
                        key=lambda r: (r["src_subject_id"], r["side"]))
    identities = [[r["src_subject_id"], r["side"]] for r in training]
    return training, validation, _digest(identities)


def training_fold_class_weight(rows, validation_fold):
    """Compute N_negative/N_positive from training knees, never validation labels."""
    training, _, identity_hash = _fold_context(rows, validation_fold)
    _require(all("PROGRESION_48M" in row for row in training), "training_targets_required")
    targets = [row["PROGRESION_48M"] for row in training]
    _require(all(type(value) is int and value in {0, 1} for value in targets),
             "binary_training_targets_required")
    positive = sum(targets)
    negative = len(targets) - positive
    _require(positive > 0 and negative > 0, "both_training_target_classes_required")
    return {"validation_fold": validation_fold, "training_knees": len(targets),
            "positive_count": positive, "negative_count": negative,
            "positive_weight": negative / positive, "train_identity_sha256": identity_hash}


def validate_fold_weight_provenance(rows, validation_fold, provenance):
    """Check an authenticated same-fold checkpoint record before fusion reuse."""
    training, validation, identity_hash = _fold_context(rows, validation_fold)
    required = {"fit_scope", "validation_fold", "train_identity_sha256", "trained_participants",
                "reserved_test_accessed"}
    _require(isinstance(provenance, dict) and set(provenance) == required,
             "exact_fold_weight_provenance_required")
    _require(provenance["fit_scope"] == "development_training_fold"
             and type(provenance["validation_fold"]) is int
             and provenance["validation_fold"] == validation_fold
             and provenance["reserved_test_accessed"] is False,
             "same_fold_development_only_weight_provenance_required")
    supplied = provenance["trained_participants"]
    _require(isinstance(supplied, list) and all(isinstance(sid, str) for sid in supplied)
             and len(supplied) == len(set(supplied)), "unique_trained_participant_list_required")
    _require(not set(supplied) & {r["src_subject_id"] for r in validation},
             "held_out_participant_in_weight_provenance")
    _require(set(supplied) == {r["src_subject_id"] for r in training}
             and provenance["train_identity_sha256"] == identity_hash,
             "exact_training_fold_weight_identity_required")
    return True


def select_modern_challengers(reports, plan, registry):
    """Rank five-fold means under the approved AP near-tie rule; fit nothing.

Reports contain one already-computed metric row per held-out participant fold.
The unweighted means, not a pooled AP, drive selection. Each pick anchors its
near-tie band to the highest remaining mean AP, avoiding a nontransitive pairwise
comparison. This validator cannot authenticate checkpoints or row provenance.
"""
    validate_plan(plan, registry)
    ids = plan["radiographic_models"]["conditional_challengers"]
    _require(isinstance(reports, list) and len(reports) <= len(ids), "bounded_challenger_reports_required")
    fields = {"model_id", "scope", "aggregation", "fold_metrics", "eligibility_gates"}
    seen, eligible = set(), []
    for report in reports:
        _require(isinstance(report, dict) and set(report) == fields, "exact_oof_report_fields_required")
        model_id = report["model_id"]
        _require(model_id in ids and model_id not in seen, "unique_registered_challenger_required")
        seen.add(model_id)
        _require(report["scope"] == "development_only"
                 and report["aggregation"] == "mean_of_five_participant_held_out_fold_metrics",
                 "complete_grouped_development_oof_required")
        gates = report["eligibility_gates"]
        _require(isinstance(gates, dict) and set(gates) == set(registry["eligibility_gates"])
                 and all(type(value) is bool for value in gates.values()), "explicit_registered_gate_results_required")
        metrics = ("average_precision_score", "roc_auc", "brier")
        fold_metrics = report["fold_metrics"]
        _require(isinstance(fold_metrics, list) and len(fold_metrics) == 5,
                 "exact_five_fold_metric_rows_required")
        fold_ids = []
        for row in fold_metrics:
            _require(isinstance(row, dict) and set(row) == {"validation_fold", *metrics}
                     and type(row["validation_fold"]) is int and row["validation_fold"] in range(5),
                     "exact_five_fold_metric_rows_required")
            fold_ids.append(row["validation_fold"])
            for metric in metrics:
                value = row[metric]
                _require(type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1,
                         "finite_unit_interval_oof_metric_required")
        _require(set(fold_ids) == set(range(5)), "unique_complete_five_fold_metrics_required")
        # Decimal respects the strict 0.01 boundary in the serialized scores;
        # no rounding of per-fold metrics or near-tie cutoff is introduced.
        means = {metric: sum(Decimal(str(row[metric])) for row in fold_metrics) / Decimal(5)
                 for metric in metrics}
        if all(gates.values()):
            eligible.append({"model_id": model_id, **means})
    selected = []
    cutoff = Decimal(str(plan["selection"]["near_tie_ap_difference_below"]))
    maximum = plan["selection"]["maximum_promoted_modern_challengers"]
    while eligible and len(selected) < maximum:
        anchor = max(report["average_precision_score"] for report in eligible)
        band = [report for report in eligible if anchor - report["average_precision_score"] < cutoff]
        winner = min(band, key=lambda r: (-r["roc_auc"], r["brier"], ids.index(r["model_id"])))
        selected.append(winner["model_id"])
        eligible.remove(winner)
    return selected


def final_training_epochs(fold_best_epochs, plan, registry):
    """Fix future refit duration from five development best epochs; train nothing.

Call separately for one selected neural model, training stage and seed, using
authenticated development logs. Five integer epochs have an integer median, so
half-up rounding is an identity and cannot introduce an extra epoch.
"""
    validate_plan(plan, registry)
    _require(isinstance(fold_best_epochs, list) and len(fold_best_epochs) == 5,
             "five_development_best_epochs_required")
    maximum = plan["future_final_refit"]["maximum_epochs"]
    folds, epochs = [], []
    for record in fold_best_epochs:
        _require(isinstance(record, dict)
                 and set(record) == {"validation_fold", "best_epoch", "scope"}
                 and record["scope"] == "development_only"
                 and type(record["validation_fold"]) is int and record["validation_fold"] in range(5)
                 and type(record["best_epoch"]) is int and 1 <= record["best_epoch"] <= maximum,
                 "valid_development_fold_best_epoch_required")
        folds.append(record["validation_fold"])
        epochs.append(record["best_epoch"])
    _require(set(folds) == set(range(5)), "unique_complete_five_best_epochs_required")
    return sorted(epochs)[2]
