"""Approved additive model release; historical notebook 32 remains immutable.

No training authority is granted. All new entrypoints resolve this descriptor,
never a mutable default registry or fallback to the older model roster.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import math
from pathlib import Path

from knee import training_plan as historical
from knee.training_plan import _read_json, _digest, _require, registered_model_ids
from knee.training_plan import training_fold_class_weight, validate_fold_weight_provenance

VERSION = "registered_training_plan_v1.1"
ADDED_IDS = ("image_biomedclip", "image_mambaout_tiny")
RELEASE_PATH = "configs/modeling_release_v1_1.json"
PINS = {
    "configs/model_registry_v1_1.json": "8d560a9eac4433c4040d2a0df8b45abaf1704ca0feaa7b2eb2ccc438e934d2f5",
    "configs/training_plan_v1_1.json": "f973586aade403c5bf4107ccb4a2a1fc29e3a6273de2c27bfafa7ef4e6ca0081",
    "configs/pretrained_weights_v1_1.json": "a53007552c61ca06c2ef734792c0d01656ca7f5db988c521dcde6bbb72ae7593",
    "configs/governance/final_model_amendment_v1.json": "c4c21908458644bc86806a191f78ef2c308bb6912b9a84f50a2a514d40781568",
    RELEASE_PATH: "21524cbbcd191a8195555d2315b1d469ee6ad78c03c5b699bdd48e68a33855ba",
}


def validate_plan(plan, registry):
    _require(_digest(plan) == PINS["configs/training_plan_v1_1.json"], "final_training_plan_changed")
    _require(_digest(registry) == PINS["configs/model_registry_v1_1.json"], "final_model_registry_changed")
    _require(plan["training_authorized"] is False and plan["reserved_test_access_authorized"] is False,
             "preparation_only_release_required")
    old_registry, old_plan = deepcopy(registry), deepcopy(plan)
    old_registry["registry_version"] = "1.0"
    old_registry["governing_change"] = "MCR-2026-002"
    old_registry.pop("architecture_search_closed")
    added = old_registry["radiographic"]["modern_challengers"][-2:]
    _require(tuple(item["id"] for item in added) == ADDED_IDS, "exact_two_approved_additions_required")
    old_registry["radiographic"]["modern_challengers"] = old_registry["radiographic"]["modern_challengers"][:-2]
    old_plan["version"] = historical.VERSION
    old_plan["registry_version"] = "1.0"
    old_plan["governing_change"] = "MCR-2026-002"
    _require(tuple(old_plan["radiographic_models"]["conditional_challengers"][-2:]) == ADDED_IDS,
             "exact_two_approved_additions_required")
    old_plan["radiographic_models"]["conditional_challengers"] = old_plan["radiographic_models"]["conditional_challengers"][:-2]
    old_plan["readiness"]["new_architectures_or_predictors"] = False
    # JSON serializers may spell an integral value 1 rather than 1.0. Compare
    # the projected recipe semantically, while authenticating BOTH documents
    # independently through their immutable full canonical digest pins.
    baseline_plan = _read_json(Path(__file__).resolve().parents[2] / "configs/training_plan_v1.json")
    historical.validate_plan(baseline_plan, old_registry)
    _require(old_plan == baseline_plan, "unapproved_training_recipe_change")
    _require(len(registered_model_ids(registry)) == 14 and registry["architecture_search_closed"] is True,
             "closed_exact_final_roster_required")
    return {"version": VERSION, "status": "configuration_verified_training_not_authorized",
            "registered_candidates": 14, "training_authorized": False,
            "reserved_test_access_authorized": False, "plan_canonical_sha256": _digest(plan),
            "registry_canonical_sha256": _digest(registry)}


def load_release(repository_root):
    """Fail closed on any missing, altered or redirected release component."""
    from knee.pretrained_verification import load_catalog as load_historical_catalog
    root = Path(repository_root).resolve()
    values = {}
    for name, expected in PINS.items():
        path = root / name
        _require(not path.is_symlink() and path.resolve().is_relative_to(root), "release_path_escape")
        value = _read_json(path)
        _require(_digest(value) == expected, "approved_release_component_changed:" + name)
        values[name] = value
    release = values[RELEASE_PATH]
    catalog = values[release["weights_catalog"]]
    plan, registry = values[release["training_plan"]], values[release["registry"]]
    validate_plan(plan, registry)
    old_catalog, _, _ = load_historical_catalog(root)
    projected = deepcopy(catalog)
    projected["version"] = old_catalog["version"]
    projected["registry_version"] = old_catalog["registry_version"]
    projected["governing_change"] = old_catalog["governing_change"]
    projected["reviewed_on"] = old_catalog["reviewed_on"]
    projected["input"].pop("normalization_selection")
    for model_id in ADDED_IDS:
        projected["models"].pop(model_id)
    _require(projected == old_catalog, "historical_checkpoint_contract_changed")
    expected_image_ids = {item["id"] for group in ("mandatory_baselines", "modern_challengers")
                          for item in registry["radiographic"][group]}
    _require(len(expected_image_ids) == 7 and set(catalog["models"]) == expected_image_ids,
             "exact_seven_image_encoders_required")
    _require(set(catalog["models"]) | set(catalog["models_without_external_pretrained_checkpoint"])
             == set(registered_model_ids(registry)), "exact_final_model_coverage_required")
    return release, catalog, plan, registry


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
