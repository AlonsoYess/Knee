"""Development-only baseline comparators and training-fold preprocessing.

The historical three-feature package is immutable. Complementary inputs travel
in a separately sealed, baseline-only development sidecar; no outcome-bearing
source workbook or reserved-test archive is opened here. Its digest must also
be bound by the caller's execution request, rather than trusted as a signature.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

import numpy as np

from knee import clinical_fold_inputs as main

VERSION = "clinical_scenario_inputs_v1.0"
SIDECAR_VERSION = "development_baseline_complements_v1.0"
SCENARIOS = {
    "clinical_main": ("AGEYEARS", "SEX", "BMI"),
    "clinical_plus_baseline_kl": ("AGEYEARS", "SEX", "BMI", "KL_BASE"),
    "clinical_plus_baseline_surgery_and_womac":
        ("AGEYEARS", "SEX", "BMI", "CIRUGIA_PREVIA", "WOMAC_TOTAL"),
}
EXTRA_FIELDS = {"KL_BASE", "CIRUGIA_PREVIA", "WOMAC_TOTAL"}
BASELINE_FIELDS = {"src_subject_id", "side", "AGEYEARS", "SEX", "BMI"} | EXTRA_FIELDS
SOURCE_COLUMNS = {
    "src_subject_id": "SRC_SUBJECT_ID", "side": "SIDE",
    "AGEYEARS": "AGEYEARS", "SEX": "SEX", "BMI": "BMI", "KL_BASE": "KL_BASE",
    "CIRUGIA_PREVIA": "CIRUGIA_PREVIA", "WOMAC_TOTAL": "WOMAC_TOTAL",
}
SOURCE_SIDE_ENCODING = {"1": "RIGHT", "2": "LEFT"}
POLICY = {
    "baseline_visit": "V00",
    "statistics_unit": "training_knee_rows",
    "main_preprocessing": main.VERSION,
    "KL_BASE": "observed_ordinal_2_or_3_standardized_training_population_ddof_0",
    "CIRUGIA_PREVIA": "binary_0_1_training_mode_imputation_ties_choose_0_no_scaling",
    "WOMAC_TOTAL": "0_to_96_training_median_imputation_population_standardization_ddof_0",
    "constant_scale": 1.0,
    "missingness_indicators": False,
    "future_predictors_allowed": False,
}
_SIDECAR_FIELDS = {"version", "scope", "baseline_visit", "development_package_sha256",
    "development_context_sha256", "baseline_source_sha256", "source_columns", "rows", "record_sha256"}
_STATE_FIELDS = {"version", "scenario", "feature_order", "policy", "validation_fold",
    "train_count", "train_participants", "train_identity_sha256", "main_state",
    "extra_imputation", "extra_mean", "extra_scale", "record_sha256"}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identity(row):
    return row["src_subject_id"], row["side"]


def _number(value, low, high):
    return type(value) in {int, float} and math.isfinite(value) and low <= value <= high


def _extras(row):
    _require(type(row["KL_BASE"]) is int and row["KL_BASE"] in {2, 3},
             "observed_eligible_baseline_kl_required")
    _require(row["CIRUGIA_PREVIA"] is None or
             (type(row["CIRUGIA_PREVIA"]) is int and row["CIRUGIA_PREVIA"] in {0, 1}),
             "binary_or_missing_baseline_surgery_required")
    _require(row["WOMAC_TOTAL"] is None or _number(row["WOMAC_TOTAL"], 0, 96),
             "finite_or_missing_baseline_womac_0_96_required")


def _base_rows(rows, *, complements=False, all_folds=False):
    _require(isinstance(rows, list) and 0 < len(rows) <= 5000,
             "bounded_development_scenario_rows_required")
    for row in rows:
        _require(isinstance(row, dict) and main._ROW_REQUIRED <= set(row)
                 <= main._ROW_ALLOWED | EXTRA_FIELDS,
                 "closed_development_scenario_fields_required")
        present = EXTRA_FIELDS & set(row)
        _require(present in (set(), EXTRA_FIELDS) and (not complements or present == EXTRA_FIELDS),
                 "complete_baseline_complements_required")
        if present:
            _extras(row)
    # Explicitly select predictors/assignments only: never access a target value.
    base = [{name: row[name] for name in main._ROW_REQUIRED} for row in rows]
    main._rows(base, require_all_folds=all_folds)
    return base


def _development_context(rows):
    return main._digest(sorted(_base_rows(rows, all_folds=True), key=_identity))


def make_baseline_sidecar(baseline_rows, development_rows, package_sha256, source_sha256):
    """Seal an already-projected V00 development source, never a full cohort.

Input rows use the exact canonical fields in BASELINE_FIELDS. A projection with
unknown participants, duplicate sides, outcomes or future visits is rejected.
The three main predictors must match the authenticated package exactly.
"""
    _require(_sha(package_sha256) and _sha(source_sha256), "baseline_source_and_package_hashes_required")
    base = _base_rows(development_rows, all_folds=True)
    _require(isinstance(baseline_rows, list) and len(baseline_rows) == len(base),
             "exact_development_baseline_row_count_required")
    expected = {_identity(row): row for row in base}
    seen = set()
    for row in baseline_rows:
        _require(isinstance(row, dict) and set(row) == BASELINE_FIELDS,
                 "baseline_only_projected_source_fields_required")
        key = _identity(row)
        _require(key in expected and key not in seen, "exact_unique_development_baseline_keys_required")
        seen.add(key)
        _require(all(type(row[name]) is type(expected[key][name]) and row[name] == expected[key][name]
                     for name in ("AGEYEARS", "SEX", "BMI")), "main_predictors_changed_in_baseline_projection")
        _extras(row)
    record = {"version": SIDECAR_VERSION, "scope": "development_only", "baseline_visit": "V00",
        "development_package_sha256": package_sha256,
        "development_context_sha256": _development_context(development_rows),
        "baseline_source_sha256": source_sha256, "source_columns": deepcopy(SOURCE_COLUMNS),
        "rows": sorted(deepcopy(baseline_rows), key=_identity)}
    record["record_sha256"] = main._digest(record)
    return record


def join_baseline_sidecar(development_rows, sidecar, package_sha256, *, expected_sidecar_sha256=None):
    """Validate the separate seal, then join by participant AND side, retaining order."""
    _require(isinstance(sidecar, dict) and set(sidecar) == _SIDECAR_FIELDS,
             "exact_development_baseline_sidecar_required")
    _require(sidecar["version"] == SIDECAR_VERSION and sidecar["scope"] == "development_only"
             and sidecar["baseline_visit"] == "V00" and sidecar["source_columns"] == SOURCE_COLUMNS,
             "baseline_only_development_sidecar_required")
    _require(_sha(sidecar["record_sha256"]) and sidecar["record_sha256"] == main._digest(
        {key: value for key, value in sidecar.items() if key != "record_sha256"}), "baseline_sidecar_seal_changed")
    if expected_sidecar_sha256 is not None:
        _require(_sha(expected_sidecar_sha256) and sidecar["record_sha256"] == expected_sidecar_sha256,
                 "baseline_sidecar_differs_from_execution_binding")
    _require(sidecar["development_package_sha256"] == package_sha256,
             "baseline_sidecar_package_binding_changed")
    expected = make_baseline_sidecar(sidecar["rows"], development_rows, package_sha256,
                                    sidecar["baseline_source_sha256"])
    _require(expected == sidecar, "baseline_sidecar_development_context_changed")
    additions = {_identity(row): row for row in sidecar["rows"]}
    result = []
    for row in development_rows:
        # Do not silently replace preexisting complementary values.
        if EXTRA_FIELDS & set(row):
            _require(all(row[name] == additions[_identity(row)][name] for name in EXTRA_FIELDS),
                     "baseline_complement_already_present_but_different")
        result.append({**row, **{name: additions[_identity(row)][name] for name in EXTRA_FIELDS}})
    return result


def fit_scenario_fold(rows, validation_fold, scenario="clinical_main"):
    """Fit preprocessing only, with no target read or estimator construction."""
    _require(scenario in SCENARIOS, "registered_clinical_scenario_required")
    base = _base_rows(rows, complements=scenario != "clinical_main", all_folds=True)
    primary = main.fit_clinical_fold(base, validation_fold)
    training = sorted((r for r in rows if r["cv_fold"] != validation_fold), key=_identity)
    imputation, means, scales = {}, {}, {}
    if scenario == "clinical_plus_baseline_surgery_and_womac":
        observed = [r["CIRUGIA_PREVIA"] for r in training if r["CIRUGIA_PREVIA"] is not None]
        _require(bool(observed), "training_fold_requires_observed_surgery")
        imputation["CIRUGIA_PREVIA"] = int(sum(observed) > len(observed) - sum(observed))
        observed = [r["WOMAC_TOTAL"] for r in training if r["WOMAC_TOTAL"] is not None]
        _require(bool(observed), "training_fold_requires_observed_womac")
        imputation["WOMAC_TOTAL"] = float(np.median(np.asarray(observed, dtype=np.float64)))
    continuous = {"clinical_main": (), "clinical_plus_baseline_kl": ("KL_BASE",),
                  "clinical_plus_baseline_surgery_and_womac": ("WOMAC_TOTAL",)}[scenario]
    for name in continuous:
        values = np.asarray([imputation[name] if r[name] is None else r[name] for r in training], dtype=np.float64)
        means[name] = float(np.mean(values, dtype=np.float64))
        scale = float(np.std(values, ddof=0, dtype=np.float64))
        scales[name] = 1.0 if scale == 0 else scale
    state = {"version": VERSION, "scenario": scenario, "feature_order": list(SCENARIOS[scenario]),
        "policy": deepcopy(POLICY), "main_state": primary,
        **{name: primary[name] for name in ("validation_fold", "train_count", "train_participants", "train_identity_sha256")},
        "extra_imputation": imputation, "extra_mean": means, "extra_scale": scales}
    state["record_sha256"] = main._digest(state)
    validate_scenario_state(state)
    return state


def validate_scenario_state(state):
    _require(isinstance(state, dict) and set(state) == _STATE_FIELDS and state["version"] == VERSION,
             "exact_clinical_scenario_state_required")
    scenario = state["scenario"]
    _require(scenario in SCENARIOS and state["feature_order"] == list(SCENARIOS[scenario])
             and main._canonical(state["policy"]) == main._canonical(POLICY),
             "registered_scenario_policy_required")
    main._validate_state(state["main_state"])
    _require(all(type(state[name]) is type(state["main_state"][name])
                 and state[name] == state["main_state"][name] for name in
             ("validation_fold", "train_count", "train_participants", "train_identity_sha256")),
             "main_and_complementary_training_identity_mismatch")
    continuous = {"clinical_main": set(), "clinical_plus_baseline_kl": {"KL_BASE"},
                  "clinical_plus_baseline_surgery_and_womac": {"WOMAC_TOTAL"}}[scenario]
    imputed = {"CIRUGIA_PREVIA", "WOMAC_TOTAL"} if "WOMAC_TOTAL" in continuous else set()
    _require(isinstance(state["extra_imputation"], dict) and set(state["extra_imputation"]) == imputed,
             "exact_scenario_imputation_required")
    if imputed:
        surgery, womac = state["extra_imputation"]["CIRUGIA_PREVIA"], state["extra_imputation"]["WOMAC_TOTAL"]
        _require(type(surgery) is int and surgery in {0, 1} and _number(womac, 0, 96),
                 "valid_fitted_baseline_imputation_required")
    for name in ("extra_mean", "extra_scale"):
        _require(isinstance(state[name], dict) and set(state[name]) == continuous,
                 "exact_scenario_statistics_required")
    for name in continuous:
        lower, upper = (2, 3) if name == "KL_BASE" else (0, 96)
        _require(_number(state["extra_mean"][name], lower, upper)
                 and main._positive_number(state["extra_scale"][name]), "finite_scenario_statistics_required")
    _require(_sha(state["record_sha256"]) and state["record_sha256"] == main._digest(
        {key: value for key, value in state.items() if key != "record_sha256"}), "clinical_scenario_state_changed")
    return True


def verify_scenario_fold_state(rows, state):
    """Authenticate fitted statistics against the exact training rows, not a self-hash."""
    validate_scenario_state(state)
    _require(fit_scenario_fold(rows, state["validation_fold"], state["scenario"]) == state,
             "scenario_state_not_fitted_on_exact_training_fold")
    return True


def transform_scenario_rows(rows, state):
    """Apply frozen statistics; preserve row order and all original values."""
    validate_scenario_state(state)
    scenario = state["scenario"]
    base = _base_rows(rows, complements=scenario != "clinical_main")
    primary = main.transform_clinical_rows(base, state["main_state"])
    if scenario == "clinical_main":
        return primary
    extra = []
    for row in rows:
        values = []
        for name in SCENARIOS[scenario][3:]:
            value = state["extra_imputation"][name] if row[name] is None else row[name]
            if name in state["extra_mean"]:
                value = (value - state["extra_mean"][name]) / state["extra_scale"][name]
            values.append(value)
        extra.append(values)
    result = np.ascontiguousarray(np.concatenate((primary, np.asarray(extra, dtype=np.float32)), axis=1))
    _require(result.shape == (len(rows), len(SCENARIOS[scenario])) and np.isfinite(result).all(),
             "finite_clinical_scenario_matrix_required")
    result.setflags(write=False)
    return result
