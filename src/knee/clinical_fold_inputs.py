"""Fit three clinical inputs on one training fold and apply them without refit.

Rows come from the sealed development package. No files, targets, neural model,
or reserved-test package are read by these functions.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np


VERSION = "clinical_fold_inputs_v1.0"
POLICY = {
    "feature_order": ["AGEYEARS", "SEX", "BMI"],
    "sex_encoding": {"F": 0, "M": 1},
    "bmi_imputation": "median_training_knee_rows_only",
    "continuous_scaling": "population_standardization_ddof_0_after_training_BMI_imputation",
    "constant_scale": 1.0,
    "statistics_unit": "training_knee_rows",
    "missingness_indicator": False,
}
_ROW_REQUIRED = {"src_subject_id", "side", "preprocessing_role", "outer_block", "cv_fold",
                 "AGEYEARS", "SEX", "BMI"}
_ROW_ALLOWED = _ROW_REQUIRED | {"PROGRESION_48M", "tensor"}
_STATE_FIELDS = {"version", "policy", "validation_fold", "train_count", "train_participants",
                 "train_identity_sha256", "BMI_median", "continuous_mean", "continuous_scale",
                 "record_sha256"}
_CONTINUOUS = ("AGEYEARS", "BMI")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _positive_number(value):
    return type(value) in {int, float} and math.isfinite(value) and value > 0


def _policy(policy):
    _require(isinstance(policy, dict) and _canonical(policy) == _canonical(POLICY),
             "fixed_three_feature_clinical_policy_required")


def _rows(rows, *, require_all_folds=False):
    _require(isinstance(rows, list) and bool(rows), "nonempty_development_clinical_rows_required")
    seen, people, folds = set(), {}, set()
    for row in rows:
        _require(isinstance(row, dict) and _ROW_REQUIRED <= set(row) <= _ROW_ALLOWED,
                 "exact_development_clinical_row_fields_required")
        sid, side = row["src_subject_id"], row["side"]
        _require(isinstance(sid, str) and sid and sid.strip() == sid
                 and side in {"RIGHT", "LEFT"}, "valid_clinical_participant_side_required")
        pair = sid, side
        _require(pair not in seen, "duplicate_clinical_participant_side")
        seen.add(pair)
        role, block, fold = row["preprocessing_role"], row["outer_block"], row["cv_fold"]
        _require(role in {"routine", "development"} and type(block) is int and block in {1, 2, 3, 4}
                 and type(fold) is int and fold in {0, 1, 2, 3, 4},
                 "development_only_clinical_rows_required")
        age, sex, bmi = row["AGEYEARS"], row["SEX"], row["BMI"]
        _require(type(age) is int and age > 0 and math.isfinite(age) and sex in {"F", "M"}
                 and (bmi is None or _positive_number(bmi)), "valid_baseline_clinical_values_required")
        signature = role, block, fold, age, sex, bmi
        _require(sid not in people or people[sid] == signature,
                 "within_participant_clinical_or_assignment_changed")
        people[sid] = signature
        folds.add(fold)
    _require(not require_all_folds or folds == set(range(5)), "all_five_development_folds_required")
    return rows


def fit_clinical_fold(rows, validation_fold, policy=None):
    """Derive BMI median and age/BMI statistics using only cv != validation_fold.

Statistics use accepted knee rows, matching the prediction unit. This operation
fits preprocessing only; it never reads PROGRESION_48M or creates a predictor.
"""
    _policy(POLICY if policy is None else policy)
    _require(type(validation_fold) is int and validation_fold in range(5),
             "integer_validation_fold_required")
    _rows(rows, require_all_folds=True)
    training = sorted((r for r in rows if r["cv_fold"] != validation_fold),
                      key=lambda r: (r["src_subject_id"], r["side"]))
    _require(bool(training), "nonempty_clinical_training_fold_required")
    observed = np.asarray([r["BMI"] for r in training if r["BMI"] is not None], dtype=np.float64)
    _require(observed.size > 0, "training_fold_requires_observed_bmi")
    median = float(np.median(observed))
    _require(_positive_number(median), "finite_training_bmi_median_required")
    values = np.asarray([[r["AGEYEARS"], median if r["BMI"] is None else r["BMI"]]
                         for r in training], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.mean(values, axis=0, dtype=np.float64)
        scales = np.std(values, axis=0, ddof=0, dtype=np.float64)
    _require(np.isfinite(values).all() and np.isfinite(means).all() and np.isfinite(scales).all(),
             "finite_training_clinical_statistics_required")
    scales = np.where(scales == 0, POLICY["constant_scale"], scales)
    state = {"version": VERSION, "policy": json.loads(_canonical(POLICY)),
        "validation_fold": validation_fold, "train_count": len(training),
        "train_participants": len({r["src_subject_id"] for r in training}),
        "train_identity_sha256": _digest([[r["src_subject_id"], r["side"]] for r in training]),
        "BMI_median": median,
        "continuous_mean": {name: float(means[i]) for i, name in enumerate(_CONTINUOUS)},
        "continuous_scale": {name: float(scales[i]) for i, name in enumerate(_CONTINUOUS)}}
    state["record_sha256"] = _digest(state)
    _validate_state(state)
    return state


def _validate_state(state):
    _require(isinstance(state, dict) and set(state) == _STATE_FIELDS and state["version"] == VERSION,
             "exact_fitted_clinical_state_required")
    _policy(state["policy"])
    _require(type(state["validation_fold"]) is int and state["validation_fold"] in range(5)
             and type(state["train_count"]) is int and state["train_count"] > 0
             and type(state["train_participants"]) is int
             and 0 < state["train_participants"] <= state["train_count"] <= 2 * state["train_participants"],
             "valid_training_clinical_state_counts_required")
    for key in ("train_identity_sha256", "record_sha256"):
        _require(isinstance(state[key], str) and re.fullmatch(r"[0-9a-f]{64}", state[key]) is not None,
                 "valid_clinical_state_hash_required")
    _require(_positive_number(state["BMI_median"]), "finite_training_bmi_median_required")
    for key in ("continuous_mean", "continuous_scale"):
        _require(isinstance(state[key], dict) and set(state[key]) == set(_CONTINUOUS)
                 and all(_positive_number(state[key][name]) for name in _CONTINUOUS),
                 "exact_finite_clinical_statistics_required")
    _require(_digest({k: v for k, v in state.items() if k != "record_sha256"}) == state["record_sha256"],
             "fitted_clinical_state_changed")


def transform_clinical_rows(rows, state):
    """Apply a previously fitted state, preserving row order and missing inputs."""
    _validate_state(state)
    _rows(rows)
    values = []
    for row in rows:
        bmi = state["BMI_median"] if row["BMI"] is None else row["BMI"]
        values.append([(row["AGEYEARS"] - state["continuous_mean"]["AGEYEARS"])
                       / state["continuous_scale"]["AGEYEARS"],
                       POLICY["sex_encoding"][row["SEX"]],
                       (bmi - state["continuous_mean"]["BMI"]) / state["continuous_scale"]["BMI"]])
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(values, dtype=np.float32)
    _require(result.shape == (len(rows), 3) and result.flags.c_contiguous and np.isfinite(result).all(),
             "finite_three_feature_clinical_inputs_required")
    result.setflags(write=False)
    return result
