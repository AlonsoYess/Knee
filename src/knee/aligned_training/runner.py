"""Prepared development-fold executor; all fitting requires separate authority.

Preparation never calls the execution path. This module cannot load the reserved
package, calibrate a model, select a threshold, or refit on a held-out test set.
All actual fold artifacts belong in a new private output directory.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from knee.aligned_training.authority import validate_authority, validate_request, digest, require
from knee.methodology_release import load_release, PINS, select_modern_challengers
from knee.colab_data import checked_path, sha_file
from knee.methodology_release import recipe_sha256

VERSION = "registered_development_fold_runner_v1.0"
JOB_FILENAME = "training_job.json"
CLINICAL_IDS = {"clinical_logistic", "clinical_xgboost"}


def read_json(path):
    from knee.training_preparation import read_json as bounded_read
    return bounded_read(path)


def save_json(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as handle:
        handle.write(raw)
    require(Path(path).read_bytes() == raw, "training_artifact_readback_failed")


def _new_private_output(output_dir, protected):
    path = checked_path(output_dir).resolve()
    require(not path.exists() and path.parent.is_dir(), "new_existing_parent_private_output_required")
    require(all(not path.is_relative_to(p) and not p.is_relative_to(path) for p in protected),
        "training_output_must_not_overlap_sources")
    return path


def completed_job(directory, package_sha256, plan, registry):
    root = checked_path(directory)
    record = read_json(root / JOB_FILENAME)
    require(record.get("version") == VERSION and record.get("scope") == "development_only"
        and record.get("eligible_for_model_selection") is True
        and record.get("training_executed") is True
        and record.get("engine_status") == "training_fold_complete"
        and record.get("reserved_test_accessed") is False
        and record.get("record_sha256") == digest({k:v for k,v in record.items() if k != "record_sha256"}),
        "authenticated_complete_development_job_required")
    request = record["request"]
    validate_request(request, plan, registry)
    require(request["development_package_sha256"] == package_sha256
        and request["operation"] == "development_fold"
        and record.get("request_sha256") == digest(request)
        and record.get("plan_sha256") == recipe_sha256(plan),
        "selection_job_package_request_and_recipe_mismatch")
    files = record.get("files_sha256")
    expected = {"best_validation_predictions.json", "clinical_state.json"}
    if request["model_id"] in CLINICAL_IDS:
        expected.add("clinical_model.json" if request["model_id"] == "clinical_logistic" else "clinical_model.ubj")
    else:
        expected.update({"best_model.safetensors", "last_model.safetensors", "training_record.json"})
    require(isinstance(files, dict) and set(files) == expected, "complete_selection_job_artifacts_required")
    require(all(sha_file(root / name) == value for name,value in files.items()), "completed_training_artifact_changed")
    predictions = read_json(root / "best_validation_predictions.json")
    require(isinstance(predictions, dict) and set(predictions) == {"targets", "logits", "identities"}
        and isinstance(predictions["identities"], list)
        and len(predictions["identities"]) == len(predictions["targets"]) == len(predictions["logits"])
        and all(isinstance(identity, list) and len(identity) == 2
            and isinstance(identity[0], str) and bool(identity[0]) and identity[1] in {"LEFT", "RIGHT"}
            for identity in predictions["identities"])
        and len({tuple(identity) for identity in predictions["identities"]}) == len(predictions["identities"]),
        "unique_aligned_validation_prediction_identities_required")
    from knee.aligned_training.engine import binary_metrics
    metrics = binary_metrics(predictions["targets"], predictions["logits"])
    require(metrics == record.get("metrics"), "selection_metrics_must_match_sealed_predictions")
    return record


def _fold_reports(entries, package_sha256, plan, registry, *, stage):
    require(isinstance(entries, list) and len(entries) <= 35, "bounded_five_fold_selection_jobs_required")
    models, seen = {}, set()
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == {"directory", "record_sha256"},
            "exact_selection_job_reference_required")
        record = completed_job(entry["directory"], package_sha256, plan, registry)
        request = record["request"]
        key = request["model_id"], request["validation_fold"]
        require(record["record_sha256"] == entry["record_sha256"] and key not in seen
            and request["stage"] == stage and request["seed"] == 2026
            and request["model_id"].startswith("image_"), "unique_registered_screening_fold_jobs_required")
        seen.add(key)
        group = models.setdefault(request["model_id"], {"gates": record["eligibility_gates"], "folds": []})
        require(group["gates"] == record["eligibility_gates"], "consistent_image_eligibility_review_required")
        group["folds"].append({"validation_fold": request["validation_fold"], **record["metrics"]})
    from knee.aligned_training.engine import aggregate_fold_metrics
    for value in models.values():
        aggregate_fold_metrics(value["folds"])
    return models


def validate_selection(path, request, plan, registry):
    require(path is not None and sha_file(checked_path(path)) == request["selection_evidence_sha256"],
        "exact_approved_selection_evidence_required")
    value = read_json(path)
    require(isinstance(value, dict) and set(value) == {"version", "package_sha256", "screening_jobs",
        "excluded_modern_models", "stage_two_jobs", "selected_image_encoder_id"}
        and value["version"] == "development_image_selection_evidence_v1.0"
        and value["package_sha256"] == request["development_package_sha256"],
        "development_selection_evidence_required")
    screened = _fold_reports(value["screening_jobs"], value["package_sha256"], plan, registry, stage="stage_1")
    require(set(plan["radiographic_models"]["mandatory"]) <= set(screened),
        "mandatory_baselines_require_complete_five_fold_screening")
    modern = set(plan["radiographic_models"]["conditional_challengers"])
    excluded = value["excluded_modern_models"]
    require(isinstance(excluded, dict) and set(excluded) <= modern
        and all(isinstance(reason, str) and bool(reason.strip()) for reason in excluded.values())
        and not set(excluded) & set(screened)
        and (set(screened) & modern) | set(excluded) == modern,
        "every_modern_candidate_requires_complete_screening_or_documented_exclusion")
    reports = [{"model_id": key, "scope": "development_only",
        "aggregation": "mean_of_five_participant_held_out_fold_metrics", "fold_metrics": item["folds"],
        "eligibility_gates": item["gates"]} for key,item in screened.items() if key in modern]
    promoted = select_modern_challengers(reports, plan, registry)
    eligible = set(plan["radiographic_models"]["mandatory"]) | set(promoted)
    if request["stage"] == "stage_2":
        require(request["model_id"] in eligible, "stage_two_requires_mandatory_or_oof_promoted_encoder")
    else:
        from knee.aligned_training.engine import aggregate_fold_metrics
        tuned = _fold_reports(value["stage_two_jobs"], value["package_sha256"], plan, registry, stage="stage_2")
        require(set(tuned) == eligible, "all_retained_image_models_require_complete_stage_two_results")
        for item in tuned.values():
            aggregate_fold_metrics(item["folds"])
        # Preserve the strict serialized 0.01 AP boundary, including when a
        # binary-float arithmetic mean would round 0.49 upward into the tie band.
        means = {key: {metric: sum(Decimal(str(row[metric])) for row in item["folds"]) / Decimal(5)
            for metric in ("average_precision_score", "roc_auc", "brier")} for key,item in tuned.items()}
        anchor = max(item["average_precision_score"] for item in means.values())
        band = [key for key,item in means.items()
            if anchor - item["average_precision_score"] < Decimal("0.01")]
        order = plan["radiographic_models"]["mandatory"] + plan["radiographic_models"]["conditional_challengers"]
        winner = min(band, key=lambda key: (-means[key]["roc_auc"], means[key]["brier"], order.index(key)))
        require(value["selected_image_encoder_id"] == request["image_encoder_id"] == winner,
            "fusion_requires_development_selected_eligible_encoder")
    return {"promoted_modern_models": promoted, "selected_image_encoder_id": value["selected_image_encoder_id"]}


def _reuse_fold_encoder(model, job_dir, request, rows, plan, registry):
    from knee.training_plan import validate_fold_weight_provenance
    record = completed_job(job_dir, request["development_package_sha256"], plan, registry)
    source_request = record["request"]
    require(record["record_sha256"] == request["fold_image_job_sha256"]
        and source_request["model_id"] == request["image_encoder_id"]
        and source_request["stage"] == "stage_2"
        and source_request["validation_fold"] == request["validation_fold"]
        and source_request["seed"] == request["seed"],
        "fusion_weights_must_be_from_selected_same_fold_image_job")
    validate_fold_weight_provenance(rows, request["validation_fold"], record["weights_provenance"])
    predictions = read_json(Path(job_dir) / "best_validation_predictions.json")
    expected_rows = [row for row in rows if row["cv_fold"] == request["validation_fold"]]
    require(predictions["identities"] == [[row["src_subject_id"], row["side"]] for row in expected_rows]
        and predictions["targets"] == [row["PROGRESION_48M"] for row in expected_rows],
        "same_fold_weight_predictions_must_match_held_out_rows")
    from safetensors.torch import load_file
    state = load_file(str(Path(job_dir) / "best_model.safetensors"), device="cpu")
    expected = model.encoder.state_dict()
    adopted = {key[8:]: value for key,value in state.items() if key.startswith("encoder.")}
    require(set(adopted) == set(expected) and all(adopted[key].shape == value.shape
        and adopted[key].dtype == value.dtype for key,value in expected.items()),
        "exact_same_fold_encoder_state_required")
    import torch
    require(all(not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in adopted.values()), "finite_same_fold_encoder_state_required")
    model.encoder.load_state_dict(adopted, strict=True)
    return record["record_sha256"]


def _fit_conventional(model_id, fold, plan, seed, output):
    from knee.aligned_training.engine import binary_metrics
    train, validation = fold.training, fold.validation
    y = np.asarray(train._targets, dtype=np.int64)
    if model_id == "clinical_logistic":
        from sklearn.linear_model import LogisticRegression
        spec = plan["clinical_models"][model_id]
        model = LogisticRegression(penalty=spec["penalty"], C=spec["C"], solver=spec["solver"],
            max_iter=spec["max_iter"], tol=spec["tol"], random_state=seed)
        model.fit(train._clinical, y, sample_weight=np.where(y == 1, fold.class_weight["positive_weight"], 1.))
        logits = model.decision_function(validation._clinical)
        save_json(output / "clinical_model.json", {"family": "logistic_regression", "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(), "classes": model.classes_.tolist(), "n_features_in": int(model.n_features_in_)})
    else:
        from xgboost import XGBClassifier
        spec = plan["clinical_models"][model_id]
        parameters = {key: spec[key] for key in ("objective", "n_estimators", "max_depth", "learning_rate",
            "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "tree_method", "device", "n_jobs")}
        model = XGBClassifier(**parameters, random_state=seed, scale_pos_weight=fold.class_weight["positive_weight"])
        model.fit(train._clinical, y)
        logits = model.predict(validation._clinical, output_margin=True)
        model.save_model(output / "clinical_model.ubj")
    predictions = {"targets": list(validation._targets), "logits": np.asarray(logits, dtype=np.float64).tolist(),
        "identities": [list(identity) for identity in validation.identities]}
    metrics = binary_metrics(predictions["targets"], predictions["logits"])
    save_json(output / "best_validation_predictions.json", predictions)
    return {"status": "training_fold_complete", "metrics": metrics, "optimizer_created": False}


def execute(request, authority, *, repository_root, package_dir, reference_path, weights_root,
            output_dir, device="cpu", allow_download=False, selection_evidence=None, image_job_dir=None,
            baseline_sidecar_path=None):
    """Future one-fold execution; no default authority, test or refit operation."""
    require(authority is not None, "separate_training_authority_required_before_execution")
    request, authority = deepcopy(request), deepcopy(authority)
    root = checked_path(repository_root).resolve()
    from knee import methodology_release as aligned
    methodology = request.get("version") == aligned.REQUEST_VERSION
    selected_release_loader = aligned.load_release if methodology else load_release
    _, catalog, plan, registry = selected_release_loader(root)
    permit = validate_authority(authority, request, plan, registry)
    request_hash, authority_hash = digest(request), digest(authority)
    require(type(allow_download) is bool, "explicit_download_flag_required")
    source = checked_path(package_dir).resolve()
    reference = checked_path(reference_path)
    protected = [root, source, reference.parent]
    baseline_sidecar = None
    if permit.baseline_sidecar_sha256 is not None:
        require(baseline_sidecar_path is not None, "required_development_baseline_sidecar_missing")
        baseline_file = checked_path(baseline_sidecar_path)
        baseline_sidecar = read_json(baseline_file)
        require(baseline_sidecar.get("record_sha256") == permit.baseline_sidecar_sha256,
            "baseline_sidecar_does_not_match_execution_authority")
        protected.append(baseline_file.parent)
    else:
        require(baseline_sidecar_path is None, "unexpected_baseline_sidecar_for_main_scenario")
    if weights_root is not None:
        protected.append(checked_path(weights_root).resolve())
    if selection_evidence is not None:
        protected.append(checked_path(selection_evidence).resolve().parent)
    if image_job_dir is not None:
        protected.append(checked_path(image_job_dir).resolve())
    output = _new_private_output(output_dir, protected)
    selection = None
    if request["stage"] != "stage_1":
        selection = validate_selection(selection_evidence, request, plan, registry)
    sealed_sources = {path: sha_file(path) for path in
        (source / "development_package.zip", source / "handoff.json", reference)}
    if baseline_sidecar_path is not None:
        sealed_sources[baseline_file] = sha_file(baseline_file)
    if selection_evidence is not None:
        sealed_sources[checked_path(selection_evidence)] = sha_file(checked_path(selection_evidence))
    if image_job_dir is not None:
        image_source = checked_path(image_job_dir)
        image_record = completed_job(image_source, permit.package_sha256, plan, registry)
        for name in (JOB_FILENAME, *image_record["files_sha256"]):
            sealed_sources[image_source / name] = sha_file(image_source / name)
    from knee.aligned_training.data import load_sealed_inputs, prepare_fold, loaders
    loaded = load_sealed_inputs(source, reference, root, permit.package_sha256)
    rows = loaded["manifest"]["rows"]
    fold = prepare_fold(rows, loaded["tensors"], permit.validation_fold,
        encoder_id=permit.image_encoder_id, package_sha256=permit.package_sha256,
        clinical_scenario=permit.clinical_scenario, baseline_sidecar=baseline_sidecar,
        baseline_sidecar_sha256=permit.baseline_sidecar_sha256, methodology=methodology, seed=permit.seed,
        augmentation_policy=plan["neural_training"]["augmentation"] if methodology and permit.image_encoder_id else None)
    smoke = request["operation"] == "single_fold_smoke"
    if permit.model_id in CLINICAL_IDS:
        output.mkdir(exist_ok=False)
        result = _fit_conventional(permit.model_id, fold, plan, permit.seed, output)
        metrics = result["metrics"]
    else:
        from knee.aligned_training.models import build_model
        from knee.aligned_training.engine import train_neural_fold, binary_metrics
        paths = None
        if permit.image_encoder_id:
            require(weights_root is not None, "verified_external_weights_cache_required")
            from knee.final_models_preflight import checkpoint_files
            paths = checkpoint_files(permit.image_encoder_id, catalog, checked_path(weights_root), allow_download)
            sealed_sources.update({path: sha_file(path) for path in paths})
        model, adoption = build_model(permit.model_id, plan, catalog, paths,
            image_encoder_id=permit.image_encoder_id if request["stage"] == "fusion" else None, seed=permit.seed,
            clinical_scenario=permit.clinical_scenario)
        if request["stage"] == "fusion":
            require(image_job_dir is not None, "selected_same_fold_image_job_required")
            _reuse_fold_encoder(model, image_job_dir, request, rows, plan, registry)
        training, validation = loaders(fold, seed=permit.seed,
            micro_batch_size=64 if permit.model_id == "clinical_mlp" else permit.image_micro_batch_size)
        result = train_neural_fold(model, training, validation, plan, device=device, seed=permit.seed,
            positive_weight=fold.class_weight["positive_weight"], stage=permit.stage, output_dir=output,
            maximum_steps=2 if smoke else None, smoke=smoke, execution_authority=permit)
        predictions = read_json(output / "best_validation_predictions.json")
        metrics = binary_metrics(predictions["targets"], predictions["logits"])
    require(digest(request) == request_hash and digest(authority) == authority_hash,
        "training_request_or_authority_changed_during_execution")
    require(all(sha_file(path) == checksum for path, checksum in sealed_sources.items()),
        "sealed_training_source_changed_during_execution")
    selected_release_loader(root)
    save_json(output / "clinical_state.json", fold.clinical_state)
    provenance = {"fit_scope": "development_training_fold", "validation_fold": permit.validation_fold,
        "train_identity_sha256": fold.clinical_state["train_identity_sha256"],
        "trained_participants": sorted({sid for sid,_ in fold.training.identities}), "reserved_test_accessed": False}
    record = {"version": VERSION, "scope": "development_only", "request": request,
        "request_sha256": digest(request), "plan_sha256": recipe_sha256(plan),
        "eligible_for_model_selection": not smoke, "reserved_test_accessed": False,
        "weights_provenance": provenance, "metrics": metrics,
        "eligibility_gates": authority["eligibility_gates"], "selection_evidence": selection,
        "engine_status": result["status"], "training_executed": True,
        "files_sha256": {p.name:sha_file(p) for p in sorted(output.iterdir()) if p.is_file()},
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    record["record_sha256"] = digest(record)
    save_json(output / JOB_FILENAME, record)
    return record
