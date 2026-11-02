"""Fixed probability fusion on authenticated, participant-held-out development jobs.

This complementary comparator has no trainable weights. Its clinical branch is
selected from the three MAIN clinical candidates by the registered five-fold
rule; its visual branch must satisfy the existing image-selection evidence
validator. The same development folds used for selection are descriptive, not
an unbiased final performance estimate. Hashes establish artifact integrity and
binding to supplied trusted receipts, not independent proof of how fitting was
performed. No reserved package or prediction is accepted by this module.
"""
from __future__ import annotations

from decimal import Decimal
import math
from pathlib import Path

import numpy as np

from knee.colab_data import checked_path, sha_file, valid_sha
from knee.aligned_training.authority import digest, require
from knee.training_plan import _fold_context, validate_fold_weight_provenance

VERSION = "development_fixed_probability_late_fusion_v1.0"
MODEL_ID = "fusion_late_fixed_mean"
FILENAME = "late_fusion_development.json"
WEIGHTS = (0.5, 0.5)


def fixed_mean_probabilities(clinical_probabilities, image_probabilities):
    """Return the fixed mean of aligned probabilities, never averaged logits."""
    clinical, image = np.asarray(clinical_probabilities), np.asarray(image_probabilities)
    require(clinical.ndim == 1 and clinical.size > 0 and image.shape == clinical.shape
        and clinical.dtype.kind in "fiu" and image.dtype.kind in "fiu"
        and np.isfinite(clinical).all() and np.isfinite(image).all()
        and ((clinical >= 0) & (clinical <= 1)).all()
        and ((image >= 0) & (image <= 1)).all(), "aligned_finite_probability_vectors_required")
    return clinical.astype(np.float64) * WEIGHTS[0] + image.astype(np.float64) * WEIGHTS[1]


def probability_metrics(targets, probabilities):
    """Compute AP, ROC-AUC and Brier without clipping exact 0/1 probabilities."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    p = fixed_mean_probabilities(probabilities, probabilities)
    require(isinstance(targets, list) and all(type(v) is int and v in {0, 1} for v in targets)
        and len(targets) == p.size and set(targets) == {0, 1}, "both_class_binary_targets_required")
    return {"average_precision_score": float(average_precision_score(targets, p)),
        "roc_auc": float(roc_auc_score(targets, p)), "brier": float(brier_score_loss(targets, p))}


def _probability(logit):
    require(type(logit) in {int, float} and math.isfinite(logit), "finite_source_logit_required")
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1 + exp_value)


def _expected(rows):
    """Require sealed-package shaped rows; all test assignments fail closed."""
    for fold in range(5):
        _fold_context(rows, fold)
    require(all(type(row.get("PROGRESION_48M")) is int and row["PROGRESION_48M"] in {0, 1}
        for row in rows), "development_binary_targets_required")
    for fold in range(5):
        require({row["PROGRESION_48M"] for row in rows if row["cv_fold"] == fold} == {0, 1},
            "both_classes_in_each_development_fold_required")
    return {(row["src_subject_id"], row["side"]): row for row in rows}


def _prediction_index(predictions, expected, fold):
    require(isinstance(predictions, dict) and set(predictions) == {"identities", "targets", "logits"}
        and all(isinstance(predictions[key], list) for key in predictions)
        and len(predictions["identities"]) == len(predictions["targets"]) == len(predictions["logits"]),
        "aligned_source_prediction_arrays_required")
    found = {}
    for identity, target, logit in zip(predictions["identities"], predictions["targets"], predictions["logits"]):
        require(isinstance(identity, list) and len(identity) == 2
            and isinstance(identity[0], str) and identity[1] in {"LEFT", "RIGHT"},
            "source_knee_identity_required")
        key = tuple(identity)
        require(key in expected and key not in found and expected[key]["cv_fold"] == fold,
            "unique_same_fold_held_out_predictions_required")
        require(type(target) is int and target == expected[key]["PROGRESION_48M"],
            "source_prediction_target_mismatch")
        found[key] = _probability(logit)
    require(set(found) == {key for key, row in expected.items() if row["cv_fold"] == fold},
        "complete_held_out_fold_predictions_required")
    return found


def fuse_held_out_predictions(clinical_predictions, image_predictions, expected_rows, validation_fold):
    """Numerical helper only; callers must authenticate source jobs separately.

    Output is keyed by participant and side, independent of source row order.
    This function itself does not issue a completion receipt or certify sources.
    """
    require(type(validation_fold) is int and validation_fold in range(5), "fixed_development_fold_required")
    expected = _expected(expected_rows)
    clinical = _prediction_index(clinical_predictions, expected, validation_fold)
    image = _prediction_index(image_predictions, expected, validation_fold)
    keys = sorted(clinical)
    fused = fixed_mean_probabilities([clinical[key] for key in keys], [image[key] for key in keys])
    return [{"src_subject_id": key[0], "side": key[1], "validation_fold": validation_fold,
        "target": expected[key]["PROGRESION_48M"], "clinical_probability": clinical[key],
        "image_probability": image[key], "probability": float(probability)}
        for key, probability in zip(keys, fused)]


def _verified_jobs(references, model_id, expected_rows, package_sha256, plan, registry):
    from knee.aligned_training.runner import completed_job, read_json
    from knee.aligned_training.engine import aggregate_fold_metrics
    require(isinstance(references, list) and len(references) == 5, "five_source_job_references_required")
    jobs, folds, sealed = {}, [], {}
    expected = _expected(expected_rows)
    for reference in references:
        require(isinstance(reference, dict) and set(reference) == {"directory", "record_sha256"}
            and valid_sha(reference["record_sha256"]), "trusted_source_job_hash_required")
        directory = checked_path(reference["directory"])
        job = completed_job(directory, package_sha256, plan, registry)
        request = job["request"]
        fold = request["validation_fold"]
        require(job["record_sha256"] == reference["record_sha256"] and request["model_id"] == model_id
            and fold not in jobs and request["seed"] == plan["selection"]["screening_seed"]
            and request.get("clinical_scenario") == "clinical_main"
            and request.get("baseline_sidecar_sha256") is None,
            "same_release_main_scenario_source_jobs_required")
        require(request["stage"] == ("stage_1" if model_id in plan["clinical_models"] else "stage_2"),
            "main_clinical_or_selected_finetuned_image_job_required")
        validate_fold_weight_provenance(expected_rows, fold, job["weights_provenance"])
        predictions = read_json(directory / "best_validation_predictions.json")
        _prediction_index(predictions, expected, fold)
        jobs[fold] = (directory, job, predictions)
        folds.append({"validation_fold": fold, **job["metrics"]})
        for filename, checksum in job["files_sha256"].items():
            sealed[directory / filename] = checksum
        sealed[directory / "training_job.json"] = sha_file(directory / "training_job.json")
    require(set(jobs) == set(range(5)), "complete_unique_source_folds_required")
    return jobs, aggregate_fold_metrics(folds), sealed


def _select_clinical(candidates, plan):
    """Apply the existing AP near-tie rule to unrounded five-fold means."""
    order = list(plan["clinical_models"])
    means = {model: {metric: sum(Decimal(str(job[1]["metrics"][metric])) for job in jobs.values()) / Decimal(5)
        for metric in ("average_precision_score", "roc_auc", "brier")} for model, jobs in candidates.items()}
    anchor = max(value["average_precision_score"] for value in means.values())
    band = [model for model, value in means.items()
        if anchor - value["average_precision_score"] < Decimal(str(plan["selection"]["near_tie_ap_difference_below"]))]
    return min(band, key=lambda model: (-means[model]["roc_auc"], means[model]["brier"], order.index(model)))


def _bind_selection_population(selection, expected_rows):
    """Bind every image-selection comparison to this same held-out population.

    ``validate_selection`` has already authenticated checkpoints and metrics.
    This additional check prevents a competing candidate from being ranked on a
    different subset, or using weights fitted with held-out participants.
    """
    from knee.aligned_training.runner import read_json
    expected, sealed = _expected(expected_rows), {}
    for key in ("screening_jobs", "stage_two_jobs"):
        for reference in selection[key]:
            directory = checked_path(reference["directory"])
            job_path = directory / "training_job.json"
            prediction_path = directory / "best_validation_predictions.json"
            job = read_json(job_path)
            require(job["record_sha256"] == reference["record_sha256"]
                == digest({field: value for field, value in job.items() if field != "record_sha256"}),
                "image_selection_job_changed_during_fusion")
            require(sha_file(prediction_path) == job["files_sha256"][prediction_path.name],
                "image_selection_predictions_changed_during_fusion")
            fold = job["request"]["validation_fold"]
            validate_fold_weight_provenance(expected_rows, fold, job["weights_provenance"])
            _prediction_index(read_json(prediction_path), expected, fold)
            sealed[job_path] = sha_file(job_path)
            sealed[prediction_path] = job["files_sha256"][prediction_path.name]
    return sealed


def combine_development_jobs(clinical_candidates, image_jobs, expected_rows, *, package_sha256,
        plan, registry, selected_image_encoder_id, image_selection_path, image_selection_sha256,
        output_dir=None):
    """Fuse reviewed source jobs; read predictions only, fit no model or threshold.

    ``clinical_candidates`` maps each of the three registered main clinical IDs
    to five ``{directory, record_sha256}`` trusted receipt references. ``image_jobs``
    contains the five selected visual stage-2 references. Image selection evidence
    is separately bound by its trusted file SHA256 and verified using the same
    mandatory/challenger selection rules as intermediate fusion.

    Historical initial-stage outputs remain intact: this entrypoint requires the
    reconciled recipe, not a mixture of old and new comparison conditions.
    """
    from knee.aligned_training.runner import read_json, save_json, validate_selection, _new_private_output
    from knee.aligned_training.engine import aggregate_fold_metrics
    from knee.methodology_release import validate_plan
    validate_plan(plan, registry)
    require(plan.get("version") == "registered_training_plan_v1.2"
        and MODEL_ID in plan["multimodal_models"].get("complementary_enabled", [])
        and valid_sha(package_sha256), "reconciled_late_fusion_recipe_required")
    _expected(expected_rows)
    require(isinstance(clinical_candidates, dict)
        and set(clinical_candidates) == set(plan["clinical_models"]), "all_three_main_clinical_candidates_required")
    image_ids = plan["radiographic_models"]["mandatory"] + plan["radiographic_models"]["conditional_challengers"]
    require(selected_image_encoder_id in image_ids, "registered_selected_image_encoder_required")
    selection_path = checked_path(image_selection_path)
    require(valid_sha(image_selection_sha256) and sha_file(selection_path) == image_selection_sha256,
        "trusted_image_selection_hash_required")
    sources, clinical_metrics, sealed = {}, {}, {selection_path: image_selection_sha256}
    for model_id in plan["clinical_models"]:
        jobs, metrics, checksums = _verified_jobs(clinical_candidates[model_id], model_id,
            expected_rows, package_sha256, plan, registry)
        sources[model_id], clinical_metrics[model_id] = jobs, metrics
        sealed.update(checksums)
    clinical_id = _select_clinical(sources, plan)
    images, _, checksums = _verified_jobs(image_jobs, selected_image_encoder_id,
        expected_rows, package_sha256, plan, registry)
    sealed.update(checksums)
    request = dict(images[0][1]["request"])
    request.update(model_id="fusion_intermediate_concat", stage="fusion",
        image_encoder_id=selected_image_encoder_id, selection_evidence_sha256=image_selection_sha256,
        fold_image_job_sha256=images[0][1]["record_sha256"])
    validate_selection(selection_path, request, plan, registry)
    selection = read_json(selection_path)
    sealed.update(_bind_selection_population(selection, expected_rows))
    # Bind the fusion predictions to exactly the selected jobs that supplied the
    # selection metrics; supplying a different run of the same architecture fails.
    recorded = {(str(checked_path(entry["directory"])), entry["record_sha256"])
        for entry in selection["stage_two_jobs"]}
    require(all((str(directory), job["record_sha256"]) in recorded
        for directory, job, _ in images.values()), "fusion_source_run_differs_from_selection_evidence")
    predictions, folds, provenance = [], [], []
    for fold in range(5):
        clinical_dir, clinical_job, clinical_predictions = sources[clinical_id][fold]
        image_dir, image_job, image_predictions = images[fold]
        require(clinical_job["weights_provenance"] == image_job["weights_provenance"],
            "identical_training_participants_required_for_fusion_branches")
        values = fuse_held_out_predictions(clinical_predictions, image_predictions, expected_rows, fold)
        predictions.extend(values)
        folds.append({"validation_fold": fold, **probability_metrics(
            [row["target"] for row in values], [row["probability"] for row in values])})
        provenance.append({"validation_fold": fold, "clinical_job_sha256": clinical_job["record_sha256"],
            "image_job_sha256": image_job["record_sha256"], "weights_provenance": clinical_job["weights_provenance"]})
    require(len(predictions) == len(expected_rows)
        and len({(row["src_subject_id"], row["side"]) for row in predictions}) == len(expected_rows),
        "one_fused_prediction_per_development_knee_required")
    result = {"version": VERSION, "model_id": MODEL_ID, "status": "development_complementary_fusion_complete",
        "scope": "development_only", "development_package_sha256": package_sha256,
        "plan_sha256": digest(plan), "weights": list(WEIGHTS), "combination_space": "probabilities",
        "clinical_scenario": "clinical_main", "clinical_model_id": clinical_id,
        "image_encoder_id": selected_image_encoder_id, "clinical_candidate_selection_metrics": clinical_metrics,
        "image_selection_sha256": image_selection_sha256, "fold_results": folds,
        "mean_fivefold_metrics": aggregate_fold_metrics(folds),
        "pooled_oof_metrics_descriptive_only": probability_metrics(
            [row["target"] for row in predictions], [row["probability"] for row in predictions]),
        "source_provenance": provenance, "prediction_rows": predictions,
        "development_knees": len(predictions), "development_participants": len({r["src_subject_id"] for r in predictions}),
        "training_executed": False, "optimizer_created": False, "source_models_previously_trained": True,
        "calibration_performed": False, "threshold_selected": False, "reserved_test_accessed": False,
        "originals_modified": False, "partitions_modified": False, "unbiased_final_performance_claimed": False,
        "limitation": "These development folds selected the component models; metrics are not independent final-test estimates."}
    result["record_sha256"] = digest(result)
    require(all(sha_file(path) == checksum for path, checksum in sealed.items()), "fusion_source_changed_during_aggregation")
    if output_dir is not None:
        protected = {path.parent.resolve() for path in sealed if path != selection_path}
        protected.add(selection_path.resolve())
        output = _new_private_output(output_dir, protected)
        output.mkdir()
        save_json(output / FILENAME, result)
    return result
