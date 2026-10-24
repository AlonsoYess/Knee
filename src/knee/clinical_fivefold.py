"""Three fixed clinical baselines on the same sealed development folds.

This module is prepared locally and executed only by the investigator in Colab.
It has no reserved-test, image-model, calibration, threshold or refit operation.
Only complete authenticated jobs can be recovered after a disconnect.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path
import shutil
import sys

from knee.colab_data import checked_path, require, sha_file, valid_sha
from knee.final_model_release import load_release, PINS, RELEASE_PATH
from knee.training_authority import digest, REQUEST_VERSION, VERSION as AUTHORITY_VERSION
from knee import training_preparation as preparation
from knee import training_smoke as previous

VERSION = "clinical_three_model_fivefold_development_v1.0"
STATUS = "clinical_fivefold_development_complete"
FILENAME = "clinical_fivefold.json"
MODEL_IDS = ("clinical_logistic", "clinical_xgboost", "clinical_mlp")
FOLDS = (0, 1, 2, 3, 4)
SEED = 2026
APPROVAL_STATEMENT = (
    "Autorizo solo el entrenamiento en CPU de regresión logística, XGBoost y MLP clínicos "
    "en los mismos cinco pliegues de desarrollo, semilla 2026 y receta registrada; "
    "sin abrir la prueba reservada, entrenar modelos de imagen ni realizar calibración o ajuste de umbral."
)


def make_request(package_sha256, model_id, validation_fold):
    require(valid_sha(package_sha256) and model_id in MODEL_IDS
        and type(validation_fold) is int and validation_fold in FOLDS,
        "exact_registered_clinical_model_package_and_fold_required")
    return {"version": REQUEST_VERSION, "operation": "development_fold", "model_id": model_id,
        "image_encoder_id": None, "stage": "stage_1", "validation_fold": validation_fold,
        "seed": SEED, "development_package_sha256": package_sha256, "image_micro_batch_size": 2,
        "selection_evidence_sha256": None, "fold_image_job_sha256": None}


def _authority(request, registry, statement):
    require(statement == APPROVAL_STATEMENT, "exact_clinical_fivefold_training_approval_required")
    require(request == make_request(request["development_package_sha256"], request["model_id"],
        request["validation_fold"]),
        "clinical_only_training_authority_required")
    # Image-pretraining gates are deliberately not asserted for clinical models.
    return {"version": AUTHORITY_VERSION, "training_authorized": True,
        "reserved_test_access_authorized": False, "approval_source": "direct_investigator_authorization",
        "approval_statement": statement, "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_sha256": digest(request), "modeling_release_sha256": PINS[RELEASE_PATH],
        "eligibility_gates": {key: False for key in registry["eligibility_gates"]},
        "full_encoder_training_authorized": False}


def sources(repository):
    root = checked_path(repository)
    expected = {**preparation.SOURCE_33, **previous.SOURCE_34, **previous.RUNTIME_SOURCE_PATCHES}
    require(all(sha_file(root / name) == value for name, value in expected.items()),
        "verified_clinical_training_sources_changed_since_preparation")
    for name in ("src/knee/clinical_fivefold.py", "src/knee/training_smoke.py",
            "src/knee/clinical_fold_inputs.py", "src/knee/model_inputs_colab.py"):
        expected[name] = sha_file(root / name)
    return expected


def verify_cpu_environment(execution_environment):
    require(execution_environment == "google_colab_cpu", "actual_colab_cpu_clinical_training_required")
    from knee.final_models_preflight import verify_environment
    environment = verify_environment(execution_environment)
    versions = {key: importlib.metadata.version(key) for key in preparation.EXTRA_PACKAGES}
    require(versions == preparation.EXTRA_PACKAGES, "registered_clinical_cpu_dependencies_required")
    import torch
    require(torch.version.cuda is None and not torch.cuda.is_available(), "clinical_cpu_build_required")
    environment.update(versions)
    environment.update(device="cpu", torch_threads=torch.get_num_threads())
    return environment


def _finite(value, minimum=0, maximum=None):
    return type(value) in {int, float} and math.isfinite(value) and value >= minimum \
        and (maximum is None or value <= maximum)


def _subsets(rows, fold):
    from knee.colab_data import validate_rows
    validate_rows(rows, "development")
    training = [row for row in rows if row["cv_fold"] != fold]
    validation = [row for row in rows if row["cv_fold"] == fold]
    train_ids = sorted([[row["src_subject_id"], row["side"]] for row in training])
    participants = sorted({row[0] for row in train_ids})
    require(training and validation and not set(participants).intersection(
        row["src_subject_id"] for row in validation), "participant_disjoint_clinical_fold_required")
    require({row["PROGRESION_48M"] for row in training} == {0, 1}
        and {row["PROGRESION_48M"] for row in validation} == {0, 1},
        "both_classes_required_in_clinical_training_and_validation")
    return training, validation, train_ids, participants


def _validate_mlp_engine(root, request, rows, job, predictions):
    from knee.training_engine import EarlyStopping, learning_rate_factor, VERSION as ENGINE_VERSION
    training, validation, train_ids, participants = _subsets(rows, request["validation_fold"])
    engine = preparation.read_json(root / "training_record.json")
    require(engine.get("version") == ENGINE_VERSION and engine.get("model_id") == "clinical_mlp"
        and engine.get("stage") == "stage_1" and engine.get("seed") == SEED
        and engine.get("smoke") is False and engine.get("scope") == "development_only"
        and engine.get("status") == "training_fold_complete"
        and engine.get("eligible_for_model_selection") is True
        and engine.get("reserved_test_accessed") is False
        and engine.get("optimizer_created") is True and engine.get("training_executed") is True
        and engine.get("micro_batch_size") == engine.get("effective_batch_size") == 64
        and engine.get("device") == "cpu" and engine.get("precision") == "float32"
        and engine.get("scheduler_horizon_epochs") == 60
        and engine.get("restored_best_checkpoint") is True and engine.get("resume_supported") is False
        and engine.get("optimizer_state_saved") is False
        and engine.get("gpu_resource_profile_recorded") is False,
        "complete_registered_cpu_clinical_mlp_required")
    require(engine.get("provenance") == {"validation_fold": request["validation_fold"],
        "development_package_sha256": request["development_package_sha256"],
        "train_identity_sha256": digest(train_ids), "request_sha256": digest(request),
        "training_knees": len(training), "validation_knees": len(validation),
        "trained_participants": participants}, "clinical_mlp_exact_training_provenance_required")
    positives = sum(row["PROGRESION_48M"] for row in training)
    require(_finite(engine.get("positive_weight"), minimum=1e-30)
        and math.isclose(engine["positive_weight"], (len(training) - positives) / positives,
            rel_tol=1e-12, abs_tol=0), "clinical_mlp_training_only_class_weight_required")
    history = engine.get("history")
    require(isinstance(history, list) and 1 <= len(history) <= 60
        and type(engine.get("completed_epochs")) is int and engine["completed_epochs"] == len(history),
        "one_to_sixty_complete_cpu_clinical_epochs_required")
    monitor, stop, scale_audit = EarlyStopping(patience=8), False, []
    updates = math.ceil(len(training) / 64)
    for epoch, row in enumerate(history, start=1):
        require(not stop and row.get("epoch") == epoch and row.get("phase") == "full"
            and type(row.get("training_knees_seen")) is int and row["training_knees_seen"] == len(training)
            and type(row.get("optimizer_updates")) is int and row["optimizer_updates"] == updates
            and _finite(row.get("weighted_loss_per_knee"))
            and row.get("learning_rate_factor") == learning_rate_factor(epoch)
            and isinstance(row.get("validation"), dict)
            and set(row["validation"]) == {"average_precision_score", "roc_auc", "brier"}
            and all(_finite(value, maximum=1) for value in row["validation"].values())
            and row.get("amp_overflow_replays") == 0,
            "complete_finite_cpu_clinical_epochs_and_schedule_required")
        audit = row.get("amp_scale_audit")
        require(isinstance(audit, list) and len(audit) == updates, "every_cpu_clinical_update_requires_audit")
        for index, entry in enumerate(audit, start=1):
            require(set(entry) == {"update_index_within_epoch", "start_scale", "successful_scale",
                "scale_after_update", "backoffs", "attempts", "gradient_norm_before_clip"}
                and type(entry["update_index_within_epoch"]) is int and entry["update_index_within_epoch"] == index
                and all(_finite(entry[name], minimum=1., maximum=1.)
                    for name in ("start_scale", "successful_scale", "scale_after_update"))
                and type(entry["backoffs"]) is int and entry["backoffs"] == 0
                and type(entry["attempts"]) is int and entry["attempts"] == 1
                and _finite(entry["gradient_norm_before_clip"]), "finite_unit_scale_cpu_clinical_audit_required")
            scale_audit.append({"epoch": epoch, **entry})
        improved, stop = monitor.update(epoch, row["validation"]["average_precision_score"])
        require(row.get("selected_best") is improved, "strict_clinical_ap_best_epoch_required")
    require((stop or len(history) == 60) and engine.get("best_epoch") == monitor.best_epoch
        and engine.get("best_validation_ap") == monitor.best_score
        and job["metrics"] == history[monitor.best_epoch - 1]["validation"]
        and type(engine.get("optimizer_updates")) is int and engine["optimizer_updates"] == updates * len(history)
        and engine.get("amp_overflow_replays") == 0 and engine.get("amp_scale_audit") == scale_audit,
        "registered_clinical_mlp_stop_and_update_accounting_required")
    require(engine.get("checkpoint_sha256") == {name: job["files_sha256"][name]
        for name in ("best_model.safetensors", "last_model.safetensors")},
        "clinical_mlp_checkpoint_hashes_required")
    resource = engine.get("resource_profile", {})
    require(resource.get("device_type") == "cpu" and resource.get("cuda_version") is None
        and resource.get("cuda_peak_measured") is False
        and _finite(resource.get("elapsed_seconds"), minimum=1e-30), "measured_cpu_clinical_profile_required")
    return engine


def _verify_model_predictions(root, request, validation, state, predictions, plan, catalog):
    """Replay saved clinical inference without fitting or touching any image."""
    import numpy as np
    from knee.clinical_fold_inputs import transform_clinical_rows
    clinical = transform_clinical_rows(validation, state)
    model_id = request["model_id"]
    if model_id == "clinical_logistic":
        model = preparation.read_json(root / "clinical_model.json")
        require(set(model) == {"family", "coef", "intercept", "classes", "n_features_in"}
            and model["family"] == "logistic_regression" and model["classes"] == [0, 1]
            and type(model["n_features_in"]) is int and model["n_features_in"] == 3,
            "three_feature_binary_clinical_logistic_artifact_required")
        coef, intercept = np.asarray(model["coef"], dtype=np.float64), np.asarray(model["intercept"], dtype=np.float64)
        require(coef.shape == (1, 3) and intercept.shape == (1,)
            and bool(np.isfinite(coef).all()) and bool(np.isfinite(intercept).all()),
            "finite_clinical_logistic_parameters_required")
        logits = (clinical.astype(np.float64) @ coef.T).reshape(-1) + intercept[0]
        # sklearn may use a different dot-product accumulation order.
        tolerance = 1e-12
    elif model_id == "clinical_xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(root / "clinical_model.ubj")
        require(model.get_booster().num_features() == 3
            and model.get_booster().num_boosted_rounds() == plan["clinical_models"][model_id]["n_estimators"],
            "registered_clinical_xgboost_shape_and_rounds_required")
        logits = model.predict(clinical, output_margin=True)
        tolerance = 0
    else:
        import torch
        from safetensors.torch import load_file
        from knee.training_models import build_model
        model, _ = build_model(model_id, plan, catalog, seed=SEED)
        initial = model.state_dict()
        states = {}
        for name in ("best_model.safetensors", "last_model.safetensors"):
            adopted = load_file(str(root / name), device="cpu")
            require(set(adopted) == set(initial) and all(value.shape == initial[key].shape
                and value.dtype == initial[key].dtype and bool(torch.isfinite(value).all())
                for key, value in adopted.items()), "finite_exact_clinical_mlp_parameters_required")
            require(any(not torch.equal(adopted[key], value) for key, value in initial.items()),
                "clinical_mlp_parameters_must_have_updated")
            states[name] = adopted
        model.load_state_dict(states["best_model.safetensors"], strict=True)
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([model(clinical=torch.from_numpy(clinical[i:i + 64].copy()))
                for i in range(0, len(clinical), 64)]).numpy()
        tolerance = 0
    require(np.asarray(logits).shape == (len(validation),)
        and bool(np.isfinite(logits).all())
        and np.allclose(np.asarray(predictions["logits"], dtype=np.float64), logits,
            rtol=tolerance, atol=tolerance), "saved_clinical_model_must_reproduce_held_out_logits")


def validate_completed_job(directory, request, plan, registry, expected_rows):
    from knee.training_runner import completed_job, JOB_FILENAME
    from knee.clinical_fold_inputs import fit_clinical_fold
    root = checked_path(directory)
    require(request == make_request(request["development_package_sha256"], request["model_id"],
        request["validation_fold"]), "exact_registered_clinical_job_request_required")
    require(root.is_dir() and all(checked_path(path).parent == root and path.is_file()
        for path in root.iterdir()), "only_plain_clinical_job_artifacts_required")
    job = completed_job(root, request["development_package_sha256"], plan, registry)
    require(job["request"] == request and job.get("selection_evidence") is None
        and job.get("eligibility_gates") == {key: False for key in registry["eligibility_gates"]}
        and set(path.name for path in root.iterdir()) == set(job["files_sha256"]) | {JOB_FILENAME},
        "exact_complete_clinical_job_artifacts_required")
    training, validation, train_ids, participants = _subsets(expected_rows, request["validation_fold"])
    require(job.get("weights_provenance") == {"fit_scope": "development_training_fold",
        "validation_fold": request["validation_fold"], "train_identity_sha256": digest(train_ids),
        "trained_participants": participants, "reserved_test_accessed": False},
        "exact_clinical_training_identity_provenance_required")
    state = fit_clinical_fold(expected_rows, request["validation_fold"])
    require(preparation.read_json(root / "clinical_state.json") == state,
        "exact_training_only_clinical_state_required")
    predictions = preparation.read_json(root / "best_validation_predictions.json")
    require(predictions["identities"] == [[row["src_subject_id"], row["side"]] for row in validation]
        and predictions["targets"] == [row["PROGRESION_48M"] for row in validation],
        "exact_clinical_validation_identity_and_target_required")
    engine = _validate_mlp_engine(root, request, expected_rows, job, predictions) \
        if request["model_id"] == "clinical_mlp" else None
    _, catalog, _, _ = load_release(Path(__file__).resolve().parents[2])
    _verify_model_predictions(root, request, validation, state, predictions, plan, catalog)
    return job, engine


def aggregate_predictions(jobs, expected_rows):
    from knee.training_engine import aggregate_fold_metrics, binary_metrics
    require(isinstance(jobs, dict) and set(jobs) == set(FOLDS),
        "all_five_clinical_fold_jobs_required_before_aggregation")
    rows, metrics, seen, participant_folds, model_id, package_sha = [], [], set(), {}, None, None
    for fold in FOLDS:
        directory, job = jobs[fold]
        request = job["request"]
        model_id = request["model_id"] if model_id is None else model_id
        package_sha = request["development_package_sha256"] if package_sha is None else package_sha
        require(request == make_request(package_sha, model_id, fold), "clinical_job_fold_identity_mismatch")
        root = checked_path(directory)
        require(preparation.read_json(root / "training_job.json") == job
            and all(sha_file(root / name) == checksum for name, checksum in job["files_sha256"].items()),
            "clinical_job_artifacts_changed_before_aggregation")
        predictions = preparation.read_json(root / "best_validation_predictions.json")
        expected = [row for row in expected_rows if row["cv_fold"] == fold]
        require(predictions["identities"] == [[row["src_subject_id"], row["side"]] for row in expected]
            and predictions["targets"] == [row["PROGRESION_48M"] for row in expected]
            and len(predictions["logits"]) == len(expected), "exact_clinical_oof_coverage_required")
        require(binary_metrics(predictions["targets"], predictions["logits"]) == job["metrics"],
            "clinical_oof_metrics_must_match_authenticated_predictions")
        metrics.append({"validation_fold": fold, **job["metrics"]})
        for identity, target, logit in zip(predictions["identities"], predictions["targets"], predictions["logits"]):
            key = tuple(identity)
            require(key not in seen and participant_folds.get(identity[0], fold) == fold,
                "clinical_oof_duplicate_or_participant_overlap")
            seen.add(key)
            participant_folds[identity[0]] = fold
            rows.append({"src_subject_id": identity[0], "side": identity[1],
                "validation_fold": fold, "target": target, "logit": logit})
    require(seen == {(row["src_subject_id"], row["side"]) for row in expected_rows}
        and len(rows) == len(expected_rows), "every_clinical_development_knee_predicted_once_required")
    return aggregate_fold_metrics(metrics), binary_metrics([row["target"] for row in rows],
        [row["logit"] for row in rows]), rows


def constant_reference(expected_rows):
    """Training-prevalence reference, without fitting a predictor or a calibrator."""
    from knee.training_engine import aggregate_fold_metrics, binary_metrics
    reports = []
    for fold in FOLDS:
        training, validation, _, _ = _subsets(expected_rows, fold)
        probability = sum(row["PROGRESION_48M"] for row in training) / len(training)
        logit = math.log(probability / (1 - probability))
        reports.append({"validation_fold": fold, "training_prevalence": probability,
            **binary_metrics([row["PROGRESION_48M"] for row in validation], [logit] * len(validation))})
    return {"scope": "development_only", "fit_source": "training_fold_prevalence_only",
        "fold_results": reports, "fivefold_selection_metrics": aggregate_fold_metrics([
            {key: value for key, value in row.items() if key != "training_prevalence"} for row in reports]),
        "calibration_performed": False, "threshold_selected": False}


def run(repository_root, preparation_dir, package_dir, reference_path, output_dir, *,
        approval_statement, execution_environment="google_colab_cpu", job_completed=None, completed_jobs=None):
    require(approval_statement == APPROVAL_STATEMENT, "exact_clinical_fivefold_training_approval_required")
    require(job_completed is None or callable(job_completed), "clinical_job_completion_callback_required")
    completed_jobs = {} if completed_jobs is None else completed_jobs
    require(isinstance(completed_jobs, dict) and all(isinstance(key, tuple) and len(key) == 2
        and key[0] in MODEL_IDS and type(key[1]) is int and key[1] in FOLDS for key in completed_jobs),
        "only_exact_completed_clinical_job_directories_may_be_reused")
    environment = verify_cpu_environment(execution_environment)
    repository, proof, package, reference, output = map(checked_path,
        (repository_root, preparation_dir, package_dir, reference_path, output_dir))
    require(checked_path(__file__) == repository / "src/knee/clinical_fivefold.py",
        "clinical_fivefold_module_must_match_published_checkout")
    require(repository.is_dir() and proof.is_dir() and package.is_dir() and reference.is_file()
        and not output.exists() and output.parent.is_dir(), "existing_clinical_inputs_and_new_private_output_required")
    protected = (repository, proof, package, reference.parent, *map(checked_path, completed_jobs.values()))
    require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in protected),
        "clinical_output_must_not_overlap_sources")
    verified = previous.validate_preparation(proof)
    source_hashes = sources(repository)
    _, catalog, plan, registry = load_release(repository)
    require([row["id"] for row in registry["clinical"]["candidates"]] == list(MODEL_IDS)
        and plan["inputs"]["clinical_features"] == ["AGEYEARS", "SEX", "BMI"],
        "same_three_registered_clinical_models_and_features_required")
    from knee.model_inputs_colab import _reference
    reference_record = preparation.read_json(reference)
    _reference(reference_record)
    package_sha = reference_record["sources_sha256"]["development_package"]
    from knee.training_data import load_sealed_inputs
    loaded = load_sealed_inputs(package, reference, repository, package_sha)
    expected_rows = loaded["manifest"]["rows"]
    participants = len({row["src_subject_id"] for row in expected_rows})
    require(len(expected_rows) == reference_record["development_knees"]
        and participants == reference_record["development_participants"], "same_sealed_clinical_development_membership_required")
    del loaded
    sealed = {path: sha_file(path) for path in (package / "development_package.zip", package / "handoff.json",
        reference, proof / previous.previous.CORE_FILENAME, proof / previous.previous.SUMMARY_FILENAME)}
    from knee.training_runner import execute, save_json
    output.mkdir(exist_ok=False)
    model_reports = {}
    for model_id in MODEL_IDS:
        model_root = output / model_id
        model_root.mkdir(exist_ok=False)
        jobs, reports = {}, []
        for fold in FOLDS:
            print(f"Clínico {model_id}: pliegue {fold + 1}/5; "
                + ("verificando recuperación completa." if (model_id, fold) in completed_jobs else "entrenando solo datos clínicos."),
                file=sys.stderr, flush=True)
            require(sources(repository) == source_hashes and all(sha_file(path) == checksum
                for path, checksum in sealed.items()), "sealed_clinical_inputs_changed_between_jobs")
            request = make_request(package_sha, model_id, fold)
            authority = _authority(request, registry, approval_statement)
            job_root = model_root / ("fold_" + str(fold))
            reused = (model_id, fold) in completed_jobs
            if reused:
                prior = checked_path(completed_jobs[(model_id, fold)])
                validate_completed_job(prior, request, plan, registry, expected_rows)
                shutil.copytree(prior, job_root)
            else:
                execute(request, authority, repository_root=repository, package_dir=package,
                    reference_path=reference, weights_root=None, output_dir=job_root,
                    device="cpu", allow_download=False)
            job, engine = validate_completed_job(job_root, request, plan, registry, expected_rows)
            jobs[fold] = (job_root, job)
            report = {"validation_fold": fold, "job_record_sha256": job["record_sha256"],
                "metrics": job["metrics"], "reused_authenticated_complete_job": reused,
                "saved_model_predictions_reproduced": True, "device": "cpu"}
            if engine is not None:
                report.update(completed_epochs=engine["completed_epochs"], best_epoch=engine["best_epoch"],
                    optimizer_updates=engine["optimizer_updates"], resource_profile=engine["resource_profile"])
            reports.append(report)
            if job_completed is not None:
                job_completed(model_id, fold, job_root, job)
            print(f"{model_id}, pliegue {fold + 1}/5 verificado y conservado.", file=sys.stderr, flush=True)
        selection, descriptive, predictions = aggregate_predictions(jobs, expected_rows)
        save_json(model_root / "out_of_fold_predictions.json", {"scope": "development_only", "model_id": model_id,
            "rows": predictions, "pooled_metric_role": "descriptive_only_not_selection_metric"})
        model_reports[model_id] = {"fold_results": reports, "fivefold_selection_metrics": selection,
            "pooled_oof_metrics_descriptive_only": descriptive, "out_of_fold_predictions": len(predictions),
            "every_development_knee_predicted_once": True}
    require(sources(repository) == source_hashes and all(sha_file(path) == checksum
        for path, checksum in sealed.items()), "sealed_clinical_inputs_changed_during_execution")
    previous.validate_preparation(proof)
    load_release(repository)
    artifacts = {path.relative_to(output).as_posix(): sha_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()}
    require(len(artifacts) == 73, "exact_fifteen_clinical_jobs_and_three_oof_artifacts_required")
    result = {"version": VERSION, "status": STATUS, "execution_environment": execution_environment,
        "colab_runtime_executed": True, "local_rehearsal": False, "scope": "development_only",
        "model_ids": list(MODEL_IDS), "stage": "stage_1", "seed": SEED, "folds_completed": list(FOLDS),
        "job_count": 15, "development_participants": participants, "development_knees": len(expected_rows),
        "clinical_features": ["AGEYEARS", "SEX", "BMI"], "model_results": model_reports,
        "constant_training_prevalence_reference": constant_reference(expected_rows),
        "approval_source": "direct_investigator_authorization", "approval_statement": approval_statement,
        "training_authorized": True, "training_executed": True, "optimizer_created": True,
        "backward_executed": True, "participant_data_opened": True, "cpu_only_guard_passed": True,
        "image_models_trained": False, "multimodal_models_trained": False,
        "image_model_inference_executed": False, "pretrained_weights_loaded": False,
        "image_arrays_authenticated_for_package_integrity": True,
        "full_encoder_training_authorized": False, "encoder_unfreezing_executed": False,
        "calibration_performed": False, "threshold_selected": False,
        "reserved_test_accessed": False, "reserved_test_predictive_evaluation_opened": False,
        "originals_modified": False, "partitions_modified": False, "fold_optimizer_resume_supported": False,
        "completed_job_reuse_supported": True, "eligible_for_model_selection": True,
        "predictive_performance_claimed": False, "unbiased_final_performance_claimed": False,
        "clinical_preprocessing_fitted_inside_training_folds_only": True,
        "selection_aggregation": "unweighted_arithmetic_mean_of_five_participant_held_out_fold_metrics",
        "every_development_knee_predicted_once_per_model": True, "participant_disjoint_folds_verified": True,
        "development_package_sha256": package_sha, "model_input_reference_sha256": sealed[reference],
        "previous_colab34_summary_sha256": sealed[proof / previous.previous.SUMMARY_FILENAME],
        "release_component_canonical_sha256": PINS, "source_files_sha256": source_hashes,
        "artifact_files_sha256": artifacts, "environment": environment,
        "next_step": "review_clinical_development_comparators_before_separate_next_model_authority",
        "limitations": ["Development folds select models and do not establish unbiased final-test performance.",
            "Only the three registered clinical models on the sealed five folds are trained.",
            "No calibration, threshold selection, image training or reserved-test evaluation is performed.",
            "Complete authenticated jobs may be reused; interrupted jobs restart without optimizer resume."],
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    save_json(output / FILENAME, result)
    require(preparation.read_json(output / FILENAME) == result, "clinical_fivefold_result_readback_failed")
    return result
