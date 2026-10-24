"""One separately approved DenseNet121 frozen-encoder development campaign.

The investigator, not notebook preparation, calls this real training path in
Colab. Complete folds may be reauthenticated after a disconnect; a partial
fold is never resumed. No reserved-test, tuning, calibration or refit API exists.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import json
from pathlib import Path
import shutil
import sys

from knee.colab_data import checked_path, require, sha_file, valid_sha
from knee.final_model_release import load_release, PINS, RELEASE_PATH
from knee.training_authority import digest, REQUEST_VERSION, VERSION as AUTHORITY_VERSION
from knee import training_preparation as preparation
from knee import training_smoke as smoke

VERSION = "densenet121_frozen_fivefold_development_v1.0"
STATUS = "frozen_fivefold_development_complete"
FILENAME = "densenet_fivefold.json"
MODEL_ID = "image_densenet121"
FOLDS = (0, 1, 2, 3, 4)
SEED = 2026
MAX_ENGINE_RECORD_BYTES = 32 * 1024**2
APPROVAL_STATEMENT = (
    "Autorizo solo el entrenamiento de DenseNet121 con codificador congelado en los cinco "
    "pliegues de desarrollo, semilla 2026 y receta registrada; sin abrir prueba reservada, "
    "sin descongelar el codificador ni entrenar otros modelos."
)


def make_request(package_sha256, validation_fold):
    require(valid_sha(package_sha256) and type(validation_fold) is int and validation_fold in FOLDS,
        "fixed_development_package_and_fivefold_index_required")
    return {"version": REQUEST_VERSION, "operation": "development_fold", "model_id": MODEL_ID,
        "image_encoder_id": MODEL_ID, "stage": "stage_1", "validation_fold": validation_fold,
        "seed": SEED, "development_package_sha256": package_sha256, "image_micro_batch_size": 2,
        "selection_evidence_sha256": None, "fold_image_job_sha256": None}


def sources(repository):
    """Retain completed34/35 lineage and bind every new training entry point."""
    result = smoke._sources(checked_path(repository))
    for name in ("src/knee/densenet_fivefold.py", "src/knee/training_smoke_evidence.py"):
        result[name] = sha_file(checked_path(repository) / name)
    return result


def _authority(request, registry, catalog, statement, verified_smoke):
    require(statement == APPROVAL_STATEMENT, "exact_frozen_fivefold_training_approval_required")
    # Reuse the already scoped catalog review, never turn a runtime check into
    # an unrestricted redistribution or external-pretraining disjointness claim.
    prior = smoke._authority(smoke.make_request(request["development_package_sha256"]),
        registry, catalog, smoke.APPROVAL_STATEMENT)
    require(verified_smoke.get("model_id") == MODEL_ID
        and verified_smoke.get("training_smoke_test_passed") is True
        and verified_smoke.get("gpu_resource_profile_recorded") is True,
        "same_densenet_actual_gpu_smoke_and_resource_evidence_required")
    return {**prior, "approval_statement": statement, "request_sha256": digest(request),
        "eligibility_gates": {name: True for name in registry["eligibility_gates"]},
        "full_encoder_training_authorized": False}


def _finite(value, *, minimum=0, maximum=None):
    return type(value) in {int, float} and math.isfinite(value) and value >= minimum \
        and (maximum is None or value <= maximum)


def read_engine_record(path):
    """Bound the larger 60-epoch AMP audit without weakening old proof limits."""
    path = checked_path(path)
    require(path.is_file() and 0 < path.stat().st_size <= MAX_ENGINE_RECORD_BYTES,
        "bounded_complete_training_engine_record_required")
    from knee.colab_data import _json_pairs
    return json.loads(path.read_bytes(), object_pairs_hook=_json_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_training_engine_record")))


def _validate_amp(engine):
    require(engine.get("numerical_repair") == {
        "version": smoke.AMP_REPAIR_VERSION, "maximum_backoffs_per_effective_group": 16,
        "same_effective_group_replayed": True, "nonfinite_gradients_never_clipped": True},
        "exact_preserved_same_group_amp_repair_required")
    expected, previous, backoffs = [], None, 0
    fields = {"update_index_within_epoch", "start_scale", "successful_scale", "scale_after_update",
        "backoffs", "attempts", "gradient_norm_before_clip"}
    for epoch in engine["history"]:
        audit = epoch.get("amp_scale_audit")
        require(isinstance(audit, list) and len(audit) == epoch["optimizer_updates"],
            "every_full_fold_optimizer_update_requires_amp_audit")
        epoch_backoffs = 0
        for index, row in enumerate(audit, start=1):
            require(isinstance(row, dict) and set(row) == fields
                and type(row["update_index_within_epoch"]) is int
                and row["update_index_within_epoch"] == index
                and type(row["backoffs"]) is int and 0 <= row["backoffs"] <= 16
                and type(row["attempts"]) is int and row["attempts"] == row["backoffs"] + 1
                and all(_finite(row[key], minimum=1) for key in
                    ("start_scale", "successful_scale", "scale_after_update"))
                and _finite(row["gradient_norm_before_clip"]), "finite_bounded_full_fold_amp_audit_required")
            require(row["successful_scale"] == row["start_scale"] * .5 ** row["backoffs"]
                and row["scale_after_update"] in {row["successful_scale"], 2 * row["successful_scale"]}
                and (previous is None or row["start_scale"] == previous),
                "continuous_successful_scale_or_registered_gradscaler_growth_required")
            previous = row["scale_after_update"]
            epoch_backoffs += row["backoffs"]
            expected.append({"epoch": epoch["epoch"], **row})
        require(type(epoch.get("amp_overflow_replays")) is int
            and epoch["amp_overflow_replays"] == epoch_backoffs, "epoch_amp_replay_accounting_mismatch")
        backoffs += epoch_backoffs
    require(engine.get("amp_scale_audit") == expected
        and type(engine.get("amp_overflow_replays")) is int
        and engine["amp_overflow_replays"] == backoffs, "full_fold_amp_replay_accounting_mismatch")


def validate_completed_fold(directory, request, plan, registry, initial_state, expected_rows):
    """Authenticate artifacts, fold identities, full epochs and unchanged encoder."""
    from knee.training_runner import completed_job, JOB_FILENAME
    from knee.training_engine import EarlyStopping, binary_metrics, learning_rate_factor, VERSION as ENGINE_VERSION
    root = checked_path(directory)
    require(root.is_dir() and all(checked_path(path).parent == root and path.is_file()
        for path in root.iterdir()), "only_plain_private_fold_artifact_files_required")
    job = completed_job(root, request["development_package_sha256"], plan, registry)
    require(job.get("request") == request and job.get("eligibility_gates") == {
        name: True for name in registry["eligibility_gates"]}
        and set(path.name for path in root.iterdir()) == smoke.JOB_FILES | {JOB_FILENAME},
        "exact_frozen_densenet_full_fold_and_private_artifacts_required")
    training = [row for row in expected_rows if row["cv_fold"] != request["validation_fold"]]
    validation = [row for row in expected_rows if row["cv_fold"] == request["validation_fold"]]
    train_ids = sorted([[row["src_subject_id"], row["side"]] for row in training])
    participants = sorted({row[0] for row in train_ids})
    require(training and validation and not set(participants).intersection(
        row["src_subject_id"] for row in validation), "participant_disjoint_full_fold_required")
    expected_provenance = {"fit_scope": "development_training_fold",
        "validation_fold": request["validation_fold"], "train_identity_sha256": digest(train_ids),
        "trained_participants": participants, "reserved_test_accessed": False}
    require(job.get("weights_provenance") == expected_provenance,
        "full_fold_weights_require_exact_training_identity_provenance")
    from knee.clinical_fold_inputs import fit_clinical_fold
    require(preparation.read_json(root / "clinical_state.json") == fit_clinical_fold(
        expected_rows, request["validation_fold"]), "exact_training_fold_clinical_state_required")
    predictions = preparation.read_json(root / "best_validation_predictions.json")
    require(predictions["identities"] == [[row["src_subject_id"], row["side"]] for row in validation]
        and predictions["targets"] == [row["PROGRESION_48M"] for row in validation],
        "exact_held_out_fold_predictions_and_targets_required")
    engine = read_engine_record(root / "training_record.json")
    require(engine.get("version") == ENGINE_VERSION and engine.get("model_id") == MODEL_ID
        and engine.get("stage") == "stage_1" and engine.get("seed") == SEED
        and engine.get("smoke") is False and engine.get("scope") == "development_only"
        and engine.get("status") == "training_fold_complete"
        and engine.get("eligible_for_model_selection") is True
        and engine.get("reserved_test_accessed") is False
        and engine.get("optimizer_created") is True and engine.get("training_executed") is True
        and engine.get("micro_batch_size") == 2 and engine.get("effective_batch_size") == 16
        and engine.get("device") == "cuda:0" and engine.get("precision") == "float16_autocast_grad_scaler"
        and engine.get("scheduler_horizon_epochs") == 60
        and engine.get("gpu_resource_profile_recorded") is True
        and engine.get("restored_best_checkpoint") is True and engine.get("resume_supported") is False
        and engine.get("optimizer_state_saved") is False,
        "complete_frozen_registered_densenet_gpu_fold_required")
    provenance = {"validation_fold": request["validation_fold"],
        "development_package_sha256": request["development_package_sha256"],
        "train_identity_sha256": digest(train_ids), "request_sha256": digest(request),
        "training_knees": len(training), "validation_knees": len(validation),
        "trained_participants": participants}
    require(engine.get("provenance") == provenance, "full_fold_engine_identity_binding_mismatch")
    positives = sum(row["PROGRESION_48M"] for row in training)
    require(0 < positives < len(training) and _finite(engine.get("positive_weight"), minimum=1e-30)
        and math.isclose(engine["positive_weight"], (len(training) - positives) / positives,
            rel_tol=1e-12, abs_tol=0), "training_only_full_fold_positive_class_weight_required")
    history = engine.get("history")
    require(isinstance(history, list) and 1 <= len(history) <= 60
        and type(engine.get("completed_epochs")) is int and engine["completed_epochs"] == len(history),
        "one_to_sixty_complete_training_epochs_required")
    monitor, stop = EarlyStopping(patience=8), False
    updates_per_epoch = math.ceil(len(training) / 16)
    for epoch, item in enumerate(history, start=1):
        require(not stop and item.get("epoch") == epoch and item.get("phase") == "head_only"
            and type(item.get("training_knees_seen")) is int and item["training_knees_seen"] == len(training)
            and type(item.get("optimizer_updates")) is int and item["optimizer_updates"] == updates_per_epoch
            and _finite(item.get("weighted_loss_per_knee"))
            and item.get("learning_rate_factor") == learning_rate_factor(epoch)
            and isinstance(item.get("validation"), dict)
            and set(item["validation"]) == {"average_precision_score", "roc_auc", "brier"}
            and all(_finite(value, maximum=1) for value in item["validation"].values()),
            "full_epochs_head_only_and_registered_schedule_required")
        improved, stop = monitor.update(epoch, item["validation"]["average_precision_score"])
        require(item.get("selected_best") is improved, "strict_ap_best_epoch_replay_required")
    require((stop or len(history) == 60)
        and engine.get("best_epoch") == monitor.best_epoch
        and engine.get("best_validation_ap") == monitor.best_score
        and job["metrics"] == history[monitor.best_epoch - 1]["validation"]
        and job["metrics"] == binary_metrics(predictions["targets"], predictions["logits"])
        and type(engine.get("optimizer_updates")) is int
        and engine["optimizer_updates"] == updates_per_epoch * len(history),
        "only_registered_early_stop_or_sixty_epochs_with_best_restored_allowed")
    require(engine.get("checkpoint_sha256") == {name: job["files_sha256"][name]
        for name in ("best_model.safetensors", "last_model.safetensors")},
        "restored_full_fold_checkpoint_hashes_required")
    _validate_amp(engine)
    resource = engine.get("resource_profile", {})
    require(resource.get("device_type") == "cuda" and resource.get("cuda_peak_measured") is True
        and resource.get("torch_version") == "2.6.0+cu124" and resource.get("cuda_version") == "12.4"
        and isinstance(resource.get("device_name"), str) and bool(resource["device_name"])
        and all(type(resource.get(key)) is int and resource[key] > 0 for key in
            ("total_memory_bytes", "peak_memory_allocated_bytes", "peak_memory_reserved_bytes"))
        and resource["peak_memory_allocated_bytes"] <= resource["peak_memory_reserved_bytes"] <= resource["total_memory_bytes"]
        and _finite(resource.get("elapsed_seconds"), minimum=1e-30), "measured_full_fold_gpu_profile_required")
    frozen = smoke._verify_frozen_encoder(root, initial_state)
    return job, engine, frozen


def aggregate_predictions(jobs, expected_rows):
    """Five unweighted fold means; pooled OOF is descriptive, never selection."""
    from knee.training_engine import aggregate_fold_metrics, binary_metrics
    require(isinstance(jobs, dict) and set(jobs) == set(FOLDS), "all_five_fold_jobs_required_before_aggregation")
    rows, metric_rows, seen, participant_folds = [], [], set(), {}
    for fold in FOLDS:
        directory, job = jobs[fold]
        require(job["request"]["validation_fold"] == fold, "fold_job_identity_mismatch")
        predictions = preparation.read_json(checked_path(directory) / "best_validation_predictions.json")
        expected = [row for row in expected_rows if row["cv_fold"] == fold]
        require(predictions["identities"] == [[row["src_subject_id"], row["side"]] for row in expected]
            and predictions["targets"] == [row["PROGRESION_48M"] for row in expected],
            "exact_fivefold_out_of_fold_coverage_required")
        metric_rows.append({"validation_fold": fold, **job["metrics"]})
        for identity, target, logit in zip(predictions["identities"], predictions["targets"], predictions["logits"]):
            key = tuple(identity)
            require(key not in seen and participant_folds.get(identity[0], fold) == fold,
                "duplicate_oof_knee_or_participant_cross_fold_forbidden")
            seen.add(key)
            participant_folds[identity[0]] = fold
            rows.append({"src_subject_id": identity[0], "side": identity[1], "validation_fold": fold,
                "target": target, "logit": logit})
    require(seen == {(row["src_subject_id"], row["side"]) for row in expected_rows}
        and len(rows) == len(expected_rows), "every_development_knee_requires_exactly_one_oof_prediction")
    return aggregate_fold_metrics(metric_rows), binary_metrics(
        [row["target"] for row in rows], [row["logit"] for row in rows]), rows


def run(repository_root, smoke_dir, package_dir, reference_path, weights_root, output_dir, *,
        approval_statement, execution_environment="google_colab_gpu", fold_completed=None, completed_folds=None):
    """Train exactly five fresh frozen DenseNet heads, preserving complete folds."""
    require(approval_statement == APPROVAL_STATEMENT, "exact_frozen_fivefold_training_approval_required")
    require(fold_completed is None or callable(fold_completed), "fold_completion_callback_must_be_callable")
    completed_folds = {} if completed_folds is None else completed_folds
    require(isinstance(completed_folds, dict) and all(type(fold) is int and fold in FOLDS
        for fold in completed_folds), "only_exact_completed_fold_directories_may_be_reused")
    environment = smoke.verify_gpu_environment(execution_environment)
    repository, proof, package, reference, weights, output = map(checked_path,
        (repository_root, smoke_dir, package_dir, reference_path, weights_root, output_dir))
    require(checked_path(__file__) == repository / "src/knee/densenet_fivefold.py",
        "fivefold_module_must_match_published_checkout")
    require(repository.is_dir() and proof.is_dir() and package.is_dir() and reference.is_file()
        and weights.is_dir() and not output.exists() and output.parent.is_dir(),
        "existing_private_inputs_and_new_fivefold_output_required")
    protected = (repository, proof, package, reference.parent, weights, *map(checked_path, completed_folds.values()))
    require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in protected)
        and all(not weights.is_relative_to(path) and not path.is_relative_to(weights)
            for path in (repository, proof, package, reference.parent)),
        "fivefold_output_and_public_cache_must_not_overlap_sources")
    from knee.training_smoke_evidence import validate_smoke
    verified = validate_smoke(proof, repository_root=repository)
    source_hashes = sources(repository)
    _, catalog, plan, registry = load_release(repository)
    from knee.model_inputs_colab import _reference
    reference_record = preparation.read_json(reference)
    _reference(reference_record)
    package_sha = reference_record["sources_sha256"]["development_package"]
    require(verified["development_package_sha256"] == package_sha
        and verified["model_input_reference_sha256"] == sha_file(reference),
        "same_verified_smoke_development_package_and_input_reference_required")
    from knee.training_data import load_sealed_inputs
    loaded = load_sealed_inputs(package, reference, repository, package_sha)
    expected_rows = loaded["manifest"]["rows"]
    participants = len({row["src_subject_id"] for row in expected_rows})
    require(len(expected_rows) == reference_record["development_knees"]
        and participants == reference_record["development_participants"],
        "same_sealed_development_participants_and_knees_required")
    del loaded
    from knee.final_models_preflight import checkpoint_files
    checkpoint_paths = checkpoint_files(MODEL_ID, catalog, weights, allow_download=True)
    from knee.training_models import build_model
    initial_model, _ = build_model(MODEL_ID, plan, catalog, checkpoint_paths, seed=SEED)
    initial = {name: tensor.detach().cpu().clone() for name, tensor in initial_model.state_dict().items()}
    del initial_model
    sealed = {path: sha_file(path) for path in (package / "development_package.zip", package / "handoff.json",
        reference, *checkpoint_paths, proof / "training_smoke.json", proof / "colab_training_smoke_check.json")}
    from knee.training_runner import execute, save_json
    output.mkdir(exist_ok=False)
    jobs, reports = {}, []
    for fold in FOLDS:
        print(f"DenseNet121 congelado: pliegue {fold + 1}/5; "
            + ("verificando recuperación completa." if fold in completed_folds else "entrenando solo la cabeza."),
            file=sys.stderr, flush=True)
        require(sources(repository) == source_hashes and all(sha_file(path) == checksum
            for path, checksum in sealed.items()), "sealed_fivefold_inputs_changed_between_folds")
        request = make_request(package_sha, fold)
        authority = _authority(request, registry, catalog, approval_statement, verified)
        job_root = output / ("fold_" + str(fold))
        reused = fold in completed_folds
        if reused:
            prior = checked_path(completed_folds[fold])
            validate_completed_fold(prior, request, plan, registry, initial, expected_rows)
            shutil.copytree(prior, job_root)
        else:
            execute(request, authority, repository_root=repository, package_dir=package,
                reference_path=reference, weights_root=weights, output_dir=job_root,
                device="cuda:0", allow_download=False)
        job, engine, frozen = validate_completed_fold(job_root, request, plan, registry, initial, expected_rows)
        jobs[fold] = (job_root, job)
        reports.append({"validation_fold": fold, "job_record_sha256": job["record_sha256"],
            "metrics": job["metrics"], "completed_epochs": engine["completed_epochs"],
            "optimizer_updates": engine["optimizer_updates"], "best_epoch": engine["best_epoch"],
            "amp_overflow_replays": engine["amp_overflow_replays"], "resource_profile": engine["resource_profile"],
            "reused_authenticated_complete_fold": reused, **frozen})
        if fold_completed is not None:
            fold_completed(fold, job_root, job)
        print(f"Pliegue {fold + 1}/5 verificado y conservado; {engine['completed_epochs']} épocas completas.",
            file=sys.stderr, flush=True)
    selection, descriptive, predictions = aggregate_predictions(jobs, expected_rows)
    require(sources(repository) == source_hashes and all(sha_file(path) == checksum
        for path, checksum in sealed.items()), "sealed_fivefold_inputs_changed_during_execution")
    validate_smoke(proof, repository_root=repository)
    load_release(repository)
    save_json(output / "out_of_fold_predictions.json", {"scope": "development_only", "rows": predictions,
        "pooled_metric_role": "descriptive_only_not_selection_metric"})
    artifacts = {path.relative_to(output).as_posix(): sha_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()}
    result = {"version": VERSION, "status": STATUS, "execution_environment": execution_environment,
        "colab_runtime_executed": True, "local_rehearsal": False, "scope": "development_only",
        "model_id": MODEL_ID, "stage": "stage_1", "seed": SEED, "folds_completed": list(FOLDS),
        "development_participants": participants, "development_knees": len(expected_rows),
        "approval_source": "direct_investigator_authorization", "approval_statement": approval_statement,
        "training_authorized": True, "training_executed": True, "optimizer_created": True,
        "backward_executed": True, "participant_data_opened": True,
        "encoder_frozen": True, "frozen_encoder_state_unchanged": True,
        "full_encoder_training_authorized": False, "encoder_unfreezing_executed": False,
        "other_models_trained": False, "calibration_performed": False, "threshold_selected": False,
        "reserved_test_accessed": False, "reserved_test_predictive_evaluation_opened": False,
        "originals_modified": False, "partitions_modified": False, "fold_optimizer_resume_supported": False,
        "completed_fold_reuse_supported": True, "eligible_for_model_selection": True,
        "predictive_performance_claimed": False, "unbiased_final_performance_claimed": False,
        "micro_batch_size": 2, "effective_batch_size": 16, "maximum_epochs": 60, "early_stopping_patience": 8,
        "selection_aggregation": "unweighted_arithmetic_mean_of_five_participant_held_out_fold_metrics",
        "fivefold_selection_metrics": selection, "pooled_oof_metrics_descriptive_only": descriptive,
        "out_of_fold_predictions": len(predictions), "every_development_knee_predicted_once": True,
        "participant_disjoint_folds_verified": True, "fold_results": reports,
        "development_package_sha256": package_sha, "model_input_reference_sha256": sealed[reference],
        "previous_colab35_summary_sha256": sealed[proof / "colab_training_smoke_check.json"],
        "release_component_canonical_sha256": PINS, "source_files_sha256": source_hashes,
        "artifact_files_sha256": artifacts, "public_checkpoint_files_sha256": {
            path.name: sealed[path] for path in checkpoint_paths},
        "eligibility_evidence": verified["eligibility_evidence"], "environment": environment,
        "next_step": "review_frozen_densenet_development_results_before_separate_next_model_authority",
        "limitations": ["Development folds select models; they are not unbiased final test performance.",
            "Only DenseNet121 frozen stage one is authorized; no unfreezing or other models are authorized.",
            "Complete authenticated folds may be reused, but interrupted folds restart from external weights.",
            "No calibration, threshold selection, reserved-test access or complete-cohort refit is performed.",
            "External pretraining disjointness is unproven; KNEEL OAI overlap and license limits remain."],
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    save_json(output / FILENAME, result)
    require(preparation.read_json(output / FILENAME) == result, "fivefold_result_readback_failed")
    return result
