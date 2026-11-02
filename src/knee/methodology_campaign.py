"""Bounded v1.2 development campaigns; preparing this module grants no training.

Only the investigator's later explicit Colab confirmation can enter ``run``.
Historical v1.1 jobs are never recovered into these campaigns. Reserved-test,
calibration, threshold, final-refit and encoder-unfreezing operations are absent.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys

from knee import clinical_scenarios as clinical
from knee import methodology_release as release
from knee import methodology_inputs_colab as inputs
from knee import methodology_preparation as preparation
from knee.aligned_training.authority import digest, validate_request, validate_authority, VERSION as AUTHORITY_VERSION
from knee.colab_data import checked_path, require, sha_file, valid_sha
from knee.training_preparation import read_json

VERSION = "methodology_v1_2_development_campaign_v1.0"
STATUS = "methodology_development_campaign_complete"
FILENAME = "methodology_campaign.json"
IMAGE_CAMPAIGNS = ("densenet_frozen", "remaining_encoders_frozen")
CAMPAIGNS = ("clinical", *IMAGE_CAMPAIGNS)
REMAINING_IMAGE_IDS = ("image_vit_b16", "image_convnextv2_tiny", "image_skelex",
    "image_biomedclip", "image_mambaout_tiny")
CLINICAL_IDS = preparation.CLINICAL_IDS
FOLDS = tuple(range(5))
SEED = 2026
CONFIRMATIONS = {"clinical": "AUTORIZO CLINICOS V1.2 5 PLIEGUES",
    "densenet_frozen": "AUTORIZO DENSENET V1.2 5 PLIEGUES",
    "remaining_encoders_frozen": "AUTORIZO CODIFICADORES V1.2 5 PLIEGUES"}
APPROVAL_STATEMENTS = {
    "clinical": "Autorizo solo los tres algoritmos clínicos y sus tres escenarios basales en los cinco pliegues de desarrollo, receta v1.2 y semilla 2026; sin abrir la prueba reservada ni realizar calibración, umbral o ajuste final.",
    "densenet_frozen": "Autorizo solo DenseNet121 con codificador congelado y aumento de intensidad v1.2 en los cinco pliegues de desarrollo, semilla 2026; sin abrir la prueba reservada, descongelar el codificador ni realizar calibración, umbral o ajuste final.",
    "remaining_encoders_frozen": "Autorizo solo ViT-B/16, ConvNeXtV2-Tiny, Skelex, BiomedCLIP y MambaOut-Tiny: primero una prueba GPU de dos actualizaciones de la cabeza por codificador en el pliegue 0; después, reiniciando los pesos y la semilla, los cinco pliegues de desarrollo con codificadores congelados y aumento de intensidad v1.2, semilla 2026; sin repetir DenseNet121, incorporar DINOv3, abrir la prueba reservada, descongelar codificadores ni ejecutar fusión, calibración, umbral o ajuste final."}


def image_models(campaign):
    require(campaign in IMAGE_CAMPAIGNS, "registered_frozen_image_campaign_required")
    return ("image_densenet121",) if campaign == "densenet_frozen" else REMAINING_IMAGE_IDS


def requests(campaign, package_sha256, sidecar_sha256, plan, registry):
    require(campaign in CAMPAIGNS and valid_sha(package_sha256) and valid_sha(sidecar_sha256),
        "exact_registered_methodology_campaign_required")
    if campaign == "clinical":
        return [preparation.make_request(package_sha256, sidecar_sha256, scenario, model, fold, plan, registry)
            for scenario in clinical.SCENARIOS for model in CLINICAL_IDS for fold in FOLDS]
    from knee.training_augmentation import VERSION as AUGMENTATION_VERSION
    result = []
    for model_id in image_models(campaign):
        for fold in FOLDS:
            request = preparation.make_request(package_sha256, sidecar_sha256, "clinical_main",
                "clinical_mlp", fold, plan, registry)
            request.update(model_id=model_id, image_encoder_id=model_id,
                augmentation_policy=AUGMENTATION_VERSION)
            validate_request(request, plan, registry)
            result.append(request)
    return result


def job_key(request):
    return f'{request["clinical_scenario"]}__{request["model_id"]}__fold_{request["validation_fold"]}'


def validate_preparation(directory, repository=None):
    """Authenticate the saved, real Colab38 preflight, including every artifact."""
    root = checked_path(directory)
    summary = read_json(root / inputs.SUMMARY_FILENAME)
    core = read_json(root / inputs.CORE_FILENAME)
    inputs.validate_summary(summary)
    inputs.validate_result(core)
    require(summary.get("core_record_sha256") == sha_file(root / inputs.CORE_FILENAME)
        and all(summary.get(key) == core[key] for key in ("development_package_sha256",
            "baseline_projection_sha256", "modeling_release_sha256", "training_plan_sha256",
            "development_participants", "development_knees")), "methodology38_summary_core_mismatch")
    expected = {**core["artifacts_sha256"], inputs.CORE_FILENAME: summary["core_record_sha256"],
        inputs.SUMMARY_FILENAME: sha_file(root / inputs.SUMMARY_FILENAME)}
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file()}
    require(set(actual) == set(expected) and len(actual) == 127
        and all(checked_path(path).is_relative_to(root) and path.stat().st_size <= inputs.MAX_OUTPUT_FILE_BYTES
            and sha_file(path) == expected[name] for name, path in actual.items()),
        "complete_unchanged_real_colab38_artifacts_required")
    require(core["modeling_release_sha256"] == release.PINS[release.RELEASE_PATH]
        and core["training_plan_sha256"] == release.PINS[release.PLAN_PATH],
        "preparation_must_use_current_methodology_recipe")
    if repository is not None:
        repository = checked_path(repository)
        require(core["implementation_sha256"] == {name: sha_file(repository / name) for name in inputs.SOURCE_FILES},
            "methodology38_implementation_changed")
    prepared = read_json(root / preparation.FILENAME)
    sidecar = read_json(root / "baseline_sidecar.json")
    require(prepared.get("version") == preparation.VERSION and prepared.get("status") == preparation.STATUS
        and prepared.get("baseline_sidecar_sha256") == sidecar.get("record_sha256")
        and prepared.get("training_authorized") is False and prepared.get("execution_authorities_created") == 0,
        "prepared_baseline_sidecar_without_training_authority_required")
    return summary, core, sidecar


def authority(request, campaign, statement, plan, registry, *, image_eligibility_verified=False,
        encoder_smoke_evidence=None):
    """Called only after the campaign's exact future investigator confirmation."""
    require(campaign in CAMPAIGNS and statement == APPROVAL_STATEMENTS[campaign],
        "exact_methodology_campaign_training_confirmation_required")
    expected = requests(campaign, request["development_package_sha256"],
        request.get("baseline_sidecar_sha256") or "0" * 64, plan, registry)
    require(request in expected and (campaign not in IMAGE_CAMPAIGNS or image_eligibility_verified is True),
        "exact_campaign_request_and_verified_image_eligibility_required")
    if campaign == "remaining_encoders_frozen":
        _require_smoke_receipt(encoder_smoke_evidence, request["model_id"], request["development_package_sha256"])
    record = {"version": AUTHORITY_VERSION, "training_authorized": True,
        "reserved_test_access_authorized": False, "approval_source": "direct_investigator_authorization",
        "approval_statement": statement, "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_sha256": digest(request), "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "eligibility_gates": {key: campaign in IMAGE_CAMPAIGNS for key in registry["eligibility_gates"]},
        "full_encoder_training_authorized": False}
    validate_authority(record, request, plan, registry)
    return record


def _finite(value, minimum=0, maximum=None):
    return type(value) in {int, float} and math.isfinite(value) and value >= minimum \
        and (maximum is None or value <= maximum)


def _validate_neural(root, request, job, rows, initial_state=None):
    from knee.aligned_training.engine import EarlyStopping, learning_rate_factor, VERSION as ENGINE_VERSION
    from knee.densenet_fivefold import read_engine_record, _validate_amp
    fold = request["validation_fold"]
    train = [row for row in rows if row["cv_fold"] != fold]
    validation = [row for row in rows if row["cv_fold"] == fold]
    train_ids = sorted([[row["src_subject_id"], row["side"]] for row in train])
    image = request["image_encoder_id"] is not None
    engine = read_engine_record(root / "training_record.json")
    require(engine.get("version") == ENGINE_VERSION and engine.get("status") == "training_fold_complete"
        and engine.get("scope") == "development_only" and engine.get("smoke") is False
        and engine.get("model_id") == request["model_id"] and engine.get("stage") == "stage_1"
        and engine.get("seed") == SEED and engine.get("training_executed") is True
        and engine.get("optimizer_created") is True and engine.get("eligible_for_model_selection") is True
        and engine.get("reserved_test_accessed") is False and engine.get("restored_best_checkpoint") is True
        and engine.get("resume_supported") is False and engine.get("optimizer_state_saved") is False
        and engine.get("scheduler_horizon_epochs") == 60,
        "complete_current_methodology_neural_job_required")
    require(engine.get("effective_batch_size") == (16 if image else 64)
        and engine.get("micro_batch_size") == (request["image_micro_batch_size"] if image else 64)
        and engine.get("device") == ("cuda:0" if image else "cpu")
        and engine.get("precision") == ("float16_autocast_grad_scaler" if image else "float32")
        and engine.get("gpu_resource_profile_recorded") is image,
        "registered_methodology_device_precision_and_batch_required")
    require(engine.get("provenance") == {"validation_fold": fold,
        "development_package_sha256": request["development_package_sha256"],
        "train_identity_sha256": digest(train_ids), "request_sha256": digest(request),
        "training_knees": len(train), "validation_knees": len(validation),
        "trained_participants": sorted({row[0] for row in train_ids})},
        "exact_methodology_engine_training_identity_required")
    positive = sum(row["PROGRESION_48M"] for row in train)
    require(0 < positive < len(train) and _finite(engine.get("positive_weight"), minimum=1e-30)
        and math.isclose(engine["positive_weight"], (len(train) - positive) / positive, rel_tol=1e-12),
        "training_only_methodology_class_weights_required")
    history = engine.get("history")
    require(isinstance(history, list) and 1 <= len(history) <= 60
        and engine.get("completed_epochs") == len(history), "complete_methodology_epochs_required")
    monitor, stop = EarlyStopping(patience=8), False
    updates = math.ceil(len(train) / (16 if image else 64))
    for epoch, item in enumerate(history, 1):
        require(not stop and item.get("epoch") == epoch and item.get("phase") == ("head_only" if image else "full")
            and item.get("training_knees_seen") == len(train) and item.get("optimizer_updates") == updates
            and _finite(item.get("weighted_loss_per_knee"))
            and item.get("learning_rate_factor") == learning_rate_factor(epoch)
            and isinstance(item.get("validation"), dict)
            and set(item["validation"]) == {"average_precision_score", "roc_auc", "brier"}
            and all(_finite(v, maximum=1) for v in item["validation"].values()),
            "whole_methodology_epochs_with_registered_schedule_required")
        improved, stop = monitor.update(epoch, item["validation"]["average_precision_score"])
        require(item.get("selected_best") is improved, "strict_methodology_ap_checkpoint_required")
    require((stop or len(history) == 60) and engine.get("best_epoch") == monitor.best_epoch
        and engine.get("best_validation_ap") == monitor.best_score
        and job["metrics"] == history[monitor.best_epoch - 1]["validation"]
        and engine.get("optimizer_updates") == updates * len(history)
        and engine.get("checkpoint_sha256") == {name: job["files_sha256"][name]
            for name in ("best_model.safetensors", "last_model.safetensors")},
        "completed_methodology_early_stop_and_checkpoint_accounting_required")
    _validate_amp(engine)
    if image:
        import torch
        from safetensors.torch import load_file
        require(isinstance(initial_state, dict) and bool(initial_state), "initial_verified_image_encoder_state_required")
        for filename in ("best_model.safetensors", "last_model.safetensors"):
            state = load_file(str(root / filename), device="cpu")
            require(set(state) == set(initial_state) and all(state[key].shape == original.shape
                and state[key].dtype == original.dtype and bool(torch.isfinite(state[key]).all())
                for key, original in initial_state.items()), "exact_finite_image_encoder_checkpoint_required")
            require(all(torch.equal(state[key], original) for key, original in initial_state.items()
                if key.startswith("encoder.")), "methodology_frozen_encoder_must_remain_exact")
            require(any(not torch.equal(state[key], original) for key, original in initial_state.items()
                if key.startswith("task_head.")), "methodology_task_head_must_have_updated")
    return engine


def validate_completed_job(directory, request, plan, registry, rows, sidecar, *, initial_state=None):
    from knee.aligned_training import runner
    root = checked_path(directory)
    job = runner.completed_job(root, request["development_package_sha256"], plan, registry)
    require(job["request"] == request and job.get("selection_evidence") is None
        and set(path.name for path in root.iterdir()) == set(job["files_sha256"]) | {runner.JOB_FILENAME}
        and all(checked_path(path).is_file() for path in root.iterdir()),
        "exact_current_methodology_job_artifacts_required")
    fold = request["validation_fold"]
    enriched = rows if request["clinical_scenario"] == "clinical_main" else clinical.join_baseline_sidecar(
        rows, sidecar, request["development_package_sha256"], expected_sidecar_sha256=request["baseline_sidecar_sha256"])
    state = clinical.fit_scenario_fold(enriched, fold, request["clinical_scenario"])
    require(read_json(root / "clinical_state.json") == state, "exact_training_only_scenario_state_required")
    training = [row for row in rows if row["cv_fold"] != fold]
    validation = [row for row in rows if row["cv_fold"] == fold]
    participants = sorted({row["src_subject_id"] for row in training})
    require(not set(participants) & {row["src_subject_id"] for row in validation}
        and job.get("weights_provenance") == {"fit_scope": "development_training_fold", "validation_fold": fold,
            "train_identity_sha256": state["train_identity_sha256"], "trained_participants": participants,
            "reserved_test_accessed": False}, "participant_disjoint_current_methodology_job_required")
    predictions = read_json(root / "best_validation_predictions.json")
    require(predictions["identities"] == [[row["src_subject_id"], row["side"]] for row in validation]
        and predictions["targets"] == [row["PROGRESION_48M"] for row in validation],
        "exact_methodology_held_out_predictions_required")
    expected_gates = {key: request["image_encoder_id"] is not None for key in registry["eligibility_gates"]}
    require(job.get("eligibility_gates") == expected_gates, "exact_campaign_eligibility_gates_required")
    engine = None
    if request["model_id"] not in runner.CLINICAL_IDS:
        engine = _validate_neural(root, request, job, rows, initial_state)
    return job, engine


def aggregate(jobs, rows):
    from knee.aligned_training.engine import aggregate_fold_metrics, binary_metrics
    require(set(jobs) == set(FOLDS), "all_five_current_methodology_folds_required")
    metrics, result, seen, participant_folds = [], [], set(), {}
    first = jobs[0][1]["request"]
    for fold in FOLDS:
        directory, job = jobs[fold]
        request = job["request"]
        require(all(request[key] == value for key, value in first.items() if key != "validation_fold")
            and request["validation_fold"] == fold, "one_recipe_scenario_and_algorithm_per_oof_required")
        predictions = read_json(directory / "best_validation_predictions.json")
        expected = [row for row in rows if row["cv_fold"] == fold]
        require(predictions["identities"] == [[r["src_subject_id"], r["side"]] for r in expected]
            and predictions["targets"] == [r["PROGRESION_48M"] for r in expected]
            and binary_metrics(predictions["targets"], predictions["logits"]) == job["metrics"],
            "authenticated_current_methodology_oof_required")
        metrics.append({"validation_fold": fold, **job["metrics"]})
        for identity, target, logit in zip(predictions["identities"], predictions["targets"], predictions["logits"]):
            key = tuple(identity)
            require(key not in seen and participant_folds.get(identity[0], fold) == fold,
                "duplicate_oof_knee_or_participant_leakage")
            seen.add(key)
            participant_folds[identity[0]] = fold
            result.append({"src_subject_id": identity[0], "side": identity[1], "validation_fold": fold,
                "target": target, "logit": logit})
    require(seen == {(r["src_subject_id"], r["side"]) for r in rows}, "exact_development_oof_coverage_required")
    return aggregate_fold_metrics(metrics), binary_metrics([r["target"] for r in result], [r["logit"] for r in result]), result


def _validate_remaining_eligibility(evidence, catalog):
    """Require each encoder's own authenticated weights and synthetic GPU check."""
    records = evidence.get("model_eligibility")
    require(isinstance(records, dict) and set(records) == set(REMAINING_IMAGE_IDS),
        "exact_remaining_encoder_eligibility_records_required")
    for model_id in REMAINING_IMAGE_IDS:
        item = records[model_id]
        _validate_encoder_probe(model_id, item, catalog)
        _require_smoke_receipt(item, model_id, evidence.get("development_package_sha256"))
    return records


def _validate_encoder_probe(model_id, item, catalog):
    spec = catalog["models"][model_id]
    require(isinstance(item, dict) and item.get("model_id") == model_id
            and item.get("image_eligibility_verified") is True
            and item.get("synthetic_gpu_forward_verified") is True
            and item.get("precision_fp16_verified") is True
            and item.get("synthetic_gpu_output_shape") == [2, spec["feature_dimension"]]
            and item.get("synthetic_gpu_logits_shape") == [2]
            and item.get("synthetic_gpu_fp16_output_shape") == [2, spec["feature_dimension"]]
            and item.get("synthetic_gpu_fp16_logits_shape") == [2]
            and valid_sha(item.get("authenticated_colab33_record_sha256"))
            and item.get("verified_checkpoint_files_sha256") == {part["name"]: part["sha256"] for part in spec["files"]}
            and item.get("license_review") == spec["license_review"]
            and item.get("overlap_review") == spec["overlap_review"],
            "model_specific_verified_remaining_encoder_evidence_required:" + model_id)


def _require_smoke_receipt(item, model_id, package_sha256):
    require(isinstance(item, dict) and item.get("model_id") == model_id
        and item.get("training_smoke_test_passed") is True
        and item.get("gpu_resource_profile_recorded") is True
        and item.get("optimizer_updates") == 2 and item.get("validation_fold") == 0
        and item.get("seed") == SEED and item.get("stage") == "stage_1"
        and item.get("development_package_sha256") == package_sha256
        and item.get("training_plan_sha256") == release.PINS[release.PLAN_PATH]
        and item.get("frozen_encoder_state_unchanged") is True
        and item.get("task_head_update_verified") is True
        and item.get("eligible_for_model_selection") is False
        and item.get("reserved_test_accessed") is False
        and all(valid_sha(item.get(key)) for key in ("smoke_request_sha256", "smoke_job_record_sha256",
            "smoke_training_record_sha256")), "own_two_update_smoke_and_resource_evidence_required:" + model_id)


def _validate_smoke_records(job, engine, request, registry):
    from knee.aligned_training import engine as training_engine, runner
    from knee.training_smoke import _validate_amp_audit, GPU_PACKAGES
    require(job.get("version") == runner.VERSION and job.get("scope") == "development_only"
        and job.get("request") == request and job.get("request_sha256") == digest(request)
        and job.get("record_sha256") == digest({k:v for k,v in job.items() if k != "record_sha256"})
        and job.get("plan_sha256") == release.PINS[release.PLAN_PATH]
        and job.get("eligible_for_model_selection") is False and job.get("reserved_test_accessed") is False
        and job.get("training_executed") is True and job.get("engine_status") == "bounded_training_smoke_complete"
        and job.get("eligibility_gates") == {key: index < 4 for index, key in enumerate(registry["eligibility_gates"])},
        "authenticated_current_encoder_smoke_job_required")
    require(engine.get("version") == training_engine.VERSION and engine.get("model_id") == request["model_id"]
        and engine.get("stage") == "stage_1" and engine.get("seed") == SEED and engine.get("smoke") is True
        and engine.get("scope") == "development_only" and engine.get("status") == "bounded_training_smoke_complete"
        and engine.get("eligible_for_model_selection") is False and engine.get("reserved_test_accessed") is False
        and engine.get("optimizer_created") is True and engine.get("training_executed") is True
        and engine.get("optimizer_updates") == 2 and engine.get("completed_epochs") == 1
        and engine.get("micro_batch_size") == request["image_micro_batch_size"] and engine.get("effective_batch_size") == 16
        and engine.get("device") == "cuda:0" and engine.get("precision") == "float16_autocast_grad_scaler"
        and engine.get("gpu_resource_profile_recorded") is True and engine.get("restored_best_checkpoint") is True
        and engine.get("resume_supported") is False and engine.get("optimizer_state_saved") is False,
        "exact_current_two_update_gpu_smoke_required")
    provenance = engine.get("provenance", {})
    require(provenance.get("validation_fold") == 0 and provenance.get("request_sha256") == digest(request)
        and provenance.get("development_package_sha256") == request["development_package_sha256"],
        "smoke_engine_package_request_binding_required")
    history = engine.get("history")
    require(isinstance(history, list) and len(history) == 1 and history[0].get("epoch") == 1
        and history[0].get("phase") == "head_only" and history[0].get("optimizer_updates") == 2
        and history[0].get("training_knees_seen") == min(32, provenance.get("training_knees", 0))
        and 16 < history[0]["training_knees_seen"] <= 32 and _finite(history[0].get("weighted_loss_per_knee"))
        and job.get("metrics") == history[0].get("validation")
        and engine.get("checkpoint_sha256") == {name: job["files_sha256"][name]
            for name in ("best_model.safetensors", "last_model.safetensors")},
        "bounded_smoke_history_and_checkpoints_required")
    _validate_amp_audit(engine)
    resource = engine.get("resource_profile", {})
    require(resource.get("device_type") == "cuda" and resource.get("cuda_peak_measured") is True
        and resource.get("torch_version") == GPU_PACKAGES["torch"] and resource.get("cuda_version") == "12.4"
        and isinstance(resource.get("device_name"), str) and bool(resource["device_name"])
        and all(type(resource.get(k)) is int and resource[k] > 0 for k in
            ("total_memory_bytes", "peak_memory_allocated_bytes", "peak_memory_reserved_bytes"))
        and resource["peak_memory_allocated_bytes"] <= resource["peak_memory_reserved_bytes"] <= resource["total_memory_bytes"]
        and _finite(resource.get("elapsed_seconds"), minimum=1e-30), "measured_current_encoder_gpu_resources_required")
    return resource


def validate_remaining_smoke_record(model_id, smoke_record, training_record, job_record, registry):
    """Reauthenticate a persisted two-update receipt, never selection metrics."""
    _require_smoke_receipt(smoke_record, model_id, smoke_record.get("development_package_sha256"))
    request = job_record["request"]
    require(model_id in REMAINING_IMAGE_IDS and request.get("model_id") == model_id
        and request.get("operation") == "single_fold_smoke" and request.get("validation_fold") == 0
        and request.get("seed") == SEED and request.get("stage") == "stage_1"
        and request.get("development_package_sha256") == smoke_record["development_package_sha256"]
        and smoke_record["smoke_request_sha256"] == digest(request)
        and smoke_record["smoke_job_record_sha256"] == job_record["record_sha256"]
        and smoke_record["smoke_training_record_sha256"] == job_record["files_sha256"]["training_record.json"],
        "persisted_smoke_receipt_job_binding_required")
    resource = _validate_smoke_records(job_record, training_record, request, registry)
    require(smoke_record.get("resource_profile") == resource, "persisted_smoke_resource_profile_changed")
    return smoke_record


def run_remaining_encoder_smoke(repository_root, preparation_dir, package_dir, reference_path,
        weights_root, output_dir, *, model_id, approval_statement, gpu_probe):
    """Only the approved Colab campaign can execute this real two-update gate."""
    require(model_id in REMAINING_IMAGE_IDS and approval_statement == APPROVAL_STATEMENTS["remaining_encoders_frozen"],
        "exact_remaining_encoder_smoke_approval_required")
    from knee.training_smoke import verify_gpu_environment, _verify_frozen_encoder
    from knee.aligned_training import runner
    verify_gpu_environment("google_colab_gpu")
    repository, proof, package, reference, weights, output = map(checked_path,
        (repository_root, preparation_dir, package_dir, reference_path, weights_root, output_dir))
    require(checked_path(__file__) == repository / "src/knee/methodology_campaign.py"
        and weights.is_dir() and not output.exists() and output.parent.is_dir()
        and all(not output.is_relative_to(p) and not p.is_relative_to(output)
            for p in (repository, proof, package, reference.parent, weights)), "new_separate_current_encoder_smoke_output_required")
    _, core38, sidecar = validate_preparation(proof, repository)
    _, catalog, plan, registry = release.load_release(repository)
    _validate_encoder_probe(model_id, gpu_probe, catalog)
    require(sha_file(reference) == core38["sources_sha256"]["notebook32_reference"], "smoke_same_development_reference_required")
    request = next(r for r in requests("remaining_encoders_frozen", core38["development_package_sha256"],
        sidecar["record_sha256"], plan, registry) if r["model_id"] == model_id and r["validation_fold"] == 0)
    request["operation"] = "single_fold_smoke"
    validate_request(request, plan, registry)
    authority_record = {"version": AUTHORITY_VERSION, "training_authorized": True,
        "reserved_test_access_authorized": False, "approval_source": "direct_investigator_authorization",
        "approval_statement": approval_statement, "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_sha256": digest(request), "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "eligibility_gates": {key: i < 4 for i,key in enumerate(registry["eligibility_gates"])},
        "full_encoder_training_authorized": False}
    validate_authority(authority_record, request, plan, registry)
    initial = _initial_image_state(model_id, plan, catalog, weights)
    sealed = {p: sha_file(p) for p in (reference, package / "development_package.zip", package / "handoff.json",
        proof / inputs.CORE_FILENAME, proof / inputs.SUMMARY_FILENAME, proof / "baseline_sidecar.json")}
    output.mkdir()
    job_root = output / "job"
    job = runner.execute(request, authority_record, repository_root=repository, package_dir=package,
        reference_path=reference, weights_root=weights, output_dir=job_root, device="cuda:0", allow_download=False)
    require(read_json(job_root / runner.JOB_FILENAME) == job
        and set(p.name for p in job_root.iterdir()) == set(job["files_sha256"]) | {runner.JOB_FILENAME}
        and all(sha_file(job_root / name) == checksum for name,checksum in job["files_sha256"].items()),
        "exact_intact_current_encoder_smoke_artifacts_required")
    engine = read_json(job_root / "training_record.json")
    resource = _validate_smoke_records(job, engine, request, registry)
    frozen = _verify_frozen_encoder(job_root, initial)
    del initial
    require(all(sha_file(path) == checksum for path, checksum in sealed.items()), "current_encoder_smoke_sources_changed")
    validate_preparation(proof, repository)
    release.load_release(repository)
    result = {"model_id": model_id, "training_smoke_test_passed": True, "gpu_resource_profile_recorded": True,
        "optimizer_updates": 2, "validation_fold": 0, "seed": SEED, "stage": "stage_1",
        "development_package_sha256": core38["development_package_sha256"],
        "training_plan_sha256": release.PINS[release.PLAN_PATH], "smoke_request_sha256": digest(request),
        "smoke_job_record_sha256": job["record_sha256"], "smoke_training_record_sha256": sha_file(job_root / "training_record.json"),
        **frozen, "resource_profile": resource, "eligible_for_model_selection": False, "reserved_test_accessed": False}
    validate_remaining_smoke_record(model_id, result, engine, job, registry)
    runner.save_json(output / "encoder_smoke.json", result)
    return result


def _initial_image_state(model_id, plan, catalog, weights):
    from knee.final_models_preflight import checkpoint_files
    from knee.aligned_training.models import build_model
    paths = checkpoint_files(model_id, catalog, weights, True)
    model, _ = build_model(model_id, plan, catalog, paths, seed=SEED)
    try:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    finally:
        del model


def run(repository_root, preparation_dir, package_dir, reference_path, output_dir, *, campaign,
        approval_statement, completed_jobs=None, job_completed=None, weights_root=None, gpu_evidence=None):
    """Execute only after exact confirmation; completed jobs are reauthenticated."""
    require(campaign in CAMPAIGNS and approval_statement == APPROVAL_STATEMENTS[campaign],
        "explicit_current_methodology_campaign_approval_required")
    repository, proof, package, reference, output = map(checked_path,
        (repository_root, preparation_dir, package_dir, reference_path, output_dir))
    require(checked_path(__file__).resolve() == repository.resolve() / "src/knee/methodology_campaign.py",
        "methodology_campaign_must_use_selected_checkout")
    require(not output.exists() and output.parent.is_dir()
        and all(not output.is_relative_to(path) and not path.is_relative_to(output)
            for path in (repository, proof, package, reference.parent)), "new_separate_methodology_output_required")
    if campaign == "clinical":
        from knee.clinical_fivefold import verify_cpu_environment
        environment = verify_cpu_environment("google_colab_cpu")
        require(weights_root is None and gpu_evidence is None, "clinical_campaign_cannot_open_pretrained_weights")
    else:
        from knee.training_smoke import verify_gpu_environment
        environment = verify_gpu_environment("google_colab_gpu")
        require(isinstance(gpu_evidence, dict) and gpu_evidence.get("image_eligibility_verified") is True
            and weights_root is not None, "verified_image_runtime_evidence_required")
    summary38, core38, sidecar = validate_preparation(proof, repository)
    _, catalog, plan, registry = release.load_release(repository)
    package_sha = core38["development_package_sha256"]
    require(sha_file(reference) == core38["sources_sha256"]["notebook32_reference"],
        "same_authenticated_development_input_reference_required")
    from knee.aligned_training.data import load_sealed_inputs
    loaded = load_sealed_inputs(package, reference, repository, package_sha)
    rows = loaded["manifest"]["rows"]
    require(len(rows) == core38["development_knees"]
        and len({r["src_subject_id"] for r in rows}) == core38["development_participants"],
        "exact_colab38_development_membership_required")
    del loaded
    all_requests = requests(campaign, package_sha, sidecar["record_sha256"], plan, registry)
    expected_keys = {job_key(request) for request in all_requests}
    completed_jobs = {} if completed_jobs is None else completed_jobs
    require(isinstance(completed_jobs, dict) and set(completed_jobs) <= expected_keys,
        "only_current_methodology_campaign_jobs_may_be_recovered")
    initial_state = None
    current_image_model = None
    if campaign in IMAGE_CAMPAIGNS:
        weights = checked_path(weights_root)
        require(weights.is_dir() and all(not weights.is_relative_to(path) and not path.is_relative_to(weights)
            for path in (repository, proof, package, reference.parent, output)),
            "separate_existing_external_weights_cache_required")
        require(gpu_evidence.get("development_package_sha256") == package_sha
            and gpu_evidence.get("model_input_reference_sha256") == sha_file(reference),
            "same_package_and_reference_gpu_evidence_required")
        if campaign == "remaining_encoders_frozen":
            _validate_remaining_eligibility(gpu_evidence, catalog)
    sealed = {path: sha_file(path) for path in (package / "development_package.zip", package / "handoff.json",
        reference, proof / inputs.CORE_FILENAME, proof / inputs.SUMMARY_FILENAME, proof / "baseline_sidecar.json")}
    from knee.aligned_training import runner
    output.mkdir()
    groups, reports = {}, []
    for index, request in enumerate(all_requests, 1):
        if campaign in IMAGE_CAMPAIGNS and current_image_model != request["model_id"]:
            # Models are contiguous in request order. Release the previous
            # encoder's full CPU reference before loading the next encoder.
            initial_state = None
            current_image_model = None
            initial_state = _initial_image_state(request["model_id"], plan, catalog, weights)
            current_image_model = request["model_id"]
        key = job_key(request)
        require(all(sha_file(path) == value for path, value in sealed.items()), "sealed_methodology_inputs_changed")
        destination = output / key
        reused = key in completed_jobs
        print(f'Campaña {campaign}: trabajo {index}/{len(all_requests)}, {key}; '
            + ("verificando trabajo completo guardado." if reused else "entrenando desarrollo."), file=sys.stderr, flush=True)
        if reused:
            previous = checked_path(completed_jobs[key])
            validate_completed_job(previous, request, plan, registry, rows, sidecar, initial_state=initial_state)
            shutil.copytree(previous, destination)
        else:
            permit = authority(request, campaign, approval_statement, plan, registry,
                image_eligibility_verified=campaign in IMAGE_CAMPAIGNS,
                encoder_smoke_evidence=gpu_evidence["model_eligibility"][request["model_id"]]
                    if campaign == "remaining_encoders_frozen" else None)
            runner.execute(request, permit, repository_root=repository, package_dir=package,
                reference_path=reference, weights_root=weights_root, output_dir=destination,
                device="cuda:0" if campaign in IMAGE_CAMPAIGNS else "cpu",
                allow_download=campaign in IMAGE_CAMPAIGNS,
                baseline_sidecar_path=proof / "baseline_sidecar.json" if request["baseline_sidecar_sha256"] else None)
        record, engine = validate_completed_job(destination, request, plan, registry, rows, sidecar, initial_state=initial_state)
        group = (request["clinical_scenario"], request["model_id"])
        groups.setdefault(group, {})[request["validation_fold"]] = (destination, record)
        reports.append({"key": key, "request_sha256": record["request_sha256"],
            "job_record_sha256": record["record_sha256"], "metrics": record["metrics"],
            "reused_authenticated_complete_job": reused})
        if job_completed:
            job_completed(key, destination, record)
    initial_state = None
    model_results = {}
    for (scenario, model_id), jobs in groups.items():
        selection, descriptive, oof = aggregate(jobs, rows)
        key = scenario + "__" + model_id
        runner.save_json(output / (key + "__oof.json"), {"scope": "development_only", "scenario": scenario,
            "model_id": model_id, "rows": oof, "pooled_metric_role": "descriptive_only_not_selection_metric"})
        model_results[key] = {"fivefold_selection_metrics": selection,
            "pooled_oof_metrics_descriptive_only": descriptive, "out_of_fold_predictions": len(oof)}
    require(all(sha_file(path) == value for path, value in sealed.items()), "sealed_methodology_sources_changed_at_close")
    validate_preparation(proof, repository)
    release.load_release(repository)
    result = {"version": VERSION, "status": STATUS, "campaign": campaign, "scope": "development_only",
        "execution_environment": "google_colab_cpu" if campaign == "clinical" else "google_colab_gpu",
        "colab_runtime_executed": True, "local_rehearsal": False, "seed": SEED, "stage": "stage_1",
        "job_count": len(all_requests), "folds_completed": list(FOLDS), "job_results": reports,
        "model_results": model_results, "clinical_scenarios": list(clinical.SCENARIOS) if campaign == "clinical" else ["clinical_main"],
        "development_participants": core38["development_participants"], "development_knees": len(rows),
        "development_package_sha256": package_sha, "model_input_reference_sha256": sha_file(reference),
        "baseline_sidecar_sha256": sidecar["record_sha256"], "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "training_plan_sha256": release.PINS[release.PLAN_PATH], "previous_colab38_summary_sha256": sha_file(proof / inputs.SUMMARY_FILENAME),
        "training_executed": True, "eligible_for_model_selection": True,
        "clinical_preprocessing_fitted_inside_training_folds_only": True,
        "participant_disjoint_folds_verified": True, "every_development_knee_predicted_once_per_model": True,
        "image_augmentation_policy": plan["neural_training"]["augmentation"] if campaign in IMAGE_CAMPAIGNS else None,
        "historical_v1_1_jobs_reused": False, "within_job_optimizer_resume_supported": False,
        "full_encoder_training_authorized": False, "encoder_unfreezing_executed": False,
        "calibration_performed": False, "threshold_selected": False, "final_refit_executed": False,
        "reserved_test_accessed": False, "reserved_test_predictive_evaluation_opened": False,
        "originals_modified": False, "partitions_modified": False, "predictive_performance_claimed": False,
        "unbiased_final_performance_claimed": False, "environment": environment,
        "artifact_files_sha256": {p.relative_to(output).as_posix(): sha_file(p) for p in output.rglob("*") if p.is_file()},
        "next_step": "review_v1_2_development_results_before_separate_next_model_authority",
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    runner.save_json(output / FILENAME, result)
    return result
