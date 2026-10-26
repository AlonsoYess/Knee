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
CAMPAIGNS = ("clinical", "densenet_frozen")
CLINICAL_IDS = preparation.CLINICAL_IDS
FOLDS = tuple(range(5))
SEED = 2026
CONFIRMATIONS = {"clinical": "AUTORIZO CLINICOS V1.2 5 PLIEGUES",
    "densenet_frozen": "AUTORIZO DENSENET V1.2 5 PLIEGUES"}
APPROVAL_STATEMENTS = {
    "clinical": "Autorizo solo los tres algoritmos clínicos y sus tres escenarios basales en los cinco pliegues de desarrollo, receta v1.2 y semilla 2026; sin abrir la prueba reservada ni realizar calibración, umbral o ajuste final.",
    "densenet_frozen": "Autorizo solo DenseNet121 con codificador congelado y aumento de intensidad v1.2 en los cinco pliegues de desarrollo, semilla 2026; sin abrir la prueba reservada, descongelar el codificador ni realizar calibración, umbral o ajuste final."}


def requests(campaign, package_sha256, sidecar_sha256, plan, registry):
    require(campaign in CAMPAIGNS and valid_sha(package_sha256) and valid_sha(sidecar_sha256),
        "exact_registered_methodology_campaign_required")
    if campaign == "clinical":
        return [preparation.make_request(package_sha256, sidecar_sha256, scenario, model, fold, plan, registry)
            for scenario in clinical.SCENARIOS for model in CLINICAL_IDS for fold in FOLDS]
    from knee.training_augmentation import VERSION as AUGMENTATION_VERSION
    result = []
    for fold in FOLDS:
        request = preparation.make_request(package_sha256, sidecar_sha256, "clinical_main",
            "clinical_mlp", fold, plan, registry)
        request.update(model_id="image_densenet121", image_encoder_id="image_densenet121",
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


def authority(request, campaign, statement, plan, registry, *, image_eligibility_verified=False):
    """Called only after the campaign's exact future investigator confirmation."""
    require(campaign in CAMPAIGNS and statement == APPROVAL_STATEMENTS[campaign],
        "exact_methodology_campaign_training_confirmation_required")
    expected = requests(campaign, request["development_package_sha256"],
        request.get("baseline_sidecar_sha256") or "0" * 64, plan, registry)
    require(request in expected and (campaign != "densenet_frozen" or image_eligibility_verified is True),
        "exact_campaign_request_and_verified_image_eligibility_required")
    record = {"version": AUTHORITY_VERSION, "training_authorized": True,
        "reserved_test_access_authorized": False, "approval_source": "direct_investigator_authorization",
        "approval_statement": statement, "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_sha256": digest(request), "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "eligibility_gates": {key: campaign == "densenet_frozen" for key in registry["eligibility_gates"]},
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
        require(isinstance(initial_state, dict) and bool(initial_state), "initial_verified_densenet_state_required")
        for filename in ("best_model.safetensors", "last_model.safetensors"):
            state = load_file(str(root / filename), device="cpu")
            require(set(state) == set(initial_state) and all(state[key].shape == original.shape
                and state[key].dtype == original.dtype and bool(torch.isfinite(state[key]).all())
                for key, original in initial_state.items()), "exact_finite_densenet_checkpoint_required")
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
            and weights_root is not None, "verified_densenet_runtime_evidence_required")
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
    if campaign == "densenet_frozen":
        weights = checked_path(weights_root)
        require(weights.is_dir() and all(not weights.is_relative_to(path) and not path.is_relative_to(weights)
            for path in (repository, proof, package, reference.parent, output)),
            "separate_existing_external_weights_cache_required")
        require(gpu_evidence.get("development_package_sha256") == package_sha
            and gpu_evidence.get("model_input_reference_sha256") == sha_file(reference),
            "same_package_and_reference_gpu_evidence_required")
        from knee.final_models_preflight import checkpoint_files
        from knee.aligned_training.models import build_model
        paths = checkpoint_files("image_densenet121", catalog, weights, True)
        model, _ = build_model("image_densenet121", plan, catalog, paths, seed=SEED)
        initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        del model
    sealed = {path: sha_file(path) for path in (package / "development_package.zip", package / "handoff.json",
        reference, proof / inputs.CORE_FILENAME, proof / inputs.SUMMARY_FILENAME, proof / "baseline_sidecar.json")}
    from knee.aligned_training import runner
    output.mkdir()
    groups, reports = {}, []
    for index, request in enumerate(all_requests, 1):
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
                image_eligibility_verified=campaign == "densenet_frozen")
            runner.execute(request, permit, repository_root=repository, package_dir=package,
                reference_path=reference, weights_root=weights_root, output_dir=destination,
                device="cuda:0" if campaign == "densenet_frozen" else "cpu",
                allow_download=campaign == "densenet_frozen",
                baseline_sidecar_path=proof / "baseline_sidecar.json" if request["baseline_sidecar_sha256"] else None)
        record, engine = validate_completed_job(destination, request, plan, registry, rows, sidecar, initial_state=initial_state)
        group = (request["clinical_scenario"], request["model_id"])
        groups.setdefault(group, {})[request["validation_fold"]] = (destination, record)
        reports.append({"key": key, "request_sha256": record["request_sha256"],
            "job_record_sha256": record["record_sha256"], "metrics": record["metrics"],
            "reused_authenticated_complete_job": reused})
        if job_completed:
            job_completed(key, destination, record)
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
        "image_augmentation_policy": plan["neural_training"]["augmentation"] if campaign == "densenet_frozen" else None,
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
