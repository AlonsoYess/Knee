"""One explicitly approved two-update DenseNet GPU smoke, never model selection.

The notebook is only prepared locally. Calling ``run`` really updates the task
head and is reserved for the investigator's separately confirmed Colab run.
No reserved-test path or complete-training operation exists in this API.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform

from knee.colab_data import checked_path, require, sha_file
from knee.final_model_release import load_release, PINS, RELEASE_PATH
from knee.training_authority import digest, REQUEST_VERSION, VERSION as AUTHORITY_VERSION
from knee import training_preparation as preparation
from knee import training_preparation_colab as previous

VERSION = "bounded_development_gpu_smoke_v1.0"
STATUS = "bounded_gpu_smoke_passed_pending_full_training_authority"
FILENAME = "training_smoke.json"
MODEL_ID = "image_densenet121"
PREPARATION_REVISION = "e6ac20c9dfe3044ca267fe6c6dcca2806c42a33d"
PREVIOUS_CHECKPOINT_SHA256 = "e6c78ad178518cad4836a18966dd9e84a3a2d30cc23771d20e58a69d3f81e771"
APPROVAL_STATEMENT = (
    "Autorizo solo la prueba breve GPU de DenseNet121 en desarrollo, pliegue 0, "
    "semilla 2026, dos actualizaciones de la cabeza; sin abrir prueba reservada "
    "ni entrenamiento completo."
)
SOURCE_34 = {
    "src/knee/training_authority.py": "55ea0d669f2662460394d3c897eadd3244b449b19298e5d2d54983844dfd47d3",
    "src/knee/training_data.py": "ee8a1196cae2630e0954a3c9b0e824956c958147df56b321118161547f5776f7",
    "src/knee/training_models.py": "89f8eacfe55ebedf8dde643273e2a3b2f739566dab3b9b423257f3af346b3696",
    "src/knee/training_engine.py": "2728f10a4e2bffb37558b9b65a0dbc2be0ac0edbe5ed5ece90472155b7f2ad93",
    "src/knee/training_runner.py": "aa118cdb3b94222e2aa805c88c14734a81c7802962e3a8beeebbbcd1ce74cdaf",
    "src/knee/training_preparation.py": "20889f94b6c24ed8b18a3001b0fd1cacd37760d0917961376a3118accec75ef3",
    "scripts/run_development_training.py": "ba7ae10b8492592bec2bdfae811cc0ce979de407fd6107769468efadcca3d9e5",
    "src/knee/training_preparation_colab.py": "9b62a6a53d0d06fa66d3146d26d24532a62f519a39883057782b1f1929adb852",
    "requirements-training-runner.txt": "03a794914de5aa5f08d0b48875fa3e21e2fdb8da5f6db808f2063fbb69a86100",
}
# Historical notebook34 proof remains bound to SOURCE_34 above. Only this
# explicitly reviewed numerical repair may differ in the current executor;
# neither the proof nor the immutable modeling release is reclassified.
AMP_REPAIR_VERSION = "same_effective_group_amp_replay_v1.0"
RUNTIME_SOURCE_PATCHES = {
    "src/knee/training_engine.py": "e732048d416011f70ed1322e03ddbf3a874af7d1c6b184fe19c5831f650d1c06",
}
GPU_PACKAGES = {**preparation.EXTRA_PACKAGES,
    "torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124", "numpy": "1.26.4"}
JOB_FILES = {"best_model.safetensors", "last_model.safetensors", "training_record.json",
    "best_validation_predictions.json", "clinical_state.json"}


def _utc(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(stamp.utcoffset() == timezone.utc.utcoffset(stamp), "utc_smoke_evidence_timestamp_required")
    except (AttributeError, TypeError, ValueError):
        raise ValueError("utc_smoke_evidence_timestamp_required") from None
    return stamp


def validate_preparation(directory):
    """Authenticate the two completed34 records, their release and sealed33 link."""
    root = checked_path(directory)
    core_path = root / previous.CORE_FILENAME
    core = preparation.read_json(core_path)
    summary = preparation.read_json(root / previous.SUMMARY_FILENAME)
    previous.validate_result(core, "google_colab_cpu")
    previous.validate_result(summary, "google_colab_cpu")
    require(summary.get("code_revision") == PREPARATION_REVISION
        and summary.get("local_rehearsal") is False
        and summary.get("cpu_only_guard_passed") is True
        and summary.get("result_files_saved_and_readback_verified") is True
        and summary.get("preparation_record_sha256") == sha_file(core_path),
        "completed_actual_colab34_with_exact_preparation_record_required")
    require(all(summary.get(key) == value for key, value in core.items() if key != "executed_at_utc"),
        "colab34_summary_and_preparation_content_mismatch")
    tests = summary.get("cpu_tests", {})
    require(tests.get("passed") is True and type(tests.get("tests_run")) is int
        and tests["tests_run"] == 313 and type(tests.get("skipped")) is int
        and tests["skipped"] == 0 and tests.get("scope") == list(previous.TEST_MODULES),
        "exact_completed_colab34_test_scope_required")
    require(core.get("source_files_sha256") == SOURCE_34
        and core.get("release_component_canonical_sha256") == PINS
        and core.get("previous_colab33_checkpoint_record_sha256") == PREVIOUS_CHECKPOINT_SHA256
        and summary.get("runtime_requirements_sha256") == SOURCE_34["requirements-training-runner.txt"],
        "colab34_code_release_and_verified_colab33_chain_required")
    require(core.get("clinical_models_implemented") == ["clinical_logistic", "clinical_xgboost", "clinical_mlp"]
        and core.get("accessible_image_models_implemented") == ["image_densenet121", "image_vit_b16",
            "image_convnextv2_tiny", "image_skelex", "image_biomedclip", "image_mambaout_tiny"]
        and core.get("conditional_fusion_models_implemented") == ["fusion_intermediate_concat", "fusion_intermediate_gated_film"],
        "same_closed_implemented_model_roster_required")
    require(_utc(summary.get("executed_at_utc")) >= _utc(core.get("executed_at_utc")),
        "colab34_publication_must_follow_preparation")
    return summary


def make_request(development_sha256):
    """The only request this notebook may authorize: no configurable model/fold."""
    require(isinstance(development_sha256, str) and len(development_sha256) == 64
        and all(value in "0123456789abcdef" for value in development_sha256),
        "sealed_development_package_hash_required")
    return {"version": REQUEST_VERSION, "operation": "single_fold_smoke", "model_id": MODEL_ID,
        "image_encoder_id": MODEL_ID, "stage": "stage_1", "validation_fold": 0, "seed": 2026,
        "development_package_sha256": development_sha256, "image_micro_batch_size": 2,
        "selection_evidence_sha256": None, "fold_image_job_sha256": None}


def verify_gpu_environment(execution_environment):
    """Fail before packages/data if this is not the pinned actual GPU Colab."""
    require(execution_environment == "google_colab_gpu" and platform.system() == "Linux"
        and bool(os.environ.get("COLAB_RELEASE_TAG")) and platform.python_version() == "3.12.14",
        "actual_isolated_colab31214_gpu_runtime_required")
    import torch
    # Set before the first operation which can initialize a CUDA context.
    from knee.training_engine import _prepare_cuda_environment
    _prepare_cuda_environment()
    from knee.training_gpu_runtime import verify_current_runtime
    full_runtime = verify_current_runtime(execution_environment)
    versions = {name: importlib.metadata.version(name) for name in GPU_PACKAGES}
    require(versions == GPU_PACKAGES and str(torch.__version__) == GPU_PACKAGES["torch"]
        and torch.version.cuda == "12.4" and torch.cuda.is_available()
        and torch.cuda.device_count() >= 1, "pinned_cu124_gpu_required_no_cpu_fallback")
    return {"runtime_verification": full_runtime, "python": platform.python_version(), **versions, "cuda": torch.version.cuda,
        "colab_release_tag": os.environ["COLAB_RELEASE_TAG"], "device": "cuda:0"}


def _sources(repository):
    expected = {**preparation.SOURCE_33, **SOURCE_34}
    expected.update(RUNTIME_SOURCE_PATCHES)
    require(all(sha_file(repository / name) == checksum for name, checksum in expected.items()),
        "verified_training_sources_changed_since_colab34")
    for name in ("src/knee/training_smoke.py", "src/knee/training_gpu_runtime.py", "requirements-training-gpu.txt"):
        expected[name] = sha_file(repository / name)
    return expected


def _authority(request, registry, catalog, statement):
    require(statement == APPROVAL_STATEMENT, "exact_separate_bounded_gpu_smoke_approval_required")
    spec = catalog["models"][MODEL_ID]
    require(spec["verification_enabled"] is True
        and spec["checkpoint_id"] == "DenseNet121_Weights.IMAGENET1K_V1"
        and spec["license_review"]["use_scope"] == "private_noncommercial_academic_research"
        and spec["license_review"]["unrestricted_redistribution_claimed"] is False
        and "ImageNet-1K" in spec["overlap_review"]
        and catalog["shared_limits"]["pretraining_subject_disjointness_proven"] is False
        and catalog["shared_limits"]["preprocessing_localizer_oai_overlap_limitation_remains"] is True,
        "scoped_official_densenet_eligibility_evidence_required")
    gates = {key: index < 4 for index, key in enumerate(registry["eligibility_gates"])}
    require(len(gates) == 6, "unchanged_six_eligibility_gates_required")
    return {"version": AUTHORITY_VERSION, "training_authorized": True,
        "reserved_test_access_authorized": False, "approval_source": "direct_investigator_authorization",
        "approval_statement": statement, "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_sha256": digest(request), "modeling_release_sha256": PINS[RELEASE_PATH],
        "eligibility_gates": gates, "full_encoder_training_authorized": False}


def _validate_job(job_root, job, request):
    """Recheck the authenticated runner record and its exact two-update engine."""
    from knee.training_runner import JOB_FILENAME, VERSION as RUNNER_VERSION
    require(preparation.read_json(job_root / JOB_FILENAME) == job
        and job.get("version") == RUNNER_VERSION and job.get("scope") == "development_only"
        and job.get("request") == request and job.get("request_sha256") == digest(request)
        and job.get("eligible_for_model_selection") is False and job.get("reserved_test_accessed") is False
        and job.get("training_executed") is True and job.get("engine_status") == "bounded_training_smoke_complete"
        and job.get("record_sha256") == digest({key: value for key, value in job.items() if key != "record_sha256"}),
        "authenticated_development_smoke_job_required")
    files = job.get("files_sha256")
    require(isinstance(files, dict) and set(files) == JOB_FILES
        and set(path.name for path in job_root.iterdir()) == JOB_FILES | {JOB_FILENAME}
        and all(sha_file(job_root / name) == checksum for name, checksum in files.items()),
        "exact_intact_private_smoke_artifacts_required")
    engine = preparation.read_json(job_root / "training_record.json")
    require(engine.get("model_id") == MODEL_ID and engine.get("stage") == "stage_1"
        and engine.get("seed") == 2026 and engine.get("smoke") is True
        and engine.get("scope") == "development_only"
        and engine.get("status") == "bounded_training_smoke_complete"
        and engine.get("eligible_for_model_selection") is False
        and engine.get("reserved_test_accessed") is False
        and engine.get("optimizer_created") is True and engine.get("training_executed") is True
        and type(engine.get("optimizer_updates")) is int and engine["optimizer_updates"] == 2
        and engine.get("completed_epochs") == 1 and engine.get("micro_batch_size") == 2
        and engine.get("effective_batch_size") == 16 and engine.get("device") == "cuda:0"
        and engine.get("precision") == "float16_autocast_grad_scaler"
        and engine.get("gpu_resource_profile_recorded") is True
        and engine.get("restored_best_checkpoint") is True and engine.get("resume_supported") is False
        and engine.get("optimizer_state_saved") is False,
        "exact_two_update_head_only_gpu_smoke_required")
    history = engine.get("history")
    require(isinstance(history, list) and len(history) == 1 and history[0].get("epoch") == 1
        and history[0].get("phase") == "head_only" and history[0].get("optimizer_updates") == 2
        and history[0].get("training_knees_seen") == 32,
        "two_updates_of_sixteen_knees_without_unfreezing_required")
    _validate_amp_audit(engine)
    resource = engine.get("resource_profile", {})
    require(resource.get("device_type") == "cuda" and resource.get("cuda_peak_measured") is True
        and resource.get("torch_version") == GPU_PACKAGES["torch"] and resource.get("cuda_version") == "12.4"
        and isinstance(resource.get("device_name"), str) and bool(resource["device_name"])
        and all(type(resource.get(key)) is int and resource[key] > 0 for key in
            ("total_memory_bytes", "peak_memory_allocated_bytes", "peak_memory_reserved_bytes"))
        and resource["peak_memory_allocated_bytes"] <= resource["peak_memory_reserved_bytes"] <= resource["total_memory_bytes"]
        and type(resource.get("elapsed_seconds")) in {int, float}
        and math.isfinite(resource["elapsed_seconds"]) and resource["elapsed_seconds"] > 0,
        "measured_finite_gpu_resource_profile_required")
    return engine


def _validate_amp_audit(engine):
    """Only finite successful updates count; no skipped or reassigned knees."""
    require(engine.get("numerical_repair") == {
        "version": AMP_REPAIR_VERSION, "maximum_backoffs_per_effective_group": 16,
        "same_effective_group_replayed": True, "nonfinite_gradients_never_clipped": True,
    }, "exact_same_effective_group_amp_repair_required")
    audit = engine.get("amp_scale_audit")
    require(isinstance(audit, list) and len(audit) == 2,
        "two_successful_update_scale_audits_required")
    fields = {"epoch", "update_index_within_epoch", "start_scale", "successful_scale",
        "scale_after_update", "backoffs", "attempts", "gradient_norm_before_clip"}
    replays = 0
    previous_scale = None
    for index, row in enumerate(audit, start=1):
        require(isinstance(row, dict) and set(row) == fields
            and type(row["epoch"]) is int and row["epoch"] == 1
            and type(row["update_index_within_epoch"]) is int
            and row["update_index_within_epoch"] == index
            and type(row["backoffs"]) is int and 0 <= row["backoffs"] <= 16
            and type(row["attempts"]) is int and row["attempts"] == row["backoffs"] + 1,
            "bounded_amp_replay_and_successful_update_accounting_required")
        for name in ("start_scale", "successful_scale", "scale_after_update", "gradient_norm_before_clip"):
            value = row[name]
            require(type(value) in {int, float} and math.isfinite(value)
                and value >= (0 if name == "gradient_norm_before_clip" else 1),
                "finite_successful_amp_scale_and_gradient_norm_required")
        require(row["successful_scale"] == row["start_scale"] * 0.5 ** row["backoffs"]
            and row["scale_after_update"] == row["successful_scale"]
            and (previous_scale is None or row["start_scale"] == previous_scale),
            "same_group_half_scale_backoff_without_omitted_update_required")
        previous_scale = row["scale_after_update"]
        replays += row["backoffs"]
    history = engine["history"][0]
    require(type(engine.get("amp_overflow_replays")) is int
        and engine["amp_overflow_replays"] == replays
        and history.get("amp_overflow_replays") == replays
        and [{"epoch": 1, **row} for row in history.get("amp_scale_audit", [])] == audit,
        "consistent_epoch_and_total_amp_replay_audit_required")


def _verify_frozen_encoder(job_root, initial):
    import torch
    from safetensors.torch import load_file
    encoder_keys = {name for name in initial if name.startswith("encoder.")}
    head_keys = {name for name in initial if name.startswith("task_head.")}
    require(encoder_keys and head_keys and encoder_keys | head_keys == set(initial),
        "exact_densenet_encoder_and_task_head_state_required")
    for name in ("best_model.safetensors", "last_model.safetensors"):
        state = load_file(str(job_root / name), device="cpu")
        require(set(state) == set(initial) and all(state[key].shape == initial[key].shape
            and state[key].dtype == initial[key].dtype and bool(torch.isfinite(state[key]).all())
            for key in initial), "finite_complete_smoke_state_required")
        require(all(torch.equal(state[key], initial[key]) for key in encoder_keys),
            "frozen_encoder_parameters_and_running_statistics_changed")
        require(any(not torch.equal(state[key], initial[key]) for key in head_keys),
            "smoke_task_head_must_actually_update")
    return {"frozen_encoder_state_unchanged": True, "task_head_update_verified": True,
        "encoder_tensors_compared": len(encoder_keys)}


def run(repository_root, preparation_dir, package_dir, reference_path, weights_root, output_dir,
        *, approval_statement, execution_environment="google_colab_gpu"):
    """Execute only the user-confirmed fixed GPU smoke; never a complete fit."""
    require(approval_statement == APPROVAL_STATEMENT, "exact_separate_bounded_gpu_smoke_approval_required")
    environment = verify_gpu_environment(execution_environment)
    repository, proof, package, reference, weights, output = map(checked_path,
        (repository_root, preparation_dir, package_dir, reference_path, weights_root, output_dir))
    require(checked_path(__file__) == repository / "src/knee/training_smoke.py",
        "smoke_module_must_match_published_checkout")
    require(repository.is_dir() and proof.is_dir() and package.is_dir() and reference.is_file()
        and weights.is_dir() and not output.exists() and output.parent.is_dir(),
        "existing_private_sources_cache_and_new_smoke_output_required")
    protected = (repository, proof, package, reference.parent, weights)
    require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in protected)
        and all(not weights.is_relative_to(path) and not path.is_relative_to(weights)
            for path in (repository, proof, package, reference.parent)),
        "smoke_output_and_public_cache_must_not_overlap_sources")
    prepared = validate_preparation(proof)
    sources = _sources(repository)
    _, catalog, plan, registry = load_release(repository)
    from knee.model_inputs_colab import _reference
    reference_record = preparation.read_json(reference)
    _reference(reference_record)
    package_sha = reference_record["sources_sha256"]["development_package"]
    request = make_request(package_sha)
    authority = _authority(request, registry, catalog, approval_statement)
    # Validate exact34 authority and sealed inputs inside the unchanged runner.
    # Download only this one public 32 MB checkpoint, never the seven-model cache.
    from knee.final_models_preflight import checkpoint_files
    paths = checkpoint_files(MODEL_ID, catalog, weights, allow_download=True)
    from knee.training_models import build_model
    initial_model, _ = build_model(MODEL_ID, plan, catalog, paths, seed=2026)
    initial = {name: tensor.detach().cpu().clone() for name, tensor in initial_model.state_dict().items()}
    del initial_model
    sealed = {path: sha_file(path) for path in (proof / previous.CORE_FILENAME,
        proof / previous.SUMMARY_FILENAME, package / "development_package.zip", package / "handoff.json",
        reference, *paths)}
    output.mkdir(exist_ok=False)
    from knee.training_runner import execute, save_json
    job_root = output / "job"
    job = execute(request, authority, repository_root=repository, package_dir=package,
        reference_path=reference, weights_root=weights, output_dir=job_root,
        device="cuda:0", allow_download=False)
    engine = _validate_job(job_root, job, request)
    frozen = _verify_frozen_encoder(job_root, initial)
    require(all(sha_file(path) == checksum for path, checksum in sealed.items())
        and _sources(repository) == sources, "sealed_smoke_sources_changed_during_execution")
    load_release(repository)
    files = {"job/" + path.name: sha_file(path) for path in sorted(job_root.iterdir())}
    result = {"version": VERSION, "status": STATUS, "execution_environment": execution_environment,
        "colab_runtime_executed": True, "local_rehearsal": False, "scope": "development_only",
        "request": request, "request_sha256": digest(request), "model_id": MODEL_ID,
        "validation_fold": 0, "seed": 2026, "stage": "stage_1", "smoke": True,
        "optimizer_updates": 2, "training_knees_seen": 32, "micro_batch_size": 2,
        "amp_numerical_repair_version": AMP_REPAIR_VERSION,
        "amp_overflow_replays": engine["amp_overflow_replays"],
        "amp_scale_audit": engine["amp_scale_audit"],
        "effective_batch_size": 16, "training_authorized": True, "training_executed": True,
        "optimizer_created": True, "backward_executed": True, "participant_data_opened": True,
        "training_smoke_test_passed": True, "gpu_resource_profile_recorded": True,
        "full_training_authorized": False, "full_training_executed": False,
        "full_encoder_training_authorized": False, "encoder_unfreezing_executed": False,
        "eligible_for_model_selection": False, "predictive_performance_claimed": False,
        "reserved_test_predictive_evaluation_opened": False, "reserved_test_accessed": False,
        "originals_modified": False, "partitions_modified": False,
        "development_package_sha256": package_sha,
        "clinical_preprocessing_fitted_inside_training_folds_only": True,
        "model_input_reference_sha256": sealed[reference],
        "previous_colab34_preparation_record_sha256": prepared["preparation_record_sha256"],
        "previous_colab33_checkpoint_record_sha256": PREVIOUS_CHECKPOINT_SHA256,
        "historical_training_source_files_sha256": SOURCE_34,
        "numerical_repair_source_lineage": {name: {
            "historical_sha256": SOURCE_34[name], "runtime_sha256": checksum,
            "repair_version": AMP_REPAIR_VERSION,
        } for name, checksum in RUNTIME_SOURCE_PATCHES.items()},
        "training_job_record_sha256": job["record_sha256"], "artifact_files_sha256": files,
        "release_component_canonical_sha256": PINS, "source_files_sha256": sources,
        "public_checkpoint_files_sha256": {path.name: sealed[path] for path in paths},
        "eligibility_gates_before_smoke": authority["eligibility_gates"],
        "eligibility_evidence": {"model_id": MODEL_ID, "scope": "private_noncommercial_academic_research",
            "catalog_sha256": PINS["configs/pretrained_weights_v1_1.json"],
            "license_review": catalog["models"][MODEL_ID]["license_review"],
            "overlap_review": catalog["models"][MODEL_ID]["overlap_review"],
            "pretraining_subject_disjointness_proven": False,
            "preprocessing_localizer_oai_overlap_limitation_remains": True,
            "normalization_and_loader_evidence": "authenticated_colab32_reference_and_colab34_frozen_sources"},
        "resource_profile": engine["resource_profile"], "environment": environment, **frozen,
        "next_step": "review_gpu_smoke_then_separately_authorize_development_training",
        "limitations": ["Two head updates are a runtime check, not predictive or clinical validation.",
            "Only DenseNet121 fold zero seed 2026 was tested; other encoders and unfreezing remain unverified on GPU.",
            "ImageNet subject-level disjointness is not independently proven; KNEEL OAI pretraining overlap remains unresolved.",
            "Checkpoint use is scoped to private noncommercial academic research; redistribution permission is not inferred.",
            "Smoke metrics are private diagnostics and cannot select architectures, hyperparameters or thresholds."],
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    save_json(output / FILENAME, result)
    require(preparation.read_json(output / FILENAME) == result, "gpu_smoke_result_readback_failed")
    return result
