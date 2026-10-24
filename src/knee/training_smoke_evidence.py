"""Read-only authentication of the completed, bounded Colab 35 GPU proof.

Only the two JSON receipts and the six sealed private job files are inspected.
Checkpoint bytes are hashed, never deserialized, and no model/data loader or
training entry point is invoked. A successful proof is not training authority.
"""
from __future__ import annotations

from pathlib import Path

from knee.colab_data import checked_path, require, sha_file, valid_sha
from knee.final_model_release import PINS, load_release
from knee.training_authority import digest
from knee.training_preparation import read_json, SOURCE_33
from knee import training_smoke as historical
from knee import training_smoke_colab as colab


VERSION = "completed_bounded_gpu_smoke_evidence_v1.0"
SMOKE_REVISION = "c157706ffa24ad924790a3e0b489013110b74169"
SMOKE_TEST_COUNT = 352
JOB_FILENAME = "training_job.json"
SOURCE_35 = {
    "src/knee/training_smoke.py": "00e4aa2cabddd0089663b71c0b8c8084535140fc4d549ad9b0a63ce4de61be56",
    "src/knee/training_gpu_runtime.py": "becc26df8f3d48b6e526cb05278d66a7b4eaa32226778ffd185b570d56afda2a",
    "requirements-training-gpu.txt": "33285509b3fa4609cfb2289899ea58f3ecdada8a411b4815e2dff44797acca43",
}
SOURCE_FILES = {**SOURCE_33, **historical.SOURCE_34,
    **historical.RUNTIME_SOURCE_PATCHES, **SOURCE_35}
ARTIFACT_NAMES = {"job/" + name for name in historical.JOB_FILES | {JOB_FILENAME}}


def _exact_inventory(root, core):
    """Bound the complete tree before hashing any sizeable checkpoint bytes."""
    require(root.is_dir() and {path.name for path in root.iterdir()} == {
        colab.CORE_FILENAME, colab.SUMMARY_FILENAME, "job"},
        "exact_completed_smoke_receipts_and_job_directory_required")
    job_root = checked_path(root / "job")
    require(job_root.is_dir() and {path.name for path in job_root.iterdir()}
        == historical.JOB_FILES | {JOB_FILENAME}, "exact_six_smoke_job_artifacts_required")
    expected = core.get("artifact_files_sha256")
    require(isinstance(expected, dict) and set(expected) == ARTIFACT_NAMES
        and all(valid_sha(value) for value in expected.values()),
        "exact_smoke_artifact_hash_inventory_required")
    total = 0
    for name, checksum in expected.items():
        path = checked_path(root / name)
        require(path.is_file() and 0 < path.stat().st_size <= colab.MAX_SINGLE_FILE_BYTES,
            "bounded_regular_smoke_artifact_required")
        total += path.stat().st_size
        require(total <= colab.MAX_OUTPUT_BYTES, "bounded_smoke_artifact_total_required")
        require(sha_file(path) == checksum, "sealed_smoke_artifact_bytes_changed")
    return job_root


def _validate_scope(core):
    colab.validate_result(core)
    for name in ("training_authorized", "training_executed", "optimizer_created",
        "backward_executed", "participant_data_opened", "training_smoke_test_passed",
        "gpu_resource_profile_recorded", "clinical_preprocessing_fitted_inside_training_folds_only"):
        require(core.get(name) is True, "actual_bounded_smoke_execution_evidence_required")
    require(core.get("reserved_test_accessed") is False and core.get("local_rehearsal") is False
        and type(core.get("encoder_tensors_compared")) is int
        and core["encoder_tensors_compared"] == 725,
        "verified_actual_colab_frozen_densenet_smoke_required")
    package_hash = core.get("development_package_sha256")
    request = core.get("request")
    require(valid_sha(package_hash) and isinstance(request, dict)
        and request == historical.make_request(package_hash)
        and all(type(request.get(name)) is int for name in
            ("validation_fold", "seed", "image_micro_batch_size"))
        and core.get("request_sha256") == digest(request),
        "exact_fixed_development_smoke_request_and_package_required")
    for name in ("model_input_reference_sha256", "previous_colab34_preparation_record_sha256",
        "training_job_record_sha256"):
        require(valid_sha(core.get(name)), "sealed_smoke_reference_and_job_chain_required")
    require(core.get("historical_training_source_files_sha256") == historical.SOURCE_34
        and core.get("previous_colab33_checkpoint_record_sha256") == historical.PREVIOUS_CHECKPOINT_SHA256
        and core.get("release_component_canonical_sha256") == PINS
        and core.get("source_files_sha256") == SOURCE_FILES,
        "historical_colab34_colab33_and_frozen_release_source_chain_required")
    expected_lineage = {name: {"historical_sha256": historical.SOURCE_34[name],
        "runtime_sha256": checksum, "repair_version": historical.AMP_REPAIR_VERSION}
        for name, checksum in historical.RUNTIME_SOURCE_PATCHES.items()}
    require(core.get("numerical_repair_source_lineage") == expected_lineage,
        "exact_approved_amp_repair_source_lineage_required")
    return request


def _validate_eligibility(core, repository):
    _, catalog, _, registry = load_release(repository)
    gates = {key: index < 4 for index, key in enumerate(registry["eligibility_gates"])}
    require(core.get("eligibility_gates_before_smoke") == gates
        and all(type(value) is bool for value in core["eligibility_gates_before_smoke"].values()),
        "historical_four_pre_smoke_gates_must_not_be_reclassified")
    spec = catalog["models"][historical.MODEL_ID]
    expected = {"model_id": historical.MODEL_ID, "scope": "private_noncommercial_academic_research",
        "catalog_sha256": PINS["configs/pretrained_weights_v1_1.json"],
        "license_review": spec["license_review"], "overlap_review": spec["overlap_review"],
        "pretraining_subject_disjointness_proven": False,
        "preprocessing_localizer_oai_overlap_limitation_remains": True,
        "normalization_and_loader_evidence": "authenticated_colab32_reference_and_colab34_frozen_sources"}
    require(core.get("eligibility_evidence") == expected,
        "exact_academic_license_and_unresolved_pretraining_overlap_evidence_required")
    require(core.get("public_checkpoint_files_sha256") == {
        item["name"]: item["sha256"] for item in spec["files"]},
        "same_official_densenet_initial_checkpoint_required")


def _validate_job_links(core, job_root, request):
    job = read_json(job_root / JOB_FILENAME)
    engine = historical._validate_job(job_root, job, request)
    require(job.get("record_sha256") == core["training_job_record_sha256"]
        and job.get("plan_sha256") == PINS["configs/training_plan_v1_1.json"]
        and job.get("selection_evidence") is None
        and job.get("eligibility_gates") == core["eligibility_gates_before_smoke"],
        "exact_smoke_runner_plan_and_job_identity_required")
    for name in ("amp_overflow_replays", "amp_scale_audit", "resource_profile"):
        require(engine.get(name) == core.get(name), "smoke_summary_and_exact_engine_evidence_mismatch")
    require(engine.get("checkpoint_sha256") == {
        name: core["artifact_files_sha256"]["job/" + name]
        for name in ("best_model.safetensors", "last_model.safetensors")},
        "engine_and_persisted_checkpoint_hashes_must_match")
    require(historical._utc(job.get("executed_at_utc")) <= historical._utc(core.get("executed_at_utc")),
        "completed_smoke_job_must_precede_core_receipt")
    return job, engine


def validate_smoke(directory, repository_root=None):
    """Authenticate an intact completed35 proof, without fitting or loading weights.

    The returned original summary preserves its UTC publication time and all
    source/package/reference digests for notebook36. The proof only establishes
    that the exact prior two-update smoke passed; it grants no additional scope.
    """
    root = checked_path(directory)
    core_path = checked_path(root / colab.CORE_FILENAME)
    core = read_json(core_path)
    summary = read_json(checked_path(root / colab.SUMMARY_FILENAME))
    require(isinstance(core, dict) and isinstance(summary, dict), "two_object_smoke_receipts_required")
    request = _validate_scope(core)
    colab.validate_result(summary)
    require(summary.get("code_revision") == SMOKE_REVISION
        and summary.get("smoke_record_sha256") == sha_file(core_path)
        and summary.get("result_files_saved_and_readback_verified") is True
        and summary.get("local_rehearsal") is False
        and summary.get("historical_cpu_preparation_authenticated") is True
        and summary.get("historical_cpu_preparation_suite_reexecuted") is False,
        "completed_actual_colab35_revision_and_exact_core_receipt_required")
    require(all(summary.get(key) == value for key, value in core.items() if key != "executed_at_utc"),
        "colab35_summary_and_core_content_mismatch")
    tests = summary.get("cpu_tests")
    require(isinstance(tests, dict) and tests.get("passed") is True
        and type(tests.get("tests_run")) is int and tests["tests_run"] == SMOKE_TEST_COUNT
        and type(tests.get("skipped")) is int and tests["skipped"] == 0
        and tests.get("scope") == list(colab.TEST_MODULES)
        and tests.get("test_runtime") == "gpu_build_cpu_compute_no_training"
        and tests.get("historical_cpu_preparation_suite_reexecuted") is False,
        "exact_completed_colab35_synthetic_test_scope_required")
    require(historical._utc(summary.get("executed_at_utc")) >= historical._utc(core.get("executed_at_utc")),
        "colab35_publication_must_follow_smoke_execution")
    repository = checked_path(repository_root if repository_root is not None else Path(__file__).resolve().parents[2])
    _validate_eligibility(core, repository)
    if repository_root is not None:
        require(all(sha_file(repository / name) == value for name, value in SOURCE_FILES.items()),
            "completed_smoke_scientific_runtime_sources_changed")
    job_root = _exact_inventory(root, core)
    _validate_job_links(core, job_root, request)
    # Recheck both receipts after the larger streamed artifact checks.
    require(read_json(core_path) == core
        and sha_file(core_path) == summary["smoke_record_sha256"]
        and read_json(root / colab.SUMMARY_FILENAME) == summary,
        "smoke_evidence_changed_during_authentication")
    return summary
