"""Thin Colab transport for the approved frozen DenseNet five-fold campaign.

The notebook host imports only the standard library. Completed folds are copied
to private Drive and read back before the next fold starts. A disconnected fold
is restarted from the public checkpoint; optimizer continuation is not supported.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

from knee import training_smoke_colab as previous


VERSION = "densenet_fivefold_colab_orchestration_v1.0"
CORE_VERSION = "densenet121_frozen_fivefold_development_v1.0"
CORE_STATUS = "frozen_fivefold_development_complete"
CORE_FILENAME = "densenet_fivefold.json"
OOF_FILENAME = "out_of_fold_predictions.json"
SUMMARY_FILENAME = "colab_densenet_fivefold_check.json"
CAMPAIGN_PREFIX = "densenet_fivefold_campaign_"
CAMPAIGN_FILENAME = "campaign_identity.json"
FOLD_RECEIPT = "fold_transport_receipt.json"
APPROVAL_STATEMENT = (
    "Autorizo solo el entrenamiento de DenseNet121 con codificador congelado "
    "en los cinco pliegues de desarrollo, semilla 2026 y receta registrada; "
    "sin abrir prueba reservada, sin descongelar el codificador ni entrenar otros modelos."
)
TEST_MODULES = previous.TEST_MODULES + (
    "test_training_smoke_evidence", "test_densenet_fivefold", "test_densenet_fivefold_colab",
)
SOURCE_FILES = (
    "src/knee/densenet_fivefold.py", "src/knee/densenet_fivefold_colab.py",
    "src/knee/training_smoke_evidence.py", "src/knee/training_engine.py",
    "src/knee/training_runner.py", "src/knee/training_authority.py",
    "src/knee/training_models.py", "src/knee/training_data.py",
    "src/knee/training_gpu_runtime.py", "src/knee/final_model_release.py",
    "requirements-training-gpu.txt", "requirements-training-runner.txt",
)
JOB_FILES = frozenset(("training_job.json", "training_record.json", "clinical_state.json",
    "best_validation_predictions.json", "best_model.safetensors", "last_model.safetensors"))
MAX_DIRECT_CHILDREN = previous.MAX_DIRECT_CHILDREN
MAX_PREVIOUS_RECORDS = previous.MAX_PREVIOUS_RECORDS
MAX_FOLD_ATTEMPTS = 128
MAX_PUBLICATION_STAGES = 64
MAX_SINGLE_FILE_BYTES = 64 * 1024**2
MAX_JOB_BYTES = 256 * 1024**2
MAX_CAMPAIGN_BYTES = 1024**3

require = previous.require
checked_path = previous.checked_path
sha_file = previous.sha_file
_json = previous._json
_save = previous._save
runtime_environment = previous.runtime_environment
setup_runtime = previous.setup_runtime
stage_development = previous.stage_development


def run_tests(repository_root):
    """Run only mocked/synthetic tests while hiding all GPUs from the process."""
    import unittest
    repository = checked_path(repository_root)
    require(checked_path(__file__) == repository / "src/knee/densenet_fivefold_colab.py",
        "fivefold_orchestration_must_match_checkout")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "synthetic_tests_must_hide_gpu")
    sys.path.insert(0, str(repository / "tests"))
    with redirect_stdout(sys.stderr):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(TEST_MODULES)
        require(not loader.errors and suite.countTestCases() > 0, "complete_fivefold_test_inventory_required")
        count = suite.countTestCases()
        result = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    require(result.wasSuccessful() and result.testsRun == count and not result.skipped
        and not result.expectedFailures and not result.unexpectedSuccesses,
        "fivefold_synthetic_tests_not_complete")
    return {"passed": True, "tests_run": count, "skipped": 0, "scope": list(TEST_MODULES),
        "test_runtime": "gpu_build_cpu_compute_no_training"}


def find_smoke(persistent_root, repository_root=None):
    """Authenticate bounded direct-child notebook35 records; no recursive search."""
    from knee.training_smoke_evidence import validate_smoke
    root = checked_path(persistent_root)
    require(root.is_dir(), "existing_private_records_folder_required")
    candidates, seen = [], 0
    with os.scandir(root) as entries:
        for count, entry in enumerate(entries, start=1):
            require(count <= MAX_DIRECT_CHILDREN, "private_records_direct_child_limit_exceeded")
            if not entry.name.startswith("training_smoke_check_"):
                continue
            path = checked_path(root / entry.name)
            if not path.is_dir():
                continue
            seen += 1
            require(seen <= MAX_PREVIOUS_RECORDS, "previous_smoke_record_limit_exceeded")
            if not (path / previous.SUMMARY_FILENAME).is_file():
                continue
            summary = validate_smoke(path, repository_root)
            candidates.append((previous.previous._utc_timestamp(summary.get("executed_at_utc")), path))
    require(candidates, "completed_notebook35_not_found_in_selected_records_folder")
    newest = max(stamp for stamp, _ in candidates)
    matches = [path for stamp, path in candidates if stamp == newest]
    require(len(matches) == 1, "latest_completed_notebook35_is_ambiguous")
    return matches[0]


def _job_inventory(root):
    root = checked_path(root)
    require(root.is_dir(), "completed_fold_job_directory_required")
    paths = list(root.iterdir())
    require({path.name for path in paths} == JOB_FILES
        and all(checked_path(path).parent == root and path.is_file() for path in paths),
        "only_exact_registered_fold_artifacts_may_be_persisted")
    require(all(0 < path.stat().st_size <= MAX_SINGLE_FILE_BYTES for path in paths)
        and sum(path.stat().st_size for path in paths) <= MAX_JOB_BYTES,
        "bounded_completed_fold_artifacts_required")
    return {path.name: sha_file(path) for path in paths}


def _copy_job(source, destination, expected):
    source, destination = map(checked_path, (source, destination))
    require(not destination.exists() and destination.parent.is_dir()
        and not destination.is_relative_to(source) and not source.is_relative_to(destination),
        "new_disjoint_fold_transport_destination_required")
    require(_job_inventory(source) == expected, "completed_fold_changed_before_transport")
    destination.mkdir(exist_ok=False)
    for name in sorted(expected):
        with (source / name).open("rb") as incoming, (destination / name).open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024**2)
        require(sha_file(destination / name) == expected[name], "completed_fold_private_readback_failed")
    require(_job_inventory(source) == _job_inventory(destination) == expected,
        "completed_fold_changed_during_transport")


def _campaign_identity(repository, smoke_dir, reference_path, revision):
    from knee.training_smoke_evidence import validate_smoke
    validate_smoke(smoke_dir, repository)
    reference = _json(reference_path)
    require(reference.get("version") == "model_inputs_loading_preflight_v1.0"
        and reference.get("status") == "development_inputs_and_cpu_loading_verified",
        "verified_development_reference_required")
    package_sha = reference.get("sources_sha256", {}).get("development_package")
    require(isinstance(package_sha, str) and re.fullmatch(r"[0-9a-f]{64}", package_sha) is not None,
        "sealed_development_package_digest_required")
    return {"code_revision": previous.previous.previous._revision(revision),
        "source_files_sha256": {name: sha_file(checked_path(repository / name)) for name in SOURCE_FILES},
        "smoke_summary_sha256": sha_file(checked_path(smoke_dir / previous.SUMMARY_FILENAME)),
        "model_input_reference_sha256": sha_file(reference_path), "development_package_sha256": package_sha,
        "model_id": "image_densenet121", "stage": "stage_1", "seed": 2026, "validation_folds": list(range(5))}


def open_campaign(persistent_root, identity):
    """Reuse only one exactly matching immutable campaign; ambiguous matches stop."""
    root = checked_path(persistent_root)
    require(root.is_dir(), "existing_private_records_folder_required")
    matches, seen = [], 0
    with os.scandir(root) as entries:
        for count, entry in enumerate(entries, start=1):
            require(count <= MAX_DIRECT_CHILDREN, "private_records_direct_child_limit_exceeded")
            if not entry.name.startswith(CAMPAIGN_PREFIX):
                continue
            path = checked_path(root / entry.name)
            if not path.is_dir():
                continue
            seen += 1
            require(seen <= MAX_PREVIOUS_RECORDS, "previous_campaign_record_limit_exceeded")
            manifest = path / CAMPAIGN_FILENAME
            if not manifest.is_file():
                continue  # A failed initialization is retained, never reused.
            value = _json(manifest)
            require(set(value) == {"version", "identity", "created_at_utc"}
                and value["version"] == VERSION, "valid_immutable_campaign_manifest_required")
            previous.previous._utc_timestamp(value["created_at_utc"])
            if value["identity"] == identity:
                matches.append(path)
    require(len(matches) <= 1, "matching_fivefold_campaigns_are_ambiguous")
    if matches:
        return matches[0]
    target = root / (CAMPAIGN_PREFIX + uuid.uuid4().hex)
    target.mkdir(exist_ok=False)
    _publish_json(target / CAMPAIGN_FILENAME, {"version": VERSION, "identity": identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    return target


def completed_fold_transports(campaign):
    """Read only receipts written after all six artifacts passed Drive readback."""
    campaign = checked_path(campaign)
    completed = {}
    paths = list(campaign.iterdir())
    require(len(paths) <= MAX_FOLD_ATTEMPTS + MAX_PUBLICATION_STAGES + 4,
        "bounded_campaign_attempt_inventory_required")
    attempts, stages = 0, 0
    for path in paths:
        if re.fullmatch(r"\.publication_[0-9a-f]{32}_(?:out_of_fold_predictions|densenet_fivefold|colab_densenet_fivefold_check|campaign_identity)\.json\.part", path.name):
            stages += 1
            require(stages <= MAX_PUBLICATION_STAGES and checked_path(path).is_file()
                and path.stat().st_size <= MAX_SINGLE_FILE_BYTES, "bounded_incomplete_publication_stages_required")
            continue  # Never parse interrupted staging files as completed evidence.
        if not re.fullmatch(r"fold_[0-4]_attempt_[0-9a-f]{32}", path.name):
            require(path.name in {CAMPAIGN_FILENAME, CORE_FILENAME, SUMMARY_FILENAME, OOF_FILENAME} and path.is_file(),
                "only_registered_campaign_artifacts_allowed")
            continue
        attempts += 1
        require(attempts <= MAX_FOLD_ATTEMPTS, "bounded_campaign_fold_attempts_required")
        safe = checked_path(path)
        require(safe.is_dir(), "fold_attempt_directory_required")
        receipt_file = safe / FOLD_RECEIPT
        if not receipt_file.is_file():
            continue  # Partial copies are preserved but are never completed folds.
        receipt = _json(receipt_file)
        fold = int(path.name[5])
        require(set(receipt) == {"version", "validation_fold", "files_sha256", "job_record_sha256"}
            and receipt["version"] == VERSION and type(receipt["validation_fold"]) is int
            and receipt["validation_fold"] == fold and fold not in completed
            and set(item.name for item in safe.iterdir()) == {"job", FOLD_RECEIPT},
            "unique_exact_completed_fold_transport_required")
        inventory = _job_inventory(safe / "job")
        require(inventory == receipt["files_sha256"]
            and _json(safe / "job/training_job.json").get("record_sha256") == receipt["job_record_sha256"],
            "private_completed_fold_transport_digest_mismatch")
        completed[fold] = {"directory": safe / "job", "receipt": receipt}
    return completed


def persist_completed_fold(campaign, fold, job_dir, job_record):
    """Checkpoint boundary: persist one fully authenticated fold before continuing."""
    require(type(fold) is int and fold in range(5)
        and job_record.get("request", {}).get("validation_fold") == fold,
        "exact_completed_fold_index_required")
    campaign, job_dir = map(checked_path, (campaign, job_dir))
    inventory = _job_inventory(job_dir)
    require(_json(job_dir / "training_job.json") == job_record,
        "completed_fold_record_must_match_transport_source")
    old = completed_fold_transports(campaign)
    if fold in old:
        require(old[fold]["receipt"]["files_sha256"] == inventory
            and old[fold]["receipt"]["job_record_sha256"] == job_record["record_sha256"],
            "completed_fold_cannot_be_overwritten_with_different_training")
        return old[fold]["directory"]
    attempt = campaign / (f"fold_{fold}_attempt_" + uuid.uuid4().hex)
    attempt.mkdir(exist_ok=False)
    _copy_job(job_dir, attempt / "job", inventory)
    _publish_json(attempt / FOLD_RECEIPT, {"version": VERSION, "validation_fold": fold,
        "files_sha256": inventory, "job_record_sha256": job_record["record_sha256"]})
    print(f"Pliegue {fold + 1}/5 guardado en Drive y verificado archivo por archivo.", file=sys.stderr, flush=True)
    return attempt / "job"


def validate_result(result):
    require(isinstance(result, dict) and result.get("version") == CORE_VERSION
        and result.get("status") == CORE_STATUS and result.get("execution_environment") == "google_colab_gpu"
        and result.get("colab_runtime_executed") is True, "completed_actual_colab_fivefold_result_required")
    require(result.get("model_id") == "image_densenet121" and result.get("stage") == "stage_1"
        and type(result.get("seed")) is int and result["seed"] == 2026
        and result.get("folds_completed") == list(range(5)) and result.get("training_executed") is True,
        "exact_frozen_densenet_fivefold_scope_required")
    for field in ("reserved_test_predictive_evaluation_opened", "originals_modified", "partitions_modified",
        "encoder_unfreezing_executed", "other_models_trained", "predictive_performance_claimed",
        "full_encoder_training_authorized", "calibration_performed", "threshold_selected",
        "reserved_test_accessed", "fold_optimizer_resume_supported", "unbiased_final_performance_claimed"):
        require(result.get(field) is False, "fivefold_blocked_scope_must_remain_false")
    folds = result.get("fold_results")
    require(isinstance(folds, list) and len(folds) == 5
        and [row.get("validation_fold") for row in folds] == list(range(5))
        and all(row.get("frozen_encoder_state_unchanged") is True for row in folds),
        "five_completed_folds_with_unchanged_encoders_required")


def _publication_stage(target):
    require(target.name in {CORE_FILENAME, OOF_FILENAME, SUMMARY_FILENAME, CAMPAIGN_FILENAME, FOLD_RECEIPT},
        "only_registered_final_records_may_be_staged")
    return target.parent / (".publication_" + uuid.uuid4().hex + "_" + target.name + ".part")


def _commit_verified_stage(stage, target, checksum):
    """Publish one verified record; the campaign has exactly one active writer.

Both paths share a directory so portable rename is the publication boundary.
Do not run two copies of notebook36 against the same campaign concurrently.
Existing completed evidence is authenticated and reused, never replaced.
"""
    stage, target = map(checked_path, (stage, target))
    require(stage.parent == target.parent and stage.is_file() and sha_file(stage) == checksum,
        "same_directory_verified_publication_stage_required")
    if target.exists():
        require(target.is_file() and sha_file(target) == checksum,
            "existing_completed_final_record_must_not_be_replaced")
        return
    require(not target.exists(), "new_final_record_destination_required")
    stage.rename(target)
    require(target.is_file() and sha_file(target) == checksum, "committed_final_record_readback_failed")


def _publish_json(target, value):
    """A failed JSON write leaves only a bounded ignored staging file."""
    stage = _publication_stage(target)
    _save(stage, value)
    checksum = sha_file(stage)
    _commit_verified_stage(stage, target, checksum)
    require(_json(target) == value, "published_final_json_readback_failed")


def _publish_copy(source, target, checksum):
    """Copy/hash the entire OOF file before exposing its final name in Drive."""
    if target.exists():
        require(target.is_file() and sha_file(target) == sha_file(source) == checksum,
            "private_oof_prediction_readback_failed")
        return
    stage = _publication_stage(target)
    with source.open("rb") as incoming, stage.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024**2)
    require(sha_file(stage) == sha_file(source) == checksum, "staged_oof_prediction_readback_failed")
    _commit_verified_stage(stage, target, checksum)


def _persist_final_artifacts(output, campaign, result):
    """Preserve all 31 declared artifacts, mapping local names to durable locations."""
    artifacts = result.get("artifact_files_sha256")
    expected = {f"fold_{fold}/{name}" for fold in range(5) for name in JOB_FILES} | {OOF_FILENAME}
    require(isinstance(artifacts, dict) and set(artifacts) == expected,
        "complete_fivefold_artifact_inventory_required")
    require(all(sha_file(checked_path(output / name)) == checksum for name, checksum in artifacts.items()),
        "local_fivefold_artifact_changed_before_finalization")
    completed = completed_fold_transports(campaign)
    require(set(completed) == set(range(5)), "all_five_folds_must_be_durable_before_summary")
    locations = {}
    for fold, item in sorted(completed.items()):
        for name in sorted(JOB_FILES):
            source_name = f"fold_{fold}/{name}"
            private_path = item["directory"] / name
            require(sha_file(private_path) == artifacts[source_name],
                "durable_fold_must_match_final_artifact_inventory")
            locations[source_name] = private_path.relative_to(campaign).as_posix()
    source, target = output / OOF_FILENAME, campaign / OOF_FILENAME
    require(0 < source.stat().st_size <= MAX_SINGLE_FILE_BYTES, "bounded_oof_predictions_required")
    _publish_copy(source, target, artifacts[OOF_FILENAME])
    require(sha_file(target) == sha_file(source) == artifacts[OOF_FILENAME],
        "private_oof_prediction_readback_failed")
    locations[OOF_FILENAME] = OOF_FILENAME
    require(all(sha_file(campaign / locations[name]) == checksum for name, checksum in artifacts.items()),
        "every_declared_artifact_requires_verified_private_location")
    return locations


def _same_completed_campaign(previous_result, recomputed):
    """Ignore only fresh invocation metadata, not metrics, checkpoints or provenance."""
    excluded = {"executed_at_utc", "environment", "fold_results"}
    require({key:value for key,value in previous_result.items() if key not in excluded}
        == {key:value for key,value in recomputed.items() if key not in excluded},
        "existing_completed_campaign_scientific_record_mismatch")
    strip_reuse = lambda rows: [{key:value for key,value in row.items()
        if key != "reused_authenticated_complete_fold"} for row in rows]
    require(strip_reuse(previous_result["fold_results"]) == strip_reuse(recomputed["fold_results"]),
        "existing_completed_campaign_fold_evidence_mismatch")


def _private_artifact_hashes(campaign, locations):
    relative = set(locations.values()) | {CORE_FILENAME, CAMPAIGN_FILENAME}
    relative.update((item["directory"].parent / FOLD_RECEIPT).relative_to(campaign).as_posix()
        for item in completed_fold_transports(campaign).values())
    staging_bytes = sum(path.stat().st_size for path in campaign.iterdir()
        if re.fullmatch(r"\.publication_[0-9a-f]{32}_(?:out_of_fold_predictions|densenet_fivefold|colab_densenet_fivefold_check|campaign_identity)\.json\.part", path.name))
    require(sum((campaign / name).stat().st_size for name in relative) + staging_bytes <= MAX_CAMPAIGN_BYTES,
        "bounded_fivefold_private_campaign_required")
    return {name: sha_file(checked_path(campaign / name)) for name in sorted(relative)}


def train(repository_root, smoke_dir, package_dir, reference_path, weights_root, output_dir,
        persistent_root, *, approval_statement, code_revision, cpu_tests, runtime):
    """Run exactly the approved five folds with a verified persistence boundary."""
    require(approval_statement == APPROVAL_STATEMENT, "explicit_fivefold_training_approval_required")
    require(cpu_tests.get("passed") is True and type(cpu_tests.get("tests_run")) is int
        and cpu_tests["tests_run"] > 0 and cpu_tests.get("skipped") == 0
        and cpu_tests.get("scope") == list(TEST_MODULES)
        and cpu_tests.get("test_runtime") == "gpu_build_cpu_compute_no_training",
        "complete_fivefold_synthetic_test_receipt_required")
    require(runtime.get("version") == "isolated_training_gpu_runtime_v1.0"
        and runtime.get("gpu_only") is True and runtime.get("cpu_only") is False
        and runtime.get("training_executed") is False, "isolated_gpu_setup_receipt_required")
    from knee import densenet_fivefold as core
    repository, smoke, reference, output, persistent = map(checked_path,
        (repository_root, smoke_dir, reference_path, output_dir, persistent_root))
    require(checked_path(core.__file__) == repository / "src/knee/densenet_fivefold.py"
        and checked_path(__file__) == repository / "src/knee/densenet_fivefold_colab.py",
        "fivefold_modules_must_match_published_checkout")
    require(not output.exists() and output.parent.is_dir() and persistent.is_dir()
        and not output.is_relative_to(persistent) and not persistent.is_relative_to(output),
        "new_temporary_output_separate_from_private_drive_required")
    require(all(not persistent.is_relative_to(source) and not source.is_relative_to(persistent)
        for source in map(checked_path, (repository, package_dir, reference.parent, weights_root))),
        "private_campaign_must_not_overlap_code_data_reference_or_public_weights")
    identity = _campaign_identity(repository, smoke, reference, code_revision)
    campaign = open_campaign(persistent, identity)
    completed = completed_fold_transports(campaign)
    reused = {}
    if completed:
        resumed = output.parent / ("completed_fold_staging_" + uuid.uuid4().hex)
        resumed.mkdir(exist_ok=False)
        for fold, item in sorted(completed.items()):
            destination = resumed / f"fold_{fold}"
            _copy_job(item["directory"], destination, item["receipt"]["files_sha256"])
            reused[fold] = destination
        print(f"Encontrados {len(reused)} pliegues completos. Se autenticarán de nuevo; no se reutilizan intentos parciales.",
            file=sys.stderr, flush=True)
    result = core.run(repository, smoke, package_dir, reference, weights_root, output,
        approval_statement=approval_statement, execution_environment="google_colab_gpu",
        completed_folds=reused, fold_completed=lambda fold, path, record:
            persist_completed_fold(campaign, fold, path, record))
    validate_result(result)
    require(_json(output / CORE_FILENAME) == result, "fivefold_core_result_readback_failed")
    require(_campaign_identity(repository, smoke, reference, code_revision) == identity,
        "campaign_inputs_or_sources_changed_during_training")
    locations = _persist_final_artifacts(output, campaign, result)
    core_path = campaign / CORE_FILENAME
    if core_path.exists():
        # A complete older campaign can be reauthenticated, never rewritten.
        stored = _json(core_path)
        validate_result(stored)
        _same_completed_campaign(stored, result)
        result = stored
    else:
        _publish_json(core_path, result)
    private_hashes = _private_artifact_hashes(campaign, locations)
    summary_path = campaign / SUMMARY_FILENAME
    if summary_path.exists():
        summary = _json(summary_path)
        validate_result(summary)
        require(all(summary.get(key) == value for key,value in result.items() if key != "executed_at_utc")
            and summary.get("fivefold_record_sha256") == sha_file(core_path)
            and summary.get("code_revision") == code_revision
            and summary.get("private_artifact_locations") == locations
            and summary.get("private_files_sha256") == private_hashes
            and summary.get("result_files_saved_and_readback_verified") is True,
            "existing_campaign_summary_readback_required")
        return summary
    summary = {**result, "code_revision": code_revision, "cpu_tests": cpu_tests,
        "result_files_saved_and_readback_verified": True, "completed_folds_saved_individually": 5,
        "persistent_campaign_folder": campaign.name, "within_fold_optimizer_resume_used": False,
        "fivefold_record_sha256": sha_file(core_path), "local_rehearsal": False,
        "private_artifact_locations": locations, "private_files_sha256": private_hashes,
        "fold_job_directories": {str(fold): item["directory"].relative_to(campaign).as_posix()
            for fold,item in sorted(completed_fold_transports(campaign).items())},
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    _publish_json(summary_path, summary)
    require(_campaign_identity(repository, smoke, reference, code_revision) == identity,
        "campaign_inputs_or_sources_changed_during_training")
    return summary
