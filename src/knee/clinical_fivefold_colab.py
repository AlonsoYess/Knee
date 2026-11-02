"""CPU Colab transport for three registered clinical models on sealed folds.

Every completed job is copied and read back before another job starts. Partial
jobs are retained but never resumed; only development data may be staged.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

from knee import training_preparation_colab as preparation
from knee import training_smoke_colab as transport

VERSION = "clinical_fivefold_colab_orchestration_v1.0"
CORE_VERSION = "clinical_three_model_fivefold_development_v1.0"
CORE_STATUS = "clinical_fivefold_development_complete"
CORE_FILENAME = "clinical_fivefold.json"
SUMMARY_FILENAME = "colab_clinical_fivefold_check.json"
CAMPAIGN_FILENAME = "campaign_identity.json"
CAMPAIGN_PREFIX = "clinical_fivefold_campaign_"
JOB_RECEIPT = "job_transport_receipt.json"
MODEL_IDS = ("clinical_logistic", "clinical_xgboost", "clinical_mlp")
APPROVAL_STATEMENT = (
    "Autorizo solo el entrenamiento en CPU de regresión logística, XGBoost y MLP clínicos "
    "en los mismos cinco pliegues de desarrollo, semilla 2026 y receta registrada; "
    "sin abrir la prueba reservada, entrenar modelos de imagen ni realizar calibración o ajuste de umbral."
)
TEST_MODULES = preparation.TEST_MODULES + (
    "test_training_amp", "test_training_smoke_evidence", "test_clinical_fivefold",
    "test_clinical_fivefold_colab", "test_clinical_fivefold_notebook",
)
SOURCE_FILES = (
    "src/knee/clinical_fivefold.py", "src/knee/clinical_fivefold_colab.py",
    "scripts/train_clinical_fivefold_colab.py", "src/knee/training_smoke.py",
    "src/knee/training_smoke_colab.py", "src/knee/training_preparation_colab.py",
    "src/knee/training_runner.py", "src/knee/training_authority.py",
    "src/knee/training_engine.py", "src/knee/training_models.py",
    "src/knee/training_data.py", "src/knee/clinical_fold_inputs.py",
    "src/knee/final_model_release.py", "src/knee/colab_package_transport.py",
    "src/knee/model_input_adapters.py", "src/knee/colab_data.py",
    "requirements-training-runner.txt", "requirements-final-models.txt",
    "requirements-model-input-loading.txt", "scripts/setup_model_inputs_cpu.py",
)
MAX_DIRECT_CHILDREN = preparation.MAX_DIRECT_CHILDREN
MAX_PREVIOUS_RECORDS = preparation.MAX_PREVIOUS_RECORDS
MAX_JOB_ATTEMPTS = 256
MAX_PUBLICATION_STAGES = 96
MAX_SINGLE_FILE_BYTES = 32 * 1024**2
MAX_JOB_BYTES = 64 * 1024**2
MAX_CAMPAIGN_BYTES = 1024**3
COMMON_FILES = {"training_job.json", "clinical_state.json", "best_validation_predictions.json"}
JOB_FILES = {
    "clinical_logistic": COMMON_FILES | {"clinical_model.json"},
    "clinical_xgboost": COMMON_FILES | {"clinical_model.ubj"},
    "clinical_mlp": COMMON_FILES | {"best_model.safetensors", "last_model.safetensors", "training_record.json"},
}

require = preparation.require
checked_path = preparation.checked_path
sha_file = preparation.sha_file
_json = preparation._json
_save = preparation._save
runtime_environment = preparation.runtime_environment
setup_runtime = preparation.setup_runtime
find_preparation = transport.find_preparation
stage_development = transport.stage_development


def run_tests(repository_root):
    import unittest
    repository = checked_path(repository_root)
    require(checked_path(__file__) == repository / "src/knee/clinical_fivefold_colab.py",
        "clinical_orchestration_must_match_checkout")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "clinical_synthetic_tests_must_hide_gpu")
    sys.path.insert(0, str(repository / "tests"))
    with redirect_stdout(sys.stderr):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(TEST_MODULES)
        require(not loader.errors and suite.countTestCases() > 0, "complete_clinical_test_inventory_required")
        count = suite.countTestCases()
        result = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    require(result.wasSuccessful() and result.testsRun == count and not result.skipped
        and not result.expectedFailures and not result.unexpectedSuccesses,
        "clinical_synthetic_tests_not_complete")
    return {"passed": True, "tests_run": count, "skipped": 0, "scope": list(TEST_MODULES),
        "test_runtime": "cpu_synthetic_no_participant_training"}


def _job_inventory(root, model_id=None):
    root = checked_path(root)
    require(root.is_dir(), "completed_clinical_job_directory_required")
    record = _json(root / "training_job.json")
    found = record.get("request", {}).get("model_id")
    require(found in MODEL_IDS and (model_id is None or model_id == found),
        "registered_clinical_model_job_required")
    paths = list(root.iterdir())
    require({p.name for p in paths} == JOB_FILES[found]
        and all(checked_path(p).parent == root and p.is_file() for p in paths),
        "only_exact_clinical_job_artifacts_may_be_persisted")
    require(all(0 < p.stat().st_size <= MAX_SINGLE_FILE_BYTES for p in paths)
        and sum(p.stat().st_size for p in paths) <= MAX_JOB_BYTES, "bounded_clinical_job_artifacts_required")
    return {p.name: sha_file(p) for p in paths}


def _copy_job(source, destination, expected):
    source, destination = map(checked_path, (source, destination))
    require(not destination.exists() and destination.parent.is_dir()
        and not destination.is_relative_to(source) and not source.is_relative_to(destination),
        "new_disjoint_clinical_transport_destination_required")
    require(_job_inventory(source) == expected, "clinical_job_changed_before_transport")
    destination.mkdir(exist_ok=False)
    for name in sorted(expected):
        with (source / name).open("rb") as incoming, (destination / name).open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024**2)
        require(sha_file(destination / name) == expected[name], "clinical_job_private_readback_failed")
    require(_job_inventory(source) == _job_inventory(destination) == expected,
        "clinical_job_changed_during_transport")


def _publication_stage(target):
    allowed = {CORE_FILENAME, SUMMARY_FILENAME, CAMPAIGN_FILENAME, JOB_RECEIPT}
    allowed.update(f"{model}_out_of_fold_predictions.json" for model in MODEL_IDS)
    require(target.name in allowed, "only_registered_clinical_records_may_be_staged")
    return target.parent / (".publication_" + uuid.uuid4().hex + "_" + target.name + ".part")


def _publish_json(target, value):
    stage = _publication_stage(target)
    _save(stage, value)
    checksum = sha_file(stage)
    _commit_verified_stage(stage, target, checksum)
    require(_json(target) == value, "clinical_record_publication_readback_failed")


def _commit_verified_stage(stage, target, checksum):
    stage, target = map(checked_path, (stage, target))
    require(stage.parent == target.parent and stage.is_file() and sha_file(stage) == checksum,
        "same_directory_verified_clinical_stage_required")
    if target.exists():
        require(target.is_file() and sha_file(target) == checksum,
            "existing_clinical_record_must_not_be_replaced")
        return
    stage.rename(target)
    require(target.is_file() and sha_file(target) == checksum, "clinical_committed_readback_failed")


def _publish_copy(source, target, checksum):
    if target.exists():
        require(target.is_file() and sha_file(source) == sha_file(target) == checksum,
            "existing_clinical_prediction_readback_failed")
        return
    stage = _publication_stage(target)
    with source.open("rb") as incoming, stage.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024**2)
    require(sha_file(stage) == sha_file(source) == checksum, "staged_clinical_prediction_readback_failed")
    _commit_verified_stage(stage, target, checksum)


def _campaign_identity(repository, preparation_dir, reference_path, revision):
    from knee.training_smoke import validate_preparation
    validate_preparation(preparation_dir)
    reference = _json(reference_path)
    require(reference.get("version") == "model_inputs_loading_preflight_v1.0"
        and reference.get("status") == "development_inputs_and_cpu_loading_verified",
        "verified_development_reference_required")
    package_sha = reference.get("sources_sha256", {}).get("development_package")
    require(isinstance(package_sha, str) and re.fullmatch(r"[0-9a-f]{64}", package_sha) is not None,
        "sealed_development_package_digest_required")
    return {"code_revision": preparation.previous._revision(revision),
        "source_files_sha256": {name: sha_file(checked_path(repository / name)) for name in SOURCE_FILES},
        "preparation_summary_sha256": sha_file(checked_path(preparation_dir / preparation.SUMMARY_FILENAME)),
        "model_input_reference_sha256": sha_file(reference_path), "development_package_sha256": package_sha,
        "model_ids": list(MODEL_IDS), "stage": "stage_1", "seed": 2026,
        "validation_folds": list(range(5)), "device": "cpu"}


def open_campaign(persistent_root, identity):
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
            require(seen <= MAX_PREVIOUS_RECORDS, "previous_clinical_campaign_record_limit_exceeded")
            manifest = path / CAMPAIGN_FILENAME
            if not manifest.is_file():
                continue
            value = _json(manifest)
            require(set(value) == {"version", "identity", "created_at_utc"} and value["version"] == VERSION,
                "valid_immutable_clinical_manifest_required")
            preparation._utc_timestamp(value["created_at_utc"])
            if value["identity"] == identity:
                matches.append(path)
    require(len(matches) <= 1, "matching_clinical_campaigns_are_ambiguous")
    if matches:
        return matches[0]
    target = root / (CAMPAIGN_PREFIX + uuid.uuid4().hex)
    target.mkdir(exist_ok=False)
    _publish_json(target / CAMPAIGN_FILENAME, {"version": VERSION, "identity": identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    return target


def completed_job_transports(campaign):
    campaign = checked_path(campaign)
    completed = {}
    paths = list(campaign.iterdir())
    require(len(paths) <= MAX_JOB_ATTEMPTS + MAX_PUBLICATION_STAGES + 6,
        "bounded_clinical_campaign_attempt_inventory_required")
    attempts, stages = 0, 0
    allowed = {CAMPAIGN_FILENAME, CORE_FILENAME, SUMMARY_FILENAME}
    allowed.update(f"{model}_out_of_fold_predictions.json" for model in MODEL_IDS)
    for path in paths:
        stage = re.fullmatch(r"\.publication_[0-9a-f]{32}_(.+)\.part", path.name)
        if stage:
            stages += 1
            require(stage[1] in allowed and stages <= MAX_PUBLICATION_STAGES and checked_path(path).is_file()
                and path.stat().st_size <= MAX_SINGLE_FILE_BYTES, "bounded_incomplete_clinical_stages_required")
            continue
        match = re.fullmatch(r"(clinical_logistic|clinical_xgboost|clinical_mlp)_fold_([0-4])_attempt_[0-9a-f]{32}", path.name)
        if not match:
            require(path.name in allowed and checked_path(path).is_file(), "only_registered_clinical_campaign_artifacts_allowed")
            continue
        attempts += 1
        require(attempts <= MAX_JOB_ATTEMPTS, "bounded_clinical_job_attempts_required")
        safe = checked_path(path)
        require(safe.is_dir(), "clinical_attempt_directory_required")
        receipt_path = safe / JOB_RECEIPT
        if not receipt_path.is_file():
            continue
        receipt = _json(receipt_path)
        key = (match[1], int(match[2]))
        require(set(receipt) == {"version", "model_id", "validation_fold", "files_sha256", "job_record_sha256"}
            and receipt["version"] == VERSION and receipt["model_id"] == key[0]
            and type(receipt["validation_fold"]) is int and receipt["validation_fold"] == key[1]
            and key not in completed and set(p.name for p in safe.iterdir()) == {"job", JOB_RECEIPT},
            "unique_exact_completed_clinical_transport_required")
        inventory = _job_inventory(safe / "job", key[0])
        job = _json(safe / "job/training_job.json")
        require(inventory == receipt["files_sha256"] and job.get("record_sha256") == receipt["job_record_sha256"]
            and job.get("request", {}).get("model_id") == key[0]
            and job.get("request", {}).get("validation_fold") == key[1],
            "private_clinical_transport_digest_or_identity_mismatch")
        completed[key] = {"directory": safe / "job", "receipt": receipt}
    return completed


def persist_completed_job(campaign, model_id, fold, job_dir, record):
    require(model_id in MODEL_IDS and type(fold) is int and fold in range(5)
        and record.get("request", {}).get("model_id") == model_id
        and record["request"].get("validation_fold") == fold, "exact_completed_clinical_job_required")
    campaign, job_dir = map(checked_path, (campaign, job_dir))
    inventory = _job_inventory(job_dir, model_id)
    require(_json(job_dir / "training_job.json") == record, "clinical_record_must_match_transport_source")
    old = completed_job_transports(campaign)
    if (model_id, fold) in old:
        item = old[(model_id, fold)]
        require(item["receipt"]["files_sha256"] == inventory
            and item["receipt"]["job_record_sha256"] == record["record_sha256"],
            "completed_clinical_job_cannot_be_overwritten")
        return item["directory"]
    attempt = campaign / (f"{model_id}_fold_{fold}_attempt_" + uuid.uuid4().hex)
    attempt.mkdir(exist_ok=False)
    _copy_job(job_dir, attempt / "job", inventory)
    _publish_json(attempt / JOB_RECEIPT, {"version": VERSION, "model_id": model_id,
        "validation_fold": fold, "files_sha256": inventory, "job_record_sha256": record["record_sha256"]})
    print(f"{model_id}: pliegue {fold + 1}/5 guardado en Drive y verificado.", file=sys.stderr, flush=True)
    return attempt / "job"


def validate_result(result):
    require(isinstance(result, dict) and result.get("version") == CORE_VERSION
        and result.get("status") == CORE_STATUS and result.get("execution_environment") == "google_colab_cpu"
        and result.get("colab_runtime_executed") is True and result.get("training_executed") is True,
        "completed_actual_colab_clinical_result_required")
    require(result.get("model_ids") == list(MODEL_IDS) and type(result.get("seed")) is int
        and result["seed"] == 2026 and result.get("stage") == "stage_1"
        and result.get("folds_completed") == list(range(5)) and type(result.get("job_count")) is int
        and result["job_count"] == 15 and result.get("scope") == "development_only"
        and result.get("clinical_features") == ["AGEYEARS", "SEX", "BMI"],
        "exact_three_registered_clinical_models_required")
    for key in ("cpu_only_guard_passed", "participant_disjoint_folds_verified",
            "clinical_preprocessing_fitted_inside_training_folds_only",
            "every_development_knee_predicted_once_per_model", "eligible_for_model_selection"):
        require(result.get(key) is True, "verified_clinical_development_scope_required")
    for key in ("reserved_test_accessed", "reserved_test_predictive_evaluation_opened", "originals_modified",
            "partitions_modified", "image_models_trained", "multimodal_models_trained",
            "pretrained_weights_loaded", "calibration_performed", "threshold_selected",
            "image_model_inference_executed", "full_encoder_training_authorized",
            "encoder_unfreezing_executed", "fold_optimizer_resume_supported",
            "predictive_performance_claimed", "unbiased_final_performance_claimed"):
        require(result.get(key) is False, "clinical_blocked_scope_must_remain_false")
    models = result.get("model_results")
    require(isinstance(models, dict) and set(models) == set(MODEL_IDS), "three_complete_clinical_model_results_required")
    for model_id in MODEL_IDS:
        rows = models[model_id].get("fold_results")
        require(isinstance(rows, list) and len(rows) == 5
            and [row.get("validation_fold") for row in rows] == list(range(5)),
            "every_clinical_model_requires_all_five_folds")


def _persist_final_artifacts(output, campaign, result):
    expected = {f"{model}/fold_{fold}/{name}" for model in MODEL_IDS for fold in range(5) for name in JOB_FILES[model]}
    expected.update(f"{model}/out_of_fold_predictions.json" for model in MODEL_IDS)
    artifacts = result.get("artifact_files_sha256")
    require(isinstance(artifacts, dict) and set(artifacts) == expected, "complete_clinical_artifact_inventory_required")
    require(all(sha_file(checked_path(output / name)) == checksum for name, checksum in artifacts.items()),
        "local_clinical_artifact_changed_before_finalization")
    completed = completed_job_transports(campaign)
    require(set(completed) == {(model, fold) for model in MODEL_IDS for fold in range(5)},
        "all_fifteen_jobs_must_be_durable_before_finalization")
    locations = {}
    for key, item in completed.items():
        model, fold = key
        for name in JOB_FILES[model]:
            source = f"{model}/fold_{fold}/{name}"
            target = item["directory"] / name
            require(sha_file(target) == artifacts[source], "durable_clinical_artifact_mismatch")
            locations[source] = target.relative_to(campaign).as_posix()
    for model in MODEL_IDS:
        source = f"{model}/out_of_fold_predictions.json"
        target = campaign / f"{model}_out_of_fold_predictions.json"
        _publish_copy(output / source, target, artifacts[source])
        locations[source] = target.name
    require(all(sha_file(checked_path(campaign / locations[name])) == checksum for name, checksum in artifacts.items()),
        "every_clinical_artifact_requires_verified_private_location")
    return locations


def _same_completed_campaign(stored, fresh):
    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items()
                if k not in {"executed_at_utc", "environment", "reused_authenticated_complete_job"}}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value
    require(stable(stored) == stable(fresh), "reused_complete_clinical_campaign_mismatch")


def train(repository_root, preparation_dir, package_dir, reference_path, output_dir, persistent_root, *,
        approval_statement, code_revision, cpu_tests, runtime):
    require(approval_statement == APPROVAL_STATEMENT, "explicit_clinical_fivefold_training_approval_required")
    require(cpu_tests.get("passed") is True and type(cpu_tests.get("tests_run")) is int
        and cpu_tests["tests_run"] > 0 and cpu_tests.get("skipped") == 0
        and cpu_tests.get("scope") == list(TEST_MODULES)
        and cpu_tests.get("test_runtime") == "cpu_synthetic_no_participant_training",
        "complete_clinical_synthetic_test_receipt_required")
    require(runtime.get("version") == preparation.VERSION and runtime.get("cpu_only") is True
        and runtime.get("training_executed") is False
        and runtime.get("pretrained_weights_downloaded") is False, "isolated_clinical_cpu_setup_receipt_required")
    from knee import clinical_fivefold as core
    require(core.APPROVAL_STATEMENT == APPROVAL_STATEMENT, "same_clinical_approval_statement_required")
    repository, proof, reference, output, persistent = map(checked_path,
        (repository_root, preparation_dir, reference_path, output_dir, persistent_root))
    require(checked_path(core.__file__) == repository / "src/knee/clinical_fivefold.py"
        and checked_path(__file__) == repository / "src/knee/clinical_fivefold_colab.py",
        "clinical_modules_must_match_published_checkout")
    require(not output.exists() and output.parent.is_dir() and persistent.is_dir()
        and not output.is_relative_to(persistent) and not persistent.is_relative_to(output),
        "new_temporary_clinical_output_separate_from_drive_required")
    require(all(not persistent.is_relative_to(source) and not source.is_relative_to(persistent)
        for source in map(checked_path, (repository, package_dir, reference.parent))),
        "private_clinical_campaign_must_not_overlap_code_data_or_reference")
    identity = _campaign_identity(repository, proof, reference, code_revision)
    campaign = open_campaign(persistent, identity)
    completed = completed_job_transports(campaign)
    reused = {}
    if completed:
        staged = output.parent / ("completed_clinical_staging_" + uuid.uuid4().hex)
        staged.mkdir(exist_ok=False)
        for (model, fold), item in sorted(completed.items()):
            destination = staged / f"{model}_fold_{fold}"
            _copy_job(item["directory"], destination, item["receipt"]["files_sha256"])
            reused[(model, fold)] = destination
        print(f"Encontrados {len(reused)}/15 trabajos completos; se verificarán de nuevo antes de reutilizarlos.",
            file=sys.stderr, flush=True)
    result = core.run(repository, proof, package_dir, reference, output,
        approval_statement=approval_statement, execution_environment="google_colab_cpu", completed_jobs=reused,
        job_completed=lambda model, fold, directory, record:
            persist_completed_job(campaign, model, fold, directory, record))
    validate_result(result)
    require(_json(output / CORE_FILENAME) == result, "clinical_core_result_readback_failed")
    require(_campaign_identity(repository, proof, reference, code_revision) == identity,
        "clinical_sources_changed_during_training")
    locations = _persist_final_artifacts(output, campaign, result)
    core_path = campaign / CORE_FILENAME
    if core_path.exists():
        stored = _json(core_path)
        validate_result(stored)
        _same_completed_campaign(stored, result)
        result = stored
    else:
        _publish_json(core_path, result)
    private_hashes = {name: sha_file(checked_path(campaign / name)) for name in locations.values()}
    for child in campaign.iterdir():
        if child.is_dir() and (child / JOB_RECEIPT).is_file():
            private_hashes[(child / JOB_RECEIPT).relative_to(campaign).as_posix()] = sha_file(child / JOB_RECEIPT)
    private_hashes[CAMPAIGN_FILENAME] = sha_file(campaign / CAMPAIGN_FILENAME)
    private_hashes[CORE_FILENAME] = sha_file(core_path)
    require(sum(checked_path(campaign / name).stat().st_size for name in private_hashes) <= MAX_CAMPAIGN_BYTES,
        "bounded_complete_clinical_campaign_required")
    summary_path = campaign / SUMMARY_FILENAME
    if summary_path.exists():
        summary = _json(summary_path)
        validate_result(summary)
        require(all(summary.get(key) == value for key, value in result.items() if key != "executed_at_utc")
            and summary.get("fivefold_record_sha256") == sha_file(core_path)
            and summary.get("code_revision") == code_revision and summary.get("private_artifact_locations") == locations
            and summary.get("private_files_sha256") == private_hashes
            and summary.get("result_files_saved_and_readback_verified") is True,
            "existing_clinical_campaign_summary_readback_required")
        return summary
    summary = {**result, "code_revision": code_revision, "cpu_tests": cpu_tests,
        "result_files_saved_and_readback_verified": True, "completed_jobs_saved_individually": 15,
        "persistent_campaign_folder": campaign.name, "within_job_optimizer_resume_used": False,
        "fivefold_record_sha256": sha_file(core_path), "local_rehearsal": False,
        "private_artifact_locations": locations, "private_files_sha256": private_hashes,
        "job_directories": {f"{model}/fold_{fold}": item["directory"].relative_to(campaign).as_posix()
            for (model, fold), item in sorted(completed_job_transports(campaign).items())},
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    _publish_json(summary_path, summary)
    require(_campaign_identity(repository, proof, reference, code_revision) == identity,
        "clinical_sources_changed_before_summary_publication")
    return summary
