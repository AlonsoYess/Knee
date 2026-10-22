"""Thin Colab orchestration for one explicitly bounded development GPU smoke.

No tensor library is imported in the notebook host. Dataset transport is limited
to the sealed development manifest; the reserved archive is not a valid input.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import uuid

from knee import training_preparation_colab as previous


VERSION = "training_smoke_colab_orchestration_v1.0"
CORE_VERSION = "bounded_development_gpu_smoke_v1.0"
CORE_STATUS = "bounded_gpu_smoke_passed_pending_full_training_authority"
CORE_FILENAME = "training_smoke.json"
SUMMARY_FILENAME = "colab_training_smoke_check.json"
REFERENCE_FILENAME = "model_inputs_loading_reference_v1.json"
APPROVAL_STATEMENT = (
    "Autorizo solo la prueba breve GPU de DenseNet121 en desarrollo, pliegue 0, "
    "semilla 2026, dos actualizaciones de la cabeza; sin abrir prueba reservada "
    "ni entrenamiento completo."
)
GPU_INCOMPATIBLE_HISTORICAL_TEST_MODULES = ("test_training_preparation", "test_pretrained_verification")
TEST_MODULES = tuple(name for name in previous.TEST_MODULES
    if name not in GPU_INCOMPATIBLE_HISTORICAL_TEST_MODULES) + (
    "test_training_gpu_runtime", "test_training_smoke", "test_training_smoke_colab", "test_training_amp",
)
MAX_DIRECT_CHILDREN = previous.MAX_DIRECT_CHILDREN
MAX_PREVIOUS_RECORDS = previous.MAX_PREVIOUS_RECORDS
MAX_OUTPUT_FILES = 32
MAX_OUTPUT_BYTES = 256 * 1024**2
MAX_SINGLE_FILE_BYTES = 64 * 1024**2

require = previous.require
checked_path = previous.checked_path
sha_file = previous.sha_file
_json = previous._json
_save = previous._save


def runtime_environment(repository):
    from knee import training_gpu_runtime
    return training_gpu_runtime.runtime_environment(repository)


def setup_runtime(repository_root, destination):
    from knee import training_gpu_runtime
    return training_gpu_runtime.setup_runtime(repository_root, destination)


def run_tests(repository_root):
    """Execute the explicit mocked/synthetic suite, never the GPU training loop."""
    import unittest
    repository = checked_path(repository_root)
    require(checked_path(__file__) == repository / "src/knee/training_smoke_colab.py",
        "smoke_orchestrator_must_match_checkout")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "synthetic_tests_must_hide_gpu")
    sys.path.insert(0, str(repository / "tests"))
    with redirect_stdout(sys.stderr):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(TEST_MODULES)
        require(not loader.errors and suite.countTestCases() > 0, "complete_smoke_test_inventory_required")
        count = suite.countTestCases()
        result = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    require(result.wasSuccessful() and result.testsRun == count and not result.skipped
        and not result.expectedFailures and not result.unexpectedSuccesses,
        "training_smoke_synthetic_tests_not_complete")
    return {"passed": True, "tests_run": count, "skipped": 0, "scope": list(TEST_MODULES),
        "test_runtime": "gpu_build_cpu_compute_no_training",
        "historical_cpu_preparation_suite_reexecuted": False}


def find_preparation(persistent_root):
    """Inspect only bounded direct child notebook34 proofs, never image folders."""
    from knee import training_smoke as core
    root = checked_path(persistent_root)
    require(root.is_dir(), "existing_private_records_folder_required")
    candidates, seen = [], 0
    with os.scandir(root) as entries:
        for count, entry in enumerate(entries, start=1):
            require(count <= MAX_DIRECT_CHILDREN, "private_records_direct_child_limit_exceeded")
            if not entry.name.startswith("training_preparation_check_"):
                continue
            path = checked_path(root / entry.name)
            if not path.is_dir():
                continue
            seen += 1
            require(seen <= MAX_PREVIOUS_RECORDS, "previous_preparation_record_limit_exceeded")
            if not (path / previous.SUMMARY_FILENAME).is_file():
                continue
            summary = core.validate_preparation(path)
            candidates.append((previous._utc_timestamp(summary.get("executed_at_utc")), path))
    require(candidates, "completed_notebook34_not_found_in_selected_records_folder")
    newest = max(stamp for stamp, _ in candidates)
    matches = [path for stamp, path in candidates if stamp == newest]
    require(len(matches) == 1, "latest_completed_notebook34_is_ambiguous")
    return matches[0]


def stage_development(remote_root, local_root, reference_destination):
    """Stage only the manifest's development parts and the approved reference."""
    from knee import colab_package_transport as transport
    remote, local, reference_target = map(checked_path, (remote_root, local_root, reference_destination))
    require(remote.is_dir() and local.parent.is_dir() and not local.exists()
        and reference_target.parent.is_dir() and not reference_target.exists(),
        "existing_development_source_and_new_local_stage_required")
    require(not local.is_relative_to(remote) and not remote.is_relative_to(local)
        and not reference_target.is_relative_to(remote) and not remote.is_relative_to(reference_target)
        and not reference_target.is_relative_to(local), "separate_development_transport_paths_required")
    source_reference = checked_path(remote / REFERENCE_FILENAME)
    require(source_reference.is_file() and 0 < source_reference.stat().st_size <= 4 * 1024**2,
        "approved_development_input_reference_missing")
    reference_sha = sha_file(source_reference)
    # Validate the small reference before streaming the larger immutable parts.
    reference = _json(source_reference)
    require(reference.get("version") == "model_inputs_loading_preflight_v1.0"
        and reference.get("status") == "development_inputs_and_cpu_loading_verified",
        "approved_development_reference_required")
    local.mkdir(exist_ok=False)
    receipt = transport.stage_development_package(remote, local)
    require(receipt.get("training_executed") is False and receipt.get("reserved_test_opened") is False
        and receipt.get("sha256") == reference.get("sources_sha256", {}).get("development_package"),
        "staged_development_must_match_verified_input_reference")
    with source_reference.open("rb") as incoming, reference_target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024**2)
    require(sha_file(source_reference) == sha_file(reference_target) == reference_sha,
        "approved_input_reference_changed_during_transport")
    return {**receipt, "reference_path": str(reference_target), "reference_sha256": reference_sha}


def validate_result(result):
    require(isinstance(result, dict) and result.get("version") == CORE_VERSION
        and result.get("status") == CORE_STATUS
        and result.get("execution_environment") == "google_colab_gpu"
        and result.get("colab_runtime_executed") is True,
        "completed_actual_colab_gpu_smoke_required")
    for field in ("reserved_test_predictive_evaluation_opened", "originals_modified", "partitions_modified",
        "full_training_authorized", "full_training_executed", "eligible_for_model_selection",
        "full_encoder_training_authorized", "encoder_unfreezing_executed", "predictive_performance_claimed"):
        require(result.get(field) is False, "bounded_smoke_scope_must_be_preserved")
    require(result.get("training_executed") is True and result.get("training_smoke_test_passed") is True
        and result.get("gpu_resource_profile_recorded") is True,
        "actual_bounded_training_and_gpu_profile_required")
    require(result.get("model_id") == "image_densenet121" and result.get("validation_fold") == 0
        and type(result.get("validation_fold")) is int and result.get("seed") == 2026
        and type(result.get("seed")) is int and result.get("optimizer_updates") == 2
        and type(result.get("optimizer_updates")) is int,
        "exact_authorized_densenet_fold0_seed2026_two_updates_required")
    require(result.get("scope") == "development_only" and result.get("stage") == "stage_1"
        and result.get("smoke") is True and result.get("training_knees_seen") == 32
        and result.get("micro_batch_size") == 2 and result.get("effective_batch_size") == 16
        and result.get("frozen_encoder_state_unchanged") is True
        and result.get("task_head_update_verified") is True,
        "verified_head_only_smoke_and_frozen_encoder_required")
    require(result.get("amp_numerical_repair_version") == "same_effective_group_amp_replay_v1.0"
        and type(result.get("amp_overflow_replays")) is int and result["amp_overflow_replays"] >= 0
        and isinstance(result.get("amp_scale_audit"), list) and len(result["amp_scale_audit"]) == 2,
        "same_effective_group_amp_repair_audit_required")


def smoke(repository_root, preparation_dir, package_dir, reference_path, weights_root, output_dir,
        *, approval_statement, execution_environment="google_colab_gpu"):
    """Call only the sealed two-update smoke API; never a general training CLI."""
    from knee import training_smoke as core
    repository = checked_path(repository_root)
    require(checked_path(core.__file__) == repository / "src/knee/training_smoke.py"
        and checked_path(__file__) == repository / "src/knee/training_smoke_colab.py",
        "smoke_modules_must_match_published_checkout")
    result = core.run(repository, preparation_dir, package_dir, reference_path, weights_root, output_dir,
        approval_statement=approval_statement, execution_environment=execution_environment)
    validate_result(result)
    require(_json(checked_path(output_dir) / CORE_FILENAME) == result, "smoke_core_record_readback_failed")
    return result


def publish_result(local_output, persistent_root, code_revision, cpu_tests, runtime):
    """Copy the exact bounded private job into a new Drive folder and read it back."""
    revision = previous.previous._revision(code_revision)
    local, persistent = map(checked_path, (local_output, persistent_root))
    require(local.is_dir() and persistent.is_dir() and not local.is_relative_to(persistent)
        and not persistent.is_relative_to(local), "distinct_existing_smoke_and_persistent_roots_required")
    result = _json(local / CORE_FILENAME)
    validate_result(result)
    require(cpu_tests.get("passed") is True and type(cpu_tests.get("tests_run")) is int
        and cpu_tests["tests_run"] > 0 and cpu_tests.get("skipped") == 0
        and cpu_tests.get("scope") == list(TEST_MODULES)
        and cpu_tests.get("test_runtime") == "gpu_build_cpu_compute_no_training"
        and cpu_tests.get("historical_cpu_preparation_suite_reexecuted") is False,
        "complete_smoke_synthetic_test_receipt_required")
    require(runtime.get("version") == "isolated_training_gpu_runtime_v1.0"
        and runtime.get("gpu_only") is True and runtime.get("cpu_only") is False
        and runtime.get("training_executed") is False, "isolated_gpu_setup_receipt_required")
    files, before, total = [], {}, 0
    for path in local.rglob("*"):
        safe = checked_path(path)
        require(safe.is_relative_to(local), "smoke_output_must_remain_inside_job")
        if path.is_dir():
            continue
        require(path.is_file() and path.suffix in {".json", ".safetensors"}, "only_registered_smoke_artifacts_may_be_persisted")
        relative = path.relative_to(local).as_posix()
        require((relative == CORE_FILENAME or relative.startswith("job/"))
            and 0 < path.stat().st_size <= MAX_SINGLE_FILE_BYTES, "bounded_smoke_artifact_required")
        files.append(path)
        total += path.stat().st_size
        require(len(files) <= MAX_OUTPUT_FILES and total <= MAX_OUTPUT_BYTES, "bounded_smoke_output_inventory_required")
        before[relative] = sha_file(path)
    artifacts = result.get("artifact_files_sha256")
    require(isinstance(artifacts, dict)
        and before == {CORE_FILENAME: sha_file(local / CORE_FILENAME), **artifacts}
        and all(name.startswith("job/") for name in artifacts), "exact_sealed_smoke_artifacts_required")
    target = persistent / ("training_smoke_check_" + uuid.uuid4().hex)
    target.mkdir(exist_ok=False)
    for path in files:
        destination = target / path.relative_to(local)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024**2)
        require(sha_file(destination) == sha_file(path) == before[path.relative_to(local).as_posix()],
            "private_smoke_artifact_readback_failed")
    summary = {**result, "code_revision": revision, "cpu_tests": cpu_tests,
        "local_rehearsal": False, "result_files_saved_and_readback_verified": True,
        "historical_cpu_preparation_authenticated": True,
        "historical_cpu_preparation_suite_reexecuted": False,
        "smoke_record_sha256": sha_file(target / CORE_FILENAME),
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    _save(target / SUMMARY_FILENAME, summary)
    require(all(sha_file(local / name) == digest for name, digest in before.items()),
        "local_smoke_result_changed_during_publication")
    return summary
