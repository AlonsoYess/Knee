"""CPU-only orchestration for preparing the registered training executor.

Notebook34 reads only the two completed notebook33 records. It never accepts
patient packages or invokes the training executor. Runtime setup remains in an
isolated temporary interpreter; the persistent output contains JSON records.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid

from knee import final_models_colab as previous


VERSION = "training_preparation_colab_orchestration_v1.0"
CORE_VERSION = "training_runner_preparation_v1.0"
CORE_STATUS = "runner_prepared_pending_training_authority"
CORE_FILENAME = "training_runner_preparation.json"
SUMMARY_FILENAME = "colab_training_preparation_check.json"
TEST_MODULES = previous.TEST_MODULES + (
    "test_training_authority", "test_training_data", "test_training_runner", "test_training_preparation",
    "test_training_models", "test_training_engine", "test_training_preparation_colab",
)
FALSE_FLAGS = (
    "training_executed", "optimizer_created", "backward_executed", "participant_data_opened",
    "reserved_test_predictive_evaluation_opened", "originals_modified", "partitions_modified",
    "training_authorized",
)
MAX_DIRECT_CHILDREN = 2048
MAX_PREVIOUS_RECORDS = 128

require = previous.require
checked_path = previous.checked_path
sha_file = previous.sha_file
_json = previous._json
_save = previous._save


def runtime_environment(repository):
    return previous.runtime_environment(repository)


def _requirements(repository, base):
    path = checked_path(repository / "requirements-training-runner.txt")
    require(path.is_file() and 0 < path.stat().st_size < 16384, "bounded_training_requirements_required")
    expected = {re.sub(r"[-_.]+", "-", name).lower(): value for name, value in base.items()}
    pins = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)", line)
        require(match is not None, "only_exact_training_requirement_pins_allowed")
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        require(name not in pins, "duplicate_training_requirement")
        require(name not in expected or version == expected[name], "training_requirements_must_preserve_cpu_base")
        pins[name] = version
    require(pins and {"scikit-learn", "xgboost-cpu"}.issubset(pins), "registered_clinical_dependencies_required")
    return path, pins


def setup_runtime(repository_root, destination):
    """Prepare exact clinical dependencies inside the approved encoder runtime."""
    previous._cpu_host("google_colab_cpu")
    repository, destination = map(checked_path, (repository_root, destination))
    require(repository.is_dir() and destination.parent.is_dir(), "existing_repository_and_runtime_parent_required")
    require(not destination.is_relative_to(repository) and not repository.is_relative_to(destination)
        and not any(part.lower() in {"drive", "gdrive", "mydrive", "shared drives"} for part in destination.parts)
        and any(destination != path and destination.is_relative_to(path)
            for path in (checked_path(tempfile.gettempdir()), Path("/content"))),
        "isolated_local_temporary_runtime_required")
    receipt_path = destination / "training_runtime.json"
    source_hashes = {name: sha_file(repository / name) for name in (
        "src/knee/training_preparation_colab.py", "requirements-training-runner.txt")}
    existed = destination.exists()
    if existed:
        require(destination.is_dir() and receipt_path.is_file(),
            "incomplete_training_runtime_use_new_session_without_deleting_files")
        receipt = _json(receipt_path)
        require(receipt.get("version") == VERSION and receipt.get("source_sha256") == source_hashes
            and receipt.get("training_executed") is False, "existing_training_runtime_receipt_mismatch")
        python = checked_path(receipt["python_path"])
        require(python.is_relative_to(destination) and python.is_file(), "isolated_training_python_required")
        previous._verify_installed(python, receipt["packages"], repository)
        return {**receipt, "existing_runtime_reverified": True}
    destination.mkdir()
    base = previous.setup_runtime(repository, destination / "encoders")
    requirements, pins = _requirements(repository, base["packages"])
    expected = {**base["packages"], **pins}
    python = checked_path(base["python_path"])
    env = runtime_environment(repository)
    print("Preparando bibliotecas clínicas fijadas en el intérprete CPU aislado…", file=sys.stderr)
    previous._run([str(python), "-I", "-m", "pip", "install", "--quiet", "--no-deps", "--no-cache-dir",
        "--only-binary=:all:", "--index-url", "https://pypi.org/simple", "-r", str(requirements)],
        env, repository, "training_runner_dependencies")
    previous._verify_installed(python, expected, repository)
    require(all(sha_file(repository / name) == digest for name, digest in source_hashes.items()),
        "training_runtime_setup_sources_changed")
    receipt = {"version": VERSION, "status": "isolated_training_preparation_cpu_runtime_ready",
        "python_path": str(python), "python_version": "3.12.14", "packages": expected,
        "source_sha256": source_hashes, "encoder_runtime_source_sha256": base["source_sha256"],
        "cpu_only": True, "host_python_modified": False, "participant_data_opened": False,
        "training_executed": False, "pretrained_weights_downloaded": False,
        "existing_runtime_reverified": False}
    _save(receipt_path, receipt)
    return receipt


def run_tests(repository_root):
    """Run the explicit synthetic preparation suite without skipped checks."""
    import unittest
    repository = checked_path(repository_root)
    require(checked_path(__file__) == repository / "src/knee/training_preparation_colab.py",
        "training_orchestration_module_must_match_checkout")
    sys.path.insert(0, str(repository / "tests"))
    with redirect_stdout(sys.stderr):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(TEST_MODULES)
        require(not loader.errors and suite.countTestCases() > 0, "complete_training_preparation_test_inventory_required")
        count = suite.countTestCases()
        result = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    require(result.wasSuccessful() and result.testsRun == count and not result.skipped
        and not result.expectedFailures and not result.unexpectedSuccesses,
        "training_preparation_tests_not_complete")
    return {"passed": True, "tests_run": count, "skipped": 0, "scope": list(TEST_MODULES)}


def _utc_timestamp(value):
    require(isinstance(value, str), "completed_preflight_timestamp_required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("completed_preflight_timestamp_required") from error
    require(result.tzinfo is not None and result.utcoffset() == timezone.utc.utcoffset(result),
        "completed_preflight_utc_timestamp_required")
    return result


def find_previous_preflight(persistent_root):
    """Read only bounded direct-child notebook33 records and select one latest proof."""
    from knee import training_preparation as core
    root = checked_path(persistent_root)
    require(root.is_dir(), "existing_private_records_folder_required")
    candidates = []
    with os.scandir(root) as entries:
        for count, entry in enumerate(entries, start=1):
            require(count <= MAX_DIRECT_CHILDREN, "private_records_direct_child_limit_exceeded")
            if not entry.name.startswith("final_model_check_"):
                continue
            path = checked_path(root / entry.name)
            if not path.is_dir():
                continue
            require(len(candidates) < MAX_PREVIOUS_RECORDS, "previous_preflight_record_limit_exceeded")
            if not (path / "colab_final_models_check.json").is_file():
                continue
            summary = core.validate_preflight(path)
            candidates.append((_utc_timestamp(summary.get("executed_at_utc")), path))
    require(candidates, "completed_notebook33_not_found_in_selected_records_folder")
    newest = max(item[0] for item in candidates)
    matches = [path for stamp, path in candidates if stamp == newest]
    require(len(matches) == 1, "latest_completed_notebook33_is_ambiguous")
    return matches[0]


def validate_result(result, execution_environment):
    require(execution_environment in {"local_cpu", "google_colab_cpu"}, "explicit_cpu_environment_required")
    require(isinstance(result, dict) and result.get("version") == CORE_VERSION
        and result.get("status") == CORE_STATUS
        and result.get("execution_environment") == execution_environment
        and result.get("colab_runtime_executed") is (execution_environment == "google_colab_cpu")
        and all(result.get(key) is False for key in FALSE_FLAGS), "bounded_training_preparation_result_required")
    require(result.get("training_runner_implemented") is True
        and result.get("synthetic_clinical_eval_forward_verified") is True
        and result.get("image_normalization_only_verified") is True
        and result.get("shortlist_search_closed") is True
        and result.get("pending_authorized_access") == ["image_dinov3_vits16"]
        and all(result.get(key) is False for key in ("actual_training_loop_executed",
            "training_smoke_test_passed", "gpu_resource_profile_recorded")),
        "complete_eval_only_training_preparation_required")


def prepare(repository_root, previous_preflight_dir, output_dir, *, execution_environment="google_colab_cpu"):
    previous._cpu_host(execution_environment)
    repository, source, output = map(checked_path, (repository_root, previous_preflight_dir, output_dir))
    require(repository.is_dir() and source.is_dir() and not output.exists(), "existing_sources_and_new_preparation_output_required")
    require(all(not left.is_relative_to(right) and not right.is_relative_to(left)
        for left, right in ((repository, source), (repository, output), (source, output))),
        "disjoint_code_preflight_and_preparation_directories_required")
    from knee import training_preparation as core
    require(checked_path(core.__file__) == repository / "src/knee/training_preparation.py"
        and checked_path(__file__) == repository / "src/knee/training_preparation_colab.py",
        "executed_training_preparation_modules_must_match_checkout")
    result = core.run(repository, source, output, execution_environment=execution_environment)
    validate_result(result, execution_environment)
    require(_json(output / CORE_FILENAME) == result, "training_preparation_core_readback_failed")
    return result


def publish_result(local_output, persistent_root, code_revision, cpu_tests, runtime):
    revision = previous._revision(code_revision)
    local, persistent = map(checked_path, (local_output, persistent_root))
    require(local.is_dir() and persistent.is_dir() and not local.is_relative_to(persistent)
        and not persistent.is_relative_to(local), "distinct_existing_local_and_persistent_roots_required")
    result = _json(local / CORE_FILENAME)
    validate_result(result, "google_colab_cpu")
    require(cpu_tests.get("passed") is True and type(cpu_tests.get("tests_run")) is int
        and cpu_tests["tests_run"] > 0 and cpu_tests.get("skipped") == 0
        and cpu_tests.get("scope") == list(TEST_MODULES), "completed_training_preparation_test_receipt_required")
    require(runtime.get("version") == VERSION and runtime.get("cpu_only") is True
        and runtime.get("training_executed") is False, "isolated_training_preparation_cpu_receipt_required")
    target = persistent / ("training_preparation_check_" + uuid.uuid4().hex)
    target.mkdir(exist_ok=False)
    _save(target / CORE_FILENAME, result)
    summary = {**result, "code_revision": revision, "cpu_tests": cpu_tests,
        "local_rehearsal": False, "cpu_only_guard_passed": True,
        "preparation_record_sha256": sha_file(target / CORE_FILENAME),
        "runtime_requirements_sha256": runtime.get("source_sha256", {}).get("requirements-training-runner.txt"),
        "result_files_saved_and_readback_verified": True,
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    _save(target / SUMMARY_FILENAME, summary)
    return summary
