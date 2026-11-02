"""Thin CPU-only Colab orchestration for the closed encoder shortlist.

No participant dataset path is accepted. Scientific checks live in
``final_models_preflight``; this module installs the isolated runtime, runs
synthetic tests and publishes a read-back-verified result without overwrites.
Imports stay stdlib-only so setup can run in the unmodified Colab interpreter.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import uuid


VERSION = "final_models_colab_orchestration_v1.0"
CORE_VERSION = "final_registered_checkpoint_cpu_preflight_v1.0"
CORE_STATUS = "accessible_checkpoints_verified_gated_candidate_pending"
IMAGE_MODELS = {"image_densenet121", "image_vit_b16", "image_convnextv2_tiny", "image_dinov3_vits16",
    "image_skelex", "image_biomedclip", "image_mambaout_tiny"}
TEST_MODULES = (
    "test_final_model_release", "test_final_models_preflight", "test_pretrained_final_encoders",
    "test_final_models_colab", "test_checkpoint_transport", "test_pretrained_verification",
    "test_pretrained_convnextv2", "test_pretrained_skelex", "test_training_plan",
    "test_model_registry", "test_model_inputs_cpu_setup",
)
FALSE_FLAGS = ("training_executed", "optimizer_created", "participant_data_opened",
    "reserved_test_predictive_evaluation_opened", "originals_modified", "partitions_modified")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_path(value):
    path = Path(value).absolute()
    for item in (path, *path.parents):
        require(not item.is_symlink() and not getattr(item, "is_junction", lambda: False)(),
            "orchestration_path_links_forbidden")
    return path.resolve()


def sha_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _json(path):
    path = checked_path(path)
    require(path.is_file() and 0 < path.stat().st_size <= 4 * 1024**2,
        "bounded_orchestration_record_required")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path, value):
    path = checked_path(path)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
    require(path.read_bytes() == raw and _json(path) == value, "orchestration_record_readback_failed")


def _revision(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None
        and value != "0" * 40, "published_exact_commit_required")
    return value


def _bootstrap(repository):
    filename = checked_path(repository / "scripts/setup_model_inputs_cpu.py")
    require(filename.is_file(), "approved_cpu_bootstrap_missing")
    spec = importlib.util.spec_from_file_location("knee_final_cpu_bootstrap", filename)
    require(spec is not None and spec.loader is not None, "cpu_bootstrap_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _requirements(repository, base):
    path = checked_path(repository / "requirements-final-models.txt")
    pins = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)", line)
        require(match is not None, "only_exact_final_model_requirement_pins_allowed")
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        require(name not in pins, "duplicate_final_model_requirement")
        base_versions = {re.sub(r"[-_.]+", "-", key).lower(): value for key, value in base.items()}
        require(name not in base_versions or version == base_versions[name],
            "final_requirements_must_not_change_frozen_cpu_base")
        pins[name] = version
    require(pins and pins.get("timm") == "1.0.15" and pins.get("transformers") == "4.50.2",
        "required_final_encoder_dependencies_missing")
    return path, pins


def runtime_environment(repository):
    bootstrap = _bootstrap(checked_path(repository))
    env = bootstrap.isolated_environment()
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
        CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = os.pathsep.join(str(checked_path(repository) / item) for item in ("src", "tests"))
    return env


def _cpu_host(execution_environment):
    require(execution_environment in {"google_colab_cpu", "local_cpu"}, "explicit_cpu_environment_required")
    require(os.environ.get("COLAB_GPU", "0") != "1"
        and not any(os.environ.get(key) for key in ("COLAB_TPU_ADDR", "TPU_NAME", "COLAB_TPU_TASK_ID")),
        "cpu_only_no_accelerator_required")
    if execution_environment == "google_colab_cpu":
        require(platform.system() == "Linux" and bool(os.environ.get("COLAB_RELEASE_TAG")),
            "actual_linux_colab_environment_required")


def _run(arguments, env, cwd, label):
    result = subprocess.run(arguments, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = re.sub(r"https?://\S+", "<url>", (result.stdout + result.stderr)[-10000:])
        detail = re.sub(r"(?i)(authorization|token|password|secret)\s*[:=]\s*\S+", r"\1=<redacted>", detail)
        print(detail, file=sys.stderr)
        raise RuntimeError(label + "_failed_exit_" + str(result.returncode))
    return result.stdout


def _verify_installed(python, expected, repository):
    code = ("import json,platform,importlib.metadata as m,torch,torchvision,numpy;"
        "print(json.dumps({'python':platform.python_version(),'packages':"
        "{p:m.version(p) for p in " + repr(list(expected)) + "},"
        "'cuda_available':torch.cuda.is_available(),'cuda_build':torch.version.cuda}))")
    env = runtime_environment(repository)
    _run([str(python), "-I", "-m", "pip", "check"], env, repository, "final_runtime_dependency_check")
    result = json.loads(_run([str(python), "-I", "-c", code], env, repository, "final_runtime_import_check"))
    require(result["python"] == "3.12.14" and result["packages"] == expected
        and result["cuda_available"] is False and result["cuda_build"] is None,
        "exact_final_cpu_runtime_required")
    return result


def setup_runtime(repository_root, destination):
    """Add pinned encoder libraries only inside the already isolated CPU runtime."""
    _cpu_host("google_colab_cpu")
    repository, destination = map(checked_path, (repository_root, destination))
    require(repository.is_dir() and destination.parent.is_dir(), "existing_repository_and_runtime_parent_required")
    require(not destination.is_relative_to(repository) and not repository.is_relative_to(destination)
        and not any(part.lower() in {"drive", "gdrive", "mydrive", "shared drives"} for part in destination.parts)
        and any(destination != path and destination.is_relative_to(path)
            for path in (checked_path(tempfile.gettempdir()), Path("/content"))),
        "isolated_local_temporary_runtime_required")
    bootstrap = _bootstrap(repository)
    requirements, pins = _requirements(repository, bootstrap.EXPECTED_PACKAGES)
    expected = {**bootstrap.EXPECTED_PACKAGES, **pins}
    source_hashes = {relative: sha_file(repository / relative) for relative in (
        "scripts/setup_model_inputs_cpu.py", "configs/model_inputs_colab_runtime_v1.json",
        "requirements-colab-model-inputs.txt", "requirements-final-models.txt", "src/knee/final_models_colab.py")}
    receipt_path = destination / "final_runtime.json"
    python = destination / "base/runtime/python/bin/python3.12"
    if destination.exists():
        require(destination.is_dir() and receipt_path.is_file() and python.is_file(),
            "incomplete_runtime_use_new_session_without_deleting_previous_files")
        receipt = _json(receipt_path)
        require(receipt.get("version") == VERSION and receipt.get("source_sha256") == source_hashes
            and receipt.get("python_path") == str(python) and receipt.get("packages") == expected
            and receipt.get("training_executed") is False, "existing_runtime_receipt_mismatch")
        _verify_installed(python, expected, repository)
        return {**receipt, "existing_runtime_reverified": True}
    destination.mkdir()
    base = bootstrap.setup(destination / "base", repository)
    require(checked_path(base["python_path"]) == python, "unexpected_isolated_python_location")
    env = runtime_environment(repository)
    print("Instalando bibliotecas de los encoders en el intérprete CPU aislado…", file=sys.stderr)
    _run([str(python), "-I", "-m", "pip", "install", "--quiet", "--no-deps", "--no-cache-dir",
        "--only-binary=:all:", "--index-url", "https://pypi.org/simple", "-r", str(requirements)],
        env, repository, "final_encoder_dependencies")
    _verify_installed(python, expected, repository)
    require(all(sha_file(repository / name) == value for name, value in source_hashes.items()),
        "final_runtime_setup_sources_changed")
    receipt = {"version": VERSION, "status": "isolated_final_model_cpu_runtime_ready",
        "python_path": str(python), "python_version": "3.12.14", "packages": expected,
        "source_sha256": source_hashes, "cpu_only": True, "host_python_modified": False,
        "participant_data_opened": False, "training_executed": False,
        "pretrained_weights_downloaded": False, "existing_runtime_reverified": False}
    _save(receipt_path, receipt)
    return receipt


def run_tests(repository_root):
    """Execute the explicit synthetic suite and reject errors, skips or empty suites."""
    import unittest
    repository = checked_path(repository_root)
    require(checked_path(__file__) == repository / "src/knee/final_models_colab.py",
        "orchestration_module_must_match_checkout")
    sys.path.insert(0, str(repository / "tests"))
    # Some tested helpers legitimately print progress. Keep CLI stdout exactly
    # one JSON object even when those test bodies exercise their logging.
    with redirect_stdout(sys.stderr):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(TEST_MODULES)
        require(not loader.errors and suite.countTestCases() > 0, "complete_final_model_test_inventory_required")
        expected = suite.countTestCases()
        result = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    require(result.wasSuccessful() and result.testsRun == expected and not result.skipped
        and not result.expectedFailures and not result.unexpectedSuccesses, "final_model_tests_not_complete")
    return {"passed": True, "tests_run": expected, "skipped": 0, "scope": list(TEST_MODULES)}


def validate_result(result, execution_environment):
    require(isinstance(result, dict) and result.get("version") == CORE_VERSION
        and result.get("status") == CORE_STATUS and result.get("execution_environment") == execution_environment
        and result.get("colab_runtime_executed") is (execution_environment == "google_colab_cpu")
        and result.get("registered_image_models") == 7 and result.get("pretrained_models_loaded") == 6
        and result.get("synthetic_inputs_only") is True
        and all(result.get(key) is False for key in FALSE_FLAGS), "bounded_final_model_result_required")
    records = result.get("model_records")
    require(isinstance(records, dict) and set(records) == IMAGE_MODELS
        and all(isinstance(item, dict) for item in records.values())
        and sum(item.get("checkpoint_loaded") is True for item in records.values()) == 6,
        "six_loaded_seven_registered_models_required")
    pending = [key for key, item in records.items() if item.get("checkpoint_loaded") is not True]
    require(pending == ["image_dinov3_vits16"]
        and records[pending[0]].get("status") == "pending_authorized_access",
        "only_registered_dinov3_access_may_remain_pending")
    for item in records.values():
        if item.get("checkpoint_loaded") is True:
            require(item.get("status") == "passed_synthetic_cpu_checkpoint_verification"
                and item.get("repeated_features_exact") is True
                and item.get("parameters_and_buffers_unchanged") is True
                and item.get("input_and_rng_unchanged") is True and item.get("gradients_created") is False,
                "complete_unchanged_synthetic_model_probe_required")


def verify(repository_root, weights_root, output_dir, *, allow_download=False, execution_environment="google_colab_cpu"):
    _cpu_host(execution_environment)
    repository, weights, output = map(checked_path, (repository_root, weights_root, output_dir))
    require(repository.is_dir() and not output.exists(), "new_verification_output_required")
    require(all(not left.is_relative_to(right) and not right.is_relative_to(left)
        for left, right in ((repository, weights), (repository, output), (weights, output))),
        "disjoint_repository_weights_and_result_directories_required")
    require(not any(part.lower() in {"drive", "gdrive", "mydrive", "shared drives"} for part in weights.parts),
        "checkpoint_cache_requires_local_storage_not_drive")
    from knee import final_models_preflight as core
    require(checked_path(core.__file__) == repository / "src/knee/final_models_preflight.py"
        and checked_path(__file__) == repository / "src/knee/final_models_colab.py",
        "executed_final_modules_must_match_checkout")
    result = core.run(repository, weights, output, allow_download=allow_download,
        execution_environment=execution_environment)
    validate_result(result, execution_environment)
    require(_json(output / "checkpoint_verification.json") == result, "core_result_readback_failed")
    return result


def publish_result(local_output, persistent_root, code_revision, cpu_tests, runtime):
    """Publish only two bounded JSON records, never patient files or checkpoints."""
    revision = _revision(code_revision)
    source, destination = map(checked_path, (local_output, persistent_root))
    require(source.is_dir() and destination.is_dir() and not source.is_relative_to(destination)
        and not destination.is_relative_to(source), "distinct_existing_local_and_persistent_roots_required")
    result = _json(source / "checkpoint_verification.json")
    validate_result(result, "google_colab_cpu")
    require(cpu_tests.get("passed") is True and type(cpu_tests.get("tests_run")) is int
        and cpu_tests["tests_run"] > 0 and cpu_tests.get("skipped") == 0
        and cpu_tests.get("scope") == list(TEST_MODULES), "completed_cpu_test_receipt_required")
    require(runtime.get("version") == VERSION and runtime.get("cpu_only") is True
        and runtime.get("training_executed") is False, "isolated_cpu_receipt_required")
    target = destination / ("final_model_check_" + uuid.uuid4().hex)
    target.mkdir(exist_ok=False)
    _save(target / "checkpoint_verification.json", result)
    summary = {"version": CORE_VERSION, "status": CORE_STATUS, "code_revision": revision,
        "colab_runtime_executed": True, "local_rehearsal": False, "cpu_only_guard_passed": True,
        "registered_image_models": 7, "pretrained_models_loaded": 6,
        "pending_authorized_access": ["image_dinov3_vits16"], "cpu_tests": cpu_tests,
        "synthetic_inputs_only": True, **{key: False for key in FALSE_FLAGS},
        "source_files_sha256": result.get("source_files_sha256", {}),
        "checkpoint_record_sha256": sha_file(target / "checkpoint_verification.json"),
        "runtime_requirements_sha256": runtime.get("source_sha256", {}).get("requirements-final-models.txt"),
        "result_files_saved_and_readback_verified": True,
        "training_smoke_test_passed": False, "gpu_resource_profile_recorded": False,
        "training_runner_implemented": result.get("training_runner_implemented", False),
        "training_authorized": False, "shortlist_search_closed": True,
        "next_step": "review_preflight_implement_training_runner_then_authorize_development_only_training_smoke_separately",
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    _save(target / "colab_final_models_check.json", summary)
    return summary
