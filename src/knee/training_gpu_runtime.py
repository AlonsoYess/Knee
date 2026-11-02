"""Pinned isolated CUDA runtime for the bounded development-only GPU smoke.

The unmodified Colab kernel runs only this stdlib bootstrap. GPU imports and
checks happen in its separate Python 3.12.14 interpreter; no dataset, model,
optimizer or backward pass is created by this module. CPU fallback is forbidden.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

from knee import final_models_colab as helpers

VERSION = "isolated_training_gpu_runtime_v1.0"
CUDA_PACKAGES = {
    "torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124",
    "nvidia-cuda-nvrtc-cu12": "12.4.127", "nvidia-cuda-runtime-cu12": "12.4.127",
    "nvidia-cuda-cupti-cu12": "12.4.127", "nvidia-cudnn-cu12": "9.1.0.70",
    "nvidia-cublas-cu12": "12.4.5.8", "nvidia-cufft-cu12": "11.2.1.3",
    "nvidia-curand-cu12": "10.3.5.147", "nvidia-cusolver-cu12": "11.6.1.9",
    "nvidia-cusparse-cu12": "12.3.1.170", "nvidia-cusparselt-cu12": "0.6.2",
    "nvidia-nccl-cu12": "2.21.5", "nvidia-nvtx-cu12": "12.4.127",
    "nvidia-nvjitlink-cu12": "12.4.127", "triton": "3.2.0",
}
CLINICAL_PACKAGES = {"scikit-learn": "1.6.1", "scipy": "1.14.1", "joblib": "1.4.2",
    "threadpoolctl": "3.5.0", "xgboost-cpu": "3.0.5"}
GPU_WHEELS = {
    "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
        "#sha256=a393b506844035c0dac2f30ea8478c343b8e95a429f06f3b3cadfc7f53adb597",
    "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
        "#sha256=efb53ea0af7bf09b7b53e2a18b9be6d245f7d46a90b51d5cf97f37e9b929a991",
}
SOURCE_FILES = ("scripts/setup_model_inputs_cpu.py", "configs/model_inputs_colab_runtime_v1.json",
    "requirements-colab-model-inputs.txt", "requirements-final-models.txt", "requirements-training-runner.txt",
    "requirements-training-gpu.txt", "src/knee/training_gpu_runtime.py", "src/knee/final_models_colab.py")
RECEIPT_FILENAME = "training_gpu_runtime.json"

require, checked_path, sha_file = helpers.require, helpers.checked_path, helpers.sha_file


def _host(execution_environment="google_colab_gpu"):
    require(execution_environment == "google_colab_gpu", "explicit_google_colab_gpu_required")
    require(platform.system() == "Linux" and platform.machine() == "x86_64"
        and bool(os.environ.get("COLAB_RELEASE_TAG")), "actual_linux_x86_64_colab_required")
    require(os.environ.get("COLAB_GPU") == "1"
        and not any(os.environ.get(key) for key in ("COLAB_TPU_ADDR", "TPU_NAME", "COLAB_TPU_TASK_ID")),
        "select_gpu_in_colab_no_cpu_or_tpu_fallback")


def runtime_environment(repository):
    """Preserve assigned GPU visibility, not host Python paths or credentials."""
    repository = checked_path(repository)
    env = helpers._bootstrap(repository).isolated_environment()
    for key in tuple(env):
        upper = key.upper()
        if any(marker in upper for marker in ("TOKEN", "PASSWORD", "SECRET", "AUTHORIZATION", "API_KEY", "CREDENTIAL")):
            env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = os.pathsep.join(str(repository / child) for child in ("src", "tests"))
    return env


def _pins(path):
    require(path.is_file() and 0 < path.stat().st_size < 16384, "bounded_gpu_requirement_file_required")
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)", line)
        require(match is not None, "exact_gpu_dependency_pins_required")
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        require(name not in result, "duplicate_gpu_requirement")
        result[name] = version
    return result


def expected_packages(repository):
    """Keep every adopted CPU numerical/model pin; replace only CUDA backends."""
    repository = checked_path(repository)
    bootstrap = helpers._bootstrap(repository)
    bootstrap._configuration(repository)
    _, encoders = helpers._requirements(repository, bootstrap.EXPECTED_PACKAGES)
    clinical = _pins(repository / "requirements-training-runner.txt")
    gpu = _pins(repository / "requirements-training-gpu.txt")
    require(clinical == CLINICAL_PACKAGES and gpu == CUDA_PACKAGES, "frozen_training_gpu_dependency_set_required")
    return {**bootstrap.EXPECTED_PACKAGES, **encoders, **clinical, **gpu}


def verify_current_runtime(execution_environment="google_colab_gpu"):
    """Check actual CUDA support with a tiny eval-only operation, never training."""
    _host(execution_environment)
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "cublas_determinism_configuration_required_before_cuda")
    import torch
    import torchvision
    import numpy
    import sklearn
    import xgboost
    repository = Path(__file__).resolve().parents[2]
    expected = expected_packages(repository)
    actual = {name: importlib.metadata.version(name) for name in expected}
    require(platform.python_version() == "3.12.14" and actual == expected,
        "actual_frozen_training_gpu_packages_required")
    require(torch.__version__ == "2.6.0+cu124" and torchvision.__version__ == "0.21.0+cu124"
        and torch.version.cuda == "12.4" and torch.cuda.is_available(),
        "actual_cuda_12_4_runtime_required_no_cpu_fallback")
    count = torch.cuda.device_count()
    require(type(count) is int and count > 0, "visible_cuda_device_required")
    properties = torch.cuda.get_device_properties(0)
    require(properties.major >= 7, "cuda_fp16_capable_device_required")
    # No random state, model weights, gradients, optimizer or training is touched.
    with torch.inference_mode():
        value = torch.ones((2, 2), dtype=torch.float32, device="cuda:0")
        product = value @ value
        require(bool(torch.isfinite(product).all()) and bool((product == 2).all()),
            "cuda_eval_only_matmul_probe_failed")
    torch.cuda.synchronize()
    return {"version": VERSION, "execution_environment": execution_environment,
        "python": platform.python_version(), "packages": actual, "gpu_only": True, "cpu_only": False,
        "cuda_build": torch.version.cuda, "cuda_available": True, "gpu_device_count": count,
        "device": "cuda:0", "gpu_name": properties.name, "gpu_total_bytes": int(properties.total_memory),
        "gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "eval_only_cuda_matmul_verified": True, "training_executed": False,
        "optimizer_created": False, "backward_executed": False,
        "participant_data_opened": False, "reserved_test_predictive_evaluation_opened": False}


def _verify_installed(python, expected, repository):
    env = runtime_environment(repository)
    helpers._run([str(python), "-I", "-m", "pip", "check"], env, repository, "gpu_runtime_dependency_check")
    # Explicit script source entry instead of -I: -I would hide the checkout module.
    code = ("import json;from knee.training_gpu_runtime import verify_current_runtime;"
        "print(json.dumps(verify_current_runtime(),sort_keys=True,allow_nan=False))")
    info = json.loads(helpers._run([str(python), "-B", "-c", code], env, repository, "gpu_runtime_import_check"))
    require(info.get("packages") == expected and info.get("gpu_only") is True
        and info.get("training_executed") is False, "verified_isolated_gpu_runtime_required")
    return info


def setup_runtime(repository_root, destination):
    """Install pinned Python/CUDA without changing the host or old CPU setup."""
    _host()
    repository, destination = map(checked_path, (repository_root, destination))
    require(repository.is_dir() and destination.parent.is_dir(), "existing_repository_and_runtime_parent_required")
    require(not destination.is_relative_to(repository) and not repository.is_relative_to(destination)
        and not any(part.lower() in {"drive", "gdrive", "mydrive", "shared drives"} for part in destination.parts)
        and any(destination != root and destination.is_relative_to(root)
            for root in (checked_path(tempfile.gettempdir()), Path("/content"))),
        "isolated_local_temporary_gpu_runtime_required")
    expected = expected_packages(repository)
    source_hashes = {name: sha_file(repository / name) for name in SOURCE_FILES}
    receipt_path = destination / RECEIPT_FILENAME
    if destination.exists():
        require(destination.is_dir() and receipt_path.is_file(),
            "incomplete_gpu_runtime_use_new_session_without_deleting_files")
        receipt = helpers._json(receipt_path)
        require(receipt.get("version") == VERSION and receipt.get("source_sha256") == source_hashes
            and receipt.get("packages") == expected and receipt.get("gpu_only") is True
            and receipt.get("training_executed") is False, "existing_gpu_runtime_receipt_mismatch")
        python = checked_path(receipt["python_path"])
        require(python.is_file() and python.is_relative_to(destination), "isolated_gpu_python_required")
        gpu_info = _verify_installed(python, expected, repository)
        return {**receipt, "gpu_info": gpu_info, "existing_runtime_reverified": True}
    bootstrap = helpers._bootstrap(repository)
    config, _, _ = bootstrap._configuration(repository)
    destination.mkdir()
    downloads = destination / "downloads"
    downloads.mkdir()
    print("Preparando Python 3.12.14 y PyTorch CUDA 12.4 aislados; sin entrenar todavía…", file=sys.stderr)
    archive = bootstrap.download(config["python_archive"], downloads / config["python_archive"]["file"])
    python = bootstrap.extract_python(archive, destination / "runtime", config)
    env = runtime_environment(repository)
    version = helpers._run([str(python), "-I", "-c", "import platform,ssl,ensurepip;print(platform.python_version())"],
        env, destination, "isolated_gpu_python_version").strip()
    require(version == "3.12.14", "isolated_gpu_python_3_12_14_required")
    helpers._run([str(python), "-I", "-m", "ensurepip", "--upgrade"], env, destination, "gpu_runtime_ensurepip")
    pypi_pins = [name + "==" + version for name, version in sorted(expected.items()) if name not in GPU_WHEELS]
    helpers._run([str(python), "-I", "-m", "pip", "install", "--quiet", "--no-deps", "--no-cache-dir",
        "--only-binary=:all:", "--index-url", "https://pypi.org/simple", *pypi_pins],
        env, destination, "pinned_gpu_numerical_cuda_dependencies")
    helpers._run([str(python), "-I", "-m", "pip", "install", "--quiet", "--no-deps", "--no-cache-dir",
        "--only-binary=:all:", "--index-url", "https://download.pytorch.org/whl/cu124", *GPU_WHEELS.values()],
        env, destination, "hash_pinned_cuda_torch_wheels")
    gpu_info = _verify_installed(python, expected, repository)
    require(all(sha_file(repository / name) == digest for name, digest in source_hashes.items()),
        "gpu_runtime_setup_sources_changed")
    receipt = {"version": VERSION, "status": "isolated_training_gpu_runtime_ready",
        "python_path": str(python), "python_version": "3.12.14", "packages": expected,
        "source_sha256": source_hashes, "gpu_only": True, "cpu_only": False,
        "host_python_modified": False, "repository_modified": False, "participant_data_opened": False,
        "pretrained_weights_downloaded": False, "training_executed": False,
        "runtime_archive_sha256": bootstrap.PYTHON_SHA, "gpu_info": gpu_info,
        "existing_runtime_reverified": False}
    helpers._save(receipt_path, receipt)
    return receipt
