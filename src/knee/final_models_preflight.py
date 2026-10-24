"""Verify the final registered image encoders without data or training.

The only inputs are repository configuration and official public checkpoints.
No dataset, outcome, partition or reserved-test argument exists in this API.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from knee.checkpoint_transport import acquire, file_digest
from knee.final_model_release import load_release, validate_plan, PINS, RELEASE_PATH, ADDED_IDS
from knee.training_plan import _require, _digest
from knee.pretrained_verification import load_encoder, probe_encoder

VERSION = "final_registered_checkpoint_cpu_preflight_v1.0"
REQUIRED_PACKAGES = {
    "torch": "2.6.0+cpu", "torchvision": "0.21.0+cpu", "numpy": "1.26.4",
    "transformers": "4.50.2", "tokenizers": "0.21.1", "regex": "2024.11.6",
    "safetensors": "0.5.3", "huggingface-hub": "0.30.2", "timm": "1.0.15",
    "packaging": "26.3", "requests": "2.34.2", "PyYAML": "6.0.3", "tqdm": "4.70.1",
    "certifi": "2026.7.22", "charset-normalizer": "3.5.2", "idna": "3.20", "urllib3": "2.8.0",
}


def _checked_path(path):
    value = Path(path).absolute()
    for parent in (value, *value.parents):
        _require(not parent.is_symlink() and not getattr(parent, "is_junction", lambda: False)(),
                 "verification_path_links_forbidden")
    return value.resolve()


def verify_environment(execution_environment):
    import torch
    _require(execution_environment in {"local_cpu", "google_colab_cpu"}, "explicit_execution_environment_required")
    _require(platform.python_version_tuple()[:2] == ("3", "12"), "isolated_python_312_required")
    if execution_environment == "google_colab_cpu":
        _require(bool(os.environ.get("COLAB_RELEASE_TAG")), "actual_colab_runtime_required")
    versions = {key: importlib.metadata.version(key) for key in REQUIRED_PACKAGES}
    _require(versions == REQUIRED_PACKAGES and torch.version.cuda is None and not torch.cuda.is_available(),
             "pinned_cpu_runtime_required")
    return {"python": platform.python_version(), **versions}


def checkpoint_files(model_id, catalog, weights_root, allow_download=False):
    _require(_digest(catalog) == PINS["configs/pretrained_weights_v1_1.json"], "final_catalog_changed")
    _require(type(allow_download) is bool, "explicit_boolean_download_authority_required")
    _require(model_id in catalog["models"], "registered_image_encoder_required")
    spec = catalog["models"][model_id]
    _require(spec["verification_enabled"] is True, "checkpoint_access_or_review_pending")
    paths = []
    for item in spec["files"]:
        path = _checked_path(Path(weights_root) / item["name"])
        if not path.is_file() and not allow_download:
            raise ValueError("checkpoint_not_cached_use_explicit_download_option")
        acquire(item["url"], path, expected_sha256=item["sha256"], expected_bytes=item["bytes"])
        paths.append(path)
    return paths


def synthetic_batch(model_id, catalog):
    import numpy as np
    import torch
    from knee.pretrained_final_encoders import normalize_sealed_image as normalize_added
    from knee.pretrained_convnextv2 import normalize_sealed_image as normalize_previous
    _require(_digest(catalog) == PINS["configs/pretrained_weights_v1_1.json"]
             and model_id in catalog["models"], "registered_catalog_normalization_required")
    spec = catalog["models"][model_id].get("input_normalization", catalog["input"])
    first = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(1, 224, 224)
    arrays = (first, np.ascontiguousarray(1 - first))
    normalized = []
    for array in arrays:
        array.setflags(write=False)
        before = hashlib.sha256(array.tobytes()).hexdigest()
        tensor = normalize_added(array, model_id) if model_id in ADDED_IDS else normalize_previous(array)
        reference = torch.from_numpy(array.copy()).repeat(3, 1, 1)
        reference = (reference - torch.tensor(spec["mean"], dtype=torch.float32)[:, None, None]) / torch.tensor(spec["std"], dtype=torch.float32)[:, None, None]
        _require(tensor.shape == (3, 224, 224) and torch.equal(tensor, reference),
                 "normalization_only_geometry_contract_failed")
        _require(before == hashlib.sha256(array.tobytes()).hexdigest() and not array.flags.writeable,
                 "source_synthetic_pixels_changed")
        normalized.append(tensor)
    return torch.stack(normalized)


def run(repository_root, weights_root, output_dir, *, allow_download=False, execution_environment="local_cpu"):
    import torch
    from knee.pretrained_final_encoders import load_final_encoder
    environment = verify_environment(execution_environment)
    release, catalog, plan, registry = load_release(repository_root)
    root, weights, output = map(_checked_path, (repository_root, weights_root, output_dir))
    _require(not output.exists() and output != root and root not in output.parents,
             "new_private_output_directory_outside_repository_required")
    _require(weights != root and root not in weights.parents and weights != output
             and weights not in output.parents and output not in weights.parents,
             "separate_private_cache_and_output_required")
    started = time.monotonic()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    records = {}
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        for model_id, spec in catalog["models"].items():
            if not spec["verification_enabled"]:
                records[model_id] = {"status": "pending_authorized_access", "checkpoint_loaded": False,
                                     "reason": spec["blocking_reason"]}
                print(f"{model_id}: acceso pendiente; no se descargaron pesos.", flush=True)
                continue
            print(f"Comprobando pesos en CPU, sin entrenamiento: {model_id}", flush=True)
            paths = checkpoint_files(model_id, catalog, weights, allow_download)
            model, adoption = load_final_encoder(model_id, paths) if model_id in ADDED_IDS else load_encoder(model_id, paths)
            probe = probe_encoder(model, synthetic_batch(model_id, catalog), spec["feature_dimension"])
            records[model_id] = {"status": "passed_synthetic_cpu_checkpoint_verification", "checkpoint_loaded": True,
                "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": file_digest(path)} for path in paths],
                "adoption": adoption, **probe, "training_smoke_test_passed": False,
                "gpu_resource_profile_recorded": False, "all_training_eligibility_gates_passed": False}
            del model
            gc.collect()
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    load_release(root)
    for model_id, record in records.items():
        if record["checkpoint_loaded"]:
            checkpoint_files(model_id, catalog, weights, False)
    _require(len(records) == 7 and sum(r["checkpoint_loaded"] for r in records.values()) == 6,
             "exact_six_accessible_checkpoints_required")
    result = {"version": VERSION, "status": "accessible_checkpoints_verified_gated_candidate_pending",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(), "execution_environment": execution_environment,
        "colab_runtime_executed": execution_environment == "google_colab_cpu", "local_rehearsal": execution_environment == "local_cpu",
        "cpu_only_guard_passed": True, "environment": environment, "model_records": records,
        "registered_image_models": 7, "pretrained_models_loaded": 6,
        "pending_model_ids": [key for key, record in records.items() if not record["checkpoint_loaded"]],
        "effective_release": {"descriptor": RELEASE_PATH, **release},
        "release_component_canonical_sha256": PINS, "training_configuration": validate_plan(plan, registry),
        "source_files_sha256": {name: file_digest(root / name) for name in (
            "src/knee/final_models_preflight.py", "src/knee/final_model_release.py",
            "src/knee/pretrained_final_encoders.py", "src/knee/pretrained_verification.py",
            "src/knee/pretrained_convnextv2.py", "src/knee/pretrained_skelex.py",
            "src/knee/checkpoint_transport.py", "requirements-final-models.txt")},
        "architecture_search_closed": True, "synthetic_inputs_only": True, "participant_data_opened": False,
        "training_executed": False, "optimizer_created": False, "gradients_created": False,
        "reserved_test_predictive_evaluation_opened": False, "originals_modified": False,
        "partitions_modified": False, "sealed_input_code_unchanged": True,
        "historical_registry_and_notebook32_unchanged": True, "new_model_selection_performed": False,
        "training_runner_implemented": False, "gpu_resource_profile_recorded": False,
        "predictive_or_clinical_validation_claimed": False,
        "limitations": ["CPU forward verification is not a training smoke test or predictive validation.",
            "DINOv3 remains pending authorized access; no gated weights were requested.",
            "External pretraining participant disjointness is not certified.",
            "Training runner and actual GPU training smoke test remain pending separate work and authority."],
        "elapsed_seconds": round(time.monotonic() - started, 3)}
    output.mkdir(parents=True, exist_ok=False)
    target = output / "checkpoint_verification.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _require(json.loads(target.read_text(encoding="utf-8")) == result, "verification_record_readback_failed")
    return result
