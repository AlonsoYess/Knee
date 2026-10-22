"""Verify registered checkpoints on synthetic inputs, with no training or data.

This deliberately has no cohort/package argument. The user can run the same
entrypoint in Colab. A successful CPU forward is not a training smoke test, a
GPU resource profile, a license grant, or evidence of predictive performance.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

from knee.checkpoint_transport import acquire, file_digest, _https, SOURCE_HOSTS
from knee.training_plan import load_training_plan, validate_plan, registered_model_ids

VERSION = "registered_checkpoint_cpu_preflight_v1.0"
CATALOG_CANONICAL_SHA256 = "8d886a77aa9f71b41a43110e9ce40df39e1f0a0caf676b42f4973d11e107454a"
SOURCE_PINS = {
    "src/knee/clinical_fold_inputs.py": "05d2165af795bd3b7e95f2d4b4120a6f9964958bde74fb333253708511dedbfd",
    "src/knee/model_input_adapters.py": "82635dad0e577a770488ae41b6647ab6942f5ba37f0716be385856542f2ebc50",
    "src/knee/model_inputs_preflight.py": "aaeb92976122af3ac18bfab11c2847d577b26e971aab2a3c1d02da74fe2adc36",
    "src/knee/colab_data.py": "5b2df195c68e9c432366a9599fd0f1e835712dd2ca60ac92a9c8e556b41d2a6f",
    "scripts/prepare_model_inputs.py": "27add7a275e6a65da3634858dd3e5982ab65f08dfc6438866be5efb8d323b063",
    "configs/model_inputs_preflight_v1.json": "c491534b0098a54a16d88264c9135d2e4c6769970b5e3fe6ff08755123e7e8fd",
    "configs/model_registry.json": "a1bec8b1d599fc7871f7e80e91a3d09f256b3aed7eeac50af816f9405902feac",
    "requirements-model-input-loading.txt": "78808930f554a419b88a3866610b6821cbc2db726943993cdd45901c900fd87d",
}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_catalog(repository_root):
    from knee.training_plan import _read_json
    root = Path(repository_root)
    catalog = _read_json(root / "configs/pretrained_weights_v1.json")
    _require(canonical_digest(catalog) == CATALOG_CANONICAL_SHA256,
        "reviewed_checkpoint_catalog_changed")
    plan, registry = load_training_plan(root / "configs/training_plan_v1.json",
        root / "configs/model_registry.json")
    image_ids = {item["id"] for key in ("mandatory_baselines", "modern_challengers")
        for item in registry["radiographic"][key]}
    _require(set(catalog["models"]) == image_ids
        and set(catalog["models"]) | set(catalog["models_without_external_pretrained_checkpoint"])
        == set(registered_model_ids(registry)), "exact_registered_checkpoint_coverage_required")
    for model_id, spec in catalog["models"].items():
        _require(type(spec["verification_enabled"]) is bool, "explicit_checkpoint_gate_required")
        for item in spec["files"]:
            _require(Path(item["name"]).name == item["name"] and "/" not in item["name"]
                and "\\" not in item["name"] and item["name"] not in {".", ".."},
                "flat_checkpoint_filename_required")
            _https(item["url"], SOURCE_HOSTS)
            _require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None
                and type(item["bytes"]) is int and 0 < item["bytes"] < 2_000_000_000,
                "full_hash_and_bounded_size_required")
        if not spec["verification_enabled"]:
            _require(bool(spec.get("blocking_reason")), "unavailable_checkpoint_reason_required")
    for name, digest in SOURCE_PINS.items():
        _require(file_digest(root / name) == digest, "previously_verified_input_source_changed:" + name)
    return catalog, plan, registry


def checkpoint_files(model_id, catalog, weights_root, allow_download=False):
    """Return only full-hash verified local paths; reject a gated model first."""
    _require(canonical_digest(catalog) == CATALOG_CANONICAL_SHA256,
        "reviewed_checkpoint_catalog_changed")
    _require(model_id in catalog["models"], "registered_image_encoder_required")
    spec = catalog["models"][model_id]
    _require(spec["verification_enabled"] is True, "checkpoint_access_or_review_pending")
    root = Path(weights_root)
    paths = []
    for item in spec["files"]:
        path = root / item["name"]
        if not path.is_file() and not allow_download:
            raise ValueError("checkpoint_not_cached_use_explicit_download_option")
        acquire(item["url"], path, expected_sha256=item["sha256"], expected_bytes=item["bytes"])
        paths.append(path)
    return paths


def _normalize_densenet_keys(state):
    """The official legacy file uses norm.1 keys, as TorchVision documents."""
    result = {}
    pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.([12])\.(weight|bias|running_mean|running_var)$")
    for key, value in state.items():
        match = pattern.match(key)
        renamed = match.group(1) + match.group(2) + "." + match.group(3) if match else key
        _require(renamed not in result, "duplicate_legacy_densenet_key")
        result[renamed] = value
    return result


def _normalize_vit_keys(state):
    """Apply TorchVision's legacy MLPBlock linear_1/2 migration explicitly."""
    result = {}
    pattern = re.compile(r"^(encoder\.layers\.encoder_layer_\d+\.mlp)\.linear_([12])\.(weight|bias)$")
    for key, value in state.items():
        match = pattern.match(key)
        renamed = (match.group(1) + "." + ("0" if match.group(2) == "1" else "3")
            + "." + match.group(3)) if match else key
        _require(renamed not in result, "duplicate_legacy_vit_key")
        result[renamed] = value
    return result


def load_encoder(model_id, paths):
    """Local paths must first pass checkpoint_files; no automatic downloads."""
    import torch
    from torch import nn
    from torchvision.models import densenet121, vit_b_16

    audit = {"classification_head_used_for_prediction": False}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        if model_id in {"image_densenet121", "image_vit_b16"}:
            model = densenet121(weights=None) if model_id == "image_densenet121" else vit_b_16(weights=None)
            state = torch.load(paths[0], map_location="cpu", weights_only=True)
            _require(isinstance(state, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor)
                and v.device.type == "cpu" and v.layout == torch.strided
                and bool(torch.isfinite(v).all()) for k, v in state.items()),
                "finite_official_tensor_state_required")
            if model_id == "image_densenet121":
                state = _normalize_densenet_keys(state)
            else:
                state = _normalize_vit_keys(state)
            expected = set(model.state_dict())
            missing = expected - set(state)
            _require(set(state) <= expected and (not missing or (model_id == "image_densenet121"
                and all(k.endswith("num_batches_tracked") for k in missing))),
                "exact_baseline_checkpoint_keys_required")
            # Only legacy BatchNorm counters may be absent; they are zero and
            # remain unchanged during eval. No learned tensor is synthesized.
            for key in missing:
                state[key] = torch.zeros_like(model.state_dict()[key])
            model.load_state_dict(state, strict=True)
            audit.update(loaded_pretrained_tensors=len(state) - len(missing),
                initialized_nonpretrained_buffer_keys=sorted(missing),
                all_source_tensors_accounted=True)
            if model_id == "image_densenet121":
                model = nn.Sequential(model.features, nn.ReLU(inplace=False),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
            else:
                model.heads = nn.Identity()
            del state
        elif model_id == "image_convnextv2_tiny":
            from knee.pretrained_convnextv2 import ConvNeXtV2Tiny, load_fcmae_state
            model = ConvNeXtV2Tiny()
            state = torch.load(paths[0], map_location="cpu", weights_only=True)
            audit.update(load_fcmae_state(model, state))
            del state
        elif model_id == "image_skelex":
            from knee.pretrained_skelex import load_skelex_encoder
            model = load_skelex_encoder(*paths)
            audit.update(complete_mae_state_including_decoder_loaded_strictly=True,
                reconstruction_decoder_executed=False, all_patches_retained=True,
                masking_disabled=True, spatial_order_preserved=True,
                all_source_tensors_accounted=True)
        else:
            raise ValueError("checkpoint_loader_not_adopted")
    return model.cpu().float().eval().requires_grad_(False), audit


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def synthetic_inputs():
    import numpy as np
    import torch
    from knee.model_input_adapters import normalize_image
    from knee.pretrained_convnextv2 import normalize_sealed_image
    first = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(1, 224, 224)
    second = np.ascontiguousarray(1 - first)
    originals = [first, second]
    batches = []
    for array in originals:
        array.setflags(write=False)
        before = hashlib.sha256(array.tobytes()).hexdigest()
        normalized = normalize_image(array, "image_densenet121")
        _require(torch.equal(normalized, normalize_image(array, "image_vit_b16"))
            and torch.equal(normalized, normalize_sealed_image(array)),
            "checkpoint_normalization_disagrees_with_verified_inputs")
        _require(hashlib.sha256(array.tobytes()).hexdigest() == before
            and not array.flags.writeable, "synthetic_geometry_or_source_changed")
        batches.append(normalized)
    return torch.stack(batches)


def probe_encoder(model, batch, feature_dimension):
    import torch
    before = state_digest(model)
    before_input = batch.clone()
    before_rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        first = model(batch)
        second = model(batch)
    _require(first.shape == (len(batch), feature_dimension)
        and first.dtype == torch.float32 and bool(torch.isfinite(first).all()),
        "finite_expected_encoder_features_required")
    _require(torch.equal(first, second), "repeated_cpu_features_differ")
    _require(state_digest(model) == before and torch.equal(before_input, batch)
        and torch.equal(before_rng, torch.get_rng_state())
        and not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters()),
        "verification_mutated_state_or_created_gradient")
    return {"feature_shape": list(first.shape), "encoder_state_sha256": before,
        "synthetic_feature_sha256": hashlib.sha256(first.cpu().numpy().tobytes()).hexdigest(),
        "repeated_features_exact": True, "parameters_and_buffers_unchanged": True,
        "input_and_rng_unchanged": True, "gradients_created": False,
        "feature_hash_scope": "same_runtime_only_not_cross_platform_acceptance_gate"}


def run(repository_root, weights_root, output_dir, *, allow_download=False, execution_environment="local_cpu"):
    import torch
    import torchvision
    import numpy as np
    _require(execution_environment in {"local_cpu", "google_colab_cpu"}, "explicit_execution_environment_required")
    if execution_environment == "google_colab_cpu":
        import os
        _require(bool(os.environ.get("COLAB_RELEASE_TAG")), "actual_colab_runtime_required")
    _require(torch.__version__ == "2.6.0+cpu" and torchvision.__version__ == "0.21.0+cpu"
        and np.__version__ == "1.26.4" and torch.version.cuda is None
        and not torch.cuda.is_available(), "verified_cpu_runtime_required")
    catalog, plan, registry = load_catalog(repository_root)
    root, output, weights = Path(repository_root).resolve(), Path(output_dir).resolve(), Path(weights_root).resolve()
    _require(not output.exists() and output != weights and root not in output.parents,
        "new_private_output_directory_outside_repository_required")
    _require(weights != root and root not in weights.parents, "checkpoint_cache_must_remain_outside_repository")
    started = time.monotonic()
    prior_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    records = {}
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        batch = synthetic_inputs()
        for model_id, spec in catalog["models"].items():
            if not spec["verification_enabled"]:
                records[model_id] = {"status": "pending_authorized_access",
                    "reason": spec["blocking_reason"], "checkpoint_loaded": False}
                print(f"{model_id}: acceso pendiente; no se descargaron pesos.", flush=True)
                continue
            print(f"Verificando pesos e inferencia CPU: {model_id}", flush=True)
            paths = checkpoint_files(model_id, catalog, weights, allow_download)
            model, adoption = load_encoder(model_id, paths)
            probe = probe_encoder(model, batch, spec["feature_dimension"])
            records[model_id] = {"status": "passed_synthetic_cpu_checkpoint_verification",
                "checkpoint_loaded": True, "files": [{"name": p.name, "bytes": p.stat().st_size,
                    "sha256": file_digest(p)} for p in paths], "adoption": adoption, **probe,
                "training_smoke_test_passed": False, "gpu_resource_profile_recorded": False,
                "all_training_eligibility_gates_passed": False}
            del model
            gc.collect()
    finally:
        torch.set_num_threads(prior_threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    # Old source pins and all downloaded bytes are checked again before closure.
    load_catalog(root)
    for model_id, record in records.items():
        if record["checkpoint_loaded"]:
            checkpoint_files(model_id, catalog, weights, False)
    result = {"version": VERSION, "status": "accessible_checkpoints_verified_gated_candidate_pending",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_environment": execution_environment,
        "colab_runtime_executed": execution_environment == "google_colab_cpu",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            **{name: importlib.metadata.version(name) for name in
                ("transformers", "tokenizers", "regex", "safetensors", "huggingface-hub")}},
        "model_records": records, "registered_image_models": len(records),
        "pretrained_models_loaded": sum(r["checkpoint_loaded"] for r in records.values()),
        "training_configuration": validate_plan(plan, registry),
        "catalog_canonical_sha256": canonical_digest(catalog),
        "source_files_sha256": {name: file_digest(root / name) for name in (
            "src/knee/pretrained_verification.py", "src/knee/checkpoint_transport.py",
            "src/knee/pretrained_convnextv2.py", "src/knee/pretrained_skelex.py",
            "src/knee/training_plan.py", "configs/pretrained_weights_v1.json",
            "configs/training_plan_v1.json", "configs/model_registry.json")},
        "synthetic_inputs_only": True, "participant_data_opened": False,
        "training_executed": False, "optimizer_created": False,
        "reserved_test_predictive_evaluation_opened": False,
        "originals_modified": False, "partitions_modified": False,
        "sealed_input_code_unchanged": True, "new_model_selection_performed": False,
        "predictive_or_clinical_validation_claimed": False,
        "elapsed_seconds": round(time.monotonic() - started, 3)}
    output.mkdir(parents=True, exist_ok=False)
    target = output / "checkpoint_verification.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _require(json.loads(target.read_text(encoding="utf-8")) == result, "verification_record_readback_failed")
    print(json.dumps({key: result[key] for key in ("version", "status", "pretrained_models_loaded",
        "registered_image_models", "training_executed", "participant_data_opened", "colab_runtime_executed")}, indent=2))
    return result
