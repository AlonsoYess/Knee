"""Atomic epoch-boundary recovery for the five frozen-image development jobs.

Only the small prediction head and optimizer are stored. The frozen encoder is
reloaded from its registered external checkpoint and checked by tensor digest.
No smoke run, partial epoch, new recipe, or participant split is adopted here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import uuid

import numpy as np
import torch

from knee.colab_data import checked_path, require, sha_file
from knee.training_authority import digest
from knee.training_preparation import read_json

VERSION = "frozen_image_epoch_resume_v1.0"
MODELS = ("image_vit_b16", "image_convnextv2_tiny", "image_skelex", "image_biomedclip", "image_mambaout_tiny")
MAX_STATE_BYTES = 64 * 1024**2
MAX_GENERATIONS = 128
_HEAD = "task_head."


def _cpu(value):
    if isinstance(value, torch.Tensor):
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "finite_epoch_tensor_state_required")
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    require(value is None or type(value) in {str, bool, int, float}, "plain_epoch_state_required")
    require(type(value) is not float or math.isfinite(value), "finite_epoch_scalar_state_required")
    return value


def _encoder_digest(model):
    result = hashlib.sha256()
    count = 0
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(_HEAD):
            continue
        require(name.startswith("encoder."), "only_encoder_and_task_head_may_resume")
        value = value.detach().cpu().contiguous()
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "finite_frozen_encoder_required")
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        result.update(value.numpy().tobytes())
        count += 1
    require(count > 0, "frozen_encoder_state_required")
    return result.hexdigest()


def _head(state):
    values = {name: _cpu(value) for name, value in state.items() if name.startswith(_HEAD)}
    require(values and sum(value.numel() for value in values.values()) <= 100000,
        "bounded_frozen_encoder_linear_head_required")
    return values


def _runtime(device):
    value = {"python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "device": str(device),
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic}
    if device.type == "cuda":
        value.update(device_name=torch.cuda.get_device_name(device),
            device_capability=list(torch.cuda.get_device_capability(device)))
    return value


def _rng(loader):
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy[0], torch.tensor(numpy[1].astype(np.int64)),
            int(numpy[2]), int(numpy[3]), float(numpy[4])],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader.generator.get_state()}


def _restore_rng(value, loader):
    random.setstate(value["python"])
    numpy = value["numpy"]
    np.random.set_state((numpy[0], numpy[1].numpy().astype(np.uint32), numpy[2], numpy[3], numpy[4]))
    torch.set_rng_state(value["torch_cpu"])
    if value["torch_cuda"]:
        require(torch.cuda.is_available() and len(value["torch_cuda"]) == torch.cuda.device_count(),
            "same_cuda_rng_devices_required")
        torch.cuda.set_rng_state_all(value["torch_cuda"])
    loader.generator.set_state(value["loader"])


class EpochCheckpoint:
    def __init__(self, directory, request):
        from knee import methodology_release as release
        self.directory = checked_path(directory)
        self.request = deepcopy(request)
        require(request.get("model_id") in MODELS and request.get("image_encoder_id") == request["model_id"]
            and request.get("operation") == "development_fold" and request.get("stage") == "stage_1"
            and request.get("clinical_scenario") == "clinical_main" and request.get("seed") == 2026
            and type(request.get("validation_fold")) is int and request["validation_fold"] in range(5),
            "epoch_resume_only_five_frozen_full_development_jobs")
        key = f'clinical_main__{request["model_id"]}__fold_{request["validation_fold"]}'
        repository = Path(__file__).resolve().parents[2]
        require(self.directory.name == key and self.directory.parent.name == "epoch_checkpoints"
            and not self.directory.is_relative_to(repository), "private_job_specific_epoch_directory_required")
        sources = ("src/knee/training_epoch_resume.py", "src/knee/aligned_training/engine.py",
            "src/knee/aligned_training/runner.py")
        self.identity = {"version": VERSION, "request_sha256": digest(request),
            "plan_sha256": release.PINS[release.PLAN_PATH],
            "source_sha256": {name: sha_file(repository / name) for name in sources}}
        self.resumed_from_epoch = 0
        self.frozen_digest = None

    def validate(self, permit, *, smoke, stage):
        require(not smoke and stage == "stage_1" and permit.request_sha256 == self.identity["request_sha256"]
            and permit.plan_sha256 == self.identity["plan_sha256"] and not permit.smoke
            and permit.model_id == self.request["model_id"] and permit.seed == 2026,
            "epoch_checkpoint_must_match_exact_full_fold_authority")

    def _latest(self):
        if not self.directory.exists():
            return None
        require(self.directory.is_dir(), "epoch_checkpoint_directory_required")
        completed = []
        for count, path in enumerate(self.directory.iterdir(), 1):
            require(count <= MAX_GENERATIONS and checked_path(path).is_dir(), "bounded_plain_epoch_generations_required")
            if path.name.startswith(".writing_"):
                continue
            require(re.fullmatch(r"epoch_[0-9]{3}_[0-9a-f]{32}", path.name), "registered_epoch_generation_name_required")
            manifest_path = path / "manifest.json"
            require(manifest_path.is_file() and 0 < manifest_path.stat().st_size < 65536,
                "committed_epoch_manifest_required")
            manifest = read_json(manifest_path)
            require(manifest.get("identity") == self.identity and type(manifest.get("epoch")) is int
                and 1 <= manifest["epoch"] <= 60 and path.name.startswith(f'epoch_{manifest["epoch"]:03d}_')
                and set(manifest) == {"identity", "epoch", "state_sha256", "state_bytes"}
                and {item.name for item in path.iterdir()} == {"manifest.json", "state.pt"},
                "exact_epoch_identity_and_inventory_required")
            completed.append((manifest["epoch"], path, manifest))
        require(len({epoch for epoch, _, _ in completed}) == len(completed), "unique_committed_epoch_required")
        return max(completed, key=lambda value: value[0]) if completed else None

    def restore(self, model, optimizer, scaler, loader, monitor, device, binding):
        require(all(name.startswith(_HEAD) == parameter.requires_grad for name, parameter in model.named_parameters()),
            "only_prediction_head_may_be_trainable_for_resume")
        require(loader.num_workers == 0 and not loader.drop_last and loader.generator is not None
            and loader.sampler.generator is loader.generator, "single_process_complete_shuffled_epoch_loader_required")
        self.frozen_digest = _encoder_digest(model)
        latest = self._latest()
        if latest is None:
            return None
        epoch, path, manifest = latest
        state_path = checked_path(path / "state.pt")
        require(0 < manifest["state_bytes"] <= MAX_STATE_BYTES and state_path.is_file()
            and state_path.stat().st_size == manifest["state_bytes"] and sha_file(state_path) == manifest["state_sha256"],
            "committed_epoch_state_hash_mismatch")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        require(set(state) == {"identity", "epoch", "runtime", "binding", "encoder_sha256", "head", "best_head",
            "optimizer", "scaler", "rng", "monitor", "history", "total_updates", "best_predictions", "stopped"}
            and state["identity"] == self.identity and state["epoch"] == epoch
            and state["runtime"] == _runtime(device) and state["binding"] == binding
            and state["encoder_sha256"] == self.frozen_digest
            and [item["epoch"] for item in state["history"]] == list(range(1, epoch + 1))
            and all(item["phase"] == "head_only" for item in state["history"])
            and state["total_updates"] == sum(item["optimizer_updates"] for item in state["history"])
            and set(state["monitor"]) == set(asdict(monitor))
            and state["monitor"]["patience"] == monitor.patience
            and state["monitor"]["monitor_starts_after_epoch"] == monitor.monitor_starts_after_epoch,
            "same_recipe_runtime_frozen_encoder_and_epoch_state_required")
        current = model.state_dict()
        expected = _head(current)
        for key in ("head", "best_head"):
            require(set(state[key]) == set(expected) and all(isinstance(value, torch.Tensor)
                and value.shape == expected[name].shape and value.dtype == expected[name].dtype
                and bool(torch.isfinite(value).all()) for name, value in state[key].items()),
                "exact_finite_saved_prediction_head_required")
        model.load_state_dict({**current, **state["head"]}, strict=True)
        optimizer.load_state_dict(_cpu(state["optimizer"]))
        scaler.load_state_dict(_cpu(state["scaler"]))
        for name, value in state["monitor"].items():
            require(name in asdict(monitor), "registered_monitor_fields_required")
            setattr(monitor, name, value)
        best_state = {name: _cpu(value) for name, value in model.state_dict().items()}
        best_state.update(state["best_head"])
        loader.dataset.set_epoch(epoch)
        _restore_rng(state["rng"], loader)
        self.resumed_from_epoch = epoch
        return {"epoch": epoch, "history": state["history"], "total_updates": state["total_updates"],
            "best_state": best_state, "best_predictions": state["best_predictions"], "stopped": state["stopped"]}

    def save(self, model, optimizer, scaler, loader, monitor, device, binding, *, epoch, history,
             total_updates, best_state, best_predictions, stopped):
        require(self.frozen_digest is not None and _encoder_digest(model) == self.frozen_digest,
            "frozen_encoder_changed_before_epoch_commit")
        require(checked_path(self.directory) == self.directory, "epoch_checkpoint_directory_changed")
        state = {"identity": self.identity, "epoch": epoch, "runtime": _runtime(device), "binding": binding,
            "encoder_sha256": self.frozen_digest, "head": _head(model.state_dict()), "best_head": _head(best_state),
            "optimizer": _cpu(optimizer.state_dict()), "scaler": _cpu(scaler.state_dict()), "rng": _cpu(_rng(loader)),
            "monitor": asdict(monitor), "history": deepcopy(history), "total_updates": total_updates,
            "best_predictions": deepcopy(best_predictions), "stopped": bool(stopped)}
        self.directory.mkdir(parents=True, exist_ok=True)
        stage = self.directory / (".writing_" + uuid.uuid4().hex)
        stage.mkdir()
        state_path = stage / "state.pt"
        with state_path.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        size, checksum = state_path.stat().st_size, sha_file(state_path)
        require(0 < size <= MAX_STATE_BYTES, "bounded_epoch_state_required")
        restored = torch.load(state_path, map_location="cpu", weights_only=True)
        require(restored["identity"] == self.identity and restored["epoch"] == epoch
            and restored["encoder_sha256"] == self.frozen_digest, "epoch_stage_readback_failed")
        manifest = {"identity": self.identity, "epoch": epoch, "state_sha256": checksum, "state_bytes": size}
        with (stage / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        require(sha_file(state_path) == checksum, "epoch_stage_changed_before_commit")
        target = self.directory / (f"epoch_{epoch:03d}_" + uuid.uuid4().hex)
        stage.rename(target)
        require(sha_file(target / "state.pt") == checksum
            and json.loads((target / "manifest.json").read_text(encoding="utf-8")) == manifest,
            "durable_epoch_commit_readback_failed")
