"""Atomic, bounded epoch recovery for the four authorized gradual image models.

The existing stage-2 engine remains the sole optimization loop. A committed
generation contains current and best complete model tensors, AdamW, AMP, all
random generators, the phase and monitor. Only the latest two committed
generations are retained; an unfinished epoch is never adopted.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

import torch

from knee.colab_data import checked_path, require, sha_file
from knee.training_authority import digest
from knee.training_epoch_resume import _cpu, _rng, _restore_rng, _runtime
from knee.training_preparation import read_json

VERSION = "gradual_image_epoch_resume_v1.0"
SCOPE = "full_model_optimizer_completed_epochs_same_runtime"
MODELS = ("image_densenet121", "image_vit_b16", "image_skelex", "image_mambaout_tiny")
MAX_STATE_BYTES = 16 * 1024**3
MAX_GENERATIONS = 128
RETAIN_COMMITTED_GENERATIONS = 2
_GENERATION = re.compile(r"epoch_([0-9]{3})_[0-9a-f]{32}")


def _sync(stream):
    """Flush always; tolerate only unsupported sync calls on a Drive mount."""
    stream.flush()
    try:
        os.fsync(stream.fileno())
    except OSError as exc:
        allowed = {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}
        if exc.errno not in allowed:
            raise


def _state_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(name, str) and isinstance(value, torch.Tensor), "plain_named_model_tensors_required")
        value = value.detach().cpu().contiguous()
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "finite_gradual_model_state_required")
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def _exact_model_state(state, expected):
    require(isinstance(state, dict) and set(state) == set(expected), "exact_full_model_state_required")
    require(all(isinstance(value, torch.Tensor) and value.shape == expected[name].shape
        and value.dtype == expected[name].dtype
        and (not value.is_floating_point() or bool(torch.isfinite(value).all()))
        for name, value in state.items()), "exact_finite_full_model_tensors_required")


def _check_plain_state(value):
    """Validate restored moments without allocating a second CPU optimizer."""
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu" and (not value.is_floating_point()
            or bool(torch.isfinite(value).all())), "finite_cpu_gradual_optimizer_state_required")
    elif isinstance(value, dict):
        for item in value.values():
            _check_plain_state(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_plain_state(item)
    else:
        require(value is None or type(value) in {str, bool, int, float}, "plain_gradual_optimizer_state_required")
        require(type(value) is not float or math.isfinite(value), "finite_gradual_optimizer_scalar_required")


class GradualEpochCheckpoint:
    version = VERSION
    scope = SCOPE

    def __init__(self, directory, request):
        from knee import methodology_release as release
        self.directory = checked_path(directory)
        self.request = deepcopy(request)
        require(request.get("model_id") in MODELS and request.get("image_encoder_id") == request["model_id"]
            and request.get("operation") == "development_fold" and request.get("stage") == "stage_2"
            and request.get("clinical_scenario") == "clinical_main" and request.get("seed") == 2026
            and type(request.get("validation_fold")) is int and request["validation_fold"] in range(5)
            and isinstance(request.get("selection_evidence_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", request["selection_evidence_sha256"]),
            "gradual_epoch_resume_only_four_selected_stage2_development_models")
        key = f'clinical_main__{request["model_id"]}__fold_{request["validation_fold"]}'
        repository = Path(__file__).resolve().parents[2]
        require(self.directory.name == key and self.directory.parent.name == "epoch_checkpoints"
            and not self.directory.is_relative_to(repository), "private_gradual_job_epoch_directory_required")
        sources = ("src/knee/gradual_epoch_resume.py", "src/knee/training_epoch_resume.py",
            "src/knee/aligned_training/engine.py", "src/knee/aligned_training/runner.py")
        self.identity = {"version": VERSION, "request_sha256": digest(request),
            "plan_sha256": release.PINS[release.PLAN_PATH],
            "source_sha256": {name: sha_file(repository / name) for name in sources}}
        self.resumed_from_epoch = 0
        self.initial_state_digest = None
        self.state_size_limit = None

    def validate(self, permit, *, smoke, stage):
        require(not smoke and stage == "stage_2" and permit.request_sha256 == self.identity["request_sha256"]
            and permit.plan_sha256 == self.identity["plan_sha256"] and not permit.smoke
            and permit.model_id == self.request["model_id"] and permit.seed == 2026,
            "gradual_checkpoint_must_match_selected_full_stage2_authority")

    def _committed(self):
        if not self.directory.exists():
            return []
        require(checked_path(self.directory) == self.directory and self.directory.is_dir(),
            "plain_gradual_epoch_directory_required")
        completed = []
        for count, path in enumerate(self.directory.iterdir(), 1):
            require(count <= MAX_GENERATIONS and checked_path(path).is_dir(), "bounded_plain_gradual_generations_required")
            if re.fullmatch(r"\.(writing|retiring)_[0-9a-f]{32}", path.name):
                continue
            match = _GENERATION.fullmatch(path.name)
            require(match is not None, "registered_gradual_generation_name_required")
            manifest_path = checked_path(path / "manifest.json")
            require(manifest_path.is_file() and 0 < manifest_path.stat().st_size < 65536,
                "committed_gradual_manifest_required")
            manifest = read_json(manifest_path)
            require(set(manifest) == {"identity", "epoch", "state_sha256", "state_bytes"}
                and manifest["identity"] == self.identity and type(manifest["epoch"]) is int
                and 1 <= manifest["epoch"] <= 60 and int(match[1]) == manifest["epoch"]
                and type(manifest["state_bytes"]) is int and 0 < manifest["state_bytes"] <= MAX_STATE_BYTES
                and isinstance(manifest["state_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", manifest["state_sha256"])
                and {item.name for item in path.iterdir()} == {"manifest.json", "state.pt"}
                and all(checked_path(path / name).is_file() for name in ("manifest.json", "state.pt")),
                "exact_gradual_identity_and_inventory_required")
            completed.append((manifest["epoch"], path, manifest))
        require(len({epoch for epoch, _, _ in completed}) == len(completed), "unique_committed_gradual_epoch_required")
        return sorted(completed, key=lambda item: item[0])

    def _latest(self):
        values = self._committed()
        return values[-1] if values else None

    def _checked_state_path(self, path, manifest):
        state_path = checked_path(path / "state.pt")
        require(state_path.is_relative_to(self.directory) and state_path.is_file()
            and state_path.stat().st_size == manifest["state_bytes"]
            and manifest["state_bytes"] <= (self.state_size_limit or MAX_STATE_BYTES)
            and sha_file(state_path) == manifest["state_sha256"], "committed_gradual_state_hash_mismatch")
        return state_path

    def restore(self, model, optimizer, scaler, loader, monitor, device, binding):
        from knee.aligned_training.engine import epoch_phase
        from knee.aligned_training.models import _structural_freeze
        require(loader.num_workers == 0 and not loader.drop_last and loader.generator is not None
            and loader.sampler.generator is loader.generator, "complete_seeded_gradual_epoch_loader_required")
        expected = model.state_dict()
        self.initial_state_digest = _state_digest(expected)
        tensor_bytes = sum(value.numel() * value.element_size() for value in expected.values())
        self.state_size_limit = min(MAX_STATE_BYTES, tensor_bytes * 6 + 64 * 1024**2)
        require({id(p) for group in optimizer.param_groups for p in group["params"]}
            == {id(p) for name, p in model.named_parameters()
                if not (name.startswith("encoder.") and _structural_freeze(
                    self.request["model_id"], name[len("encoder."):]))},
            "all_gradual_parameters_must_be_registered_in_optimizer")
        latest = self._latest()
        if latest is None:
            return None
        epoch, path, manifest = latest
        state = torch.load(self._checked_state_path(path, manifest), map_location="cpu", weights_only=True)
        required = {"identity", "epoch", "runtime", "binding", "initial_state_sha256", "current_model", "best_model",
            "optimizer", "scaler", "rng", "monitor", "history", "total_updates", "best_predictions", "stopped",
            "phase", "trainable_parameters", "scheduler_epoch"}
        require(isinstance(state, dict) and set(state) == required
            and state["identity"] == self.identity and state["epoch"] == epoch
            and state["runtime"] == _runtime(device) and state["binding"] == binding
            and state["initial_state_sha256"] == self.initial_state_digest
            and state["scheduler_epoch"] == epoch
            and state["phase"] == epoch_phase(self.request["model_id"], "stage_2", epoch)
            and [item["epoch"] for item in state["history"]] == list(range(1, epoch + 1))
            and all(item["phase"] == epoch_phase(self.request["model_id"], "stage_2", item["epoch"])
                for item in state["history"])
            and state["total_updates"] == sum(item["optimizer_updates"] for item in state["history"])
            and set(state["monitor"]) == set(asdict(monitor))
            and state["monitor"]["patience"] == monitor.patience
            and state["monitor"]["monitor_starts_after_epoch"] == monitor.monitor_starts_after_epoch == 10
            and type(state["stopped"]) is bool,
            "same_gradual_recipe_runtime_initial_weights_and_epoch_state_required")
        require((epoch <= 10 and state["best_model"] is None and state["best_predictions"] is None
                 and state["monitor"]["best_epoch"] is None and state["monitor"]["best_score"] is None
                 and not state["stopped"])
            or (epoch > 10 and isinstance(state["best_model"], dict) and isinstance(state["best_predictions"], dict)
                 and type(state["monitor"]["best_epoch"]) is int and 11 <= state["monitor"]["best_epoch"] <= epoch),
            "stage2_best_selection_starts_only_after_epoch10")
        named_parameters = dict(model.named_parameters())
        require(isinstance(state["trainable_parameters"], list) and state["trainable_parameters"]
            and state["trainable_parameters"] == sorted(set(state["trainable_parameters"]))
            and set(state["trainable_parameters"]).issubset(named_parameters), "registered_gradual_trainable_names_required")
        _exact_model_state(state["current_model"], expected)
        if state["best_model"] is not None:
            _exact_model_state(state["best_model"], expected)
        model.load_state_dict(state["current_model"], strict=True)
        _check_plain_state(state["optimizer"])
        _check_plain_state(state["scaler"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        for name, value in state["monitor"].items():
            setattr(monitor, name, value)
        loader.dataset.set_epoch(epoch)
        _restore_rng(state["rng"], loader)
        self.resumed_from_epoch = epoch
        next_action = "finalizando sin nuevas actualizaciones" if state["stopped"] or epoch == 60 else f"siguiente epoca {epoch + 1}"
        print(f'Reanudando {self.request["model_id"]}, pliegue {self.request["validation_fold"]}: '
            f'epoca {epoch} completa recuperada; {next_action}.', file=sys.stderr, flush=True)
        return {"epoch": epoch, "history": state["history"], "total_updates": state["total_updates"],
            "best_state": state["best_model"], "best_predictions": state["best_predictions"],
            "stopped": state["stopped"], "phase": state["phase"], "trainable_parameters": state["trainable_parameters"]}

    def _retire_old_generations(self):
        """Remove only named, authenticated files in older committed generations."""
        completed = self._committed()
        for _, directory, manifest in completed[:-RETAIN_COMMITTED_GENERATIONS]:
            require(checked_path(directory).parent == self.directory and _GENERATION.fullmatch(directory.name),
                "retirement_must_stay_in_exact_gradual_job_directory")
            state_path = checked_path(directory / "state.pt")
            require(state_path.parent == directory and state_path.is_file()
                and state_path.stat().st_size == manifest["state_bytes"],
                "retired_state_must_match_authenticated_generation_inventory")
            # A disconnect during cleanup leaves an ignored retirement directory,
            # never a half-deleted generation that could be chosen for recovery.
            retired = self.directory / (".retiring_" + uuid.uuid4().hex)
            directory.rename(retired)
            require(checked_path(retired).parent == self.directory
                and {item.name for item in retired.iterdir()} == {"state.pt", "manifest.json"},
                "exact_retired_generation_inventory_required")
            for name in ("state.pt", "manifest.json"):
                item = checked_path(retired / name)
                require(item.parent == retired and item.is_file(), "plain_retired_generation_file_required")
                item.unlink()
            retired.rmdir()

    def save(self, model, optimizer, scaler, loader, monitor, device, binding, *, epoch, history,
             total_updates, best_state, best_predictions, stopped):
        from knee.aligned_training.engine import epoch_phase
        require(self.initial_state_digest is not None and self.state_size_limit is not None
            and checked_path(self.directory) == self.directory, "initialized_exact_gradual_checkpoint_required")
        latest = self._latest()
        require(type(epoch) is int and 1 <= epoch <= 60
            and epoch == (latest[0] + 1 if latest else 1), "next_uncommitted_gradual_epoch_required")
        phase = epoch_phase(self.request["model_id"], "stage_2", epoch)
        require((epoch <= 10 and best_state is None and best_predictions is None and monitor.best_epoch is None)
            or (epoch > 10 and best_state is not None and best_predictions is not None
                and monitor.best_epoch is not None and 11 <= monitor.best_epoch <= epoch),
            "stage2_best_selection_starts_only_after_epoch10")
        if best_state is not None:
            _exact_model_state(best_state, model.state_dict())
            require(all(value.device.type == "cpu" and value.is_contiguous() for value in best_state.values()),
                "existing_best_cpu_snapshot_required")
        state = {"identity": self.identity, "epoch": epoch, "runtime": _runtime(device), "binding": binding,
            "initial_state_sha256": self.initial_state_digest, "current_model": _cpu(model.state_dict()),
            "best_model": best_state, "optimizer": _cpu(optimizer.state_dict()), "scaler": _cpu(scaler.state_dict()),
            "rng": _cpu(_rng(loader)), "monitor": asdict(monitor), "history": deepcopy(history),
            "total_updates": total_updates, "best_predictions": deepcopy(best_predictions), "stopped": bool(stopped),
            "phase": phase, "trainable_parameters": sorted(name for name, p in model.named_parameters() if p.requires_grad),
            "scheduler_epoch": epoch}
        self.directory.mkdir(parents=True, exist_ok=True)
        stage = self.directory / (".writing_" + uuid.uuid4().hex)
        stage.mkdir()
        state_path = stage / "state.pt"
        with state_path.open("xb") as stream:
            torch.save(state, stream)
            _sync(stream)
        size, checksum = state_path.stat().st_size, sha_file(state_path)
        require(0 < size <= self.state_size_limit, "bounded_complete_gradual_state_required")
        del state
        manifest = {"identity": self.identity, "epoch": epoch, "state_sha256": checksum, "state_bytes": size}
        with (stage / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, allow_nan=False)
            _sync(stream)
        target = self.directory / (f"epoch_{epoch:03d}_" + uuid.uuid4().hex)
        stage.rename(target)
        require(sha_file(target / "state.pt") == checksum and read_json(target / "manifest.json") == manifest,
            "durable_gradual_epoch_commit_readback_failed")
        self._retire_old_generations()
        print(f'Epoca {epoch}/60 guardada: {self.request["model_id"]}, pliegue {self.request["validation_fold"]}, fase {phase}.',
            file=sys.stderr, flush=True)
