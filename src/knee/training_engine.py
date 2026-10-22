"""Prepared neural fold engine; importing or testing this module fits nothing.

The public runner must authenticate development-only execution authority, fold
identity, checkpoint eligibility, and stage eligibility BEFORE calling
``train_neural_fold``. This module never opens a dataset/package or authorizes
itself. Notebook 34 does not call the training function. Checkpoints contain
model tensors only: optimizer/scaler continuation is deliberately unsupported.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from knee.training_plan import _digest, _require
from knee.final_model_release import PINS

VERSION = "prepared_neural_fold_engine_v1.0"
METRICS = ("average_precision_score", "roc_auc", "brier")
AMP_NUMERICAL_REPAIR_VERSION = "same_effective_group_amp_replay_v1.0"
AMP_MAX_BACKOFFS = 16


def validate_recipe(plan):
    """Authenticate the complete immutable recipe without granting authority."""
    _require(isinstance(plan, dict) and _digest(plan) == PINS["configs/training_plan_v1_1.json"],
             "approved_training_recipe_required")
    return True


def learning_rate_factor(epoch, *, maximum_epochs=60, warmup_epochs=5, minimum_fraction=0.01):
    """One-based epoch schedule, with the approved fixed 60-epoch horizon."""
    _require(type(epoch) is int and type(maximum_epochs) is int and type(warmup_epochs) is int
             and 0 < warmup_epochs < maximum_epochs and 1 <= epoch <= maximum_epochs,
             "valid_one_based_schedule_epoch_required")
    _require(type(minimum_fraction) in {int, float} and 0 < minimum_fraction <= 1,
             "valid_minimum_learning_rate_fraction_required")
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (maximum_epochs - warmup_epochs)
    return minimum_fraction + (1 - minimum_fraction) * (1 + math.cos(math.pi * progress)) / 2


def epoch_phase(model_id, stage, epoch):
    _require(stage in {"stage_1", "stage_2", "fusion"} and type(epoch) is int and 1 <= epoch <= 60,
             "registered_stage_and_epoch_required")
    _require(isinstance(model_id, str), "registered_neural_model_required")
    if model_id == "clinical_mlp":
        _require(stage == "stage_1", "clinical_mlp_has_no_image_unfreezing_stage")
        return "full"
    _require(model_id.startswith("image_") or model_id in
             {"fusion_intermediate_concat", "fusion_intermediate_gated_film"},
             "registered_neural_model_required")
    _require((stage == "fusion") == model_id.startswith("fusion_"), "fusion_stage_required_for_fusion_models")
    if stage == "stage_1" and model_id.startswith("image_"):
        return "head_only"
    if epoch <= 5:
        return "head_only"
    if epoch <= 10:
        return "last_block"
    return "full"


@dataclass
class EarlyStopping:
    """Strict AP improvement preserves the earliest epoch on an exact tie."""
    patience: int = 8
    monitor_starts_after_epoch: int = 0
    best_score: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0

    def update(self, epoch, score):
        _require(type(epoch) is int and epoch > 0 and type(score) in {int, float}
                 and math.isfinite(score) and 0 <= score <= 1,
                 "finite_validation_ap_and_epoch_required")
        if epoch <= self.monitor_starts_after_epoch:
            return False, False
        improved = self.best_score is None or score > self.best_score
        if improved:
            self.best_score, self.best_epoch, self.bad_epochs = float(score), epoch, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def binary_metrics(targets, logits):
    """Compute sklearn AP (not trapezoidal PR-AUC) and both-class metrics."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y, z = np.asarray(targets), np.asarray(logits, dtype=np.float64)
    _require(y.ndim == 1 and z.shape == y.shape and y.size > 0
             and np.isfinite(y).all() and np.isfinite(z).all()
             and set(np.unique(y).tolist()) == {0, 1},
             "finite_both_class_validation_targets_and_logits_required")
    # Stable sigmoid for large finite logits; probabilities remain unrounded.
    p = np.empty_like(z)
    positive = z >= 0
    p[positive] = 1 / (1 + np.exp(-z[positive]))
    negative_exp = np.exp(z[~positive])
    p[~positive] = negative_exp / (1 + negative_exp)
    result = {"average_precision_score": float(average_precision_score(y, p)),
              "roc_auc": float(roc_auc_score(y, p)),
              "brier": float(brier_score_loss(y, p))}
    _require(all(math.isfinite(value) and 0 <= value <= 1 for value in result.values()),
             "finite_validation_metrics_required")
    return result


def aggregate_fold_metrics(records):
    """Unweighted five-fold means; pooled predictions cannot enter this API."""
    _require(isinstance(records, list) and len(records) == 5,
             "exact_five_fold_metric_rows_required")
    folds = []
    for record in records:
        _require(isinstance(record, dict) and set(record) == {"validation_fold", *METRICS}
                 and type(record["validation_fold"]) is int and record["validation_fold"] in range(5),
                 "exact_five_fold_metric_rows_required")
        folds.append(record["validation_fold"])
        _require(all(type(record[key]) in {int, float} and math.isfinite(record[key])
                     and 0 <= record[key] <= 1 for key in METRICS), "finite_fold_metrics_required")
    _require(set(folds) == set(range(5)), "complete_unique_five_fold_metrics_required")
    return {key: math.fsum(record[key] for record in records) / 5 for key in METRICS}


def accumulation_denominators(micro_batch_sizes, effective_batch_size=16):
    """Pure accounting helper: each gradient sum is divided by actual knees."""
    _require(type(effective_batch_size) is int and effective_batch_size > 0,
             "positive_effective_batch_size_required")
    groups, pending, total = [], [], 0
    for size in micro_batch_sizes:
        _require(type(size) is int and 0 < size <= effective_batch_size,
                 "valid_micro_batch_size_required")
        _require(total + size <= effective_batch_size, "micro_batches_must_not_cross_effective_group")
        pending.append(size)
        total += size
        if total == effective_batch_size:
            groups.append((tuple(pending), total))
            pending, total = [], 0
    if pending:
        groups.append((tuple(pending), total))
    return groups


def _target_count(batch):
    _require(isinstance(batch, dict) and "target" in batch, "aligned_batch_dictionary_required")
    target = batch["target"]
    _require(isinstance(target, torch.Tensor) and target.ndim == 1 and target.numel() > 0
             and torch.isfinite(target).all().item()
             and ((target == 0) | (target == 1)).all().item(), "finite_binary_batch_targets_required")
    return target.numel()


def _batch_groups(loader, effective_batch_size):
    pending, total = [], 0
    for batch in loader:
        count = _target_count(batch)
        _require(count <= effective_batch_size and total + count <= effective_batch_size,
                 "micro_batches_must_not_cross_effective_group")
        pending.append(batch)
        total += count
        if total == effective_batch_size:
            yield pending, total
            pending, total = [], 0
    if pending:
        yield pending, total


def _forward(model, batch, device):
    target = batch["target"].to(device=device, dtype=torch.float32)
    kwargs = {}
    model_id = model.model_id
    needed = (("clinical", "clinical"),) if model_id == "clinical_mlp" else (("image", "images"),)
    if model_id.startswith("fusion_"):
        needed += (("clinical", "clinical"),)
    for key, parameter in needed:
        _require(key in batch, "required_registered_batch_modality_missing")
        if key in batch:
            value = batch[key]
            _require(isinstance(value, torch.Tensor) and value.shape[0] == target.numel()
                     and torch.isfinite(value).all().item(), "finite_aligned_batch_input_required")
            kwargs[parameter] = value.to(device=device, dtype=torch.float32)
    logits = model(**kwargs)
    _require(isinstance(logits, torch.Tensor) and logits.shape == target.shape
             and torch.isfinite(logits).all().item(), "finite_one_logit_per_knee_required")
    return logits.float(), target


def _new_optimizer(groups, plan):
    recipe = plan["neural_training"]
    _require(bool(groups), "nonempty_optimizer_parameter_groups_required")
    seen = set()
    copied = []
    for group in groups:
        parameters = list(group["params"])
        _require(parameters and all(isinstance(p, torch.nn.Parameter) for p in parameters),
                 "valid_optimizer_parameters_required")
        _require(not seen.intersection(id(p) for p in parameters)
                 and len({id(p) for p in parameters}) == len(parameters),
                 "optimizer_parameters_must_be_disjoint")
        seen.update(id(p) for p in parameters)
        rate, decay = group["lr"], group["weight_decay"]
        _require(type(rate) in {int, float} and math.isfinite(rate) and rate > 0
                 and type(decay) in {int, float} and math.isfinite(decay) and decay >= 0,
                 "valid_registered_optimizer_group_recipe_required")
        copied.append({"params": parameters, "lr": rate, "initial_lr": rate,
                       "weight_decay": decay, "name": group.get("name", str(len(copied)))})
    return torch.optim.AdamW(copied, betas=tuple(recipe["betas"]), eps=recipe["epsilon"])


def _capture_group_state(model, device):
    """Save only state a rejected forward may mutate, never fitted parameters."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
            "buffers": {name: value.detach().clone() for name, value in model.named_buffers()}}


def _restore_group_state(model, state, device):
    buffers = dict(model.named_buffers())
    _require(set(buffers) == set(state["buffers"]), "stable_registered_model_buffers_required_for_amp_replay")
    with torch.no_grad():
        for name, cached in state["buffers"].items():
            value = buffers[name]
            _require(value.shape == cached.shape and value.dtype == cached.dtype and value.device == cached.device,
                     "stable_registered_model_buffers_required_for_amp_replay")
            value.copy_(cached)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _gradient_parameters(optimizer):
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]
                  if parameter.grad is not None]
    _require(bool(parameters), "nonempty_accumulated_training_gradients_required")
    return parameters


def _gradients_finite(parameters):
    return all(bool(torch.isfinite(parameter.grad).all().item()) for parameter in parameters)


def _accumulate_group(model, batches, actual_knees, scaler, device, positive_weight):
    group_loss = 0.0
    for batch in batches:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits, target = _forward(model, batch, device)
            summed_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=positive_weight, reduction="sum")
            _require(torch.isfinite(summed_loss).item(), "finite_weighted_training_loss_required")
            normalized_loss = summed_loss / actual_knees
        scaler.scale(normalized_loss).backward()
        group_loss += float(summed_loss.detach().cpu())
    return group_loss


def _execute_effective_group(model, batches, actual_knees, optimizer, scaler, device, positive_weight):
    """Back off AMP on the same retained group; never omit a training knee.

    PyTorch's scaler skips the unsafe optimizer step before lowering its scale.
    A rejected pass does not advance the loader, loss accounting, RNG, buffers,
    scheduler or successful-update count. Invalid inputs/losses remain fatal.
    """
    state = _capture_group_state(model, device)
    start_scale = float(scaler.get_scale())
    _require(math.isfinite(start_scale) and start_scale > 0, "finite_positive_amp_scale_required")
    backoffs = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        may_restore = True
        try:
            group_loss = _accumulate_group(model, batches, actual_knees, scaler, device, positive_weight)
            scaler.unscale_(optimizer)
            parameters = _gradient_parameters(optimizer)
            if not _gradients_finite(parameters):
                _require(device.type == "cuda" and scaler.is_enabled(),
                         "nonfinite_unscaled_gradients_without_amp_recovery")
                previous_scale = float(scaler.get_scale())
                _require(backoffs < AMP_MAX_BACKOFFS and math.isfinite(previous_scale) and previous_scale > 1,
                         "amp_nonfinite_gradients_persist_after_bounded_replay")
                # unscale_ has already registered inf/NaN with GradScaler.
                # Its public step API must skip optimizer.step in this branch.
                scaler.step(optimizer)
                scaler.update()
                next_scale = float(scaler.get_scale())
                _require(math.isfinite(next_scale) and next_scale >= 1
                         and next_scale == previous_scale * 0.5,
                         "amp_overflow_requires_finite_standard_backoff")
                backoffs += 1
                optimizer.zero_grad(set_to_none=True)
                _restore_group_state(model, state, device)
                continue
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
            norm = float(norm)
            _require(math.isfinite(norm), "finite_unscaled_gradient_norm_required")
            previous_scale = float(scaler.get_scale())
            _require(math.isfinite(previous_scale) and previous_scale > 0,
                     "finite_positive_amp_scale_required")
            # No rejected-pass restoration is allowed after a valid optimizer
            # step may have changed parameters and optimizer moments.
            may_restore = False
            scaler.step(optimizer)
            scaler.update()
            successful_scale = float(scaler.get_scale())
            _require(math.isfinite(successful_scale) and successful_scale >= previous_scale,
                     "amp_overflow_after_finite_gradient_check_requires_numerical_review")
            optimizer.zero_grad(set_to_none=True)
            return {"summed_loss": group_loss, "amp_overflow_replays": backoffs,
                    "amp_scale_audit": {"start_scale": start_scale,
                        "successful_scale": previous_scale, "scale_after_update": successful_scale,
                        "backoffs": backoffs, "attempts": backoffs + 1,
                        "gradient_norm_before_clip": norm}}
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            if may_restore:
                _restore_group_state(model, state, device)
            raise


def _execute_training_epoch(model, loader, optimizer, scaler, device, positive_weight,
                            effective_batch_size, maximum_updates=None):
    """Execute only when explicitly called by an authorized public runner."""
    loss_sum, knees_seen, updates, replays = 0.0, 0, 0, 0
    scale_audit = []
    weight = torch.tensor(float(positive_weight), device=device, dtype=torch.float32)
    try:
        for batches, actual_knees in _batch_groups(loader, effective_batch_size):
            group = _execute_effective_group(model, batches, actual_knees, optimizer,
                                             scaler, device, weight)
            loss_sum += group["summed_loss"]
            knees_seen += actual_knees
            updates += 1
            replays += group["amp_overflow_replays"]
            scale_audit.append({"update_index_within_epoch": updates, **group["amp_scale_audit"]})
            if maximum_updates is not None and updates >= maximum_updates:
                break
    except torch.cuda.OutOfMemoryError as exc:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("cuda_oom_restart_entire_fold_same_seed_with_smaller_registered_micro_batch") from exc
    _require(knees_seen > 0, "nonempty_training_epoch_required")
    return {"weighted_loss_per_knee": loss_sum / knees_seen,
            "training_knees_seen": knees_seen, "optimizer_updates": updates,
            "amp_overflow_replays": replays, "amp_scale_audit": scale_audit}


def _evaluate(model, loader, device):
    targets, logits = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            _target_count(batch)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                values, target = _forward(model, batch, device)
            logits.extend(values.detach().cpu().tolist())
            targets.extend(target.detach().cpu().tolist())
    return binary_metrics(targets, logits), {"targets": targets, "logits": logits}


def _cpu_state(model):
    result = {name: value.detach().cpu().contiguous().clone()
              for name, value in model.state_dict().items()}
    _require(result and all(not value.is_floating_point() or torch.isfinite(value).all().item()
                           for value in result.values()), "finite_model_state_required")
    return result


def _write_checkpoint(output, name, state):
    from safetensors.torch import save_file
    target = output / name
    _require(not target.exists(), "new_checkpoint_file_required")
    save_file(state, str(target))
    checksum = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _new_output(output_dir):
    requested = Path(output_dir)
    _require(requested.is_absolute(), "absolute_private_training_output_required")
    for parent in (requested, *requested.parents):
        _require(not parent.is_symlink() and not (hasattr(parent, "is_junction") and parent.is_junction()),
                 "training_output_redirect_forbidden")
    output = requested.resolve()
    repository = Path(__file__).resolve().parents[2]
    _require(not output.is_relative_to(repository), "private_training_outputs_outside_public_repository_required")
    _require(not output.exists() and output.parent.is_dir(), "new_private_training_output_required")
    return output


def _prepare_cuda_environment():
    """Select deterministic cuBLAS behavior before its first CUDA use."""
    key = "CUBLAS_WORKSPACE_CONFIG"
    current = os.environ.get(key)
    if current is None and not torch.cuda.is_initialized():
        os.environ[key] = ":4096:8"
        current = os.environ[key]
    _require(current in {":4096:8", ":16:8"},
             "restart_cuda_runtime_with_cublas_workspace_config_4096_8_before_initialization")


def _validate_loader_bindings(training_loader, validation_loader, permit, positive_weight):
    """Recheck the narrow permit against actual datasets before any mutation."""
    from knee.training_data import TrainingFoldDataset
    datasets = (getattr(training_loader, "dataset", None), getattr(validation_loader, "dataset", None))
    for dataset, split in zip(datasets, ("training", "validation")):
        _require(isinstance(dataset, TrainingFoldDataset) and dataset.split == split
                 and dataset.package_sha256 == permit.package_sha256
                 and dataset.validation_fold == permit.validation_fold
                 and dataset.encoder_id == permit.image_encoder_id,
                 "training_datasets_must_match_execution_authority")
        _require(len(dataset.identities) > 0 and len(dataset.identities) == len(set(dataset.identities))
                 and len(dataset.identities) == len(dataset._targets)
                 and all(type(target) is int and target in {0, 1} for target in dataset._targets)
                 and set(dataset._targets) == {0, 1}, "nonempty_unique_both_class_fold_datasets_required")
    training, validation = datasets
    _require(not {row[0] for row in training.identities}.intersection(row[0] for row in validation.identities),
             "training_validation_participant_overlap_forbidden")
    expected_hash = _digest(sorted([list(identity) for identity in training.identities]))
    _require(training.train_identity_sha256 == validation.train_identity_sha256 == expected_hash,
             "exact_training_identity_binding_required")
    positive = sum(training._targets)
    expected_weight = (len(training._targets) - positive) / positive
    _require(math.isclose(positive_weight, expected_weight, rel_tol=1e-12, abs_tol=0),
             "class_weight_must_be_computed_from_training_targets_only")
    from torch.utils.data import RandomSampler, SequentialSampler
    sampler = getattr(training_loader, "sampler", None)
    generator = getattr(training_loader, "generator", None)
    _require(isinstance(sampler, RandomSampler) and sampler.replacement is False
             and generator is not None and generator.initial_seed() == permit.seed
             and isinstance(getattr(validation_loader, "sampler", None), SequentialSampler),
             "ordinary_seeded_training_shuffle_and_sequential_validation_required")
    return {"validation_fold": permit.validation_fold, "development_package_sha256": permit.package_sha256,
            "train_identity_sha256": expected_hash, "request_sha256": permit.request_sha256,
            "training_knees": len(training), "validation_knees": len(validation),
            "trained_participants": sorted({row[0] for row in training.identities})}


@contextmanager
def _seeded_execution(seed):
    py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def train_neural_fold(model, training_loader, validation_loader, plan, *, device, seed,
                      positive_weight, stage="stage_1", output_dir, phase_callback=None,
                      groups_callback=None, maximum_steps=None, smoke=False,
                      execution_authority=None):
    """Future fold training; caller must validate separate execution authority.

    No resumes, silent OOM retries, split changes, or hyperparameter searches.
    ``maximum_steps`` is reserved for an explicitly authorized <=2-update smoke;
    smoke metrics cannot be promoted into full five-fold model selection.
    """
    from knee.training_authority import require_permit
    validate_recipe(plan)
    model_id = getattr(model, "model_id", None)
    require_permit(execution_authority, model_id=model_id, stage=stage, smoke=smoke)
    _require(seed == execution_authority.seed, "training_seed_must_match_execution_authority")
    _require(type(seed) is int and seed in plan["selection"]["neural_finalist_seeds"],
             "registered_training_seed_required")
    _require(type(positive_weight) in {int, float} and math.isfinite(positive_weight)
             and positive_weight > 0, "positive_finite_training_fold_class_weight_required")
    _require(type(smoke) is bool and ((not smoke and maximum_steps is None)
             or (smoke and type(maximum_steps) is int and 1 <= maximum_steps <= 2)),
             "explicit_bounded_smoke_or_complete_training_required")
    allowed = {"clinical_mlp", *plan["radiographic_models"]["mandatory"],
               *plan["radiographic_models"]["conditional_challengers"],
               *plan["multimodal_models"]["primary"]}
    _require(model_id in allowed, "registered_neural_model_required")
    epoch_phase(model_id, stage, 1)
    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        _prepare_cuda_environment()
    _require(selected_device.type in {"cpu", "cuda"}
             and (selected_device.type != "cuda" or torch.cuda.is_available()), "available_cpu_or_cuda_device_required")
    micro = getattr(training_loader, "batch_size", None)
    clinical = model_id == "clinical_mlp"
    _require(type(micro) is int and micro in ({64} if clinical else {4, 2, 1}),
             "registered_clinical_or_image_micro_batch_required")
    _require(clinical or micro == execution_authority.image_micro_batch_size,
             "training_micro_batch_must_match_execution_authority")
    _require(getattr(model, "image_encoder_id", None) == execution_authority.image_encoder_id,
             "training_model_encoder_must_match_execution_authority")
    for loader in (training_loader, validation_loader):
        _require(getattr(loader, "drop_last", None) is False
                 and getattr(loader, "num_workers", None) == 0, "complete_single_process_loaders_required")
    binding = _validate_loader_bindings(training_loader, validation_loader, execution_authority, positive_weight)
    output = _new_output(output_dir)
    if phase_callback is None or groups_callback is None:
        from knee.training_models import set_phase, optimizer_groups
        phase_callback = phase_callback or set_phase
        groups_callback = groups_callback or optimizer_groups
    recipe = plan["neural_training"]
    monitor = EarlyStopping(patience=recipe["early_stopping"]["patience_epochs"],
                            monitor_starts_after_epoch=(10 if stage in {"stage_2", "fusion"} else 0))
    # Fusion follows its approved initial five frozen epochs before unfreezing.
    # Enumerating all phase-2 groups first preserves AdamW state across phases;
    # frozen parameters have no gradient, so AdamW applies no decay/update.
    enumerate_phase = ("full" if stage == "stage_2" or model_id.startswith("fusion_")
                       or clinical else "head_only")
    started = time.perf_counter()
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
        torch.cuda.reset_peak_memory_stats(selected_device)
    model.to(selected_device)
    phase_callback(model, enumerate_phase)
    groups = groups_callback(model, plan, enumerate_phase, stage="stage_2" if stage == "fusion" else stage)
    phase_callback(model, epoch_phase(model_id, stage, 1))
    output.mkdir(exist_ok=False)
    history, total_updates, best_state, best_predictions = [], 0, None, None
    with _seeded_execution(seed):
        # A caller may have inspected a batch already. Rewind the authenticated
        # ordinary-shuffle generator to the requested run seed before epoch 1.
        training_loader.generator.manual_seed(seed)
        optimizer = _new_optimizer(groups, plan)
        scaler = torch.amp.GradScaler("cuda", enabled=selected_device.type == "cuda")
        for epoch in range(1, recipe["maximum_epochs"] + 1):
            phase = epoch_phase(model_id, stage, epoch)
            model.train()
            phase_callback(model, phase)
            factor = learning_rate_factor(epoch, maximum_epochs=recipe["maximum_epochs"],
                                          warmup_epochs=recipe["scheduler"]["warmup_epochs"],
                                          minimum_fraction=recipe["scheduler"]["minimum_learning_rate_fraction"])
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            remaining = None if maximum_steps is None else maximum_steps - total_updates
            training = _execute_training_epoch(model, training_loader, optimizer, scaler,
                                               selected_device, positive_weight,
                                               64 if clinical else recipe["image_effective_batch_size"], remaining)
            total_updates += training["optimizer_updates"]
            metrics, predictions = _evaluate(model, validation_loader, selected_device)
            improved, stop = monitor.update(epoch, metrics["average_precision_score"])
            # Smoke terminates before the stage-2 monitor starts; its snapshot
            # is explicitly a smoke artifact, never a selected best fold.
            if improved or (smoke and best_state is None):
                best_state, best_predictions = _cpu_state(model), predictions
            history.append({"epoch": epoch, "phase": phase, "learning_rate_factor": factor,
                            **training, "validation": metrics, "selected_best": improved})
            if stop or (maximum_steps is not None and total_updates >= maximum_steps):
                break
        _require(best_state is not None, "monitored_best_checkpoint_required")
        last_state = _cpu_state(model)
        last_hash = _write_checkpoint(output, "last_model.safetensors", last_state)
        best_hash = _write_checkpoint(output, "best_model.safetensors", best_state)
        model.load_state_dict(best_state, strict=True)
        model.eval()
    resource_profile = {"device_type": selected_device.type, "torch_version": str(torch.__version__),
                        "cuda_version": torch.version.cuda, "cuda_peak_measured": False}
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
        properties = torch.cuda.get_device_properties(selected_device)
        resource_profile.update({"device_name": properties.name, "total_memory_bytes": properties.total_memory,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(selected_device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(selected_device),
            "cuda_peak_measured": True})
    resource_profile["elapsed_seconds"] = time.perf_counter() - started
    result = {"version": VERSION, "model_id": model_id, "stage": stage, "seed": seed,
              "status": "bounded_training_smoke_complete" if smoke else "training_fold_complete",
              "scope": "development_only", "smoke": smoke,
              "eligible_for_model_selection": not smoke,
              "best_epoch": monitor.best_epoch, "best_validation_ap": monitor.best_score,
              "completed_epochs": len(history), "optimizer_updates": total_updates,
              "restored_best_checkpoint": True, "resume_supported": False,
              "optimizer_state_saved": False, "positive_weight": float(positive_weight),
              "training_executed": True, "optimizer_created": True,
              "scheduler_horizon_epochs": recipe["maximum_epochs"],
              "gpu_resource_profile_recorded": selected_device.type == "cuda",
              "resource_profile": resource_profile,
              "micro_batch_size": micro, "effective_batch_size": 64 if clinical else 16,
              "device": str(selected_device), "precision": "float16_autocast_grad_scaler" if selected_device.type == "cuda" else "float32",
              "numerical_repair": {"version": AMP_NUMERICAL_REPAIR_VERSION,
                  "maximum_backoffs_per_effective_group": AMP_MAX_BACKOFFS,
                  "same_effective_group_replayed": True, "nonfinite_gradients_never_clipped": True},
              "amp_overflow_replays": sum(item.get("amp_overflow_replays", 0) for item in history),
              "amp_scale_audit": [{"epoch": item["epoch"], **audit}
                  for item in history for audit in item.get("amp_scale_audit", [])],
              "checkpoint_sha256": {"best_model.safetensors": best_hash, "last_model.safetensors": last_hash},
              "history": history, "reserved_test_accessed": False, "provenance": binding}
    best_predictions["identities"] = [list(identity) for identity in validation_loader.dataset.identities]
    for filename, content in (("training_record.json", result), ("best_validation_predictions.json", best_predictions)):
        with (output / filename).open("x", encoding="utf-8") as handle:
            json.dump(content, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    return result
