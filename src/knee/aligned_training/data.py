"""Development-only training datasets; no resplitting or image geometry changes."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from knee.colab_data import validate_rows, load_development_package, sha_file, checked_path, require
from knee.clinical_fold_inputs import fit_clinical_fold, transform_clinical_rows
from knee.training_plan import training_fold_class_weight


@dataclass(frozen=True)
class FoldInputs:
    training: object
    validation: object
    clinical_state: dict
    class_weight: dict


class TrainingFoldDataset(Dataset):
    def __init__(self, selected, tensors, clinical, state, split, encoder_id, package_sha256,
                 *, clinical_scenario="clinical_main", baseline_sidecar_sha256=None,
                 augmentation_policy=None, seed=2026):
        require(split in {"training", "validation"}, "training_or_validation_dataset_required")
        self.split, self.validation_fold = split, state["validation_fold"]
        self.package_sha256, self.encoder_id = package_sha256, encoder_id
        self.train_identity_sha256 = state["train_identity_sha256"]
        self.identities = tuple((r["src_subject_id"], r["side"]) for r in selected)
        self._targets = tuple(r["PROGRESION_48M"] for r in selected)
        self._clinical = clinical
        self._tensors = tensors
        self.clinical_scenario = clinical_scenario
        self.baseline_sidecar_sha256 = baseline_sidecar_sha256
        self.augmentation_policy = deepcopy(augmentation_policy)
        self.seed, self.epoch = seed, 1
        if augmentation_policy is not None:
            from knee.training_augmentation import validate_policy
            validate_policy(augmentation_policy)
            require(split == "training" and encoder_id is not None, "augmentation_only_image_training")

    def set_epoch(self, epoch):
        require(type(epoch) is int and 1 <= epoch <= 60, "registered_training_epoch_required")
        self.epoch = epoch

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, index):
        identity = self.identities[index]
        result = {"clinical": torch.from_numpy(self._clinical[index].copy()),
            "target": torch.tensor(self._targets[index], dtype=torch.float32),
            "src_subject_id": identity[0], "side": identity[1]}
        if self.encoder_id is not None:
            from knee.aligned_training.models import normalize_for_model
            image = self._tensors[identity]
            if self.augmentation_policy is not None:
                from knee.training_augmentation import augment_training_image
                image = augment_training_image(image, seed=self.seed, validation_fold=self.validation_fold,
                    epoch=self.epoch, src_subject_id=identity[0], side=identity[1], policy=self.augmentation_policy)
            result["image"] = normalize_for_model(image, self.encoder_id)
        return result


def prepare_fold(rows, tensors, validation_fold, *, encoder_id=None, package_sha256=None,
                 clinical_scenario="clinical_main", baseline_sidecar=None, baseline_sidecar_sha256=None,
                 augmentation_policy=None, seed=2026, methodology=False):
    validate_rows(rows, "development")
    if package_sha256 is not None:
        from knee.colab_data import valid_sha
        require(valid_sha(package_sha256), "valid_development_package_binding_required")
    expected = {(r["src_subject_id"], r["side"]) for r in rows}
    require(isinstance(tensors, dict) and set(tensors) == expected, "exact_training_tensor_mapping_required")
    for source in tensors.values():
        require(isinstance(source, np.ndarray) and source.dtype == np.dtype("float32")
            and source.shape == (1, 224, 224) and source.flags.c_contiguous and not source.flags.writeable
            and bool(np.isfinite(source).all()) and source.min() >= 0 and source.max() <= 1,
            "sealed_readonly_training_image_required")
    enriched = rows
    transform = transform_clinical_rows
    if methodology:
        from knee.clinical_scenarios import join_baseline_sidecar, fit_scenario_fold, transform_scenario_rows
        if clinical_scenario != "clinical_main":
            require(baseline_sidecar is not None and baseline_sidecar_sha256 is not None,
                "pinned_development_baseline_sidecar_required")
            enriched = join_baseline_sidecar(rows, baseline_sidecar, package_sha256,
                expected_sidecar_sha256=baseline_sidecar_sha256)
        else:
            require(baseline_sidecar is None and baseline_sidecar_sha256 is None, "main_scenario_has_no_auxiliary_inputs")
        state = fit_scenario_fold(enriched, validation_fold, clinical_scenario)
        transform = transform_scenario_rows
    else:
        require(clinical_scenario == "clinical_main" and baseline_sidecar is None
            and baseline_sidecar_sha256 is None and augmentation_policy is None,
            "historical_inputs_cannot_enable_new_scenarios_or_augmentation")
        state = fit_clinical_fold(rows, validation_fold)
    weight = training_fold_class_weight(rows, validation_fold)
    require(state["train_identity_sha256"] == weight["train_identity_sha256"], "clinical_target_training_identity_mismatch")
    subsets = [[r for r in enriched if (r["cv_fold"] == validation_fold) == validation]
        for validation in (False, True)]
    require(all(subsets), "nonempty_training_and_validation_required")
    people = [{r["src_subject_id"] for r in subset} for subset in subsets]
    require(not people[0] & people[1], "training_validation_participant_overlap")
    require({r["PROGRESION_48M"] for r in subsets[1]} == {0, 1}, "validation_requires_both_classes_for_auc")
    datasets = [TrainingFoldDataset(subset, tensors, transform(subset, state), state,
        split, encoder_id, package_sha256, clinical_scenario=clinical_scenario,
        baseline_sidecar_sha256=baseline_sidecar_sha256,
        augmentation_policy=augmentation_policy if split == "training" else None, seed=seed)
        for subset, split in zip(subsets, ("training", "validation"))]
    return FoldInputs(*datasets, state, weight)


def loaders(fold, *, seed, micro_batch_size):
    require(isinstance(fold, FoldInputs) and type(seed) is int and seed in {2026, 2027, 2028},
        "registered_seed_and_fold_inputs_required")
    expected = {1, 2, 4} if fold.training.encoder_id else {64}
    require(type(micro_batch_size) is int and micro_batch_size in expected, "registered_loader_batch_size_required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (DataLoader(fold.training, batch_size=micro_batch_size, shuffle=True, generator=generator,
        num_workers=0, drop_last=False), DataLoader(fold.validation, batch_size=micro_batch_size,
        shuffle=False, num_workers=0, drop_last=False))


def load_sealed_inputs(package_dir, reference_path, repository_root, expected_package_sha256):
    """Future training path: authenticate notebook32 inputs before opening ZIP."""
    from knee.model_inputs_colab import _reference, SCIENTIFIC_FILES, _read_json
    package, reference_file, root = map(checked_path, (package_dir, reference_path, repository_root))
    require(package.is_dir() and reference_file.is_file(), "existing_development_package_and_reference_required")
    reference = _read_json(reference_file)
    _reference(reference)
    for name, relative in SCIENTIFIC_FILES.items():
        require(sha_file(root / relative) == reference["implementation_sha256"][name],
            "verified_clinical_or_input_implementation_changed")
    for name, relative in (("policy", "configs/model_inputs_preflight_v1.json"),
                           ("registry", "configs/model_registry.json")):
        require(sha_file(root / relative) == reference["sources_sha256"][name],
                "verified_input_policy_or_registry_changed")
    handoff_path = package / "handoff.json"
    require(sha_file(handoff_path) == reference["sources_sha256"]["handoff"], "sealed_handoff_changed")
    handoff = _read_json(handoff_path)
    require(handoff.get("version") == "participant_split_colab_preparation_v1.0"
        and handoff.get("training_executed") is False
        and handoff.get("reserved_test_predictive_evaluation_opened") is False
        and handoff.get("development", {}).get("file") == "development_package.zip",
        "development_handoff_with_closed_test_required")
    require(expected_package_sha256 == reference["sources_sha256"]["development_package"],
        "training_request_package_does_not_match_verified_inputs")
    archive = checked_path(package / "development_package.zip")
    development = handoff["development"]
    require(archive.is_file() and development.get("sha256") == expected_package_sha256
        and development.get("bytes") == archive.stat().st_size,
        "sealed_development_package_size_or_hash_changed")
    loaded = load_development_package(archive, expected_package_sha256)
    require(loaded["manifest"]["bindings"]["partition_sha256"] == handoff.get("partition_sha256")
        and loaded["manifest"]["bindings"]["split_policy_sha256"] == handoff.get("split_policy_sha256")
        and len(loaded["manifest"]["rows"]) == development.get("knees") == reference["development_knees"]
        and loaded["verification"]["participants"] == development.get("participants") == reference["development_participants"],
        "verified_development_membership_changed")
    verify_clinical_reference(loaded["manifest"]["rows"], reference, root)
    return loaded


def verify_clinical_reference(rows, reference, repository_root):
    """Authenticate all five previously verified states and transformed copies.

    Deterministic fold calculations are replayed using the sealed functions.
    Nothing is fitted on validation rows, and no predictor is constructed.
    """
    from knee import model_inputs_preflight as previous
    from knee.colab_data import canonical_json
    root = checked_path(repository_root)
    policy = json.loads((root / "configs/model_inputs_preflight_v1.json").read_text(encoding="utf-8"))
    registry = json.loads((root / "configs/model_registry.json").read_text(encoding="utf-8"))
    previous.validate_policy(policy, registry)
    states = previous.prepare_fold_states(rows, policy)
    records, artifacts = [], {}
    for fold, state in enumerate(states):
        artifacts[f"clinical/fold_{fold}_state.json"] = hashlib.sha256(canonical_json(state)).hexdigest()
        record = {"validation_fold": fold, "clinical_state_sha256": state["record_sha256"], "splits": {}}
        for split in ("training", "validation"):
            selected = [row for row in rows if (row["cv_fold"] == fold) == (split == "validation")]
            identities = [[row["src_subject_id"], row["side"]] for row in selected]
            clinical = transform_clinical_rows(selected, state)
            buffer = io.BytesIO()
            np.save(buffer, clinical, allow_pickle=False)
            artifacts[f"clinical/fold_{fold}_{split}.npy"] = hashlib.sha256(buffer.getvalue()).hexdigest()
            artifacts[f"clinical/fold_{fold}_{split}_order.json"] = hashlib.sha256(canonical_json(identities)).hexdigest()
            record["splits"][split] = {"rows": len(selected),
                "participants": len({row["src_subject_id"] for row in selected}),
                "imputed_BMI_rows": sum(row["BMI"] is None for row in selected),
                "ordered_identity_sha256": previous.digest(identities),
                "clinical_sha256": hashlib.sha256(clinical.tobytes(order="C")).hexdigest()}
        records.append(record)
    require(records == reference["folds"], "previously_verified_clinical_fold_states_or_values_changed")
    require(artifacts == reference["clinical_artifacts_sha256"], "previously_verified_clinical_artifact_bytes_changed")
    return {"clinical_folds_verified": len(states), "clinical_states_and_transformed_hashes_match_reference": True}
