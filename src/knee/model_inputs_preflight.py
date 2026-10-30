"""Fold-local clinical preparation and exhaustive CPU loading, never training.

This stage only reads the sealed development archive. Image architecture probes
use random weights to test tensor compatibility; they do not validate pretrained
checkpoint loading, candidate eligibility, anatomical quality, or prediction.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torchvision
from torchvision.models import densenet121, vit_b_16

from knee.clinical_fold_inputs import POLICY as CLINICAL_POLICY, fit_clinical_fold, transform_clinical_rows
from knee.colab_data import canonical_json, checked_path, load_development_package, require, sha_file, validate_rows
from knee.model_input_adapters import DevelopmentFoldDataset, MODELS, make_development_loader

VERSION = "model_inputs_loading_preflight_v1.0"
ENCODERS = ["image_densenet121", "image_vit_b16"]
POLICY = {
    "version": VERSION, "registry_version": "1.0", "image_encoders": ENCODERS,
    "clinical_features": ["AGEYEARS", "SEX", "BMI"], "clinical_policy": CLINICAL_POLICY,
    "image_geometry": "preserve_sealed_physical_letterbox_224_without_resize_crop_or_flip",
    "checkpoint_input_metadata": "explicit_torchvision_IMAGENET1K_V1_not_full_transform",
    "loader": {"batch_size": 8, "num_workers": 0, "shuffle": False, "drop_last": False},
    "architecture_probe": {"weights": None, "training": False, "samples_per_fold": 2,
        "feature_extraction_only": True},
    "folds": [0, 1, 2, 3, 4], "seed": 2026, "maximum_cpu_threads": 2,
    "pretrained_downloads_authorized": False, "predictive_training_authorized": False,
    "reserved_test_access_authorized": False,
}


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate_policy(policy, registry):
    require(isinstance(policy, dict) and canonical_json(policy) == canonical_json(POLICY),
        "exact_registered_input_preflight_policy_required")
    require(isinstance(registry, dict) and registry.get("registry_version") == "1.0"
        and registry.get("status") == "approved_for_development_screening"
        and registry.get("test_block_must_remain_closed") is True,
        "approved_closed_test_model_registry_required")
    common = registry.get("common_contract", {})
    require(common.get("clinical_predictors") == ["AGEYEARS_V00", "SEX", "BMI_V00"]
        and common.get("group_unit") == "participant"
        and common.get("outcome") == "PROGRESION_48M"
        and common.get("radiographic_visit") == "V00"
        and common.get("outcome_visit") == "V06"
        and common.get("same_cohort_and_partitions") is True, "registered_three_feature_contract_required")
    baselines = registry.get("radiographic", {}).get("mandatory_baselines", [])
    require(isinstance(baselines, list) and [r.get("id") for r in baselines] == ENCODERS
        and [r.get("family") for r in baselines] == ["densenet121", "vit_b_16"]
        and all(r.get("pretraining") == "imagenet" and r.get("input_resolutions") == [224]
            for r in baselines), "registered_224_baselines_required")


def prepare_fold_states(rows, policy):
    require(canonical_json(policy) == canonical_json(POLICY), "exact_registered_input_preflight_policy_required")
    validate_rows(rows, "development")
    return [fit_clinical_fold(rows, fold, policy["clinical_policy"]) for fold in policy["folds"]]


def check_loader(dataset, batch_size=8):
    """Traverse every selected row, checking paired inputs and final partial batch."""
    loader = make_development_loader(dataset, batch_size)
    identities, total, batches = [], 0, 0
    image_hash, clinical_hash, target_hash = (hashlib.sha256() for _ in range(3))
    for batch in loader:
        require(set(batch) == {"image", "clinical", "target", "src_subject_id", "side"},
            "exact_aligned_batch_fields_required")
        image, clinical, target = (batch[k] for k in ("image", "clinical", "target"))
        n = len(batch["src_subject_id"])
        require(0 < n <= batch_size and len(batch["side"]) == n
            and tuple(image.shape) == (n, 3, 224, 224) and tuple(clinical.shape) == (n, 3)
            and tuple(target.shape) == (n,), "aligned_image_clinical_target_shapes_required")
        for value in (image, clinical, target):
            require(value.device.type == "cpu" and value.dtype == torch.float32
                and value.is_contiguous() and not value.requires_grad and bool(torch.isfinite(value).all()),
                "finite_contiguous_cpu_inputs_without_grad_required")
        require(bool(((target == 0) | (target == 1)).all()), "binary_target_only_required")
        identities.extend(zip(batch["src_subject_id"], batch["side"]))
        image_hash.update(image.numpy().tobytes(order="C"))
        clinical_hash.update(clinical.numpy().tobytes(order="C"))
        target_hash.update(target.numpy().tobytes(order="C"))
        total += n
        batches += 1
    require(tuple(identities) == dataset.identities and total == len(dataset),
        "complete_ordered_development_loading_required")
    return {"rows": total, "participants": len({sid for sid, _ in identities}), "batches": batches,
        "image_shape": [3, 224, 224], "clinical_features": 3, "target_shape": [],
        "ordered_identity_sha256": digest(identities), "image_sha256": image_hash.hexdigest(),
        "clinical_sha256": clinical_hash.hexdigest(), "targets_sha256": target_hash.hexdigest()}


def _state_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(canonical_json([name, str(array.dtype), list(array.shape)]))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def architecture_probe(encoder_id, fold_samples):
    """Random-weight eval forward, bounded CPU memory, no optimizer or backward."""
    require(encoder_id in ENCODERS, "registered_architecture_probe_required")
    require(isinstance(fold_samples, list) and len(fold_samples) == 5, "five_fold_probe_samples_required")
    for sample in fold_samples:
        require(isinstance(sample, torch.Tensor) and sample.device.type == "cpu"
            and sample.dtype == torch.float32 and sample.ndim == 4
            and 1 <= sample.shape[0] <= 2 and tuple(sample.shape[1:]) == (3, 224, 224)
            and not sample.requires_grad and bool(torch.isfinite(sample).all()), "bounded_cpu_probe_sample_required")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        if encoder_id == "image_densenet121":
            model, width = densenet121(weights=None).cpu(), 1024
            model.classifier = torch.nn.Identity()
        else:
            model, width = vit_b_16(weights=None).cpu(), 768
            model.heads = torch.nn.Identity()
        model.eval()
        require(not any(module.training for module in model.modules()), "every_probe_module_must_be_eval")
        before = _state_digest(model)
        outputs = []
        with torch.inference_mode():
            for fold, sample in enumerate(fold_samples):
                features = model(sample)
                require(tuple(features.shape) == (len(sample), width) and features.device.type == "cpu"
                    and not features.requires_grad and bool(torch.isfinite(features).all()),
                    "finite_expected_architecture_features_required")
                outputs.append({"validation_fold": fold, "rows": len(sample), "feature_width": width,
                    "features_sha256": hashlib.sha256(features.numpy().tobytes(order="C")).hexdigest()})
        after = _state_digest(model)
        require(before == after and all(p.grad is None for p in model.parameters()),
            "probe_changed_parameters_buffers_or_gradients")
        del model
    gc.collect()
    return {"encoder_id": encoder_id, "initialization": "random_weights_none",
        "pretrained_checkpoint_loaded": False, "pretrained_download_attempted": False,
        "mode": "eval_torch_inference_mode", "parameters_and_buffers_unchanged": True,
        "all_parameter_gradients_none": True, "state_sha256_before": before,
        "state_sha256_after": after, "folds": outputs}


def _read_bound(spec):
    require(isinstance(spec, dict) and set(spec) == {"path", "sha256"}, "bound_source_specification_required")
    path = checked_path(spec["path"])
    require(path.is_file() and sha_file(path) == spec["sha256"], "bound_source_hash_mismatch")
    return path


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(path, payload):
    path = checked_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, "existing_model_input_artifact_changed")
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def _save_array(path, array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    _save(path, stream.getvalue())


def _array_digest(tensors):
    result = hashlib.sha256()
    for identity, array in sorted(tensors.items()):
        result.update(canonical_json(identity))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def execute(config_path):
    started = time.monotonic()
    config_path = checked_path(config_path)
    config_hash = sha_file(config_path)
    config = _read_json(config_path)
    source_names = {"development_package", "handoff", "package_closure", "colab_confirmation",
        "policy", "registry", "authority"}
    require(isinstance(config, dict) and set(config) == source_names | {"version", "source_pins", "output_dir"}
        and config["version"] == VERSION, "exact_model_inputs_private_config_required")
    sources = {name: _read_bound(config[name]) for name in sorted(source_names)}
    pins = config["source_pins"]
    require(isinstance(pins, list) and len(pins) >= 7, "implementation_source_pins_required")
    pinned_paths = [_read_bound(spec) for spec in pins]
    required_files = {Path(__file__).resolve(), Path(__file__).with_name("clinical_fold_inputs.py").resolve(),
        Path(__file__).with_name("model_input_adapters.py").resolve(),
        Path(__file__).with_name("colab_data.py").resolve()}
    require(required_files <= {p.resolve() for p in pinned_paths}
        and len({p.resolve() for p in pinned_paths}) == len(pins), "exact_executed_source_pins_required")
    output = checked_path(config["output_dir"])
    require(not output.exists() or output.is_dir(), "new_stage_output_directory_required")
    require(all(output != p.parent and output not in p.parents for p in [*sources.values(), *pinned_paths, config_path]),
        "stage_output_must_not_contain_bound_sources")
    authority = _read_json(sources["authority"])
    required_authority = {"development_package_only": True,
        "fit_clinical_preprocessing_inside_each_training_fold": True,
        "derive_image_input_copies_without_changing_geometry": True,
        "cpu_loading_and_random_architecture_forward": True, "predictive_model_training": False,
        "pretrained_weight_download": False, "reserved_test_access": False,
        "original_modification": False, "partition_modification": False, "new_model_or_predictor_addition": False}
    require(authority.get("version") == "model_inputs_loading_authority_v1.0"
        and all(authority.get(k) is v for k, v in required_authority.items()), "bounded_model_inputs_authority_required")
    policy, registry = _read_json(sources["policy"]), _read_json(sources["registry"])
    validate_policy(policy, registry)
    require(np.__version__ == "1.26.4" and torch.__version__ == "2.6.0+cpu"
        and torchvision.__version__ == "0.21.0+cpu" and not torch.cuda.is_available()
        and os.environ.get("COLAB_GPU", "0") != "1" and not os.environ.get("COLAB_TPU_ADDR"),
        "frozen_cpu_loading_runtime_required")
    torch.set_num_threads(min(policy["maximum_cpu_threads"], os.cpu_count() or 1))
    handoff, prior, colab = (_read_json(sources[k]) for k in ("handoff", "package_closure", "colab_confirmation"))
    dev = handoff["development"]
    require(dev["sha256"] == config["development_package"]["sha256"]
        and dev["bytes"] == sources["development_package"].stat().st_size
        and prior.get("status") == "participant_partitions_and_private_colab_packages_verified"
        and prior.get("handoff_sha256") == config["handoff"]["sha256"], "sealed_development_handoff_required")
    require(colab.get("version") == "colab_development_transport_preflight_v1.1"
        and colab.get("status") == "passed" and colab.get("colab_runtime_executed") is True
        and colab.get("local_rehearsal") is False and colab.get("cpu_only_guard_passed") is True
        and colab.get("development_package_sha256") == dev["sha256"]
        and colab.get("all_tensor_file_and_pixel_hashes_verified") is True
        and colab.get("training_executed") is False
        and colab.get("reserved_test_predictive_evaluation_opened") is False,
        "prior_real_colab_development_confirmation_required")
    loaded = load_development_package(sources["development_package"], dev["sha256"])
    rows, tensors = loaded["manifest"]["rows"], loaded["tensors"]
    require(loaded["manifest"]["bindings"]["partition_sha256"] == handoff["partition_sha256"]
        and len(rows) == dev["knees"] == colab["verified_development_tensors"]
        and len({r["src_subject_id"] for r in rows}) == dev["participants"] == colab["verified_development_participants"],
        "exact_previously_verified_development_membership_required")
    rows_before, tensors_before = digest(rows), _array_digest(tensors)
    states = prepare_fold_states(rows, policy)
    records, file_hashes = [], {}
    for fold, state in enumerate(states):
        record = {"validation_fold": fold, "clinical_state_sha256": state["record_sha256"], "splits": {}}
        state_path = output / "clinical" / f"fold_{fold}_state.json"
        _save(state_path, canonical_json(state))
        file_hashes[state_path.relative_to(output).as_posix()] = sha_file(state_path)
        for split in ("training", "validation"):
            selected = [r for r in rows if (r["cv_fold"] == fold) == (split == "validation")]
            identities = [[r["src_subject_id"], r["side"]] for r in selected]
            clinical = transform_clinical_rows(selected, state)
            prefix = output / "clinical" / f"fold_{fold}_{split}"
            array_path, order_path = Path(str(prefix) + ".npy"), Path(str(prefix) + "_order.json")
            _save_array(array_path, clinical)
            _save(order_path, canonical_json(identities))
            for path in (array_path, order_path):
                file_hashes[path.relative_to(output).as_posix()] = sha_file(path)
            record["splits"][split] = {"rows": len(selected),
                "participants": len({r["src_subject_id"] for r in selected}),
                "imputed_BMI_rows": sum(r["BMI"] is None for r in selected),
                "ordered_identity_sha256": digest(identities),
                "clinical_sha256": hashlib.sha256(clinical.tobytes(order="C")).hexdigest()}
        training_people = {r["src_subject_id"] for r in rows if r["cv_fold"] != fold}
        validation_people = {r["src_subject_id"] for r in rows if r["cv_fold"] == fold}
        require(not training_people & validation_people, "participant_overlap_detected")
        records.append(record)
    loaders, probes = {}, {}
    for encoder in policy["image_encoders"]:
        samples, fold_results = [], []
        for fold, state in enumerate(states):
            result = {"validation_fold": fold, "splits": {}}
            for split in ("training", "validation"):
                dataset = DevelopmentFoldDataset(rows, tensors, state, encoder, split)
                summary = check_loader(dataset, policy["loader"]["batch_size"])
                expected = records[fold]["splits"][split]
                require(all(summary[k] == expected[k] for k in ("rows", "participants", "ordered_identity_sha256", "clinical_sha256")),
                    "image_clinical_pairing_changed")
                result["splits"][split] = summary
                if split == "validation":
                    samples.append(torch.stack([dataset[i]["image"] for i in range(min(2, len(dataset)))]))
            fold_results.append(result)
            print(f"Carga CPU: {encoder}, pliegue {fold + 1}/5 verificado.", flush=True)
        loaders[encoder] = fold_results
        probes[encoder] = architecture_probe(encoder, samples)
        print(f"Compatibilidad de arquitectura sin entrenamiento: {encoder} verificada.", flush=True)
    require(digest(rows) == rows_before and _array_digest(tensors) == tensors_before,
        "in_memory_sealed_inputs_changed")
    for name in source_names:
        _read_bound(config[name])
    for spec in pins:
        _read_bound(spec)
    require(sha_file(config_path) == config_hash, "private_stage_configuration_changed")
    summary = {"version": VERSION, "status": "development_inputs_and_cpu_loading_verified",
        "execution_environment": "local_frozen_cpu_not_colab", "prior_real_colab_data_preflight_verified": True,
        "development_participants": dev["participants"], "development_knees": dev["knees"],
        "clinical_features": policy["clinical_features"], "clinical_states_fitted": 5,
        "clinical_preprocessing_fitted": True, "BMI_imputation_applied_to_copies": True,
        "missing_BMI_development_knees_retained": sum(r["BMI"] is None for r in rows),
        "statistics_unit": "training_knee_rows", "validation_or_target_used_for_fitting": False,
        "clinical_inputs_shared_by_registered_clinical_and_multimodal_scenarios": True,
        "aligned_multimodal_input_pairs_verified": True,
        "folds": records, "image_loading": loaders, "architecture_probes": probes,
        "total_loader_row_visits": sum(s["rows"] for v in loaders.values() for f in v for s in f["splits"].values()),
        "loader_batch_size": 8, "new_image_files_written": 0,
        "additional_resize_crop_flip_or_augmentation": False,
        "pretrained_weights_loaded": False, "pretrained_weight_eligibility_certified": False,
        "modern_challengers_unchanged_and_deferred": [r["id"] for r in registry["radiographic"]["modern_challengers"]],
        "clinical_predictor_or_fusion_network_instantiated": False,
        "predictive_model_training_executed": False, "optimizer_created": False, "backward_executed": False,
        "reserved_test_opened": False, "originals_modified": False, "partitions_modified": False,
        "source_files_unchanged": True, "development_rows_and_pixels_unchanged": True,
        "sources_sha256": {k: config[k]["sha256"] for k in sorted(source_names)},
        "implementation_sha256": {p.name: spec["sha256"] for p, spec in zip(pinned_paths, pins)},
        "private_configuration_sha256": config_hash, "clinical_artifacts_sha256": file_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "torchvision": torchvision.__version__, "torch_threads": torch.get_num_threads()},
        "limitations": ["Random-weight architecture probes do not verify pretrained checkpoint loading or quality.",
            "No clinical predictor, fusion network, optimizer, loss, prediction metric or training was executed.",
            "This local CPU check is not a new Google Colab execution or a GPU memory certification.",
            "External KNEEL/OAI pretraining overlap remains unresolved; no clinical validation is claimed."],
        "next_step": "review_pretrained_checkpoint_eligibility_and_freeze_training_configuration_before_any_training"}
    _save(output / "closure_summary.json", canonical_json(summary))
    print(json.dumps({"status": summary["status"], "development_knees": len(rows),
        "folds": 5, "total_loader_row_visits": summary["total_loader_row_visits"],
        "predictive_model_training_executed": False, "reserved_test_opened": False,
        "elapsed_seconds": round(time.monotonic() - started, 2)}, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    execute(args.config)
