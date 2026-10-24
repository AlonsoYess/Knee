"""Repeat the frozen input-loading check on CPU against an approved reference.

All clinical calculations, loaders, and architecture probes reuse the previously
verified core. Only deterministic input hashes are compared across platforms;
random-weight feature or model-state hashes are checked within each run only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import time

from knee import model_inputs_preflight as core
from knee.model_inputs_preflight import (POLICY, _array_digest, _save, _save_array,
    architecture_probe, check_loader, digest, prepare_fold_states, validate_policy)
from knee.colab_data import (canonical_json, checked_path, load_development_package,
    require, sha_file, valid_sha)

VERSION = "colab_model_input_loading_preflight_v1.0"
SCIENTIFIC_FILES = {
    "clinical_fold_inputs.py": "src/knee/clinical_fold_inputs.py",
    "colab_data.py": "src/knee/colab_data.py",
    "model_input_adapters.py": "src/knee/model_input_adapters.py",
    "model_inputs_preflight.py": "src/knee/model_inputs_preflight.py",
    "model_inputs_preflight_v1.json": "configs/model_inputs_preflight_v1.json",
    "model_registry.json": "configs/model_registry.json",
    "prepare_model_inputs.py": "scripts/prepare_model_inputs.py",
    "requirements-model-input-loading.txt": "requirements-model-input-loading.txt",
}


def _read_json(path, maximum=4 * 1024**2):
    path = checked_path(path)
    require(path.is_file() and 0 < path.stat().st_size <= maximum, "bounded_preflight_json_required")
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime(execution_environment):
    require(execution_environment in {"google_colab_cpu", "local_rehearsal"},
        "explicit_colab_or_local_rehearsal_environment_required")
    require(core.np.__version__ == "1.26.4" and core.torch.__version__ == "2.6.0+cpu"
        and core.torchvision.__version__ == "0.21.0+cpu" and not core.torch.cuda.is_available()
        and os.environ.get("COLAB_GPU", "0") != "1"
        and not any(os.environ.get(key) for key in ("COLAB_TPU_ADDR", "TPU_NAME", "COLAB_TPU_TASK_ID")),
        "frozen_cpu_loading_runtime_required")
    actual_colab = execution_environment == "google_colab_cpu" and bool(os.environ.get("COLAB_RELEASE_TAG"))
    require(execution_environment != "google_colab_cpu" or actual_colab,
        "google_colab_runtime_evidence_required")
    return actual_colab


def _reference(value):
    require(isinstance(value, dict) and value.get("version") == core.VERSION
        and value.get("status") == "development_inputs_and_cpu_loading_verified",
        "approved_local_model_input_reference_required")
    false_flags = ("validation_or_target_used_for_fitting", "predictive_model_training_executed",
        "optimizer_created", "backward_executed", "reserved_test_opened", "originals_modified",
        "partitions_modified", "pretrained_weights_loaded", "additional_resize_crop_flip_or_augmentation")
    true_flags = ("clinical_preprocessing_fitted", "aligned_multimodal_input_pairs_verified",
        "source_files_unchanged", "development_rows_and_pixels_unchanged")
    require(all(value.get(key) is False for key in false_flags)
        and all(value.get(key) is True for key in true_flags)
        and value.get("clinical_features") == POLICY["clinical_features"]
        and value.get("clinical_states_fitted") == len(POLICY["folds"]), "bounded_reference_scope_required")
    require(type(value.get("development_knees")) is int and type(value.get("development_participants")) is int
        and 0 < value["development_participants"] <= value["development_knees"] <= 2 * value["development_participants"],
        "reference_membership_counts_required")
    require(isinstance(value.get("sources_sha256"), dict)
        and all(valid_sha(value["sources_sha256"].get(key)) for key in ("development_package", "handoff", "policy", "registry"))
        and isinstance(value.get("implementation_sha256"), dict)
        and set(value["implementation_sha256"]) == set(SCIENTIFIC_FILES)
        and all(valid_sha(v) for v in value["implementation_sha256"].values()), "reference_source_hashes_required")
    require(isinstance(value.get("folds"), list) and len(value["folds"]) == len(POLICY["folds"])
        and all(isinstance(r, dict) for r in value["folds"])
        and [r.get("validation_fold") for r in value["folds"]] == POLICY["folds"]
        and isinstance(value.get("image_loading"), dict) and set(value["image_loading"]) == set(POLICY["image_encoders"])
        and all(isinstance(v, list) and all(isinstance(r, dict) for r in v)
            and [r.get("validation_fold") for r in v] == POLICY["folds"]
            for v in value["image_loading"].values())
        and isinstance(value.get("clinical_artifacts_sha256"), dict)
        and len(value["clinical_artifacts_sha256"]) == len(POLICY["folds"]) * 5,
        "complete_five_fold_reference_required")


def run(package_dir, reference_record, repository_root, output_dir, execution_environment):
    """Verify every development loader and bounded eval forward without training."""
    started = time.monotonic()
    actual_colab = _runtime(execution_environment)
    package, reference_path, repository, output = map(checked_path,
        (package_dir, reference_record, repository_root, output_dir))
    require(package.is_dir() and repository.is_dir() and reference_path.is_file(), "existing_preflight_input_paths_required")
    protected_roots = [package, repository]
    require(all(not output.is_relative_to(root) and not root.is_relative_to(output) for root in protected_roots),
        "separate_preflight_output_directory_required")
    require(not reference_path.is_relative_to(output), "preflight_output_must_not_contain_reference_record")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
        "new_empty_preflight_output_required")
    reference = _read_json(reference_path)
    _reference(reference)
    sources = {"reference_record": reference_path, "handoff": checked_path(package / "handoff.json"),
        "development_package": checked_path(package / "development_package.zip")}
    for name, relative in SCIENTIFIC_FILES.items():
        path = checked_path(repository / relative)
        require(path.is_file() and sha_file(path) == reference["implementation_sha256"][name],
            "reference_scientific_implementation_changed")
        sources["implementation/" + relative] = path
    require(checked_path(core.__file__) == repository / "src/knee/model_inputs_preflight.py"
        and checked_path(core.transform_clinical_rows.__code__.co_filename) == repository / "src/knee/clinical_fold_inputs.py"
        and checked_path(core.DevelopmentFoldDataset.__init__.__code__.co_filename) == repository / "src/knee/model_input_adapters.py"
        and checked_path(load_development_package.__code__.co_filename) == repository / "src/knee/colab_data.py"
        and checked_path(__file__) == repository / "src/knee/model_inputs_colab.py",
        "executed_preflight_modules_must_match_checkout")
    sources["implementation/src/knee/model_inputs_colab.py"] = checked_path(__file__)
    cli = checked_path(repository / "scripts/check_model_inputs_colab.py")
    require(cli.is_file(), "colab_preflight_cli_required")
    sources["implementation/scripts/check_model_inputs_colab.py"] = cli
    before = {name: sha_file(path) for name, path in sources.items()}
    for key in ("development_package", "handoff"):
        require(before[key] == reference["sources_sha256"][key], "reference_development_package_or_handoff_changed")
    policy = _read_json(repository / "configs/model_inputs_preflight_v1.json")
    registry = _read_json(repository / "configs/model_registry.json")
    validate_policy(policy, registry)
    for key, relative in (("policy", "configs/model_inputs_preflight_v1.json"), ("registry", "configs/model_registry.json")):
        require(before["implementation/" + relative] == reference["sources_sha256"][key], "reference_policy_or_registry_changed")
    handoff = _read_json(sources["handoff"])
    require(handoff.get("version") == "participant_split_colab_preparation_v1.0"
        and handoff.get("training_executed") is False and handoff.get("reserved_test_predictive_evaluation_opened") is False
        and handoff.get("development", {}).get("file") == "development_package.zip",
        "sealed_development_handoff_required")
    development = handoff["development"]
    require(development.get("sha256") == before["development_package"]
        and development.get("bytes") == sources["development_package"].stat().st_size,
        "sealed_development_package_size_or_hash_changed")
    loaded = load_development_package(sources["development_package"], development["sha256"])
    rows, tensors = loaded["manifest"]["rows"], loaded["tensors"]
    require(loaded["manifest"]["bindings"]["partition_sha256"] == handoff.get("partition_sha256")
        and loaded["manifest"]["bindings"]["split_policy_sha256"] == handoff.get("split_policy_sha256")
        and len(rows) == development.get("knees") == reference["development_knees"]
        and len({r["src_subject_id"] for r in rows}) == development.get("participants") == reference["development_participants"],
        "exact_reference_development_membership_and_partition_required")
    core.torch.set_num_threads(min(policy["maximum_cpu_threads"], os.cpu_count() or 1))
    rows_before, tensors_before = digest(rows), _array_digest(tensors)
    states = prepare_fold_states(rows, policy)
    records, artifacts = [], {}
    for fold, state in enumerate(states):
        require(state["record_sha256"] == reference["folds"][fold]["clinical_state_sha256"],
            "reference_training_only_clinical_state_mismatch")
        record = {"validation_fold": fold, "clinical_state_sha256": state["record_sha256"], "splits": {}}
        state_path = output / "clinical" / f"fold_{fold}_state.json"
        _save(state_path, canonical_json(state))
        artifacts[state_path.relative_to(output).as_posix()] = sha_file(state_path)
        for split in ("training", "validation"):
            selected = [row for row in rows if (row["cv_fold"] == fold) == (split == "validation")]
            identities = [[r["src_subject_id"], r["side"]] for r in selected]
            clinical = core.transform_clinical_rows(selected, state)
            prefix = output / "clinical" / f"fold_{fold}_{split}"
            array_path, order_path = Path(str(prefix) + ".npy"), Path(str(prefix) + "_order.json")
            _save_array(array_path, clinical)
            _save(order_path, canonical_json(identities))
            for path in (array_path, order_path):
                artifacts[path.relative_to(output).as_posix()] = sha_file(path)
            record["splits"][split] = {"rows": len(selected), "participants": len({r["src_subject_id"] for r in selected}),
                "imputed_BMI_rows": sum(r["BMI"] is None for r in selected),
                "ordered_identity_sha256": digest(identities), "clinical_sha256": hashlib.sha256(clinical.tobytes(order="C")).hexdigest()}
        require(record == reference["folds"][fold], "reference_clinical_fold_record_mismatch")
        records.append(record)
    require(artifacts == reference["clinical_artifacts_sha256"], "reference_clinical_artifact_bytes_mismatch")
    loaders, probes = {}, {}
    for encoder in policy["image_encoders"]:
        samples, fold_results = [], []
        for fold, state in enumerate(states):
            result = {"validation_fold": fold, "splits": {}}
            for split in ("training", "validation"):
                dataset = core.DevelopmentFoldDataset(rows, tensors, state, encoder, split)
                summary = check_loader(dataset, policy["loader"]["batch_size"])
                require(summary == reference["image_loading"][encoder][fold]["splits"][split],
                    "reference_complete_loader_input_hashes_mismatch")
                result["splits"][split] = summary
                if split == "validation":
                    samples.append(core.torch.stack([dataset[i]["image"] for i in range(min(2, len(dataset)))]))
            fold_results.append(result)
            print(f"Entradas CPU: {encoder}, pliegue {fold + 1}/{len(states)} verificado.", flush=True)
        loaders[encoder] = fold_results
        # This core asserts finite expected feature widths and within-run state
        # identity; no random feature/model hash is compared with another host.
        probes[encoder] = architecture_probe(encoder, samples)
        print(f"Arquitectura {encoder}: compatibilidad comprobada sin entrenamiento.", flush=True)
    require(digest(rows) == rows_before and _array_digest(tensors) == tensors_before,
        "development_rows_or_pixels_modified")
    require(all(sha_file(path) == before[name] for name, path in sources.items()), "preflight_source_modified")
    visits = sum(s["rows"] for values in loaders.values() for fold in values for s in fold["splits"].values())
    require(visits == reference["total_loader_row_visits"], "reference_loader_visit_count_mismatch")
    summary = {"version": VERSION, "status": "development_inputs_and_cpu_loading_verified",
        "execution_environment": execution_environment, "colab_runtime_executed": actual_colab,
        "local_rehearsal": execution_environment == "local_rehearsal", "cpu_only_guard_passed": True,
        "reference_matches_training_only_states_and_all_loader_hashes": True,
        "reference_record_sha256": before["reference_record"], "development_package_sha256": before["development_package"],
        "partition_sha256": handoff["partition_sha256"], "split_policy_sha256": handoff["split_policy_sha256"],
        "development_participants": development["participants"], "development_knees": len(rows),
        "clinical_features": policy["clinical_features"], "clinical_states_fitted": len(states),
        "clinical_preprocessing_fitted": True, "BMI_imputation_applied_to_copies": True,
        "missing_BMI_development_knees_retained": sum(r["BMI"] is None for r in rows),
        "statistics_unit": "training_knee_rows", "validation_or_target_used_for_fitting": False,
        "aligned_multimodal_input_pairs_verified": True, "folds": records, "image_loading": loaders,
        "architecture_probes": probes, "architecture_hashes_compared_across_platforms": False,
        "total_loader_row_visits": visits, "loader_batch_size": policy["loader"]["batch_size"],
        "clinical_artifacts_sha256": artifacts, "sources_sha256": before,
        "source_files_unchanged": True, "development_rows_and_pixels_unchanged": True,
        "training_executed": False, "predictive_model_training_executed": False,
        "optimizer_created": False, "backward_executed": False, "reserved_test_opened": False,
        "reserved_test_predictive_evaluation_opened": False, "originals_modified": False, "partitions_modified": False,
        "pretrained_weights_loaded": False, "pretrained_weight_eligibility_certified": False,
        "new_image_files_written": 0, "additional_resize_crop_flip_or_augmentation": False,
        "environment": {"python": platform.python_version(), "numpy": core.np.__version__,
            "torch": core.torch.__version__, "torchvision": core.torchvision.__version__,
            "torch_threads": core.torch.get_num_threads()},
        "limitations": ["Random-weight eval probes certify input compatibility only; pretrained weights were not loaded.",
            "No predictor training or reserved-test access occurred; external pretraining overlap remains unresolved."],
        "next_step": "review_pretrained_checkpoint_eligibility_and_freeze_training_configuration_before_any_training"}
    _save(output / "closure_summary.json", canonical_json(summary))
    print(json.dumps({key: summary[key] for key in ("version", "status", "execution_environment", "colab_runtime_executed",
        "development_participants", "development_knees", "clinical_states_fitted", "total_loader_row_visits",
        "reference_matches_training_only_states_and_all_loader_hashes", "training_executed", "reserved_test_opened")}, indent=2), flush=True)
    print(f"Verificación completada en {time.monotonic() - started:.1f} segundos.", flush=True)
    return summary
