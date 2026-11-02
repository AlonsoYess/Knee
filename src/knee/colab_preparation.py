"""Seal grouped partitions and portable private inputs, without any model fitting."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

from knee.participant_partitions import POLICY, build_partitions, canonical, digest, identity, require


VERSION = "participant_split_colab_preparation_v1.0"
SOURCE_NAMES = {"cohort_closure", "membership", "labels", "development_manifest", "development_closure",
                "routine_manifest", "routine_closure", "routine_verification", "split_policy"}
CODE_FILES = ("src/knee/__init__.py", "src/knee/colab_data.py", "src/knee/colab_data_package.py",
              "src/knee/participant_partitions.py", "scripts/verify_colab_data.py",
              "configs/participant_split_v1.json", "requirements-colab-data.txt")
PINNED_COMPONENTS = set(CODE_FILES) | {"src/knee/colab_preparation.py", "scripts/prepare_colab_data.py",
    "notebooks/31_verificar_paquete_desarrollo_sin_entrenar.ipynb"}
ALLOWED_ACTIONS = ["participant_level_partition", "fixed_outcome_stratification", "class_presence_check",
                   "private_development_package", "separate_reserved_test_packaging", "cpu_package_verification",
                   "colab_preflight_notebook"]
FORBIDDEN_ACTIONS = ["training", "original_edit", "qc_regrading", "imputation_fit", "model_changes",
                     "reserved_test_predictive_evaluation", "drive_upload", "data_deletion", "new_selection"]


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path, *, directory=False):
    raw = Path(path)
    require(raw.is_absolute(), "absolute_private_path_required")
    for ancestor in [raw] + list(raw.parents):
        require(not ancestor.is_symlink() and not getattr(ancestor, "is_junction", lambda: False)(),
                "linked_private_path_forbidden")
    require(raw.is_dir() if directory else raw.is_file(), "private_path_missing")
    return raw.resolve()


def safe(root, relative):
    require(isinstance(relative, str) and "\\" not in relative and ":" not in relative
            and not relative.startswith("/") and all(p not in {"", ".", ".."} for p in relative.split("/")),
            "safe_relative_path_required")
    target = root.joinpath(*relative.split("/"))
    for ancestor in [target] + list(target.parents):
        require(not ancestor.is_symlink() and not getattr(ancestor, "is_junction", lambda: False)(),
                "linked_file_forbidden")
    require(target.resolve().is_relative_to(root), "path_escape")
    return target


def write_bytes_once(path, payload):
    require(not path.is_symlink() and not getattr(path, "is_junction", lambda: False)(), "linked_output_forbidden")
    if path.exists():
        require(path.read_bytes() == payload, "immutable_output_changed")
        return True
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == payload, "output_readback_failed")
    return False


def write_json(path, value):
    return write_bytes_once(path, canonical(value) + b"\n")


def read_pinned(spec):
    require(isinstance(spec, dict) and set(spec) == {"path", "sha256"}, "exact_pinned_source_required")
    path = checked(spec["path"])
    require(sha_file(path) == spec["sha256"], "source_hash_changed")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def code_bundle(project, output):
    data, hashes = io.BytesIO(), {}
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_STORED) as archive:
        for relative in CODE_FILES:
            payload = safe(project, relative).read_bytes()
            item = zipfile.ZipInfo(relative, date_time=(2000, 1, 1, 0, 0, 0))
            item.compress_type, item.create_system, item.external_attr = zipfile.ZIP_STORED, 3, 0o100644 << 16
            archive.writestr(item, payload)
            hashes[relative] = hashlib.sha256(payload).hexdigest()
    target = output / "colab_preparation_code.zip"
    write_bytes_once(target, data.getvalue())
    return {"file": target.name, "sha256": sha_file(target), "files_sha256": hashes}


def run(config_path):
    from knee.colab_data_package import build_packages
    from knee.colab_data import verify_development_package

    config_path = checked(config_path)
    config_sha = sha_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(set(config) == {"version", "project_root", "sources", "authority", "component_pins", "output_root"}
            and config["version"] == VERSION and set(config["sources"]) == SOURCE_NAMES,
            "exact_preparation_config_required")
    project = checked(config["project_root"], directory=True)
    require(checked(__file__) == project / "src/knee/colab_preparation.py", "configured_project_mismatch")
    require(set(config["component_pins"]) == PINNED_COMPONENTS, "complete_preparation_component_pins_required")
    for relative, sha in config["component_pins"].items():
        require(sha_file(safe(project, relative)) == sha, "preparation_component_changed")
    authority = read_pinned(config["authority"])
    require(authority.get("version") == VERSION and authority.get("approved_by") == "researcher"
            and authority.get("user_statement") and authority.get("allowed_actions") == ALLOWED_ACTIONS
            and authority.get("forbidden_actions") == FORBIDDEN_ACTIONS,
            "separate_partition_preparation_authority_required")
    data = {name: read_pinned(spec) for name, spec in config["sources"].items()}
    require(canonical(data["split_policy"]) == canonical(POLICY), "adopted_split_policy_changed")
    cohort, members, labels = data["cohort_closure"], data["membership"], data["labels"]
    require(cohort.get("status") == "common_cohort_frozen_input_tensor_and_splits_pending"
            and labels.get("membership_sha256") == digest(members)
            and cohort["files_sha256"]["membership_without_outcomes.json"] == config["sources"]["membership"]["sha256"]
            and cohort["files_sha256"]["outcomes.json"] == config["sources"]["labels"]["sha256"],
            "sealed_common_cohort_required")
    tensors, source_paths = {}, {}
    contract_hash = None
    for prefix in ("development", "routine"):
        manifest, closure = data[prefix + "_manifest"], data[prefix + "_closure"]
        require(closure.get("manifest_sha256") == config["sources"][prefix + "_manifest"]["sha256"]
                and closure.get("training_executed") is False and closure.get("partitions_created") is False
                and manifest["tensor_contract_version"] == "physical_letterbox_224_p01_p99_v1.0",
                "sealed_image_input_closure_required")
        if contract_hash is None:
            contract_hash = manifest["tensor_contract_sha256"]
        require(contract_hash == manifest["tensor_contract_sha256"], "image_transform_contract_mismatch")
        root = checked(config["sources"][prefix + "_manifest"]["path"]).parent
        for row in manifest["rows"]:
            pair = identity(row)
            require(pair not in tensors and row["role"] == prefix, "duplicate_or_wrong_image_input_role")
            tensor = row["tensor"]
            require(set(tensor) == {"relative_path", "sha256", "pixel_sha256", "shape", "dtype"}
                    and tensor["shape"] == [1, 224, 224] and tensor["dtype"] == "float32",
                    "sealed_image_input_descriptor_required")
            source = safe(root, tensor["relative_path"])
            require(sha_file(source) == tensor["sha256"], "derived_input_source_changed")
            tensors[pair] = row
            source_paths[pair] = source
    require(data["routine_verification"].get("status") == "passed"
            and data["routine_verification"].get("closure_summary_sha256") == config["sources"]["routine_closure"]["sha256"]
            and data["routine_verification"].get("development_and_routine_inputs_disjoint_and_complete") is True,
            "independent_image_verification_required")
    require(set(tensors) == {identity(row) for row in members["rows"]}, "complete_2161_common_image_join_required")
    output = Path(config["output_root"])
    require(output.is_absolute(), "absolute_output_root_required")
    for ancestor in [output] + list(output.parents):
        require(not ancestor.is_symlink() and not getattr(ancestor, "is_junction", lambda: False)(), "linked_output_root_forbidden")
    for spec in list(config["sources"].values()) + [config["authority"], {"path": str(config_path)}]:
        source = checked(spec["path"])
        require(not output.resolve().is_relative_to(source.parent)
                and not source.is_relative_to(output.resolve()), "output_overlaps_sealed_input")
    require(not output.resolve().is_relative_to(project) and not project.is_relative_to(output.resolve()),
            "private_package_must_remain_outside_public_repository")
    output.mkdir(parents=True, exist_ok=True)
    # Persist the unique adopted rule before deriving any participant assignments.
    write_json(output / "adopted_split_policy.json", POLICY)
    partitions = build_partitions(members, labels, policy=data["split_policy"])
    partitions["source_files_sha256"] = {name: spec["sha256"] for name, spec in config["sources"].items()}
    partition_path = output / "participant_partitions.json"
    reused_partitions = write_json(partition_path, partitions)
    assignments = {row["src_subject_id"]: row for row in partitions["participants"]}
    outcomes = {identity(row): row["PROGRESION_48M"] for row in labels["rows"]}
    rows, tensor_sources = [], {}
    for index, member in enumerate(sorted(members["rows"], key=identity), 1):
        pair = identity(member)
        assignment = assignments[pair[0]]
        relative = f"tensors/knee_{index:06d}.npy"
        tensor = dict(tensors[pair]["tensor"], relative_path=relative)
        require(member["role"] == tensors[pair]["role"] == assignment["preprocessing_role"],
                "cohort_image_assignment_role_mismatch")
        rows.append({"src_subject_id": pair[0], "side": pair[1],
            "preprocessing_role": assignment["preprocessing_role"],
            "outer_block": assignment["outer_block"], "cv_fold": assignment["cv_fold"],
            **member["clinical"], "PROGRESION_48M": outcomes[pair], "tensor": tensor})
        tensor_sources[relative] = source_paths[pair]
    bindings = {"partition_sha256": sha_file(partition_path), "split_policy_sha256": digest(POLICY),
        "cohort_closure_sha256": config["sources"]["cohort_closure"]["sha256"],
        "tensor_contract_sha256": contract_hash,
        "development_closure_sha256": config["sources"]["development_closure"]["sha256"],
        "routine_closure_sha256": config["sources"]["routine_closure"]["sha256"]}
    packages = build_packages(output, rows, tensor_sources, bindings)
    code = code_bundle(project, output)
    handoff = {"version": VERSION, "development": {}, "reserved_test": {}, "code": code,
        "partition_sha256": bindings["partition_sha256"], "split_policy_sha256": digest(POLICY),
        "training_executed": False, "reserved_test_predictive_evaluation_opened": False,
        "reserved_test_access_policy": "separate_archive_never_opened_by_development_notebook",
        "imputation": "BMI_median_fit_inside_each_training_fold_only_not_fitted_yet",
        "model_specific_rgb_and_normalization": "pending_selected_encoder_integration",
        "external_pretraining_disjointness_verified": False}
    for kind in ("development", "reserved_test"):
        info = packages["packages"][kind]
        handoff[kind] = {"file": Path(info["path"]).name, **{k: info[k] for k in ("sha256", "bytes", "participants", "knees")}}
    write_json(output / "handoff.json", handoff)
    notebook = "notebooks/31_verificar_paquete_desarrollo_sin_entrenar.ipynb"
    write_bytes_once(output / Path(notebook).name, safe(project, notebook).read_bytes())
    verification = verify_development_package(output / handoff["development"]["file"],
                                             handoff["development"]["sha256"])
    write_json(output / "development_package_verification.json", verification)
    # End-to-end source integrity includes copied derived tensors; originals are never opened for writing.
    for spec in config["sources"].values():
        require(sha_file(checked(spec["path"])) == spec["sha256"], "source_changed_during_preparation")
    for pair, path in source_paths.items():
        require(sha_file(path) == tensors[pair]["tensor"]["sha256"], "image_input_changed_during_packaging")
    for relative, sha in config["component_pins"].items():
        require(sha_file(safe(project, relative)) == sha, "component_changed_during_preparation")
    require(sha_file(config_path) == config_sha, "config_changed_during_preparation")
    result = {"version": VERSION, "status": "participant_partitions_and_private_colab_packages_verified",
        "partition_summary": partitions["summary"], "packages": handoff,
        "participant_partitions_sha256": sha_file(partition_path),
        "handoff_sha256": sha_file(output / "handoff.json"),
        "development_package_verification_sha256": sha_file(output / "development_package_verification.json"),
        "source_files_unchanged": True, "originals_modified": False, "cohort_membership_changed": False,
        "same_cohort_for_three_main_models": True, "outcomes_used_for_stratification_only": True,
        "imputation_performed": False, "model_specific_normalization_applied": False,
        "training_executed": False, "reserved_test_predictive_evaluation_opened": False,
        "reserved_test_used_for_model_development": False, "drive_upload_performed": False,
        "external_pretraining_disjointness_verified": False, "statistical_precision_or_power_certified": False,
        "limitations": ["Technical assisted image QC is not independent clinical validation.",
            "External KNEEL/OAI pretraining overlap remains unresolved.",
            "Reserved test archive is logically separated and hashed, not encrypted.",
            "RGB and pretrained-encoder normalization remain pending; no model was fitted.",
            "Colab preparation was verified locally; no remote Colab execution is claimed."],
        "next_step": "private_drive_publication_then_colab_cpu_package_preflight_before_training_authority"}
    reused_closure = write_json(output / "closure_summary.json", result)
    return {"status": result["status"], "partition_summary": partitions["summary"],
        "packages": {kind: handoff[kind] for kind in ("development", "reserved_test")},
        "existing_partitions_reused": reused_partitions, "existing_closure_reused": reused_closure,
        "training_executed": False, "drive_upload_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
