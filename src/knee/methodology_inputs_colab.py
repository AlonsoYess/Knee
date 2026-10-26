"""Notebook38: baseline fold inputs and image transforms, never model fitting.

Only the previously authenticated development package and an independently
pinned baseline-only development projection are accepted. The image review is
a deterministic software/pixel check, not anatomical or predictive validation.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

import numpy as np

from knee import methodology_preparation as preparation
from knee import methodology_release as release
from knee import training_preparation_colab as runtime_base
from knee import training_smoke_colab as transport
from knee.colab_data import checked_path, require, sha_file, valid_sha
from knee.training_plan import _digest
from knee.training_preparation import read_json

VERSION = "methodology_inputs_colab_orchestration_v1.0"
CORE_VERSION = "development_methodology_inputs_preflight_v1.0"
CORE_STATUS = "development_baseline_and_augmentation_verified_no_training"
CORE_FILENAME = "methodology_inputs_preflight.json"
SUMMARY_FILENAME = "colab_methodology_inputs_check.json"
PROJECTION_FILENAME = "development_baseline_projection_v1.json"
DESCRIPTOR_FILENAME = "development_baseline_descriptor_v1.json"
DESCRIPTOR_VERSION = "development_baseline_projection_transport_v1.0"
PREVIEW_FILES = tuple(f"augmentation_previews/comparison_{index:02}.png" for index in range(1, 4))
IMAGE_MODELS = ("image_densenet121", "image_vit_b16", "image_convnextv2_tiny", "image_skelex",
                "image_biomedclip", "image_mambaout_tiny")
TEST_MODULES = ("test_baseline_development_projection", "test_clinical_scenarios",
    "test_methodology_source_line_endings", "test_methodology_portable_release",
    "test_methodology_release", "test_training_augmentation", "test_methodology_preparation",
    "test_methodology_clinical_integration", "test_methodology_augmentation_integration",
    "test_methodology_inputs_colab", "test_methodology_augmentation_preflight",
    "test_methodology_inputs_notebook")
SOURCE_FILES = ("src/knee/methodology_inputs_colab.py", "scripts/check_methodology_inputs_colab.py",
    "src/knee/baseline_development_projection.py", "src/knee/methodology_preparation.py",
    "src/knee/clinical_scenarios.py", "src/knee/training_augmentation.py",
    "src/knee/methodology_release.py", "src/knee/aligned_training/data.py",
    "src/knee/aligned_training/models.py", "src/knee/aligned_training/authority.py")
FALSE_FLAGS = ("training_executed", "optimizer_created", "backward_executed",
    "training_authorized", "pretrained_weights_loaded", "pretrained_weights_downloaded",
    "reserved_test_predictive_evaluation_opened", "originals_modified", "partitions_modified",
    "new_studies_selected", "predictive_performance_claimed", "clinical_validation_claimed")
MAX_PROJECTION_BYTES = 2 * 1024**2
MAX_OUTPUT_BYTES = 32 * 1024**2
MAX_OUTPUT_FILE_BYTES = 4 * 1024**2
runtime_environment = runtime_base.runtime_environment
setup_runtime = runtime_base.setup_runtime
stage_development = transport.stage_development
_save = runtime_base._save
_json = read_json


def _descriptor(value):
    fields = {"version", "scope", "baseline_visit", "development_package_sha256",
        "development_participants", "development_knees", "baseline_source_sha256", "projection", "record_sha256"}
    require(isinstance(value, dict) and set(value) == fields and value["version"] == DESCRIPTOR_VERSION
        and value["scope"] == "development_only" and value["baseline_visit"] == "V00",
        "exact_baseline_development_descriptor_required")
    require(all(valid_sha(value[name]) for name in ("development_package_sha256", "baseline_source_sha256", "record_sha256"))
        and value["record_sha256"] == _digest({k: v for k, v in value.items() if k != "record_sha256"}),
        "baseline_descriptor_seal_changed")
    require(type(value["development_participants"]) is int and type(value["development_knees"]) is int
        and 1 <= value["development_participants"] <= value["development_knees"] <= 5000,
        "bounded_development_descriptor_counts_required")
    projection = value["projection"]
    require(isinstance(projection, dict) and set(projection) == {"file", "sha256", "bytes"}
        and projection["file"] == PROJECTION_FILENAME and valid_sha(projection["sha256"])
        and type(projection["bytes"]) is int and 0 < projection["bytes"] <= MAX_PROJECTION_BYTES,
        "exact_bounded_baseline_projection_required")
    return value


def make_baseline_descriptor(projection_path, *, package_sha256, participants, knees, source_sha256):
    path = checked_path(projection_path)
    require(path.is_file() and path.name == PROJECTION_FILENAME, "canonical_baseline_projection_filename_required")
    value = {"version": DESCRIPTOR_VERSION, "scope": "development_only", "baseline_visit": "V00",
        "development_package_sha256": package_sha256, "development_participants": participants,
        "development_knees": knees, "baseline_source_sha256": source_sha256,
        "projection": {"file": PROJECTION_FILENAME, "sha256": sha_file(path), "bytes": path.stat().st_size}}
    value["record_sha256"] = _digest(value)
    return _descriptor(value)


def stage_baseline(remote_root, local_root):
    """Copy only the descriptor and the two-way verified development projection."""
    remote, local = map(checked_path, (remote_root, local_root))
    require(remote.is_dir() and not local.exists() and local.parent.is_dir()
        and not local.is_relative_to(remote) and not remote.is_relative_to(local),
        "new_separate_baseline_stage_required")
    descriptor_path = checked_path(remote / DESCRIPTOR_FILENAME)
    descriptor_hash = sha_file(descriptor_path)
    descriptor = _descriptor(read_json(descriptor_path))
    projection_path = checked_path(remote / descriptor["projection"]["file"])
    spec = descriptor["projection"]
    require(projection_path.is_file() and projection_path.stat().st_size == spec["bytes"]
        and sha_file(projection_path) == spec["sha256"], "baseline_projection_transport_mismatch")
    local.mkdir()
    for source in (descriptor_path, projection_path):
        destination = local / source.name
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024**2)
        require(sha_file(source) == sha_file(destination), "baseline_projection_stage_readback_failed")
    require(sha_file(descriptor_path) == descriptor_hash
        and sha_file(projection_path) == spec["sha256"], "baseline_source_changed_during_stage")
    return {"descriptor_path": str(local / DESCRIPTOR_FILENAME), "projection_path": str(local / PROJECTION_FILENAME),
        "descriptor_sha256": descriptor_hash, "projection_sha256": spec["sha256"],
        "development_only": True, "training_executed": False, "reserved_test_opened": False}


def _cpu_environment(execution_environment):
    runtime_base.previous._cpu_host(execution_environment)
    import torch
    import torchvision
    import platform
    require(torch.cuda.is_available() is False and torch.version.cuda is None, "cpu_only_pytorch_build_required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "preflight_requires_hidden_cuda")
    require(np.__version__ == "1.26.4" and str(torch.__version__) == "2.6.0+cpu"
        and str(torchvision.__version__) == "0.21.0+cpu" and platform.python_version() == "3.12.14",
        "frozen_methodology_cpu_runtime_required")
    return {"execution_environment": execution_environment,
        "colab_runtime_executed": execution_environment == "google_colab_cpu",
        "local_rehearsal": execution_environment == "local_cpu", "cpu_only_guard_passed": True,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__)}}


def _pixels(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _verified_text_proofs(proofs):
    """Check the loader's bounded, pinned LF/CRLF equivalence receipts."""
    from knee.model_inputs_colab import SCIENTIFIC_FILES
    expected = {"implementation_sha256/" + name: relative for name, relative in SCIENTIFIC_FILES.items()}
    expected.update({"sources_sha256/policy": "configs/model_inputs_preflight_v1.json",
                     "sources_sha256/registry": "configs/model_registry.json"})
    require(isinstance(proofs, list) and len(proofs) == len(expected)
        and {value.get("reference_field") for value in proofs if isinstance(value, dict)} == set(expected),
        "complete_pinned_source_text_proofs_required")
    for value in proofs:
        require(value.get("relative_name") == expected[value["reference_field"]]
            and valid_sha(value.get("expected_sha256")) and valid_sha(value.get("actual_sha256"))
            and value.get("matched_variant") in {"exact_bytes", "lf_bytes", "crlf_bytes"}
            and (value["actual_sha256"] == value["expected_sha256"]) is (value["matched_variant"] == "exact_bytes")
            and value.get("line_endings_only_equivalence") is (value["matched_variant"] != "exact_bytes"),
            "pinned_source_text_proof_invalid")
    return True


def _preview(source, transformed, destination, *, index, gamma, applied):
    from PIL import Image, ImageDraw
    left = Image.fromarray(np.rint(source[0] * 255).astype(np.uint8), mode="L").convert("RGB")
    right = Image.fromarray(np.rint(transformed[0] * 255).astype(np.uint8), mode="L").convert("RGB")
    canvas = Image.new("RGB", (480, 285), "white")
    canvas.paste(left, (8, 42))
    canvas.paste(right, (248, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"Desarrollo: muestra {index}. Vista de software, sin diagnostico.", fill="black")
    draw.text((8, 25), "Original", fill="black")
    draw.text((248, 25), f"Gamma={gamma:.5f}; aplicado={applied}", fill="black")
    draw.text((8, 268), "Copias de visualizacion. Se conservan geometria y lateralidad.", fill="black")
    destination.parent.mkdir(exist_ok=True)
    with destination.open("xb") as stream:
        canvas.save(stream, format="PNG")
    require(destination.is_file() and destination.stat().st_size > 0, "augmentation_preview_missing")


def _select_preview_keys(rows, policy, seed):
    """Illustrate existing draws without inspecting targets or image appearance.

    Choose the first canonical fold-zero training key for gamma below one,
    gamma above one, and an unapplied control. If a category is absent, fill the
    remaining slots with distinct canonical training keys. No draw is changed.
    """
    from knee.training_augmentation import sample_parameters
    eligible = sorted((row["src_subject_id"], row["side"]) for row in rows if row["cv_fold"] != 0)
    require(len(eligible) >= 3 and len(eligible) == len(set(eligible)),
        "three_distinct_development_training_preview_samples_required")
    realized = {key: sample_parameters(seed=seed, validation_fold=0, epoch=1,
        src_subject_id=key[0], side=key[1], policy=policy) for key in eligible}
    predicates = (lambda value: value["applied"] and value["gamma"] < 1,
                  lambda value: value["applied"] and value["gamma"] > 1,
                  lambda value: not value["applied"])
    selected = []
    for predicate in predicates:
        candidate = next((key for key in eligible if key not in selected and predicate(realized[key])), None)
        if candidate is not None:
            selected.append(candidate)
    selected.extend(key for key in eligible if key not in selected)
    return selected[:3]


def verify_augmentation(rows, tensors, plan, package_sha256, output_dir):
    """Exercise all development training/validation positions for epoch one."""
    import torch
    from knee.training_augmentation import augment_training_image, apply_image_policy, sample_parameters, validate_policy
    from knee.aligned_training.data import prepare_fold
    from knee.aligned_training.models import normalize_for_model
    from knee.colab_data import validate_rows
    validate_rows(rows, "development")
    expected = {(row["src_subject_id"], row["side"]) for row in rows}
    require(isinstance(tensors, dict) and set(tensors) == expected, "exact_development_tensor_mapping_required")
    policy = plan["neural_training"]["augmentation"]
    validate_policy(policy)
    seed = plan["selection"]["screening_seed"]
    before = {key: _pixels(value) for key, value in tensors.items()}
    output = checked_path(output_dir)
    require(output.is_dir(), "existing_private_augmentation_output_required")
    training_visits = validation_visits = changed = applied_count = normalized = 0
    folds, preview_records = [], []
    preview_keys = _select_preview_keys(rows, policy, seed)
    require(len(preview_keys) == 3, "three_development_training_preview_samples_required")
    for fold in range(5):
        fold_training = fold_validation = fold_changed = 0
        for row in rows:
            key = row["src_subject_id"], row["side"]
            source = tensors[key]
            if row["cv_fold"] == fold:
                original = apply_image_policy(source, split="validation", policy=policy)
                require(original is source and _pixels(original) == before[key], "validation_image_policy_must_be_identity")
                validation_visits += 1
                fold_validation += 1
                continue
            kwargs = dict(seed=seed, validation_fold=fold, epoch=1, src_subject_id=key[0], side=key[1], policy=policy)
            transformed = augment_training_image(source, **kwargs)
            repeated = augment_training_image(source, **kwargs)
            params = sample_parameters(**kwargs)
            require(np.array_equal(transformed, repeated) and transformed.shape == source.shape
                and np.array_equal(transformed == 0, source == 0) and not transformed.flags.writeable,
                "deterministic_anatomy_preserving_augmentation_required")
            pixel_changed = not np.array_equal(source, transformed)
            require(params["applied"] or not pixel_changed, "identity_draw_changed_image")
            changed += int(pixel_changed)
            applied_count += int(params["applied"])
            training_visits += 1
            fold_training += 1
            fold_changed += int(pixel_changed)
            if fold == 0 and key in preview_keys:
                index = preview_keys.index(key)
                relative = PREVIEW_FILES[index]
                _preview(source, transformed, output / relative, index=index + 1,
                    gamma=params["gamma"], applied=params["applied"])
                preview_records.append({"sample": index + 1, "validation_fold": 0, "epoch": 1,
                    "file": relative, "source_pixel_sha256": before[key], "transformed_pixel_sha256": _pixels(transformed),
                    "gamma": params["gamma"], "applied": params["applied"],
                    "draw_category": ("unapplied_control" if not params["applied"] else
                        "applied_gamma_below_one" if params["gamma"] < 1 else
                        "applied_gamma_above_one" if params["gamma"] > 1 else "applied_gamma_equal_one")})
        # Use actual dataset entrypoints for a bounded pair per split and encoder.
        for encoder in IMAGE_MODELS:
            prepared = prepare_fold(rows, tensors, fold, encoder_id=encoder, package_sha256=package_sha256,
                augmentation_policy=policy, seed=seed, methodology=True)
            prepared.training.set_epoch(1)
            for dataset in (prepared.training, prepared.validation):
                for index in range(min(2, len(dataset))):
                    key = dataset.identities[index]
                    value = tensors[key]
                    if dataset.split == "training":
                        value = augment_training_image(value, seed=seed, validation_fold=fold, epoch=1,
                            src_subject_id=key[0], side=key[1], policy=policy)
                    actual = dataset[index]["image"]
                    expected_normalized = normalize_for_model(value, encoder)
                    require(torch.equal(actual, expected_normalized) and tuple(actual.shape) == (3, 224, 224)
                        and actual.dtype == torch.float32 and actual.device.type == "cpu"
                        and bool(torch.isfinite(actual).all()) and not actual.requires_grad,
                        "augmentation_before_registered_normalization_required")
                    normalized += 1
        folds.append({"validation_fold": fold, "training_image_visits": fold_training,
            "validation_image_visits": fold_validation, "changed_training_images": fold_changed})
        print(f"Comprobacion de aumento: pliegue {fold + 1}/5 verificado.", file=sys.stderr, flush=True)
    require(training_visits == 4 * len(rows) and validation_visits == len(rows)
        and all(_pixels(value) == before[key] for key, value in tensors.items()),
        "all_development_visits_and_original_pixel_hashes_required")
    require(set(record["file"] for record in preview_records) == set(PREVIEW_FILES), "all_fixed_previews_required")
    return {"policy_version": policy["version"], "epoch_checked": 1, "seed": seed,
        "training_image_visits": training_visits, "validation_image_visits": validation_visits,
        "augmentation_draws_applied": applied_count, "changed_training_images": changed,
        "normalization_pairs_verified": normalized, "image_encoders_verified": list(IMAGE_MODELS),
        "fold_checks": folds, "deterministic_replay_verified": True, "zero_padding_preserved": True,
        "image_geometry_unchanged": True, "laterality_unchanged": True,
        "validation_images_unchanged": True, "original_tensor_hashes_unchanged": True,
        "training_executed": False, "previews": sorted(preview_records, key=lambda value: value["sample"]),
        "selection_rule": "fold_zero_training_first_canonical_gamma_below_one_then_above_one_then_unapplied_control_with_distinct_canonical_fallback_no_target_or_pixel_selection",
        "predictive_improvement_claimed": False, "anatomical_validation_claimed": False}


def verify(repository_root, package_dir, reference_path, projection_path, descriptor_path, output_dir,
           *, execution_environment="google_colab_cpu"):
    environment = _cpu_environment(execution_environment)
    repository, package, reference, projection, descriptor_file, output = map(checked_path,
        (repository_root, package_dir, reference_path, projection_path, descriptor_path, output_dir))
    require(not output.exists() and output.parent.is_dir()
        and all(not output.resolve().is_relative_to(path.resolve()) and not path.resolve().is_relative_to(output.resolve())
            for path in (repository, package, reference.parent, projection.parent, descriptor_file.parent)),
        "new_separate_methodology_preflight_output_required")
    require(checked_path(__file__).resolve() == repository.resolve() / "src/knee/methodology_inputs_colab.py",
        "methodology_colab_module_must_match_selected_repository")
    descriptor = _descriptor(read_json(descriptor_file))
    require(projection.name == PROJECTION_FILENAME and projection.stat().st_size == descriptor["projection"]["bytes"]
        and sha_file(projection) == descriptor["projection"]["sha256"], "pinned_baseline_projection_required")
    source_paths = {"descriptor": descriptor_file, "projection": projection,
        "development_package": package / "development_package.zip", "handoff": package / "handoff.json",
        "notebook32_reference": reference}
    source_hashes = {name: sha_file(path) for name, path in source_paths.items()}
    implementation = {name: sha_file(repository / name) for name in SOURCE_FILES}
    _, _, plan, _ = release.load_release(repository)
    prepared = preparation.prepare(repository, package, reference, projection, descriptor["projection"]["sha256"], output,
        expected_package_sha256=descriptor["development_package_sha256"])
    require(prepared["development_participants"] == descriptor["development_participants"]
        and prepared["development_knees"] == descriptor["development_knees"], "development_baseline_descriptor_counts_changed")
    from knee.aligned_training.data import load_sealed_inputs
    loaded = load_sealed_inputs(package, reference, repository, descriptor["development_package_sha256"])
    text_proofs = loaded["source_text_verification"]
    _verified_text_proofs(text_proofs)
    augmentation = verify_augmentation(loaded["manifest"]["rows"], loaded["tensors"], plan,
        descriptor["development_package_sha256"], output)
    require(all(sha_file(path) == source_hashes[name] for name, path in source_paths.items())
        and all(sha_file(repository / name) == value for name, value in implementation.items())
        and all(sha_file(repository / proof["relative_name"]) == proof["actual_sha256"] for proof in text_proofs),
        "methodology_preflight_sources_changed")
    artifacts = {str(path.relative_to(output).as_posix()): sha_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()}
    result = {"version": CORE_VERSION, "status": CORE_STATUS, **environment,
        "development_participants": prepared["development_participants"], "development_knees": prepared["development_knees"],
        "development_package_sha256": descriptor["development_package_sha256"],
        "baseline_projection_sha256": descriptor["projection"]["sha256"],
        "baseline_descriptor_sha256": source_hashes["descriptor"],
        "baseline_source_sha256": descriptor["baseline_source_sha256"],
        "clinical_scenarios": prepared["clinical_scenarios"],
        "clinical_preprocessing_states_verified": prepared["clinical_preprocessing_states_verified"],
        "prepared_clinical_requests": prepared["prepared_clinical_requests"],
        "clinical_preprocessing_fitted_inside_training_folds_only": True,
        "baseline_projection_matched_exact_development_keys": True,
        "augmentation": augmentation, "development_data_opened": True,
        **{field: False for field in FALSE_FLAGS}, "source_files_unchanged": True,
        "sources_sha256": source_hashes, "implementation_sha256": implementation,
        "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "training_plan_sha256": release.PINS[release.PLAN_PATH],
        "source_text_verification": text_proofs, "source_text_cross_platform_verified": True,
        "source_text_line_ending_equivalences": sum(value["line_endings_only_equivalence"] for value in text_proofs),
        "artifacts_sha256": artifacts, "review_previews_are_development_only": True,
        "real_image_visual_review_status": "pending_investigator_review_of_three_previews",
        "limitations": ["Input and transformation verification does not establish predictive performance.",
            "The three preview samples are development-only visualization copies, not anatomical validation.",
            "Prepared requests grant no training authority; the reserved test remains closed."],
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    validate_result(result, execution_environment)
    _save(output / CORE_FILENAME, result)
    return result


def _expected_artifacts():
    from knee.clinical_scenarios import SCENARIOS
    names = {"baseline_sidecar.json", preparation.FILENAME, *PREVIEW_FILES}
    for scenario in SCENARIOS:
        for fold in range(5):
            names.add(f"clinical/{scenario}/fold_{fold}_state.json")
            for split in ("training", "validation"):
                names.add(f"clinical/{scenario}/fold_{fold}_{split}.npy")
                names.add(f"clinical/{scenario}/fold_{fold}_{split}_order.json")
            for model in preparation.CLINICAL_IDS:
                names.add(f"requests/{scenario}/{model}/fold_{fold}.json")
    return names


def validate_result(result, execution_environment="google_colab_cpu"):
    require(isinstance(result, dict) and result.get("version") == CORE_VERSION and result.get("status") == CORE_STATUS
        and result.get("execution_environment") == execution_environment
        and result.get("colab_runtime_executed") is (execution_environment == "google_colab_cpu")
        and result.get("local_rehearsal") is (execution_environment == "local_cpu")
        and result.get("cpu_only_guard_passed") is True, "completed_methodology_cpu_preflight_required")
    require(all(result.get(field) is False for field in FALSE_FLAGS), "preflight_must_not_train_or_open_test")
    from knee.clinical_scenarios import SCENARIOS
    require(result.get("clinical_scenarios") == list(SCENARIOS)
        and result.get("clinical_preprocessing_states_verified") == 15
        and result.get("prepared_clinical_requests") == 45
        and result.get("clinical_preprocessing_fitted_inside_training_folds_only") is True
        and result.get("baseline_projection_matched_exact_development_keys") is True,
        "complete_clinical_scenario_preflight_required")
    require(type(result.get("development_knees")) is int and result["development_knees"] > 0
        and type(result.get("development_participants")) is int
        and 0 < result["development_participants"] <= result["development_knees"],
        "development_preflight_counts_required")
    augmentation = result.get("augmentation", {})
    require(augmentation.get("training_image_visits") == 4 * result["development_knees"]
        and augmentation.get("validation_image_visits") == result["development_knees"]
        and augmentation.get("image_encoders_verified") == list(IMAGE_MODELS)
        and augmentation.get("normalization_pairs_verified") == 120,
        "complete_training_validation_augmentation_visits_required")
    require(all(augmentation.get(field) is True for field in ("deterministic_replay_verified", "zero_padding_preserved",
        "image_geometry_unchanged", "laterality_unchanged", "validation_images_unchanged", "original_tensor_hashes_unchanged")),
        "augmentation_geometry_or_replay_not_verified")
    require(result.get("source_files_unchanged") is True and result.get("review_previews_are_development_only") is True
        and isinstance(result.get("artifacts_sha256"), dict)
        and set(result["artifacts_sha256"]) == _expected_artifacts()
        and all(valid_sha(value) for value in result["artifacts_sha256"].values()),
        "exact_methodology_output_inventory_required")
    _verified_text_proofs(result.get("source_text_verification"))
    require(result.get("source_text_cross_platform_verified") is True, "source_text_equivalence_must_be_verified")
    return True


def _inventory(root, result):
    expected = {**result["artifacts_sha256"], CORE_FILENAME: sha_file(root / CORE_FILENAME)}
    paths = {str(path.relative_to(root).as_posix()): path for path in root.rglob("*") if path.is_file()}
    require(set(paths) == set(expected) and len(paths) == 126, "only_registered_methodology_outputs_may_be_saved")
    require(all(checked_path(path).resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        and 0 < path.stat().st_size <= MAX_OUTPUT_FILE_BYTES for path in paths.values())
        and sum(path.stat().st_size for path in paths.values()) <= MAX_OUTPUT_BYTES,
        "bounded_private_methodology_outputs_required")
    require(all(sha_file(paths[name]) == value for name, value in expected.items()), "methodology_artifact_changed")
    return expected


def save_result(repository_root, local_output, records_root, *, code_revision, cpu_tests, runtime,
                execution_environment="google_colab_cpu"):
    """Persist this preflight into a fresh private folder and verify every byte."""
    require(isinstance(code_revision, str) and re.fullmatch(r"[0-9a-f]{40}", code_revision)
        and code_revision != "0" * 40, "published_exact_methodology_revision_required")
    repository, local, persistent = map(checked_path, (repository_root, local_output, records_root))
    require(local.is_dir() and persistent.is_dir() and not local.is_relative_to(persistent)
        and not persistent.is_relative_to(local), "separate_existing_methodology_output_and_private_records_required")
    result = read_json(local / CORE_FILENAME)
    validate_result(result, execution_environment)
    require(isinstance(cpu_tests, dict) and cpu_tests.get("passed") is True and cpu_tests.get("skipped") == 0
        and cpu_tests.get("scope") == list(TEST_MODULES) and type(cpu_tests.get("tests_run")) is int
        and cpu_tests["tests_run"] > 0, "complete_methodology_cpu_tests_required")
    require(isinstance(runtime, dict) and runtime.get("cpu_only") is True
        and runtime.get("training_executed") is False and runtime.get("host_python_modified") is False
        and runtime.get("pretrained_weights_downloaded") is False
        and runtime.get("version") == runtime_base.VERSION
        and runtime.get("python_version") == "3.12.14"
        and runtime.get("source_sha256") == {name: sha_file(repository / name) for name in
            ("src/knee/training_preparation_colab.py", "requirements-training-runner.txt")},
        "isolated_methodology_cpu_runtime_receipt_required")
    require(result["implementation_sha256"] == {name: sha_file(repository / name) for name in SOURCE_FILES},
        "methodology_checkout_changed_before_save")
    inventory = _inventory(local, result)
    destination = persistent / ("methodology_inputs38_" + uuid.uuid4().hex)
    require(not destination.exists(), "new_private_methodology_record_directory_required")
    destination.mkdir()
    for name, checksum in sorted(inventory.items()):
        source, target = local / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024**2)
        require(sha_file(target) == sha_file(source) == checksum, "methodology_private_readback_failed")
    require(_inventory(local, result) == _inventory(destination, result) == inventory
        and read_json(destination / CORE_FILENAME) == result, "methodology_persisted_inventory_mismatch")
    summary = {key: result[key] for key in ("version", "status", "execution_environment", "colab_runtime_executed",
        "local_rehearsal", "cpu_only_guard_passed", "development_participants", "development_knees",
        "environment",
        "development_package_sha256", "baseline_projection_sha256", "baseline_descriptor_sha256", "baseline_source_sha256",
        "clinical_scenarios", "clinical_preprocessing_states_verified", "prepared_clinical_requests",
        "clinical_preprocessing_fitted_inside_training_folds_only", "baseline_projection_matched_exact_development_keys",
        "modeling_release_sha256", "training_plan_sha256", *FALSE_FLAGS)}
    summary.update(source_text_cross_platform_verified=result["source_text_cross_platform_verified"],
        source_text_line_ending_equivalences=result["source_text_line_ending_equivalences"])
    summary.update(code_revision=code_revision, cpu_tests=cpu_tests, augmentation={key: result["augmentation"][key]
        for key in ("policy_version", "epoch_checked", "training_image_visits", "validation_image_visits",
            "augmentation_draws_applied", "changed_training_images", "normalization_pairs_verified", "image_encoders_verified",
            "deterministic_replay_verified", "zero_padding_preserved", "image_geometry_unchanged", "laterality_unchanged",
            "validation_images_unchanged", "original_tensor_hashes_unchanged")},
        previews_saved=len(PREVIEW_FILES), result_files_saved_and_readback_verified=True,
        result_files_verified=len(inventory), core_record_sha256=inventory[CORE_FILENAME],
        runtime_cpu_only=True, executed_at_utc=datetime.now(timezone.utc).isoformat(),
        visual_review_status="review_three_development_previews_before_training_authority",
        next_step="interpret_development_baseline_and_augmentation_preflight_before_any_training_authority")
    validate_summary(summary, execution_environment)
    _save(destination / SUMMARY_FILENAME, summary)
    require(read_json(destination / SUMMARY_FILENAME) == summary, "methodology_summary_readback_failed")
    return summary


def validate_summary(summary, execution_environment="google_colab_cpu"):
    require(isinstance(summary, dict) and summary.get("version") == CORE_VERSION and summary.get("status") == CORE_STATUS
        and summary.get("execution_environment") == execution_environment
        and summary.get("colab_runtime_executed") is (execution_environment == "google_colab_cpu")
        and summary.get("local_rehearsal") is (execution_environment == "local_cpu")
        and all(summary.get(field) is False for field in FALSE_FLAGS)
        and summary.get("result_files_saved_and_readback_verified") is True
        and summary.get("result_files_verified") == 126 and summary.get("previews_saved") == 3,
        "saved_methodology_preflight_summary_required")
    return True


def run_tests(repository_root):
    import unittest
    repository = checked_path(repository_root)
    require(checked_path(__file__).resolve() == repository.resolve() / "src/knee/methodology_inputs_colab.py",
        "methodology_test_module_must_match_checkout")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "methodology_tests_must_hide_cuda")
    sys.path.insert(0, str(repository / "tests"))
    with redirect_stdout(sys.stderr):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(TEST_MODULES)
        if loader.errors:
            print("\n".join(loader.errors), file=sys.stderr)
        require(not loader.errors and suite.countTestCases() > 0, "complete_methodology_test_inventory_required")
        count = suite.countTestCases()
        result = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    require(result.wasSuccessful() and result.testsRun == count and not result.skipped
        and not result.expectedFailures and not result.unexpectedSuccesses, "methodology_tests_not_complete")
    return {"passed": True, "tests_run": count, "skipped": 0, "scope": list(TEST_MODULES),
        "test_runtime": "cpu_synthetic_no_participant_training"}
