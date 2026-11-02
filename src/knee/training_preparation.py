"""Authenticate completed Colab33 and prepare the executor without fitting."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

from knee.final_model_release import load_release, PINS
from knee.final_models_colab import validate_result, TEST_MODULES as PREVIOUS_TESTS
from knee.final_models_preflight import verify_environment
from knee.colab_data import checked_path, require, sha_file

VERSION = "training_runner_preparation_v1.0"
STATUS = "runner_prepared_pending_training_authority"
FILENAME = "training_runner_preparation.json"
SOURCE_33 = {
    "src/knee/final_models_preflight.py": "6dece58a5ba9cdcf4f4769494a9d4d36b20fc07a2d96c41a828565853109597e",
    "src/knee/final_model_release.py": "345398854e512feff8c0759346389b9533dc49d78a5af3d9d50a1bf4c11790c6",
    "src/knee/pretrained_final_encoders.py": "ff2da5b23a75db6fefa1679cb95030a7650ca53a8771bea74da6d9c5225ba1a2",
    "src/knee/pretrained_verification.py": "be09314da8b2edce4b73b75aea1697e51bcd4e505d093e84862d8d5dab5ce028",
    "src/knee/pretrained_convnextv2.py": "5bb0b5c41de547698961e36ccd3fafb6e4290dbb950dff6ce12a408eb5e0371d",
    "src/knee/pretrained_skelex.py": "8ba4ca6888a66afc0f81d6a9e12b7da6fb33219c1ba2343287a2c8c072e1be9a",
    "src/knee/checkpoint_transport.py": "a5fa4b88854da11bd3a86c6ba36f01ae3078678ee46297efff2405b046be88be",
    "requirements-final-models.txt": "d8f68576efddbe0a9cf1f933a38ad9edb5ef0451b5f3b156ade49dedd5340440",
}
EXTRA_PACKAGES = {"scikit-learn": "1.6.1", "scipy": "1.14.1", "joblib": "1.4.2",
    "threadpoolctl": "3.5.0", "xgboost-cpu": "3.0.5"}
RUNNER_FILES = ("src/knee/training_authority.py", "src/knee/training_data.py",
    "src/knee/training_models.py", "src/knee/training_engine.py", "src/knee/training_runner.py",
    "src/knee/training_preparation.py", "scripts/run_development_training.py")


def read_json(path):
    path = checked_path(path)
    require(path.is_file() and 0 < path.stat().st_size <= 4 * 1024**2, "bounded_preparation_record_required")
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate_preparation_record_key")
            value[key] = item
        return value
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_preparation_record")))


def validate_preflight(directory):
    """Validate only the two notebook33 records; never scan participant data."""
    root = checked_path(directory)
    require(root.is_dir(), "existing_completed_preflight_directory_required")
    summary = read_json(root / "colab_final_models_check.json")
    checkpoint_path = root / "checkpoint_verification.json"
    checkpoint = read_json(checkpoint_path)
    validate_result(checkpoint, "google_colab_cpu")
    require(summary.get("version") == checkpoint["version"]
        and summary.get("status") == checkpoint["status"]
        and summary.get("code_revision") == "b687d64e18d8e34b106c34e4cc28abd683afc000"
        and summary.get("colab_runtime_executed") is True and summary.get("local_rehearsal") is False
        and summary.get("cpu_only_guard_passed") is True
        and summary.get("result_files_saved_and_readback_verified") is True
        and summary.get("checkpoint_record_sha256") == sha_file(checkpoint_path),
        "completed_actual_colab33_and_exact_checkpoint_record_required")
    tests = summary.get("cpu_tests", {})
    require(tests.get("passed") is True and tests.get("tests_run") == 195
        and type(tests.get("tests_run")) is int and tests.get("skipped") == 0
        and tests.get("scope") == list(PREVIOUS_TESTS), "completed_colab33_test_scope_required")
    require(summary.get("source_files_sha256") == checkpoint.get("source_files_sha256") == SOURCE_33
        and checkpoint.get("release_component_canonical_sha256") == PINS
        and summary.get("runtime_requirements_sha256") == SOURCE_33["requirements-final-models.txt"],
        "approved_colab33_code_and_release_hashes_required")
    for field in ("training_executed", "optimizer_created", "participant_data_opened",
        "reserved_test_predictive_evaluation_opened", "originals_modified", "partitions_modified",
        "training_authorized", "training_smoke_test_passed", "gpu_resource_profile_recorded"):
        require(summary.get(field) is False, "preparation_does_not_inherit_training_authority")
    require(summary.get("shortlist_search_closed") is True
        and summary.get("pending_authorized_access") == ["image_dinov3_vits16"]
        and summary.get("registered_image_models") == 7 and summary.get("pretrained_models_loaded") == 6,
        "same_closed_model_roster_required")
    try:
        timestamp = datetime.fromisoformat(summary["executed_at_utc"])
        require(timestamp.utcoffset() is not None, "aware_preflight_timestamp_required")
    except (KeyError, TypeError, ValueError):
        raise ValueError("valid_preflight_timestamp_required") from None
    return summary


def run(repository_root, previous_preflight_dir, output_dir, *, execution_environment="local_cpu"):
    """Dry preparation only: syntax, authority guard and synthetic eval forward."""
    import numpy as np
    import torch
    from knee.training_models import build_model, normalize_for_model, IMAGE_DIMENSIONS
    from knee.pretrained_verification import state_digest
    from knee import training_runner
    environment = verify_environment(execution_environment)
    environment.update({key: importlib.metadata.version(key) for key in EXTRA_PACKAGES})
    require(all(environment[key] == value for key, value in EXTRA_PACKAGES.items()),
        "pinned_training_preparation_dependencies_required")
    repository, proof, output = map(checked_path, (repository_root, previous_preflight_dir, output_dir))
    require(not output.exists() and all(not output.is_relative_to(p) and not p.is_relative_to(output)
        for p in (repository, proof)), "new_separate_training_preparation_output_required")
    previous = validate_preflight(proof)
    release, catalog, plan, registry = load_release(repository)
    require(all(sha_file(repository / name) == value for name, value in SOURCE_33.items()),
        "verified_colab33_implementation_changed")
    sources = {}
    for name in RUNNER_FILES:
        path = repository / name
        ast.parse(path.read_text(encoding="utf-8"))
        sources[name] = sha_file(path)
    for name in ("src/knee/training_preparation_colab.py", "requirements-training-runner.txt"):
        sources[name] = sha_file(repository / name)
    require(checked_path(__file__) == repository / "src/knee/training_preparation.py",
        "preparation_module_must_match_published_checkout")
    # A default invocation must stop before opening any package or optimizer.
    try:
        training_runner.execute(None, None, repository_root=repository, package_dir=None,
            reference_path=None, weights_root=None, output_dir=None)
    except ValueError as error:
        require(str(error) == "separate_training_authority_required_before_execution",
            "training_guard_failed_before_any_data_access")
    else:
        raise ValueError("training_executor_must_not_run_without_authority")
    rng_before = torch.get_rng_state().clone()
    model, adoption = build_model("clinical_mlp", plan, catalog)
    before = state_digest(model)
    with torch.inference_mode():
        logits = model(clinical=torch.tensor([[0., 0., 0.], [1., 1., 1.]]))
    require(tuple(logits.shape) == (2,) and bool(torch.isfinite(logits).all())
        and state_digest(model) == before and torch.equal(rng_before, torch.get_rng_state())
        and all(p.grad is None for p in model.parameters()), "synthetic_eval_only_model_factory_required")
    gray = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(1, 224, 224)
    gray.setflags(write=False)
    pixels = hashlib.sha256(gray.tobytes()).hexdigest()
    for encoder_id in IMAGE_DIMENSIONS:
        normalized = normalize_for_model(gray, encoder_id)
        require(normalized.shape == (3, 224, 224) and bool(torch.isfinite(normalized).all()),
            "registered_training_input_normalization_required")
    require(hashlib.sha256(gray.tobytes()).hexdigest() == pixels and not gray.flags.writeable,
        "synthetic_preparation_input_changed")
    result = {"version": VERSION, "status": STATUS, "execution_environment": execution_environment,
        "colab_runtime_executed": execution_environment == "google_colab_cpu",
        "training_runner_implemented": True, "training_authorized": False,
        "training_executed": False, "optimizer_created": False, "backward_executed": False,
        "participant_data_opened": False, "reserved_test_predictive_evaluation_opened": False,
        "originals_modified": False, "partitions_modified": False,
        "clinical_models_implemented": [r["id"] for r in registry["clinical"]["candidates"]],
        "accessible_image_models_implemented": list(IMAGE_DIMENSIONS),
        "conditional_fusion_models_implemented": plan["multimodal_models"]["primary"],
        "pending_authorized_access": ["image_dinov3_vits16"], "shortlist_search_closed": True,
        "previous_colab33_checkpoint_record_sha256": previous["checkpoint_record_sha256"],
        "release_component_canonical_sha256": PINS, "source_files_sha256": sources,
        "synthetic_clinical_eval_forward_verified": True, "image_normalization_only_verified": True,
        "actual_training_loop_executed": False, "training_smoke_test_passed": False,
        "gpu_resource_profile_recorded": False, "environment": environment,
        "resume_supported": False,
        "next_step": "review_preparation_then_separately_authorize_bounded_development_gpu_training_smoke",
        "limitations": ["Preparation does not establish predictive performance or GPU backward compatibility.",
            "Training requires a separate exact-request authority, all relevant eligibility gates and development-only inputs.",
            "Stage two and fusion require authenticated development selection; no test or calibration operation exists here."],
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=False)
    path = output / FILENAME
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    require(read_json(path) == result, "training_preparation_readback_failed")
    return result
