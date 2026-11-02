"""Prepare the aligned clinical inputs and requests without predictor training.

Requires a separately supplied, baseline-only JSON projection containing ONLY
the authenticated development participants/sides. It must be obtained from the
audited V00 source; this module does not fabricate it, read an outcome workbook,
or search a full cohort for additional records. A caller-supplied SHA256 fixes
that projection before it is parsed. Existing images and partitions are reused.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from knee import clinical_scenarios as clinical
from knee import methodology_release as release
from knee.colab_data import checked_path, require, sha_file, valid_sha
from knee.aligned_training.authority import validate_request, digest
from knee.training_preparation import read_json

VERSION = "development_methodology_inputs_preparation_v1.0"
STATUS = "development_complementary_inputs_prepared_no_training"
FILENAME = "methodology_preparation.json"
CLINICAL_IDS = ("clinical_logistic", "clinical_xgboost", "clinical_mlp")
SOURCE_FILES = ("src/knee/methodology_preparation.py", "src/knee/clinical_scenarios.py",
    "src/knee/methodology_release.py", "src/knee/aligned_training/authority.py", "src/knee/aligned_training/data.py")


def _write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    require(path.read_bytes() == raw and read_json(path) == value, "methodology_json_readback_failed")


def _write_matrix(path, matrix):
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(buffer.getvalue())
    restored = np.load(path, allow_pickle=False)
    require(restored.dtype == np.dtype("float32") and restored.shape == matrix.shape
        and np.array_equal(restored, matrix), "methodology_matrix_readback_failed")


def make_request(package_sha256, sidecar_sha256, scenario, model_id, fold, plan, registry):
    """A validated request describes future work; it contains NO authority."""
    request = {"version": release.REQUEST_VERSION, "operation": "development_fold", "model_id": model_id,
        "image_encoder_id": None, "stage": "stage_1", "validation_fold": fold,
        "seed": plan["selection"]["screening_seed"], "development_package_sha256": package_sha256,
        "image_micro_batch_size": plan["neural_training"]["initial_image_micro_batch_size"],
        "selection_evidence_sha256": None, "fold_image_job_sha256": None,
        "clinical_scenario": scenario, "baseline_sidecar_sha256": None if scenario == "clinical_main" else sidecar_sha256,
        "augmentation_policy": "identity", "modeling_release_sha256": release.PINS[release.RELEASE_PATH]}
    validate_request(request, plan, registry)
    return request


def prepare(repository_root, package_dir, reference_path, baseline_projection_path,
            expected_projection_sha256, output_dir, *, expected_package_sha256):
    """Verify development inputs and fit preprocessing only, never an estimator."""
    require(valid_sha(expected_projection_sha256) and valid_sha(expected_package_sha256),
            "external_development_package_and_projection_hashes_required")
    repository, package, reference, projection, output = map(checked_path,
        (repository_root, package_dir, reference_path, baseline_projection_path, output_dir))
    repository, package, reference, projection, output = (p.resolve() for p in
        (repository, package, reference, projection, output))
    require(repository.is_dir() and package.is_dir() and reference.is_file() and projection.is_file(),
            "existing_repository_development_package_reference_and_projection_required")
    require(not output.exists() and output.parent.is_dir()
        and all(not output.is_relative_to(p) and not p.is_relative_to(output)
                for p in (repository, package, reference.parent, projection.parent)),
        "new_separate_methodology_preparation_output_required")
    require(checked_path(__file__).resolve() == repository / "src/knee/methodology_preparation.py",
            "methodology_preparation_must_use_selected_repository")
    require(sha_file(projection) == expected_projection_sha256, "external_baseline_projection_hash_mismatch")
    _, _, plan, registry = release.load_release(repository)
    source_paths = {"development_package": package / "development_package.zip",
        "handoff": package / "handoff.json", "notebook32_reference": reference,
        "baseline_development_projection": projection}
    source_hashes = {key: sha_file(path) for key, path in source_paths.items()}
    implementation = {name: sha_file(repository / name) for name in SOURCE_FILES}
    # The loader authenticates the immutable package and historical fold states.
    from knee.aligned_training.data import load_sealed_inputs
    loaded = load_sealed_inputs(package, reference, repository, expected_package_sha256)
    rows = loaded["manifest"]["rows"]
    projection_rows = read_json(projection)
    sidecar = clinical.make_baseline_sidecar(projection_rows, rows, expected_package_sha256,
                                             expected_projection_sha256)
    enriched = clinical.join_baseline_sidecar(rows, sidecar, expected_package_sha256,
                                              expected_sidecar_sha256=sidecar["record_sha256"])
    jobs, fitted = [], []
    for scenario in clinical.SCENARIOS:
        scenario_rows = rows if scenario == "clinical_main" else enriched
        for fold in range(5):
            state = clinical.fit_scenario_fold(scenario_rows, fold, scenario)
            clinical.verify_scenario_fold_state(scenario_rows, state)
            parts = {}
            for split in ("training", "validation"):
                selected = [row for row in scenario_rows if (row["cv_fold"] == fold) == (split == "validation")]
                matrix = clinical.transform_scenario_rows(selected, state)
                parts[split] = (matrix, [[row["src_subject_id"], row["side"]] for row in selected])
            require(not {sid for sid, _ in parts["training"][1]} & {sid for sid, _ in parts["validation"][1]},
                    "methodology_training_validation_participant_overlap")
            fitted.append((scenario, fold, state, parts))
            for model_id in CLINICAL_IDS:
                jobs.append(make_request(expected_package_sha256, sidecar["record_sha256"],
                                         scenario, model_id, fold, plan, registry))
    require(len(fitted) == 15 and len(jobs) == 45, "complete_three_scenario_five_fold_preparation_required")
    output.mkdir()
    _write_json(output / "baseline_sidecar.json", sidecar)
    restored = read_json(output / "baseline_sidecar.json")
    require(clinical.join_baseline_sidecar(rows, restored, expected_package_sha256,
                expected_sidecar_sha256=sidecar["record_sha256"]) == enriched, "baseline_sidecar_readback_failed")
    checks = []
    for scenario, fold, state, parts in fitted:
        prefix = output / "clinical" / scenario
        _write_json(prefix / f"fold_{fold}_state.json", state)
        split_checks = {}
        for split, (matrix, identities) in parts.items():
            _write_matrix(prefix / f"fold_{fold}_{split}.npy", matrix)
            _write_json(prefix / f"fold_{fold}_{split}_order.json", identities)
            split_checks[split] = {"rows": len(identities), "dimensions": matrix.shape[1],
                "ordered_identity_sha256": digest(identities),
                "matrix_pixel_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest()}
        checks.append({"scenario": scenario, "validation_fold": fold,
            "state_sha256": state["record_sha256"], "splits": split_checks})
    for job in jobs:
        _write_json(output / "requests" / job["clinical_scenario"] / job["model_id"]
                    / f'fold_{job["validation_fold"]}.json', job)
    require(all(sha_file(path) == source_hashes[name] for name, path in source_paths.items())
        and all(sha_file(repository / name) == value for name, value in implementation.items()),
        "methodology_source_or_implementation_changed_during_preparation")
    release.load_release(repository)
    artifacts = {str(path.relative_to(output).as_posix()): sha_file(path)
                 for path in sorted(output.rglob("*")) if path.is_file()}
    result = {"version": VERSION, "status": STATUS,
        "development_participants": len({row["src_subject_id"] for row in rows}),
        "development_knees": len(rows), "clinical_scenarios": list(clinical.SCENARIOS),
        "clinical_preprocessing_states_verified": len(fitted), "prepared_clinical_requests": len(jobs),
        "baseline_sidecar_sha256": sidecar["record_sha256"],
        "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "training_plan_sha256": release.PINS[release.PLAN_PATH],
        "sources_sha256": source_hashes, "implementation_sha256": implementation,
        "fold_checks": checks, "artifacts_sha256": artifacts,
        "clinical_preprocessing_fitted_inside_training_folds_only": True,
        "baseline_projection_matched_exact_development_keys": True,
        "result_files_saved_and_readback_verified": True,
        "training_authorized": False, "training_executed": False, "optimizer_created": False,
        "backward_executed": False, "reserved_test_predictive_evaluation_opened": False,
        "originals_modified": False, "partitions_modified": False,
        "new_studies_selected": False, "image_augmentation_applied": False,
        "execution_authorities_created": 0,
        "next_step": "review_development_preparation_then_separate_colab_training_authority",
        "limitations": ["Preparation verifies inputs and preprocessing, not predictive performance.",
            "The supplied baseline-only development projection must come from the audited V00 source.",
            "Prepared requests do not grant training authority and no reserved-test data are accepted."],
        "executed_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(output / FILENAME, result)
    return result
