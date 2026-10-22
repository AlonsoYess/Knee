"""Independently recompute development clinical and image loading numerics.

No clinical fitter, image adapter, PyTorch model or reserved-test archive is
imported or opened. The established package reader checks source integrity only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knee.colab_data import load_development_package


CLINICAL_POLICY = {
    "feature_order": ["AGEYEARS", "SEX", "BMI"], "sex_encoding": {"F": 0, "M": 1},
    "bmi_imputation": "median_training_knee_rows_only",
    "continuous_scaling": "population_standardization_ddof_0_after_training_BMI_imputation",
    "constant_scale": 1.0, "statistics_unit": "training_knee_rows", "missingness_indicator": False,
}
ENCODERS = {"image_densenet121": 1024, "image_vit_b16": 768}


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value, *, newline=False):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (raw + ("\n" if newline else "")).encode("utf-8")


def digest(value, *, newline=False):
    return hashlib.sha256(canonical(value, newline=newline)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identities(rows):
    return [[row["src_subject_id"], row["side"]] for row in rows]


def independent_statistics(rows):
    observed = sorted(row["BMI"] for row in rows if row["BMI"] is not None)
    assert observed, "training_only_observed_BMI_required"
    middle = len(observed) // 2
    median = float(observed[middle] if len(observed) % 2
                   else (observed[middle - 1] + observed[middle]) / 2)
    values = np.asarray([[row["AGEYEARS"], row["BMI"] if row["BMI"] is not None else median]
                         for row in rows], dtype=np.float64)
    mean = np.add.reduce(values, axis=0) / len(rows)
    variance = np.add.reduce((values - mean) ** 2, axis=0) / len(rows)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    assert np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()
    return median, mean, scale


def independent_clinical(rows, stats):
    median, mean, scale = stats
    # Scalar operations and explicit order reproduce the specified float32 input,
    # without reading a production preprocessing state to construct the values.
    return np.asarray([[(r["AGEYEARS"] - float(mean[0])) / float(scale[0]),
                        0 if r["SEX"] == "F" else 1,
                        ((median if r["BMI"] is None else r["BMI"]) - float(mean[1])) / float(scale[1])]
                       for r in rows], dtype=np.float32)


def independent_images_sha(rows, tensors):
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    result = hashlib.sha256()
    for row in rows:
        source = tensors[(row["src_subject_id"], row["side"])]
        assert source.shape == (1, 224, 224) and source.dtype == np.float32 and not source.flags.writeable
        rgb = np.array(np.broadcast_to(source, (3, 224, 224)), dtype=np.float32, order="C", copy=True)
        np.subtract(rgb, mean, out=rgb)
        np.divide(rgb, std, out=rgb)
        assert np.isfinite(rgb).all() and rgb.flags.c_contiguous
        result.update(rgb.tobytes(order="C"))
    return result.hexdigest()


def verify(root, package, expected_sha, config_path=None):
    root, package = Path(root), Path(package)
    assert np.__version__ == "1.26.4", "same_frozen_numerical_runtime_required"
    assert sha_file(package) == expected_sha
    closure_path = root / "closure_summary.json"
    closure_sha = sha_file(closure_path)
    closure = read_json(closure_path)
    assert closure["status"] == "development_inputs_and_cpu_loading_verified"
    assert closure["sources_sha256"]["development_package"] == expected_sha
    assert closure["clinical_features"] == ["AGEYEARS", "SEX", "BMI"]
    pins = {str(package): expected_sha, str(closure_path): closure_sha}
    if config_path is not None:
        config_path = Path(config_path)
        config = read_json(config_path)
        pins[str(config_path)] = sha_file(config_path)
        assert pins[str(config_path)] == closure["private_configuration_sha256"]
        assert Path(config["output_dir"]).resolve() == root.resolve()
        assert Path(config["development_package"]["path"]).resolve() == package.resolve()
        names = ("development_package", "handoff", "package_closure", "colab_confirmation", "policy", "registry", "authority")
        assert set(closure["sources_sha256"]) == set(names)
        for name in names:
            spec = config[name]
            assert closure["sources_sha256"][name] == spec["sha256"] == sha_file(spec["path"])
            pins[str(Path(spec["path"]))] = spec["sha256"]
        for spec in config["source_pins"]:
            assert sha_file(spec["path"]) == spec["sha256"]
            assert closure["implementation_sha256"][Path(spec["path"]).name] == spec["sha256"]
            pins[str(Path(spec["path"]))] = spec["sha256"]
    for relative, expected in closure["clinical_artifacts_sha256"].items():
        path = root / relative
        assert path.resolve().is_relative_to(root.resolve()) and sha_file(path) == expected
        pins[str(path)] = expected
    loaded = load_development_package(package, expected_sha)
    rows, tensors = loaded["manifest"]["rows"], loaded["tensors"]
    assert len(rows) == closure["development_knees"]
    assert len({row["src_subject_id"] for row in rows}) == closure["development_participants"]
    assert {row["cv_fold"] for row in rows} == set(range(5))
    assert all(row["outer_block"] != 0 for row in rows)
    assert len({tuple(pair) for pair in identities(rows)}) == len(rows)
    baseline_snapshot = canonical(rows)
    tensor_snapshot = {pair: hashlib.sha256(value.tobytes(order="C")).hexdigest() for pair, value in tensors.items()}
    records = closure["folds"]
    assert len(records) == closure["clinical_states_fitted"] == 5
    expected_loading, transformed_rows = {}, 0
    for fold in range(5):
        training_sorted = sorted([r for r in rows if r["cv_fold"] != fold],
                                 key=lambda r: (r["src_subject_id"], r["side"]))
        validation = [r for r in rows if r["cv_fold"] == fold]
        train_people = {r["src_subject_id"] for r in training_sorted}
        valid_people = {r["src_subject_id"] for r in validation}
        assert not train_people & valid_people
        assert train_people | valid_people == {r["src_subject_id"] for r in rows}
        stats = independent_statistics(training_sorted)
        state_path = root / "clinical" / f"fold_{fold}_state.json"
        state = read_json(state_path)
        assert state["version"] == "clinical_fold_inputs_v1.0" and state["policy"] == CLINICAL_POLICY
        assert state["validation_fold"] == fold and state["train_count"] == len(training_sorted)
        assert state["train_participants"] == len(train_people)
        assert state["train_identity_sha256"] == digest(identities(training_sorted))
        assert state["BMI_median"] == stats[0]
        assert state["continuous_mean"] == dict(zip(("AGEYEARS", "BMI"), map(float, stats[1])))
        assert state["continuous_scale"] == dict(zip(("AGEYEARS", "BMI"), map(float, stats[2])))
        assert digest({k: v for k, v in state.items() if k != "record_sha256"}) == state["record_sha256"]
        assert records[fold]["validation_fold"] == fold and records[fold]["clinical_state_sha256"] == state["record_sha256"]
        expected_loading[fold] = {}
        for split in ("training", "validation"):
            selected = [r for r in rows if (r["cv_fold"] == fold) == (split == "validation")]
            order = identities(selected)
            prefix = root / "clinical" / f"fold_{fold}_{split}"
            assert read_json(str(prefix) + "_order.json") == order
            values = np.load(str(prefix) + ".npy", allow_pickle=False)
            expected = independent_clinical(selected, stats)
            assert values.shape == expected.shape == (len(selected), 3) and values.dtype == np.float32
            assert values.flags.c_contiguous and np.isfinite(values).all()
            assert np.array_equal(values, expected), "clinical_values_differ_from_training_only_reconstruction"
            assert np.array_equal(values[:, 1], np.asarray([0 if r["SEX"] == "F" else 1 for r in selected], dtype=np.float32))
            clinical_sha = hashlib.sha256(expected.tobytes(order="C")).hexdigest()
            identity_sha = digest(order, newline=True)
            target_sha = hashlib.sha256(np.asarray([r["PROGRESION_48M"] for r in selected], dtype=np.float32).tobytes()).hexdigest()
            row_record = records[fold]["splits"][split]
            assert row_record == {"rows": len(selected), "participants": len({r["src_subject_id"] for r in selected}),
                "imputed_BMI_rows": sum(r["BMI"] is None for r in selected),
                "ordered_identity_sha256": identity_sha, "clinical_sha256": clinical_sha}
            expected_loading[fold][split] = {"rows": len(selected),
                "participants": len({r["src_subject_id"] for r in selected}),
                "batches": math.ceil(len(selected) / 8), "image_shape": [3, 224, 224],
                "clinical_features": 3, "target_shape": [], "ordered_identity_sha256": identity_sha,
                "clinical_sha256": clinical_sha, "targets_sha256": target_sha,
                "image_sha256": independent_images_sha(selected, tensors)}
            transformed_rows += len(selected)
    assert set(closure["image_loading"]) == set(ENCODERS)
    loader_visits = 0
    for encoder, width in ENCODERS.items():
        loading = closure["image_loading"][encoder]
        assert len(loading) == 5
        for fold, result in enumerate(loading):
            assert result == {"validation_fold": fold, "splits": expected_loading[fold]}
            loader_visits += sum(r["rows"] for r in result["splits"].values())
        probe = closure["architecture_probes"][encoder]
        assert probe["encoder_id"] == encoder and probe["initialization"] == "random_weights_none"
        assert probe["pretrained_checkpoint_loaded"] is False and probe["pretrained_download_attempted"] is False
        assert probe["mode"] == "eval_torch_inference_mode"
        assert probe["parameters_and_buffers_unchanged"] is True and probe["all_parameter_gradients_none"] is True
        assert probe["state_sha256_before"] == probe["state_sha256_after"]
        assert len(probe["folds"]) == 5
        assert all(r["validation_fold"] == i and r["rows"] == 2 and r["feature_width"] == width
                   for i, r in enumerate(probe["folds"]))
    assert loader_visits == closure["total_loader_row_visits"] == transformed_rows * len(ENCODERS)
    assert transformed_rows == len(rows) * 5
    assert closure["clinical_preprocessing_fitted"] is True and closure["BMI_imputation_applied_to_copies"] is True
    assert closure["missing_BMI_development_knees_retained"] == sum(row["BMI"] is None for row in rows)
    for flag in ("validation_or_target_used_for_fitting", "additional_resize_crop_flip_or_augmentation",
                 "pretrained_weights_loaded", "pretrained_weight_eligibility_certified", "predictive_model_training_executed",
                 "optimizer_created", "backward_executed", "reserved_test_opened", "originals_modified", "partitions_modified"):
        assert closure[flag] is False
    assert closure["new_image_files_written"] == 0
    assert canonical(rows) == baseline_snapshot
    assert {pair: hashlib.sha256(value.tobytes(order="C")).hexdigest() for pair, value in tensors.items()} == tensor_snapshot
    for path, expected in pins.items():
        assert sha_file(path) == expected, "source_or_artifact_changed_during_independent_check"
    return {"version": "model_input_independent_numerical_verification_v1.0", "status": "passed",
        "development_knees_checked": len(rows), "development_participants_checked": len({r["src_subject_id"] for r in rows}),
        "training_only_states_independently_recomputed": 5, "clinical_arrays_checked": 10,
        "clinical_float32_rows_compared_exactly": transformed_rows,
        "clinical_feature_values_compared_exactly": transformed_rows * 3,
        "ordered_identity_records_checked": 10, "encoder_fold_split_records_checked": len(ENCODERS) * 10,
        "loader_image_clinical_target_hashes_match": True, "total_loader_row_visits_checked": loader_visits,
        "fixed_RGB_mean_std_and_geometry_verified": True, "no_train_validation_participant_overlap": True,
        "validation_statistics_used": False, "architecture_probe_records_checked": True,
        "architecture_forward_reexecuted_by_verifier": False, "pretrained_checkpoint_verified": False,
        "source_and_artifact_hashes_unchanged": True, "private_source_config_verified": config_path is not None,
        "reserved_test_opened": False, "predictive_model_training_executed": False,
        "development_package_sha256": expected_sha, "closure_summary_sha256": closure_sha,
        "verifier_sha256": sha_file(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.package, args.expected_sha, args.config)
    output = args.output or args.root / "independent_verification.json"
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if output.exists():
        assert output.read_text(encoding="utf-8") == payload, "existing_independent_record_changed"
    else:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(payload, end="")
