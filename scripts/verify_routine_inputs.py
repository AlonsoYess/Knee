"""Independently verify routine inputs against the sealed development recipe.

This verifier does not import the production tensor transformation. It rebuilds
the expected pixels directly from the frozen recipe, reads routine native ROIs
only, and returns aggregate checks without clinical data or future outcomes.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import PIL
from PIL import Image
from knee.roi_global_retention_sources import safe, sha_file


def _read_pinned(descriptor):
    path = Path(descriptor["path"])
    assert path.suffix == ".json" and not path.is_symlink()
    assert sha_file(path) == descriptor["sha256"], "pinned_input_changed"
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _assert_outcome_free(value):
    if isinstance(value, dict):
        assert not {str(key).upper() for key in value}.intersection(
            {"KL_48", "KL_48M", "PROGRESION_48M", "FECHA_V06", "Y_TRUE", "OUTCOME", "LABEL"}
        ), "outcome_fields_forbidden_in_image_inputs"
        for child in value.values():
            _assert_outcome_free(child)
    elif isinstance(value, list):
        for child in value:
            _assert_outcome_free(child)


def _expected_tensor(native, source):
    # An independent reconstruction: no call to normalize_native/transform_native.
    raw = native.astype(np.float64)
    low, high = np.quantile(raw, [.01, .99], method="linear")
    assert math.isfinite(low) and math.isfinite(high) and high > low
    unit = np.clip((raw - low) / (high - low), 0.0, 1.0)
    inverted = source["photometric"] == "MONOCHROME1"
    if inverted:
        unit = 1.0 - unit
    unit = np.ascontiguousarray(unit, dtype=np.float32)
    spacing_h, spacing_w = source["row_spacing_mm"], source["column_spacing_mm"]
    assert all(type(x) in {float, int} and math.isfinite(x) and x > 0
               for x in (spacing_h, spacing_w))
    physical_h, physical_w = native.shape[0] * spacing_h, native.shape[1] * spacing_w
    longest = max(physical_h, physical_w)
    scaled_h, scaled_w = 224 * (physical_h / longest), 224 * (physical_w / longest)
    height = max(1, min(224, math.floor(scaled_h + .5)))
    width = max(1, min(224, math.floor(scaled_w + .5)))
    top, left = (224 - height) // 2, (224 - width) // 2
    bottom, right = 224 - height - top, 224 - width - left
    resized = np.asarray(Image.fromarray(unit).resize(
        (width, height), resample=Image.Resampling.BILINEAR, reducing_gap=None), dtype=np.float32)
    expected = np.zeros((1, 224, 224), dtype=np.float32)
    expected[0, top:top + height, left:left + width] = np.clip(resized, 0.0, 1.0)
    return expected, {"percentile_window": [float(low), float(high)],
        "resized_shape": [height, width], "padding_ltrb": [left, top, right, bottom],
        "physical_size_mm": [physical_h, physical_w], "inversion_applied": inverted,
        "rounding_error_pixels": [height - scaled_h, width - scaled_w]}


def verify(config_path):
    config_path = Path(config_path)
    config_sha = sha_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_root"])
    membership = _read_pinned(config["membership"])
    provenance = _read_pinned(config["cohort_provenance"])
    contract = _read_pinned(config["contract"])
    development_closure = _read_pinned(config["development_closure"])
    development_manifest = _read_pinned(config["development_manifest"])
    manifest_path, closure_path = output / "manifest.json", output / "closure_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    assert output.name == "routine_inputs_224_001"
    assert sha_file(config["development_manifest"]["path"]) == development_closure["manifest_sha256"]
    assert sha_file(manifest_path) == closure["manifest_sha256"]
    version = contract["version"]
    assert version == "physical_letterbox_224_p01_p99_v1.0"
    assert contract["output_shape"] == [1, 224, 224] and contract["dtype"] == "float32"
    assert contract["runtime"] == {"numpy": np.__version__, "Pillow": PIL.__version__}
    assert contract["geometry"]["orientation"] == "original_acquisition_no_mirroring"
    assert contract["geometry"]["crop"] is False
    assert contract["intensity"]["percentiles"] == [1.0, 99.0]
    assert contract["intensity"]["percentile_method"] == "linear"
    assert contract["intensity"]["photometric"] == "invert_MONOCHROME1_after_clipping_before_padding"
    assert not any(contract[k] for k in ("rgb_replication_applied", "checkpoint_normalization_applied", "augmentation_applied"))
    assert development_manifest["tensor_contract_version"] == development_closure["tensor_contract_version"] == version
    assert manifest["tensor_contract_version"] == closure["tensor_contract_version"] == version
    assert development_manifest["tensor_contract_sha256"] == manifest["tensor_contract_sha256"] == _digest(contract)
    contract_path = str(Path(config["contract"]["path"]))
    assert development_manifest["source_files_sha256"][contract_path] == config["contract"]["sha256"]
    assert manifest["source_files_sha256"][contract_path] == config["contract"]["sha256"]
    for name in ("membership", "cohort_provenance", "development_closure", "development_manifest"):
        descriptor = config[name]
        assert manifest["source_files_sha256"][str(Path(descriptor["path"]))] == descriptor["sha256"]
    assert membership["outcomes_used_for_membership"] is False
    _assert_outcome_free(membership)
    _assert_outcome_free(manifest)
    key = lambda row: (row["src_subject_id"], row["side"])
    all_membership = {key(row) for row in membership["rows"]}
    assert len(all_membership) == len(membership["rows"]), "unique_common_membership_required"
    assert {row["role"] for row in membership["rows"]} == {"development", "routine"}
    selected = [row for row in membership["rows"] if row["role"] == "routine"]
    eligible = {key(row): row for row in selected}
    expected_knees = membership["summary"]["routine_knees"]
    expected_participants = membership["summary"]["routine_participants"]
    assert len(selected) == len(eligible) == expected_knees == len(manifest["rows"])
    assert len({k[0] for k in eligible}) == expected_participants
    assert {key(row) for row in manifest["rows"]} == set(eligible)
    assert {row["role"] for row in manifest["rows"]} == {"routine"}
    development_keys = {key(row) for row in development_manifest["rows"]}
    expected_development_keys = {key(row) for row in membership["rows"] if row["role"] == "development"}
    assert len(development_keys) == len(development_manifest["rows"]) == membership["summary"]["development_knees"]
    assert development_keys == expected_development_keys
    assert {row["role"] for row in development_manifest["rows"]} == {"development"}
    assert not development_keys & set(eligible)
    assert development_keys | set(eligible) == all_membership
    assert not ({k[0] for k in eligible} & set(membership["development_reservations"]))
    assert manifest["summary"]["development_images_processed"] == 0
    assert manifest["summary"]["routine_images_processed"] == expected_knees
    assert manifest["summary"]["knees"] == expected_knees
    assert manifest["summary"]["participants"] == expected_participants
    assert closure["summary"] == manifest["summary"]
    assert all(manifest["summary"][flag] is False for flag in (
        "outcome_rows_loaded", "training_executed", "partitions_created", "native_edits", "qc_regraded"))
    tensor_paths, shapes, photo_counts, inversion_count = set(), [], Counter(), 0
    for row in manifest["rows"]:
        source = eligible[key(row)]
        assert source["side"] in {"RIGHT", "LEFT"}
        assert source["native_spatial_orientation"] == "original_acquisition_no_mirroring"
        assert source["photometric"] in {"MONOCHROME1", "MONOCHROME2"}
        native_path = safe(provenance["native_roots"][source["crop"]["root_key"]], source["crop"]["relative_path"])
        native_sha = sha_file(native_path)
        assert native_sha == source["crop"]["sha256"] == row["source_native_sha256"]
        assert row["native_crop"] == source["crop"]
        native = np.load(native_path, allow_pickle=False)
        native.setflags(write=False)
        assert native.ndim == 2 and native.dtype.kind in {"u", "i"}
        assert list(native.shape) == source["crop"]["shape"] and str(native.dtype) == source["crop"]["dtype"]
        before = hashlib.sha256(native.tobytes(order="C")).hexdigest()
        assert before == source["crop"]["pixel_sha256"]
        path = safe(output, row["tensor"]["relative_path"])
        assert path not in tensor_paths, "unique_tensor_paths_required"
        tensor_paths.add(path)
        assert sha_file(path) == row["tensor"]["sha256"]
        tensor = np.load(path, allow_pickle=False)
        assert tensor.shape == (1, 224, 224) and tensor.dtype == np.float32 and tensor.flags.c_contiguous
        assert np.isfinite(tensor).all() and tensor.min() >= 0 and tensor.max() <= 1 and tensor.std() > 0
        assert row["tensor"]["shape"] == [1, 224, 224] and row["tensor"]["dtype"] == "float32"
        assert hashlib.sha256(tensor.tobytes(order="C")).hexdigest() == row["tensor"]["pixel_sha256"]
        expected, geometry = _expected_tensor(native, source)
        assert np.array_equal(tensor, expected), "independent_pixel_reconstruction_failed"
        record = row["transform"]
        for name in ("resized_shape", "padding_ltrb", "percentile_window", "physical_size_mm", "rounding_error_pixels"):
            np.testing.assert_allclose(record[name], geometry[name], rtol=0, atol=1e-9)
        assert record["native_shape"] == list(native.shape)
        assert record["inversion_applied"] == geometry["inversion_applied"]
        assert record["photometric"] == source["photometric"]
        assert record["orientation_preserved"] and not record["spatial_crop_applied"]
        assert record["output_shape"] == [1, 224, 224] and record["dtype"] == "float32"
        left, top, right, bottom = record["padding_ltrb"]
        height, width = record["resized_shape"]
        assert height + top + bottom == width + left + right == 224
        assert 0 <= right - left <= 1 and 0 <= bottom - top <= 1 and max(height, width) == 224
        assert all(abs(x) <= 1 for x in record["rounding_error_pixels"])
        for border in (tensor[:, :top, :], tensor[:, :, :left],
                       tensor[:, 224 - bottom:, :] if bottom else tensor[:, 0:0, :],
                       tensor[:, :, 224 - right:] if right else tensor[:, :, 0:0]):
            assert np.count_nonzero(border) == 0
        assert sha_file(native_path) == native_sha
        assert hashlib.sha256(native.tobytes(order="C")).hexdigest() == before
        inversion_count += record["inversion_applied"]
        photo_counts[source["photometric"]] += 1
        shapes.append((height, width))
    assert not closure["training_executed"] and not closure["partitions_created"]
    assert closure["routine_application_executed"] and not closure["cohort_membership_changed"]
    assert not closure["clinical_or_predictive_validation_claimed"]
    assert manifest["summary"]["photometric_counts"] == dict(photo_counts)
    assert manifest["summary"]["resized_height_range"] == [min(h for h, _ in shapes), max(h for h, _ in shapes)]
    assert manifest["summary"]["resized_width_range"] == [min(w for _, w in shapes), max(w for _, w in shapes)]
    for relative, sha in manifest["qa_files_sha256"].items():
        assert sha_file(safe(output, relative)) == sha
    assert sha_file(config_path) == config_sha
    for name in ("membership", "cohort_provenance", "contract", "development_closure", "development_manifest"):
        assert sha_file(config[name]["path"]) == config[name]["sha256"]
    return {"version": "routine_input_independent_verification_v1.0", "status": "passed",
        "routine_knees_checked": len(eligible), "routine_participants_checked": expected_participants,
        "development_images_processed": 0, "routine_images_processed": len(eligible),
        "whole_cohort_input_keys_checked": len(all_membership),
        "whole_cohort_participants_checked": len({pair[0] for pair in all_membership}),
        "development_manifest_keys_checked_without_pixels": len(development_keys),
        "development_and_routine_inputs_disjoint_and_complete": True,
        "same_sealed_development_contract_verified": True,
        "native_hashes_unchanged": True, "derived_file_and_pixel_hashes_verified": True,
        "finite_float32_1x224x224": True, "centered_zero_padding_checked": True,
        "physical_aspect_and_percentile_windows_checked": True,
        "all_pixels_independently_reconstructed": True, "orientation_and_no_flip_checked": True,
        "monochrome1_inversions_checked": inversion_count, "development_arrays_opened": False,
        "outcome_rows_loaded": False, "training_executed": False, "partitions_created": False,
        "closure_summary_sha256": sha_file(closure_path), "verifier_sha256": sha_file(__file__)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.record.exists():
        assert args.record.read_text(encoding="utf-8") == text, "existing_verification_changed"
    else:
        with args.record.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print(text, end="")
