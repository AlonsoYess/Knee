"""Prepare and technically verify development-only model inputs, without fitting."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import platform

import numpy as np
import PIL
from PIL import Image, ImageDraw
import pydicom

from knee.common_cohort import read_pinned, write_once
from knee.common_cohort_assets import inspect_npy
from knee.roi_global_retention_monitor import checked_path
from knee.roi_global_retention_sources import digest, read_inventory, require, safe, sha_file
from knee.model_input_tensor import CONTRACT, VERSION as TENSOR_VERSION, normalize_native, transform_native


VERSION = "development_only_model_inputs_224_v1.0"


def _blob(path, data):
    require(not path.is_symlink(), "derived_output_link_forbidden")
    expected = hashlib.sha256(data).hexdigest()
    if path.exists():
        require(sha_file(path) == expected, "existing_derived_output_changed")
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    require(sha_file(path) == expected, "derived_output_readback_failed")
    return expected


def _png(path, image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _blob(path, buffer.getvalue())


def _context(config_path):
    config_path = checked_path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(set(config) == {"version", "project_root", "membership", "cohort_provenance",
        "cohort_closure", "inventory", "contract", "authority", "component_pins", "output_root"}
        and config["version"] == VERSION, "exact_development_inputs_config_required")
    pins = {str(config_path): sha_file(config_path)}
    project = checked_path(config["project_root"], directory=True)
    require(checked_path(__file__) == project / "src/knee/development_model_inputs.py",
            "adopted_development_input_module_required")
    require(checked_path(transform_native.__code__.co_filename) == project / "src/knee/model_input_tensor.py"
        and checked_path(inspect_npy.__code__.co_filename) == project / "src/knee/common_cohort_assets.py",
        "adopted_tensor_dependencies_required")
    files = {"src/knee/model_input_tensor.py", "src/knee/development_model_inputs.py",
        "src/knee/development_dicom_headers.py", "src/knee/common_cohort_assets.py",
        "src/knee/common_cohort.py"}
    require(set(config["component_pins"]) == files, "exact_development_input_components_required")
    for relative, sha in config["component_pins"].items():
        path = checked_path(project / relative)
        require(sha_file(path) == sha, "development_input_component_changed")
        pins[str(path)] = sha
    authority = read_pinned(config["authority"])
    require(authority.get("version") == VERSION and authority.get("approved_by") == "researcher"
        and authority.get("scope") == "development_accepted_native_to_224_only"
        and authority.get("user_statement")
        and authority.get("allowed_actions") == ["development_dicom_header_audit", "development_native_read",
            "derived_tensor_write", "development_visualization", "technical_verification", "contract_seal"]
        and authority.get("forbidden_actions") == ["routine_pixel_access", "outcome_rows_access", "training",
            "partitioning", "qc_regrading", "native_edit", "new_selection", "drive_upload", "data_deletion"],
        "development_input_authority_changed")
    membership, provenance, contract = (read_pinned(config[k]) for k in ("membership", "cohort_provenance", "contract"))
    # Hash the parent closure without loading its outcome aggregates.
    closure = checked_path(config["cohort_closure"]["path"])
    require(sha_file(closure) == config["cohort_closure"]["sha256"], "parent_cohort_closure_changed")
    require(contract == CONTRACT and np.__version__ == CONTRACT["runtime"]["numpy"]
        and PIL.__version__ == CONTRACT["runtime"]["Pillow"] and pydicom.__version__ == "3.0.1",
        "frozen_tensor_runtime_or_contract_required")
    require(membership["outcomes_used_for_membership"] is False, "outcome_free_membership_required")
    roots = {k: checked_path(v, directory=True) for k, v in provenance["native_roots"].items()}
    selected = [row for row in membership["rows"] if row["role"] == "development"]
    require(len(selected) == membership["summary"]["development_knees"]
        and len({r["src_subject_id"] for r in selected}) == membership["summary"]["development_participants"]
        and {r["src_subject_id"] for r in selected} <= set(membership["development_reservations"]),
        "exact_accepted_development_membership_required")
    require(len({(r["src_subject_id"], r["side"]) for r in selected}) == len(selected), "unique_development_inputs_required")
    output = checked_path(config["output_root"], directory=True)
    require(output.name == "development_inputs_224_001"
        and all(not output.is_relative_to(root) for root in roots.values())
        and not output.is_relative_to(Path(config["membership"]["path"]).parent), "dedicated_tensor_output_required")
    require(checked_path(config["authority"]["path"]).is_relative_to(output / "governance"), "own_tensor_authority_required")
    for k in ("membership", "cohort_provenance", "cohort_closure", "inventory", "contract", "authority"):
        pins[str(checked_path(config[k]["path"]))] = config[k]["sha256"]
    inventory = checked_path(config["inventory"]["path"])
    require(sha_file(inventory) == config["inventory"]["sha256"], "pinned_tensor_inventory_changed")
    return config, selected, roots, output, pins, read_inventory(inventory)


def _sheet(tiles, *, columns=5):
    width, height = 240, 254
    sheet = Image.new("RGB", (columns*width, math.ceil(len(tiles)/columns)*height), "#ececec")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns)*width, (i // columns)*height))
    return sheet


def prepare(config_path):
    config, selected, roots, output, pins, inventory = _context(config_path)
    from knee.development_dicom_headers import audit_development_headers
    audit = audit_development_headers(selected, roots, inventory)
    require(audit["summary"]["compatible_with_fixed_quantile_native_recipe"] is True,
            "development_dicom_intensity_policy_unresolved")
    for path, sha in audit["pins"].items():
        require(path not in pins or pins[path] == sha, "conflicting_dicom_pin")
        pins[path] = sha
    write_once(output / "dicom_headers.json", audit)
    records, tiles, comparison_candidates, all_deterministic = [], [], [], True
    for index, row in enumerate(selected, 1):
        require(row["role"] == "development", "routine_native_access_forbidden")
        native_path = safe(roots[row["crop"]["root_key"]], row["crop"]["relative_path"])
        crop = row["crop"]
        inspect_npy(native_path, expected_file_sha=crop["sha256"], expected_pixel_sha=crop["pixel_sha256"],
                    shape=crop["shape"], dtype=crop["dtype"], verify_payloads=True)
        native = np.load(native_path, allow_pickle=False)
        native.setflags(write=False)
        before = hashlib.sha256(native.tobytes(order="C")).hexdigest()
        tensor, record = transform_native(native, row, CONTRACT)
        again, repeated_record = transform_native(native, row, CONTRACT)
        require(np.array_equal(tensor, again) and record == repeated_record, "nondeterministic_development_tensor")
        require(hashlib.sha256(native.tobytes(order="C")).hexdigest() == before == crop["pixel_sha256"]
            and sha_file(native_path) == crop["sha256"], "native_modified_during_tensor_conversion")
        require(tensor.shape == (1,224,224) and tensor.dtype == np.float32
            and tensor.flags.c_contiguous and np.isfinite(tensor).all()
            and tensor.min() >= 0 and tensor.max() <= 1, "model_input_array_contract_failed")
        # Quantization of the shorter side is at most half a pixel, except the
        # explicit minimum-one-pixel guard for extremely narrow fields.
        require(max(abs(x) for x in record["rounding_error_pixels"]) <= 1,
                "physical_aspect_rounding_bound_failed")
        relative = f"tensors/dev_{index:03d}.npy"
        buffer = io.BytesIO()
        np.save(buffer, tensor, allow_pickle=False)
        file_sha = _blob(safe(output, relative), buffer.getvalue())
        records.append({"index": index, "src_subject_id": row["src_subject_id"], "side": row["side"],
            "role": "development", "block": row["block"], "native_crop": crop,
            "source_native_sha256": crop["sha256"], "transform": record,
            "tensor": {"relative_path": relative, "sha256": file_sha,
                "pixel_sha256": hashlib.sha256(tensor.tobytes(order="C")).hexdigest(),
                "shape": [1,224,224], "dtype": "float32"}})
        pins[str(native_path)] = crop["sha256"]
        tile = Image.new("RGB", (240,254), "#ececec")
        ImageDraw.Draw(tile).text((8,6), f"DEV {index:03d} / {row['side']}", fill="black")
        gray = Image.fromarray(np.rint(tensor[0]*255).astype(np.uint8)).convert("RGB")
        tile.paste(gray, (8,26))
        tiles.append(tile)
        normalized, _ = normalize_native(native, row)
        source_view = Image.fromarray(np.rint(normalized*255).astype(np.uint8)).convert("RGB")
        # QA reference preserves physical aspect; training always reads float32.
        h, w = record["resized_shape"]
        source_view = source_view.resize((w,h), Image.Resampling.NEAREST)
        pair = Image.new("RGB", (480,254), "#ececec")
        draw = ImageDraw.Draw(pair)
        draw.text((8,6), f"DEV {index:03d}: ROI ventana fija", fill="black")
        draw.text((248,6), "Entrada 224 x 224", fill="black")
        pair.paste(source_view, (8+(224-w)//2,26+(224-h)//2))
        pair.paste(gray, (248,26))
        comparison_candidates.append((row["photometric"], row["side"], h, w, index, pair))
    visual = {}
    for offset in range(0, len(tiles), 20):
        relative = f"qa/sheet_{offset//20+1:03d}.png"
        visual[relative] = _png(safe(output, relative), _sheet(tiles[offset:offset+20]))
    chosen = {min(comparison_candidates, key=lambda x:x[2])[4], max(comparison_candidates, key=lambda x:x[2])[4],
              min(comparison_candidates, key=lambda x:x[3])[4], max(comparison_candidates, key=lambda x:x[3])[4]}
    for photo in {r[0] for r in comparison_candidates}:
        for side in {r[1] for r in comparison_candidates}:
            same = [r for r in comparison_candidates if r[0] == photo and r[1] == side]
            if same:
                chosen.add(same[0][4])
    pairs = [r[5] for r in comparison_candidates if r[4] in chosen]
    comparison = Image.new("RGB", (960,254*math.ceil(len(pairs)/2)), "#ececec")
    for n, pair in enumerate(pairs):
        comparison.paste(pair, ((n%2)*480,(n//2)*254))
    visual["qa/reference_comparisons.png"] = _png(safe(output,"qa/reference_comparisons.png"), comparison)
    for path, sha in pins.items():
        require(sha_file(checked_path(path)) == sha, "input_changed_during_development_preparation")
    manifest = {"version": VERSION, "tensor_contract_version": TENSOR_VERSION,
        "tensor_contract_sha256": digest(CONTRACT), "rows": records,
        "selection_rule": "all_accepted_membership_rows_with_permanent_development_role",
        "source_files_sha256": dict(sorted(pins.items())), "qa_files_sha256": visual,
        "summary": {"participants": len({r["src_subject_id"] for r in records}), "knees": len(records),
            "photometric_counts": dict(Counter(r["transform"]["photometric"] for r in records)),
            "resized_height_range": [min(r["transform"]["resized_shape"][0] for r in records),max(r["transform"]["resized_shape"][0] for r in records)],
            "resized_width_range": [min(r["transform"]["resized_shape"][1] for r in records),max(r["transform"]["resized_shape"][1] for r in records)],
            "native_inputs_unchanged": True, "all_repeated_transforms_identical": all_deterministic,
            "all_finite_float32_1x224x224": True, "physical_aspect_preserved_with_pixel_rounding": True,
            "whole_native_crop_retained": True, "orientation_preserved": True,
            "dicom_header_audit": audit["summary"], "routine_images_processed": 0,
            "outcome_rows_loaded": False, "training_executed": False, "partitions_created": False,
            "native_edits": False, "qc_regraded": False, "drive_upload_performed": False},
        "execution_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                  "Pillow": PIL.__version__, "pydicom": pydicom.__version__}}
    reused = write_once(output / "manifest.json", manifest)
    return {"status": "development_inputs_numerically_verified_pending_visual_review", "summary": manifest["summary"],
        "manifest_sha256": sha_file(output / "manifest.json"), "existing_manifest_reused": reused,
        "qa_files": list(visual), "native_inputs_unchanged": True}


def close(config_path, review_path):
    config, selected, roots, output, pins, inventory = _context(config_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("version") == VERSION and manifest.get("tensor_contract_version") == TENSOR_VERSION
        and manifest.get("tensor_contract_sha256") == digest(CONTRACT)
        and all(manifest.get("source_files_sha256",{}).get(path) == sha for path,sha in pins.items()),
        "adopted_tensor_manifest_and_source_pins_required")
    review = json.loads(checked_path(review_path).read_text(encoding="utf-8"))
    require(review.get("version") == VERSION and review.get("manifest_sha256") == sha_file(manifest_path)
        and review.get("reviewed_qa_files_sha256") == manifest["qa_files_sha256"]
        and review.get("development_outputs_visually_inspected") == len(selected)
        and review.get("technical_visualization_passed") is True
        and review.get("review_scope") == "transformation_geometry_polarity_and_display_only_not_anatomical_regrading"
        and review.get("reviewer") == "Codex_assisted_technical_review", "complete_technical_visual_review_required")
    require(len(manifest["rows"]) == len(selected)
        and {(r["src_subject_id"],r["side"]) for r in manifest["rows"]}
        == {(r["src_subject_id"],r["side"]) for r in selected}, "development_manifest_membership_changed")
    for path, sha in manifest["source_files_sha256"].items():
        require(sha_file(checked_path(path)) == sha, "source_changed_before_tensor_closure")
    for row in manifest["rows"]:
        path = safe(output,row["tensor"]["relative_path"])
        require(sha_file(path) == row["tensor"]["sha256"], "derived_tensor_changed_before_closure")
    for relative, sha in manifest["qa_files_sha256"].items():
        require(sha_file(safe(output,relative)) == sha, "technical_qa_image_changed")
    summary = {"version": VERSION, "status": "development_tensor_contract_verified_routine_application_pending",
        "tensor_contract_version": TENSOR_VERSION, "summary": manifest["summary"],
        "manifest_sha256": sha_file(manifest_path), "technical_visual_review_sha256": sha_file(review_path),
        "model_specific_rgb_and_normalization": "pending_selected_encoder_integration",
        "routine_application_executed": False, "training_executed": False, "partitions_created": False,
        "cohort_membership_changed": False, "clinical_or_predictive_validation_claimed": False,
        "next_step": "apply_identical_frozen_transform_to_remaining_accepted_routine_inputs_before_partitions"}
    reused = write_once(output / "closure_summary.json", summary)
    return dict(summary, existing_closure_reused=reused)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--close-review", type=Path)
    args = parser.parse_args()
    result = close(args.config,args.close_review) if args.close_review else prepare(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
