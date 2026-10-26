"""Apply the development-frozen image-input contract to sealed routine cases."""
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
from knee.development_model_inputs import _blob, _png, _sheet
from knee.model_input_tensor import CONTRACT, VERSION as TENSOR_VERSION, normalize_native, transform_native
from knee.roi_global_retention_monitor import checked_path
from knee.roi_global_retention_sources import digest, read_inventory, require, safe, sha_file


VERSION = "routine_model_inputs_224_v1.0"
SCOPE_FLAGS = {"native_inputs_unchanged": True, "all_repeated_transforms_identical": True,
    "all_finite_float32_1x224x224": True, "physical_aspect_preserved_with_pixel_rounding": True,
    "whole_native_crop_retained": True, "orientation_preserved": True,
    "outcome_rows_loaded": False, "training_executed": False, "partitions_created": False,
    "native_edits": False, "qc_regraded": False, "drive_upload_performed": False}


def _context(config_path):
    config_path = checked_path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(set(config) == {"version", "project_root", "membership", "cohort_provenance", "cohort_closure",
        "inventory", "contract", "authority", "development_closure", "development_manifest", "component_pins", "output_root",
        "private_sources_root", "original_recovery_record"}
        and config["version"] == VERSION, "exact_routine_inputs_config_required")
    pins = {str(config_path): sha_file(config_path)}
    project = checked_path(config["project_root"], directory=True)
    require(checked_path(__file__) == project / "src/knee/routine_model_inputs.py"
        and checked_path(transform_native.__code__.co_filename) == project / "src/knee/model_input_tensor.py"
        and checked_path(inspect_npy.__code__.co_filename) == project / "src/knee/common_cohort_assets.py",
        "adopted_routine_input_modules_required")
    components = {"src/knee/model_input_tensor.py", "src/knee/routine_model_inputs.py", "src/knee/routine_dicom_headers.py",
        "src/knee/development_model_inputs.py", "src/knee/development_dicom_headers.py",
        "src/knee/common_cohort.py", "src/knee/common_cohort_assets.py"}
    require(set(config["component_pins"]) == components, "exact_routine_input_components_required")
    for relative, sha in config["component_pins"].items():
        path = checked_path(project / relative)
        require(sha_file(path) == sha, "routine_input_component_changed")
        pins[str(path)] = sha
    authority = read_pinned(config["authority"])
    require(authority.get("version") == VERSION and authority.get("approved_by") == "researcher"
        and authority.get("scope") == "accepted_routine_apply_development_frozen_224_only" and authority.get("user_statement")
        and authority.get("allowed_actions") == ["restore_pinned_originals_from_existing_drive", "routine_dicom_header_audit", "routine_native_read",
            "same_frozen_tensor_write", "routine_visualization", "technical_verification", "routine_input_seal"]
        and authority.get("forbidden_actions") == ["training", "partitioning", "outcome_rows_access", "qc_regrading",
            "native_edit", "transform_retuning", "new_selection", "drive_upload", "data_deletion", "development_output_edit"],
        "routine_input_authority_changed")
    membership, provenance, contract, devclosed, devmanifest = (read_pinned(config[k]) for k in
        ("membership", "cohort_provenance", "contract", "development_closure", "development_manifest"))
    require(devclosed["status"] == "development_tensor_contract_verified_routine_application_pending"
        and devclosed["manifest_sha256"] == config["development_manifest"]["sha256"]
        and devclosed["tensor_contract_version"] == devmanifest["tensor_contract_version"] == TENSOR_VERSION
        and devmanifest["tensor_contract_sha256"] == digest(CONTRACT)
        and devmanifest["source_files_sha256"][str(checked_path(config["contract"]["path"]))] == config["contract"]["sha256"]
        and devmanifest["source_files_sha256"][str(project / "src/knee/model_input_tensor.py")]
            == config["component_pins"]["src/knee/model_input_tensor.py"], "development_frozen_transform_changed")
    require(contract == CONTRACT and np.__version__ == CONTRACT["runtime"]["numpy"]
        and PIL.__version__ == CONTRACT["runtime"]["Pillow"] and pydicom.__version__ == "3.0.1",
        "frozen_routine_tensor_runtime_required")
    closure = checked_path(config["cohort_closure"]["path"])
    require(sha_file(closure) == config["cohort_closure"]["sha256"], "common_cohort_closure_changed")
    require(membership["outcomes_used_for_membership"] is False, "outcome_free_routine_membership_required")
    roots = {k: checked_path(v, directory=True) for k,v in provenance["native_roots"].items()}
    roots["private_sources_root"] = checked_path(config["private_sources_root"],directory=True)
    selected = [row for row in membership["rows"] if row["role"] == "routine"]
    ids = {row["src_subject_id"] for row in selected}
    require(len(selected) == membership["summary"]["routine_knees"] and selected
        and len(ids) == membership["summary"]["routine_participants"]
        and not ids & set(membership["development_reservations"])
        and len({(r["src_subject_id"],r["side"]) for r in selected}) == len(selected),
        "exact_frozen_routine_membership_required")
    output = checked_path(config["output_root"], directory=True)
    require(output.name == "routine_inputs_224_001" and all(not output.is_relative_to(p) for p in roots.values())
        and not output.is_relative_to(Path(config["membership"]["path"]).parent)
        and not output.is_relative_to(Path(config["development_manifest"]["path"]).parent), "dedicated_routine_output_required")
    require(checked_path(config["authority"]["path"]).is_relative_to(output / "governance"), "own_routine_input_authority_required")
    for k in ("membership", "cohort_provenance", "cohort_closure", "inventory", "contract", "authority",
              "development_closure", "development_manifest", "original_recovery_record"):
        pins[str(checked_path(config[k]["path"]))] = config[k]["sha256"]
    require(sha_file(checked_path(config["original_recovery_record"]["path"])) == config["original_recovery_record"]["sha256"],
        "original_recovery_record_changed")
    inventory = checked_path(config["inventory"]["path"])
    require(sha_file(inventory) == config["inventory"]["sha256"], "routine_inventory_pin_changed")
    return config, selected, roots, output, pins, read_inventory(inventory)


def prepare(config_path):
    config, selected, roots, output, pins, inventory = _context(config_path)
    from knee.routine_dicom_headers import audit_routine_headers
    progress = lambda done,total: print(json.dumps({"stage":"routine_original_header_audit","completed":done,"total":total}),flush=True)
    audit = audit_routine_headers(selected, roots, inventory, progress=progress)
    require(audit["summary"]["compatible_with_fixed_quantile_native_recipe"] is True,
        "routine_dicom_intensity_incompatible_no_retuning_allowed")
    for path,sha in audit["pins"].items():
        require(path not in pins or pins[path] == sha, "conflicting_routine_dicom_pin")
        pins[path] = sha
    write_once(output / "dicom_headers.json", audit)
    records, tiles, visual, pair_candidates = [], [], {}, {}
    extrema = {}
    groups = {}
    for index,row in enumerate(selected,1):
        require(row["role"] == "routine", "only_accepted_routine_native_access")
        crop = row["crop"]
        native_path = safe(roots[crop["root_key"]],crop["relative_path"])
        inspect_npy(native_path,expected_file_sha=crop["sha256"],expected_pixel_sha=crop["pixel_sha256"],
                    shape=crop["shape"],dtype=crop["dtype"],verify_payloads=True)
        native = np.load(native_path,allow_pickle=False)
        native.setflags(write=False)
        require(hashlib.sha256(native.tobytes(order="C")).hexdigest() == crop["pixel_sha256"],"native_pixel_pin_changed")
        tensor,record = transform_native(native,row,CONTRACT)
        again,again_record = transform_native(native,row,CONTRACT)
        require(np.array_equal(tensor,again) and record == again_record,"routine_tensor_not_deterministic")
        require(sha_file(native_path) == crop["sha256"]
            and hashlib.sha256(native.tobytes(order="C")).hexdigest() == crop["pixel_sha256"],"routine_native_modified")
        require(tensor.shape == (1,224,224) and tensor.dtype == np.float32 and tensor.flags.c_contiguous
            and np.isfinite(tensor).all() and tensor.min() >= 0 and tensor.max() <= 1
            and max(abs(x) for x in record["rounding_error_pixels"]) <= 1,"routine_tensor_array_contract_failed")
        relative = f"tensors/routine_{index:04d}.npy"
        buffer = io.BytesIO()
        np.save(buffer,tensor,allow_pickle=False)
        file_sha = _blob(safe(output,relative),buffer.getvalue())
        records.append({"index":index,"src_subject_id":row["src_subject_id"],"side":row["side"],"role":"routine",
            "block":row["block"],"native_crop":crop,"source_native_sha256":crop["sha256"],"transform":record,
            "tensor":{"relative_path":relative,"sha256":file_sha,
                "pixel_sha256":hashlib.sha256(tensor.tobytes(order="C")).hexdigest(),"shape":[1,224,224],"dtype":"float32"}})
        pins[str(native_path)] = crop["sha256"]
        tile = Image.new("RGB",(240,254),"#ececec")
        ImageDraw.Draw(tile).text((8,6),f"R {index:04d} / {row['side']}",fill="black")
        gray = Image.fromarray(np.rint(tensor[0]*255).astype(np.uint8)).convert("RGB")
        tile.paste(gray,(8,26))
        tiles.append(tile)
        if len(tiles) == 20 or index == len(selected):
            relative = f"qa/sheet_{(index-1)//20+1:03d}.png"
            visual[relative] = _png(safe(output,relative),_sheet(tiles))
            tiles = []
        h,w = record["resized_shape"]
        group = (row["photometric"],row["side"])
        needs_pair = group not in groups
        if needs_pair:
            groups[group] = index
        for name,value,prefer in (("min_h",h,min),("max_h",h,max),("min_w",w,min),("max_w",w,max)):
            if name not in extrema or prefer(value,extrema[name][0]) != extrema[name][0]:
                extrema[name] = (value,index)
                needs_pair = True
        retained = set(groups.values()) | {item[1] for item in extrema.values()}
        pair_candidates = {i:p for i,p in pair_candidates.items() if i in retained}
        if needs_pair:
            normalized,_ = normalize_native(native,row)
            source_view = Image.fromarray(np.rint(normalized*255).astype(np.uint8)).convert("RGB")
            source_view = source_view.resize((w,h),Image.Resampling.NEAREST)
            pair = Image.new("RGB",(480,254),"#ececec")
            draw = ImageDraw.Draw(pair)
            draw.text((8,6),f"R {index:04d}: ROI ventana fija",fill="black")
            draw.text((248,6),"Entrada 224 x 224",fill="black")
            pair.paste(source_view,(8+(224-w)//2,26+(224-h)//2))
            pair.paste(gray,(248,26))
            pair_candidates[index] = pair
        if index % 100 == 0 or index == len(selected):
            print(json.dumps({"stage":"routine_tensors_prepared","completed":index,"total":len(selected)}),flush=True)
    comparison = Image.new("RGB",(960,254*math.ceil(len(pair_candidates)/2)),"#ececec")
    for n,(_,pair) in enumerate(sorted(pair_candidates.items())):
        comparison.paste(pair,((n%2)*480,(n//2)*254))
    visual["qa/reference_comparisons.png"] = _png(safe(output,"qa/reference_comparisons.png"),comparison)
    for path,sha in pins.items():
        require(sha_file(checked_path(path)) == sha,"routine_input_changed_during_preparation")
    manifest = {"version":VERSION,"tensor_contract_version":TENSOR_VERSION,"tensor_contract_sha256":digest(CONTRACT),
        "development_manifest_sha256":config["development_manifest"]["sha256"],
        "rows":records,"selection_rule":"all_frozen_accepted_membership_rows_with_routine_role",
        "source_files_sha256":dict(sorted(pins.items())),"qa_files_sha256":visual,
        "reference_comparison_indices":sorted(pair_candidates),
        "summary":{"participants":len({r["src_subject_id"] for r in records}),"knees":len(records),
            "routine_images_processed":len(records),"development_images_processed":0,
            "photometric_counts":dict(Counter(r["transform"]["photometric"] for r in records)),
            "resized_height_range":[min(r["transform"]["resized_shape"][0] for r in records),max(r["transform"]["resized_shape"][0] for r in records)],
            "resized_width_range":[min(r["transform"]["resized_shape"][1] for r in records),max(r["transform"]["resized_shape"][1] for r in records)],
            **SCOPE_FLAGS,"dicom_header_audit":audit["summary"]},
        "execution_environment":{"python":platform.python_version(),"numpy":np.__version__,"Pillow":PIL.__version__,"pydicom":pydicom.__version__}}
    reused = write_once(output / "manifest.json",manifest)
    return {"status":"routine_inputs_numerically_verified_pending_visual_review","summary":manifest["summary"],
        "manifest_sha256":sha_file(output / "manifest.json"),"existing_manifest_reused":reused,
        "qa_sheet_count":len(visual)-1,"reference_comparison_indices":sorted(pair_candidates)}


def close(config_path,review_path):
    config,selected,roots,output,pins,inventory = _context(config_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("version") == VERSION and manifest.get("tensor_contract_version") == TENSOR_VERSION
        and manifest.get("tensor_contract_sha256") == digest(CONTRACT)
        and all(manifest.get("source_files_sha256",{}).get(path) == sha for path,sha in pins.items()),
        "adopted_routine_tensor_manifest_required")
    review = json.loads(checked_path(review_path).read_text(encoding="utf-8"))
    require(review.get("version") == VERSION and review.get("manifest_sha256") == sha_file(manifest_path)
        and review.get("reviewed_qa_files_sha256") == manifest["qa_files_sha256"]
        and review.get("routine_outputs_visually_inspected") == len(selected)
        and review.get("technical_visualization_passed") is True
        and review.get("review_scope") == "transformation_geometry_polarity_and_display_only_not_anatomical_regrading"
        and review.get("reviewer") == "Codex_assisted_technical_review","complete_routine_technical_visual_review_required")
    require(len(manifest["rows"]) == len(selected)
        and {(r["src_subject_id"],r["side"]) for r in manifest["rows"]} == {(r["src_subject_id"],r["side"]) for r in selected}
        and all(r["role"] == "routine" for r in manifest["rows"]),"routine_tensor_membership_changed")
    summary = manifest["summary"]
    require(all(summary.get(k) is v for k,v in SCOPE_FLAGS.items())
        and all(type(summary.get(k)) is int for k in ("participants","knees","routine_images_processed","development_images_processed"))
        and summary.get("knees") == summary.get("routine_images_processed") == len(selected)
        and summary.get("development_images_processed") == 0
        and summary.get("participants") == len({r["src_subject_id"] for r in selected}),"routine_tensor_scope_or_checks_changed")
    for path,sha in manifest["source_files_sha256"].items():
        require(sha_file(checked_path(path)) == sha,"routine_source_changed_before_closure")
    for row in manifest["rows"]:
        require(sha_file(safe(output,row["tensor"]["relative_path"])) == row["tensor"]["sha256"],"routine_tensor_changed_before_closure")
    for relative,sha in manifest["qa_files_sha256"].items():
        require(sha_file(safe(output,relative)) == sha,"routine_visual_qa_changed")
    result = {"version":VERSION,"status":"accepted_routine_inputs_224_technically_verified",
        "tensor_contract_version":TENSOR_VERSION,"summary":summary,"manifest_sha256":sha_file(manifest_path),
        "technical_visual_review_sha256":sha_file(review_path),"development_contract_applied_unchanged":True,
        "model_specific_rgb_and_normalization":"pending_selected_encoder_integration",
        "routine_application_executed":True,"training_executed":False,"partitions_created":False,
        "cohort_membership_changed":False,"clinical_or_predictive_validation_claimed":False,
        "next_step":"participant_level_partition_plan_and_verified_private_training_package_before_training"}
    reused = write_once(output / "closure_summary.json",result)
    return dict(result,existing_closure_reused=reused)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--close-review",type=Path)
    args = parser.parse_args()
    result = close(args.config,args.close_review) if args.close_review else prepare(args.config)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
