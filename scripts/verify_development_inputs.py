"""Independently check development model inputs and their unchanged native ROIs."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from knee.roi_global_retention_sources import safe, sha_file


def verify(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(config["output_root"])
    membership = json.loads(Path(config["membership"]["path"]).read_text(encoding="utf-8"))
    provenance = json.loads(Path(config["cohort_provenance"]["path"]).read_text(encoding="utf-8"))
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    closure = json.loads((output/"closure_summary.json").read_text(encoding="utf-8"))
    assert sha_file(config["membership"]["path"]) == config["membership"]["sha256"]
    assert sha_file(output/"manifest.json") == closure["manifest_sha256"]
    key = lambda row: (row["src_subject_id"],row["side"])
    eligible = {key(row):row for row in membership["rows"] if row["role"] == "development"}
    assert len(eligible) == len(manifest["rows"])
    assert {key(row) for row in manifest["rows"]} == set(eligible)
    assert {row["role"] for row in manifest["rows"]} == {"development"}
    assert manifest["summary"]["routine_images_processed"] == 0
    assert not manifest["summary"]["outcome_rows_loaded"]
    shapes, inversion_count = [], 0
    for row in manifest["rows"]:
        source = eligible[key(row)]
        native_path = safe(provenance["native_roots"][source["crop"]["root_key"]],source["crop"]["relative_path"])
        assert sha_file(native_path) == source["crop"]["sha256"] == row["source_native_sha256"]
        native = np.load(native_path,allow_pickle=False)
        assert hashlib.sha256(native.tobytes(order="C")).hexdigest() == source["crop"]["pixel_sha256"]
        path = safe(output,row["tensor"]["relative_path"])
        assert sha_file(path) == row["tensor"]["sha256"]
        tensor = np.load(path,allow_pickle=False)
        assert tensor.shape == (1,224,224) and tensor.dtype == np.float32
        assert np.isfinite(tensor).all() and tensor.min() >= 0 and tensor.max() <= 1 and tensor.std() > 0
        assert hashlib.sha256(tensor.tobytes(order="C")).hexdigest() == row["tensor"]["pixel_sha256"]
        record = row["transform"]
        left,top,right,bottom = record["padding_ltrb"]
        height,width = record["resized_shape"]
        assert height + top + bottom == width + left + right == 224
        assert 0 <= right-left <= 1 and 0 <= bottom-top <= 1
        for border in (tensor[:,:top,:],tensor[:,:, :left],tensor[:,224-bottom:,:] if bottom else tensor[:,0:0,:],
                       tensor[:,:,224-right:] if right else tensor[:,:,0:0]):
            assert np.count_nonzero(border) == 0
        physical_h = native.shape[0]*source["row_spacing_mm"]
        physical_w = native.shape[1]*source["column_spacing_mm"]
        scale = 224/max(physical_h,physical_w)
        assert max(height,width) == 224
        assert abs(height-physical_h*scale) <= 1 and abs(width-physical_w*scale) <= 1
        window = np.quantile(native.astype(np.float64),[.01,.99],method="linear")
        np.testing.assert_allclose(record["percentile_window"],window,rtol=0,atol=1e-9)
        assert record["inversion_applied"] == (source["photometric"] == "MONOCHROME1")
        assert record["orientation_preserved"] and not record["spatial_crop_applied"]
        inversion_count += record["inversion_applied"]
        shapes.append((height,width))
    assert not closure["training_executed"] and not closure["partitions_created"]
    assert not closure["routine_application_executed"] and not closure["cohort_membership_changed"]
    for relative,sha in manifest["qa_files_sha256"].items():
        assert sha_file(safe(output,relative)) == sha
    return {"version":"development_input_independent_verification_v1.0","status":"passed",
        "development_knees_checked":len(eligible),
        "development_participants_checked":len({pair[0] for pair in eligible}),
        "native_hashes_unchanged":True,"derived_file_and_pixel_hashes_verified":True,
        "finite_float32_1x224x224":True,"centered_zero_padding_checked":True,
        "physical_aspect_and_percentile_windows_checked":True,
        "monochrome1_inversions_checked":inversion_count,"routine_arrays_opened":False,
        "outcome_rows_loaded":False,"training_executed":False,
        "closure_summary_sha256":sha_file(output/"closure_summary.json"),
        "verifier_sha256":sha_file(__file__)}


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--record",type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.config)
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if args.record.exists():
        assert args.record.read_text(encoding="utf-8") == text,"existing_verification_changed"
    else:
        with args.record.open("x",encoding="utf-8",newline="\n") as stream:
            stream.write(text)
    print(text,end="")
