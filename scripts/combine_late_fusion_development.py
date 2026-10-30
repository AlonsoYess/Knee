"""Combine sealed development predictions with fixed equal weights; never train."""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--development-package-dir", required=True, type=Path)
    parser.add_argument("--input-reference", required=True, type=Path)
    parser.add_argument("--development-package-sha256", required=True)
    parser.add_argument("--clinical-job-index", required=True, type=Path)
    parser.add_argument("--image-job-index", required=True, type=Path)
    parser.add_argument("--selected-image-encoder", required=True)
    parser.add_argument("--image-selection", required=True, type=Path)
    parser.add_argument("--image-selection-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from knee.methodology_release import load_release
    from knee.aligned_training.data import load_sealed_inputs
    from knee.aligned_training.runner import read_json
    from knee.late_fusion import combine_development_jobs
    _, _, plan, registry = load_release(args.repository_root)
    # The sole dataset loader accepts the authenticated development package and
    # rejects reserved-test membership before it reads tensor payloads.
    loaded = load_sealed_inputs(args.development_package_dir, args.input_reference,
        args.repository_root, args.development_package_sha256)
    result = combine_development_jobs(read_json(args.clinical_job_index), read_json(args.image_job_index),
        loaded["manifest"]["rows"], package_sha256=args.development_package_sha256,
        plan=plan, registry=registry, selected_image_encoder_id=args.selected_image_encoder,
        image_selection_path=args.image_selection, image_selection_sha256=args.image_selection_sha256,
        output_dir=args.output_dir)
    # Person-level probabilities and identifiers remain only in the private file.
    print(json.dumps({key: result[key] for key in ("version", "status", "model_id", "scope",
        "clinical_model_id", "image_encoder_id", "weights", "development_knees",
        "training_executed", "reserved_test_accessed", "unbiased_final_performance_claimed")}, indent=2))


if __name__ == "__main__":
    main()
