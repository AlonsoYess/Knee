"""Future development-fold training CLI; separate explicit authority required."""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--development-package-dir", required=True, type=Path)
    parser.add_argument("--input-reference", required=True, type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--selection-evidence", type=Path)
    parser.add_argument("--image-job-dir", type=Path)
    args = parser.parse_args()
    from knee.training_runner import execute, read_json
    result = execute(read_json(args.request), read_json(args.authority), repository_root=args.repository_root,
        package_dir=args.development_package_dir, reference_path=args.input_reference, weights_root=args.weights_root,
        output_dir=args.output_dir, device=args.device, allow_download=args.allow_download,
        selection_evidence=args.selection_evidence, image_job_dir=args.image_job_dir)
    # Identifiers and fold predictions remain only in private output files.
    print(json.dumps({key:result[key] for key in ("version", "scope", "engine_status", "training_executed",
        "eligible_for_model_selection", "reserved_test_accessed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
