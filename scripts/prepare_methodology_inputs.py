"""Prepare aligned clinical development inputs, with no predictor training."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--development-package-dir", required=True, type=Path)
    parser.add_argument("--development-package-sha256", required=True)
    parser.add_argument("--input-reference", required=True, type=Path)
    parser.add_argument("--baseline-development-projection", required=True, type=Path,
        help="Baseline-only JSON list of exact development keys, from audited V00; not the full cohort.")
    parser.add_argument("--baseline-projection-sha256", required=True,
        help="Externally established SHA256 of the development-only baseline projection.")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from knee.methodology_preparation import prepare
    result = prepare(args.repository_root, args.development_package_dir, args.input_reference,
        args.baseline_development_projection, args.baseline_projection_sha256, args.output_dir,
        expected_package_sha256=args.development_package_sha256)
    # Private row identities and paths remain in the separately selected output.
    print(json.dumps({key: result[key] for key in ("version", "status", "development_participants",
        "development_knees", "clinical_preprocessing_states_verified", "prepared_clinical_requests",
        "training_authorized", "training_executed", "optimizer_created",
        "reserved_test_predictive_evaluation_opened", "originals_modified", "partitions_modified")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
