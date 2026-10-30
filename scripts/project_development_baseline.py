"""Project audited V00 baseline fields for the sealed development package only."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--development-package-dir", required=True, type=Path)
    parser.add_argument("--development-package-sha256", required=True)
    parser.add_argument("--input-reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from knee.aligned_training.data import load_sealed_inputs
    from knee.baseline_development_projection import prepare_projection
    loaded = load_sealed_inputs(args.development_package_dir, args.input_reference,
        args.repository_root, args.development_package_sha256)
    record = prepare_projection(args.source_csv, args.source_sha256, loaded["manifest"]["rows"],
        args.output_dir, development_package_sha256=args.development_package_sha256)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
