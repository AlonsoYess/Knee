"""Prepare, test or execute only the separately authorized two-update GPU smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import training_smoke_colab as orchestration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, choices=("setup", "tests", "locate", "smoke"))
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--preparation-dir", type=Path)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--approval-statement")
    args = parser.parse_args()
    if args.operation == "setup":
        if args.destination is None:
            parser.error("setup requires --destination")
        result = orchestration.setup_runtime(args.repository_root, args.destination)
    elif args.operation == "tests":
        result = orchestration.run_tests(args.repository_root)
    elif args.operation == "locate":
        if args.records_root is None:
            parser.error("locate requires --records-root")
        result = {"preparation_dir": str(orchestration.find_preparation(args.records_root))}
    else:
        names = ("preparation_dir", "package_dir", "reference_path", "weights_root", "output_dir", "approval_statement")
        if any(getattr(args, name) is None for name in names):
            parser.error("smoke requires exact preparation, sealed development, reference, weights, output and approval")
        result = orchestration.smoke(args.repository_root, args.preparation_dir, args.package_dir,
            args.reference_path, args.weights_root, args.output_dir, approval_statement=args.approval_statement)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
