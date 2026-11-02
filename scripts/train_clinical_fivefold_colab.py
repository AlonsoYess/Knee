"""Prepare or execute the three registered clinical five-fold CPU campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import clinical_fivefold_colab as orchestration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, choices=("setup", "tests", "locate", "train"))
    parser.add_argument("--repository-root", required=True, type=Path)
    for name in ("destination", "records-root", "preparation-dir", "package-dir", "reference-path",
            "output-dir", "test-receipt", "runtime-receipt"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--code-revision")
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
        required = ("preparation_dir", "package_dir", "reference_path", "output_dir", "records_root",
            "approval_statement", "code_revision", "test_receipt", "runtime_receipt")
        if any(getattr(args, name) is None for name in required):
            parser.error("train requires authenticated preparation, development, CPU receipts and exact approval")
        result = orchestration.train(args.repository_root, args.preparation_dir, args.package_dir,
            args.reference_path, args.output_dir, args.records_root, approval_statement=args.approval_statement,
            code_revision=args.code_revision, cpu_tests=orchestration._json(args.test_receipt),
            runtime=orchestration._json(args.runtime_receipt))
        orchestration._save(args.output_dir / orchestration.SUMMARY_FILENAME, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
