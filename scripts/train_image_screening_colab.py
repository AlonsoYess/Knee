"""Prepare or run only the confirmed five-encoder development screening."""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from knee import image_campaign_recovery as orchestration
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, choices=("setup", "tests", "locate", "train"))
    parser.add_argument("--campaign", required=True, choices=("remaining_encoders_frozen",))
    parser.add_argument("--repository-root", required=True, type=Path)
    for name in ("destination", "records-root", "preparation-dir", "package-dir", "reference-path",
                 "weights-root", "output-dir", "test-receipt", "runtime-receipt"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--code-revision")
    parser.add_argument("--approval-statement")
    args = parser.parse_args()
    if args.operation == "setup":
        if args.destination is None:
            parser.error("setup requires --destination")
        result = orchestration.setup_runtime(args.repository_root, args.destination, args.campaign)
    elif args.operation == "tests":
        result = orchestration.run_tests(args.repository_root, campaign=args.campaign)
    elif args.operation == "locate":
        if args.records_root is None:
            parser.error("locate requires --records-root")
        proof33 = orchestration.find_image_preflight(args.records_root, args.repository_root)
        result = {"preparation_dir": str(orchestration.find_preparation(args.records_root)),
                  "checkpoint_preflight_dir": str(proof33)}
    else:
        required = ("preparation_dir", "package_dir", "reference_path", "output_dir", "records_root",
                    "weights_root", "approval_statement", "code_revision", "test_receipt", "runtime_receipt")
        if any(getattr(args, name) is None for name in required):
            parser.error("train requires development-only inputs, cache, runtime/tests and explicit confirmation")
        result = orchestration.train(args.repository_root, args.preparation_dir, args.package_dir,
            args.reference_path, args.output_dir, args.records_root, campaign=args.campaign,
            approval_statement=args.approval_statement, code_revision=args.code_revision,
            cpu_tests=orchestration._json(args.test_receipt), runtime=orchestration._json(args.runtime_receipt),
            weights_root=args.weights_root)
        orchestration._save(args.output_dir / orchestration.SUMMARY_FILENAME, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
