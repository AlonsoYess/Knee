"""Prepare or execute only the authorized four-model gradual Colab campaign."""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from knee import gradual_campaign_colab as orchestration
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, choices=("setup", "tests", "prepare", "train"))
    parser.add_argument("--campaign", required=True, choices=("gradual_four_models",))
    parser.add_argument("--repository-root", required=True, type=Path)
    for name in ("destination", "records-root", "preparation-dir", "package-dir", "reference-path",
            "weights-root", "output-dir", "test-receipt", "runtime-receipt"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--code-revision")
    args = parser.parse_args()
    if args.operation == "setup":
        if args.destination is None:
            parser.error("setup requires --destination")
        result = orchestration.setup_runtime(args.repository_root, args.destination, args.campaign)
    elif args.operation == "tests":
        result = orchestration.run_tests(args.repository_root, args.campaign)
    elif args.operation == "prepare":
        required = ("records_root", "package_dir", "reference_path", "output_dir", "code_revision",
            "test_receipt", "runtime_receipt")
        if any(getattr(args, name) is None for name in required):
            parser.error("prepare requires prior private records, development package and verified runtime/tests")
        result = orchestration.prepare(args.repository_root, args.records_root, args.package_dir,
            args.reference_path, args.output_dir, code_revision=args.code_revision,
            cpu_tests=orchestration._json(args.test_receipt), runtime=orchestration._json(args.runtime_receipt),
            campaign=args.campaign)
    else:
        required = ("records_root", "preparation_dir", "package_dir", "reference_path", "output_dir",
            "weights_root", "code_revision")
        if any(getattr(args, name) is None for name in required):
            parser.error("train requires the verified gradual preparation and development-only paths")
        result = orchestration.train(args.repository_root, args.preparation_dir, args.package_dir,
            args.reference_path, args.output_dir, weights_root=args.weights_root,
            persistent_root=args.records_root, code_revision=args.code_revision, campaign=args.campaign)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
