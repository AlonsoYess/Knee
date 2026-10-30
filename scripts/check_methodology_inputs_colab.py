"""Notebook38 CPU preparation and transformation check; never train a model."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, choices=("setup", "tests", "verify", "save"))
    parser.add_argument("--repository-root", required=True, type=Path)
    for name in ("destination", "package-dir", "reference-path", "projection-path", "descriptor-path",
                 "output-dir", "records-root", "test-receipt", "runtime-receipt"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--execution-environment", choices=("google_colab_cpu", "local_cpu"), default="google_colab_cpu")
    parser.add_argument("--code-revision")
    args = parser.parse_args()
    from knee import methodology_inputs_colab as orchestration
    if args.operation == "setup":
        if args.destination is None:
            parser.error("setup requires --destination")
        result = orchestration.setup_runtime(args.repository_root, args.destination)
    elif args.operation == "tests":
        result = orchestration.run_tests(args.repository_root)
    elif args.operation == "verify":
        required = ("package_dir", "reference_path", "projection_path", "descriptor_path", "output_dir")
        if any(getattr(args, name) is None for name in required):
            parser.error("verify requires development package, reference, baseline projection, descriptor and new output")
        result = orchestration.verify(args.repository_root, args.package_dir, args.reference_path,
            args.projection_path, args.descriptor_path, args.output_dir, execution_environment=args.execution_environment)
        result = {key: result[key] for key in ("version", "status", "development_participants", "development_knees",
            "clinical_preprocessing_states_verified", "prepared_clinical_requests", *orchestration.FALSE_FLAGS)}
    else:
        required = ("output_dir", "records_root", "test_receipt", "runtime_receipt", "code_revision")
        if any(getattr(args, name) is None for name in required):
            parser.error("save requires local result, records root, exact revision and CPU receipts")
        result = orchestration.save_result(args.repository_root, args.output_dir, args.records_root,
            code_revision=args.code_revision, cpu_tests=orchestration._json(args.test_receipt),
            runtime=orchestration._json(args.runtime_receipt), execution_environment=args.execution_environment)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
