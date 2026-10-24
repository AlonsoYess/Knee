"""Prepare/test the registered executor in CPU Colab without executing training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import training_preparation_colab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, choices=("setup", "tests", "locate", "prepare"))
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--source-preflight-dir", type=Path)
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execution-environment", default="google_colab_cpu", choices=("google_colab_cpu", "local_cpu"))
    args = parser.parse_args()
    if args.operation == "setup":
        if args.destination is None:
            parser.error("setup requires --destination")
        result = training_preparation_colab.setup_runtime(args.repository_root, args.destination)
    elif args.operation == "tests":
        result = training_preparation_colab.run_tests(args.repository_root)
    elif args.operation == "locate":
        if args.records_root is None:
            parser.error("locate requires --records-root")
        result = {"source_preflight_dir": str(training_preparation_colab.find_previous_preflight(args.records_root))}
    else:
        if args.source_preflight_dir is None or args.output_dir is None:
            parser.error("prepare requires --source-preflight-dir and --output-dir")
        result = training_preparation_colab.prepare(args.repository_root, args.source_preflight_dir,
            args.output_dir, execution_environment=args.execution_environment)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
