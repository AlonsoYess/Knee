"""Run only the approved CPU setup/tests/synthetic encoder checks for notebook33."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import final_models_colab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, choices=("setup", "tests", "verify"))
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--execution-environment", default="google_colab_cpu", choices=("google_colab_cpu", "local_cpu"))
    args = parser.parse_args()
    if args.operation == "setup":
        if args.destination is None:
            parser.error("setup requires --destination")
        result = final_models_colab.setup_runtime(args.repository_root, args.destination)
    elif args.operation == "tests":
        result = final_models_colab.run_tests(args.repository_root)
    else:
        if args.weights_root is None or args.output_dir is None:
            parser.error("verify requires --weights-root and --output-dir")
        result = final_models_colab.verify(args.repository_root, args.weights_root, args.output_dir,
            allow_download=args.allow_download, execution_environment=args.execution_environment)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
