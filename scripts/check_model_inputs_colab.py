"""Verify frozen development inputs and CPU architectures against prior hashes."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee.model_inputs_colab import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--reference-record", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-environment", required=True, choices=("google_colab_cpu", "local_rehearsal"))
    args = parser.parse_args()
    run(args.package_dir, args.reference_record, args.repository_root, args.output_dir, args.execution_environment)


if __name__ == "__main__":
    main()
