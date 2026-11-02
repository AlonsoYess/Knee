"""Check public registered model weights and the plan; never train a model."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee.pretrained_verification import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--weights-root", required=True, help="Private/local cache; not inside the public repository")
    parser.add_argument("--output-dir", required=True, help="New private directory for the verification JSON")
    parser.add_argument("--allow-download", action="store_true", help="Download only the four public, pinned checkpoints")
    parser.add_argument("--execution-environment", choices=("local_cpu", "google_colab_cpu"), default="local_cpu")
    args = parser.parse_args()
    run(args.repository_root, args.weights_root, args.output_dir,
        allow_download=args.allow_download, execution_environment=args.execution_environment)


if __name__ == "__main__":
    main()
