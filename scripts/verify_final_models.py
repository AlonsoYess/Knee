"""Verify final registered model weights on synthetic CPU inputs; never train."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knee.final_models_preflight import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--execution-environment", choices=("local_cpu", "google_colab_cpu"), default="local_cpu")
    args = parser.parse_args()
    result = run(args.repository_root, args.weights_root, args.output_dir,
                 allow_download=args.allow_download, execution_environment=args.execution_environment)
    print(json.dumps({key: result[key] for key in ("version", "status", "registered_image_models",
        "pretrained_models_loaded", "pending_model_ids", "training_executed", "colab_runtime_executed",
        "participant_data_opened", "architecture_search_closed")}, indent=2))


if __name__ == "__main__":
    main()
