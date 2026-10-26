"""Split the sealed development package for bounded private Drive transport."""
import argparse
import json

from knee.colab_package_transport import split_development_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=96000000)
    args = parser.parse_args()
    result = split_development_package(args.package_root, args.output_root, chunk_bytes=args.chunk_bytes)
    print(json.dumps({"version": result["version"], "parts": len(result["parts"]),
        "bytes": result["development"]["bytes"], "sha256": result["development"]["sha256"],
        "reserved_test_opened": False, "training_executed": False}, indent=2))


if __name__ == "__main__":
    main()
