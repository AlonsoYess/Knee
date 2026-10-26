"""Verify only the protected development dataset, printing aggregate results."""
import argparse
import json

from knee.colab_data import verify_development_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--record")
    args = parser.parse_args()
    print(json.dumps(verify_development_package(args.package, args.expected_sha256, args.record),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
