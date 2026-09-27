"""Create an aggregate, reproducible phase-0 run record outside Git."""

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import openpyxl

from knee.config import load_paths


def _revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    paths = load_paths(args.config)
    report = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
    if report.get("status") != "ok":
        raise SystemExit("Cohort audit has not passed; run record was not created.")
    record = {
        "phase": "0_data_integrity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _revision(),
        "source_sha256": report["source_sha256"],
        "counts": report["counts"],
        "environment": {
            "python": platform.python_version(),
            "openpyxl": openpyxl.__version__,
            "platform": platform.platform(),
        },
        "audit_status": report["status"],
    }
    target = paths["run_record_json"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Run record written under KNEE_DATA_ROOT ({target.name}).")


if __name__ == "__main__":
    main()

