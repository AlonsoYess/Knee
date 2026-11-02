"""Resolve local/Drive paths without storing private locations in Git."""

import json
import os
from pathlib import Path


PATH_KEYS = ("cohort_csv", "manifest_xlsx", "audit_json", "run_record_json")


def load_paths(config_path: str | Path) -> dict[str, Path]:
    root_value = os.environ.get("KNEE_DATA_ROOT")
    if not root_value:
        raise ValueError("Set KNEE_DATA_ROOT to the authorized data directory.")
    root = Path(root_value).expanduser().resolve()
    with Path(config_path).open(encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict) or any(key not in values for key in PATH_KEYS):
        raise ValueError("Path configuration is missing a required key.")

    paths = {}
    for key in PATH_KEYS:
        value = values[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid relative path for {key}.")
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"The path for {key} must stay below KNEE_DATA_ROOT.")
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"The path for {key} escapes KNEE_DATA_ROOT.")
        paths[key] = resolved
    return paths

