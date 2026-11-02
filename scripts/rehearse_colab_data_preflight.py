"""Execute the exact Colab preparation cells locally, with only mount/input stubbed.

This is explicitly NOT a remote Colab run. The rehearsal directory contains no
reserved-test archive; only disposable copies of the development handoff exist.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve(strict=True)
    require(root.is_dir() and not args.package_root.is_symlink(), "unlinked_package_root_required")
    closure = json.loads((root / "closure_summary.json").read_text(encoding="utf-8"))
    handoff_path = root / "handoff.json"
    require(sha(handoff_path) == closure["handoff_sha256"], "sealed_handoff_required")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    notebook_path = root / "31_verificar_paquete_desarrollo_sin_entrenar.ipynb"
    notebook_sha = sha(notebook_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source_names = ["handoff.json", handoff["development"]["file"], handoff["code"]["file"]]
    require(source_names == ["handoff.json", "development_package.zip", "colab_preparation_code.zip"],
            "development_only_rehearsal_sources_required")
    originals = {name: sha(root / name) for name in source_names + ["reserved_test_package.zip"]}
    require(originals["development_package.zip"] == handoff["development"]["sha256"]
            and originals["colab_preparation_code.zip"] == handoff["code"]["sha256"]
            and originals["reserved_test_package.zip"] == handoff["reserved_test"]["sha256"],
            "sealed_package_hashes_required")
    require(not any(name == "knee" or name.startswith("knee.") for name in sys.modules),
            "fresh_process_without_repository_imports_required")
    captured = io.StringIO()
    mounted = []
    with tempfile.TemporaryDirectory(prefix=".development_rehearsal_", dir=root) as temporary:
        staging = Path(temporary).resolve(strict=True)
        require(staging.parent == root and staging.name.startswith(".development_rehearsal_"),
                "task_owned_rehearsal_directory_required")
        for name in source_names:
            shutil.copyfile(root / name, staging / name)
            require(sha(staging / name) == originals[name], "rehearsal_copy_changed")
        require(not (staging / "reserved_test_package.zip").exists(), "reserved_test_must_not_enter_rehearsal")
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=lambda path: mounted.append(path))
        google.colab = colab
        sys.modules["google"], sys.modules["google.colab"] = google, colab
        old_input = builtins.input
        builtins.input = lambda prompt="": str(staging)
        namespace = {"__name__": "__main__"}
        try:
            with redirect_stdout(captured):
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), f"notebook_cell_{index}", "exec"), namespace)
        finally:
            builtins.input = old_input
        summary = namespace["summary"]
        require(summary["status"] == "development_package_verified_no_training"
                and summary["knees"] == handoff["development"]["knees"]
                and summary["participants"] == handoff["development"]["participants"]
                and summary["reserved_test_opened"] is False and summary["training_executed"] is False,
                "complete_development_only_notebook_result_required")
        require(mounted == ["/content/drive"] and not (staging / "reserved_test_package.zip").exists(),
                "mount_stub_or_rehearsal_scope_changed")
        local_record = json.loads((staging / "colab_development_verification.json").read_text(encoding="utf-8"))
        require(local_record == summary, "notebook_record_mismatch")
        # Notebook owns another temporary directory for verified public code.
        namespace["CODE_DIRECTORY"].cleanup()
    require(all(sha(root / name) == expected for name, expected in originals.items())
            and sha(notebook_path) == notebook_sha, "source_packages_changed_during_rehearsal")
    result = {"version": "colab_development_local_rehearsal_v1.0", "status": "passed",
        "colab_runtime_executed": False, "exact_notebook_cells_executed_locally": True,
        "only_google_drive_mount_and_path_input_stubbed": True, "notebook_sha256": notebook_sha,
        "handoff_sha256": originals["handoff.json"], "development_package_sha256": originals["development_package.zip"],
        "verified_development_participants": summary["participants"], "verified_development_tensors": summary["knees"],
        "reserved_test_archive_present_in_rehearsal": False, "reserved_test_predictive_evaluation_opened": False,
        "all_original_packages_unchanged": True, "temporary_rehearsal_copies_removed": True,
        "training_executed": False, "imputation_performed": False}
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    require(args.record.is_absolute() and args.record.parent.resolve() == root
            and not args.record.is_symlink(), "private_rehearsal_record_required")
    if args.record.exists():
        require(args.record.read_bytes() == payload, "immutable_rehearsal_record_changed")
    else:
        with args.record.open("xb") as stream:
            stream.write(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
