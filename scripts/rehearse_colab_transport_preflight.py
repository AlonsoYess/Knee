"""Execute the exact fragment notebook in a fresh local CPU Python process.

Only Google Drive mounting and the directory input are replaced. A rehearsal
flag prevents any remote-directory report write and cannot certify Colab.
"""
from __future__ import annotations

import argparse
import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types


BASE_NOTEBOOK_SHA256 = "231ca2151400028c19cf57c46419c7a3ef57ff9f0fb2c399ac2e04c3c13e8018"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, code):
    if not condition:
        raise ValueError(code)


def notebook_code(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "code"]


def verify_preserved_cells(base_path, notebook_path):
    require(sha(base_path) == BASE_NOTEBOOK_SHA256, "frozen_original_notebook_changed")
    base, updated = notebook_code(base_path), notebook_code(notebook_path)
    require(len(base) == 3 and len(updated) == 5, "expected_notebook_code_cells_required")
    transformed = "\n".join("PACKAGE_DIR = LOCAL_PACKAGE_DIR" if line.startswith("PACKAGE_DIR = Path(input(") else line
        for line in base[0].split("\n") if line != "drive.mount('/content/drive')")
    require(updated[1] == transformed and updated[2:4] == base[1:3], "frozen_verification_cells_changed")
    return updated


def worker(args):
    require("knee" not in sys.modules, "fresh_rehearsal_process_required")
    remote, source = Path(args.remote_root).absolute(), Path(args.package_root).absolute()
    notebook, original = Path(args.notebook).absolute(), Path(args.base_notebook).absolute()
    cells = verify_preserved_cells(original, notebook)
    protected = [source / name for name in ("development_package.zip", "colab_preparation_code.zip", "handoff.json", "closure_summary.json")]
    protected += [original, notebook]
    before = {str(path): sha(path) for path in protected}
    handoff = json.loads((source / "handoff.json").read_text(encoding="utf-8"))
    require(before[str(source / "development_package.zip")] == handoff["development"]["sha256"], "original_development_package_changed")
    manifest = json.loads((remote / "transport.json").read_text(encoding="utf-8"))
    paths = [remote / "transport.json", remote / manifest["transport_module"]["file"]]
    paths += [remote / name for name in manifest["support_files"]]
    paths += [remote / spec["file"] for spec in manifest["parts"]]
    remote_before = {str(path): sha(path) for path in paths}
    directory_before = {path.name for path in remote.iterdir()}
    google, colab, drive = (types.ModuleType(name) for name in ("google", "google.colab", "google.colab.drive"))
    mounts = []
    drive.mount = lambda location: mounts.append(location)
    colab.drive = drive
    google.colab = colab
    sys.modules.update({"google": google, "google.colab": colab, "google.colab.drive": drive})
    previous_input = builtins.input
    builtins.input = lambda prompt="": str(remote)
    namespace = {"__name__": "__colab_rehearsal__", "_LOCAL_REHEARSAL": True}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for index, cell in enumerate(cells, 1):
                exec(compile(cell, str(notebook) + f":cell{index}", "exec"), namespace)
        final = namespace["FINAL_PREFLIGHT"]
        local = namespace["LOCAL_PACKAGE_DIR"]
        require(final["colab_runtime_executed"] is False and final["local_rehearsal"] is True
            and final["training_executed"] is False and final["reserved_test_predictive_evaluation_opened"] is False,
            "rehearsal_must_not_claim_remote_or_training_execution")
        require(namespace["FINAL_REPORT_PATH"].parent == local and not (local / "reserved_test_package.zip").exists(),
            "local_only_rehearsal_outputs_required")
        require(sha(local / "development_package.zip") == handoff["development"]["sha256"]
            and (local / "development_package.zip").stat().st_size == handoff["development"]["bytes"],
            "reconstructed_original_package_not_identical")
        require({str(path): sha(path) for path in protected} == before
            and {str(path): sha(path) for path in paths} == remote_before
            and {path.name for path in remote.iterdir()} == directory_before, "source_or_remote_transport_modified")
        require(mounts == ["/content/drive"], "only_one_stubbed_mount_required")
        result = {**final, "exact_notebook_cells_executed_locally": True,
            "only_google_drive_mount_and_path_input_stubbed": True,
            "all_original_packages_and_transport_unchanged": True,
            "original_notebook_sha256": before[str(original)], "notebook_sha256": before[str(notebook)],
            "transport_sha256": remote_before[str(remote / "transport.json")],
            "no_remote_report_written": True}
    finally:
        builtins.input = previous_input
        for name in ("CODE_DIRECTORY", "LOCAL_PACKAGE_DIRECTORY"):
            if name in namespace:
                namespace[name].cleanup()
    result["temporary_rehearsal_copies_removed"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--base-notebook", required=True)
    parser.add_argument("--record")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = worker(args)
    else:
        command = [sys.executable, "-I", "-B", str(Path(__file__).absolute()), "--worker",
            "--remote-root", args.remote_root, "--package-root", args.package_root,
            "--notebook", args.notebook, "--base-notebook", args.base_notebook]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        require(completed.returncode == 0, "isolated_notebook_rehearsal_failed: " + completed.stderr[-4000:])
        result = json.loads(completed.stdout)
        if args.record:
            path = Path(args.record).absolute()
            require(path.parent.is_dir() and not path.exists(), "new_rehearsal_record_path_required")
            with path.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
