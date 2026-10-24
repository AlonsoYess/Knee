"""Run exact notebook cells in a fresh isolated local worker without Drive writes.

Only Drive mounting, interactive path entry, GitHub cloning, and the Linux runtime
installation helper are replaced. Checkout verification, unit tests, transport
staging, CPU input checking, and result publication execute the actual cells.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types


VERSION = "model_inputs_notebook_local_rehearsal_v1.0"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def checked(value, *, directory=False):
    raw = Path(value).absolute()
    for item in (raw, *raw.parents):
        require(not item.is_symlink() and not getattr(item, "is_junction", lambda: False)(),
                "rehearsal_linked_path_forbidden")
    require(raw.is_dir() if directory else raw.is_file(), "rehearsal_input_missing")
    return raw.resolve()


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def snapshot(root):
    result = {}
    for path in sorted(root.rglob("*")):
        checked(path, directory=path.is_dir())
        relative = path.relative_to(root).as_posix()
        result[relative] = {"kind": "directory"} if path.is_dir() else {
            "kind": "file", "sha256": sha_file(path), "bytes": path.stat().st_size}
    return result


def _record_path(value, protected):
    if value is None:
        return None
    raw = Path(value).absolute()
    for item in (raw, *raw.parents):
        require(not item.is_symlink() and not getattr(item, "is_junction", lambda: False)(),
                "rehearsal_record_link_forbidden")
    path = raw.resolve()
    require(path.parent.is_dir() and not path.exists(), "new_rehearsal_record_in_existing_directory_required")
    require(all(not path.is_relative_to(root) for root in protected),
            "rehearsal_record_must_remain_outside_inputs")
    return path


def worker(args):
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "fresh_isolated_no_bytecode_worker_required")
    require(not any(name == "knee" or name.startswith("knee.") for name in sys.modules),
            "scientific_modules_preloaded_before_notebook")
    notebook = checked(args.notebook)
    repository = checked(args.repository_root, directory=True)
    transport = checked(args.transport_root, directory=True)
    reference = checked(args.reference_record)
    python = checked(args.python)
    require(Path(sys.executable).resolve() == python, "worker_must_use_declared_frozen_python")
    record_path = _record_path(args.record, [transport, reference.parent, repository])
    original_hashes = {"notebook": sha_file(notebook), "reference": sha_file(reference)}
    original_transport = snapshot(transport)
    require("transport.json" in original_transport, "sealed_transport_manifest_required")
    notebook_value = json.loads(notebook.read_text(encoding="utf-8"))
    require(notebook_value.get("nbformat") == 4 and isinstance(notebook_value.get("cells"), list),
            "valid_notebook_required")
    namespace = {"__name__": "__exact_notebook_local_rehearsal__", "_LOCAL_REHEARSAL": True}
    original_run, original_input, original_mkdtemp = subprocess.run, builtins.input, tempfile.mkdtemp
    previous_env = {k: os.environ.get(k) for k in ("KNEE_DATA_ROOT", "KNEE_INPUT_REFERENCE_PATH")}
    previous_google = {key: sys.modules.get(key) for key in ("google", "google.colab", "google.colab.drive")}
    mount_calls, substitutions, owned_sessions, executed = [], [], [], []
    # TemporaryDirectory owns only new rehearsal files, including the notebook's
    # session. Its cleanup handles read-only Windows Git files within this root.
    with tempfile.TemporaryDirectory(prefix="knee_notebook_rehearsal_") as owned_name:
        owned_root = Path(owned_name).resolve()
        require(all(not owned_root.is_relative_to(p) and not p.is_relative_to(owned_root)
                    for p in (transport, reference.parent, repository)), "separate_owned_temporary_root_required")

        def session_mkdtemp(*positional, **kwargs):
            require(not positional and kwargs == {"prefix": "knee_input_check_"},
                    "unexpected_notebook_temporary_directory")
            path = Path(original_mkdtemp(prefix=kwargs["prefix"], dir=owned_root)).resolve()
            require(path.parent == owned_root, "owned_session_path_escape")
            owned_sessions.append(path)
            return str(path)

        def interactive_path(prompt=""):
            substitutions.append("interactive_transport_path")
            return str(transport)

        def local_run(arguments, **kwargs):
            require(not isinstance(arguments, str), "argument_list_required_for_notebook_process")
            command = [str(x) for x in arguments]
            if command[:2] == ["git", "clone"]:
                require(command[:-2] == ["git", "clone", "--quiet", "--config", "core.autocrlf=false",
                                        "--filter=blob:none", "--no-checkout"]
                        and command[-2] == "https://github.com/AlonsoYess/Knee.git"
                        and len(owned_sessions) == 1
                        and Path(command[-1]).resolve() == owned_sessions[0] / "repository",
                        "unexpected_notebook_clone_command")
                substitutions.append("github_clone_replaced_by_committed_local_git_clone")
                options = [value for value in command[2:-2] if value != "--filter=blob:none"]
                return original_run(["git", "clone", *options, "--no-hardlinks",
                                     str(repository), command[-1]], **kwargs)
            if len(command) >= 3 and Path(command[2]).name == "setup_model_inputs_cpu.py":
                require(len(command) == 7 and command[1] == "-B" and command[3] == "--destination"
                        and command[5] == "--repository-root" and len(owned_sessions) == 1
                        and Path(command[2]).resolve() == owned_sessions[0] / "repository/scripts/setup_model_inputs_cpu.py"
                        and Path(command[4]).resolve() == owned_sessions[0] / "cpu_runtime"
                        and Path(command[6]).resolve() == owned_sessions[0] / "repository",
                        "unexpected_notebook_runtime_setup_command")
                substitutions.append("linux_runtime_install_replaced_by_verified_frozen_python")
                return subprocess.CompletedProcess(arguments, 0,
                    stdout=json.dumps({"python_path": str(python), "local_runtime_reused": True}), stderr="")
            return original_run(arguments, **kwargs)

        google = types.ModuleType("google")
        google.__path__ = []
        colab, drive = types.ModuleType("google.colab"), types.ModuleType("google.colab.drive")
        drive.mount = lambda location: mount_calls.append(location)
        colab.drive, google.colab = drive, colab
        sys.modules.update({"google": google, "google.colab": colab, "google.colab.drive": drive})
        os.environ.pop("KNEE_DATA_ROOT", None)
        os.environ["KNEE_INPUT_REFERENCE_PATH"] = str(reference)
        subprocess.run, builtins.input, tempfile.mkdtemp = local_run, interactive_path, session_mkdtemp
        try:
            for index, cell in enumerate(notebook_value["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                source = cell.get("source", [])
                text = "".join(source) if isinstance(source, list) else source
                require(isinstance(text, str), "notebook_cell_source_required")
                print(f"Rehearsal: exact notebook cell {index}", flush=True)
                exec(compile(text, f"{notebook}:cell_{index}", "exec"), namespace)
                executed.append(index)
            require(len(owned_sessions) == 1 and mount_calls == ["/content/drive"],
                    "single_notebook_session_and_drive_mount_stub_required")
            session = checked(namespace["SESSION_ROOT"], directory=True)
            require(session == owned_sessions[0] and session.parent == owned_root,
                    "notebook_session_not_owned_by_rehearsal")
            report_path = checked(namespace["FINAL_REPORT_PATH"])
            require(report_path.is_relative_to(session), "local_rehearsal_report_escaped_session")
            final = json.loads(report_path.read_text(encoding="utf-8"))
            require(final == namespace["FINAL_CHECK"] and final.get("status") == "passed"
                    and final.get("local_rehearsal") is True and final.get("colab_runtime_executed") is False
                    and final.get("training_executed") is False
                    and final.get("reserved_test_predictive_evaluation_opened") is False,
                    "local_only_no_training_notebook_result_required")
            require(snapshot(transport) == original_transport
                    and sha_file(reference) == original_hashes["reference"]
                    and sha_file(notebook) == original_hashes["notebook"],
                    "original_notebook_transport_or_reference_changed")
            pinned = namespace["CODE_REVISION"]
            actual = original_run(["git", "rev-parse", "HEAD"], cwd=namespace["REPO_DIR"],
                                  check=True, capture_output=True, text=True).stdout.strip()
            require(actual == pinned, "actual_checkout_does_not_match_notebook_revision")
            record = {"version": VERSION, "status": "passed", "fresh_isolated_worker": True,
                "scientific_modules_preloaded_before_notebook": False, "exact_code_cells_executed": executed,
                "notebook_sha256": original_hashes["notebook"], "reference_sha256": original_hashes["reference"],
                "transport_files_and_directories_unchanged": True, "transport_snapshot": original_transport,
                "reference_unchanged": True, "code_revision": actual, "python": str(python),
                "substitutions": substitutions, "drive_mount_stub_calls": mount_calls,
                "linux_runtime_install_executed": False, "network_git_clone_executed": False,
                "local_rehearsal": True, "colab_runtime_executed": False,
                "training_executed": False, "reserved_test_predictive_evaluation_opened": False,
                "notebook_final_record": final,
                "local_published_files_sha256": {p.relative_to(report_path.parent).as_posix(): sha_file(p)
                    for p in sorted(report_path.parent.rglob("*")) if p.is_file()},
                "owned_temporary_session_cleanup_requested": True}
            if record_path is not None:
                with record_path.open("xb") as stream:
                    stream.write(canonical(record))
                require(json.loads(record_path.read_text(encoding="utf-8")) == record,
                        "rehearsal_record_readback_changed")
            print("REHEARSAL_RECORD=" + json.dumps(record, sort_keys=True), flush=True)
            return record
        finally:
            subprocess.run, builtins.input, tempfile.mkdtemp = original_run, original_input, original_mkdtemp
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            for name, value in previous_google.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--transport-root", required=True)
    parser.add_argument("--reference-record", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--record")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        worker(args)
        return
    python = checked(args.python)
    command = [str(python), "-I", "-B", str(Path(__file__).resolve()), "--_worker"]
    for name in ("notebook", "repository_root", "transport_root", "reference_record", "python", "record"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "KNEE_DATA_ROOT"):
        environment.pop(name, None)
    completed = subprocess.run(command, env=environment)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
