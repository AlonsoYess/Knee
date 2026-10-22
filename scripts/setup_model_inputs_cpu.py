"""Install the pinned Linux CPU interpreter without changing the Colab kernel.

Standalone stdlib bootstrap: no Knee import, patient data, model checkpoint,
training, or host-environment installation. Progress goes to stderr; successful
CLI stdout contains exactly one JSON result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request


VERSION = "isolated_model_inputs_cpu_runtime_v1.0"
PYTHON_URL = "https://github.com/astral-sh/python-build-standalone/releases/download/20260929/cpython-3.12.14%2B20260929-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PYTHON_SHA = "ef605200f8174e87ecfc308e52a88127543f85dd5c940dc5e92cab244b98a003"
WHEEL_PINS = {
    "torch": ("https://download-r2.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl",
              "59e78aa0c690f70734e42670036d6b541930b8eabbaa18d94e090abf14cc4d91"),
    "torchvision": ("https://download-r2.pytorch.org/whl/cpu/torchvision-0.21.0%2Bcpu-cp312-cp312-linux_x86_64.whl",
                   "d6874431e678ba107b60a83f255c33f3755f06bad587b1b919aa514ec325dcd8"),
}
EXPECTED_PACKAGES = {
    "numpy": "1.26.4", "Pillow": "11.3.0", "filelock": "4.0.9", "typing_extensions": "4.16.0",
    "networkx": "3.7", "Jinja2": "3.1.6", "fsspec": "2026.9.0", "setuptools": "84.0.0",
    "sympy": "1.13.1", "mpmath": "1.3.0", "MarkupSafe": "3.0.3",
    "torch": "2.6.0+cpu", "torchvision": "0.21.0+cpu",
}
HTTPS_HOSTS = {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com",
               "download-r2.pytorch.org", "download.pytorch.org"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _inside(path, root):
    return path == root or root in path.parents


def _checked(path):
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        require(not ancestor.is_symlink() and not getattr(ancestor, "is_junction", lambda: False)(),
                "bootstrap_path_links_forbidden")
    return path.resolve()


def _url(url):
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == "https" and parsed.hostname in HTTPS_HOSTS
            and parsed.username is None and parsed.password is None and parsed.port in {None, 443},
            "bootstrap_https_host_not_allowed")


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        _url(newurl)
        return super().redirect_request(request, response, code, message, headers, newurl)


def download(spec, target):
    """Stream one fixed pinned artifact with a hard upper byte bound."""
    allowed = {PYTHON_URL: PYTHON_SHA, **{url: sha for url, sha in WHEEL_PINS.values()}}
    require(allowed.get(spec["url"]) == spec["sha256"], "unregistered_bootstrap_artifact")
    _url(spec["url"])
    target = _checked(target)
    require(not target.exists(), "bootstrap_download_target_already_exists")
    opener = urllib.request.build_opener(HTTPSRedirects())
    with opener.open(spec["url"], timeout=60) as response:
        _url(response.geturl())
        length = response.headers.get("Content-Length")
        if length is not None:
            require(str(length).isdigit() and 0 < int(length) <= spec["max_bytes"], "bootstrap_download_size_limit")
        value, count = hashlib.sha256(), 0
        with target.open("xb") as stream:
            while True:
                chunk = response.read(min(1024 * 1024, spec["max_bytes"] - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                require(count <= spec["max_bytes"], "bootstrap_download_size_limit")
                value.update(chunk)
                stream.write(chunk)
    require(count > 0 and ("bytes" not in spec or count == spec["bytes"])
            and value.hexdigest() == spec["sha256"], "bootstrap_download_hash_or_size_mismatch")
    return target


def extract_python(archive_path, destination, config):
    """Allow regular interpreter files and internal links, never outside python/."""
    require(hasattr(tarfile, "data_filter"), "safe_tar_data_filter_required")
    destination = _checked(destination)
    require(not destination.exists(), "fresh_python_extraction_required")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        require(0 < len(members) <= config["maximum_tar_members"]
                and all(type(item.size) is int and item.size >= 0 for item in members)
                and sum(item.size for item in members) <= config["maximum_tar_expanded_bytes"],
                "bounded_python_tar_inventory_required")
        seen = set()
        for member in members:
            name = member.name.rstrip("/")
            path = PurePosixPath(name)
            require(name and "\\" not in name and ":" not in name and "\0" not in name
                    and not path.is_absolute() and path.parts[0] == "python"
                    and all(part not in {"", ".", ".."} for part in name.split("/"))
                    and name == path.as_posix() and name not in seen,
                    "unsafe_or_duplicate_python_tar_path")
            seen.add(name)
            require(member.isfile() or member.isdir() or member.issym() or member.islnk(),
                    "python_tar_special_file_forbidden")
            if member.issym() or member.islnk():
                link = PurePosixPath(member.linkname)
                require(not link.is_absolute() and "\\" not in member.linkname and ":" not in member.linkname,
                        "unsafe_python_tar_link")
                base = destination.joinpath(*path.parent.parts) if member.issym() else destination
                require(_inside((base / member.linkname).resolve(), destination / "python"),
                        "escaping_python_tar_link")
            tarfile.data_filter(member, str(destination))
        destination.mkdir()
        archive.extractall(destination, filter="data")
    executable = destination / "python/bin/python3.12"
    require(executable.is_file() and _inside(executable.resolve(), destination / "python"),
            "isolated_python_executable_missing")
    return executable


def isolated_environment():
    result = dict(os.environ)
    for key in tuple(result):
        if key in {"HOME", "PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV", "CONDA_PREFIX"} \
                or key.startswith("PIP_") or key.upper().endswith("_PROXY"):
            result.pop(key, None)
    result.update(PYTHONNOUSERSITE="1", PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return result


def _run(command, env, cwd, label):
    completed = subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode:
        # Never echo an inherited index/proxy credential or request URL.
        detail = re.sub(r"https?://\S+", "<url>", (completed.stderr or completed.stdout)[-4000:])
        detail = re.sub(r"(?i)(authorization|token|password|secret)\s*[:=]\s*\S+", r"\1=<redacted>", detail)
        print(label + ": " + detail, file=sys.stderr)
        raise RuntimeError(label + "_failed_exit_" + str(completed.returncode))
    return completed.stdout.strip()


def _configuration(repository):
    config_path = _checked(repository / "configs/model_inputs_colab_runtime_v1.json")
    requirements = _checked(repository / "requirements-colab-model-inputs.txt")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config.get("version") == VERSION and config.get("platform") == "linux_x86_64"
            and config.get("python") == "3.12.14" and config.get("packages") == EXPECTED_PACKAGES
            and config.get("pypi_index") == "https://pypi.org/simple", "fixed_cpu_runtime_configuration_required")
    archive = config["python_archive"]
    require(archive.get("url") == PYTHON_URL and archive.get("sha256") == PYTHON_SHA
            and archive.get("bytes") == archive.get("max_bytes") == 34277962,
            "fixed_python_runtime_pin_required")
    require(len(config["cpu_wheels"]) == 2
            and {(item["url"], item["sha256"]) for item in config["cpu_wheels"]} == set(WHEEL_PINS.values()),
            "fixed_cpu_wheel_pins_required")
    for spec in [archive, *config["cpu_wheels"]]:
        require(Path(spec["file"]).name == spec["file"] and not any(c in spec["file"] for c in ("\\", "/", ":"))
                and type(spec["max_bytes"]) is int and 0 < spec["max_bytes"] <= 300000000,
                "bounded_fixed_download_descriptor_required")
    require(config.get("maximum_tar_members") == 10000 and config.get("maximum_tar_expanded_bytes") == 500000000
            and all(config.get(flag) is False for flag in ("data_opened", "training_executed", "pretrained_weights_downloaded")),
            "fixed_bootstrap_scope_and_tar_bounds_required")
    lines = [line.strip() for line in requirements.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    expected = {name + "==" + version for name, version in EXPECTED_PACKAGES.items() if name not in WHEEL_PINS}
    require(set(lines) == expected and len(lines) == len(expected), "exact_minimal_pypi_requirements_required")
    return config, config_path, requirements


def setup(destination, repository_root):
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "linux_x86_64_bootstrap_required")
    require(os.environ.get("COLAB_GPU", "0") != "1" and not any(os.environ.get(k)
            for k in ("COLAB_TPU_ADDR", "TPU_NAME", "COLAB_TPU_TASK_ID")), "cpu_colab_runtime_required")
    repository, destination = _checked(repository_root), _checked(destination)
    require(repository.is_dir() and destination.parent.is_dir() and not destination.exists(), "fresh_bootstrap_destination_required")
    require(not _inside(destination, repository) and not _inside(repository, destination)
            and not any(part.lower() in {"drive", "gdrive", "mydrive", "shared drives"} for part in destination.parts)
            and any(_inside(destination, root) and destination != root
                    for root in (_checked(tempfile.gettempdir()), Path("/content"))), "local_temporary_runtime_outside_drive_required")
    config, config_path, requirements = _configuration(repository)
    before = {path: sha_file(path) for path in (config_path, requirements, Path(__file__))}
    destination.mkdir()
    downloads = destination / "downloads"
    downloads.mkdir()
    print("Preparando Python 3.12.14 aislado para CPU…", file=sys.stderr)
    archive = download(config["python_archive"], downloads / config["python_archive"]["file"])
    python = extract_python(archive, destination / "runtime", config)
    env = isolated_environment()
    version = _run([str(python), "-I", "-c", "import platform,ssl,ensurepip;print(platform.python_version())"], env, destination, "isolated_python_version")
    require(version == config["python"], "actual_python_3_12_14_required")
    _run([str(python), "-I", "-m", "ensurepip", "--upgrade"], env, destination, "standalone_ensurepip")
    print("Instalando únicamente las dependencias congeladas y las ruedas CPU verificadas…", file=sys.stderr)
    _run([str(python), "-I", "-m", "pip", "install", "--quiet", "--no-deps", "--no-cache-dir",
          "--only-binary=:all:", "--index-url", config["pypi_index"], "-r", str(requirements)],
         env, destination, "minimal_cpu_dependencies")
    wheels = [download(spec, downloads / spec["file"]) for spec in config["cpu_wheels"]]
    _run([str(python), "-I", "-m", "pip", "install", "--quiet", "--no-deps", "--no-cache-dir",
          *map(str, wheels)], env, destination, "pinned_local_cpu_wheels")
    _run([str(python), "-I", "-m", "pip", "check"], env, destination, "cpu_runtime_pip_check")
    code = ("import json,platform,importlib.metadata as m,numpy,torch,torchvision,PIL;"
            "print(json.dumps({'python':platform.python_version(),'packages':{p:m.version(p) for p in "
            + repr(list(EXPECTED_PACKAGES)) + "},'cuda_available':torch.cuda.is_available(),'cuda_build':torch.version.cuda}))")
    actual = json.loads(_run([str(python), "-I", "-c", code], env, destination, "cpu_runtime_import_versions"))
    require(actual["python"] == config["python"] and actual["packages"] == EXPECTED_PACKAGES
            and actual["cuda_available"] is False and actual["cuda_build"] is None,
            "actual_frozen_cpu_runtime_versions_required")
    require(all(sha_file(path) == expected for path, expected in before.items()), "bootstrap_sources_changed")
    return {"version": VERSION, "status": "isolated_cpu_runtime_ready", "python_path": str(python),
        "python_version": actual["python"], "packages": actual["packages"], "cpu_only": True,
        "host_python_modified": False, "repository_modified": False, "data_opened": False,
        "pretrained_weights_downloaded": False, "training_executed": False,
        "runtime_archive_sha256": PYTHON_SHA, "runtime_config_sha256": before[config_path],
        "requirements_sha256": before[requirements]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    args = parser.parse_args()
    result = setup(args.destination, args.repository_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
