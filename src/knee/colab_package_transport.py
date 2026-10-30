"""Bounded byte transport of the sealed development ZIP, without test access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile

VERSION = "colab_development_transport_v1.0"
MAX_PART_BYTES = 96000000
MAX_PARTS = 16
SUPPORT = {"handoff.json": 1024**2, "colab_preparation_code.zip": 16 * 1024**2}
MODULE_NAME = "colab_package_transport.py"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def _sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _path(value):
    path = Path(value).absolute()
    for item in (path, *path.parents):
        require(not item.is_symlink() and not getattr(item, "is_junction", lambda: False)(), "transport_links_forbidden")
    return path


def _name(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.-]+", value) is not None
        and value not in {".", ".."}, "transport_safe_basename_required")
    return value


def sha_file(path):
    with _path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "transport_duplicate_json_key")
        result[key] = value
    return result


def _json(path, limit=1024**2):
    path = _path(path)
    require(path.is_file() and 0 < path.stat().st_size <= limit, "bounded_transport_json_required")
    return json.loads(path.read_bytes(), object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("transport_nonfinite_json")))


def _canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _spec(spec, maximum, filename=None):
    fields = {"sha256", "bytes"} | ({"file"} if filename is not None else set())
    require(isinstance(spec, dict) and set(spec) == fields and _sha(spec["sha256"])
        and type(spec["bytes"]) is int and 0 < spec["bytes"] <= maximum,
        "bounded_transport_file_descriptor_required")
    if filename is not None:
        require(_name(spec["file"]) == filename, "transport_exact_filename_required")


def _file(root, name, spec):
    path = _path(root / _name(name))
    require(path.is_file() and path.stat().st_size == spec["bytes"], "transport_source_missing_or_size_mismatch")
    return path


def _chunks(stream, limit):
    remaining = limit
    while remaining:
        chunk = stream.read(min(1024**2, remaining))
        require(bool(chunk), "transport_source_truncated")
        remaining -= len(chunk)
        yield chunk


def _write(target, chunks, expected_bytes, expected_sha=None):
    target = _path(target)
    require(target.parent.is_dir(), "existing_transport_output_parent_required")
    with tempfile.NamedTemporaryFile(prefix=".transport-", suffix=".partial", dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        count, digest = 0, hashlib.sha256()
        try:
            for chunk in chunks:
                count += len(chunk)
                require(count <= expected_bytes, "transport_source_longer_than_declared")
                digest.update(chunk)
                stream.write(chunk)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        actual = digest.hexdigest()
        require(count == expected_bytes and (expected_sha is None or actual == expected_sha), "transport_stream_hash_or_size_mismatch")
        if target.exists():
            require(target.is_file() and target.stat().st_size == count and sha_file(target) == actual,
                "existing_transport_file_changed_no_overwrite")
        else:
            # The target is created exclusively; an existing user file is never replaced.
            with target.open("xb") as destination, temporary.open("rb") as incoming:
                for chunk in iter(lambda: incoming.read(1024**2), b""):
                    destination.write(chunk)
        require(target.stat().st_size == count and sha_file(target) == actual, "transport_written_file_changed")
        return {"sha256": actual, "bytes": count}
    finally:
        temporary.unlink(missing_ok=True)


def _copy(source, destination, spec):
    with _path(source).open("rb") as incoming:
        result = _write(destination, _chunks(incoming, spec["bytes"]), spec["bytes"], spec["sha256"])
        require(not incoming.read(1), "transport_source_longer_than_declared")
    return result


def _handoff(value):
    require(isinstance(value, dict) and value.get("version") == "participant_split_colab_preparation_v1.0"
        and value.get("training_executed") is False
        and value.get("reserved_test_predictive_evaluation_opened") is False,
        "sealed_preparation_handoff_required")
    development = value.get("development", {})
    require(development.get("file") == "development_package.zip" and _sha(development.get("sha256"))
        and type(development.get("bytes")) is int and 0 < development["bytes"] <= MAX_PART_BYTES * MAX_PARTS,
        "bounded_development_handoff_required")
    require(value.get("code", {}).get("file") == "colab_preparation_code.zip"
        and _sha(value["code"].get("sha256")), "sealed_code_handoff_required")


def split_development_package(package_root, output_root, module_source=None, chunk_bytes=MAX_PART_BYTES):
    """Write deterministic immutable parts and support copies from sealed packages."""
    source, output = _path(package_root), _path(output_root)
    require(source.is_dir() and output.is_dir() and not source.is_relative_to(output)
        and not output.is_relative_to(source), "separate_existing_transport_roots_required")
    require(type(chunk_bytes) is int and 0 < chunk_bytes <= MAX_PART_BYTES, "bounded_transport_chunk_size_required")
    handoff = _json(source / "handoff.json")
    _handoff(handoff)
    closure = _json(source / "closure_summary.json")
    handoff_sha = sha_file(source / "handoff.json")
    require(closure.get("status") == "participant_partitions_and_private_colab_packages_verified"
        and closure.get("handoff_sha256") == handoff_sha and closure.get("packages") == handoff
        and closure.get("training_executed") is False
        and closure.get("reserved_test_predictive_evaluation_opened") is False,
        "sealed_package_closure_and_handoff_required")
    dev = {k: handoff["development"][k] for k in ("file", "sha256", "bytes")}
    require((dev["bytes"] + chunk_bytes - 1) // chunk_bytes <= MAX_PARTS, "bounded_transport_part_count_required")
    package = _file(source, dev["file"], dev)
    require(sha_file(package) == dev["sha256"], "sealed_development_package_hash_mismatch")
    support = {}
    for name, maximum in SUPPORT.items():
        path = _path(source / name)
        require(path.is_file() and 0 < path.stat().st_size <= maximum, "bounded_transport_support_required")
        support[name] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    require(support["handoff.json"]["sha256"] == handoff_sha
        and support["colab_preparation_code.zip"]["sha256"] == handoff["code"]["sha256"], "sealed_support_hash_mismatch")
    module_path = _path(module_source if module_source is not None else __file__)
    require(module_path.is_file() and 0 < module_path.stat().st_size <= 1024**2, "bounded_transport_module_required")
    module = {"file": MODULE_NAME, "sha256": sha_file(module_path), "bytes": module_path.stat().st_size}
    parts, full_hash = [], hashlib.sha256()
    remaining = dev["bytes"]
    with package.open("rb") as stream:
        while remaining:
            name = f"development_package.zip.part{len(parts) + 1:03d}"
            size = min(chunk_bytes, remaining)
            def pieces():
                for chunk in _chunks(stream, size):
                    full_hash.update(chunk)
                    yield chunk
            part = _write(output / name, pieces(), size)
            parts.append({"file": name, **part})
            remaining -= size
        require(not stream.read(1) and full_hash.hexdigest() == dev["sha256"], "development_package_changed_during_transport")
    for name, spec in support.items():
        _copy(source / name, output / name, spec)
    _copy(module_path, output / MODULE_NAME, module)
    manifest = {"version": VERSION, "handoff_sha256": handoff_sha, "development": dev,
        "parts": parts, "support_files": support, "transport_module": module}
    raw = _canonical(manifest)
    _write(output / "transport.json", [raw], len(raw), hashlib.sha256(raw).hexdigest())
    require(sha_file(package) == dev["sha256"] and sha_file(source / "handoff.json") == handoff_sha,
        "original_package_or_handoff_changed")
    return manifest


def stage_development_package(remote_root, local_root):
    """Stream verified ordered parts into a local development ZIP; no test reads."""
    remote, local = _path(remote_root), _path(local_root)
    require(remote.is_dir() and local.is_dir() and not remote.is_relative_to(local)
        and not local.is_relative_to(remote), "separate_existing_transport_roots_required")
    manifest_path = remote / "transport.json"
    manifest_sha = sha_file(manifest_path)
    manifest = _json(manifest_path)
    require(isinstance(manifest, dict) and set(manifest) == {"version", "handoff_sha256", "development", "parts", "support_files", "transport_module"}
        and manifest["version"] == VERSION and _sha(manifest["handoff_sha256"]), "exact_transport_manifest_required")
    _spec(manifest["development"], MAX_PART_BYTES * MAX_PARTS, "development_package.zip")
    require(isinstance(manifest["support_files"], dict) and set(manifest["support_files"]) == set(SUPPORT), "exact_transport_support_required")
    for name, maximum in SUPPORT.items():
        _spec(manifest["support_files"][name], maximum)
    _spec(manifest["transport_module"], 1024**2, MODULE_NAME)
    parts = manifest["parts"]
    require(isinstance(parts, list) and 0 < len(parts) <= MAX_PARTS, "bounded_transport_parts_required")
    for i, part in enumerate(parts, 1):
        _spec(part, MAX_PART_BYTES, f"development_package.zip.part{i:03d}")
    require(sum(p["bytes"] for p in parts) == manifest["development"]["bytes"]
        and all(p["bytes"] == parts[0]["bytes"] for p in parts[:-1])
        and parts[-1]["bytes"] <= parts[0]["bytes"], "consistent_ordered_transport_part_sizes_required")
    found = {p.name for p in remote.glob("development_package.zip.part*")}
    require(found == {p["file"] for p in parts}, "exact_transport_part_file_set_required")
    for part in parts:
        _file(remote, part["file"], part)
    for name, spec in manifest["support_files"].items():
        path = _file(remote, name, spec)
        require(sha_file(path) == spec["sha256"], "transport_support_hash_mismatch")
    module = manifest["transport_module"]
    module_path = _file(remote, module["file"], module)
    require(sha_file(module_path) == module["sha256"], "transport_module_hash_mismatch")
    handoff = _json(remote / "handoff.json")
    _handoff(handoff)
    require(manifest["handoff_sha256"] == manifest["support_files"]["handoff.json"]["sha256"]
        and manifest["development"] == {k: handoff["development"][k] for k in ("file", "sha256", "bytes")}
        and manifest["support_files"]["colab_preparation_code.zip"]["sha256"] == handoff["code"]["sha256"],
        "transport_handoff_binding_mismatch")
    def pieces():
        for part in parts:
            digest = hashlib.sha256()
            with _file(remote, part["file"], part).open("rb") as incoming:
                for chunk in _chunks(incoming, part["bytes"]):
                    digest.update(chunk)
                    yield chunk
                require(not incoming.read(1), "transport_source_longer_than_declared")
            require(digest.hexdigest() == part["sha256"], "transport_part_hash_mismatch")
    dev = manifest["development"]
    _write(local / dev["file"], pieces(), dev["bytes"], dev["sha256"])
    for name, spec in manifest["support_files"].items():
        _copy(remote / name, local / name, spec)
    require(sha_file(manifest_path) == manifest_sha, "transport_manifest_changed_during_staging")
    return {"package_root": str(local), "sha256": dev["sha256"], "bytes": dev["bytes"], "parts": len(parts),
        "reserved_test_opened": False, "training_executed": False}
