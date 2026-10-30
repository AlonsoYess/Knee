"""Build deterministic, separate development and reserved-test data packages."""
from __future__ import annotations

import hashlib
from pathlib import Path
import stat
import tempfile
import zipfile

from knee.colab_data import (VERSION, canonical_json, checked_path, require, sha_file,
    validate_bindings, validate_rows)


def _info(name):
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_STORED
    entry.create_system = 3
    entry.external_attr = (stat.S_IFREG | 0o600) << 16
    return entry


def build_packages(output_root, rows, tensor_sources, bindings):
    """Package byte-identical tensors without fitting or opening reserved pixels.

    Sources must cover exactly the declared relative paths; identities and
    assignments are checked jointly before writing either archive. Existing
    packages are never replaced, and repetition must reproduce identical bytes.
    """
    validate_bindings(bindings)
    validate_rows(rows)
    require(isinstance(tensor_sources, dict) and set(tensor_sources) ==
        {r["tensor"]["relative_path"] for r in rows}, "exact_tensor_source_set_required")
    expected = {r["tensor"]["relative_path"]: r["tensor"]["sha256"] for r in rows}
    checked = {}
    for relative, source in sorted(tensor_sources.items()):
        require(Path(source).is_absolute(), "absolute_tensor_source_required")
        source = checked_path(source)
        require(source.is_file() and source.stat().st_size <= 262144
            and sha_file(source) == expected[relative], "pinned_tensor_source_required")
        checked[relative] = source
    root = checked_path(output_root)
    require(root.is_dir(), "existing_package_output_root_required")
    require(all(not p.is_relative_to(root) for p in checked.values()), "package_output_must_not_contain_sources")
    subsets = {kind: sorted([r for r in rows if (r["outer_block"] == 0) == (kind == "reserved_test")],
        key=lambda r: (r["src_subject_id"], r["side"])) for kind in ("development", "reserved_test")}
    require(all(subsets.values()), "development_and_reserved_packages_required")
    packages = {}
    for kind, subset in subsets.items():
        target = checked_path(root / (kind + "_package.zip"))
        manifest = {"version": VERSION, "package_kind": kind, "bindings": bindings, "rows": subset}
        with tempfile.NamedTemporaryFile(prefix=".package-", suffix=".partial", dir=root, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
                archive.writestr(_info("manifest.json"), canonical_json(manifest))
                for row in subset:
                    relative = row["tensor"]["relative_path"]
                    source = checked[relative]
                    raw = source.read_bytes()
                    require(hashlib.sha256(raw).hexdigest() == expected[relative], "tensor_source_changed_during_packaging")
                    archive.writestr(_info(relative), raw)
            sha = sha_file(temporary)
            if target.exists():
                require(target.is_file() and sha_file(target) == sha, "existing_package_changed_no_overwrite")
            else:
                # Exclusive creation works without a platform-specific replace.
                with target.open("xb") as destination, temporary.open("rb") as incoming:
                    for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                        destination.write(chunk)
            require(sha_file(target) == sha, "written_package_hash_mismatch")
            packages[kind] = {"path": str(target), "sha256": sha, "bytes": target.stat().st_size,
                "participants": len({r["src_subject_id"] for r in subset}), "knees": len(subset)}
        finally:
            temporary.unlink(missing_ok=True)
    require(all(sha_file(checked[p]) == h for p, h in expected.items()), "tensor_sources_changed_during_packaging")
    return {"version": VERSION, "packages": packages, "source_tensor_files_unchanged": True,
        "reserved_pixels_decoded": False, "training_executed": False, "imputation_applied": False}
