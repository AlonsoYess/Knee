"""Portable packages keep labels/grouping sealed and refuse reserved test loads."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from knee import colab_data as data
from knee import colab_data_package as package


class ColabPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "packages"
        self.sources = self.root / "sources"
        self.output.mkdir()
        self.sources.mkdir()
        self.rows, self.paths = [], {}
        for index, (subject, side, role, block, fold) in enumerate([
            ("syntheticA", "LEFT", "development", 1, 0),
            ("syntheticA", "RIGHT", "development", 1, 0),
            ("syntheticB", "LEFT", "routine", 2, 1),
            ("syntheticT", "RIGHT", "routine", 0, None)]):
            array = np.full((1, 224, 224), index / 4, dtype=np.float32)
            relative = f"tensors/{index:03d}.npy"
            path = self.sources / f"{index:03d}.npy"
            np.save(path, array, allow_pickle=False)
            self.paths[relative] = path
            self.rows.append({"src_subject_id": subject, "side": side, "preprocessing_role": role,
                "outer_block": block, "cv_fold": fold, "AGEYEARS": 60, "SEX": "F",
                "BMI": None if subject == "syntheticA" else 25.0, "PROGRESION_48M": index % 2,
                "tensor": {"relative_path": relative, "sha256": data.sha_file(path),
                    "pixel_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    "shape": [1, 224, 224], "dtype": "float32"}})
        self.bindings = {key: "a" * 64 for key in data.REQUIRED_BINDINGS}

    def build(self, rows=None, paths=None, bindings=None):
        return package.build_packages(self.output, self.rows if rows is None else rows,
            self.paths if paths is None else paths, self.bindings if bindings is None else bindings)

    def development(self):
        entry = self.build()["packages"]["development"]
        return Path(entry["path"]), entry["sha256"]

    def rewrite(self, mutate_manifest=None, mutate_files=None, *, duplicate=None):
        path, _ = self.development()
        with zipfile.ZipFile(path) as archive:
            files = {i.filename: archive.read(i) for i in archive.infolist()}
        manifest = json.loads(files["manifest.json"])
        if mutate_manifest:
            mutate_manifest(manifest)
        files["manifest.json"] = data.canonical_json(manifest)
        if mutate_files:
            mutate_files(files, manifest)
        with zipfile.ZipFile(path, "w") as archive:
            for name, raw in files.items():
                archive.writestr(package._info(name), raw)
            if duplicate:
                archive.writestr(package._info(duplicate), b"duplicate")
        return path, data.sha_file(path)

    def test_deterministic_packages_byte_identical_sources_and_input_unchanged(self):
        before = deepcopy(self.rows)
        first = self.build()
        second = self.build(rows=list(reversed(self.rows)))
        self.assertEqual(first, second)
        self.assertEqual(before, self.rows)
        self.assertEqual(first["packages"]["development"]["knees"], 3)
        self.assertEqual(first["packages"]["reserved_test"]["knees"], 1)
        for entry in first["packages"].values():
            with zipfile.ZipFile(entry["path"]) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                for row in manifest["rows"]:
                    relative = row["tensor"]["relative_path"]
                    self.assertEqual(archive.read(relative), self.paths[relative].read_bytes())

    def test_verify_streams_aggregates_and_optional_record_idempotently(self):
        path, sha = self.development()
        record = self.root / "verification.json"
        result = data.verify_development_package(path, sha, record)
        self.assertEqual(data.verify_development_package(path, sha, record), result)
        self.assertEqual(result["participants"], 2)
        self.assertEqual(result["bmi_missing_knees"], 2)
        self.assertNotIn("synthetic", str(result))
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["files_extracted"])

    def test_explicit_load_is_readonly_and_has_no_imputation(self):
        path, sha = self.development()
        loaded = data.load_development_package(path, sha)
        self.assertEqual(len(loaded["tensors"]), 3)
        self.assertIsNone(loaded["manifest"]["rows"][0]["BMI"])
        for array in loaded["tensors"].values():
            self.assertFalse(array.flags.writeable)

    def test_reserved_package_refused_before_any_tensor_load(self):
        entry = self.build()["packages"]["reserved_test"]
        with patch.object(np, "load", side_effect=AssertionError("reserved pixels forbidden")):
            for function in (data.verify_development_package, data.load_development_package):
                with self.subTest(function=function.__name__), self.assertRaisesRegex(ValueError, "reserved_test_package_open_forbidden"):
                    function(entry["path"], entry["sha256"])

    def test_builder_does_not_decode_any_array_including_reserved(self):
        with patch.object(np, "load", side_effect=AssertionError("packaging never decodes")):
            self.build()

    def test_unknown_row_and_tensor_columns_and_bindings_rejected(self):
        for key in ("KL_BASE", "KL_48", "FECHA_V06", "private_path"):
            rows = deepcopy(self.rows)
            rows[0][key] = "forbidden"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "row_allowlist"):
                self.build(rows=rows)
        rows = deepcopy(self.rows)
        rows[0]["tensor"]["native_path"] = "private"
        with self.assertRaisesRegex(ValueError, "tensor_descriptor"):
            self.build(rows=rows)
        with self.assertRaisesRegex(ValueError, "hash_only"):
            self.build(bindings=dict(self.bindings, private_path="D:/private"))

    def test_source_set_missing_or_extra_and_hash_change_rejected(self):
        for paths in ({}, dict(self.paths, extra=self.sources / "other.npy")):
            with self.assertRaisesRegex(ValueError, "exact_tensor_source_set"):
                self.build(paths=paths)
        self.paths["tensors/000.npy"].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "pinned_tensor_source"):
            self.build()

    def test_duplicates_and_subject_cross_block_or_fold_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate_package_knee"):
            self.build(rows=self.rows + [self.rows[0]])
        for key, value in (("cv_fold", 4), ("outer_block", 2), ("preprocessing_role", "routine")):
            rows = deepcopy(self.rows)
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "participant_crosses"):
                self.build(rows=rows)

    def test_reserved_preprocessing_development_and_reserved_cv_rejected(self):
        for key, value in (("preprocessing_role", "development"), ("cv_fold", 0)):
            rows = deepcopy(self.rows)
            rows[-1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "reserved_role"):
                self.build(rows=rows)

    def test_bilateral_baseline_consistency_required(self):
        for key, value in (("BMI", 30.0), ("AGEYEARS", 70), ("SEX", "M")):
            rows = deepcopy(self.rows)
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "baseline_values_inconsistent"):
                self.build(rows=rows)

    def test_path_traversal_absolute_windows_and_manifest_collision_rejected(self):
        for relative in ("../outside.npy", "/absolute.npy", "C:/private.npy", "tensors\\x.npy", "tensors//x.npy", "./x.npy"):
            rows = deepcopy(self.rows)
            rows[0]["tensor"]["relative_path"] = relative
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "safe_package_relative"):
                self.build(rows=rows)

    def test_source_symlink_and_package_symlink_forbidden(self):
        original = Path.is_symlink
        source = self.paths["tensors/000.npy"]
        with patch.object(Path, "is_symlink", lambda p: p == source or original(p)):
            with self.assertRaisesRegex(ValueError, "package_links_forbidden"):
                self.build()
        path, sha = self.development()
        with patch.object(Path, "is_symlink", lambda p: p == path or original(p)):
            with self.assertRaisesRegex(ValueError, "package_links_forbidden"):
                data.verify_development_package(path, sha)

    def test_reserved_rows_cannot_be_disguised_as_development(self):
        path, sha = self.rewrite(lambda m: m["rows"][0].update(outer_block=0, cv_fold=None, preprocessing_role="routine"))
        with self.assertRaisesRegex(ValueError, "package_role_membership_mismatch"):
            data.verify_development_package(path, sha)

    def test_loader_fileset_extra_and_missing_rejected(self):
        for extra in (True, False):
            path, sha = self.rewrite(mutate_files=(lambda f, m: f.update({"extra.txt": b"unexpected"})) if extra
                else (lambda f, m: f.pop(m["rows"][0]["tensor"]["relative_path"])))
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "exact_package_file_set"):
                data.verify_development_package(path, sha)
            path.unlink()

    def test_loader_duplicate_and_zip_symlink_rejected(self):
        path, sha = self.rewrite(duplicate="tensors/000.npy")
        with self.assertRaisesRegex(ValueError, "duplicate_zip_member"):
            data.verify_development_package(path, sha)
        path.unlink()
        path, _ = self.development()
        with zipfile.ZipFile(path, "a") as archive:
            info = package._info("link.npy")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../outside.npy")
        with self.assertRaisesRegex(ValueError, "regular_unencrypted"):
            data.verify_development_package(path, data.sha_file(path))

    def test_wrong_package_file_and_pixel_hashes_rejected(self):
        path, sha = self.development()
        with self.assertRaisesRegex(ValueError, "package_sha256_mismatch"):
            data.verify_development_package(path, "a" * 64)
        for field, code in (("sha256", "file_hash"), ("pixel_sha256", "pixel_hash")):
            path, sha = self.rewrite(lambda m: m["rows"][0]["tensor"].update({field: "b" * 64}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, code):
                data.verify_development_package(path, sha)
            path.unlink()

    def test_invalid_arrays_object_nonfinite_range_shape_dtype_and_trailing_payload(self):
        for array in (np.full((1, 224, 224), np.nan, np.float32), np.full((1, 224, 224), -0.1, np.float32),
                      np.full((1, 224, 224), 1.1, np.float32), np.zeros((224, 224), np.float32),
                      np.zeros((1, 224, 224), np.float64), np.array([object()], dtype=object), None):
            def mutate(files, manifest):
                row = manifest["rows"][0]
                spec = row["tensor"]
                if array is None:
                    raw = files[spec["relative_path"]] + b"trailing"
                else:
                    stream = io.BytesIO()
                    np.save(stream, array, allow_pickle=True)
                    raw = stream.getvalue()
                    spec["pixel_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
                spec["sha256"] = hashlib.sha256(raw).hexdigest()
                files[spec["relative_path"]] = raw
                files["manifest.json"] = data.canonical_json(manifest)
            path, sha = self.rewrite(mutate_files=mutate)
            with self.subTest(kind=str(None if array is None else (array.shape, array.dtype))), self.assertRaises(ValueError):
                data.verify_development_package(path, sha)
            path.unlink()

    def test_changed_existing_package_and_record_never_overwritten(self):
        path, sha = self.development()
        path.write_bytes(b"user changes")
        with self.assertRaisesRegex(ValueError, "existing_package_changed_no_overwrite"):
            self.build()
        self.assertEqual(path.read_bytes(), b"user changes")
        path.unlink()
        path, sha = self.development()
        record = self.root / "record.json"
        record.write_bytes(b"user record")
        with self.assertRaisesRegex(ValueError, "record_changed_no_overwrite"):
            data.verify_development_package(path, sha, record)
        self.assertEqual(record.read_bytes(), b"user record")


if __name__ == "__main__":
    unittest.main()
