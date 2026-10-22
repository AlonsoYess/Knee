"""Transport pieces preserve the sealed development bytes and never read test."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee import colab_package_transport as t


class ColabTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.remote, self.local = (self.root / n for n in ("source", "remote", "local"))
        for folder in (self.source, self.remote, self.local):
            folder.mkdir()
        self.raw = bytes(range(256)) * 40
        (self.source / "development_package.zip").write_bytes(self.raw)
        (self.source / "reserved_test_package.zip").write_bytes(b"MUST NOT OPEN")
        code = b"synthetic code bundle bytes"
        (self.source / "colab_preparation_code.zip").write_bytes(code)
        self.handoff = {"version": "participant_split_colab_preparation_v1.0",
            "development": {"file": "development_package.zip", "sha256": hashlib.sha256(self.raw).hexdigest(), "bytes": len(self.raw)},
            "reserved_test": {"file": "reserved_test_package.zip", "sha256": "f" * 64},
            "code": {"file": "colab_preparation_code.zip", "sha256": hashlib.sha256(code).hexdigest()},
            "training_executed": False, "reserved_test_predictive_evaluation_opened": False}
        self.write_json(self.source / "handoff.json", self.handoff)
        self.closure = {"status": "participant_partitions_and_private_colab_packages_verified",
            "handoff_sha256": t.sha_file(self.source / "handoff.json"), "packages": deepcopy(self.handoff),
            "training_executed": False, "reserved_test_predictive_evaluation_opened": False}
        self.write_json(self.source / "closure_summary.json", self.closure)
        self.module = self.root / "module.py"
        self.module.write_bytes(b"# synthetic transport module\n")

    @staticmethod
    def write_json(path, value):
        path.write_bytes(t._canonical(value))

    def split(self, **kwargs):
        return t.split_development_package(self.source, self.remote, module_source=self.module,
            chunk_bytes=kwargs.get("chunk_bytes", 3000))

    def mutate(self, function):
        path = self.remote / "transport.json"
        value = json.loads(path.read_bytes())
        function(value)
        self.write_json(path, value)

    def test_four_deterministic_parts_roundtrip_idempotent_originals_unchanged(self):
        before = {p.name: t.sha_file(p) for p in self.source.iterdir()}
        first = self.split()
        self.assertEqual(len(first["parts"]), 4)
        self.assertEqual(first, self.split())
        result = t.stage_development_package(self.remote, self.local)
        self.assertEqual(result, t.stage_development_package(self.remote, self.local))
        self.assertEqual((self.local / "development_package.zip").read_bytes(), self.raw)
        self.assertEqual({p.name: t.sha_file(p) for p in self.source.iterdir()}, before)
        self.assertEqual({p.name for p in self.local.iterdir()}, {"handoff.json", "colab_preparation_code.zip", "development_package.zip"})
        self.assertFalse(result["reserved_test_opened"])

    def test_reserved_payload_not_opened_in_split_or_stage(self):
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == "reserved_test_package.zip":
                raise AssertionError("reserved package opened")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            self.split()
            t.stage_development_package(self.remote, self.local)

    def test_invalid_closure_hash_handoff_and_training_fail(self):
        for field, value in (("handoff_sha256", "a" * 64), ("training_executed", True),
                             ("status", "other"), ("reserved_test_predictive_evaluation_opened", True)):
            changed = dict(self.closure, **{field: value})
            self.write_json(self.source / "closure_summary.json", changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "sealed_package_closure"):
                self.split()
        self.write_json(self.source / "closure_summary.json", self.closure)
        changed = deepcopy(self.handoff)
        changed["training_executed"] = True
        self.write_json(self.source / "handoff.json", changed)
        with self.assertRaisesRegex(ValueError, "sealed_preparation_handoff"):
            self.split()

    def test_bad_development_and_code_hash_refused_before_transport_manifest(self):
        (self.source / "development_package.zip").write_bytes(b"x" * len(self.raw))
        with self.assertRaisesRegex(ValueError, "sealed_development_package_hash"):
            self.split()
        self.assertFalse((self.remote / "transport.json").exists())
        (self.source / "development_package.zip").write_bytes(self.raw)
        (self.source / "colab_preparation_code.zip").write_bytes(b"different code")
        with self.assertRaisesRegex(ValueError, "sealed_support_hash"):
            self.split()

    def test_chunk_hard_limits_and_count(self):
        for chunk in (0, -1, True, t.MAX_PART_BYTES + 1, 0.5, 1):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                self.split(chunk_bytes=chunk)

    def test_missing_extra_and_mutated_part_fail_without_local_zip(self):
        manifest = self.split()
        first = self.remote / manifest["parts"][0]["file"]
        raw = first.read_bytes()
        first.unlink()
        with self.assertRaisesRegex(ValueError, "exact_transport_part_file_set"):
            t.stage_development_package(self.remote, self.local)
        first.write_bytes(raw)
        extra = self.remote / "development_package.zip.part005"
        extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "exact_transport_part_file_set"):
            t.stage_development_package(self.remote, self.local)
        extra.unlink()
        first.write_bytes(b"z" * len(raw))
        with self.assertRaisesRegex(ValueError, "transport_part_hash"):
            t.stage_development_package(self.remote, self.local)
        self.assertFalse((self.local / "development_package.zip").exists())
        self.assertFalse(list(self.local.glob(".transport*")))

    def test_swapped_order_and_duplicate_names_fail(self):
        self.split()
        before = json.loads((self.remote / "transport.json").read_bytes())
        for function in (lambda m: m["parts"].reverse(),
                         lambda m: m["parts"][1].update(file=m["parts"][0]["file"])):
            self.write_json(self.remote / "transport.json", before)
            self.mutate(function)
            with self.assertRaisesRegex(ValueError, "transport_exact_filename"):
                t.stage_development_package(self.remote, self.local)

    def test_traversal_and_absolute_descriptors_fail(self):
        self.split()
        before = json.loads((self.remote / "transport.json").read_bytes())
        for name in ("../private", "/absolute", "D:/private", "x\\private", ".."):
            self.write_json(self.remote / "transport.json", before)
            self.mutate(lambda m: m["parts"][0].update(file=name))
            with self.subTest(name=name), self.assertRaises(ValueError):
                t.stage_development_package(self.remote, self.local)

    def test_unknown_manifest_or_descriptor_fields_fail(self):
        self.split()
        before = json.loads((self.remote / "transport.json").read_bytes())
        for function in (lambda m: m.update(unexpected="field"),
                         lambda m: m["parts"][0].update(path="private")):
            self.write_json(self.remote / "transport.json", before)
            self.mutate(function)
            with self.assertRaises(ValueError):
                t.stage_development_package(self.remote, self.local)

    def test_bad_part_size_and_total_count_fail(self):
        self.split()
        before = json.loads((self.remote / "transport.json").read_bytes())
        for value in (0, True, t.MAX_PART_BYTES + 1, 3001):
            self.write_json(self.remote / "transport.json", before)
            self.mutate(lambda m: m["parts"][0].update(bytes=value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                t.stage_development_package(self.remote, self.local)
        self.write_json(self.remote / "transport.json", before)
        self.mutate(lambda m: m.update(parts=[]))
        with self.assertRaises(ValueError):
            t.stage_development_package(self.remote, self.local)

    def test_support_module_and_whole_binding_integrity(self):
        self.split()
        code = self.remote / "colab_preparation_code.zip"
        before = code.read_bytes()
        code.write_bytes(b"x" * len(before))
        with self.assertRaisesRegex(ValueError, "support_hash"):
            t.stage_development_package(self.remote, self.local)
        code.write_bytes(before)
        module = self.remote / t.MODULE_NAME
        before = module.read_bytes()
        module.write_bytes(b"x" * len(before))
        with self.assertRaisesRegex(ValueError, "module_hash"):
            t.stage_development_package(self.remote, self.local)
        module.write_bytes(before)
        self.mutate(lambda m: m["development"].update(sha256="b" * 64))
        with self.assertRaisesRegex(ValueError, "handoff_binding"):
            t.stage_development_package(self.remote, self.local)

    def test_existing_user_files_not_overwritten(self):
        manifest = self.split()
        part = self.remote / manifest["parts"][0]["file"]
        part.write_bytes(b"user part")
        with self.assertRaisesRegex(ValueError, "existing_transport_file_changed_no_overwrite"):
            self.split()
        self.assertEqual(part.read_bytes(), b"user part")
        part.unlink()
        self.split()
        destination = self.local / "development_package.zip"
        destination.write_bytes(b"user package")
        with self.assertRaisesRegex(ValueError, "existing_transport_file_changed_no_overwrite"):
            t.stage_development_package(self.remote, self.local)
        self.assertEqual(destination.read_bytes(), b"user package")

    def test_whole_package_digest_checked_after_all_valid_part_digests(self):
        self.split()
        handoff_path = self.remote / "handoff.json"
        changed = deepcopy(self.handoff)
        changed["development"]["sha256"] = "a" * 64
        self.write_json(handoff_path, changed)
        def mutate(manifest):
            manifest["development"]["sha256"] = "a" * 64
            manifest["handoff_sha256"] = t.sha_file(handoff_path)
            manifest["support_files"]["handoff.json"] = {
                "sha256": t.sha_file(handoff_path), "bytes": handoff_path.stat().st_size}
        self.mutate(mutate)
        with self.assertRaisesRegex(ValueError, "transport_stream_hash_or_size_mismatch"):
            t.stage_development_package(self.remote, self.local)
        self.assertFalse((self.local / "development_package.zip").exists())

    def test_duplicate_json_keys_rejected(self):
        self.split()
        path = self.remote / "transport.json"
        raw = path.read_bytes()
        path.write_bytes(b'{"version":"duplicated",' + raw[1:])
        with self.assertRaisesRegex(ValueError, "transport_duplicate_json_key"):
            t.stage_development_package(self.remote, self.local)

    def test_junction_ancestor_rejected(self):
        original = getattr(Path, "is_junction", lambda p: False)
        with patch.object(Path, "is_junction", lambda p: p == self.source or original(p), create=True):
            with self.assertRaisesRegex(ValueError, "transport_links_forbidden"):
                self.split()

    def test_links_and_overlapping_directories_forbidden(self):
        self.split()
        target = self.remote / "development_package.zip.part001"
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda p: p == target or original(p)):
            with self.assertRaisesRegex(ValueError, "transport_links_forbidden"):
                t.stage_development_package(self.remote, self.local)
        with self.assertRaisesRegex(ValueError, "separate_existing_transport_roots"):
            t.stage_development_package(self.remote, self.remote)


if __name__ == "__main__":
    unittest.main()
