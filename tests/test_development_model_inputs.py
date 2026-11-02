"""Synthetic-only closure and immutable-output tests for development inputs.

The controller context is replaced with temporary fixtures. No patient images,
private membership, source archives, outcomes, checkpoints, or training are read.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from knee import development_model_inputs as subject


class DevelopmentModelInputClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="knee_tensor_close_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "development_inputs_224_001"
        self.output.mkdir()
        self.native = self.root / "synthetic_native.npy"
        np.save(self.native, np.arange(12, dtype=np.uint16).reshape(3, 4), allow_pickle=False)
        self.source = self.root / "synthetic_source.json"
        self.source.write_text('{"synthetic":true}\n', encoding="utf-8")
        self.pins = {str(self.source): subject.sha_file(self.source),
                     str(self.native): subject.sha_file(self.native)}
        self.selected = [{"src_subject_id": "synthetic_development", "side": "RIGHT",
                          "role": "development", "block": "synthetic_block"}]
        self.tensor = np.full((1, 224, 224), 0.25, dtype=np.float32)
        self.tensor_path = self.output / "tensors/dev_001.npy"
        self.tensor_path.parent.mkdir()
        np.save(self.tensor_path, self.tensor, allow_pickle=False)
        self.qa_path = self.output / "qa/sheet_001.png"
        self.qa_path.parent.mkdir()
        Image.new("RGB", (2, 2), "black").save(self.qa_path, format="PNG")
        self.manifest = {
            "version": subject.VERSION,
            "tensor_contract_version": subject.TENSOR_VERSION,
            "tensor_contract_sha256": subject.digest(subject.CONTRACT),
            "rows": [{**self.selected[0], "index": 1,
                "source_native_sha256": self.pins[str(self.native)],
                "tensor": {"relative_path": "tensors/dev_001.npy",
                    "sha256": subject.sha_file(self.tensor_path),
                    "pixel_sha256": hashlib.sha256(self.tensor.tobytes()).hexdigest(),
                    "shape": [1, 224, 224], "dtype": "float32"}}],
            "source_files_sha256": deepcopy(self.pins),
            "qa_files_sha256": {"qa/sheet_001.png": subject.sha_file(self.qa_path)},
            "summary": {
                "participants": 1, "knees": 1, "photometric_counts": {"MONOCHROME2": 1},
                "native_inputs_unchanged": True, "all_repeated_transforms_identical": True,
                "all_finite_float32_1x224x224": True,
                "physical_aspect_preserved_with_pixel_rounding": True,
                "whole_native_crop_retained": True, "orientation_preserved": True,
                "routine_images_processed": 0, "outcome_rows_loaded": False,
                "training_executed": False, "partitions_created": False,
                "native_edits": False, "qc_regraded": False, "drive_upload_performed": False,
            },
        }
        self.manifest_path = self.output / "manifest.json"
        subject.write_once(self.manifest_path, self.manifest)
        self.review_path = self.root / "synthetic_visual_review.json"
        self.review = {
            "version": subject.VERSION,
            "manifest_sha256": subject.sha_file(self.manifest_path),
            "reviewed_qa_files_sha256": deepcopy(self.manifest["qa_files_sha256"]),
            "development_outputs_visually_inspected": 1,
            "technical_visualization_passed": True,
            "review_scope": "transformation_geometry_polarity_and_display_only_not_anatomical_regrading",
            "reviewer": "Codex_assisted_technical_review",
        }
        self.write_review()

    def write_review(self):
        self.review_path.write_text(json.dumps(self.review, allow_nan=False) + "\n", encoding="utf-8")

    def rewrite_manifest(self, *, update_review_hash=True):
        self.manifest_path.write_text(json.dumps(self.manifest, allow_nan=False) + "\n", encoding="utf-8")
        if update_review_hash:
            self.review["manifest_sha256"] = subject.sha_file(self.manifest_path)
            self.write_review()

    def close(self):
        context = ({}, self.selected, {"synthetic": self.root}, self.output, self.pins, {})
        with patch.object(subject, "_context", return_value=context):
            return subject.close(self.root / "unused_synthetic_config.json", self.review_path)

    def assert_rejected_without_closure(self):
        with self.assertRaises((ValueError, TypeError, KeyError)):
            self.close()
        self.assertFalse((self.output / "closure_summary.json").exists())

    def test_positive_closure_uses_actual_canonical_contract_digest(self):
        # digest() is production canonical hashing, not a stubbed fixture value.
        canonical = json.dumps(subject.CONTRACT, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.assertEqual(self.manifest["tensor_contract_sha256"], hashlib.sha256(canonical).hexdigest())
        before = {path: subject.sha_file(path) for path in (
            self.native, self.source, self.tensor_path, self.qa_path, self.manifest_path, self.review_path)}
        result = self.close()
        self.assertFalse(result["existing_closure_reused"])
        self.assertEqual(result["summary"]["knees"], 1)
        self.assertFalse(result["routine_application_executed"])
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["partitions_created"])
        self.assertFalse(result["cohort_membership_changed"])
        self.assertFalse(result["clinical_or_predictive_validation_claimed"])
        self.assertEqual(before, {path: subject.sha_file(path) for path in before})
        stored = json.loads((self.output / "closure_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(stored, {k: v for k, v in result.items() if k != "existing_closure_reused"})

    def test_repeated_closure_reuses_exact_immutable_result(self):
        first = self.close()
        closure = self.output / "closure_summary.json"
        before = closure.read_bytes()
        second = self.close()
        self.assertFalse(first.pop("existing_closure_reused"))
        self.assertTrue(second.pop("existing_closure_reused"))
        self.assertEqual(first, second)
        self.assertEqual(closure.read_bytes(), before)

    def test_changed_existing_closure_cannot_be_silently_overwritten(self):
        self.close()
        closure = self.output / "closure_summary.json"
        modified = json.loads(closure.read_text(encoding="utf-8"))
        modified["summary"]["knees"] = 999
        closure.write_text(json.dumps(modified) + "\n", encoding="utf-8")
        before = closure.read_bytes()
        with self.assertRaisesRegex(ValueError, "immutable_output_changed"):
            self.close()
        self.assertEqual(closure.read_bytes(), before)

    def test_foreign_manifest_version_is_rejected_even_with_current_review_hash(self):
        self.manifest["version"] = "another_controller_v1.0"
        self.rewrite_manifest()
        self.assert_rejected_without_closure()

    def test_foreign_tensor_version_is_rejected(self):
        self.manifest["tensor_contract_version"] = "another_tensor_v1.0"
        self.rewrite_manifest()
        self.assert_rejected_without_closure()

    def test_foreign_tensor_contract_digest_is_rejected(self):
        self.manifest["tensor_contract_sha256"] = "0" * 64
        self.rewrite_manifest()
        self.assert_rejected_without_closure()

    def test_manifest_missing_current_source_pin_is_rejected(self):
        del self.manifest["source_files_sha256"][str(self.source)]
        self.rewrite_manifest()
        self.assert_rejected_without_closure()

    def test_manifest_changed_current_source_pin_is_rejected(self):
        self.manifest["source_files_sha256"][str(self.source)] = "f" * 64
        self.rewrite_manifest()
        self.assert_rejected_without_closure()

    def test_review_cannot_point_to_another_manifest(self):
        self.review["manifest_sha256"] = "0" * 64
        self.write_review()
        self.assert_rejected_without_closure()

    def test_review_must_cover_all_qa_files_exactly(self):
        self.review["reviewed_qa_files_sha256"] = {}
        self.write_review()
        self.assert_rejected_without_closure()

    def test_review_cannot_change_qa_file_digest(self):
        self.review["reviewed_qa_files_sha256"]["qa/sheet_001.png"] = "a" * 64
        self.write_review()
        self.assert_rejected_without_closure()

    def test_review_must_inspect_all_development_outputs(self):
        self.review["development_outputs_visually_inspected"] = 0
        self.write_review()
        self.assert_rejected_without_closure()

    def test_failed_visual_review_cannot_be_closed(self):
        self.review["technical_visualization_passed"] = False
        self.write_review()
        self.assert_rejected_without_closure()

    def test_review_must_have_exact_scope_and_reviewer(self):
        for key, value in (("review_scope", "clinical_approval"),
                           ("reviewer", "unidentified"), ("version", "foreign_review")):
            with self.subTest(key=key):
                original = self.review[key]
                self.review[key] = value
                self.write_review()
                self.assert_rejected_without_closure()
                self.review[key] = original
                self.write_review()

    def test_extra_routine_membership_cannot_enter_closure(self):
        extra = deepcopy(self.manifest["rows"][0])
        extra.update(src_subject_id="synthetic_routine", role="routine", index=2)
        self.manifest["rows"].append(extra)
        self.rewrite_manifest()
        self.assert_rejected_without_closure()

    def test_replacement_routine_membership_cannot_enter_closure(self):
        self.manifest["rows"][0].update(src_subject_id="synthetic_routine", role="routine")
        self.rewrite_manifest()
        self.assert_rejected_without_closure()

    def test_duplicate_membership_cannot_enter_closure(self):
        self.manifest["rows"].append(deepcopy(self.manifest["rows"][0]))
        self.rewrite_manifest()
        self.assert_rejected_without_closure()

    def test_changed_derived_tensor_is_rejected(self):
        np.save(self.tensor_path, np.zeros_like(self.tensor), allow_pickle=False)
        self.assert_rejected_without_closure()

    def test_changed_visualization_file_is_rejected(self):
        Image.new("RGB", (2, 2), "white").save(self.qa_path, format="PNG")
        self.assert_rejected_without_closure()

    def test_changed_source_file_is_rejected(self):
        self.source.write_text('{"synthetic":false}\n', encoding="utf-8")
        self.assert_rejected_without_closure()

    def test_changed_native_source_is_rejected(self):
        np.save(self.native, np.zeros((3, 4), dtype=np.uint16), allow_pickle=False)
        self.assert_rejected_without_closure()

    def test_blob_write_once_is_idempotent_and_checks_readback(self):
        path = self.output / "synthetic/new_output.bin"
        data = b"synthetic derived content"
        expected = hashlib.sha256(data).hexdigest()
        self.assertEqual(subject._blob(path, data), expected)
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(subject._blob(path, data), expected)
        self.assertEqual(path.read_bytes(), data)

    def test_blob_cannot_overwrite_changed_derived_content(self):
        path = self.output / "synthetic/new_output.bin"
        subject._blob(path, b"first")
        with self.assertRaisesRegex(ValueError, "existing_derived_output_changed"):
            subject._blob(path, b"second")
        self.assertEqual(path.read_bytes(), b"first")

    def test_png_uses_immutable_binary_output_path(self):
        path = self.output / "synthetic/new_visualization.png"
        first = subject._png(path, Image.new("RGB", (2, 2), "black"))
        self.assertEqual(subject.sha_file(path), first)
        self.assertEqual(subject._png(path, Image.new("RGB", (2, 2), "black")), first)
        with self.assertRaisesRegex(ValueError, "existing_derived_output_changed"):
            subject._png(path, Image.new("RGB", (2, 2), "white"))


if __name__ == "__main__":
    unittest.main()
