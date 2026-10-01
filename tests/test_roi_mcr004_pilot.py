"""Gating and per-knee failure tests for the historical-only MCR004 runner."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from knee.dicom_audit import sha256_file

DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("cv2", "scipy"))
if DEPENDENCIES:
    from knee.roi_mcr004_pilot import (
        ALGORITHM_VERSION, run_historical_regression, validate_prior_v04_closure,
    )


def closed_v04(review_hash):
    return {
        "status": "rejected_after_blinded_review",
        "algorithm_version": "tibiofemoral_crop_v0.4_pilot",
        "reviewed_knees": 20, "acceptable_crops": 8, "rejected_crops": 12,
        "technical_exclusions": 0,
        "rejection_reason_counts": {
            "joint_not_centered": 9, "peripheral_artifact_present": 6,
        },
        "parameters_frozen": False, "mass_processing_executed": False,
        "partitions_created": False, "training_executed": False,
        "reserved_test_opened": False,
        "review_csv_sha256": review_hash,
    }


@unittest.skipUnless(DEPENDENCIES, "run in .venv-knee-crop with OpenCV and SciPy")
class RoiMcr004PilotTests(unittest.TestCase):
    def test_requires_matching_colab_closure_and_unchanged_review_csv(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            summary = root/"summary.json"
            record = root/"record.json"
            review_csv = root/"review.csv"
            review_csv.write_bytes(b"reviewed-cases\n")
            review_hash = sha256_file(review_csv)
            summary.write_text(json.dumps(closed_v04(review_hash)), encoding="utf-8")
            payload = {**closed_v04(review_hash),
                       "closure_execution_environment": "Google Colab, libreta 12"}
            # The actual notebook 12 record omits this field.
            del payload["review_csv_sha256"]
            record.write_text(json.dumps(payload), encoding="utf-8")
            result = validate_prior_v04_closure(summary, record, review_csv)
            self.assertIn("summary_sha256", result)
            self.assertEqual(result["review_csv_sha256"], review_hash)
            review_csv.write_bytes(b"changed-review\n")
            with self.assertRaisesRegex(ValueError, "v04_review_csv_hash_mismatch"):
                validate_prior_v04_closure(summary, record, review_csv)
            review_csv.write_bytes(b"reviewed-cases\n")
            payload["acceptable_crops"] = 9
            record.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v04_closure_mismatch"):
                validate_prior_v04_closure(summary, record, review_csv)
            payload["acceptable_crops"] = 8
            payload["closure_execution_environment"] = "local"
            record.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "colab_closure_not_proven"):
                validate_prior_v04_closure(summary, record, review_csv)
            payload["closure_execution_environment"] = "Google Colab, libreta 12"
            payload["review_csv_sha256"] = "wrong-hash"
            record.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v04_review_csv_hash_mismatch"):
                validate_prior_v04_closure(summary, record, review_csv)

    def test_historical_denominator_and_write_failure_becomes_abstain(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            config = {
                "algorithm_version": ALGORITHM_VERSION,
                "expected_unique_studies": 10, "expected_knees": 20,
                "source_dir": root, "bilateral_output_dir": root,
                "pilot_audit_json": root/"audit.json",
                "v04_review_summary_json": root/"summary.json",
                "v04_closure_record_json": root/"record.json",
                "v04_review_csv": root/"review.csv",
                "output_dir": root/"new-output",
                "max_preview_width": 1600,
            }
            bilateral = [{
                "case_alias": f"case_{i:03d}",
                "manifest_key": f"key_{i}",
                "split_column": "600",
            } for i in range(1, 11)]
            audit = {f"key_{i}": {
                "package_relative_path": f"case_{i}.tar",
                "package_sha256": "digest", "dicom_sha256": "digest",
                "pixel_sha256": "digest",
            } for i in range(1, 11)}
            pixels = np.ones((1200, 1200), np.uint16)
            dataset = SimpleNamespace(PhotometricInterpretation="MONOCHROME2",
                                      ImagerPixelSpacing=[.15, .2])
            def fake_candidate(image, side, photometric, half_x0, row_mm, col_mm, cfg):
                return {
                    "status": "candidate", "crop": image[:10, :10].copy(),
                    "native_half_box": [0, 10, 0, 10],
                    "native_dicom_box": [half_x0, half_x0+10, 0, 10],
                    "requested_working_box": [0, 10, 0, 10],
                    "crop_rows": 10, "crop_columns": 10,
                    "crop_height_mm": 1.5, "crop_width_mm": 2.0,
                    "trace": {"synthetic": True},
                }
            preview_calls = 0
            def fake_preview(image, photometric, box, target, max_width):
                nonlocal preview_calls
                preview_calls += 1
                if preview_calls == 1:
                    raise OSError("synthetic write failure")
                target.write_bytes(b"synthetic-preview")
            with (
                patch("knee.roi_mcr004_pilot.validate_prior_v04_closure",
                      return_value={"verified": True}),
                patch("knee.roi_mcr004_pilot._load_closed_bilateral_pilot",
                      return_value=bilateral),
                patch("knee.roi_mcr004_pilot._load_verified_pilot",
                      return_value=audit),
                patch("knee.roi_mcr004_pilot._resolve_package",
                      return_value=root/"source.tar"),
                patch("knee.roi_mcr004_pilot.sha256_file", return_value="digest"),
                patch("knee.roi_mcr004_pilot.sha256_bytes", return_value="digest"),
                patch("knee.roi_mcr004_pilot._read_single_image",
                      return_value=(b"dicom", dataset, pixels)),
                patch("knee.roi_mcr004_pilot.candidate_for_half",
                      side_effect=fake_candidate),
                patch("knee.roi_mcr004_pilot._preview", side_effect=fake_preview),
            ):
                summary = run_historical_regression(config)
            self.assertEqual(summary["expected_knees"], 20)
            self.assertEqual(summary["candidate_knees"], 19)
            self.assertEqual(summary["abstained_knees"], 1)
            with (config["output_dir"]/"resultados_roi_privados.csv").open(
                newline="", encoding="utf-8-sig"
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter=";"))
            self.assertEqual(len(rows), 20)
            self.assertEqual(rows[0]["status"], "abstain")
            self.assertTrue(rows[0]["error_code"].startswith("output_failure:"))
            self.assertEqual(rows[1]["status"], "candidate")
            self.assertFalse((config["output_dir"]/"recortes_nativos"/
                              "case_001_RIGHT_crop.npy").exists())
            with self.assertRaises(FileExistsError):
                run_historical_regression(config)

    def test_rejects_changed_pilot_denominator_before_any_io(self):
        with self.assertRaisesRegex(ValueError, "10_studies_20_knees"):
            run_historical_regression({
                "algorithm_version": ALGORITHM_VERSION,
                "expected_unique_studies": 20, "expected_knees": 40,
            })


if __name__ == "__main__":
    unittest.main()
