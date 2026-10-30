"""Coordinate importer tests, without clinical data or quality labels."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from knee.roi_external_candidate import candidate_from_prediction, load_predictions


class ExternalCandidateTests(unittest.TestCase):
    def setUp(self):
        self.half = np.arange(120, dtype=np.uint16).reshape(12, 10)
        self.pred = dict(knee_alias="case_001_RIGHT", status="candidate",
                         source_half_sha256=hashlib.sha256(self.half.tobytes()).hexdigest(),
                         native_half_box=[1.2, 8.6, 2.4, 10.8], metadata={"source": "fixture"})

    def candidate(self):
        return candidate_from_prediction(self.half, self.pred, 100, .2, .3)

    def test_native_slice_offset_and_outward_rounding(self):
        result = self.candidate()
        self.assertEqual(result["native_half_box"], [1, 9, 2, 11])
        self.assertEqual(result["native_dicom_box"], [101, 109, 2, 11])
        np.testing.assert_array_equal(result["crop"], self.half[2:11, 1:9])
        self.assertEqual(result["crop"].dtype, self.half.dtype)
        self.assertAlmostEqual(result["crop_height_mm"], 1.8)
        self.assertAlmostEqual(result["crop_width_mm"], 2.4)

    def test_copies_not_mutates_or_shares_source(self):
        result = self.candidate()
        result["crop"][0, 0] = 999
        self.assertEqual(self.half[2, 1], 21)

    def test_changed_source_hash_stops(self):
        self.half[0, 0] = 333
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            self.candidate()

    def test_no_clipping_padding_or_repair_of_bad_box(self):
        for box in ([-.1, 8, 2, 10], [1, 10.01, 2, 10], [1, 8, -1, 10],
                    [1, 8, 2, 13], [5, 4, 2, 10], [1, 8, 5, 5]):
            with self.subTest(box=box):
                self.pred["native_half_box"] = box
                result = self.candidate()
                self.assertEqual(result["status"], "abstain")
                self.assertNotIn("crop", result)
                self.assertEqual(result["requested_native_half_box"], box)

    def test_nonfinite_and_malformed_boxes_rejected(self):
        for box in ([0, 1, 2, float("nan")], [0, 1, 2, float("inf")],
                    [0, 1, 2, True], [0, 1, 2], "0,1,2,3"):
            self.pred["native_half_box"] = box
            with self.assertRaises(ValueError):
                self.candidate()

    def test_original_abstention_stays_abstention(self):
        self.pred.update(status="abstain", error_code="fixture_error")
        result = self.candidate()
        self.assertEqual(result["error_code"], "fixture_error")
        self.assertNotIn("crop", result)

    def test_file_requires_matching_unique_aliases_manifest_status_and_digest(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"predictions.json"
            data = {"manifest": {"weights_sha256": "fixture"}, "rows": [self.pred]}
            path.write_text(json.dumps(data), encoding="utf-8")
            rows, manifest, sha = load_predictions(path, [self.pred["knee_alias"]])
            self.assertEqual(sha, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(rows[self.pred["knee_alias"]], self.pred)
            with self.assertRaises(ValueError):
                load_predictions(path, ["other"])
            cases = [dict(data, rows=[self.pred, self.pred]), dict(data, manifest={}),
                     dict(data, rows=[dict(self.pred, status="accept")]),
                     dict(data, rows=[dict(self.pred, source_half_sha256=None)])]
            for changed in cases:
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_predictions(path, [self.pred["knee_alias"]])


if __name__ == "__main__":
    unittest.main()
