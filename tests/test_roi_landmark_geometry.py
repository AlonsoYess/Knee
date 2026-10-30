"""Physical geometry and provenance checks for the frozen landmark experiment."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from knee.roi_landmark_geometry import build_predictions, landmark_box


def surface_points(xs=(100, 120, 170, 200), ys=(144, 150, 154, 160)):
    return np.array([(x, y) for y in ys for x in xs], dtype=float)


class LandmarkGeometryTests(unittest.TestCase):
    def test_anisotropic_physical_square_and_outward_rounding(self):
        result = landmark_box(surface_points(), (400, 400), .25, .5)
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["native_half_box"], [90, 210, 32, 272])
        self.assertAlmostEqual(result["requested_side_mm"], 60)
        x0, x1, y0, y1 = result["native_half_box"]
        self.assertAlmostEqual((x1-x0)*.5, (y1-y0)*.25)
        np.testing.assert_allclose(result["center_native_half_xy"], [150, 152])

    def test_translation_preserves_geometry_and_does_not_mutate_points(self):
        points = surface_points()
        original = points.copy()
        first = landmark_box(points, (500, 500), .4, .4)
        shifted = landmark_box(points+[17, 23], (500, 500), .4, .4)
        np.testing.assert_array_equal(points, original)
        np.testing.assert_allclose(shifted["requested_native_half_box"],
                                   np.array(first["requested_native_half_box"])+[17, 17, 23, 23])
        np.testing.assert_array_equal(shifted["native_half_box"],
                                      np.array(first["native_half_box"])+[17, 17, 23, 23])

    def test_sampling_scale_keeps_physical_roi_when_lateral_context_dominates(self):
        points = surface_points()
        first = landmark_box(points, (400, 400), .25, .5)
        finer = landmark_box(points*3, (1200, 1200), .25/3, .5/3)
        self.assertAlmostEqual(first["requested_side_mm"], finer["requested_side_mm"])
        np.testing.assert_allclose(np.array(finer["requested_native_half_box"])/3,
                                   first["requested_native_half_box"], atol=1e-12)

    def test_sampling_scale_changes_only_physical_rounding_guard_when_vertical_dominates(self):
        points = surface_points(xs=(100, 103, 108, 110), ys=(80, 100, 120, 140))
        first = landmark_box(points, (400, 400), .25, .5)
        finer = landmark_box(points*2, (800, 800), .125, .25)
        self.assertEqual(first["status"], "candidate")
        self.assertEqual(finer["status"], "candidate")
        self.assertAlmostEqual(first["requested_side_mm"], 30.5)
        self.assertAlmostEqual(finer["requested_side_mm"], 30.25)

    def test_lateral_reflection_and_point_order_do_not_change_rule(self):
        points = surface_points()
        first = landmark_box(points, (400, 400), .25, .5)
        reflected = points.copy()
        reflected[:, 0] = 399-reflected[:, 0]
        other = landmark_box(reflected[::-1], (400, 400), .25, .5)
        a, b, c, d = first["requested_native_half_box"]
        np.testing.assert_allclose(other["requested_native_half_box"], [399-b, 399-a, c, d])
        self.assertEqual(other["requested_side_mm"], first["requested_side_mm"])

    def test_fractional_extents_stay_inside_vertical_band_after_rounding(self):
        base = surface_points(xs=(100, 103, 108, 110), ys=(80, 100, 120, 140))
        for shift in (.001, .249, .501, .999):
            with self.subTest(shift=shift):
                points = base+[.17, shift]
                result = landmark_box(points, (400, 400), .23, .17)
                self.assertEqual(result["status"], "candidate")
                x0, x1, y0, y1 = result["native_half_box"]
                fractions = (points[:, 1]-y0)/(y1-y0)
                self.assertGreaterEqual(float(fractions.min()), .25)
                self.assertLessEqual(float(fractions.max()), .75)
                a, b, c, d = result["requested_native_half_box"]
                self.assertTrue(x0 <= a < x0+1 and x1-1 < b <= x1)
                self.assertTrue(y0 <= c < y0+1 and y1-1 < d <= y1)

    def test_invalid_points_spacing_and_shape_abstain(self):
        points = surface_points()
        invalid_points = [None, [], points[:15], points[:, :1], points.astype(str),
                          np.ones((16, 2), dtype=bool), points.copy(), points.copy()]
        invalid_points[-2][0, 0] = np.nan
        invalid_points[-1][0, 1] = np.inf
        for bad in invalid_points:
            with self.subTest(points=repr(bad)[:60]):
                result = landmark_box(bad, (400, 400), .2, .2)
                self.assertEqual(result["error_code"], "invalid_landmarks")
        for dr, dc in ((0, .2), (-1, .2), (True, .2), (.2, np.nan), (.2, np.inf), (.2, "0.2")):
            with self.subTest(spacing=(dr, dc)):
                self.assertEqual(landmark_box(points, (400, 400), dr, dc)["error_code"],
                                 "invalid_pixel_spacing")
        for shape in (None, (), (400,), (400, 0), (400, -1), (400., 400), (True, 400), (400, 400, 1)):
            with self.subTest(shape=shape):
                self.assertEqual(landmark_box(points, shape, .2, .2)["error_code"],
                                 "invalid_half_shape")

    def test_collapsed_envelope_abstains_but_single_coincident_pair_does_not(self):
        points = surface_points()
        for axis in (0, 1):
            bad = points.copy()
            bad[:, axis] = 100
            with self.subTest(axis=axis):
                self.assertEqual(landmark_box(bad, (400, 400), .2, .2)["error_code"],
                                 "degenerate_landmarks")
        points[5] = points[6]
        self.assertEqual(landmark_box(points, (400, 400), .2, .2)["status"], "candidate")

    def test_outside_landmarks_and_overflowing_box_never_clip(self):
        points = surface_points()
        for x, y in ((-.01, 100), (400, 100), (100, -.01), (100, 400)):
            bad = points.copy()
            bad[0] = [x, y]
            with self.subTest(point=(x, y)):
                result = landmark_box(bad, (400, 400), .2, .2)
                self.assertEqual(result["error_code"], "landmarks_outside_frozen_half")
                self.assertNotIn("native_half_box", result)
        bad = surface_points(xs=(0, 20, 70, 100))
        result = landmark_box(bad, (400, 400), .2, .2)
        self.assertEqual(result["error_code"], "landmark_box_outside_frozen_half")
        self.assertLess(result["requested_native_half_box"][0], 0)
        self.assertNotIn("native_half_box", result)


class LandmarkBuilderIntegrityTests(unittest.TestCase):
    def fixture(self, root):
        upstream, verified = root/"upstream", root/"verified"
        upstream.mkdir()
        (verified/"halves").mkdir(parents=True)
        pixels = np.arange(160000, dtype=np.uint16).reshape(400, 400)
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        rows, references = [], []
        for case in range(1, 11):
            for side in ("RIGHT", "LEFT"):
                alias = f"case_{case:03d}_{side}"
                np.save(verified/"halves"/(alias+".npy"), pixels, allow_pickle=False)
                rows.append(dict(knee_alias=alias, source_half_sha256=digest,
                                 status="abstain", native_half_box=[-1, 400, 0, 400],
                                 metadata={"landmarks_native_half_xy": surface_points().tolist()}))
                references.append(dict(knee_alias=alias, row_spacing_mm=.25, column_spacing_mm=.5))
        path = upstream/"predictions.json"
        path.write_text(json.dumps({"manifest": {"source": "synthetic"}, "rows": rows}), encoding="utf-8")
        evidence = {"rows": references, "summary": {"exact_baseline_replays": 20,
                    "external_predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
        (verified/"results.json").write_text(json.dumps(evidence), encoding="utf-8")
        return path, verified

    def test_hash_mismatch_stops_and_does_not_publish_predictions(self):
        for tamper in ("predictions", "half"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, verified = self.fixture(root)
                if tamper == "predictions":
                    source.write_bytes(source.read_bytes()+b"\n")
                    error = "frozen_landmark_predictions_hash_mismatch"
                else:
                    half_path = verified/"halves"/"case_001_RIGHT.npy"
                    pixels = np.load(half_path, allow_pickle=False)
                    pixels[0, 0] ^= 1
                    np.save(half_path, pixels, allow_pickle=False)
                    error = "landmark_source_half_hash_mismatch"
                with self.assertRaisesRegex(ValueError, error):
                    build_predictions(source, verified, root/"derived")
                self.assertFalse((root/"derived"/"predictions.json").exists())

    def test_new_geometry_preserves_upstream_abstention_and_never_assigns_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, verified = self.fixture(root)
            before = source.read_bytes()
            path = build_predictions(source, verified, root/"derived")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(data["rows"]), 20)
            self.assertFalse(data["manifest"]["acceptability_assigned"])
            self.assertFalse(data["manifest"]["production_method_promoted"])
            for row in data["rows"]:
                self.assertEqual(row["status"], "candidate")
                self.assertEqual(row["metadata"]["upstream_140mm_status"], "abstain")
            self.assertTrue((root/"derived"/"rule_before_evaluation.json").is_file())


if __name__ == "__main__":
    unittest.main()
