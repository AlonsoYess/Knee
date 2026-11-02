"""Synthetic edge recovery, exact path optimization and native-pixel contracts."""
import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from knee.roi_contour_geometry import _trace, contour_boxes, minimum_cost_path, run
from knee.roi_external_candidate import candidate_from_prediction


def synthetic_knee(dr=.5, dc=.5, left=60., right=140., shift_y=0.):
    """Known physical rectangular bones separated by a 10-mm dark joint gap."""
    yy, xx = np.meshgrid(np.arange(round(120/dr)+1)*dr,
                         np.arange(round(200/dc)+1)*dc, indexing="ij")
    sigmoid = lambda z: 1/(1+np.exp(np.clip(-z, -700, 700)))
    lateral = sigmoid((xx-left)/.6)*sigmoid((right-xx)/.6)
    bone = lateral*(sigmoid((55+shift_y-yy)/.6)+sigmoid((yy-65-shift_y)/.6))
    raw = np.rint(100+1800*bone).astype(np.uint16)
    points_mm = np.vstack((
        np.column_stack((np.linspace(left, right, 9), np.full(9, 65+shift_y))),
        np.column_stack((np.linspace(left, right, 7), np.full(7, 55+shift_y))),
    ))
    return raw, points_mm/[dc, dr]


class MinimumCostPathTests(unittest.TestCase):
    def test_matches_exhaustive_optimization_on_small_problem(self):
        costs = np.array([[.7, -.3, .8, .2], [.1, .6, -.4, .9], [.9, .2, .5, -.7]])
        weight, max_step = .37, 1
        path = minimum_cost_path(costs, weight, max_step)
        def energy(p):
            return sum(costs[p[i], i] for i in range(4)) + weight*sum(
                (p[i]-p[i-1])**2 for i in range(1, 4))
        feasible = [p for p in itertools.product(range(3), repeat=4)
                    if max(abs(p[i]-p[i-1]) for i in range(1, 4)) <= max_step]
        self.assertAlmostEqual(energy(path), min(map(energy, feasible)))
        self.assertLessEqual(int(np.abs(np.diff(path)).max()), max_step)

    def test_transition_constraint_prevents_attractive_unreachable_jumps(self):
        costs = np.zeros((5, 5))
        costs[0, ::2] = -100
        costs[4, 1::2] = -100
        path = minimum_cost_path(costs, max_step=1)
        self.assertTrue(np.all(np.abs(np.diff(path)) <= 1))
        stationary = minimum_cost_path(costs, max_step=0)
        self.assertTrue(np.all(stationary == stationary[0]))

    def test_single_column_and_invalid_costs(self):
        np.testing.assert_array_equal(minimum_cost_path([[2], [-3], [1]]), [1])
        for bad in ([], [1, 2], np.empty((0, 3)), [[np.nan]], [[np.inf]]):
            with self.subTest(cost=repr(bad)), self.assertRaisesRegex(ValueError, "invalid_path_cost"):
                minimum_cost_path(bad)

    def test_gradient_sign_and_physical_search_recover_known_ridge(self):
        gradient = np.zeros((101, 101), dtype=np.float32)
        gradient[50, :] = 1
        prior = np.column_stack((np.linspace(5, 45, 81), np.full(81, 24.)))
        before = prior.copy()
        for sign in (-1, 1):
            trace, metrics = _trace(sign*gradient, prior, 1, sign, 3., .5)
            np.testing.assert_allclose(trace[:, 1], 25.)
            np.testing.assert_allclose(trace[:, 0], prior[:, 0])
            self.assertEqual(metrics["maximum_displacement_mm"], 1.)
        np.testing.assert_array_equal(prior, before)
        with self.assertRaisesRegex(ValueError, "no_gradient_evidence"):
            _trace(np.zeros_like(gradient), prior, 1, 1, 3., .5)


class ContourGeometryTests(unittest.TestCase):
    def test_known_bone_edges_and_both_geometries_retain_surface_context(self):
        raw, points = synthetic_knee()
        boxes, metadata = contour_boxes(raw, points, .5, .5, "MONOCHROME2")
        for name, target_y in (("femur", 55.), ("tibia", 65.)):
            trace = np.asarray(metadata["traces_native_half_xy"][name])*.5
            np.testing.assert_allclose(trace[:, 1], target_y, atol=.5)
        for name, target_x in (("femur_left", 60.), ("tibia_left", 60.),
                               ("femur_right", 140.), ("tibia_right", 140.)):
            trace = np.asarray(metadata["traces_native_half_xy"][name])*.5
            np.testing.assert_allclose(trace[:, 0], target_x, atol=.5)
        for box in boxes.values():
            self.assertEqual(box["status"], "candidate")
            x0, x1, y0, y1 = box["native_half_box"]
            self.assertLess(x0, points[:, 0].min())
            self.assertGreater(x1, points[:, 0].max())
            fractions = (points[:, 1]-y0)/(y1-y0)
            self.assertGreaterEqual(float(fractions.min()), .25)
            self.assertLessEqual(float(fractions.max()), .75)
        self.assertLess(boxes["contour_compact"]["crop_height_mm"],
                        boxes["contour_full_height"]["crop_height_mm"])
        self.assertFalse(metadata["coverage_certified"])
        self.assertFalse(metadata["artifact_absence_certified"])

    def test_input_pixels_points_and_supervised_native_slices_are_unchanged(self):
        raw, points = synthetic_knee()
        original, original_points = raw.copy(), points.copy()
        boxes, _ = contour_boxes(raw, points, .5, .5, "MONOCHROME2")
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        for box in boxes.values():
            prediction = dict(box, source_half_sha256=digest, metadata={})
            result = candidate_from_prediction(raw, prediction, 31, .5, .5)
            a, b, c, d = box["native_half_box"]
            np.testing.assert_array_equal(result["crop"], original[c:d, a:b])
            self.assertEqual(result["native_dicom_box"], [31+a, 31+b, c, d])
            self.assertEqual(result["crop"].dtype, raw.dtype)
            result["crop"][0, 0] ^= 1
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(points, original_points)

    def test_monochrome1_complement_recovers_same_boxes_and_traces(self):
        raw, points = synthetic_knee()
        direct, direct_meta = contour_boxes(raw, points, .5, .5, "MONOCHROME2")
        inverse = np.uint16(4095)-raw
        inverted, inverse_meta = contour_boxes(inverse, points, .5, .5, "MONOCHROME1")
        for name in direct:
            self.assertEqual(direct[name]["native_half_box"], inverted[name]["native_half_box"])
        for name, trace in direct_meta["traces_native_half_xy"].items():
            np.testing.assert_allclose(trace, inverse_meta["traces_native_half_xy"][name])
        np.testing.assert_array_equal(inverse, np.uint16(4095)-raw)

    def test_anisotropic_and_finer_sampling_preserve_physical_geometry(self):
        reference = None
        for dr, dc in ((.5, .5), (.25, .5), (.5, .25), (.25, .25)):
            raw, points = synthetic_knee(dr, dc)
            boxes, metadata = contour_boxes(raw, points, dr, dc, "MONOCHROME2")
            physical = {name: np.array(v["native_half_box"])*[dc, dc, dr, dr]
                        for name, v in boxes.items()}
            if reference is None:
                reference = physical
            else:
                for name in reference:
                    np.testing.assert_allclose(physical[name], reference[name], atol=.5)
            for name, target_y in (("tibia", 65.), ("femur", 55.)):
                trace = np.asarray(metadata["traces_native_half_xy"][name])*[dc, dr]
                np.testing.assert_allclose(trace[:, 1], target_y, atol=.5)

    def test_physical_translation_moves_both_boxes_without_changing_size(self):
        raw, points = synthetic_knee()
        first, _ = contour_boxes(raw, points, .5, .5, "MONOCHROME2")
        shifted_raw, shifted_points = synthetic_knee(left=67., right=147., shift_y=4.)
        shifted, _ = contour_boxes(shifted_raw, shifted_points, .5, .5, "MONOCHROME2")
        for name in first:
            np.testing.assert_array_equal(shifted[name]["native_half_box"],
                np.array(first[name]["native_half_box"])+[14, 14, 8, 8])

    def test_new_requested_margin_outside_source_abstains_without_repair(self):
        raw, points = synthetic_knee(left=2., right=22.)
        boxes, _ = contour_boxes(raw, points, .5, .5, "MONOCHROME2")
        for box in boxes.values():
            self.assertEqual(box["status"], "abstain")
            self.assertEqual(box["error_code"], "contour_box_outside_frozen_half")
            self.assertLess(box["native_half_box"][0], 0)
            prediction = dict(box, source_half_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
            result = candidate_from_prediction(raw, prediction, 0, .5, .5)
            self.assertNotIn("crop", result)

    def test_invalid_pixels_photometric_and_degenerate_surface_raise(self):
        raw, points = synthetic_knee()
        with self.assertRaisesRegex(ValueError, "unsupported_photometric"):
            contour_boxes(raw, points, .5, .5, "RGB")
        with self.assertRaisesRegex(ValueError, "constant_signal"):
            contour_boxes(np.zeros_like(raw), points, .5, .5, "MONOCHROME2")
        for invalid in (np.nan, np.inf):
            broken = raw.astype(float)
            broken[50, 50] = invalid
            with self.assertRaisesRegex(ValueError, "invalid_native_pixels"):
                contour_boxes(broken, points, .5, .5, "MONOCHROME2")
        points[:9, 0] = 200
        with self.assertRaisesRegex(ValueError, "collapsed_surface_x"):
            contour_boxes(raw, points, .5, .5, "MONOCHROME2")


class ContourRunnerIntegrityTests(unittest.TestCase):
    def fixture(self, root):
        upstream, verified = root/"upstream", root/"verified"
        upstream.mkdir()
        (verified/"halves").mkdir(parents=True)
        raw, points = synthetic_knee()
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        rows, references = [], []
        for case in range(1, 11):
            for side in ("RIGHT", "LEFT"):
                alias = f"case_{case:03d}_{side}"
                np.save(verified/"halves"/(alias+".npy"), raw, allow_pickle=False)
                rows.append(dict(knee_alias=alias, source_half_sha256=digest,
                    status="candidate", native_half_box=[0, 401, 0, 241],
                    metadata={"landmarks_native_half_xy": points.tolist()}))
                references.append(dict(knee_alias=alias, row_spacing_mm=.5,
                                       column_spacing_mm=.5, photometric="MONOCHROME2"))
        path = upstream/"predictions.json"
        path.write_text(json.dumps(dict(manifest={"source": "synthetic"}, rows=rows)), encoding="utf-8")
        evidence = dict(rows=references, summary=dict(exact_baseline_replays=20,
            external_predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (verified/"results.json").write_text(json.dumps(evidence), encoding="utf-8")
        return path, verified

    def test_all_synthetic_outputs_have_provenance_and_no_quality_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, verified = self.fixture(root)
            before_source, before_results = source.read_bytes(), (verified/"results.json").read_bytes()
            before_halves = {p.name: p.read_bytes() for p in (verified/"halves").glob("*.npy")}
            target = root/"derived"
            with contextlib.redirect_stdout(io.StringIO()):
                run(source, verified, target)
            self.assertTrue((target/"configuration_before_execution.json").is_file())
            self.assertEqual(source.read_bytes(), before_source)
            self.assertEqual((verified/"results.json").read_bytes(), before_results)
            for p in (verified/"halves").glob("*.npy"):
                self.assertEqual(p.read_bytes(), before_halves[p.name])
            for mode in ("contour_full_height", "contour_compact"):
                data = json.loads((target/(mode+"_predictions.json")).read_text())
                self.assertEqual(len(data["rows"]), 20)
                self.assertFalse(data["manifest"]["acceptability_assigned"])
                self.assertTrue(data["manifest"]["experimental_only"])
                self.assertFalse(data["manifest"]["source_pixel_modification"])
                self.assertTrue(all(r["status"] == "candidate" for r in data["rows"]))

    def test_source_hash_mismatch_never_publishes_predictions(self):
        for tamper in ("predictions", "half"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, verified = self.fixture(root)
                if tamper == "predictions":
                    source.write_bytes(source.read_bytes()+b"\n")
                    message = "upstream_predictions_hash_mismatch"
                else:
                    path = verified/"halves"/"case_001_RIGHT.npy"
                    array = np.load(path, allow_pickle=False)
                    array[0, 0] ^= 1
                    np.save(path, array, allow_pickle=False)
                    message = "source_half_hash_mismatch"
                with self.assertRaisesRegex(ValueError, message):
                    run(source, verified, root/"derived")
                self.assertFalse(list((root/"derived").glob("*_predictions.json")))


if __name__ == "__main__":
    unittest.main()
