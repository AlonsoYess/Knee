"""Independent synthetic tests for the one approved contact-anchor change.

These tests do not load radiographs, review labels, sealed reference centers or
the development sample.  Frozen fixtures use the previous envelope-midpoint
geometry; only the new vertical anchor is under test.
"""
import copy
import inspect
import math
import unittest

import numpy as np

from knee.roi_contact_geometry import contact_box


CONTACT_INDICES = (1, 7, 10, 14)


def fixture(points=None, traces=None, shape=(500, 600), dr=.5, dc=.25):
    """Construct a self-consistent previous record without production helpers."""
    if points is None:
        points = np.vstack((
            np.column_stack((np.linspace(120., 480., 9),
                             [214., 220., 219., 210., 198., 206., 217., 228., 220.])),
            np.column_stack((np.linspace(120., 480., 7),
                             [190., 192., 194., 180., 193., 198., 196.])),
        ))
    points = np.array(points, dtype=float, copy=True)
    if traces is None:
        traces = {
            "tibia": [[120., 214.], [300., 196.], [480., 230.]],
            "femur": [[120., 188.], [300., 178.], [480., 196.]],
            "tibia_left": [[100., 230.], [104., 270.]],
            "tibia_right": [[496., 230.], [500., 270.]],
            "femur_left": [[100., 130.], [104., 178.]],
            "femur_right": [[496., 130.], [500., 178.]],
        }
    traces = copy.deepcopy(traces)
    surface_y = np.concatenate((points[:, 1],
        np.asarray(traces["tibia"], dtype=float)[:, 1],
        np.asarray(traces["femur"], dtype=float)[:, 1])) * dr
    sy0, sy1 = float(surface_y.min()), float(surface_y.max())
    lateral_x = [points[:, 0]] + [np.asarray(v, dtype=float)[:, 0]
        for key, v in traces.items() if key not in ("tibia", "femur")]
    all_x = np.concatenate(lateral_x)
    left = float(all_x.min() * dc - 5.)
    right = float(all_x.max() * dc + 5.)
    height = max((right-left)/2.4, 2.*(sy1-sy0)+2.*dr)
    old_center = (sy0+sy1)/2.
    requested = [left/dc, right/dc,
                 (old_center-height/2.)/dr, (old_center+height/2.)/dr]
    a, b, c, d = (math.floor(requested[0]), math.ceil(requested[1]),
                  math.floor(requested[2]), math.ceil(requested[3]))
    inside = 0 <= a < b <= shape[1] and 0 <= c < d <= shape[0]
    geometry = {
        "status": "candidate" if inside else "abstain",
        "error_code": "" if inside else "contour_box_outside_frozen_half",
        "requested_native_half_box": requested,
        "native_half_box": [a, b, c, d],
        "surface_vertical_fraction": [(sy0/dr-c)/(d-c), (sy1/dr-c)/(d-c)],
        "crop_width_mm": (b-a)*dc,
        "crop_height_mm": (d-c)*dr,
    }
    metadata = {"traces_native_half_xy": traces,
                "surface_envelope_y_mm": [sy0, sy1],
                "coverage_certified": False,
                "artifact_absence_certified": False}
    return points, shape, dr, dc, geometry, metadata


def narrow_fixture(low, high, selected_at_low=True, shape=(500, 600)):
    points = np.column_stack((np.linspace(280., 360., 16),
                              np.full(16, high if selected_at_low else low)))
    points[list(CONTACT_INDICES), 1] = low if selected_at_low else high
    traces = {"tibia": [[280., low], [360., high]],
              "femur": [[280., low], [360., high]]}
    return fixture(points, traces, shape)


class ContactAnchorGeometryTests(unittest.TestCase):
    def test_exact_fixed_zero_based_quartet_and_equal_pair_weight(self):
        args = fixture()
        result = contact_box(*args)
        # First compartment: (220+192)/2; second: (228+198)/2.
        expected_native_center = (206.+213.)/2.
        self.assertEqual(result["vertical_anchor_native_y"], expected_native_center)
        self.assertEqual(result["vertical_anchor_y_mm"], 104.75)
        self.assertNotEqual(expected_native_center, float(args[0][:, 1].mean()))
        self.assertNotEqual(expected_native_center,
                            float(np.mean(args[5]["surface_envelope_y_mm"]))/args[2])

    def test_each_selected_landmark_has_exactly_one_quarter_weight(self):
        points, shape, dr, dc, _, metadata = fixture()
        reference = contact_box(*fixture(points, metadata["traces_native_half_xy"]))
        for index in CONTACT_INDICES:
            with self.subTest(index=index):
                changed = points.copy()
                changed[index, 1] += 4.
                output = contact_box(*fixture(changed, metadata["traces_native_half_xy"]))
                self.assertEqual(output["vertical_anchor_native_y"],
                                 reference["vertical_anchor_native_y"]+1.)
                self.assertEqual(output["vertical_anchor_y_mm"],
                                 reference["vertical_anchor_y_mm"]+dr)

    def test_nonselected_landmark_can_expand_envelope_but_not_anchor(self):
        args = fixture()
        before = contact_box(*args)
        points = args[0].copy()
        points[4, 1] = 160.
        changed = fixture(points, args[5]["traces_native_half_xy"])
        after = contact_box(*changed)
        self.assertEqual(after["vertical_anchor_native_y"], before["vertical_anchor_native_y"])
        self.assertGreater(after["crop_height_mm"], before["crop_height_mm"])
        y0, y1 = after["native_half_box"][2:]
        self.assertGreaterEqual((points[4, 1]-y0)*args[2], args[2])
        self.assertGreaterEqual((y1-points[:, 1].max())*args[2], args[2])

    def test_both_frozen_surface_paths_still_determine_full_height_envelope(self):
        args = fixture()
        original = contact_box(*args)
        for name, y in (("femur", 150.), ("tibia", 260.)):
            with self.subTest(surface=name):
                traces = copy.deepcopy(args[5]["traces_native_half_xy"])
                traces[name][1][1] = y
                changed = fixture(args[0], traces)
                result = contact_box(*changed)
                self.assertEqual(result["vertical_anchor_native_y"],
                                 original["vertical_anchor_native_y"])
                self.assertGreater(result["crop_height_mm"], original["crop_height_mm"])
                c, d = result["native_half_box"][2:]
                envelope = np.asarray(changed[5]["surface_envelope_y_mm"])/changed[2]
                np.testing.assert_allclose(result["surface_vertical_fraction"],
                    (envelope-c)/(d-c))
                self.assertLessEqual(c+1., envelope.min())
                self.assertGreaterEqual(d-1., envelope.max())

    def test_lateral_trace_vertical_extrema_are_not_contact_anchor_inputs(self):
        args = fixture()
        baseline = contact_box(*args)
        traces = copy.deepcopy(args[5]["traces_native_half_xy"])
        for name in ("femur_left", "femur_right", "tibia_left", "tibia_right"):
            traces[name][0][1] = 0.
            traces[name][1][1] = 499.
        result = contact_box(*fixture(args[0], traces))
        self.assertEqual(result["vertical_anchor_y_mm"], baseline["vertical_anchor_y_mm"])
        self.assertEqual(result["requested_native_half_box"],
                         baseline["requested_native_half_box"])
        self.assertEqual(result["native_half_box"], baseline["native_half_box"])

    def test_horizontal_mirror_preserves_contact_anchor_and_mirrors_x_bounds(self):
        args = fixture()
        result = contact_box(*args)
        points = args[0].copy()
        points[:, 0] = args[1][1]-points[:, 0]
        traces = copy.deepcopy(args[5]["traces_native_half_xy"])
        for path in traces.values():
            for point in path:
                point[0] = args[1][1]-point[0]
        mirrored = contact_box(*fixture(points, traces))
        self.assertEqual(mirrored["vertical_anchor_y_mm"], result["vertical_anchor_y_mm"])
        self.assertEqual(mirrored["native_half_box"][2:], result["native_half_box"][2:])
        self.assertEqual(mirrored["native_half_box"][:2],
            [args[1][1]-result["native_half_box"][1],
             args[1][1]-result["native_half_box"][0]])

    def test_physical_spacing_is_axis_specific_and_row_guard_remains_one_pixel(self):
        reference = fixture()
        points_mm = reference[0]*[reference[3], reference[2]]
        paths_mm = {name: np.asarray(path)*[reference[3], reference[2]]
                    for name, path in reference[5]["traces_native_half_xy"].items()}
        for dr, dc, shape in ((.5, .25, (500, 600)), (.25, .5, (1000, 300)),
                              (.25, .25, (1000, 600))):
            with self.subTest(dr=dr, dc=dc):
                points = points_mm/[dc, dr]
                paths = {name: (path/[dc, dr]).tolist() for name, path in paths_mm.items()}
                args = fixture(points, paths, shape, dr, dc)
                result = contact_box(*args)
                self.assertEqual(result["vertical_anchor_y_mm"], 104.75)
                self.assertEqual(result["vertical_anchor_native_y"], 104.75/dr)
                requested = result["requested_native_half_box"]
                np.testing.assert_allclose(np.asarray(requested[:2])*dc, [20., 130.])
                expected_height_mm = max(110./2.4, 2.*26.+2.*dr)
                self.assertAlmostEqual((requested[3]-requested[2])*dr, expected_height_mm)

    def test_native_rounding_is_outward_and_x_and_height_are_unchanged(self):
        args = fixture()
        result = contact_box(*args)
        self.assertEqual(result["requested_native_half_box"], [80., 520., 156.5, 262.5])
        self.assertEqual(result["native_half_box"], [80, 520, 156, 263])
        self.assertEqual(result["requested_native_half_box"][:2],
                         args[4]["requested_native_half_box"][:2])
        old_requested = args[4]["requested_native_half_box"]
        new_requested = result["requested_native_half_box"]
        self.assertEqual(new_requested[3]-new_requested[2], old_requested[3]-old_requested[2])
        self.assertEqual(result["crop_width_mm"], 110.)
        self.assertEqual(result["crop_height_mm"], 53.5)

    def test_surface_envelope_has_at_least_one_native_row_margin_at_anchor_extreme(self):
        args = narrow_fixture(300., 360.)
        result = contact_box(*args)
        self.assertEqual(result["status"], "candidate")
        c, d = result["native_half_box"][2:]
        sy0, sy1 = args[5]["surface_envelope_y_mm"]
        self.assertGreaterEqual(sy0-c*args[2], args[2])
        self.assertGreaterEqual(d*args[2]-sy1, args[2])
        self.assertAlmostEqual(d*args[2]-sy1, args[2])

    def test_contact_rule_does_not_claim_former_all_surface_central_half_guarantee(self):
        result = contact_box(*narrow_fixture(300., 360.))
        self.assertGreater(result["surface_vertical_fraction"][1], .75)
        self.assertEqual(result["status"], "candidate")

    def test_lower_source_overflow_abstains_without_clipping_padding_or_rescue(self):
        args = narrow_fixture(60., 120.)
        self.assertEqual(args[4]["status"], "candidate")
        result = contact_box(*args)
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["error_code"], "contact_box_outside_frozen_half")
        self.assertEqual(result["native_half_box"][2:], [-1, 121])
        self.assertEqual(result["requested_native_half_box"][2:], [-1., 121.])
        self.assertNotIn("crop", result)

    def test_upper_source_overflow_abstains_without_clipping(self):
        args = narrow_fixture(380., 440., selected_at_low=False)
        self.assertEqual(args[4]["status"], "candidate")
        result = contact_box(*args)
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["native_half_box"][2:], [379, 501])
        self.assertEqual(result["requested_native_half_box"][3], 501.)

    def test_half_open_box_may_touch_original_source_boundary(self):
        result = contact_box(*narrow_fixture(379., 439., selected_at_low=False))
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["native_half_box"][3], 500)
        self.assertEqual(result["error_code"], "")

    def test_inputs_and_frozen_records_are_not_mutated(self):
        args = fixture()
        points_before = args[0].copy()
        geometry_before, traces_before = copy.deepcopy(args[4:])
        result = contact_box(*args)
        np.testing.assert_array_equal(args[0], points_before)
        self.assertEqual(args[4], geometry_before)
        self.assertEqual(args[5], traces_before)
        result["native_half_box"][0] += 1
        self.assertEqual(args[4], geometry_before)

    def test_no_manual_source_center_or_quality_labels_are_algorithm_inputs(self):
        parameters = tuple(inspect.signature(contact_box).parameters)
        self.assertEqual(len(parameters), 6)
        for name in parameters:
            self.assertNotIn("review", name)
            self.assertNotIn("manual", name)
            self.assertNotIn("source_center", name)
            self.assertNotIn("quality", name)

    def test_full_sixteen_points_must_be_finite_and_have_xy_shape(self):
        args = fixture()
        invalid = [args[0][:15], np.zeros((16, 3)), args[0].reshape(-1)]
        for value in (np.nan, np.inf, -np.inf):
            points = args[0].copy()
            points[4, 1] = value
            invalid.append(points)
        for points in invalid:
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                contact_box(points, *args[1:])

    def test_full_sixteen_points_must_lie_in_original_half(self):
        args = fixture()
        for axis, value in ((0, -1.), (0, 600.), (1, -1.), (1, 500.)):
            with self.subTest(axis=axis, value=value):
                points = args[0].copy()
                points[4, axis] = value
                with self.assertRaises(ValueError):
                    contact_box(points, *args[1:])

    def test_spacing_must_be_positive_and_finite_in_both_axes(self):
        points, shape, dr, dc, geometry, traces = fixture()
        for value in (0., -1., np.nan, np.inf, -np.inf):
            for axis in ("row", "column"):
                with self.subTest(axis=axis, spacing=value), self.assertRaises(ValueError):
                    contact_box(points, shape, value if axis == "row" else dr,
                                value if axis == "column" else dc, geometry, traces)

    def test_source_shape_must_be_two_positive_integer_dimensions(self):
        args = fixture()
        for shape in ((0, 600), (-1, 600), (500,), (500, 600, 2), (500.5, 600)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                contact_box(args[0], shape, *args[2:])

    def test_both_surface_paths_require_finite_nonempty_xy_arrays(self):
        args = fixture()
        for name in ("tibia", "femur"):
            for path in ([], [1., 2.], [[1., 2., 3.]], [[np.nan, 200.]], [[200., np.inf]]):
                with self.subTest(name=name, path=path):
                    metadata = copy.deepcopy(args[5])
                    metadata["traces_native_half_xy"][name] = path
                    with self.assertRaises(ValueError):
                        contact_box(*args[:5], metadata)

    def test_stored_surface_envelope_must_match_all_original_points_and_paths(self):
        args = fixture()
        for envelope in ([89., 115.1], [90., 115.], [0., 0.], [np.nan, 115.], [89.], None):
            with self.subTest(envelope=envelope):
                metadata = copy.deepcopy(args[5])
                metadata["surface_envelope_y_mm"] = envelope
                with self.assertRaises(ValueError):
                    contact_box(*args[:5], metadata)

    def test_frozen_geometry_height_and_original_midpoint_are_checked(self):
        args = fixture()
        for shift in ((0., 0., -1., 1.), (0., 0., 1., 1.)):
            with self.subTest(shift=shift):
                geometry = copy.deepcopy(args[4])
                geometry["requested_native_half_box"] = (np.asarray(
                    geometry["requested_native_half_box"])+shift).tolist()
                with self.assertRaises(ValueError):
                    contact_box(*args[:4], geometry, args[5])

    def test_inherited_auxiliary_square_validation_is_not_bypassed(self):
        args = fixture()
        points = args[0].copy()
        points[:, 0] -= 115.
        # Original landmarks remain in source, but the inherited square margin
        # extends beyond it.  The anchor-only derivative must still refuse it.
        self.assertGreaterEqual(points[:, 0].min(), 0.)
        with self.assertRaises(ValueError):
            contact_box(points, *args[1:])


if __name__ == "__main__":
    unittest.main()
