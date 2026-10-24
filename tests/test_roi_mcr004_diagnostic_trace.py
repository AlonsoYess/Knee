"""Synthetic-only checks for read-only, equivalent MCR004 instrumentation."""

import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np

DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("cv2", "scipy"))
if DEPENDENCIES:
    from knee import roi_mcr004_diagnostic_trace as diagnostic
    from knee.roi_mcr004 import candidate_for_half
    from knee.third_party.emory_hiti.config import CropConfig


def synthetic_knee():
    image = np.zeros((1200, 600), dtype=np.uint16)
    yy, xx = np.ogrid[:1200, :600]
    image[((xx-300)**2/170**2+(yy-450)**2/250**2) < 1] = 5000
    image[((xx-300)**2/130**2+(yy-840)**2/260**2) < 1] = 4500
    return image


@unittest.skipUnless(DEPENDENCIES, "requires the isolated MCR004 environment")
class DiagnosticTraceTests(unittest.TestCase):
    def test_both_sides_match_production_and_preserve_native_pixels(self):
        native = synthetic_knee()
        original = native.copy()
        for side in ("RIGHT", "LEFT"):
            with self.subTest(side=side):
                reference = candidate_for_half(native, side, "MONOCHROME2", 0, .15, .20)
                traced = diagnostic.trace_for_half(native, side)
                self.assertEqual(traced["status"], "equivalent_candidate")
                self.assertEqual(traced["requested_working_box"], reference["requested_working_box"])
                self.assertEqual(traced["native_half_box"], reference["native_half_box"])
                self.assertEqual(traced["trace"], reference["trace"])
                self.assertEqual(traced["native_crop_shape"], list(reference["crop"].shape))
                self.assertEqual(traced["native_crop_sha256"], diagnostic._array_hash(reference["crop"]))
                self.assertTrue(all(traced["equivalence_checks"].values()))
        np.testing.assert_array_equal(native, original)

    def test_both_photometric_polarities_have_equal_geometry(self):
        native = synthetic_knee()
        for side in ("LEFT", "RIGHT"):
            direct = diagnostic.trace_for_half(native, side)
            inverted = diagnostic.trace_for_half(5000-native, side, "MONOCHROME1")
            self.assertEqual(direct["requested_working_box"], inverted["requested_working_box"])
            self.assertEqual(direct["native_half_box"], inverted["native_half_box"])

    def test_json_contains_no_arrays_or_production_crop(self):
        traced = diagnostic.trace_for_half(synthetic_knee(), "RIGHT")
        decoded = json.loads(json.dumps(traced, allow_nan=False))
        self.assertEqual(decoded, traced)
        self.assertNotIn("crop", traced)
        self.assertNotIn("crop_height_mm", traced)
        self.assertFalse(traced["production_crop_written"])
        self.assertFalse(traced["review_decision_changed"])
        self.assertTrue(traced["historical_record_comparison_required"])
        self.assertFalse(traced["semantic_mask_confirmation"])

    def test_invalid_inputs_stop_instead_of_inventing_diagnostics(self):
        for native, side, photometric in (
            (np.zeros((1200, 600), np.uint16), "RIGHT", "MONOCHROME2"),
            (synthetic_knee(), "UNKNOWN", "MONOCHROME2"),
            (synthetic_knee(), "RIGHT", "RGB"),
        ):
            with self.subTest(side=side, photometric=photometric):
                with self.assertRaisesRegex(ValueError, "diagnostic_requires_existing_candidate"):
                    diagnostic.trace_for_half(native, side, photometric)

    def test_detector_preserves_existing_area_rule_and_row_selection(self):
        image = np.full((600, 500), 200, np.uint8)
        image[50:150, 20:120] = 0  # Exactly 10000: upstream's comparison is strict.
        rows, records, _ = diagnostic._black_box_trace(image, CropConfig())
        self.assertEqual(rows, [])
        self.assertEqual(records, [])
        image[250:400, 250:400] = 0
        rows, records, _ = diagnostic._black_box_trace(image, CropConfig())
        self.assertEqual(rows, [250, 399])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["lateral_bbox"], [250, 400, 250, 400])

    def test_intersection_uses_offset_and_half_open_roi(self):
        image = np.full((600, 500), 200, np.uint8)
        image[250:400, 250:400] = 0
        _, records, contours = diagnostic._black_box_trace(image, CropConfig())
        diagnostic._attach_intersections(records, contours, image.shape, 100, [400, 500, 300, 350])
        self.assertEqual(records[0]["working_bbox"], [350, 500, 250, 400])
        self.assertEqual(records[0]["final_roi_intersection_pixels"], 5000)
        self.assertTrue(records[0]["intersects_final_roi"])
        diagnostic._attach_intersections(records, contours, image.shape, 100, [500, 550, 300, 350])
        self.assertEqual(records[0]["final_roi_intersection_pixels"], 0)
        self.assertFalse(records[0]["intersects_final_roi"])

    def test_three_detector_routes_and_search_band_changes_with_fixed_parameters(self):
        # These are controlled dark rectangles, never radiographs or fitted cases.
        for interval, expected_route in (
            (None, "none_detected"),
            ((100, 280), "detected_outside_final_roi"),
            ((350, 520), "detected_but_final_roi_intersects"),
            ((750, 920), "detected_outside_final_roi"),
        ):
            with self.subTest(interval=interval):
                native = synthetic_knee()
                native[native == 0] = 100
                if interval:
                    native[interval[0]:interval[1], 30:180] = 0
                traced = diagnostic.trace_for_half(native, "RIGHT")
                self.assertEqual(traced["mask_route"], expected_route)
                self.assertEqual(traced["search_band_before"], [200, 800])
                start, end = traced["search_band_before"]
                before, after = traced["black_box_rows_before"], traced["black_box_rows_after"]
                if before:
                    start = max(start, max(before)+CropConfig().black_box_offset)
                if after:
                    end = min(end, min(after)-CropConfig().black_box_offset)
                self.assertEqual(traced["search_band_after"], [start, end])
                self.assertGreaterEqual(traced["notch"], start)
                self.assertLess(traced["notch"], end)
                self.assertEqual(traced["mask_intersects_final_roi"],
                                 expected_route == "detected_but_final_roi_intersects")
                if interval:
                    self.assertNotEqual(traced["search_band_before"], traced["search_band_after"])
                self.assertTrue(all(traced["equivalence_checks"].values()))

    def test_left_detector_coordinates_include_lateral_offset(self):
        native = synthetic_knee()
        native[native == 0] = 100
        native[350:520, 30:180] = 0
        traced = diagnostic.trace_for_half(np.fliplr(native), "LEFT")
        offset = traced["black_box_lateral_offset_x"]
        self.assertEqual(offset, traced["horizontal_peak"])
        self.assertGreater(offset, 0)
        self.assertTrue(traced["black_box_contours"])
        for record in traced["black_box_contours"]:
            x0, x1, y0, y1 = record["lateral_bbox"]
            self.assertEqual(record["working_bbox"], [x0+offset, x1+offset, y0, y1])
            for (lx, ly), (wx, wy) in zip(record["lateral_polygon"], record["working_polygon"]):
                self.assertEqual([wx, wy], [lx+offset, ly])

    def test_upstream_detector_disagreement_stops(self):
        image = np.full((600, 500), 200, np.uint8)
        with patch.object(diagnostic.upstream, "get_black_box_rows", return_value=[1]):
            with self.assertRaisesRegex(diagnostic.DiagnosticMismatch, "black_box_rows_mismatch"):
                diagnostic._black_box_trace(image, CropConfig())

    def test_instrumented_box_disagreement_stops(self):
        instrument = diagnostic._instrument_core
        def changed_box(*args):
            crop, details = instrument(*args)
            details["requested_working_box"][0] += 1
            return crop, details
        with patch.object(diagnostic, "_instrument_core", side_effect=changed_box):
            with self.assertRaisesRegex(diagnostic.DiagnosticMismatch, "upstream_working_box_mismatch"):
                diagnostic.trace_for_half(synthetic_knee(), "RIGHT")

    def test_instrumented_pixel_disagreement_stops(self):
        instrument = diagnostic._instrument_core
        def changed_pixels(*args):
            crop, details = instrument(*args)
            crop = crop.copy()
            crop[0, 0] ^= 1
            return crop, details
        with patch.object(diagnostic, "_instrument_core", side_effect=changed_pixels):
            with self.assertRaisesRegex(diagnostic.DiagnosticMismatch, "upstream_working_pixels_mismatch"):
                diagnostic.trace_for_half(synthetic_knee(), "RIGHT")

    def test_production_native_disagreement_stops(self):
        production = diagnostic.adapter.candidate_for_half
        for changed, expected in (("box", "adapter_native_box_mismatch"),
                                  ("pixels", "adapter_native_pixels_mismatch")):
            def changed_reference(*args):
                result = production(*args)
                if changed == "box":
                    result["native_half_box"][0] += 1
                else:
                    result["crop"] = result["crop"].copy()
                    result["crop"][0, 0] ^= 1
                return result
            with self.subTest(changed=changed):
                with patch.object(diagnostic.adapter, "candidate_for_half", side_effect=changed_reference):
                    with self.assertRaisesRegex(diagnostic.DiagnosticMismatch, expected):
                        diagnostic.trace_for_half(synthetic_knee(), "RIGHT")


if __name__ == "__main__":
    unittest.main()
