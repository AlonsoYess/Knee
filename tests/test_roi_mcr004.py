"""Contract tests for MCR-2026-004; run in the isolated pinned environment."""

import importlib.util
import unittest

import numpy as np

DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("cv2", "scipy"))

if DEPENDENCIES:
    from knee.roi_mcr004 import (
        Abstain, Trace, candidate_for_half, normalize_half,
        preprocess_with_trace,
    )
    from knee.third_party.emory_hiti.config import CropConfig
    from knee.third_party.emory_hiti.pipeline import (
        get_mins, preprocess_and_crop, process_knee_side,
    )


def synthetic_knee():
    image = np.zeros((1200, 600), dtype=np.uint16)
    yy, xx = np.ogrid[:1200, :600]
    image[((xx-300)**2/170**2+(yy-450)**2/250**2) < 1] = 5000
    image[((xx-300)**2/130**2+(yy-840)**2/260**2) < 1] = 4500
    return image


@unittest.skipUnless(DEPENDENCIES, "run in .venv-knee-crop with OpenCV and SciPy")
class RoiMcr004Tests(unittest.TestCase):
    def test_working_copy_does_not_modify_native_pixels(self):
        native = synthetic_knee()
        before = native.copy()
        working = normalize_half(native, "MONOCHROME2")
        self.assertEqual(working.dtype, np.uint8)
        self.assertTrue(np.array_equal(native, before))
        inverted = normalize_half(native, "MONOCHROME1")
        self.assertTrue(np.array_equal(inverted, 255-working))

    def test_pinned_preprocess_matches_upstream(self):
        working = normalize_half(synthetic_knee(), "MONOCHROME2")
        config = CropConfig()
        result, trace = preprocess_with_trace(working, config)
        self.assertTrue(np.array_equal(result, preprocess_and_crop(working, config)))
        self.assertEqual(result.shape, (trace.line_y1-trace.line_y0,
                                        trace.line_x1-trace.line_x0))

    def test_instrumented_core_preserves_upstream_crop_for_both_sides(self):
        working, _ = preprocess_with_trace(
            normalize_half(synthetic_knee(), "MONOCHROME2"), CropConfig()
        )
        for is_right in (True, False):
            with self.subTest(is_right=is_right):
                reference = process_knee_side(working, is_right_knee=is_right)
                traced, (x0, x1, y0, y1) = process_knee_side(
                    working, is_right_knee=is_right, return_geometry=True
                )
                self.assertTrue(np.array_equal(reference, traced))
                self.assertTrue(np.array_equal(traced, working[y0:y1, x0:x1]))
                peak = (
                    100+np.argmax(working.sum(0)[100:-100]) if is_right
                    else np.argmax(working.sum(0)[:-100])
                )
                self.assertEqual((x0, x1), tuple(map(int, get_mins(working, peak))))

    def test_candidate_preserves_native_crop_and_anisotropic_spacing(self):
        native = synthetic_knee()
        before = native.copy()
        for side in ("RIGHT", "LEFT"):
            with self.subTest(side=side):
                result = candidate_for_half(native, side, "MONOCHROME2",
                                            800, 0.15, 0.20)
                self.assertEqual(result["status"], "candidate", result)
                x0, x1, y0, y1 = result["native_half_box"]
                self.assertEqual(result["native_dicom_box"],
                                 [x0+800, x1+800, y0, y1])
                self.assertTrue(np.array_equal(result["crop"], native[y0:y1, x0:x1]))
                self.assertAlmostEqual(result["crop_height_mm"], (y1-y0)*0.15)
                self.assertAlmostEqual(result["crop_width_mm"], (x1-x0)*0.20)
        self.assertTrue(np.array_equal(native, before))

    def test_boundary_transform_rounds_outward_and_rejects_truncation(self):
        trace = Trace(200, 200, 10, 190, 20, 180, 100, 100,
                      5, 0, 100, 0, 90)
        self.assertEqual(trace.to_native_bounds((11, 22, 2, 13)),
                         (29, 50, 31, 49))
        with self.assertRaises(Abstain):
            trace.to_native_bounds((0, 100, -1, 10))
        with self.assertRaises(Abstain):
            trace.to_native_bounds((0, 101, 0, 10))

    def test_uniform_small_invalid_laterality_and_spacing_abstain(self):
        native = synthetic_knee()
        samples = [
            (np.zeros((1200, 600), np.uint16), "RIGHT", .15, .2),
            (np.zeros((20, 20), np.uint16), "RIGHT", .15, .2),
            (native, "UNKNOWN", .15, .2),
            (native, "RIGHT", .15, -1),
        ]
        for image, side, row_mm, col_mm in samples:
            with self.subTest(side=side, shape=image.shape):
                result = candidate_for_half(image, side, "MONOCHROME2",
                                            0, row_mm, col_mm)
                self.assertEqual(result["status"], "abstain")
                self.assertNotIn("crop", result)

    def test_repeated_box_is_exact(self):
        native = synthetic_knee()
        a = candidate_for_half(native, "RIGHT", "MONOCHROME2", 0, .15, .2)
        b = candidate_for_half(native, "RIGHT", "MONOCHROME2", 0, .15, .2)
        self.assertEqual(a["status"], "candidate")
        self.assertEqual(a["native_half_box"], b["native_half_box"])
        self.assertTrue(np.array_equal(a["crop"], b["crop"]))

    def test_unapproved_parameter_change_abstains(self):
        changed = CropConfig()
        changed.clahe_clip = 3.0
        result = candidate_for_half(synthetic_knee(), "RIGHT", "MONOCHROME2",
                                    0, .15, .2, changed)
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["error_code"], "non_pinned_core_parameters")


if __name__ == "__main__":
    unittest.main()
