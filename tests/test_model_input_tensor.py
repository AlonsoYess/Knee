"""Synthetic checks of the development-only, geometry-preserving input transform.

These tests never load patient pixels, labels, checkpoints, or source archives.
"""
from copy import deepcopy
import math
import unittest

import numpy as np

from knee import model_input_tensor as subject


def metadata(**changes):
    result = {
        "photometric": "MONOCHROME2",
        "row_spacing_mm": 1.0,
        "column_spacing_mm": 1.0,
        "side": "RIGHT",
        "native_spatial_orientation": "original_acquisition_no_mirroring",
    }
    result.update(changes)
    return result


def ramp(height=100, width=200, dtype=np.uint16):
    return np.broadcast_to(np.arange(width, dtype=dtype), (height, width)).copy()


class ModelInputGeometryTests(unittest.TestCase):
    def test_isotropic_rectangle_preserves_full_extent_and_adds_padding(self):
        source = ramp()
        tensor, record = subject.transform_native(source, metadata())
        self.assertEqual(record["native_shape"], [100, 200])
        self.assertEqual(record["physical_size_mm"], [100.0, 200.0])
        self.assertEqual(record["resized_shape"], [112, 224])
        self.assertEqual(record["padding_ltrb"], [0, 56, 0, 56])
        np.testing.assert_array_equal(tensor[0, :56], 0)
        np.testing.assert_array_equal(tensor[0, 168:], 0)
        self.assertAlmostEqual(float(tensor[0, 56:168, 0].max()), 0, places=6)
        self.assertAlmostEqual(float(tensor[0, 56:168, -1].min()), 1, places=6)
        self.assertGreater(float(tensor[0, 56:168, 112].mean()), 0.4)

    def test_anisotropic_pixels_use_physical_ratio_instead_of_array_ratio(self):
        tensor, record = subject.transform_native(
            ramp(), metadata(row_spacing_mm=0.2, column_spacing_mm=0.1))
        self.assertEqual(record["physical_size_mm"], [20.0, 20.0])
        self.assertEqual(record["resized_shape"], [224, 224])
        self.assertEqual(record["padding_ltrb"], [0, 0, 0, 0])
        self.assertGreater(float(tensor[0, :, 112].mean()), 0.4)

    def test_tall_rectangle_keeps_top_and_bottom(self):
        source = np.broadcast_to(np.arange(200, dtype=np.uint16)[:, None], (200, 100)).copy()
        tensor, record = subject.transform_native(source, metadata())
        self.assertEqual(record["resized_shape"], [224, 112])
        self.assertEqual(record["padding_ltrb"], [56, 0, 56, 0])
        self.assertAlmostEqual(float(tensor[0, 0, 56:168].max()), 0, places=6)
        self.assertAlmostEqual(float(tensor[0, -1, 56:168].min()), 1, places=6)
        np.testing.assert_array_equal(tensor[0, :, :56], 0)
        np.testing.assert_array_equal(tensor[0, :, 168:], 0)

    def test_odd_padding_places_extra_pixel_at_bottom(self):
        _, record = subject.transform_native(ramp(7, 12), metadata())
        self.assertEqual(record["resized_shape"], [131, 224])
        self.assertEqual(record["padding_ltrb"], [0, 46, 0, 47])

    def test_odd_padding_places_extra_pixel_at_right(self):
        _, record = subject.transform_native(ramp(12, 7), metadata())
        self.assertEqual(record["resized_shape"], [224, 131])
        self.assertEqual(record["padding_ltrb"], [46, 0, 47, 0])

    def test_half_pixel_rounding_is_half_up_not_bankers(self):
        _, record = subject.transform_native(ramp(5, 448), metadata())
        self.assertEqual(record["resized_shape"], [3, 224])
        self.assertEqual(record["padding_ltrb"], [0, 110, 0, 111])

    def test_small_nonzero_physical_extent_never_rounds_to_zero(self):
        _, record = subject.transform_native(ramp(1, 1000), metadata())
        self.assertEqual(record["resized_shape"], [1, 224])
        self.assertEqual(record["padding_ltrb"], [0, 111, 0, 112])

    def test_both_sides_preserve_original_horizontal_orientation(self):
        source = ramp(32, 64)
        right, right_record = subject.transform_native(source, metadata(side="RIGHT"))
        left, left_record = subject.transform_native(source, metadata(side="LEFT"))
        np.testing.assert_array_equal(right, left)
        self.assertLess(float(right[0, 60:164, 5].mean()), 0.1)
        self.assertGreater(float(right[0, 60:164, -6].mean()), 0.9)
        self.assertTrue(right_record["orientation_preserved"])
        self.assertTrue(left_record["orientation_preserved"])


class ModelInputIntensityAndPurityTests(unittest.TestCase):
    def test_percentiles_are_linear_native_sample_percentiles(self):
        source = np.arange(100, dtype=np.uint16).reshape(10, 10)
        _, record = subject.transform_native(source, metadata())
        np.testing.assert_allclose(record["percentile_window"], [0.99, 98.01], rtol=0, atol=1e-12)

    def test_identity_geometry_uses_percentile_clipping_not_minmax_scaling(self):
        source = ramp(224, 224)
        tensor, record = subject.transform_native(source, metadata())
        self.assertEqual(record["resized_shape"], [224, 224])
        np.testing.assert_allclose(record["percentile_window"], [2.0, 221.0], rtol=0, atol=0)
        self.assertAlmostEqual(float(tensor[0, 112, 2]), 0, places=7)
        self.assertAlmostEqual(float(tensor[0, 112, 221]), 1, places=7)
        self.assertAlmostEqual(float(tensor[0, 112, 112]), 110 / 219, places=6)

    def test_bilinear_downsampling_filters_high_frequency_content(self):
        rows, columns = np.indices((448, 448))
        source = ((rows + columns) % 2).astype(np.uint16)
        tensor, _ = subject.transform_native(source, metadata())
        self.assertGreater(float(tensor[0, 5:-5, 5:-5].min()), 0.49)
        self.assertLess(float(tensor[0, 5:-5, 5:-5].max()), 0.51)

    def test_equivalent_photometric_encodings_have_equivalent_output(self):
        source = ramp(50, 100)
        inverse = np.uint16(65535) - source
        mono2, record2 = subject.transform_native(source, metadata())
        mono1, record1 = subject.transform_native(inverse, metadata(photometric="MONOCHROME1"))
        np.testing.assert_allclose(mono1, mono2, rtol=0, atol=2e-7)
        self.assertFalse(record2["inversion_applied"])
        self.assertTrue(record1["inversion_applied"])
        self.assertEqual(record1["photometric"], "MONOCHROME1")
        self.assertEqual(record2["photometric"], "MONOCHROME2")

    def test_mono1_padding_is_black_not_inverted_to_white(self):
        tensor, record = subject.transform_native(ramp(), metadata(photometric="MONOCHROME1"))
        self.assertEqual(record["padding_ltrb"], [0, 56, 0, 56])
        np.testing.assert_array_equal(tensor[0, :56], 0)
        np.testing.assert_array_equal(tensor[0, -56:], 0)
        self.assertGreater(float(tensor[0, 56:168, 0].mean()), 0.99)
        self.assertLess(float(tensor[0, 56:168, -1].mean()), 0.01)

    def test_signed_native_pixels_are_supported_without_integer_overflow(self):
        unsigned = ramp(30, 60)
        signed = unsigned.astype(np.int16) - np.int16(30000)
        expected, _ = subject.transform_native(unsigned, metadata())
        actual, _ = subject.transform_native(signed, metadata())
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)

    def test_unsigned_dynamic_range_is_not_subtracted_in_integer_dtype(self):
        source = np.array([[0, 60000], [200, 65535]], dtype=np.uint16)
        tensor, _ = subject.transform_native(source, metadata())
        self.assertTrue(np.isfinite(tensor).all())
        self.assertGreater(float(tensor[0, -1, -1]), 0.99)
        self.assertLess(float(tensor[0, 0, 0]), 0.01)

    def test_output_contract_and_repeatability(self):
        source = ramp(31, 59)
        first, first_record = subject.transform_native(source, metadata())
        second, second_record = subject.transform_native(source, metadata())
        self.assertEqual(first.shape, (1, 224, 224))
        self.assertEqual(first.dtype, np.dtype("float32"))
        self.assertTrue(first.flags.c_contiguous)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreaterEqual(float(first.min()), 0)
        self.assertLessEqual(float(first.max()), 1)
        self.assertEqual(first_record["output_shape"], [1, 224, 224])
        self.assertEqual(first_record["dtype"], "float32")
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_record, second_record)

    def test_readonly_input_and_metadata_remain_unchanged(self):
        source = ramp(25, 44)
        original_bytes = source.tobytes()
        source.flags.writeable = False
        details = metadata()
        before_details = deepcopy(details)
        tensor, _ = subject.transform_native(source, details)
        self.assertEqual(source.tobytes(), original_bytes)
        self.assertFalse(source.flags.writeable)
        self.assertEqual(details, before_details)
        self.assertFalse(np.shares_memory(source, tensor))

    def test_noncontiguous_integer_view_is_not_modified(self):
        base = ramp(50, 100)
        source = base[::2, ::2]
        before = base.tobytes()
        tensor, _ = subject.transform_native(source, metadata())
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(base.tobytes(), before)


class ModelInputRejectionTests(unittest.TestCase):
    def test_rejects_flat_native_or_flat_percentile_window(self):
        almost_flat = np.zeros((100, 100), dtype=np.uint16)
        almost_flat[0, 0] = 65535
        for source in (np.zeros((10, 10), dtype=np.uint16), almost_flat):
            with self.subTest(shape=source.shape, maximum=int(source.max())):
                with self.assertRaises((ValueError, TypeError)):
                    subject.transform_native(source, metadata())

    def test_rejects_wrong_array_shape(self):
        for source in (np.arange(10, dtype=np.uint16),
                       np.zeros((2, 3, 1), dtype=np.uint16),
                       np.empty((0, 2), dtype=np.uint16),
                       np.empty((2, 0), dtype=np.uint16)):
            with self.subTest(shape=source.shape):
                with self.assertRaises((ValueError, TypeError)):
                    subject.transform_native(source, metadata())

    def test_rejects_noninteger_native_pixels(self):
        for source in (ramp().astype(np.float32),
                       np.array([[0.0, math.nan], [1.0, 2.0]]),
                       np.array([[False, True], [True, False]]),
                       np.array([[0j, 1j], [2j, 3j]])):
            with self.subTest(dtype=str(source.dtype)):
                with self.assertRaises((ValueError, TypeError)):
                    subject.transform_native(source, metadata())

    def test_rejects_unsupported_photometric_encoding(self):
        for value in ("RGB", "PALETTE COLOR", "", None):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    subject.transform_native(ramp(), metadata(photometric=value))

    def test_rejects_invalid_pixel_spacing(self):
        for key in ("row_spacing_mm", "column_spacing_mm"):
            for value in (0, -1, math.nan, math.inf, -math.inf, None, "invalid"):
                with self.subTest(key=key, value=value):
                    with self.assertRaises((ValueError, TypeError, KeyError)):
                        subject.transform_native(ramp(), metadata(**{key: value}))

    def test_rejects_unknown_side_or_pretransformed_orientation(self):
        changes = ({"side": "UNKNOWN"}, {"side": "left"},
                   {"native_spatial_orientation": "mirrored_left_to_right"},
                   {"native_spatial_orientation": ""})
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    subject.transform_native(ramp(), metadata(**change))

    def test_rejects_missing_required_metadata(self):
        for key in metadata():
            details = metadata()
            del details[key]
            with self.subTest(missing=key):
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    subject.transform_native(ramp(), details)


if __name__ == "__main__":
    unittest.main()
