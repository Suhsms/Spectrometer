import json
import unittest

import numpy as np

from processing import (
    Calibration, compensate_sensitivity, correct_spectrum, detect_features,
    extract_spectrum, smooth_spectrum,
)


class ProcessingTests(unittest.TestCase):
    def test_sensitivity_compensation_normalized_at_550_nm(self):
        np.testing.assert_allclose(compensate_sensitivity([100], [550]), [100])

    def test_sensitivity_compensation_recovers_flat_relative_optical_power(self):
        wavelengths = np.array([450, 550, 700, 900])
        efficiency = np.array([60, 62, 41, 16])
        measured = 100 * efficiency * wavelengths / (62 * 550)
        np.testing.assert_allclose(compensate_sensitivity(measured, wavelengths), 100)

    def test_sensitivity_compensation_masks_unknown_and_weak_response(self):
        result = compensate_sensitivity([100] * 6, [350, 399, 400, 1000, 1050, 1200])
        self.assertTrue(np.isnan(result[[0, 1, 4, 5]]).all())
        self.assertTrue(np.isfinite(result[[2, 3]]).all())

    def test_sensitivity_compensation_preserves_gaps_and_rejects_mismatched_bins(self):
        result = compensate_sensitivity([100, np.nan], [550, 600])
        self.assertEqual(result[0], 100)
        self.assertTrue(np.isnan(result[1]))
        with self.assertRaises(ValueError):
            compensate_sensitivity([100, 100], [550])

    def setUp(self):
        self.calibration = Calibration("camera", 100, 20, ((10, 400), (90, 700)))

    def test_crop_uses_only_selected_pixels_and_keeps_full_frame_coordinates(self):
        frame = np.zeros((20, 100, 3), dtype=np.uint8)
        frame[5:10, 20:25] = 100
        pixels, intensity = extract_spectrum(frame, (20, 5, 5, 5))
        np.testing.assert_array_equal(pixels, np.arange(20, 25))
        np.testing.assert_allclose(intensity, 100)
        np.testing.assert_allclose(self.calibration.wavelengths(pixels)[0], 437.5)

    def test_piecewise_calibration_and_endpoint_extrapolation(self):
        calibration = Calibration("camera", 100, 20, ((10, 400), (50, 500), (90, 700)))
        np.testing.assert_allclose(calibration.wavelengths([0, 30, 70, 99]), [375, 450, 600, 745])

    def test_decreasing_wavelengths_are_supported(self):
        calibration = Calibration("camera", 100, 20, ((10, 700), (90, 400)))
        np.testing.assert_allclose(calibration.wavelengths([10, 50, 90]), [700, 550, 400])

    def test_invalid_calibration_rejected(self):
        for points in [((10, 400), (10, 700)), ((-1, 400), (90, 700)),
                       ((10, 400), (50, 700), (90, 600)), ((10, 400), (90, float("nan")))]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                Calibration("camera", 100, 20, points)

    def test_calibration_roundtrip_and_camera_compatibility(self):
        restored = Calibration.from_dict(json.loads(json.dumps(self.calibration.to_dict())))
        self.assertEqual(restored, self.calibration)
        self.assertTrue(restored.matches("camera", 100, 20))
        self.assertFalse(restored.matches("other", 100, 20))
        self.assertFalse(restored.matches("camera", 200, 20))

    def test_invalid_crop_rejected(self):
        with self.assertRaises(ValueError):
            extract_spectrum(np.zeros((20, 100, 3)), (99, 0, 2, 1))

    def test_dark_corrected_ratio_and_weak_reference_mask(self):
        result = correct_spectrum([60, 110, 10, 10], [110, 210, 10, 11], [10, 10, 10, 10])
        np.testing.assert_allclose(result[:2], [0.5, 0.5])
        self.assertTrue(np.isnan(result[2:]).all())

    def test_reference_is_not_rescaled_by_current_maximum(self):
        np.testing.assert_allclose(correct_spectrum([50, 25], [100, 100]), [0.5, 0.25])
        np.testing.assert_allclose(correct_spectrum([100, 50], [100, 100]), [1, 0.5])

    def test_mismatched_captures_rejected(self):
        with self.assertRaises(ValueError):
            correct_spectrum([1, 2], [1])
        with self.assertRaises(ValueError):
            correct_spectrum([1, 2], dark=[1])

    def test_peaks_and_dips_do_not_bridge_masked_regions(self):
        peaks, dips = detect_features([0, 4, 0, 2, 0, 3, 0, np.nan, 0, 5, 0], 10, 1)
        np.testing.assert_array_equal(peaks, [1, 3, 5, 9])
        np.testing.assert_array_equal(dips, [2, 4])

    def test_smoothing_preserves_mask_and_handles_short_crop(self):
        values = [0, 1, 2, np.nan, 3, 4]
        np.testing.assert_allclose(smooth_spectrum(values, 11), values, equal_nan=True)
        self.assertEqual(len(smooth_spectrum(np.arange(20.0), 10)), 20)


if __name__ == "__main__":
    unittest.main()