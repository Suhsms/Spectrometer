import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtMultimedia import QVideoFrame
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from app import Spectrometer, spectrum_fill_brush
from processing import compensate_sensitivity


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = Spectrometer()
        self.window.camera_id = "test-camera"
        self.frame = np.zeros((60, 120, 3), dtype=np.uint8)
        self.frame[:, 30:35] = 180
        self.frame[:, 80:85] = 240
        self.window.set_frame(self.frame)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()

    def calibrate(self):
        for row, point in enumerate(((30, 436), (80, 546))):
            for column, value in enumerate(point):
                self.window.calibration_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.window.apply_calibration()

    def test_sensitivity_toggle_requires_calibration_and_restores_original_when_off(self):
        self.assertFalse(self.window.sensitivity_toggle.isChecked())
        self.assertFalse(self.window.sensitivity_toggle.isEnabled())
        self.calibrate()
        self.assertTrue(self.window.sensitivity_toggle.isEnabled())
        original = self.window.latest[3].copy()
        self.window.sensitivity_toggle.setChecked(True)
        np.testing.assert_allclose(
            self.window.latest[3],
            compensate_sensitivity(original, self.window.latest[1]), equal_nan=True,
        )
        self.assertTrue(self.window.sensitivity_active())
        self.window.sensitivity_toggle.setChecked(False)
        np.testing.assert_allclose(self.window.latest[3], original)

    def test_wavelength_axis_keeps_nanometer_units_for_near_infrared(self):
        for row, point in enumerate(((0, 400), (119, 1100))):
            for column, value in enumerate(point):
                self.window.calibration_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.window.apply_calibration()
        self.assertFalse(self.window.plot.getAxis("bottom").autoSIPrefix)
        self.assertEqual(self.window.plot.getAxis("bottom").labelUnits, "nm")

    def test_sensitivity_is_bypassed_for_reference_ratio(self):
        self.calibrate()
        self.window.sensitivity_toggle.setChecked(True)
        corrected = self.window.latest[3].copy()
        self.window.exposure_auto.setChecked(False)
        self.window.capture_reference()
        self.assertFalse(self.window.sensitivity_active())
        self.assertFalse(self.window.sensitivity_toggle.isEnabled())
        self.assertTrue(self.window.sensitivity_toggle.isChecked())
        finite = self.window.latest[3][np.isfinite(self.window.latest[3])]
        np.testing.assert_allclose(finite, 1)
        self.window.use_reference.setChecked(False)
        self.assertTrue(self.window.sensitivity_toggle.isEnabled())
        np.testing.assert_allclose(self.window.latest[3], corrected, equal_nan=True)

    def test_sensitivity_is_unavailable_without_calibration_and_resets_on_disconnect(self):
        self.calibrate()
        self.window.sensitivity_toggle.setChecked(True)
        self.window.clear_calibration()
        self.assertFalse(self.window.sensitivity_active())
        self.assertFalse(self.window.sensitivity_toggle.isEnabled())
        np.testing.assert_allclose(self.window.latest[3], self.window.latest[2])
        self.window.disconnect_camera()
        self.assertFalse(self.window.sensitivity_toggle.isChecked())

    def test_preview_calibration_and_crop_preserve_wavelengths(self):
        self.calibrate()
        self.window.set_crop(20, 10, 80, 10)
        pixels, wavelengths, raw, values = self.window.latest
        self.assertEqual(pixels[10], 30)
        self.assertEqual(wavelengths[10], 436)
        self.assertTrue(self.window.csv_button.isEnabled())

    def test_feature_labels_use_calibrated_wavelengths_not_pixels(self):
        self.assertEqual(self.window.peak_labels, [])
        self.calibrate()
        labels = [label.textItem.toPlainText() for label in self.window.peak_labels]
        self.assertIn("440.4 nm", labels)
        self.assertIn("550.4 nm", labels)
        self.window.set_crop(20, 10, 80, 10)
        cropped_labels = [label.textItem.toPlainText() for label in self.window.peak_labels]
        self.assertEqual(set(cropped_labels), set(labels))
        self.window.clear_calibration()
        self.assertEqual(self.window.peak_labels, [])

    def test_spectrum_fill_colors_follow_wavelength_and_are_gray_outside_visible(self):
        image = QImage(1000, 10, QImage.Format_RGB32)
        painter = QPainter(image)
        painter.fillRect(image.rect(), spectrum_fill_brush())
        painter.end()
        self.assertEqual(image.pixelColor(350, 5), QColor("#55575e"))
        self.assertEqual(image.pixelColor(800, 5), QColor("#55575e"))
        blue, green, red = [image.pixelColor(wavelength, 5) for wavelength in (450, 530, 650)]
        self.assertGreater(blue.blue(), blue.red())
        self.assertGreater(green.green(), green.red())
        self.assertGreater(green.green(), green.blue())
        self.assertGreater(red.red(), red.green())
        self.assertGreater(red.red(), red.blue())

    def test_spectrum_fill_fades_gradually_at_visible_edges(self):
        image = QImage(1000, 10, QImage.Format_RGB32)
        painter = QPainter(image)
        painter.fillRect(image.rect(), spectrum_fill_brush())
        painter.end()
        colors = np.array([image.pixelColor(wavelength, 5).getRgb()[:3]
                           for wavelength in range(1000)], dtype=float)
        for boundary in (380, 780):
            differences = np.abs(np.diff(colors[boundary - 2:boundary + 3], axis=0))
            self.assertLessEqual(differences.max(), 8)
        gray = np.array([85, 87, 94])
        self.assertLess(np.linalg.norm(colors[385] - gray), np.linalg.norm(colors[405] - gray))
        self.assertLess(np.linalg.norm(colors[770] - gray), np.linalg.norm(colors[740] - gray))

    def test_fill_uses_wavelength_gradient_only_after_calibration(self):
        self.assertIsNone(self.window.curve.opts["fillBrush"].gradient())
        self.calibrate()
        self.assertIsNotNone(self.window.curve.opts["fillBrush"].gradient())
        self.window.clear_calibration()
        self.assertIsNone(self.window.curve.opts["fillBrush"].gradient())

    def test_reference_dark_capture_and_crop_invalidation(self):
        self.window.exposure_auto.setChecked(False)
        self.window.capture_reference()
        self.assertTrue(self.window.use_reference.isChecked())
        self.window.capture_dark()
        self.assertIsNotNone(self.window.dark)
        self.window.set_crop(0, 0, 60, 10)
        self.assertIsNone(self.window.reference)
        self.assertIsNone(self.window.dark)
        self.assertFalse(self.window.use_reference.isChecked())

    def test_camera_or_resolution_change_clears_calibration(self):
        self.calibrate()
        self.window.set_frame(np.zeros((30, 60, 3), dtype=np.uint8))
        self.assertIsNone(self.window.calibration)
        self.assertEqual(len(self.window.latest[0]), 60)

    def test_export_and_calibration_roundtrip(self):
        self.calibrate()
        with tempfile.TemporaryDirectory() as folder:
            calibration = str(Path(folder) / "calibration.json")
            spectrum = str(Path(folder) / "spectrum.csv")
            image = str(Path(folder) / "spectrum.png")
            with patch("app.QFileDialog.getSaveFileName", return_value=(calibration, "")):
                self.window.save_calibration()
            self.window.clear_calibration()
            with patch("app.QFileDialog.getOpenFileName", return_value=(calibration, "")):
                self.window.load_calibration()
            self.assertIsNotNone(self.window.calibration)
            self.window.sensitivity_toggle.setChecked(True)
            with patch("app.QFileDialog.getSaveFileName", return_value=(spectrum, "")):
                self.window.save_csv()
            with Path(spectrum).open(encoding="utf-8", newline="") as output:
                metadata = json.loads(output.readline()[2:])
                rows = list(csv.DictReader(output))
            self.assertEqual(metadata["camera_id"], "test-camera")
            self.assertTrue(metadata["sensitivity_compensation_applied"])
            self.assertTrue(metadata["sensitivity_profile"]["approximate"])
            self.assertEqual(metadata["sensitivity_profile"]["normalization_nm"], 550)
            self.assertEqual(len(rows), 120)
            self.assertEqual(float(rows[30]["wavelength_nm"]), 436)
            with patch("app.QFileDialog.getSaveFileName", return_value=(image, "")):
                self.window.save_graph()
            self.assertGreater(Path(image).stat().st_size, 1000)

    def test_mismatched_calibration_file_rejected(self):
        self.calibrate()
        data = self.window.calibration.to_dict()
        data["camera_id"] = "different-camera"
        self.window.clear_calibration()
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "calibration.json"
            filename.write_text(json.dumps(data), encoding="utf-8")
            with patch("app.QFileDialog.getOpenFileName", return_value=(str(filename), "")), \
                    patch.object(self.window, "report_error") as report:
                self.window.load_calibration()
            report.assert_called_once()
            self.assertIsNone(self.window.calibration)

    def test_video_frame_rgb_conversion_and_pause(self):
        image = QImage(self.frame.data, 120, 60, self.frame.strides[0], QImage.Format_RGB888)
        video_frame = QVideoFrame(image)
        self.window.camera = object()
        self.window.last_frame_time = 0
        self.window.receive_frame(video_frame)
        np.testing.assert_array_equal(self.window.frame, self.frame)
        previous = self.window.frame
        self.window.pause_button.setChecked(True)
        self.window.last_frame_time = 0
        self.window.receive_frame(video_frame)
        self.assertIs(self.window.frame, previous)
        self.window.camera = None

    def test_disconnect_clears_stale_measurement(self):
        self.window.disconnect_camera()
        self.assertIsNone(self.window.latest)
        self.assertFalse(self.window.csv_button.isEnabled())

    def test_active_camera_updates_status_and_exposure_capability(self):
        camera = Mock()
        camera.isExposureModeSupported.return_value = False
        camera.minimumExposureTime.return_value = 0
        camera.maximumExposureTime.return_value = 0
        self.window.camera = camera
        self.window.camera_active_changed(True)
        self.assertEqual(self.window.statusBar().currentMessage(), "Live camera")
        self.assertFalse(self.window.exposure_auto.isEnabled())
        self.assertFalse(self.window.exposure.isEnabled())
        self.window.camera = None


if __name__ == "__main__":
    unittest.main()