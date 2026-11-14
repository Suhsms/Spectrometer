from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter


SENSITIVITY_PROFILE_NAME = "IDS UI-3242LE-M (approx. QE)"
IDS_QE_PROFILE = np.array([
    (400, 55), (425, 58), (450, 60), (475, 61), (500, 61),
    (525, 63), (550, 62), (575, 58), (600, 58), (625, 56),
    (650, 49), (675, 46), (700, 41), (725, 37), (750, 35),
    (775, 30), (800, 27), (825, 24), (850, 21), (875, 18),
    (900, 16), (925, 12), (950, 9), (975, 7), (1000, 5),
    (1025, 3), (1050, 1.5), (1075, 0.5), (1100, 0.2),
], dtype=float)


def compensate_sensitivity(intensity, wavelengths):
    intensity = np.asarray(intensity, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if intensity.shape != wavelengths.shape:
        raise ValueError("Sensitivity correction requires a wavelength for every intensity bin.")
    profile_wavelengths, quantum_efficiency = IDS_QE_PROFILE.T
    efficiency = np.interp(wavelengths, profile_wavelengths, quantum_efficiency,
                          left=np.nan, right=np.nan)
    reference_efficiency = np.interp(550, profile_wavelengths, quantum_efficiency)
    relative_response = efficiency * wavelengths / (reference_efficiency * 550)
    valid = np.isfinite(relative_response) & (efficiency >= quantum_efficiency.max() * 0.05)
    result = np.full_like(intensity, np.nan)
    np.divide(intensity, relative_response, out=result, where=valid)
    return result


@dataclass(frozen=True)
class Calibration:
    camera_id: str
    width: int
    height: int
    points: tuple[tuple[float, float], ...]

    def __post_init__(self):
        if not self.camera_id or self.width < 2 or self.height < 1:
            raise ValueError("Calibration requires a camera and valid frame dimensions.")
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("Enter at least two pixel/wavelength pairs.")
        if not np.isfinite(points).all():
            raise ValueError("Calibration values must be finite numbers.")
        pixels, wavelengths = points.T
        if np.any(np.diff(pixels) <= 0):
            raise ValueError("Pixel positions must be distinct and in increasing order.")
        if pixels[0] < 0 or pixels[-1] > self.width - 1:
            raise ValueError("Calibration pixels must be inside the full camera frame.")
        differences = np.diff(wavelengths)
        if np.any(wavelengths <= 0) or not (
            np.all(differences > 0) or np.all(differences < 0)
        ):
            raise ValueError("Wavelengths must be positive and strictly increasing or decreasing.")

    def wavelengths(self, pixels):
        pixels = np.asarray(pixels, dtype=float)
        anchors, wavelengths = np.asarray(self.points, dtype=float).T
        values = np.interp(pixels, anchors, wavelengths)
        left = wavelengths[0] + (pixels - anchors[0]) * (
            (wavelengths[1] - wavelengths[0]) / (anchors[1] - anchors[0])
        )
        right = wavelengths[-1] + (pixels - anchors[-1]) * (
            (wavelengths[-1] - wavelengths[-2]) / (anchors[-1] - anchors[-2])
        )
        return np.where(pixels < anchors[0], left, np.where(pixels > anchors[-1], right, values))

    def matches(self, camera_id, width, height):
        return (self.camera_id, self.width, self.height) == (camera_id, width, height)

    def to_dict(self):
        return {
            "version": 1,
            "camera_id": self.camera_id,
            "width": self.width,
            "height": self.height,
            "points": [list(point) for point in self.points],
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != 1:
            raise ValueError("Unsupported calibration file version.")
        return cls(
            str(data["camera_id"]),
            int(data["width"]),
            int(data["height"]),
            tuple(sorted((float(pixel), float(nm)) for pixel, nm in data["points"])),
        )


def extract_spectrum(frame, roi):
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an RGB camera frame.")
    left, top, width, height = (int(value) for value in roi)
    if (
        left < 0 or top < 0 or width < 1 or height < 1
        or left + width > frame.shape[1] or top + height > frame.shape[0]
    ):
        raise ValueError("Crop must be inside the camera frame and at least one pixel wide/high.")
    cropped = frame[top:top + height, left:left + width]
    intensity = cropped.mean(axis=0) @ np.array([0.299, 0.587, 0.114])
    return np.arange(left, left + width, dtype=float), intensity


def correct_spectrum(sample, reference=None, dark=None):
    sample = np.asarray(sample, dtype=float)
    baseline = np.zeros_like(sample) if dark is None else np.asarray(dark, dtype=float)
    if baseline.shape != sample.shape:
        raise ValueError("Dark capture does not match the current crop.")
    corrected = sample - baseline
    if reference is None:
        return np.maximum(corrected, 0)
    reference = np.asarray(reference, dtype=float)
    if reference.shape != sample.shape:
        raise ValueError("Reference capture does not match the current crop.")
    denominator = reference - baseline
    threshold = max(float(np.max(denominator)) * 0.01, 1e-6)
    valid = denominator > threshold
    result = np.full_like(sample, np.nan)
    np.divide(corrected, denominator, out=result, where=valid)
    return result


def finite_segments(values):
    indices = np.flatnonzero(np.isfinite(values))
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


def smooth_spectrum(values, window):
    result = np.asarray(values, dtype=float).copy()
    for segment in finite_segments(result):
        size = min(int(window), len(segment))
        if size % 2 == 0:
            size -= 1
        if size >= 5:
            result[segment] = savgol_filter(result[segment], size, 2)
    return result


def detect_features(values, prominence_percent=5, distance=10):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 3 or np.ptp(finite) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    prominence = float(np.ptp(finite)) * prominence_percent / 100
    peaks, dips = [], []
    for segment in finite_segments(values):
        if len(segment) < 3:
            continue
        maxima, _ = find_peaks(values[segment], prominence=prominence, distance=max(1, distance))
        minima, _ = find_peaks(-values[segment], prominence=prominence, distance=max(1, distance))
        peaks.extend(segment[maxima])
        dips.extend(segment[minima])
    return np.asarray(peaks, dtype=int), np.asarray(dips, dtype=int)