# Simple Spectrometer

A standalone, live-camera-only desktop spectrometer for Windows. The original
Theremino sources are unchanged. No video-file playback, serial sensors,
automation, irradiance correction, or network service is included.

## Run

Python 3.11 or newer is recommended. Development was tested with Python 3.14.
From this directory, run:

```powershell
.\run.ps1
```

The launcher creates a local `.venv` and installs dependencies on first use.
After updating the app, install any added dependencies with
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
Create a separate environment on each computer; do not copy `.venv` between PCs.
If PowerShell script execution is restricted, use these commands directly:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The application does not connect to a camera automatically. Choose a camera and
resolution, then select **Connect**. Camera access may require enabling Windows
Settings > Privacy & security > Camera > Let desktop apps access your camera.

## IDS and DirectShow Cameras

If your camera works with `cv2.VideoCapture(1, cv2.CAP_DSHOW)` but is not listed by
Qt, select **OpenCV / DirectShow** in the **Capture** tab. Choose **Camera index 1**
in the top camera selector, then **Connect**. The constant is `cv2.CAP_DSHOW`.
Index 1 is the second DirectShow camera; index 0 is the first.

This mode lists manual indices 0-15, not detected camera names. It remains
available even when Qt detects no cameras. Use the same index as your working
script. Close that script, IDS peak, and any other camera application before
connecting if the driver permits only one application at a time.

OpenCV is installed through the requirements above. The IDS/DirectShow driver
must also be installed on the target computer; installing OpenCV does not install
the IDS camera driver. Availability depends on the camera's driver exposing a
DirectShow interface.

DirectShow mode uses the driver's default resolution, frame rate, and exposure
settings. Qt exposure and format controls are disabled in this mode. Frames are
read in a background thread; pause freezes the measurement while acquisition
continues. Disconnect requests a clean stop. If a driver read is still blocked,
the app keeps the window open rather than destroying a running capture thread;
retry Disconnect after the exposure/driver operation finishes.

DirectShow calibration identity is index-based, not a physical camera serial
number. Indices can change after reconnecting cameras or changing computers.
Verify the camera, frame resolution, and optics before reusing a calibration.
CSV metadata includes the selected backend and DirectShow index; exposure state
is recorded as unknown in DirectShow mode.

## Capture and Crop

The upper view is the live camera preview. Drag or resize the green rectangle,
or enter its pixel coordinates in **Crop**. Only this rectangle contributes to
the spectrum. The initial crop is a narrow horizontal strip across the middle.
Align the dispersed spectrum horizontally before measuring.

Each spectrum bin is the vertical mean of RGB-weighted camera brightness:
`0.299 R + 0.587 G + 0.114 B`. Preview processing is limited to 15 updates per
second. The pause button freezes the preview and measurement; the camera stays
connected. Disconnect releases the camera and clears the displayed measurement.

## Wavelength Calibration

1. Illuminate the instrument with a source containing known spectral lines.
2. Select a pixel cell in the calibration table and click its line on the graph.
3. Enter that line's known wavelength in nanometers in the adjacent cell.
4. Repeat for at least one other line, then select **Apply**.
5. Use **Save calibration** to retain the mapping and crop as a JSON file.

Calibration uses piecewise linear interpolation and linear extrapolation beyond
the outer points. Both increasing and decreasing wavelength directions work.
Prefer widely separated known lines; do not treat extrapolated wavelengths as
equally trustworthy. The axis remains in pixels until calibration is applied.

Pixel positions are always full-frame coordinates, so moving the crop does not
shift wavelengths. Calibration files are checked against camera identity and
frame dimensions. Recalibrate after moving the camera, slit, or grating even
when those checks still match. This app does not import Theremino calibration
files, whose coordinates and format differ.

## Reference and Dark

Capture dark with the light blocked. Then illuminate the instrument without a
sample and select **Capture reference**. Insert the sample to measure a
sample/reference ratio. Uncheck reference correction to return to relative
intensity; dark subtraction stays active until **Clear captures** is selected.

With a dark capture, the ratio is `(sample - dark) / (reference - dark)`.
Reference-corrected values are ratios, not absolute optical power. Weak reference
bins (at or below 1% of the maximum dark-corrected reference, or 0.000001 camera
units) are masked rather than divided. They appear as graph gaps and CSV `nan`.

This intentionally differs from Theremino's reference display, which also scales
the ratio by the current signal maximum. No reference overlay is implemented.

Keep exposure, gain, lighting, crop, and optics fixed between captures. Manual
exposure is offered only when Qt reports driver support; gain and white balance
may still be controlled automatically by the camera. Reference capture warns
when automatic exposure is enabled. Crop, resolution, camera, or app exposure
changes clear reference and dark captures. Captures are held in memory only.

## Camera Sensitivity Compensation

In **Analysis**, enable **Compensate sensitivity** after applying or loading a
wavelength calibration. It is off by default. Turning it off restores the
uncompensated spectrum while retaining any dark subtraction and smoothing.

The built-in profile is only for the **IDS UI-3242LE-M / EV76C560ABT monochrome
sensor**. Its quantum-efficiency values were estimated visually from the graph
on page 1 of the supplied datasheet; they are not manufacturer-provided numerical
calibration data. Do not apply this profile to another camera model.

Quantum efficiency is electrons per incident photon, not response per unit
optical power. For a relative optical-power display, the app compensates for
both QE and photon energy, normalized to unity at 550 nm:

$$
I_{\mathrm{corrected}}(\lambda) = I_{\mathrm{dark-subtracted}}(\lambda)
\frac{QE(550)\,550}{QE(\lambda)\,\lambda}
$$

Correction is performed before smoothing. The graph identifies corrected
intensity, and CSV metadata records the profile points, formula, normalization,
and whether correction was requested and actually applied. Raw values are kept.

The estimated profile spans 400-1100 nm. Wavelengths outside it, or with QE below
5% of the profile's peak QE, are masked instead of extrapolated or excessively
amplified. These bins appear as gaps and CSV `nan` values. At the long-wavelength
end, this threshold masks part of the nominal profile range.

Sample/reference ratios already cancel a stable sensor response. Compensation is
therefore automatically bypassed while reference correction is active, and
resumes when reference mode is turned off if the checkbox is still selected.
Disconnecting a camera resets the checkbox to off.

This is an approximate sensor-only correction, not radiometric calibration or a
correction for grating efficiency, lenses, filters, slit geometry, or wavelength
bin width. It assumes linear, unsaturated camera values with fixed gain and
exposure. The current preview path converts frames to 8-bit RGB, so it does not
preserve the camera's full native ADC precision. Quantitative measurements need
accurate response data and ideally a measured response for the complete instrument.

## Peaks, Dips, and Export

Peaks and dips can be enabled separately. **Prominence** rejects small features
relative to the displayed signal range; **Separation** is measured in pixels.
**Smoothing** controls a Savitzky-Golay window; values below 5 disable it. Up to
12 wavelength labels in nanometers are shown for each feature type after
calibration, with markers for all detected features. Before calibration, markers
remain available but numeric peak/dip labels are hidden. These are local extrema,
not automatic element identification.

After calibration, the area between the curve and zero is filled with a continuous
illustrative visible-spectrum gradient from approximately 380 to 780 nm. Outside
that range, the fill is dark gray. Violet fades in gradually over 380-410 nm,
and the red end fades to gray over 700-780 nm, without sharp boundary jumps.
Without calibration, the entire fill stays gray
because camera pixels alone do not determine wavelengths. The colors follow the
wavelength axis, not the crop width, and are not measured camera RGB values.

**Save CSV** exports the displayed measurement, including unsmoothed raw values,
reference/dark values when active, full-frame pixels, and calibrated wavelengths.
The first line starts with `# ` and contains JSON measurement metadata; the next
line is the CSV header. Wavelength cells are blank before calibration. The
`intensity` column includes enabled corrections and smoothing. Use
`numpy.genfromtxt(..., delimiter=',', names=True, skip_header=1)` to read it.
**Save graph** exports a PNG. Pausing first retains a specific measurement.

Camera brightness is not radiometrically calibrated. Gamma, compression, sensor
response, clipping, and automatic camera adjustments limit quantitative accuracy.
The near-clipping readout flags ROI pixels with any channel at least 250/255.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Processing and offscreen desktop tests use synthetic frames, exercise file
exports, and do not open a real camera. An offscreen Qt font/backend warning
does not indicate a failed test.

## Theremino Study Notes

The design was informed by the local V5.0 source and bundled documentation:

- `Module_DirectShow_VideoInput.vb`: camera selection/capture in the original.
- `Module_Spectrometer.vb`: vertical RGB-weighted averaging, reference division,
  background subtraction, and extrema marking.
- `Module_Calibrations.vb`: multipoint piecewise linear wavelength mapping.
- `Module_SaveLoad.vb`: separate spectrum and calibration persistence.
- `Annotations.txt`: processing-stage notes; `Language_ENG.txt`: UI terminology.

The local `Docs` folder contains translations and linear-sensor datasheets, not
a complete application manual. This implementation uses Qt Multimedia or
OpenCV/DirectShow instead of porting Theremino's DirectShow code, and SciPy for
filtering/extrema detection.