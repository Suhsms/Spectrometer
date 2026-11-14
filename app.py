import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtCore import QSignalBlocker, QThread, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QGradient, QIcon, QImage, QLinearGradient, QPainter, QPalette
from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QStyle, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from processing import (
    IDS_QE_PROFILE, SENSITIVITY_PROFILE_NAME, Calibration, compensate_sensitivity,
    correct_spectrum, detect_features, extract_spectrum, smooth_spectrum,
)


pg.setConfigOptions(antialias=True, background="#24262a", foreground="#d1d5dc", imageAxisOrder="row-major")


def spectrum_fill_brush():
    gradient = QLinearGradient(380, 0, 780, 0)
    gradient.setSpread(QGradient.PadSpread)
    for position, color in (
        (0, "#55575e"),
        (0.075, "#7900ba"),
        (0.1, "#6600ff"),
        (0.175, "#003bff"),
        (0.275, "#00dfff"),
        (0.325, "#00ee44"),
        (0.425, "#aaff00"),
        (0.5, "#ffff00"),
        (0.55, "#ffac00"),
        (0.6, "#ff5000"),
        (0.6625, "#ff1600"),
        (0.8, "#dc0000"),
        (0.9, "#8b3439"),
        (1, "#55575e"),
    ):
        gradient.setColorAt(position, QColor(color))
    return QBrush(gradient)


class DirectShowCapture(QThread):
    frame_ready = Signal(object)
    opened = Signal()
    failed = Signal(str)

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index

    def run(self):
        capture = None
        try:
            import cv2

            capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                self.failed.emit(
                    f"Cannot open DirectShow camera index {self.index}. "
                    "Check the index, driver, and whether another application is using it."
                )
                return
            self.opened.emit()
            last_emission = 0
            while not self.isInterruptionRequested():
                success, frame = capture.read()
                if self.isInterruptionRequested():
                    break
                if not success or frame is None:
                    self.failed.emit(f"No frame received from DirectShow camera {self.index}.")
                    break
                now = time.monotonic()
                if now - last_emission < 1 / 15:
                    continue
                last_emission = now
                if frame.ndim == 2:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.ndim == 3 and frame.shape[2] == 3:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif frame.ndim == 3 and frame.shape[2] == 4:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                else:
                    raise ValueError("Unsupported DirectShow frame format.")
                if rgb.dtype != np.uint8:
                    raise ValueError("DirectShow capture requires 8-bit preview frames.")
                self.frame_ready.emit(rgb.copy())
        except ImportError:
            self.failed.emit("OpenCV is unavailable. Run: python -m pip install -r requirements.txt")
        except Exception as error:
            self.failed.emit(f"DirectShow capture error: {error}")
        finally:
            if capture is not None:
                capture.release()


class Spectrometer(QMainWindow):
    def __init__(self):
        super().__init__()
        application = QApplication.instance()
        application.setStyle("Fusion")
        palette = application.palette()
        for role, color in (
            (QPalette.Window, "#2b2d31"), (QPalette.WindowText, "#e6e8eb"),
            (QPalette.Base, "#35373d"), (QPalette.AlternateBase, "#303238"),
            (QPalette.Text, "#e6e8eb"), (QPalette.Button, "#404249"),
            (QPalette.ButtonText, "#e6e8eb"), (QPalette.Highlight, "#7bd1b5"),
            (QPalette.HighlightedText, "#24262a"), (QPalette.ToolTipBase, "#404249"),
            (QPalette.ToolTipText, "#e6e8eb"), (QPalette.Light, "#62656e"),
            (QPalette.Midlight, "#4a4d55"), (QPalette.Mid, "#42454d"),
            (QPalette.Dark, "#202226"), (QPalette.Shadow, "#1c1e22"),
        ):
            palette.setColor(role, QColor(color))
        for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
            palette.setColor(QPalette.Disabled, role, QColor("#8c909b"))
        application.setPalette(palette)
        self.setWindowTitle("Simple Spectrometer")
        self.resize(1240, 850)
        self.setMinimumSize(820, 650)
        self.camera = None
        self.directshow = None
        self.camera_id = ""
        self.frame = None
        self.frame_size = None
        self.calibration = None
        self.reference = None
        self.dark = None
        self.last_frame_time = 0
        self.latest = None
        self.peak_labels = []
        self.capture_time = ""
        self.media_devices = QMediaDevices(self)
        self.session = QMediaCaptureSession(self)
        self.sink = QVideoSink(self)
        self.session.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(self.receive_frame)
        self.build_ui()
        self.media_devices.videoInputsChanged.connect(self.refresh_devices)
        self.refresh_devices()

    def tool_button(self, icon, tooltip, callback, checkable=False):
        button = QToolButton()
        pixmap = self.style().standardIcon(icon).pixmap(18, 18)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), QColor("#e6e8eb"))
        painter.end()
        button.setIcon(QIcon(pixmap))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setCheckable(checkable)
        button.clicked.connect(callback)
        return button

    def command(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def heading(self, text, layout):
        label = QLabel(text)
        label.setObjectName("sectionHeading")
        layout.addWidget(label)

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 12)
        toolbar = QHBoxLayout()
        title = QLabel("Simple Spectrometer")
        title.setObjectName("appTitle")
        toolbar.addWidget(title)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("Camera"))
        self.devices = QComboBox()
        self.devices.setMinimumWidth(200)
        self.devices.setMaximumWidth(360)
        self.devices.currentIndexChanged.connect(self.device_changed)
        toolbar.addWidget(self.devices)
        toolbar.addWidget(self.tool_button(QStyle.SP_BrowserReload, "Refresh cameras", self.refresh_devices))
        self.connect_button = self.command("Connect", self.toggle_camera)
        toolbar.addWidget(self.connect_button)
        self.pause_button = self.tool_button(QStyle.SP_MediaPause, "Pause/resume measurement", self.pause_changed, True)
        self.pause_button.setEnabled(False)
        toolbar.addWidget(self.pause_button)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)
        workspace = QWidget()
        views = QVBoxLayout(workspace)
        views.setContentsMargins(0, 0, 12, 0)
        self.preview = pg.GraphicsLayoutWidget()
        self.preview.setBackground("#25272b")
        self.preview.setMinimumHeight(190)
        self.preview_view = self.preview.addViewBox(lockAspect=True, enableMenu=False)
        self.preview_view.invertY(True)
        self.preview_view.setMouseEnabled(x=False, y=False)
        self.image = pg.ImageItem()
        self.preview_view.addItem(self.image)
        self.roi = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("#41deac", width=2), rotatable=False)
        self.roi.addScaleHandle([1, 0], [0, 1])
        self.roi.addScaleHandle([0, 1], [1, 0])
        self.roi.setVisible(False)
        self.preview_view.addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self.roi_changed)
        views.addWidget(self.preview, 2)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(240)
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.setLabel("bottom", "Pixel position")
        self.plot.setLabel("left", "Relative intensity", units="a.u.")
        self.plot.showGrid(x=True, y=True, alpha=0.16)
        self.plot.plotItem.vb.setDefaultPadding(0.1)
        self.spectrum_brush = spectrum_fill_brush()
        self.uncalibrated_brush = pg.mkBrush("#55575e")
        self.curve = self.plot.plot(
            pen=pg.mkPen("#d1d5dc", width=1.2), connect="finite",
            fillLevel=0, fillBrush=self.uncalibrated_brush,
        )
        self.peak_markers = self.plot.plot(pen=None, symbol="t", symbolBrush="#ff8296", symbolSize=9)
        self.dip_markers = self.plot.plot(pen=None, symbol="t1", symbolBrush="#87b8ff", symbolSize=9)
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#86938f", style=Qt.DashLine))
        self.plot.addItem(self.cursor, ignoreBounds=True)
        self.cursor.hide()
        self.plot.scene().sigMouseMoved.connect(self.cursor_moved)
        self.plot.scene().sigMouseClicked.connect(self.graph_clicked)
        views.addWidget(self.plot, 3)
        self.readout = QLabel("No measurement")
        self.readout.setMinimumHeight(22)
        self.readout.setWordWrap(True)
        views.addWidget(self.readout)
        splitter.addWidget(workspace)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(290)
        scroll.setMaximumWidth(370)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        self.control_tabs = QTabWidget()
        panel_layout.addWidget(self.control_tabs, 1)
        capture_tab = QWidget()
        self.control_tabs.addTab(capture_tab, "Capture")
        controls = QVBoxLayout(capture_tab)
        controls.setSpacing(8)
        self.capture_backend = QComboBox()
        self.capture_backend.addItem("Qt Multimedia", "qt")
        self.capture_backend.addItem("OpenCV / DirectShow", "dshow")
        self.capture_backend.setToolTip("Select OpenCV / DirectShow for cameras accessible via cv2.CAP_DSHOW")
        self.capture_backend.currentIndexChanged.connect(self.capture_backend_changed)
        controls.addWidget(self.capture_backend)
        self.format_select = QComboBox()
        self.format_select.setToolTip("Camera resolution and frame rate")
        controls.addWidget(self.format_select)
        self.exposure_auto = QCheckBox("Automatic exposure")
        self.exposure_auto.setChecked(True)
        self.exposure_auto.toggled.connect(self.set_auto_exposure)
        controls.addWidget(self.exposure_auto)
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(0.1, 1000)
        self.exposure.setSuffix(" ms")
        self.exposure.setValue(20)
        self.exposure.setEnabled(False)
        self.exposure.valueChanged.connect(self.set_exposure)
        exposure_form = QFormLayout()
        exposure_form.addRow("Exposure", self.exposure)
        controls.addLayout(exposure_form)
        self.exposure_note = QLabel("Manual exposure: driver-dependent")
        self.exposure_note.setWordWrap(True)
        controls.addWidget(self.exposure_note)

        self.heading("Crop", controls)
        crop_form = QFormLayout()
        self.crop_inputs = []
        for name in ("Left", "Top", "Width", "Height"):
            control = QSpinBox()
            control.setRange(0 if name in ("Left", "Top") else 1, 10000)
            control.setEnabled(False)
            control.valueChanged.connect(self.crop_inputs_changed)
            crop_form.addRow(name, control)
            self.crop_inputs.append(control)
        controls.addLayout(crop_form)
        self.reset_crop_button = self.command("Reset crop", self.reset_crop)
        self.reset_crop_button.setEnabled(False)
        controls.addWidget(self.reset_crop_button)
        controls.addStretch()

        calibration_tab = QWidget()
        self.control_tabs.addTab(calibration_tab, "Calibration")
        controls = QVBoxLayout(calibration_tab)
        self.heading("Wavelength Calibration", controls)
        self.calibration_status = QLabel("Uncalibrated: pixel axis")
        self.calibration_status.setWordWrap(True)
        controls.addWidget(self.calibration_status)
        self.calibration_table = QTableWidget(2, 2)
        self.calibration_table.setHorizontalHeaderLabels(["Full-frame pixel", "Wavelength (nm)"])
        self.calibration_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.calibration_table.setMaximumHeight(145)
        self.calibration_table.setToolTip("Select a pixel cell, then click a known spectral line on the graph.")
        self.calibration_table.setCurrentCell(0, 0)
        controls.addWidget(self.calibration_table)
        points_bar = QHBoxLayout()
        points_bar.addWidget(self.tool_button(QStyle.SP_FileDialogNewFolder, "Add calibration point", self.add_point))
        points_bar.addWidget(self.tool_button(QStyle.SP_TrashIcon, "Remove selected calibration point", self.remove_point))
        self.apply_calibration_button = self.command("Apply", self.apply_calibration)
        points_bar.addWidget(self.apply_calibration_button)
        points_bar.addWidget(self.command("Clear", self.clear_calibration))
        controls.addLayout(points_bar)
        calibration_files = QHBoxLayout()
        self.load_calibration_button = self.command("Load calibration", self.load_calibration)
        self.save_calibration_button = self.command("Save calibration", self.save_calibration)
        calibration_files.addWidget(self.load_calibration_button)
        calibration_files.addWidget(self.save_calibration_button)
        controls.addLayout(calibration_files)
        controls.addStretch()

        analysis_tab = QWidget()
        self.control_tabs.addTab(analysis_tab, "Analysis")
        controls = QVBoxLayout(analysis_tab)
        self.heading("Reference", controls)
        self.reference_button = self.command("Capture reference", self.capture_reference)
        controls.addWidget(self.reference_button)
        self.use_reference = QCheckBox("Reference correction (sample/reference)")
        self.use_reference.toggled.connect(self.process_frame)
        controls.addWidget(self.use_reference)
        dark_bar = QHBoxLayout()
        self.dark_button = self.command("Capture dark", self.capture_dark)
        dark_bar.addWidget(self.dark_button)
        self.clear_captures_button = self.command("Clear captures", self.clear_captures)
        dark_bar.addWidget(self.clear_captures_button)
        controls.addLayout(dark_bar)
        self.capture_status = QLabel("Reference: none | Dark: none")
        self.capture_status.setWordWrap(True)
        controls.addWidget(self.capture_status)

        self.heading("Camera Sensitivity", controls)
        profile_label = QLabel(SENSITIVITY_PROFILE_NAME)
        profile_label.setWordWrap(True)
        controls.addWidget(profile_label)
        self.sensitivity_toggle = QCheckBox("Compensate sensitivity")
        self.sensitivity_toggle.setToolTip(
            "Only for the IDS UI-3242LE-M monochrome sensor. Approximate QE values "
            "read from page 1 of the supplied datasheet. Corrects relative optical "
            "power, normalized at 550 nm; requires linear, unsaturated camera data. "
            "Not a calibration of the grating, optics, or other camera models. "
            "Bypassed for sample/reference ratios."
        )
        self.sensitivity_toggle.toggled.connect(self.process_frame)
        controls.addWidget(self.sensitivity_toggle)
        self.sensitivity_status = QLabel("Unavailable: no wavelength calibration")
        self.sensitivity_status.setWordWrap(True)
        controls.addWidget(self.sensitivity_status)

        self.heading("Peaks and Dips", controls)
        feature_bar = QHBoxLayout()
        self.show_peaks = QCheckBox("Peaks")
        self.show_dips = QCheckBox("Dips")
        self.show_peaks.setChecked(True)
        for checkbox in (self.show_peaks, self.show_dips):
            checkbox.toggled.connect(self.process_frame)
            feature_bar.addWidget(checkbox)
        controls.addLayout(feature_bar)
        feature_form = QFormLayout()
        self.prominence = QDoubleSpinBox()
        self.prominence.setRange(0.1, 100)
        self.prominence.setValue(5)
        self.prominence.setSuffix(" %")
        self.prominence.setToolTip("Minimum feature prominence as a percentage of the displayed signal range")
        self.distance = QSpinBox()
        self.distance.setRange(1, 10000)
        self.distance.setValue(10)
        self.distance.setSuffix(" px")
        self.smoothing = QSpinBox()
        self.smoothing.setRange(1, 101)
        self.smoothing.setSingleStep(2)
        self.smoothing.setValue(1)
        self.smoothing.setToolTip("Savitzky-Golay window in pixels; values below 5 disable smoothing")
        for name, widget in (("Prominence", self.prominence), ("Separation", self.distance), ("Smoothing", self.smoothing)):
            widget.valueChanged.connect(self.process_frame)
            feature_form.addRow(name, widget)
        controls.addLayout(feature_form)
        controls.addStretch()

        controls = panel_layout
        self.heading("Export", controls)
        export_bar = QHBoxLayout()
        self.csv_button = self.command("Save CSV", self.save_csv)
        self.image_button = self.command("Save graph", self.save_graph)
        export_bar.addWidget(self.csv_button)
        export_bar.addWidget(self.image_button)
        controls.addLayout(export_bar)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)
        splitter.setSizes([890, 320])
        self.statusBar().showMessage("Disconnected")
        self.setStyleSheet("""
            QWidget { font-family: 'Segoe UI'; font-size: 9pt; color: #e6e8eb; }
            QWidget:disabled { color: #8c909b; }
            QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #24262a; }
            QLabel#appTitle { font-family: 'Bahnschrift'; font-size: 22px; font-weight: 600; }
            QLabel#sectionHeading { font-weight: 700; color: #7bd1b5; margin-top: 9px; }
            QPushButton, QToolButton {
                min-height: 28px; padding: 2px 6px;
                background: #3a3d44; border: 1px solid #555862; border-radius: 4px;
            }
            QPushButton:hover, QToolButton:hover { background: #484c54; border-color: #7bd1b5; }
            QPushButton:disabled, QToolButton:disabled { background: #303238; border-color: #42454d; }
            QToolButton:checked { background: #36584f; border-color: #7bd1b5; }
            QToolButton { min-width: 28px; }
            QComboBox, QSpinBox, QDoubleSpinBox { min-height: 26px; }
            QScrollArea { border: 0; }
            QTableWidget { background: #35373d; gridline-color: #555862; }
            QHeaderView::section { background: #404249; color: #e6e8eb; padding: 5px; }
            QTableCornerButton::section { background: #404249; }
            QTabWidget::pane { border: 1px solid #4a4d55; }
            QTabBar::tab { background: #303238; color: #c4c7cf; padding: 8px; }
            QTabBar::tab:selected { background: #404249; color: #7bd1b5; }
            QToolTip { background: #404249; color: #e6e8eb; border: 1px solid #62656e; }
        """)
        self.update_actions()

    def capture_connected(self):
        return self.camera is not None or self.directshow is not None

    def capture_backend_changed(self):
        if self.capture_connected():
            return
        self.disconnect_camera()
        self.refresh_devices()

    def refresh_devices(self):
        previous = self.devices.currentData()
        if self.capture_backend.currentData() == "dshow":
            with QSignalBlocker(self.devices):
                self.devices.clear()
                for index in range(16):
                    self.devices.addItem(f"Camera index {index}", index)
                selected = self.devices.findData(previous)
                self.devices.setCurrentIndex(selected if selected >= 0 else 1)
            self.devices.setEnabled(not self.capture_connected())
            self.connect_button.setEnabled(True)
            self.device_changed()
            return
        available = QMediaDevices.videoInputs()
        with QSignalBlocker(self.devices):
            self.devices.clear()
            for device in available:
                self.devices.addItem(device.description(), bytes(device.id()).hex())
            index = self.devices.findData(previous)
            self.devices.setCurrentIndex(max(0, index))
        if self.camera and self.devices.findData(self.camera_id) < 0:
            self.disconnect_camera()
            self.statusBar().showMessage("Camera disconnected")
        self.devices.setEnabled(self.camera is None)
        self.device_changed()
        self.connect_button.setEnabled(bool(available))
        if not available:
            self.statusBar().showMessage("No cameras detected. Check Windows camera privacy permissions.")

    def selected_device(self):
        return next((device for device in QMediaDevices.videoInputs()
                     if bytes(device.id()).hex() == self.devices.currentData()), None)

    def device_changed(self):
        if self.capture_connected():
            return
        self.format_select.clear()
        if self.capture_backend.currentData() == "dshow":
            self.format_select.addItem("Driver default resolution / frame rate")
            self.format_select.setEnabled(False)
            return
        self.format_select.setEnabled(True)
        device = self.selected_device()
        if device is None:
            return
        formats = sorted(device.videoFormats(), key=lambda item: (
            abs(item.resolution().width() - 1280), abs(item.maxFrameRate() - 30)
        ))
        seen = set()
        for camera_format in formats:
            size = camera_format.resolution()
            key = (size.width(), size.height(), round(camera_format.maxFrameRate()))
            if key not in seen:
                seen.add(key)
                self.format_select.addItem(f"{key[0]} x {key[1]} | {key[2]} fps", camera_format)

    def toggle_camera(self):
        if self.capture_connected():
            self.disconnect_camera()
            return
        if self.capture_backend.currentData() == "dshow":
            if sys.platform != "win32":
                self.statusBar().showMessage("DirectShow capture requires Windows.")
                return
            index = self.devices.currentData()
            if index is None:
                return
            self.camera_id = f"opencv-dshow:index:{index}"
            self.directshow = DirectShowCapture(index, self)
            self.directshow.frame_ready.connect(self.receive_directshow_frame)
            self.directshow.opened.connect(self.directshow_opened)
            self.directshow.failed.connect(self.directshow_error)
            self.directshow.finished.connect(self.directshow_finished)
            self.capture_backend.setEnabled(False)
            self.devices.setEnabled(False)
            self.format_select.setEnabled(False)
            self.exposure_auto.setEnabled(False)
            self.exposure.setEnabled(False)
            self.exposure_note.setText("Exposure: driver-managed")
            self.connect_button.setText("Disconnect")
            self.statusBar().showMessage(f"Connecting DirectShow camera {index}...")
            self.directshow.start()
            return
        device = self.selected_device()
        if device is None:
            return
        self.camera_id = bytes(device.id()).hex()
        self.camera = QCamera(device, self)
        if self.format_select.currentData() is not None:
            self.camera.setCameraFormat(self.format_select.currentData())
        self.camera.errorOccurred.connect(self.camera_error)
        self.camera.activeChanged.connect(self.camera_active_changed)
        self.session.setCamera(self.camera)
        self.capture_backend.setEnabled(False)
        self.devices.setEnabled(False)
        self.format_select.setEnabled(False)
        self.connect_button.setText("Disconnect")
        self.statusBar().showMessage("Connecting camera...")
        self.camera.start()

    @Slot()
    def directshow_opened(self):
        if self.sender() is self.directshow:
            self.statusBar().showMessage("Live camera | OpenCV / DirectShow")

    @Slot(object)
    def receive_directshow_frame(self, frame):
        if self.directshow is not None and self.sender() is self.directshow and not self.pause_button.isChecked():
            self.set_frame(frame)

    @Slot(str)
    def directshow_error(self, message):
        if self.directshow is not None and self.sender() is self.directshow:
            self.camera_error(None, message)

    @Slot()
    def directshow_finished(self):
        worker = self.sender()
        if worker is self.directshow:
            self.directshow = None
            self.disconnect_camera()
        worker.deleteLater()

    def camera_active_changed(self, active):
        if not active or self.camera is None:
            return
        self.statusBar().showMessage("Live camera")
        supported = self.camera.isExposureModeSupported(QCamera.ExposureManual)
        with QSignalBlocker(self.exposure_auto):
            self.exposure_auto.setChecked(True)
        self.exposure_auto.setEnabled(supported)
        self.exposure.setEnabled(False)
        minimum = self.camera.minimumExposureTime() * 1000
        maximum = self.camera.maximumExposureTime() * 1000
        if maximum > minimum >= 0:
            self.exposure.setRange(max(0.01, minimum), maximum)
        self.exposure_note.setText("Manual exposure: available" if supported else "Manual exposure: unavailable")

    def set_auto_exposure(self, enabled):
        if self.camera is None:
            return
        self.clear_captures()
        self.exposure.setEnabled(not enabled)
        if enabled:
            self.camera.setExposureMode(QCamera.ExposureAuto)
        else:
            self.camera.setExposureMode(QCamera.ExposureManual)
            self.camera.setManualExposureTime(self.exposure.value() / 1000)

    def set_exposure(self):
        if self.camera and not self.exposure_auto.isChecked():
            self.clear_captures()
            self.camera.setManualExposureTime(self.exposure.value() / 1000)

    def disconnect_camera(self):
        if self.directshow is not None:
            self.directshow.requestInterruption()
            if not self.directshow.wait(3000):
                self.statusBar().showMessage("Waiting for the camera driver. Retry Disconnect after the current exposure.")
                return False
            self.directshow = None
        if self.camera:
            camera = self.camera
            self.camera = None
            camera.stop()
            self.session.setCamera(None)
            camera.deleteLater()
        self.frame = None
        self.frame_size = None
        self.latest = None
        with QSignalBlocker(self.sensitivity_toggle):
            self.sensitivity_toggle.setChecked(False)
        self.image.clear()
        self.roi.hide()
        self.curve.clear()
        self.clear_feature_markers()
        self.clear_captures()
        self.readout.setText("No measurement")
        self.capture_backend.setEnabled(True)
        self.devices.setEnabled(True)
        self.format_select.setEnabled(self.capture_backend.currentData() == "qt")
        self.connect_button.setText("Connect")
        self.pause_button.setChecked(False)
        self.statusBar().showMessage("Disconnected")
        self.update_actions()

    def camera_error(self, error, message):
        self.disconnect_camera()
        self.statusBar().showMessage(f"Camera error: {message}")

    def pause_changed(self):
        paused = self.pause_button.isChecked()
        self.statusBar().showMessage("Measurement paused" if paused else "Live camera")
        self.reference_button.setEnabled(self.frame is not None and not paused)
        self.dark_button.setEnabled(self.frame is not None and not paused)
        self.process_frame()

    def receive_frame(self, video_frame):
        now = time.monotonic()
        if self.camera is None or self.pause_button.isChecked() or now - self.last_frame_time < 1 / 15:
            return
        image = video_frame.toImage().convertToFormat(QImage.Format_RGB888)
        if image.isNull():
            return
        self.last_frame_time = now
        pixels = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
        frame = pixels.reshape(image.height(), image.bytesPerLine())[:, :image.width() * 3]
        self.set_frame(frame.reshape(image.height(), image.width(), 3).copy())

    def set_frame(self, frame):
        self.frame = frame
        self.capture_time = datetime.now(timezone.utc).isoformat()
        height, width = frame.shape[:2]
        self.image.setImage(frame, autoLevels=False, levels=(0, 255))
        if self.frame_size != (width, height):
            self.frame_size = (width, height)
            if self.calibration and not self.calibration.matches(self.camera_id, width, height):
                self.calibration = None
                self.calibration_status.setText("Uncalibrated: pixel axis")
                self.plot.enableAutoRange()
                self.statusBar().showMessage("Camera/resolution changed: wavelength calibration cleared.", 10000)
            for control, maximum in zip(self.crop_inputs, (width - 1, height - 1, width, height)):
                with QSignalBlocker(control):
                    control.setMaximum(maximum)
            self.reset_crop()
            self.preview_view.setRange(xRange=(0, width), yRange=(0, height), padding=0.02)
        self.process_frame()
        self.update_actions()

    def roi_values(self):
        return tuple(control.value() for control in self.crop_inputs)

    def set_crop(self, left, top, width, height):
        if self.frame is None:
            return
        frame_height, frame_width = self.frame.shape[:2]
        left = min(max(int(round(left)), 0), frame_width - 1)
        top = min(max(int(round(top)), 0), frame_height - 1)
        width = min(max(int(round(width)), 1), frame_width - left)
        height = min(max(int(round(height)), 1), frame_height - top)
        with QSignalBlocker(self.roi):
            self.roi.setPos([left, top])
            self.roi.setSize([width, height])
        for control, value in zip(self.crop_inputs, (left, top, width, height)):
            with QSignalBlocker(control):
                control.setValue(value)
        self.roi.show()
        self.clear_captures()
        self.process_frame()

    def reset_crop(self):
        if self.frame is not None:
            height, width = self.frame.shape[:2]
            self.set_crop(0, height // 2 - max(1, height // 20) // 2, width, max(1, height // 20))

    def roi_changed(self):
        self.set_crop(self.roi.pos().x(), self.roi.pos().y(), self.roi.size().x(), self.roi.size().y())

    def crop_inputs_changed(self):
        self.set_crop(*self.roi_values())

    def clear_captures(self):
        self.reference = None
        self.dark = None
        with QSignalBlocker(self.use_reference):
            self.use_reference.setChecked(False)
        self.use_reference.setEnabled(False)
        self.update_capture_status()
        self.process_frame()

    def update_capture_status(self):
        self.capture_status.setText(
            f"Reference: {'captured' if self.reference is not None else 'none'} | "
            f"Dark: {'captured' if self.dark is not None else 'none'}"
        )

    def capture_reference(self):
        if self.frame is None or self.pause_button.isChecked():
            return
        if self.exposure_auto.isChecked() and QMessageBox.warning(
            self, "Automatic exposure", "Automatic exposure may invalidate reference measurements. "
            "Keep exposure, gain, lighting and optics fixed. Capture anyway?",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel
        ) != QMessageBox.Yes:
            return
        self.reference = extract_spectrum(self.frame, self.roi_values())[1].copy()
        self.use_reference.setEnabled(True)
        self.use_reference.setChecked(True)
        self.update_capture_status()
        self.process_frame()

    def capture_dark(self):
        if self.frame is None or self.pause_button.isChecked():
            return
        self.dark = extract_spectrum(self.frame, self.roi_values())[1].copy()
        self.update_capture_status()
        self.process_frame()

    def sensitivity_active(self):
        return (
            self.frame is not None and self.calibration is not None
            and self.sensitivity_toggle.isChecked()
            and not (self.use_reference.isChecked() and self.reference is not None)
        )

    def update_sensitivity_state(self):
        has_calibration = self.frame is not None and self.calibration is not None
        ratio = self.use_reference.isChecked() and self.reference is not None
        self.sensitivity_toggle.setEnabled(has_calibration and not ratio)
        if ratio:
            self.sensitivity_status.setText("Bypassed: sample/reference ratio")
        elif not has_calibration:
            self.sensitivity_status.setText("Unavailable: no wavelength calibration")
        else:
            self.sensitivity_status.setText(
                "On | Relative optical power" if self.sensitivity_active() else "Off"
            )

    def process_frame(self):
        self.update_sensitivity_state()
        if self.frame is None:
            return
        pixels, raw = extract_spectrum(self.frame, self.roi_values())
        reference = self.reference if self.use_reference.isChecked() else None
        horizontal = self.calibration.wavelengths(pixels) if self.calibration else pixels
        values = correct_spectrum(raw, reference, self.dark)
        if self.sensitivity_active():
            values = compensate_sensitivity(values, horizontal)
        values = smooth_spectrum(values, self.smoothing.value())
        self.latest = (pixels, horizontal, raw, values)
        self.curve.setFillBrush(self.spectrum_brush if self.calibration else self.uncalibrated_brush)
        self.curve.setData(horizontal, values, connect="finite")
        self.plot.setLabel("bottom", "Wavelength" if self.calibration else "Pixel position", units="nm" if self.calibration else None)
        intensity_label = "Response-corrected intensity" if self.sensitivity_active() else "Relative intensity"
        self.plot.setLabel("left", "Sample/reference" if reference is not None else intensity_label, units="" if reference is not None else "a.u.")
        self.clear_feature_markers()
        peaks, dips = detect_features(values, self.prominence.value(), self.distance.value())
        for indices, checkbox, markers, color, anchor in (
            (peaks, self.show_peaks, self.peak_markers, "#ff8296", (0.5, 1.3)),
            (dips, self.show_dips, self.dip_markers, "#87b8ff", (0.5, -0.3)),
        ):
            if not checkbox.isChecked():
                continue
            markers.setData(horizontal[indices], values[indices])
            if self.calibration is None:
                continue
            strongest = sorted(indices, key=lambda index: values[index], reverse=checkbox is self.show_peaks)[:12]
            for index in strongest:
                label = pg.TextItem(f"{horizontal[index]:.1f} nm", color=color, anchor=anchor)
                label.setPos(horizontal[index], values[index])
                self.plot.addItem(label, ignoreBounds=True)
                self.peak_labels.append(label)
        left, top, width, height = self.roi_values()
        clipped = np.any(self.frame[top:top + height, left:left + width] >= 250, axis=2).mean() * 100
        invalid = np.count_nonzero(~np.isfinite(values))
        state = "Paused" if self.pause_button.isChecked() else "Live"
        self.readout.setText(f"{state} | {self.frame.shape[1]} x {self.frame.shape[0]} | ROI {width} x {height} | "
                             f"Near-clipped pixels: {clipped:.1f}% | Masked bins: {invalid}")

    def clear_feature_markers(self):
        self.peak_markers.clear()
        self.dip_markers.clear()
        for label in self.peak_labels:
            self.plot.removeItem(label)
        self.peak_labels.clear()

    def cursor_moved(self, position):
        if self.latest is None or not self.plot.sceneBoundingRect().contains(position):
            return
        coordinate = self.plot.plotItem.vb.mapSceneToView(position).x()
        pixels, horizontal, raw, values = self.latest
        index = int(np.argmin(np.abs(horizontal - coordinate)))
        self.cursor.setPos(horizontal[index])
        self.cursor.show()
        unit = "nm" if self.calibration else "px"
        self.statusBar().showMessage(f"{horizontal[index]:.2f} {unit} | Full-frame pixel {pixels[index]:.0f} | Value {values[index]:.4g}")

    def graph_clicked(self, event):
        if event.button() != Qt.LeftButton or self.latest is None or not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        row = self.calibration_table.currentRow()
        if row < 0:
            return
        coordinate = self.plot.plotItem.vb.mapSceneToView(event.scenePos()).x()
        pixels, horizontal, raw, values = self.latest
        index = int(np.argmin(np.abs(horizontal - coordinate)))
        self.calibration_table.setItem(row, 0, QTableWidgetItem(str(int(pixels[index]))))
        self.calibration_table.setCurrentCell(row, 1)

    def add_point(self):
        self.calibration_table.insertRow(self.calibration_table.rowCount())
        self.calibration_table.setCurrentCell(self.calibration_table.rowCount() - 1, 0)

    def remove_point(self):
        row = self.calibration_table.currentRow()
        if row >= 0 and self.calibration_table.rowCount() > 2:
            self.calibration_table.removeRow(row)

    def apply_calibration(self):
        if self.frame is None:
            return
        try:
            points = []
            for row in range(self.calibration_table.rowCount()):
                items = [self.calibration_table.item(row, column) for column in range(2)]
                if any(item is None or not item.text().strip() for item in items):
                    raise ValueError("Complete every calibration row or remove unused rows.")
                points.append(tuple(float(item.text()) for item in items))
            height, width = self.frame.shape[:2]
            calibration = Calibration(self.camera_id, width, height, tuple(sorted(points)))
        except ValueError as error:
            self.report_error(error)
            return
        self.calibration = calibration
        self.calibration_status.setText(f"Calibrated | {len(points)} points | Outside anchors: extrapolated")
        self.process_frame()
        self.plot.enableAutoRange()
        self.update_actions()

    def clear_calibration(self):
        self.calibration = None
        self.calibration_status.setText("Uncalibrated: pixel axis")
        self.process_frame()
        self.plot.enableAutoRange()
        self.update_actions()

    def save_calibration(self):
        if self.calibration is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save calibration", "calibration.json", "Calibration (*.json)")
        if not filename:
            return
        try:
            data = self.calibration.to_dict()
            data["roi"] = list(self.roi_values())
            Path(filename).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as error:
            self.report_error(error)

    def load_calibration(self):
        if self.frame is None:
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Load calibration", "", "Calibration (*.json)")
        if not filename:
            return
        try:
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
            calibration = Calibration.from_dict(data)
            height, width = self.frame.shape[:2]
            if not calibration.matches(self.camera_id, width, height):
                raise ValueError("Calibration belongs to another camera or frame resolution. Recalibrate this setup.")
            roi = data.get("roi", list(self.roi_values()))
            if len(roi) != 4 or not all(isinstance(value, int) for value in roi):
                raise ValueError("Invalid crop in calibration file.")
            extract_spectrum(self.frame, roi)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
            self.report_error(error)
            return
        self.calibration = calibration
        self.calibration_table.setRowCount(len(calibration.points))
        for row, point in enumerate(calibration.points):
            for column, value in enumerate(point):
                self.calibration_table.setItem(row, column, QTableWidgetItem(f"{value:g}"))
        self.set_crop(*roi)
        self.calibration_status.setText(f"Calibrated | {len(calibration.points)} points | Outside anchors: extrapolated")
        self.plot.enableAutoRange()
        self.update_actions()

    def save_csv(self):
        if self.latest is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save spectrum", "spectrum.csv", "CSV (*.csv)")
        if not filename:
            return
        try:
            pixels, horizontal, raw, values = self.latest
            reference = self.reference if self.use_reference.isChecked() else None
            metadata = {
                "timestamp_utc": self.capture_time,
                "camera_id": self.camera_id,
                "capture_backend": self.capture_backend.currentData(),
                "directshow_index": self.devices.currentData() if self.capture_backend.currentData() == "dshow" else None,
                "frame_size": self.frame_size,
                "roi": self.roi_values(),
                "calibration": self.calibration.to_dict() if self.calibration else None,
                "reference_enabled": reference is not None,
                "dark_enabled": self.dark is not None,
                "smoothing_window": self.smoothing.value(),
                "intensity_units": (
                    "sample/reference" if reference is not None else
                    "relative optical power (sensor QE corrected)" if self.sensitivity_active()
                    else "relative camera units"
                ),
                "sensitivity_compensation_requested": self.sensitivity_toggle.isChecked(),
                "sensitivity_compensation_applied": self.sensitivity_active(),
                "sensitivity_profile": {
                    "name": SENSITIVITY_PROFILE_NAME,
                    "source": "User-supplied IDS UI-3242LE-M datasheet, page 1 QE graph",
                    "approximate": True,
                    "normalization_nm": 550,
                    "minimum_relative_qe": 0.05,
                    "method": "I_corrected = I * QE(550) * 550 / (QE(wavelength) * wavelength)",
                    "qe_percent_points": IDS_QE_PROFILE.tolist(),
                } if self.sensitivity_toggle.isChecked() else None,
                "automatic_exposure": self.exposure_auto.isChecked() if self.capture_backend.currentData() == "qt" else None,
            }
            with Path(filename).open("w", newline="", encoding="utf-8") as output:
                output.write("# " + json.dumps(metadata) + "\n")
                writer = csv.writer(output)
                writer.writerow(["pixel", "wavelength_nm", "intensity", "raw_intensity", "reference", "dark"])
                for index, pixel in enumerate(pixels):
                    writer.writerow([
                        int(pixel), horizontal[index] if self.calibration else "",
                        values[index], raw[index],
                        reference[index] if reference is not None else "",
                        self.dark[index] if self.dark is not None else "",
                    ])
            self.statusBar().showMessage(f"Saved {filename}", 10000)
        except OSError as error:
            self.report_error(error)

    def save_graph(self):
        if self.latest is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save graph", "spectrum.png", "PNG (*.png)")
        if not filename:
            return
        try:
            self.cursor.hide()
            exporter = pg.exporters.ImageExporter(self.plot.plotItem)
            exporter.parameters()["width"] = 1800
            exporter.export(filename)
            self.statusBar().showMessage(f"Saved {filename}", 10000)
        except (OSError, ValueError, RuntimeError) as error:
            self.report_error(error)

    def report_error(self, error):
        QMessageBox.warning(self, "Simple Spectrometer", str(error))

    def update_actions(self):
        has_frame = self.frame is not None
        self.update_sensitivity_state()
        self.pause_button.setEnabled(has_frame)
        for control in self.crop_inputs:
            control.setEnabled(has_frame)
        for control in (self.reset_crop_button, self.apply_calibration_button, self.load_calibration_button):
            control.setEnabled(has_frame)
        for control in (self.reference_button, self.dark_button):
            control.setEnabled(has_frame and not self.pause_button.isChecked())
        self.save_calibration_button.setEnabled(has_frame and self.calibration is not None)
        self.csv_button.setEnabled(self.latest is not None)
        self.image_button.setEnabled(self.latest is not None)

    def closeEvent(self, event):
        if self.disconnect_camera() is False:
            event.ignore()
            return
        event.accept()


def main():
    application = QApplication(sys.argv)
    application.setApplicationName("Simple Spectrometer")
    window = Spectrometer()
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()