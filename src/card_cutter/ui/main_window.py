from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QSettings, QSignalBlocker, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent, QColor, QImage, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from card_cutter.config import (
    APP_NAME,
    DEFAULT_MARGIN_PX,
    INPUT_MARGIN_MAX,
    ORGANIZATION_NAME,
    SLIDER_MARGIN_MAX,
)
from card_cutter.io import ImageValidationError, load_image
from card_cutter.models import ProcessingOptions, ProcessingStatus, ProcessResult
from card_cutter.services import process_folder, process_selected


class ProcessingWorker(QObject):
    progress = Signal(int, int, object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        mode: str,
        inputs: list[Path] | Path,
        output_root: Path,
        options: ProcessingOptions,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.inputs = inputs
        self.output_root = output_root
        self.options = options
        self.cancel_event = cancel_event

    @Slot()
    def run(self) -> None:
        try:
            def callback(current: int, total: int, result: ProcessResult) -> None:
                self.progress.emit(current, total, result)

            if self.mode == "folder":
                if not isinstance(self.inputs, Path):
                    raise TypeError("Folder mode requires one input directory")
                results = process_folder(
                    self.inputs,
                    self.output_root,
                    self.options,
                    cancel_event=self.cancel_event,
                    progress=callback,
                )
            else:
                if isinstance(self.inputs, Path):
                    raise TypeError("Selected-image mode requires a list of files")
                results = process_selected(
                    self.inputs,
                    self.output_root,
                    self.options,
                    cancel_event=self.cancel_event,
                    progress=callback,
                )
            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class ThemeSwitch(QAbstractButton):
    """A compact, accessible two-position theme switch."""

    def __init__(self) -> None:
        super().__init__()
        self.setCheckable(True)
        self.setFixedSize(50, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("Light and dark mode")
        self.setToolTip("Switch to dark mode")

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1570ef") if self.isChecked() else QColor("#98a2b3"))
        painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)
        knob_size = track.height() - 6
        knob_x = track.right() - knob_size - 2 if self.isChecked() else track.left() + 3
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(knob_x, track.top() + 3, knob_size, knob_size)


class ImagePreview(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("imagePreview")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setObjectName("previewHeading")
        self.image_label = QLabel("No image")
        self.image_label.setObjectName("previewImage")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(260, 260)
        layout.addWidget(heading)
        layout.addWidget(self.image_label, 1)
        self._pixmap: QPixmap | None = None

    def set_path(self, path: Path | None) -> None:
        if path is None:
            self._pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("No image")
            return
        try:
            pixels = load_image(path).pixels
            self._pixmap = self._pixmap_from_array(pixels)
            self.image_label.setText("")
            self._scale_pixmap()
        except ImageValidationError as exc:
            self._pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f"Preview unavailable\n{exc}")

    @staticmethod
    def _pixmap_from_array(pixels: np.ndarray) -> QPixmap:
        contiguous = np.ascontiguousarray(pixels)
        height, width, channels = contiguous.shape
        image = QImage(
            contiguous.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(image)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._scale_pixmap()

    def _scale_pixmap(self) -> None:
        if self._pixmap is None:
            return
        target = self.image_label.size() - QSize(12, 12)
        self.image_label.setPixmap(
            self._pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1120, 760)
        self.settings = QSettings(ORGANIZATION_NAME, APP_NAME)
        self._thread: QThread | None = None
        self._worker: ProcessingWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._selected_paths: list[Path] = []
        self._dark_mode = False
        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(9)
        self.title_label = QLabel("Card Cutter")
        self.title_label.setObjectName("appTitle")
        self.credit_label = QLabel("— by the GOAT, yours truly, Goliver")
        self.credit_label.setObjectName("appCredit")
        self.theme_toggle = ThemeSwitch()
        self.theme_toggle.toggled.connect(self._theme_toggled)
        theme_widget = QWidget()
        theme_widget.setObjectName("themeControls")
        theme_layout = QHBoxLayout(theme_widget)
        theme_layout.setContentsMargins(0, 0, 0, 0)
        theme_layout.setSpacing(7)
        sun_label = QLabel("☀")
        sun_label.setObjectName("themeIcon")
        sun_label.setToolTip("Light mode")
        moon_label = QLabel("☾")
        moon_label.setObjectName("themeIcon")
        moon_label.setToolTip("Dark mode")
        theme_layout.addWidget(sun_label)
        theme_layout.addWidget(self.theme_toggle)
        theme_layout.addWidget(moon_label)
        header.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignBottom)
        header.addWidget(self.credit_label, 0, Qt.AlignmentFlag.AlignBottom)
        header.addStretch()
        header.addWidget(theme_widget, 0, Qt.AlignmentFlag.AlignVCenter)
        subtitle = QLabel("Crop, straighten, and orient card or graded-slab photos locally.")
        subtitle.setObjectName("appSubtitle")
        root.addLayout(header)
        root.addWidget(subtitle)

        options_box = QGroupBox("Processing settings")
        options_layout = QHBoxLayout(options_box)
        margin_label = QLabel("Outside margin")
        self.margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.margin_slider.setRange(0, SLIDER_MARGIN_MAX)
        self.margin_slider.setValue(DEFAULT_MARGIN_PX)
        self.margin_slider.setMinimumWidth(220)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, INPUT_MARGIN_MAX)
        self.margin_spin.setSuffix(" px")
        self.margin_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.margin_spin.setMinimumWidth(96)
        self.margin_spin.setToolTip("Type a margin or use the visible - and + buttons")
        self.margin_decrease_button = QPushButton("-")
        self.margin_decrease_button.setObjectName("spinStepButton")
        self.margin_decrease_button.setAccessibleName("Decrease outside margin")
        self.margin_decrease_button.setToolTip("Decrease outside margin by 1 pixel")
        self.margin_decrease_button.setFixedSize(30, 30)
        self.margin_decrease_button.clicked.connect(self.margin_spin.stepDown)
        self.margin_increase_button = QPushButton("+")
        self.margin_increase_button.setObjectName("spinStepButton")
        self.margin_increase_button.setAccessibleName("Increase outside margin")
        self.margin_increase_button.setToolTip("Increase outside margin by 1 pixel")
        self.margin_increase_button.setFixedSize(30, 30)
        self.margin_increase_button.clicked.connect(self.margin_spin.stepUp)
        margin_number_widget = QWidget()
        margin_number_layout = QHBoxLayout(margin_number_widget)
        margin_number_layout.setContentsMargins(0, 0, 0, 0)
        margin_number_layout.setSpacing(4)
        margin_number_layout.addWidget(self.margin_spin)
        margin_number_layout.addWidget(self.margin_decrease_button)
        margin_number_layout.addWidget(self.margin_increase_button)
        self.margin_spin.setValue(DEFAULT_MARGIN_PX)
        self.straighten_check = QCheckBox("Straighten perspective")
        self.straighten_check.setChecked(True)
        self.rotate_check = QCheckBox("Auto-rotate using text")
        self.rotate_check.setChecked(True)
        self.margin_slider.valueChanged.connect(self._slider_margin_changed)
        self.margin_spin.valueChanged.connect(self._spin_margin_changed)
        options_layout.addWidget(margin_label)
        options_layout.addWidget(self.margin_slider, 1)
        options_layout.addWidget(margin_number_widget)
        options_layout.addSpacing(16)
        options_layout.addWidget(self.straighten_check)
        options_layout.addWidget(self.rotate_check)
        root.addWidget(options_box)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_selected_tab(), "Selected Images")
        self.tabs.addTab(self._build_batch_tab(), "Batch Folder")
        root.addWidget(self.tabs)

        previews = QSplitter(Qt.Orientation.Horizontal)
        self.before_preview = ImagePreview("Before")
        self.after_preview = ImagePreview("After / needs review")
        previews.addWidget(self.before_preview)
        previews.addWidget(self.after_preview)
        previews.setSizes([540, 540])
        root.addWidget(previews, 1)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_processing)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress_bar, 2)
        status_row.addWidget(self.cancel_button)
        root.addLayout(status_row)

        self.setCentralWidget(central)
        self._apply_theme(False)

    def _apply_theme(self, dark: bool) -> None:
        self._dark_mode = dark
        blocker = QSignalBlocker(self.theme_toggle)
        self.theme_toggle.setChecked(dark)
        del blocker
        self.theme_toggle.setToolTip(
            "Switch to light mode" if dark else "Switch to dark mode"
        )
        if dark:
            self.setStyleSheet(
                """
                QMainWindow, QWidget { background: #0b1526; color: #e8edf5; }
                QLabel#appTitle { font-size: 28px; font-weight: 700; color: #f9fafb; }
                QLabel#appCredit { font-size: 13px; color: #f9fafb; }
                QLabel#appSubtitle { color: #aeb9ca; margin-bottom: 5px; }
                QLabel#themeIcon { color: #f9fafb; font-size: 18px; }
                QLabel#previewHeading { font-size: 14px; font-weight: 600; color: #f1f5f9; }
                QLabel#previewImage { color: #aeb9ca; }
                QLabel, QCheckBox, QRadioButton { color: #e8edf5; }
                QGroupBox { color: #e8edf5; font-weight: 600; border: 1px solid #5b6c84;
                            border-radius: 8px; margin-top: 9px; padding-top: 12px;
                            background: #192a40; }
                QGroupBox::title { color: #f1f5f9; subcontrol-origin: margin;
                                   left: 12px; padding: 0 4px; }
                QGroupBox QLabel, QGroupBox QCheckBox { background: #192a40; color: #e8edf5; }
                QTabWidget::pane { border: 1px solid #5b6c84; border-radius: 6px;
                                   background: #122036; }
                QTabBar::tab { background: #3b4d66; color: #eef2f7; padding: 6px 14px;
                               border-top-left-radius: 5px; border-top-right-radius: 5px; }
                QTabBar::tab:selected { background: #1570ef; color: white; }
                QTabBar::tab:disabled { color: #aeb9ca; }
                QFrame#imagePreview { background: #111f34; border: 1px solid #52647d;
                                      border-radius: 4px; }
                QFrame#imagePreview QLabel { background: #111f34; border: none; }
                QSplitter::handle { background: #52647d; width: 2px; }
                QPushButton { padding: 7px 13px; border-radius: 5px; }
                QPushButton:enabled { background: #1570ef; color: white; }
                QPushButton:disabled { background: #33445c; color: #aeb9ca; }
                QPushButton#spinStepButton { padding: 0; background: #40536e;
                                             color: #f9fafb; border: 1px solid #718198;
                                             font-size: 18px; font-weight: 700; }
                QLineEdit, QListWidget, QSpinBox { background: #0e1a2d; color: #f9fafb;
                                                  border: 1px solid #718198; padding: 5px; }
                QLineEdit:disabled, QListWidget:disabled, QSpinBox:disabled {
                    background: #24354c; color: #aeb9ca;
                }
                QSpinBox::up-button, QSpinBox::down-button { width: 0; border: none; }
                QProgressBar { background: #354861; color: #f9fafb;
                               border: 1px solid #667891; border-radius: 5px;
                               text-align: center; }
                QProgressBar::chunk { background: #1570ef; border-radius: 4px; }
                """
            )
            return
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f7f8fa; color: #344054; }
            QLabel#appTitle { font-size: 28px; font-weight: 700; color: #172b4d; }
            QLabel#appCredit { font-size: 13px; color: #172b4d; }
            QLabel#appSubtitle { color: #5e6c84; margin-bottom: 5px; }
            QLabel#themeIcon { color: #172b4d; font-size: 18px; }
            QLabel#previewHeading { font-size: 14px; font-weight: 600; color: #344054; }
            QLabel#previewImage { color: #667085; }
            QLabel, QCheckBox, QRadioButton { color: #344054; }
            QGroupBox { color: #344054; font-weight: 600; border: 1px solid #d0d5dd;
                        border-radius: 8px; margin-top: 9px; padding-top: 12px;
                        background: white; }
            QGroupBox::title { color: #344054; subcontrol-origin: margin;
                               left: 12px; padding: 0 4px; }
            QGroupBox QLabel, QGroupBox QCheckBox { background: white; color: #344054; }
            QTabWidget::pane { border: 1px solid #d0d5dd; border-radius: 6px; background: white; }
            QTabBar::tab { background: #e4e7ec; color: #344054; padding: 6px 14px;
                           border-top-left-radius: 5px; border-top-right-radius: 5px; }
            QTabBar::tab:selected { background: #155eef; color: white; }
            QTabBar::tab:disabled { color: #667085; }
            QPushButton { padding: 7px 13px; border-radius: 5px; }
            QPushButton:enabled { background: #155eef; color: white; }
            QPushButton:disabled { background: #d0d5dd; color: #667085; }
            QLineEdit, QListWidget, QSpinBox { background: white; color: #101828;
                                              border: 1px solid #98a2b3; padding: 5px; }
            QLineEdit:disabled, QListWidget:disabled, QSpinBox:disabled {
                background: #f2f4f7; color: #667085;
            }
            QSpinBox::up-button, QSpinBox::down-button { width: 0; border: none; }
            QPushButton#spinStepButton { padding: 0; background: #f2f4f7;
                                         color: #172b4d; border: 1px solid #98a2b3;
                                         font-size: 18px; font-weight: 700; }
            QProgressBar { background: #e4e7ec; color: #101828;
                           border: 1px solid #d0d5dd; border-radius: 5px;
                           text-align: center; }
            QProgressBar::chunk { background: #155eef; border-radius: 4px; }
            """
        )

    @Slot(bool)
    def _theme_toggled(self, checked: bool) -> None:
        self._apply_theme(checked)

    def _build_selected_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        button_row = QHBoxLayout()
        add_button = QPushButton("Add images…")
        remove_button = QPushButton("Remove selected")
        clear_button = QPushButton("Clear")
        add_button.clicked.connect(self._add_images)
        remove_button.clicked.connect(self._remove_selected_images)
        clear_button.clicked.connect(self._clear_images)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(clear_button)
        button_row.addStretch()
        layout.addLayout(button_row)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._selected_preview_changed)
        layout.addWidget(self.image_list)
        output_row = QHBoxLayout()
        self.selected_output = QLineEdit()
        self.selected_output.setReadOnly(True)
        choose_output = QPushButton("Choose output folder…")
        choose_output.clicked.connect(lambda: self._choose_folder(self.selected_output))
        self.selected_process_button = QPushButton("Process selected images")
        self.selected_process_button.clicked.connect(self._start_selected)
        output_row.addWidget(QLabel("Output"))
        output_row.addWidget(self.selected_output, 1)
        output_row.addWidget(choose_output)
        output_row.addWidget(self.selected_process_button)
        layout.addLayout(output_row)
        return tab

    def _build_batch_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        input_widget = QWidget()
        input_row = QHBoxLayout(input_widget)
        input_row.setContentsMargins(0, 0, 0, 0)
        self.batch_input = QLineEdit()
        self.batch_input.setReadOnly(True)
        input_button = QPushButton("Choose input folder…")
        input_button.clicked.connect(
            lambda: self._choose_folder(self.batch_input, input_folder=True)
        )
        input_row.addWidget(self.batch_input, 1)
        input_row.addWidget(input_button)
        form.addRow("Input folder", input_widget)

        output_widget = QWidget()
        output_row = QHBoxLayout(output_widget)
        output_row.setContentsMargins(0, 0, 0, 0)
        self.batch_output = QLineEdit()
        self.batch_output.setReadOnly(True)
        output_button = QPushButton("Choose output folder…")
        output_button.clicked.connect(lambda: self._choose_folder(self.batch_output))
        output_row.addWidget(self.batch_output, 1)
        output_row.addWidget(output_button)
        form.addRow("Output folder", output_widget)
        self.batch_process_button = QPushButton("Process folder recursively")
        self.batch_process_button.clicked.connect(self._start_batch)
        form.addRow("", self.batch_process_button)
        return tab

    def _options(self) -> ProcessingOptions:
        return ProcessingOptions(
            margin_px=self.margin_spin.value(),
            straighten=self.straighten_check.isChecked(),
            auto_rotate=self.rotate_check.isChecked(),
        )

    @Slot(int)
    def _slider_margin_changed(self, value: int) -> None:
        self.margin_spin.setValue(value)

    @Slot(int)
    def _spin_margin_changed(self, value: int) -> None:
        blocker = QSignalBlocker(self.margin_slider)
        self.margin_slider.setValue(min(value, SLIDER_MARGIN_MAX))
        del blocker

    @Slot()
    def _add_images(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose card or slab images",
            "",
            "Images (*.jpg *.jpeg *.png *.heic);;All files (*)",
        )
        existing = set(self._selected_paths)
        for name in names:
            path = Path(name)
            if path not in existing:
                self._selected_paths.append(path)
                item = QListWidgetItem(path.name)
                item.setToolTip(str(path))
                self.image_list.addItem(item)
                existing.add(path)
        if self.image_list.count() and self.image_list.currentRow() < 0:
            self.image_list.setCurrentRow(0)

    @Slot()
    def _remove_selected_images(self) -> None:
        rows = sorted({index.row() for index in self.image_list.selectedIndexes()}, reverse=True)
        for row in rows:
            self.image_list.takeItem(row)
            self._selected_paths.pop(row)
        self._selected_preview_changed(self.image_list.currentRow())

    @Slot()
    def _clear_images(self) -> None:
        self.image_list.clear()
        self._selected_paths.clear()
        self.before_preview.set_path(None)
        self.after_preview.set_path(None)

    @Slot(int)
    def _selected_preview_changed(self, row: int) -> None:
        path = self._selected_paths[row] if 0 <= row < len(self._selected_paths) else None
        self.before_preview.set_path(path)
        self.after_preview.set_path(None)

    def _choose_folder(self, line_edit: QLineEdit, *, input_folder: bool = False) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose folder", line_edit.text())
        if chosen:
            line_edit.setText(chosen)
            if input_folder and not self.batch_output.text():
                source = Path(chosen)
                suggested = source.parent / f"{source.name} - Card Cutter Output"
                self.batch_output.setText(str(suggested))

    @Slot()
    def _start_selected(self) -> None:
        if not self._selected_paths:
            self._show_input_error("Choose at least one image.")
            return
        if not self.selected_output.text():
            self._show_input_error("Choose an output folder.")
            return
        self._start_worker(
            "selected", list(self._selected_paths), Path(self.selected_output.text())
        )

    @Slot()
    def _start_batch(self) -> None:
        if not self.batch_input.text() or not self.batch_output.text():
            self._show_input_error("Choose both the input and output folders.")
            return
        self._start_worker("folder", Path(self.batch_input.text()), Path(self.batch_output.text()))

    def _start_worker(self, mode: str, inputs: list[Path] | Path, output_root: Path) -> None:
        if self._thread is not None:
            return
        self._cancel_event = threading.Event()
        self._thread = QThread(self)
        self._worker = ProcessingWorker(
            mode, inputs, output_root, self._options(), self._cancel_event
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_finished)
        self._set_processing_state(True)
        self.progress_bar.setRange(0, len(inputs) if isinstance(inputs, list) else 0)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing…")
        self._thread.start()

    @Slot(int, int, object)
    def _on_progress(self, current: int, total: int, result_object: object) -> None:
        result = result_object if isinstance(result_object, ProcessResult) else None
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)
        if result is not None:
            self.before_preview.set_path(result.input_path)
            self.after_preview.set_path(result.output_path)
            item_status = result.status.value.replace("_", " ")
            self.status_label.setText(
                f"{current}/{total}: {result.input_path.name} — {item_status}"
            )

    @Slot(object)
    def _on_finished(self, results_object: object) -> None:
        results = results_object if isinstance(results_object, list) else []
        success = sum(
            isinstance(item, ProcessResult) and item.status is ProcessingStatus.SUCCESS
            for item in results
        )
        review = sum(
            isinstance(item, ProcessResult) and item.status is ProcessingStatus.NEEDS_REVIEW
            for item in results
        )
        errors = sum(
            isinstance(item, ProcessResult) and item.status is ProcessingStatus.ERROR
            for item in results
        )
        cancelled = sum(
            isinstance(item, ProcessResult) and item.status is ProcessingStatus.CANCELLED
            for item in results
        )
        summary = (
            f"Finished: {success} processed, {review} need review, "
            f"{errors} errors, {cancelled} cancelled"
        )
        self.status_label.setText(summary)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.status_label.setText("Processing stopped safely")
        QMessageBox.critical(self, "Card Cutter", message)

    @Slot()
    def _thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._cancel_event = None
        self._set_processing_state(False)

    @Slot()
    def _cancel_processing(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling after the current image…")

    def _set_processing_state(self, running: bool) -> None:
        self.tabs.setEnabled(not running)
        self.margin_slider.setEnabled(not running)
        self.margin_spin.setEnabled(not running)
        self.margin_decrease_button.setEnabled(not running)
        self.margin_increase_button.setEnabled(not running)
        self.straighten_check.setEnabled(not running)
        self.rotate_check.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _show_input_error(self, message: str) -> None:
        QMessageBox.warning(self, "Card Cutter", message)

    def _restore_settings(self) -> None:
        margin_value = self.settings.value("margin", DEFAULT_MARGIN_PX)
        self.margin_spin.setValue(int(str(margin_value)))
        straighten_value = self.settings.value("straighten", True)
        rotate_value = self.settings.value("auto_rotate", True)
        self.straighten_check.setChecked(str(straighten_value).casefold() in {"1", "true"})
        self.rotate_check.setChecked(str(rotate_value).casefold() in {"1", "true"})
        self.selected_output.setText(str(self.settings.value("selected_output", "")))
        self.batch_input.setText(str(self.settings.value("batch_input", "")))
        self.batch_output.setText(str(self.settings.value("batch_output", "")))
        dark_mode_value = self.settings.value("dark_mode", False)
        self._apply_theme(str(dark_mode_value).casefold() in {"1", "true"})

    def _save_settings(self) -> None:
        self.settings.setValue("margin", self.margin_spin.value())
        self.settings.setValue("straighten", self.straighten_check.isChecked())
        self.settings.setValue("auto_rotate", self.rotate_check.isChecked())
        self.settings.setValue("selected_output", self.selected_output.text())
        self.settings.setValue("batch_input", self.batch_input.text())
        self.settings.setValue("batch_output", self.batch_output.text())
        self.settings.setValue("dark_mode", self._dark_mode)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None:
            answer = QMessageBox.question(
                self,
                "Processing is running",
                "Cancel processing and close Card Cutter?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._thread.quit()
            if not self._thread.wait(30_000):
                QMessageBox.information(
                    self,
                    "Card Cutter",
                    "The current image is still finishing safely. "
                    "Card Cutter will remain open; try closing again in a moment.",
                )
                event.ignore()
                return
        self._save_settings()
        event.accept()


def create_application() -> QApplication:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    application.setApplicationName(APP_NAME)
    application.setOrganizationName(ORGANIZATION_NAME)
    return application

