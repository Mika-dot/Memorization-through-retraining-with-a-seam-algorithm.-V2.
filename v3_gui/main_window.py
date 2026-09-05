from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mtr3.codec import PRESETS
from mtr3.seam import seam_importance

from .image_view import ImageView
from .workers import DecodeWorker, EncodeWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MTRSA GUI 3.1")
        self.resize(1380, 860)
        self.setAcceptDrops(True)

        self.image_path: str | None = None
        self.bitstream_path: str | None = None
        self._thread: QThread | None = None
        self._worker = None

        root = QWidget()
        self.setCentralWidget(root)
        page = QVBoxLayout(root)
        page.setContentsMargins(16, 16, 16, 16)
        page.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("MTRSA · learned image compression")
        title.setObjectName("appTitle")
        subtitle = QLabel("Per-image neural representation · seam-aware rate allocation · .mtr3")
        subtitle.setObjectName("subTitle")
        title_box = QVBoxLayout()
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        page.addLayout(header)

        controls = QFrame()
        controls.setObjectName("controlBar")
        row = QHBoxLayout(controls)
        row.setContentsMargins(12, 10, 12, 10)

        self.open_image_btn = QPushButton("Open image")
        self.open_bitstream_btn = QPushButton("Open .mtr3")
        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        self.preset.setCurrentText("balanced")
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu"] + (["cuda"] if torch.cuda.is_available() else []))
        self.steps = QSpinBox()
        self.steps.setRange(0, 100000)
        self.steps.setSpecialValueText("preset")
        self.steps.setToolTip("0 uses the selected preset's default optimization budget")
        self.compress_btn = QPushButton("Compress → .mtr3")
        self.compress_btn.setObjectName("primaryButton")
        self.decode_btn = QPushButton("Restore → PNG")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)

        for label, widget in (("Preset", self.preset), ("Device", self.device), ("Steps", self.steps)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.insertWidget(0, self.open_image_btn)
        row.insertWidget(1, self.open_bitstream_btn)
        row.addStretch(1)
        row.addWidget(self.compress_btn)
        row.addWidget(self.decode_btn)
        row.addWidget(self.cancel_btn)
        page.addWidget(controls)

        previews = QHBoxLayout()
        self.original = ImageView("1 · Original")
        self.importance = ImageView("2 · Seam / detail importance")
        self.restored = ImageView("3 · Reconstructed")
        previews.addWidget(self.original, 1)
        previews.addWidget(self.importance, 1)
        previews.addWidget(self.restored, 1)
        page.addLayout(previews, 1)

        bottom = QHBoxLayout()
        metrics_card = QFrame()
        metrics_card.setObjectName("metricsCard")
        form = QFormLayout(metrics_card)
        self.metric_labels: dict[str, QLabel] = {}
        for key, title_text in (
            ("source", "Source"),
            ("source_size", "Source file"),
            ("compressed_size", "MTR3 file"),
            ("ratio", "Ratio vs source"),
            ("bpp", "Bits / pixel"),
            ("psnr", "PSNR"),
            ("encode", "Encode time"),
            ("decode", "Decode time"),
            ("device", "Device"),
        ):
            value = QLabel("—")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metric_labels[key] = value
            form.addRow(title_text, value)
        bottom.addWidget(metrics_card, 1)

        status_card = QFrame()
        status_card.setObjectName("metricsCard")
        status_layout = QVBoxLayout(status_card)
        self.status = QLabel("Drop an image here or open a file.")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.path_label = QLabel("Ready")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_layout.addWidget(self.status)
        status_layout.addStretch(1)
        status_layout.addWidget(self.progress)
        status_layout.addWidget(self.path_label)
        bottom.addWidget(status_card, 2)
        page.addLayout(bottom)

        self.open_image_btn.clicked.connect(self._choose_image)
        self.open_bitstream_btn.clicked.connect(self._choose_bitstream)
        self.compress_btn.clicked.connect(self._compress)
        self.decode_btn.clicked.connect(self._decode)
        self.cancel_btn.clicked.connect(self._cancel)

        self.setStyleSheet(STYLE)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path.lower().endswith(".mtr3"):
            self._load_bitstream(path)
        else:
            self._load_image(path)

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)")
        if path:
            self._load_image(path)

    def _choose_bitstream(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open MTR3", "", "MTRSA bitstream (*.mtr3)")
        if path:
            self._load_bitstream(path)

    def _load_image(self, path: str) -> None:
        try:
            img = Image.open(path).convert("RGB")
            rgb = np.asarray(img)
            imp = seam_importance(rgb)
            vis = np.repeat((imp[..., None] * 255.0).clip(0, 255).astype(np.uint8), 3, axis=2)
        except Exception as exc:
            QMessageBox.critical(self, "Open image", str(exc))
            return
        self.image_path = path
        self.original.set_rgb_array(rgb)
        self.importance.set_rgb_array(vis)
        self.restored.clear("Compress to preview reconstruction")
        self.metric_labels["source"].setText(Path(path).name)
        self.metric_labels["source_size"].setText(_human_size(os.path.getsize(path)))
        self.path_label.setText(path)
        self.status.setText("Image loaded. Choose preset and compress.")

    def _load_bitstream(self, path: str) -> None:
        self.bitstream_path = path
        self.path_label.setText(path)
        self.metric_labels["compressed_size"].setText(_human_size(os.path.getsize(path)))
        self.status.setText("MTR3 bitstream loaded. Choose Restore → PNG.")

    def _compress(self) -> None:
        if not self.image_path:
            QMessageBox.information(self, "Compress", "Open an image first.")
            return
        suggested = str(Path(self.image_path).with_suffix(".mtr3"))
        output, _ = QFileDialog.getSaveFileName(self, "Save compressed image", suggested, "MTRSA bitstream (*.mtr3)")
        if not output:
            return
        if not output.lower().endswith(".mtr3"):
            output += ".mtr3"
        steps = self.steps.value() or None
        worker = EncodeWorker(self.image_path, output, self.preset.currentText(), self.device.currentText(), steps)
        self._start_worker(worker, self._on_encoded)

    def _decode(self) -> None:
        source = self.bitstream_path
        if not source:
            QMessageBox.information(self, "Restore", "Open an .mtr3 file first, or compress an image in this session.")
            return
        suggested = str(Path(source).with_suffix(".restored.png"))
        output, _ = QFileDialog.getSaveFileName(self, "Save restored image", suggested, "PNG image (*.png)")
        if not output:
            return
        if not output.lower().endswith(".png"):
            output += ".png"
        worker = DecodeWorker(source, output, self.device.currentText())
        self._start_worker(worker, self._on_decoded)

    def _start_worker(self, worker, finished_slot) -> None:
        if self._thread is not None:
            return
        self._set_busy(True)
        self.progress.setValue(0)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(finished_slot)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        for signal in (worker.finished, worker.failed, worker.cancelled):
            signal.connect(thread.quit)
        thread.finished.connect(self._worker_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_progress(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        self.status.setText(message)

    def _on_encoded(self, info: dict) -> None:
        self.bitstream_path = str(info["output_path"])
        self.restored.set_path(str(info["preview_path"]))
        source_bytes = os.path.getsize(self.image_path) if self.image_path else 0
        compressed_bytes = int(info["bytes"])
        ratio = (source_bytes / compressed_bytes) if compressed_bytes else 0.0
        self.metric_labels["compressed_size"].setText(_human_size(compressed_bytes))
        self.metric_labels["ratio"].setText(f"{ratio:.2f}×")
        self.metric_labels["bpp"].setText(f"{float(info['bpp']):.4f}")
        self.metric_labels["psnr"].setText(f"{float(info['psnr_db']):.2f} dB")
        self.metric_labels["encode"].setText(f"{float(info['encode_seconds']):.3f} s")
        self.metric_labels["decode"].setText(f"{float(info['decode_seconds']):.3f} s")
        self.metric_labels["device"].setText(str(info["device"]))
        self.path_label.setText(str(info["output_path"]))
        self.status.setText("Compression completed. The .mtr3 file is self-contained and can be restored later.")

    def _on_decoded(self, info: dict) -> None:
        self.restored.set_path(str(info["output_path"]))
        self.metric_labels["decode"].setText(f"{float(info['decode_seconds']):.3f} s")
        self.metric_labels["device"].setText(str(info["device"]))
        self.path_label.setText(str(info["output_path"]))
        self.status.setText("Image restored successfully.")

    def _on_failed(self, message: str) -> None:
        self.status.setText("Operation failed")
        QMessageBox.critical(self, "MTRSA", message)

    def _on_cancelled(self) -> None:
        self.status.setText("Operation cancelled")

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Cancelling…")
            self.cancel_btn.setEnabled(False)

    def _worker_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.open_image_btn, self.open_bitstream_btn, self.compress_btn, self.decode_btn, self.preset, self.device, self.steps):
            widget.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)


def _human_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"


STYLE = """
QMainWindow, QWidget { background: #111318; color: #e8ebf0; font-size: 13px; }
#appTitle { font-size: 24px; font-weight: 700; }
#subTitle { color: #9ca6b5; }
#controlBar, #metricsCard, #imageCard { background: #191d24; border: 1px solid #2a303b; border-radius: 10px; }
#imageTitle { font-weight: 600; color: #cfd6e2; }
#imageSurface { background: #0c0e12; border-radius: 6px; color: #687386; }
QPushButton { background: #252b35; border: 1px solid #343c49; padding: 8px 12px; border-radius: 6px; }
QPushButton:hover { background: #303745; }
QPushButton:disabled { color: #667080; background: #1b1f26; }
#primaryButton { background: #3158d4; border-color: #4168e2; font-weight: 600; }
#primaryButton:hover { background: #3a63e6; }
QComboBox, QSpinBox { background: #0f1217; border: 1px solid #343c49; padding: 6px; border-radius: 5px; min-width: 78px; }
QProgressBar { background: #0d1015; border: 1px solid #343c49; border-radius: 5px; text-align: center; }
QProgressBar::chunk { background: #3158d4; border-radius: 4px; }
"""
