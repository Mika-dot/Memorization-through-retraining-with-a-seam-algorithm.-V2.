from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class ImageView(QFrame):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setObjectName("imageCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title = QLabel(title)
        self.title.setObjectName("imageTitle")
        self.image = QLabel("No image")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(260, 220)
        self.image.setObjectName("imageSurface")
        layout.addWidget(self.title)
        layout.addWidget(self.image, 1)

    def clear(self, text: str = "No image") -> None:
        self._pixmap = None
        self.image.setPixmap(QPixmap())
        self.image.setText(text)

    def set_path(self, path: str) -> None:
        pix = QPixmap(path)
        if pix.isNull():
            self.clear("Unable to preview")
            return
        self._pixmap = pix
        self.image.setText("")
        self._refresh()

    def set_rgb_array(self, rgb: np.ndarray) -> None:
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.image.setText("")
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap is None:
            return
        size = self.image.size()
        self.image.setPixmap(self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
