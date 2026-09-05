from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from mtr3.codec import EncodeCancelled, decode, encode


class EncodeWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, source: str, output: str, preset: str, device: str, steps: int | None = None) -> None:
        super().__init__()
        self.source = source
        self.output = output
        self.preset = preset
        self.device = device
        self.steps = steps
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _progress(self, done: int, total: int, loss: float) -> None:
        pct = int(round(100 * done / max(1, total)))
        self.progress.emit(max(0, min(100, pct)), f"Encoding {done}/{total} · loss {loss:.6f}")

    @Slot()
    def run(self) -> None:
        try:
            info = encode(
                self.source,
                self.output,
                preset=self.preset,
                device=self.device,
                steps=self.steps,
                quiet=True,
                progress_callback=self._progress,
                cancel_check=lambda: self._cancelled,
            )
            if self._cancelled:
                raise EncodeCancelled("Operation cancelled")

            preview = Path(tempfile.gettempdir()) / "mtrsa_gui_last_preview.png"
            dec = decode(self.output, preview, device=self.device, cancel_check=lambda: self._cancelled)
            info = dict(info)
            info["preview_path"] = str(preview)
            info["decode_seconds"] = dec["decode_seconds"]
            info["output_path"] = self.output
            self.progress.emit(100, "Done")
            self.finished.emit(info)
        except EncodeCancelled:
            self.cancelled.emit()
        except Exception as exc:  # GUI boundary
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DecodeWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, source: str, output: str, device: str) -> None:
        super().__init__()
        self.source = source
        self.output = output
        self.device = device
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _progress(self, done: int, total: int) -> None:
        pct = int(round(100 * done / max(1, total)))
        self.progress.emit(max(0, min(100, pct)), f"Decoding {pct}%")

    @Slot()
    def run(self) -> None:
        try:
            info = decode(
                self.source,
                self.output,
                device=self.device,
                progress_callback=self._progress,
                cancel_check=lambda: self._cancelled,
            )
            info = dict(info)
            info["output_path"] = self.output
            self.progress.emit(100, "Done")
            self.finished.emit(info)
        except EncodeCancelled:
            self.cancelled.emit()
        except Exception as exc:  # GUI boundary
            self.failed.emit(f"{type(exc).__name__}: {exc}")
