from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# Development convenience: running directly from a repository checkout works
# even before `pip install -e v3`.
REPO_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = REPO_ROOT / "v3"
if V3_ROOT.exists() and str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from v3_gui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MTRSA GUI")
    app.setApplicationVersion("3.1.0")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
