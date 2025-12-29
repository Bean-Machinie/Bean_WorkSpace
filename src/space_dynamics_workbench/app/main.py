from __future__ import annotations

import logging
import sys

from PySide6 import QtWidgets

from .window import MainWindow


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
