"""Application entry point for the PyTracerLab GUI."""

import os
import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication

from .main_window import MainWindow

# make high dpi adjustments
QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"


def main():
    """Launch the Qt application and show the main window."""
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
