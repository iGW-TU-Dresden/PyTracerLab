"""Application entry point for the PyTracerLab GUI."""

import sys

from PyQt5.QtWidgets import QApplication

from .main_window import MainWindow


def main():
    """Launch the Qt application and show the main window."""
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
