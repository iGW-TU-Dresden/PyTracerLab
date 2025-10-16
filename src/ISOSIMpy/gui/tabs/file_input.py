"""Tab for frequency/tracer selection and CSV input loading."""

from datetime import datetime

import numpy as np
from PyQt5.QtCore import QSize, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..database import Tracers


class FileInputTab(QWidget):
    """Tab to choose frequency, tracer, and load input/observation CSV files.

    Signals
    -------
    changed
        Emitted whenever user selections or loaded files change.
    """

    # Define signals that this tab can emit
    changed = pyqtSignal()

    def __init__(self, state, parent=None):
        """Create the tab and wire UI controls to update ``state``."""
        super().__init__(parent)
        self.state = state
        lay = QVBoxLayout(self)

        ### Temporal reolution selection radio buttons
        self.monthly = QRadioButton("Monthly")
        self.monthly.setChecked(True)
        self.yearly = QRadioButton("Yearly")
        g1 = QButtonGroup(self)
        g1.addButton(self.monthly)
        g1.addButton(self.yearly)
        # Set title, add widgets
        lbl = QLabel("Frequency")
        lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(lbl)
        lay.addWidget(self.monthly)
        lay.addWidget(self.yearly)

        ### Tracer Selection dropdowns (dual-tracer)
        lbl = QLabel("Tracers")
        lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(lbl)
        self.cb_t1 = QComboBox()
        self.cb_t1.setFixedSize(QSize(200, 20))
        self.cb_t1.addItems(list(Tracers.tracer_data.keys()))  # primary cannot be None
        self.cb_t2 = QComboBox()
        self.cb_t2.setFixedSize(QSize(200, 20))
        self.cb_t2.addItems(
            ["None"] + list(Tracers.tracer_data.keys())  # use None to indicate no second tracer
        )  # optional second tracer
        lay.addWidget(QLabel("Tracer 1"))
        lay.addWidget(self.cb_t1)
        lay.addWidget(QLabel("Tracer 2"))
        lay.addWidget(self.cb_t2)

        ### Input and target series selection buttons
        self.lbl_in = QLabel("No input series selected")
        self.lbl_tg = QLabel("No observation series selected")
        b_in = QPushButton("Select Input CSV")
        b_in.setFixedSize(QSize(200, 40))
        b_in.clicked.connect(self._open_input)
        b_tg = QPushButton("Select Observation CSV")
        b_tg.setFixedSize(QSize(200, 40))
        b_tg.clicked.connect(self._open_target)
        # Set title, add widgets
        lbl = QLabel("Input and Observation Series")
        lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(lbl)
        lay.addWidget(b_in)
        lay.addWidget(self.lbl_in)
        lay.addWidget(b_tg)
        lay.addWidget(self.lbl_tg)

        # Signal connections
        self.monthly.toggled.connect(self._freq_changed)
        self.cb_t1.currentTextChanged.connect(self._tracer_changed)
        self.cb_t2.currentTextChanged.connect(self._tracer_changed)

    def _freq_changed(self, checked):
        self.state.is_monthly = checked
        self.changed.emit()

    def _tracer_changed(self):
        self.state.tracer1 = self.cb_t1.currentText()
        t2 = self.cb_t2.currentText()
        self.state.tracer2 = None if t2 == "None" else t2
        # legacy compatibility
        self.state.tracer = self.state.tracer1
        self.changed.emit()

    def _read_csv(self, path, monthly=True):
        """Read a CSV with first column timestamps and remaining one or two tracer columns.

        Parameters
        ----------
        path : str
            File path to read.
        monthly : bool
            Interpret timestamp format as ``"%Y-%m"`` if ``True`` else ``"%Y"``.

        Returns
        -------
        tuple(ndarray, ndarray)
            Parsed datetimes and corresponding float matrix of shape (N, K).
        """
        import csv

        fmt = "%Y-%m" if monthly else "%Y"
        times = []
        rows: list[list[float]] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            _ = next(r, None)  # skip header
            for rec in r:
                if not rec:
                    continue
                try:
                    t = datetime.strptime(rec[0].strip(), fmt)
                except Exception:
                    # skip malformed timestamps
                    continue
                times.append(t)
                vals = []
                for tok in rec[1:]:
                    tok = tok.strip()
                    if tok == "" or tok.lower() == "nan":
                        vals.append(np.nan)
                    else:
                        try:
                            vals.append(float(tok))
                        except Exception:
                            vals.append(np.nan)
                rows.append(vals)
        if len(times) == 0:
            return np.array([]), np.array([])
        # Ensure at least one value column
        max_cols = max((len(r) for r in rows), default=0)
        if max_cols == 0:
            vals_arr = np.full((len(times), 1), np.nan)
        else:
            # Pad rows to equal length
            vals_arr = np.array([ri + [np.nan] * (max_cols - len(ri)) for ri in rows], dtype=float)
        return np.array(times, dtype=object), vals_arr

    def _open_input(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Open Input Series CSV", "", "CSV Files (*.csv)"
        )
        if file:
            self.state.input_series = self._read_csv(file, self.state.is_monthly)
            shape = self.state.input_series[1].shape if self.state.input_series else None
            cols = shape[1] if (shape is not None and len(shape) == 2) else 1
            self.lbl_in.setText(f"Loaded: {file} (columns: {cols})")
            self.changed.emit()

    def _open_target(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Open Observation Series CSV", "", "CSV Files (*.csv)"
        )
        if file:
            self.state.target_series = self._read_csv(file, self.state.is_monthly)
            shape = self.state.target_series[1].shape if self.state.target_series else None
            cols = shape[1] if (shape is not None and len(shape) == 2) else 1
            self.lbl_tg.setText(f"Loaded: {file} (columns: {cols})")
            self.changed.emit()
