"""Tab for frequency/tracer selection and CSV input loading."""

from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QSize, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..database import Tracers


def _current_tracer_names(state) -> List[str]:
    """Return tracer names selected in the shared state."""
    names: List[str] = []
    t1 = getattr(state, "tracer1", None)
    if t1 and str(t1).lower() not in {"", "none"}:
        names.append(t1)
    t2 = getattr(state, "tracer2", None)
    if t2 and str(t2).lower() not in {"", "none"}:
        names.append(t2)
    if not names:
        t_fallback = getattr(state, "tracer", None)
        if t_fallback:
            names.append(t_fallback)
    return names or ["Tracer 1"]


def _format_timestamp(ts: datetime, monthly: bool) -> str:
    fmt = "%Y-%m" if monthly else "%Y"
    return ts.strftime(fmt)


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
        self._manual_target_active = False
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
        self.cb_t1.addItems(["None"] + list(Tracers.tracer_data.keys()))  # use None initially
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
        b_manual = QPushButton("Manual Observation Input")
        b_manual.setFixedSize(QSize(200, 40))
        b_manual.clicked.connect(self._open_manual_observations)
        # Set title, add widgets
        lbl = QLabel("Input and Observation Series")
        lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(lbl)
        lay.addWidget(b_in)
        lay.addWidget(self.lbl_in)
        btn_row = QHBoxLayout()
        btn_row.addWidget(b_tg)
        btn_row.addWidget(b_manual)
        btn_row.addStretch()
        lay.addLayout(btn_row)
        lay.addWidget(self.lbl_tg)

        # Signal connections
        self.monthly.toggled.connect(self._freq_changed)
        self.cb_t1.currentTextChanged.connect(self._tracer_changed)
        self.cb_t2.currentTextChanged.connect(self._tracer_changed)

    def _freq_changed(self, checked):
        self.state.is_monthly = checked
        cleared = self._clear_loaded_series()
        if cleared:
            frequency = "monthly" if checked else "yearly"
            QMessageBox.information(
                self,
                "Temporal Resolution Changed",
                "Loaded input and observation data were cleared after changing "
                f"the temporal resolution to {frequency}. Please reload compatible data.",
            )
        self.changed.emit()

    def _tracer_changed(self):
        self.state.tracer1 = self.cb_t1.currentText()
        t2 = self.cb_t2.currentText()
        self.state.tracer2 = None if t2 == "None" else t2
        self._clear_manual_observations()
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
            # try reading with comma separator
            try:
                r = csv.reader(f, delimiter=";")
                try:
                    r = csv.reader(f, delimiter=",")
                except Exception:
                    pass
            except Exception:
                # still continue
                pass
            # skip header
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
                            # optionally replace commas with decimal points
                            if "," in tok:
                                tok = tok.replace(",", ".")
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

    def _clear_manual_observations(self) -> None:
        """Reset manual observations when context changes."""
        if not self.state.manual_observations and not self._manual_target_active:
            return
        self.state.manual_observations.clear()
        if self._manual_target_active:
            self.state.target_series = None
            self.lbl_tg.setText("No observation series selected")
        self._manual_target_active = False

    def _clear_loaded_series(self) -> bool:
        """Clear loaded input/output data after a frequency change."""
        had_input = self.state.input_series is not None
        had_target = (
            self.state.target_series is not None
            or bool(self.state.manual_observations)
            or self._manual_target_active
        )
        if not had_input and not had_target:
            return False

        self.state.input_series = None
        self.state.target_series = None
        self.state.manual_observations.clear()
        self._manual_target_active = False
        self.lbl_in.setText("No input series selected")
        self.lbl_tg.setText("No observation series selected")
        return True

    def _active_tracer_names(self) -> List[str]:
        """Return the currently selected tracers for manual observation input."""
        return _current_tracer_names(self.state)

    def _rebuild_manual_target_series(self) -> bool:
        """Convert manual observations into the ``state.target_series`` tuple."""
        if not self.state.manual_observations:
            self.state.target_series = None
            self._manual_target_active = False
            return False
        if not self.state.input_series or len(self.state.input_series[0]) == 0:
            return False
        times = self.state.input_series[0]
        tracer_names = self._active_tracer_names()
        n_tracers = max(1, len(tracer_names))
        obs = np.full((len(times), n_tracers), np.nan, dtype=float)
        for idx, ts in enumerate(times):
            vals = self.state.manual_observations.get(ts)
            if not vals:
                continue
            for col in range(n_tracers):
                if col < len(vals):
                    obs[idx, col] = float(vals[col])
        self.state.target_series = (times.copy(), obs)
        self._manual_target_active = True
        return True

    def _manual_observations_updated(self) -> None:
        """Update labels/state when manual observations are changed."""
        if not self.state.manual_observations:
            self._clear_manual_observations()
            self.changed.emit()
            return
        if not self._rebuild_manual_target_series():
            QMessageBox.warning(self, "Missing Input Series", "Load an input series first.")
            return
        count = len(self.state.manual_observations)
        label = f"Manual observations ({count} entries)"
        self.lbl_tg.setText(label)
        self.changed.emit()

    def _open_manual_observations(self):
        if not self.state.input_series or len(self.state.input_series[0]) == 0:
            QMessageBox.warning(
                self,
                "No Input Series",
                "Please load an input series before adding manual observations.",
            )
            return
        dlg = ManualObservationDialog(self.state, self)
        dlg.observations_changed.connect(self._manual_observations_updated)
        dlg.exec_()

    def _open_input(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Open Input Series CSV", "", "CSV Files (*.csv)"
        )
        if file:
            self.state.input_series = self._read_csv(file, self.state.is_monthly)
            self._clear_manual_observations()
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
            self.state.manual_observations.clear()
            self._manual_target_active = False
            shape = self.state.target_series[1].shape if self.state.target_series else None
            cols = shape[1] if (shape is not None and len(shape) == 2) else 1
            self.lbl_tg.setText(f"Loaded: {file} (columns: {cols})")
            self.changed.emit()


class ManualObservationDialog(QDialog):
    """Dialog for managing manual observation entries."""

    observations_changed = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Observation Input")
        self.state = state
        self._monthly = getattr(state, "is_monthly", True)
        self._timestamps = list(state.input_series[0]) if state.input_series else []
        self._tracer_names = _current_tracer_names(state)

        layout = QVBoxLayout(self)

        headers = ["Date"] + self._tracer_names
        self._table = QTableWidget(0, len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Observation")
        add_btn.clicked.connect(self._add_observation)
        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh_table()

    def _refresh_table(self) -> None:
        manual = self.state.manual_observations
        rows = sorted(
            ((ts, manual[ts]) for ts in manual if ts in self._timestamps),
            key=lambda item: item[0],
        )
        self._table.setRowCount(len(rows))
        for row_idx, (ts, values) in enumerate(rows):
            date_item = QTableWidgetItem(_format_timestamp(ts, self._monthly))
            self._table.setItem(row_idx, 0, date_item)
            for col_idx, tracer in enumerate(self._tracer_names, start=1):
                val = values[col_idx - 1] if col_idx - 1 < len(values) else np.nan
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    text = ""
                else:
                    text = f"{float(val):g}"
                item = QTableWidgetItem(text)
                self._table.setItem(row_idx, col_idx, item)

    def _add_observation(self) -> None:
        dlg = ObservationEntryDialog(self._timestamps, self._tracer_names, self._monthly, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        timestamp, values = dlg.result()
        self.state.manual_observations[timestamp] = values
        self._refresh_table()
        self.observations_changed.emit()


class ObservationEntryDialog(QDialog):
    """Dialog that collects a timestamp and tracer concentrations from the user."""

    def __init__(self, timestamps, tracer_names, monthly, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Observation")
        self._timestamps = timestamps
        self._monthly = monthly
        self._tracer_names = tracer_names
        self._values: List[float] = []
        self._selected_timestamp: Optional[datetime] = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._timestamp_box = QComboBox()
        for ts in self._timestamps:
            self._timestamp_box.addItem(_format_timestamp(ts, self._monthly), ts)
        form.addRow("Timestamp", self._timestamp_box)

        self._value_edits: List[QLineEdit] = []
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-1e100, 1e100, 12)
        for name in self._tracer_names:
            line = QLineEdit()
            line.setValidator(validator)
            line.setPlaceholderText("Enter concentration")
            form.addRow(name, line)
            self._value_edits.append(line)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if not self._timestamps:
            QMessageBox.warning(self, "No Timestamps", "No timestamps available for input.")
            return
        values: List[float] = []
        for edit, tracer_name in zip(self._value_edits, self._tracer_names):
            text = edit.text().strip()
            if text == "":
                QMessageBox.warning(
                    self,
                    "Missing Value",
                    f"Please enter a concentration for {tracer_name}.",
                )
                return
            try:
                values.append(float(text))
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Value",
                    f"Could not parse the value for {tracer_name}.",
                )
                return
        self._values = values
        self._selected_timestamp = self._timestamp_box.currentData()
        self.accept()

    def result(self) -> Tuple[datetime, List[float]]:
        if self._selected_timestamp is None:
            raise RuntimeError("Dialog accepted without selecting a timestamp.")
        return self._selected_timestamp, self._values
