"""Tab to run simulations/calibrations and visualize results."""

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...model import solver as ms
from .solver_params import SolverParamsDialog


class SimulationTab(QWidget):
    """Tab to select solvers, run simulation/calibration, and plot results.

    Signals
    -------
    simulate_requested
        Trigger a forward simulation using the current model.
    calibrate_requested
        Trigger running the selected solver to calibrate parameters.
    report_requested : str
        Trigger writing a report to the provided file path.
    """

    simulate_requested = pyqtSignal()
    calibrate_requested = pyqtSignal()
    report_requested = pyqtSignal(str)  # carries the file path

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        lay = QVBoxLayout(self)

        # Solver selection row
        row = QHBoxLayout()
        lbl_solver = QLabel("Solver:")
        row.addWidget(lbl_solver)
        self.cb_solver = QComboBox()
        self._solver_keys = []
        for key, name in ms.available_solvers():
            self.cb_solver.addItem(name, userData=key)
            self._solver_keys.append(key)
        # default selection from state
        try:
            idx = self._solver_keys.index(getattr(self.state, "solver_key", "de"))
        except ValueError:
            idx = 0
        self.cb_solver.setCurrentIndex(idx)
        self.cb_solver.currentIndexChanged.connect(self._on_solver_changed)
        row.addWidget(self.cb_solver)
        row.addStretch(1)

        # Simulation label
        lbl_sim = QLabel("Simulation")
        lbl_sim.setStyleSheet("font-weight: 600;")

        # Simulation button
        b_sim = QPushButton("Run Simulation")
        b_sim.clicked.connect(self.simulate_requested)

        # Calibration label
        lbl_cal = QLabel("Calibration")
        lbl_cal.setStyleSheet("font-weight: 600;")

        # Edit solver parameters button
        b_edit = QPushButton("Edit Solver Parameters")
        b_edit.clicked.connect(self._edit_solver_params)

        # Calibration button
        b_cal = QPushButton("Run Calibration")
        b_cal.clicked.connect(self.calibrate_requested)

        # Plotting label
        lbl_plot = QLabel("Plotting")
        lbl_plot.setStyleSheet("font-weight: 600;")

        # Plot button
        b_plot = QPushButton("Plot Results")
        b_plot.clicked.connect(self._plot)

        # Report label
        lbl_report = QLabel("Report")
        lbl_report.setStyleSheet("font-weight: 600;")

        # Report button
        b_report = QPushButton("Write Report")
        b_report.clicked.connect(self._choose_report_file)

        # Add widgets
        lay.addWidget(lbl_sim)
        lay.addWidget(b_sim)
        lay.addStretch(1)
        lay.addWidget(lbl_cal)
        lay.addLayout(row)
        lay.addWidget(b_edit)
        lay.addWidget(b_cal)
        lay.addStretch(1)
        lay.addWidget(lbl_plot)
        lay.addWidget(b_plot)
        lay.addStretch(1)
        lay.addWidget(lbl_report)
        lay.addWidget(b_report)

    def _on_solver_changed(self, idx: int):
        key = self.cb_solver.itemData(idx)
        if key is None:
            return
        self.state.solver_key = key

    def _edit_solver_params(self):
        dlg = SolverParamsDialog(self.state, self)
        dlg.exec_()

    def _choose_report_file(self):
        """Open save dialog and emit signal with chosen filename."""
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report",
            "report.txt",  # default filename
            "Text Files (*.txt);;All Files (*)",  # filters
        )
        if fname:  # user clicked OK
            self.report_requested.emit(fname)

    def show_results(self, result):
        """Accept either a plain ndarray or a standardized payload dict."""
        if isinstance(result, np.ndarray):
            payload = {"solver": "sim", "sim": result, "envelope": None}
        else:
            payload = result
        self.state.last_simulation = payload

    def show_status(self, msg):
        """Display a non-blocking status message dialog."""
        QMessageBox.information(self, "Status", msg)

    def _plot(self):
        if self.state.last_simulation is None or self.state.input_series is None:
            QMessageBox.information(self, "Plot", "Nothing to plot yet.")
            return
        times = self.state.input_series[0]
        obs = self.state.target_series[1] if self.state.target_series else None
        fig, ax = plt.subplots(figsize=(8, 4))

        # Observations
        if obs is not None:
            ax.scatter(times, obs, label="Observations", marker="x", zorder=100, c="red")

        payload = self.state.last_simulation
        if isinstance(payload, dict):
            sim = payload.get("sim")
            env_1_99 = payload.get("envelope_1_99")
            env_20_80 = payload.get("envelope_20_80")
            label = "Simulation"
            if payload.get("solver") == "mcmc":
                label = "Median simulation"
                if (
                    isinstance(env_1_99, dict)
                    and env_1_99.get("low") is not None
                    and env_1_99.get("high") is not None
                ):
                    ax.fill_between(
                        times,
                        np.asarray(env_1_99["low"], dtype=float),
                        np.asarray(env_1_99["high"], dtype=float),
                        color="0.7",
                        alpha=0.4,
                        label="1–99% percentile",
                    )
                if (
                    isinstance(env_20_80, dict)
                    and env_20_80.get("low") is not None
                    and env_20_80.get("high") is not None
                ):
                    ax.fill_between(
                        times,
                        np.asarray(env_20_80["low"], dtype=float),
                        np.asarray(env_20_80["high"], dtype=float),
                        color="0.7",
                        alpha=0.7,
                        label="20–80% percentile",
                    )
            if sim is not None:
                ax.plot(times, np.asarray(sim, dtype=float), label=label, c="k")
        else:
            # Legacy: just a single simulation array
            ax.plot(times, np.asarray(payload, dtype=float), label="Simulation", c="k")

        ax.set_xlabel("Time")
        ax.set_ylabel("Tracer Concentration")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        plt.show()
