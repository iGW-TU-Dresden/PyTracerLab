"""Tab to run simulations/calibrations and visualize results."""

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import QSize, pyqtSignal
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
    savedata_requested : str
        Trigger saving simulation data to the provided file path.
    """

    # Define signals that this tab can emit
    simulate_requested = pyqtSignal()
    calibrate_requested = pyqtSignal()
    report_requested = pyqtSignal(str)  # carries the file path
    savedata_requested = pyqtSignal(str)  # carries the file path

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        lay = QVBoxLayout(self)

        # Solver selection row
        row = QHBoxLayout()
        lbl_solver = QLabel("Solver:")
        row.addWidget(lbl_solver)
        self.cb_solver = QComboBox()
        self.cb_solver.setFixedSize(QSize(200, 20))
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
        b_sim.setFixedSize(QSize(200, 40))
        # b_sim.setStyleSheet("background-color: rgb(140, 230, 170)")
        b_sim.clicked.connect(self.simulate_requested)

        # Calibration label
        lbl_cal = QLabel("Calibration")
        lbl_cal.setStyleSheet("font-weight: 600;")

        # Edit solver parameters button
        b_edit = QPushButton("Edit Solver Parameters")
        b_edit.setFixedSize(QSize(200, 40))
        # b_edit.setStyleSheet("background-color: rgb(0, 125, 75); color: white")
        b_edit.clicked.connect(self._edit_solver_params)

        # Calibration button
        b_cal = QPushButton("Run Calibration")
        b_cal.setFixedSize(QSize(200, 40))
        # b_cal.setStyleSheet("background-color: rgb(140, 230, 170)")
        b_cal.clicked.connect(self.calibrate_requested)

        # Plotting label
        lbl_plot = QLabel("Plotting")
        lbl_plot.setStyleSheet("font-weight: 600;")

        # Plot button
        b_plot = QPushButton("Plot Results")
        b_plot.setFixedSize(QSize(200, 40))
        b_plot.clicked.connect(self._plot)

        # Report label
        lbl_report = QLabel("Report")
        lbl_report.setStyleSheet("font-weight: 600;")

        # Report button and save simulation data button
        b_report = QPushButton("Write Report")
        b_report.setFixedSize(QSize(200, 40))
        b_report.clicked.connect(self._choose_report_file)
        b_savedata = QPushButton("Save Simulation Data")
        b_savedata.setFixedSize(QSize(200, 40))
        b_savedata.clicked.connect(self._choose_savedata_file)

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
        btn_row = QHBoxLayout()
        btn_row.addWidget(b_report)
        btn_row.addWidget(b_savedata)
        btn_row.addStretch()
        lay.addLayout(btn_row)

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

    def _choose_savedata_file(self):
        """Open save dialog and emit signal with chosen filename."""
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save Simulation Data",
            "sim_data.txt",  # default filename
            "Text Files (*.txt);;All Files (*)",  # filters
        )
        if fname:  # user clicked OK
            self.savedata_requested.emit(fname)

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

        # Coerce obs to 2D for consistent handling
        if obs is not None:
            obs_arr = np.asarray(obs, dtype=float)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(-1, 1)
        else:
            obs_arr = None

        # Prepare axes: one subplot per tracer (up to 2 for now)
        n_tr = 1 if (obs_arr is None or obs_arr.shape[1] == 1) else obs_arr.shape[1]
        fig, axes = plt.subplots(n_tr, 1, figsize=(8, 3.5 * n_tr), sharex=True)
        if n_tr == 1:
            axes = [axes]

        # Observations per tracer
        if obs_arr is not None:
            for j in range(obs_arr.shape[1]):
                axes[j].scatter(
                    times,
                    obs_arr[:, j],
                    label="Observations",
                    marker="x",
                    zorder=100,
                    c="red",
                )

        payload = self.state.last_simulation
        if isinstance(payload, dict):
            sim = payload.get("sim")
            env_1_99 = payload.get("envelope_1_99")
            env_20_80 = payload.get("envelope_20_80")
            label = "Simulation"
            if payload.get("solver") == "mcmc" or payload.get("solver") == "dream":
                label = "Median simulation"
                # Handle envelopes for single or dual tracer
                if (
                    isinstance(env_1_99, dict)
                    and env_1_99.get("low") is not None
                    and env_1_99.get("high") is not None
                ):
                    low = np.asarray(env_1_99["low"], dtype=float)
                    high = np.asarray(env_1_99["high"], dtype=float)
                    if low.ndim == 1:
                        low = low.reshape(-1, 1)
                        high = high.reshape(-1, 1)
                    for j in range(low.shape[1]):
                        axes[j].fill_between(
                            times,
                            low[:, j],
                            high[:, j],
                            color="0.7",
                            alpha=0.4,
                            label="1–99% percentile",
                        )
                if (
                    isinstance(env_20_80, dict)
                    and env_20_80.get("low") is not None
                    and env_20_80.get("high") is not None
                ):
                    low = np.asarray(env_20_80["low"], dtype=float)
                    high = np.asarray(env_20_80["high"], dtype=float)
                    if low.ndim == 1:
                        low = low.reshape(-1, 1)
                        high = high.reshape(-1, 1)
                    for j in range(low.shape[1]):
                        axes[j].fill_between(
                            times,
                            low[:, j],
                            high[:, j],
                            color="0.7",
                            alpha=0.7,
                            label="20–80% percentile",
                        )
            if sim is not None:
                sim_arr = np.asarray(sim, dtype=float)
                if sim_arr.ndim == 1:
                    sim_arr = sim_arr.reshape(-1, 1)
                for j in range(sim_arr.shape[1]):
                    axes[j].plot(times, sim_arr[:, j], label=label, c="k")
        else:
            # Legacy: just a single simulation array
            axes[0].plot(times, np.asarray(payload, dtype=float), label="Simulation", c="k")

        # Get tracer names for axes titles
        has_two_tracers = self._has_dual_tracers()
        tracer1 = getattr(self.state, "tracer1", "Tracer 1") or "Tracer 1"
        tracer_labels = [tracer1]
        if has_two_tracers:
            tracer2 = getattr(self.state, "tracer2", "Tracer 2") or "Tracer 2"
            tracer_labels.append(tracer2)

        for j, ax in enumerate(axes):
            ax.set_ylabel(tracer_labels[j])
            ax.set_yscale("log")
            ax.legend()
        axes[-1].set_xlabel("Time")
        fig.tight_layout()
        plt.show()

    def _has_dual_tracers(self) -> bool:
        tracer2 = getattr(self.state, "tracer2", None)
        if not tracer2:
            return False
        target = self.state.target_series
        if target is None or target[1] is None:
            return False
        obs = np.asarray(target[1], dtype=float)
        if obs.ndim == 1:
            return False
        return obs.shape[1] >= 2
