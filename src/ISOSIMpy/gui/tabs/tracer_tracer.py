"""Tab for tracer-tracer sweep configuration and visualization."""

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import QSize, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TracerTracerTab(QWidget):
    """UI for running tracer-tracer sweeps and plotting results."""

    sweep_requested = pyqtSignal(float, float, int, str)

    def __init__(self, state, registry, parent=None):
        super().__init__(parent)
        self.state = state
        self.registry = registry
        self._param_keys = []
        self._date_indices = []
        self._timestamps = None
        self._observations = None
        self._is_running = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        lbl_sweep = QLabel("Tracer-Tracer Sweep")
        lbl_sweep.setStyleSheet("font-weight: 600;")
        outer.addWidget(lbl_sweep)

        row_param = QHBoxLayout()
        lbl_param = QLabel("Mean Travel Time Parameter:")
        row_param.addWidget(lbl_param)

        self.cb_param = QComboBox(self)
        self.cb_param.setFixedSize(QSize(260, 22))
        row_param.addWidget(self.cb_param)
        row_param.addStretch(1)
        outer.addLayout(row_param)

        row_range = QHBoxLayout()
        lbl_start = QLabel("Start:")
        row_range.addWidget(lbl_start)

        self.sb_start = QDoubleSpinBox(self)
        self.sb_start.setDecimals(2)
        self.sb_start.setRange(1.0, 5000.0)
        self.sb_start.setValue(60.0)
        self.sb_start.setSingleStep(12.0)
        row_range.addWidget(self.sb_start)

        lbl_stop = QLabel("Stop:")
        row_range.addWidget(lbl_stop)

        self.sb_stop = QDoubleSpinBox(self)
        self.sb_stop.setDecimals(2)
        self.sb_stop.setRange(1.0, 20000.0)
        self.sb_stop.setValue(480.0)
        self.sb_stop.setSingleStep(12.0)
        row_range.addWidget(self.sb_stop)

        lbl_count = QLabel("Points:")
        row_range.addWidget(lbl_count)

        self.sb_count = QSpinBox(self)
        self.sb_count.setRange(2, 400)
        self.sb_count.setValue(21)
        row_range.addWidget(self.sb_count)

        self.lbl_units = QLabel(self._unit_label_text())
        self.lbl_units.setStyleSheet("color: #666;")
        row_range.addWidget(self.lbl_units)

        row_range.addStretch(1)
        outer.addLayout(row_range)

        self.btn_run = QPushButton("Run Sweep", self)
        self.btn_run.setFixedSize(QSize(200, 40))
        self.btn_run.clicked.connect(self._on_run_clicked)
        outer.addWidget(self.btn_run)

        outer.addStretch(1)

        lbl_obs = QLabel("Observation Selection")
        lbl_obs.setStyleSheet("font-weight: 600;")
        outer.addWidget(lbl_obs)

        row_obs = QHBoxLayout()
        self.cb_date = QComboBox(self)
        self.cb_date.setFixedSize(QSize(260, 22))
        self.cb_date.currentIndexChanged.connect(self._on_date_changed)
        row_obs.addWidget(self.cb_date)
        row_obs.addStretch(1)
        outer.addLayout(row_obs)

        self.btn_plot = QPushButton("Plot Tracer-Tracer", self)
        self.btn_plot.setFixedSize(QSize(200, 40))
        self.btn_plot.clicked.connect(self._plot)
        outer.addWidget(self.btn_plot)

        outer.addStretch(2)

        self._refresh_param_choices()
        self.reset_results()
        self.notify_sweep_finished()

    def _unit_label_text(self) -> str:
        if getattr(self.state, "is_monthly", True):
            return "Units: months"
        return "Units: years"

    def refresh(self) -> None:
        """Refresh parameter choices and enable states."""
        self.lbl_units.setText(self._unit_label_text())
        self._refresh_param_choices()
        self._update_enable_state()

    def reset_results(self) -> None:
        """Clear cached sweep outputs and disable plotting controls."""
        self._date_indices = []
        self._timestamps = None
        self._observations = None
        self._is_running = False
        self.cb_date.clear()
        self.cb_date.setEnabled(False)
        self.btn_plot.setEnabled(False)

    def handle_tracer_tracer_ready(self, payload) -> None:
        """Populate observation selectors after a sweep finished."""
        if payload is None:
            return
        self._timestamps = payload.get("timestamps")
        self._observations = payload.get("observations")
        indices = payload.get("obs_indices")
        self._date_indices = (
            list(int(i) for i in np.asarray(indices, dtype=int)) if indices is not None else []
        )

        self.cb_date.clear()
        if self._timestamps is not None:
            for idx in self._date_indices:
                label = self._format_timestamp(self._timestamps[idx])
                self.cb_date.addItem(label, idx)

        has_entries = self.cb_date.count() > 0
        self.cb_date.setEnabled(has_entries)
        if has_entries:
            self.cb_date.setCurrentIndex(0)
        self.btn_plot.setEnabled(has_entries)
        self.notify_sweep_finished()

    def _refresh_param_choices(self) -> None:
        """Fill parameter combo box with available mean travel time keys."""
        self.cb_param.blockSignals(True)
        self.cb_param.clear()
        self._param_keys = []

        instances = list(getattr(self.state, "design_instances", []) or [])
        for inst in instances:
            unit_name = inst.get("name")
            prefix = inst.get("prefix")
            if not unit_name or unit_name not in self.registry:
                continue
            cls = self.registry[unit_name]
            for meta in getattr(cls, "PARAMS", []):
                if meta.get("key") != "mtt":
                    continue
                param_key = f"{prefix}.{meta['key']}"
                label = meta.get("label", meta["key"])
                display = f"{unit_name} ({prefix}) - {label}"
                self.cb_param.addItem(display, param_key)
                self._param_keys.append(param_key)

        self.cb_param.blockSignals(False)
        if self._param_keys:
            self.cb_param.setCurrentIndex(0)

    def _on_run_clicked(self) -> None:
        if not self._param_keys:
            QMessageBox.information(self, "Tracer-Tracer", "No mean travel time parameter found.")
            return
        start = float(self.sb_start.value())
        stop = float(self.sb_stop.value())
        if stop <= start:
            QMessageBox.information(
                self, "Tracer-Tracer", "Stop value must be greater than start value."
            )
            return
        param_key = self.cb_param.currentData()
        if not param_key:
            QMessageBox.information(self, "Tracer-Tracer", "Select a parameter to sweep.")
            return
        count = int(self.sb_count.value())
        self._is_running = True
        self.btn_run.setEnabled(False)
        self.sweep_requested.emit(start, stop, count, param_key)

    def notify_sweep_finished(self) -> None:
        """Re-enable sweep button after controller finished processing."""
        self._is_running = False
        self._update_enable_state()

    def sweep_failed(self) -> None:
        """Ensure sweep button is re-enabled after an error."""
        self._is_running = False
        self._update_enable_state()

    def _on_date_changed(self, index: int) -> None:
        self.btn_plot.setEnabled(
            index >= 0 and index < len(self._date_indices) and self.state.tt_results is not None
        )

    def _update_enable_state(self) -> None:
        has_two_tracers = self._has_dual_tracers()
        has_param = bool(self._param_keys)
        can_run = (
            has_two_tracers
            and has_param
            and self.state.input_series is not None
            and self.state.target_series is not None
        )
        self.cb_param.setEnabled(has_param)
        self.btn_run.setEnabled(can_run and not self._is_running)
        has_results = self.state.tt_results is not None and self.state.tt_mtt_values is not None
        enable_plot = has_results and self.cb_date.currentIndex() >= 0 and self.cb_date.count() > 0
        self.btn_plot.setEnabled(enable_plot)

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

    def _format_timestamp(self, ts) -> str:
        try:
            if hasattr(ts, "strftime"):
                fmt = "%Y-%m" if getattr(self.state, "is_monthly", True) else "%Y"
                return ts.strftime(fmt)
        except Exception:
            pass
        return str(ts)

    def _plot(self) -> None:
        if self.state.tt_results is None or self.state.tt_mtt_values is None:
            QMessageBox.information(self, "Tracer-Tracer", "Run a sweep before plotting.")
            return
        if not self._date_indices or self.cb_date.currentIndex() < 0:
            QMessageBox.information(self, "Tracer-Tracer", "Select an observation date.")
            return

        obs_idx = self._date_indices[self.cb_date.currentIndex()]
        results = np.asarray(self.state.tt_results, dtype=float)
        if results.shape[0] < 2:
            QMessageBox.information(
                self, "Tracer-Tracer", "Results must contain two tracers to plot."
            )
            return

        x = results[0, :, obs_idx]
        y = results[1, :, obs_idx]
        mtt_values = np.asarray(self.state.tt_mtt_values, dtype=float)
        scale = 12.0 if getattr(self.state, "is_monthly", True) else 1.0
        mtt_years = mtt_values / scale

        obs_vals = None
        if self._observations is not None and len(self._observations) > obs_idx:
            obs_array = np.asarray(self._observations, dtype=float)
            if obs_array.ndim == 1:
                obs_array = obs_array.reshape(-1, 1)
            if obs_array.shape[0] > obs_idx and obs_array.shape[1] >= 2:
                obs_vals = obs_array[obs_idx, :2]

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(x, y, c="black", lw=2.0)
        scatter = ax.scatter(x, y, c=mtt_years, edgecolor="k", s=60, zorder=10)

        if obs_vals is not None and not np.isnan(obs_vals).any():
            ax.scatter(
                obs_vals[0],
                obs_vals[1],
                c="r",
                edgecolor="k",
                marker="X",
                s=200,
                zorder=20,
                label="Observation",
            )

        ax.scatter([0.0], [0.0], c="k", edgecolor="k", marker="o", s=20, zorder=15, label="Origin")

        step = max(1, len(mtt_values) // 10)
        for idx in range(0, len(mtt_values), step):
            label = "Binary Mixing" if idx == 0 else None
            ax.plot([0.0, x[idx]], [0.0, y[idx]], c="k", lw=1.0, ls="--", alpha=0.3, label=label)

        for frac in (0.75, 0.5, 0.25):
            label = f"{int((1 - frac) * 100)}% Tracer-Free Water"
            ax.plot(x * frac, y * frac, c="k", lw=1.0, alpha=0.8 * frac, label=label)

        plt.colorbar(scatter, ax=ax, label="Mean residence time [years]")

        tracer1 = getattr(self.state, "tracer1", "Tracer 1")
        tracer2 = getattr(self.state, "tracer2", "Tracer 2") or "Tracer 2"
        ax.set_xlabel(tracer1 or "Tracer 1")
        ax.set_ylabel(tracer2)

        if self._timestamps is not None and len(self._timestamps) > obs_idx:
            ax.set_title(
                f"Date of Observation: {self._format_timestamp(self._timestamps[obs_idx])}"
            )

        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        plt.show()
