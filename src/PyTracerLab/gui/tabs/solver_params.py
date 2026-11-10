"""Solver parameter settings window."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QRadioButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..uncertainty import file_uncertainty
from .file_input import _current_tracer_names


def _int_edit(parent, lo: int, hi: int) -> QLineEdit:
    edit = QLineEdit()
    edit.setValidator(QIntValidator(lo, hi, parent))
    edit.setAlignment(Qt.AlignRight)
    return edit


def _float_edit(parent, lo: float, hi: float, decimals: int, scientific: bool) -> QLineEdit:
    edit = QLineEdit()
    val = QDoubleValidator(parent)
    val.setNotation(
        QDoubleValidator.ScientificNotation if scientific else QDoubleValidator.StandardNotation
    )
    val.setDecimals(decimals)
    val.setRange(lo, hi, decimals)
    edit.setValidator(val)
    edit.setAlignment(Qt.AlignRight)
    return edit


def _parse_float(edit: QLineEdit) -> Optional[float]:
    """Float value of ``edit`` (decimal point or comma), ``None`` if empty/invalid."""
    txt = edit.text().strip().replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return None


class SolverParamsDialog(QDialog):
    """Dialog for editing solver parameters and the measurement uncertainty.

    One tab per solver (DE, Least Squares, DREAM, MCMC) and, below the tabs,
    the measurement uncertainty of the observations, which applies to all
    solvers (see :mod:`PyTracerLab.gui.uncertainty`). The dialog reads current
    values from the state and writes back updated values on accept.
    """

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Solver Parameters")
        self.state = state

        tabs = QTabWidget()

        # DE tab
        de_box = QGroupBox("Differential Evolution")
        de_form = QFormLayout(de_box)
        self._de_widgets = {
            "maxiter": _int_edit(self, 1, 1_000_000),
            "popsize": _int_edit(self, 1, 10_000),
            "mutation_lo": _float_edit(self, 0.0, 3.0, 4, False),
            "mutation_hi": _float_edit(self, 0.0, 3.0, 4, False),
            "recombination": _float_edit(self, 0.0, 1.0, 4, False),
            "tol": _float_edit(self, 1e-12, 1.0, 9, False),
        }
        for label, key in [
            ("maxiter", "maxiter"),
            ("popsize", "popsize"),
            ("mutation low", "mutation_lo"),
            ("mutation high", "mutation_hi"),
            ("recombination", "recombination"),
            ("tol", "tol"),
        ]:
            de_form.addRow(label, self._de_widgets[key])

        # Least Squares tab
        lsq_box = QGroupBox("Least Squares")
        lsq_form = QFormLayout(lsq_box)
        self._lsq_widgets = {
            "ftol": _float_edit(self, 1e-12, 1.0, 12, True),
            "xtol": _float_edit(self, 1e-12, 1.0, 12, True),
            "gtol": _float_edit(self, 1e-12, 1.0, 12, True),
            "max_nfev": _int_edit(self, 1, 10_000_000),
        }
        for key in ("ftol", "xtol", "gtol", "max_nfev"):
            lsq_form.addRow(key, self._lsq_widgets[key])

        # MCMC tab
        mcmc_box = QGroupBox("MCMC")
        mcmc_form = QFormLayout(mcmc_box)
        self._mcmc_widgets = {
            "n_samples": _int_edit(self, 1, 1_000_000),
            "burn_in": _int_edit(self, 0, 1_000_000),
            "thin": _int_edit(self, 1, 10_000),
            "rw_scale": _float_edit(self, 1e-9, 10.0, 6, False),
        }
        for key in ("n_samples", "burn_in", "thin", "rw_scale"):
            mcmc_form.addRow(key, self._mcmc_widgets[key])

        # DREAM tab
        dream_box = QGroupBox("DREAM")
        dream_form = QFormLayout(dream_box)
        self._dream_widgets = {
            "n_samples": _int_edit(self, 1, 1_000_000),
            "n_chains": _int_edit(self, 1, 100),
            "n_diff_pairs": _int_edit(self, 1, 10_000),
            "burn_in": _int_edit(self, 0, 1_000_000),
            "thin": _int_edit(self, 1, 10_000),
            "n_cr": _int_edit(self, 1, 100),
        }
        for key in ("n_samples", "n_chains", "n_diff_pairs", "n_cr", "burn_in", "thin"):
            dream_form.addRow(key, self._dream_widgets[key])

        # Pack tabs
        for box, title in [
            (de_box, "DE"),
            (lsq_box, "Least Squares"),
            (dream_box, "DREAM"),
            (mcmc_box, "MCMC"),
        ]:
            page = QWidget()
            page_lay = QVBoxLayout(page)
            page_lay.addWidget(box)
            tabs.addTab(page, title)

        # Measurement uncertainty (all solvers)
        unc_box = self._build_uncertainty_box()

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        # Custom reset button
        btn_reset = buttons.addButton("Reset to defaults", QDialogButtonBox.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_reset.clicked.connect(self._on_reset)

        # Main layout
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(unc_box)
        lay.addWidget(buttons)

        self._load_from_state()

    # ------------------------------------------------------ uncertainty box
    def _build_uncertainty_box(self) -> QGroupBox:
        box = QGroupBox("Measurement uncertainty (all solvers)")
        grid = QGridLayout(box)

        self.cb_file = QCheckBox(
            "Use uncertainties from the observation file ('<tracer>_sd' columns)"
        )
        self._file_available = file_uncertainty(self.state) is not None
        if not self._file_available:
            self.cb_file.setEnabled(False)
            self.cb_file.setText(self.cb_file.text() + " – none loaded")
        self.lbl_other = QLabel()

        self.rb_none = QRadioButton(
            "Not specified (DE, least squares: unweighted; MCMC, DREAM: estimated per tracer)"
        )
        self.rb_abs = QRadioButton("Absolute standard deviation:")
        self.rb_rel = QRadioButton("Relative standard deviation [% of observed value]:")
        self._mode_group = QButtonGroup(self)
        for rb in (self.rb_none, self.rb_abs, self.rb_rel):
            self._mode_group.addButton(rb)

        names = _current_tracer_names(self.state)
        abs_row = QHBoxLayout()
        self.ed_abs: List[QLineEdit] = []
        for i in range(2):
            lbl = QLabel(names[i] if i < len(names) else f"Tracer {i + 1}")
            edit = _float_edit(self, 0.0, 1e100, 12, False)
            edit.setPlaceholderText("σ (tracer unit)")
            if i >= len(names):  # second tracer not selected
                lbl.hide()
                edit.hide()
            abs_row.addWidget(lbl)
            abs_row.addWidget(edit)
            self.ed_abs.append(edit)
        self._n_abs = min(len(names), 2)
        self.ed_rel = _float_edit(self, 0.0, 1000.0, 6, False)
        self.ed_rel.setPlaceholderText("e.g. 5")

        grid.addWidget(self.cb_file, 0, 0, 1, 2)
        grid.addWidget(self.lbl_other, 1, 0, 1, 2)
        grid.addWidget(self.rb_none, 2, 0, 1, 2)
        grid.addWidget(self.rb_abs, 3, 0)
        grid.addLayout(abs_row, 3, 1)
        grid.addWidget(self.rb_rel, 4, 0)
        grid.addWidget(self.ed_rel, 4, 1)

        self.cb_file.toggled.connect(self._update_uncertainty_widgets)
        for rb in (self.rb_none, self.rb_abs, self.rb_rel):
            rb.toggled.connect(self._update_uncertainty_widgets)
        return box

    def _update_uncertainty_widgets(self) -> None:
        """Enable the input fields of the selected option; adapt the label."""
        for edit in self.ed_abs:
            edit.setEnabled(self.rb_abs.isChecked())
        self.ed_rel.setEnabled(self.rb_rel.isChecked())
        if self.cb_file.isEnabled() and self.cb_file.isChecked():
            self.lbl_other.setText("For observations without uncertainty in the file:")
        else:
            self.lbl_other.setText("For all observations:")

    def _load_uncertainty(self, use_file: bool, mode: str, sigma_abs, sigma_rel_percent) -> None:
        self.cb_file.setChecked(bool(use_file) and self._file_available)
        {"absolute": self.rb_abs, "relative": self.rb_rel}.get(mode, self.rb_none).setChecked(True)
        sigma_abs = list(sigma_abs or []) + [None, None]
        for edit, val in zip(self.ed_abs, sigma_abs):
            edit.setText("" if val is None else f"{float(val):g}")
        self.ed_rel.setText("" if sigma_rel_percent is None else f"{float(sigma_rel_percent):g}")
        self._update_uncertainty_widgets()

    def _save_uncertainty(self) -> bool:
        """Validate and write the uncertainty settings to the state."""
        mode = "absolute" if self.rb_abs.isChecked() else "none"
        if self.rb_rel.isChecked():
            mode = "relative"
        sigma_abs = [_parse_float(e) for e in self.ed_abs]
        sigma_rel = _parse_float(self.ed_rel)
        if mode == "absolute" and any(v is None or v <= 0 for v in sigma_abs[: self._n_abs]):
            QMessageBox.warning(
                self,
                "Measurement Uncertainty",
                "Please enter a positive absolute standard deviation for each tracer.",
            )
            return False
        if mode == "relative" and (sigma_rel is None or sigma_rel <= 0):
            QMessageBox.warning(
                self,
                "Measurement Uncertainty",
                "Please enter a positive relative standard deviation (in %).",
            )
            return False
        if self._file_available:  # keep the setting if the checkbox is disabled
            self.state.use_file_uncertainty = self.cb_file.isChecked()
        self.state.uncertainty_mode = mode
        self.state.sigma_abs = [v if v is not None and v > 0 else None for v in sigma_abs]
        self.state.sigma_rel_percent = (
            sigma_rel if sigma_rel is not None and sigma_rel > 0 else None
        )
        return True

    # ------------------------------------------------------------- loading
    def _load_from_state(self) -> None:
        """Populate the UI from ``state.solver_params`` with safe defaults."""
        params = getattr(self.state, "solver_params", {})

        de = params.get("de") or {}
        self._de_widgets["maxiter"].setText(str(int(de.get("maxiter", 10000))))
        self._de_widgets["popsize"].setText(str(int(de.get("popsize", 100))))
        mut = de.get("mutation", (0.5, 1.99))
        try:
            lo, hi = float(mut[0]), float(mut[1])  # type: ignore[index]
        except Exception:
            lo, hi = 0.5, 1.99
        self._de_widgets["mutation_lo"].setText(str(lo))
        self._de_widgets["mutation_hi"].setText(str(hi))
        self._de_widgets["recombination"].setText(str(float(de.get("recombination", 0.5))))
        self._de_widgets["tol"].setText(str(float(de.get("tol", 1e-3))))

        lsq = params.get("lsq") or {}
        self._lsq_widgets["ftol"].setText(str(float(lsq.get("ftol", 1e-8))))
        self._lsq_widgets["xtol"].setText(str(float(lsq.get("xtol", 1e-8))))
        self._lsq_widgets["gtol"].setText(str(float(lsq.get("gtol", 1e-8))))
        self._lsq_widgets["max_nfev"].setText(str(int(lsq.get("max_nfev", 10000))))

        dr = params.get("dream") or {}
        self._dream_widgets["n_samples"].setText(str(int(dr.get("n_samples", 2000))))
        self._dream_widgets["n_chains"].setText(str(int(dr.get("n_chains", 8))))
        self._dream_widgets["burn_in"].setText(str(int(dr.get("burn_in", 2000))))
        self._dream_widgets["thin"].setText(str(int(dr.get("thin", 1))))
        self._dream_widgets["n_diff_pairs"].setText(str(int(dr.get("n_diff_pairs", 2))))
        self._dream_widgets["n_cr"].setText(str(int(dr.get("n_cr", 4))))

        mc = params.get("mcmc") or {}
        self._mcmc_widgets["n_samples"].setText(str(int(mc.get("n_samples", 1000))))
        self._mcmc_widgets["burn_in"].setText(str(int(mc.get("burn_in", 2000))))
        self._mcmc_widgets["thin"].setText(str(int(mc.get("thin", 1))))
        self._mcmc_widgets["rw_scale"].setText(str(float(mc.get("rw_scale", 0.05))))

        self._load_uncertainty(
            getattr(self.state, "use_file_uncertainty", True),
            getattr(self.state, "uncertainty_mode", "none"),
            getattr(self.state, "sigma_abs", None),
            getattr(self.state, "sigma_rel_percent", None),
        )

    def _default_params(self) -> Dict[str, Dict[str, Any]]:
        """Return built-in defaults for DE, Least Squares, MCMC and DREAM parameters."""
        return {
            "de": {
                "maxiter": 1000,
                "popsize": 15,
                "mutation": (0.5, 1.0),
                "recombination": 0.7,
                "tol": 1e-2,
            },
            "lsq": {
                "ftol": 1e-8,
                "xtol": 1e-8,
                "gtol": 1e-8,
                "max_nfev": 10_000,
            },
            "mcmc": {
                "n_samples": 10000,
                "burn_in": 2000,
                "thin": 2,
                "rw_scale": 0.05,
            },
            "dream": {
                "n_samples": 2000,
                "burn_in": 2000,
                "n_chains": 8,
                "thin": 1,
                "n_diff_pairs": 2,
                "n_cr": 4,
            },
        }

    def _load_defaults(self) -> None:
        """Reset the UI to built-in defaults (does not modify state)."""
        defaults = self._default_params()
        de = defaults["de"]
        self._de_widgets["maxiter"].setText(str(int(de["maxiter"])))
        self._de_widgets["popsize"].setText(str(int(de["popsize"])))
        self._de_widgets["mutation_lo"].setText(str(float(de["mutation"][0])))
        self._de_widgets["mutation_hi"].setText(str(float(de["mutation"][1])))
        self._de_widgets["recombination"].setText(str(float(de["recombination"])))
        self._de_widgets["tol"].setText(str(float(de["tol"])))

        lsq = defaults["lsq"]
        for key in ("ftol", "xtol", "gtol"):
            self._lsq_widgets[key].setText(str(float(lsq[key])))
        self._lsq_widgets["max_nfev"].setText(str(int(lsq["max_nfev"])))

        mc = defaults["mcmc"]
        for key in ("n_samples", "burn_in", "thin"):
            self._mcmc_widgets[key].setText(str(int(mc[key])))
        self._mcmc_widgets["rw_scale"].setText(str(float(mc["rw_scale"])))

        dr = defaults["dream"]  # was defaults["mcmc"] -> KeyError (O.22)
        for key in ("n_samples", "burn_in", "thin", "n_chains", "n_diff_pairs", "n_cr"):
            self._dream_widgets[key].setText(str(int(dr[key])))

        self._load_uncertainty(True, "none", None, None)

    def _on_reset(self) -> None:
        """Handle the reset button by reloading built-in defaults."""
        self._load_defaults()

    # --------------------------------------------------------------- saving
    def accept(self) -> None:  # noqa: D401
        """Validate and save parameters back to the state."""
        if not self._save_uncertainty():
            return
        defaults = self._default_params()

        def _to_int(edit: QLineEdit, default_val: int) -> int:
            try:
                return int(edit.text().strip())
            except ValueError:
                return default_val

        def _to_float(edit: QLineEdit, default_val: float) -> float:
            val = _parse_float(edit)
            return default_val if val is None else val

        d_de, d_lsq, d_mc, d_dr = (defaults[k] for k in ("de", "lsq", "mcmc", "dream"))
        w = self._de_widgets
        de = {
            "maxiter": _to_int(w["maxiter"], int(d_de["maxiter"])),
            "popsize": _to_int(w["popsize"], int(d_de["popsize"])),
            "mutation": (
                _to_float(w["mutation_lo"], float(d_de["mutation"][0])),
                _to_float(w["mutation_hi"], float(d_de["mutation"][1])),
            ),
            "recombination": _to_float(w["recombination"], float(d_de["recombination"])),
            "tol": _to_float(w["tol"], float(d_de["tol"])),
        }
        w = self._lsq_widgets
        lsq = {
            "ftol": _to_float(w["ftol"], float(d_lsq["ftol"])),
            "xtol": _to_float(w["xtol"], float(d_lsq["xtol"])),
            "gtol": _to_float(w["gtol"], float(d_lsq["gtol"])),
            "max_nfev": _to_int(w["max_nfev"], int(d_lsq["max_nfev"])),
        }
        w = self._mcmc_widgets
        mcmc = {
            "n_samples": _to_int(w["n_samples"], int(d_mc["n_samples"])),
            "burn_in": _to_int(w["burn_in"], int(d_mc["burn_in"])),
            "thin": _to_int(w["thin"], int(d_mc["thin"])),
            "rw_scale": _to_float(w["rw_scale"], float(d_mc["rw_scale"])),
        }
        w = self._dream_widgets
        dream = {
            key: _to_int(w[key], int(d_dr[key]))
            for key in ("n_samples", "burn_in", "thin", "n_chains", "n_diff_pairs", "n_cr")
        }

        # Write back to state
        params = getattr(self.state, "solver_params", {})
        params["de"] = de
        params["lsq"] = lsq
        params["mcmc"] = mcmc
        params["dream"] = dream
        self.state.solver_params = params

        super().accept()
