"""Solver parameter settings window."""

from __future__ import annotations

from typing import Any, Dict

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class SolverParamsDialog(QDialog):
    """Tabbed dialog for editing DE and MCMC solver parameters.

    The dialog reads current values from ``state.solver_params`` and writes
    back updated values on accept.
    """

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Solver Parameters")
        self.state = state

        tabs = QTabWidget()

        # DE tab
        self._de_widgets = {}

        de_box = QGroupBox("Differential Evolution")
        de_form = QFormLayout(de_box)

        w_maxiter = QLineEdit()
        maxiter_validator = QIntValidator(1, 1_000_000, self)
        w_maxiter.setValidator(maxiter_validator)
        w_maxiter.setAlignment(Qt.AlignRight)

        w_popsize = QLineEdit()
        popsize_validator = QIntValidator(1, 10_000, self)
        w_popsize.setValidator(popsize_validator)
        w_popsize.setAlignment(Qt.AlignRight)

        w_mut_lo = QLineEdit()
        mut_lo_val = QDoubleValidator(self)
        mut_lo_val.setNotation(QDoubleValidator.StandardNotation)
        mut_lo_val.setDecimals(4)
        mut_lo_val.setRange(0.0, 3.0, 4)
        w_mut_lo.setValidator(mut_lo_val)
        w_mut_lo.setAlignment(Qt.AlignRight)

        w_mut_hi = QLineEdit()
        mut_hi_val = QDoubleValidator(self)
        mut_hi_val.setNotation(QDoubleValidator.StandardNotation)
        mut_hi_val.setDecimals(4)
        mut_hi_val.setRange(0.0, 3.0, 4)
        w_mut_hi.setValidator(mut_hi_val)
        w_mut_hi.setAlignment(Qt.AlignRight)

        w_recomb = QLineEdit()
        recomb_val = QDoubleValidator(self)
        recomb_val.setNotation(QDoubleValidator.StandardNotation)
        recomb_val.setDecimals(4)
        recomb_val.setRange(0.0, 1.0, 4)
        w_recomb.setValidator(recomb_val)
        w_recomb.setAlignment(Qt.AlignRight)

        w_tol = QLineEdit()
        tol_val = QDoubleValidator(self)
        tol_val.setNotation(QDoubleValidator.StandardNotation)
        tol_val.setDecimals(9)
        tol_val.setRange(1e-12, 1.0, 9)
        w_tol.setValidator(tol_val)
        w_tol.setAlignment(Qt.AlignRight)

        self._de_widgets = {
            "maxiter": w_maxiter,
            "popsize": w_popsize,
            "mutation_lo": w_mut_lo,
            "mutation_hi": w_mut_hi,
            "recombination": w_recomb,
            "tol": w_tol,
        }
        de_form.addRow("maxiter", w_maxiter)
        de_form.addRow("popsize", w_popsize)
        de_form.addRow("mutation low", w_mut_lo)
        de_form.addRow("mutation high", w_mut_hi)
        de_form.addRow("recombination", w_recomb)
        de_form.addRow("tol", w_tol)

        # MCMC tab
        self._mcmc_widgets = {}

        mcmc_box = QGroupBox("MCMC")
        mcmc_form = QFormLayout(mcmc_box)

        w_nsamp = QLineEdit()
        nsamp_validator = QIntValidator(1, 1_000_000, self)
        w_nsamp.setValidator(nsamp_validator)
        w_nsamp.setAlignment(Qt.AlignRight)

        w_burn = QLineEdit()
        burn_validator = QIntValidator(0, 1_000_000, self)
        w_burn.setValidator(burn_validator)
        w_burn.setAlignment(Qt.AlignRight)

        w_thin = QLineEdit()
        thin_validator = QIntValidator(1, 10_000, self)
        w_thin.setValidator(thin_validator)
        w_thin.setAlignment(Qt.AlignRight)

        w_rw = QLineEdit()
        rw_val = QDoubleValidator(self)
        rw_val.setNotation(QDoubleValidator.StandardNotation)
        rw_val.setDecimals(6)
        rw_val.setRange(1e-9, 10.0, 6)
        w_rw.setValidator(rw_val)
        w_rw.setAlignment(Qt.AlignRight)

        w_sigma = QLineEdit()
        sigma_val = QDoubleValidator(self)
        sigma_val.setNotation(QDoubleValidator.StandardNotation)
        sigma_val.setDecimals(12)
        sigma_val.setRange(-1e100, 1e100, 12)
        w_sigma.setValidator(sigma_val)
        w_sigma.setPlaceholderText("leave empty to infer")

        self._mcmc_widgets = {
            "n_samples": w_nsamp,
            "burn_in": w_burn,
            "thin": w_thin,
            "rw_scale": w_rw,
            "sigma": w_sigma,
        }
        mcmc_form.addRow("n_samples", w_nsamp)
        mcmc_form.addRow("burn_in", w_burn)
        mcmc_form.addRow("thin", w_thin)
        mcmc_form.addRow("rw_scale", w_rw)
        mcmc_form.addRow("sigma", w_sigma)

        # Pack tabs
        w_de = QVBoxLayout()
        w_de.addWidget(de_box)
        w_mcmc = QVBoxLayout()
        w_mcmc.addWidget(mcmc_box)
        de_widget = QWidget()
        de_widget.setLayout(w_de)
        mcmc_widget = QWidget()
        mcmc_widget.setLayout(w_mcmc)
        tabs.addTab(de_widget, "DE")
        tabs.addTab(mcmc_widget, "MCMC")

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
        lay.addWidget(buttons)

        self._load_from_state()

    def _load_from_state(self) -> None:
        """Populate the UI from ``state.solver_params`` with safe defaults."""
        # Load DE
        de = getattr(self.state, "solver_params", {}).get("de") or {}
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

        # Load MCMC
        mc = getattr(self.state, "solver_params", {}).get("mcmc") or {}
        self._mcmc_widgets["n_samples"].setText(str(int(mc.get("n_samples", 1000))))
        self._mcmc_widgets["burn_in"].setText(str(int(mc.get("burn_in", 2000))))
        self._mcmc_widgets["thin"].setText(str(int(mc.get("thin", 1))))
        self._mcmc_widgets["rw_scale"].setText(str(float(mc.get("rw_scale", 0.05))))
        sigma_val = mc.get("sigma", None)
        self._mcmc_widgets["sigma"].setText("" if sigma_val is None else str(sigma_val))

    def _default_params(self) -> Dict[str, Dict[str, Any]]:
        """Return built-in defaults for DE and MCMC parameters."""
        return {
            "de": {
                "maxiter": 1000,
                "popsize": 15,
                "mutation": (0.5, 1.0),
                "recombination": 0.7,
                "tol": 1e-2,
            },
            "mcmc": {
                "n_samples": 10000,
                "burn_in": 2000,
                "thin": 2,
                "rw_scale": 0.05,
                "sigma": None,
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

        mc = defaults["mcmc"]
        self._mcmc_widgets["n_samples"].setText(str(int(mc["n_samples"])))
        self._mcmc_widgets["burn_in"].setText(str(int(mc["burn_in"])))
        self._mcmc_widgets["thin"].setText(str(int(mc["thin"])))
        self._mcmc_widgets["rw_scale"].setText(str(float(mc["rw_scale"])))
        self._mcmc_widgets["sigma"].setText("")

    def _on_reset(self) -> None:
        """Handle the reset button by reloading built-in defaults."""
        self._load_defaults()

    def accept(self) -> None:  # noqa: D401
        """Validate and save parameters back to ``state.solver_params``."""
        # Save DE
        defaults = self._default_params()

        def _to_int(edit: QLineEdit, default_val: int) -> int:
            txt = edit.text().strip()
            try:
                val = int(txt)
            except Exception:
                return default_val
            return val

        def _to_float(edit: QLineEdit, default_val: float) -> float:
            txt = edit.text().strip()
            try:
                val = float(txt)
            except Exception:
                return default_val
            return val

        de = {
            "maxiter": _to_int(self._de_widgets["maxiter"], int(defaults["de"]["maxiter"])),
            "popsize": _to_int(self._de_widgets["popsize"], int(defaults["de"]["popsize"])),
            "mutation": (
                _to_float(self._de_widgets["mutation_lo"], float(defaults["de"]["mutation"][0])),
                _to_float(self._de_widgets["mutation_hi"], float(defaults["de"]["mutation"][1])),
            ),
            "recombination": _to_float(
                self._de_widgets["recombination"], float(defaults["de"]["recombination"])
            ),
            "tol": _to_float(self._de_widgets["tol"], float(defaults["de"]["tol"])),
        }

        # Save MCMC
        sigma_txt = self._mcmc_widgets["sigma"].text().strip()
        sigma_val: float | None
        if sigma_txt == "":
            sigma_val = None
        else:
            try:
                sigma_val = float(sigma_txt)
            except ValueError:
                sigma_val = None
        mcmc = {
            "n_samples": _to_int(
                self._mcmc_widgets["n_samples"], int(defaults["mcmc"]["n_samples"])
            ),
            "burn_in": _to_int(self._mcmc_widgets["burn_in"], int(defaults["mcmc"]["burn_in"])),
            "thin": _to_int(self._mcmc_widgets["thin"], int(defaults["mcmc"]["thin"])),
            "rw_scale": _to_float(
                self._mcmc_widgets["rw_scale"], float(defaults["mcmc"]["rw_scale"])
            ),
            "sigma": sigma_val,
        }

        # Write back to state
        params = getattr(self.state, "solver_params", {})
        params["de"] = de
        params["mcmc"] = mcmc
        self.state.solver_params = params

        super().accept()
