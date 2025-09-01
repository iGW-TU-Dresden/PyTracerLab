from __future__ import annotations

from typing import Any, Dict

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class SolverParamsDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Solver Parameters")
        self.state = state

        tabs = QTabWidget()

        # DE tab
        self._de_widgets = {}
        de_box = QGroupBox("Differential Evolution")
        de_form = QFormLayout(de_box)
        w_maxiter = QSpinBox()
        w_maxiter.setRange(1, 1_000_000)
        w_popsize = QSpinBox()
        w_popsize.setRange(1, 10_000)
        w_mut_lo = QDoubleSpinBox()
        w_mut_lo.setDecimals(4)
        w_mut_lo.setRange(0.0, 3.0)
        w_mut_hi = QDoubleSpinBox()
        w_mut_hi.setDecimals(4)
        w_mut_hi.setRange(0.0, 3.0)
        w_recomb = QDoubleSpinBox()
        w_recomb.setDecimals(4)
        w_recomb.setRange(0.0, 1.0)
        w_tol = QDoubleSpinBox()
        w_tol.setDecimals(9)
        w_tol.setRange(1e-12, 1.0)
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
        w_nsamp = QSpinBox()
        w_nsamp.setRange(1, 1_000_000)
        w_burn = QSpinBox()
        w_burn.setRange(0, 1_000_000)
        w_thin = QSpinBox()
        w_thin.setRange(1, 10_000)
        w_rw = QDoubleSpinBox()
        w_rw.setDecimals(6)
        w_rw.setRange(1e-9, 10.0)
        w_sigma = QLineEdit()
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
        # Load DE
        de = getattr(self.state, "solver_params", {}).get("de") or {}
        self._de_widgets["maxiter"].setValue(int(de.get("maxiter", 10000)))
        self._de_widgets["popsize"].setValue(int(de.get("popsize", 100)))
        mut = de.get("mutation", (0.5, 1.99))
        try:
            lo, hi = float(mut[0]), float(mut[1])  # type: ignore[index]
        except Exception:
            lo, hi = 0.5, 1.99
        self._de_widgets["mutation_lo"].setValue(lo)
        self._de_widgets["mutation_hi"].setValue(hi)
        self._de_widgets["recombination"].setValue(float(de.get("recombination", 0.5)))
        self._de_widgets["tol"].setValue(float(de.get("tol", 1e-3)))

        # Load MCMC
        mc = getattr(self.state, "solver_params", {}).get("mcmc") or {}
        self._mcmc_widgets["n_samples"].setValue(int(mc.get("n_samples", 1000)))
        self._mcmc_widgets["burn_in"].setValue(int(mc.get("burn_in", 2000)))
        self._mcmc_widgets["thin"].setValue(int(mc.get("thin", 1)))
        self._mcmc_widgets["rw_scale"].setValue(float(mc.get("rw_scale", 0.05)))
        sigma_val = mc.get("sigma", None)
        self._mcmc_widgets["sigma"].setText("" if sigma_val is None else str(sigma_val))

    def _default_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "de": {
                "maxiter": 10000,
                "popsize": 100,
                "mutation": (0.5, 1.99),
                "recombination": 0.5,
                "tol": 1e-3,
            },
            "mcmc": {
                "n_samples": 1000,
                "burn_in": 2000,
                "thin": 1,
                "rw_scale": 0.05,
                "sigma": None,
            },
        }

    def _load_defaults(self) -> None:
        defaults = self._default_params()
        de = defaults["de"]
        self._de_widgets["maxiter"].setValue(int(de["maxiter"]))
        self._de_widgets["popsize"].setValue(int(de["popsize"]))
        self._de_widgets["mutation_lo"].setValue(float(de["mutation"][0]))
        self._de_widgets["mutation_hi"].setValue(float(de["mutation"][1]))
        self._de_widgets["recombination"].setValue(float(de["recombination"]))
        self._de_widgets["tol"].setValue(float(de["tol"]))

        mc = defaults["mcmc"]
        self._mcmc_widgets["n_samples"].setValue(int(mc["n_samples"]))
        self._mcmc_widgets["burn_in"].setValue(int(mc["burn_in"]))
        self._mcmc_widgets["thin"].setValue(int(mc["thin"]))
        self._mcmc_widgets["rw_scale"].setValue(float(mc["rw_scale"]))
        self._mcmc_widgets["sigma"].setText("")

    def _on_reset(self) -> None:
        self._load_defaults()

    def accept(self) -> None:  # noqa: D401
        # Save DE
        de = {
            "maxiter": int(self._de_widgets["maxiter"].value()),
            "popsize": int(self._de_widgets["popsize"].value()),
            "mutation": (
                float(self._de_widgets["mutation_lo"].value()),
                float(self._de_widgets["mutation_hi"].value()),
            ),
            "recombination": float(self._de_widgets["recombination"].value()),
            "tol": float(self._de_widgets["tol"].value()),
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
            "n_samples": int(self._mcmc_widgets["n_samples"].value()),
            "burn_in": int(self._mcmc_widgets["burn_in"].value()),
            "thin": int(self._mcmc_widgets["thin"].value()),
            "rw_scale": float(self._mcmc_widgets["rw_scale"].value()),
            "sigma": sigma_val,
        }

        # Write back to state
        params = getattr(self.state, "solver_params", {})
        params["de"] = de
        params["mcmc"] = mcmc
        self.state.solver_params = params

        super().accept()
