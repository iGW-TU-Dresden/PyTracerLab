"""Measurement uncertainty in the GUI: settings, dialog, controller (WP2)."""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

from PyTracerLab.gui import controller as gui_controller  # noqa: E402
from PyTracerLab.gui import uncertainty as unc  # noqa: E402
from PyTracerLab.gui.controller import Controller  # noqa: E402
from PyTracerLab.gui.state import AppState  # noqa: E402
from PyTracerLab.gui.tabs import file_input, solver_params  # noqa: E402
from PyTracerLab.gui.tabs.file_input import FileInputTab  # noqa: E402
from PyTracerLab.gui.tabs.solver_params import SolverParamsDialog  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[1] / "docs" / "examples" / "templates"
PAPER = Path(__file__).resolve().parents[1] / "docs" / "examples" / "paper"


class _MessageBoxStub:
    def __init__(self):
        self.calls = []

    def warning(self, parent, title, text, *args):
        self.calls.append(("warning", title, text))

    def information(self, parent, title, text, *args):
        self.calls.append(("information", title, text))


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def messages(monkeypatch):
    stub = _MessageBoxStub()
    monkeypatch.setattr(file_input, "QMessageBox", stub)
    monkeypatch.setattr(solver_params, "QMessageBox", stub)
    return stub


def _loaded_state(observations="observations_template.csv", inp=TEMPLATES / "input_template.csv"):
    """State with yearly template input and observations loaded via the Input tab."""
    tab = FileInputTab(AppState())
    tab.yearly.setChecked(True)
    assert tab._load_input(str(inp))
    folder = TEMPLATES if observations.startswith("observations_template") else PAPER
    assert tab._load_target(str(folder / observations))
    return tab.state


# -------------------------------------------------------------- settings
def test_solver_kwargs_from_file_and_fallbacks(qapp, messages):
    state = _loaded_state()
    y = unc.observations(state)
    observed = ~np.isnan(y)
    assert observed.sum() == 3

    # default: values from the file, nothing else
    kw = unc.solver_sigma_kwargs(state)
    assert kw["sigma_rel"] is None
    np.testing.assert_allclose(kw["sigma"][observed], [0.3, 0.3, 0.2])

    state.uncertainty_mode, state.sigma_rel_percent = "relative", 5.0
    kw = unc.solver_sigma_kwargs(state)
    assert kw["sigma_rel"] == pytest.approx(0.05)
    np.testing.assert_allclose(kw["sigma"][observed], [0.3, 0.3, 0.2])

    state.use_file_uncertainty = False
    kw = unc.solver_sigma_kwargs(state)
    assert kw["sigma"] is None and kw["sigma_rel"] == pytest.approx(0.05)
    np.testing.assert_allclose(unc.effective_sigma(state)[observed], 0.05 * y[observed])

    state.uncertainty_mode, state.sigma_abs = "absolute", [2.0, None]
    kw = unc.solver_sigma_kwargs(state)
    np.testing.assert_allclose(kw["sigma"], [2.0])

    state.uncertainty_mode = "none"
    assert unc.solver_sigma_kwargs(state) == {"sigma": None, "sigma_rel": None}
    assert unc.effective_sigma(state) is None


def test_file_without_uncertainty_columns(qapp, messages):
    state = _loaded_state("tritium_observations.csv", inp=PAPER / "tritium_input.csv")
    assert unc.file_uncertainty(state) is None
    assert unc.solver_sigma_kwargs(state) == {"sigma": None, "sigma_rel": None}
    assert "not specified" in unc.summary(state)


def test_summary(qapp, messages):
    state = _loaded_state()
    assert "from the observation file (3 values)" in unc.summary(state)
    state.use_file_uncertainty = False
    state.uncertainty_mode, state.sigma_rel_percent = "relative", 5.0
    assert "5 % of the observed value" in unc.summary(state)


# ---------------------------------------------------------------- dialog
def test_dialog_saves_uncertainty_settings(qapp, messages):
    state = _loaded_state()
    dlg = SolverParamsDialog(state)
    assert dlg.cb_file.isEnabled() and dlg.cb_file.isChecked()
    dlg.rb_rel.setChecked(True)
    dlg.ed_rel.setText("5")
    dlg.accept()
    assert state.uncertainty_mode == "relative"
    assert state.sigma_rel_percent == pytest.approx(5.0)
    assert state.use_file_uncertainty is True
    assert "sigma" not in state.solver_params["de"]


def test_dialog_without_file_uncertainty(qapp, messages):
    state = AppState()
    dlg = SolverParamsDialog(state)
    assert not dlg.cb_file.isEnabled()
    assert dlg.lbl_other.text() == "For all observations:"
    dlg.rb_abs.setChecked(True)
    dlg.ed_abs[0].setText("0,5")  # decimal comma
    dlg.accept()
    assert state.use_file_uncertainty is True  # unchanged while not available
    assert state.uncertainty_mode == "absolute"
    assert state.sigma_abs[0] == pytest.approx(0.5)


def test_dialog_rejects_missing_values(qapp, messages):
    state = AppState()
    dlg = SolverParamsDialog(state)
    dlg.rb_abs.setChecked(True)
    dlg.accept()
    dlg.rb_rel.setChecked(True)
    dlg.accept()
    assert [c[0] for c in messages.calls] == ["warning", "warning"]
    assert state.uncertainty_mode == "none"


def test_dialog_reset_to_defaults(qapp, messages):
    """Reset used to raise KeyError: 'n_chains' (O.22)."""
    state = _loaded_state()
    state.uncertainty_mode, state.sigma_rel_percent = "relative", 5.0
    dlg = SolverParamsDialog(state)
    assert dlg.rb_rel.isChecked()
    dlg._on_reset()
    assert dlg._dream_widgets["n_chains"].text() == "3"
    assert dlg.rb_none.isChecked() and dlg.ed_rel.text() == ""


# ------------------------------------------------------------ controller
def test_calibration_uses_uncertainties_from_file(qapp, messages, monkeypatch):
    state = _loaded_state()
    state.design_instances = [{"name": "EPM", "prefix": "epm", "fraction": 1.0}]
    state.solver_params["de"].update(maxiter=3, popsize=5)
    captured = {}
    original = gui_controller.ms.run_solver

    def spy(model, key, params):
        captured.update(params)
        return original(model, key, params)

    monkeypatch.setattr(gui_controller.ms, "run_solver", spy)
    ctrl = Controller(state)
    errors, results = [], []
    ctrl.error.connect(errors.append)
    ctrl.calibrated.connect(results.append)
    ctrl.calibrate()
    assert errors == [] and len(results) == 1
    sig = np.asarray(captured["sigma"])
    np.testing.assert_allclose(sig[~np.isnan(sig)], [0.3, 0.3, 0.2])

    # missing uncertainties are reported, not ignored
    state.use_file_uncertainty = False
    state.uncertainty_mode, state.sigma_abs = "absolute", [None, None]
    ctrl.calibrate()
    assert len(errors) == 1 and "no positive measurement uncertainty" in errors[0]
