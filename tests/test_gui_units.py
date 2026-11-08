"""New units in the GUI (WP4, D22/D23) and the PM fix for yearly data (O.27)."""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

from PyTracerLab.gui.controller import Controller  # noqa: E402
from PyTracerLab.gui.state import AppState  # noqa: E402
from PyTracerLab.gui.tabs import file_input  # noqa: E402
from PyTracerLab.gui.tabs.file_input import FileInputTab  # noqa: E402
from PyTracerLab.gui.tabs.model_design import ModelDesignTab  # noqa: E402
from PyTracerLab.gui.tabs.parameters import ParametersTab  # noqa: E402
from PyTracerLab.model.registry import UNIT_REGISTRY  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[1] / "docs" / "examples" / "templates"


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _tabs(state=None):
    state = state or AppState()
    design = ModelDesignTab(state, UNIT_REGISTRY)
    params = ParametersTab(state, UNIT_REGISTRY)
    design.selection_changed.connect(params.refresh)
    return design, params


def _select(design, *units):
    for row, name in zip(design.rows, units):
        row["combo"].setCurrentIndex(row["combo"].findData(name))


def _labels(params):
    grid = params.grid
    return [
        grid.itemAt(i).widget().text()
        for i in range(grid.count())
        if isinstance(grid.itemAt(i).widget(), QtWidgets.QLabel)
    ]


def _loaded_tab(monkeypatch):
    monkeypatch.setattr(
        file_input, "QMessageBox", type("Stub", (), {"warning": print, "information": print})
    )
    tab = FileInputTab(AppState())
    tab.yearly.setChecked(True)
    assert tab._load_input(str(TEMPLATES / "input_template.csv"))
    return tab


def test_new_units_selectable(qapp):
    design, _ = _tabs()
    combo = design.rows[0]["combo"]
    names = [combo.itemData(i) for i in range(combo.count())]
    assert {"GM", "LM", "PEM"} <= set(names)


def test_parameter_rows_labels_and_fixed_geometry(qapp):
    design, params = _tabs()
    _select(design, "PEM", "GM")
    pem = {ed.key: ed for ed in params.editors if ed.prefix.startswith("pem")}
    assert list(pem) == ["mtt", "above", "screen", "below", "fraction"]  # binary mixture
    assert not pem["mtt"].fixed.isChecked()
    assert all(pem[k].fixed.isChecked() for k in ("above", "screen", "below"))
    gm = {ed.key: ed for ed in params.editors if ed.prefix.startswith("gm")}
    assert not gm["alpha"].fixed.isChecked() and gm["alpha"].val.text() == "1.0"
    labels = _labels(params)
    assert "PEM (1) - Screen Length [length]" in labels
    assert "PEM (1) - Mean Travel Time [years]" in labels
    assert "GM (1) - Shape Factor [-]" in labels


def test_controller_builds_and_calibrates_pem(qapp, monkeypatch):
    tab = _loaded_tab(monkeypatch)
    assert tab._load_target(str(TEMPLATES / "observations_template.csv"))
    state = tab.state
    design, params = _tabs(state)
    _select(design, "PEM")
    params.commit()
    state.uncertainty_mode, state.sigma_abs = "absolute", [0.3, None]
    state.solver_params["de"].update(maxiter=3, popsize=5)

    ctrl = Controller(state)
    errors, status = [], []
    ctrl.error.connect(errors.append)
    ctrl.status.connect(status.append)
    ctrl.calibrate()
    assert errors == []
    prefix = state.design_instances[0]["prefix"]
    assert ctrl.ml.param_keys(free_only=True) == [f"{prefix}.mtt"]  # geometry fixed
    assert ctrl.ml.params[f"{prefix}.screen"]["value"] == 20.0  # length: no month conversion
    assert status[-1].startswith("Calibration finished.")


def test_controller_uses_default_fixed_flag_without_parameters_tab(qapp, monkeypatch):
    """A design that never visited the Parameters tab still gets a fixed geometry."""
    state = _loaded_tab(monkeypatch).state
    state.design_instances = [{"name": "PEM", "prefix": "pem1", "fraction": 1.0}]
    state.params = {}
    ctrl = Controller(state)
    ctrl.build_model()
    assert ctrl.ml.param_keys(free_only=True) == ["pem1.mtt"]


def test_pm_with_yearly_data(qapp, monkeypatch):
    """O.27: the GUI works in months (dt = 12 for yearly data); PM output was 1/12."""
    state = _loaded_tab(monkeypatch).state
    state.design_instances = [{"name": "PM", "prefix": "pm1", "fraction": 1.0}]
    state.params = {"pm1": {"mtt": {"val": 5.0, "lb": 1.0, "ub": 50.0, "fixed": False}}}
    ctrl = Controller(state)
    out, errors = [], []
    ctrl.simulated.connect(out.append)
    ctrl.error.connect(errors.append)
    ctrl.simulate()
    assert errors == [] and ctrl.ml.dt == 12.0
    x = np.asarray(state.input_series[1]).ravel()
    lam_month = ctrl.ml.lambda_
    expected = x[-6] * np.exp(-lam_month * 60.0)  # 5 years = 60 months
    assert np.asarray(out[0]).ravel()[-1] == pytest.approx(expected)
