"""Mixing fraction of binary mixtures in the GUI (WP3, D17/D18) and O.25."""

import os
from pathlib import Path

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
    """Model and Parameters tab on one state, wired as in the main window."""
    state = state or AppState()
    design = ModelDesignTab(state, UNIT_REGISTRY)
    params = ParametersTab(state, UNIT_REGISTRY)
    design.selection_changed.connect(params.refresh)
    return design, params


def _select(design, *units):
    for row, name in zip(design.rows, units):
        row["combo"].setCurrentIndex(row["combo"].findData(name))


def _editor(params, prefix, key):
    return next(ed for ed in params.editors if ed.prefix == prefix and ed.key == key)


def test_model_tab_second_fraction_is_remainder(qapp):
    design, _ = _tabs()
    _select(design, "EM", "PM")
    first, second = design.rows[0]["frac"], design.rows[1]["frac"]
    first.setText("0.7")
    assert second.text() == "0.3000" and not second.isEnabled()
    assert not design.binary_note.isHidden()
    assert [i["fraction"] for i in design.state.design_instances] == pytest.approx([0.7, 0.3])

    _select(design, "EM", "PM", "DM")  # three units: all fractions editable
    assert second.isEnabled() and design.rows[2]["frac"].isEnabled()
    assert design.binary_note.isHidden()


def test_parameters_tab_fraction_row(qapp):
    design, params = _tabs()
    _select(design, "EM", "PM")
    design.rows[0]["frac"].setText("0.6")
    ed = params.fraction_editor
    assert ed is not None and ed.prefix == "em1" and ed.key == "fraction"
    assert ed.fixed.isChecked()  # fixed by default
    assert ed.val.text() == "0.6" and not ed.val.isEnabled()
    assert (ed.lb.text(), ed.ub.text()) == ("0.0", "1.0")

    _select(design, "EM", "PM", "DM")
    assert params.fraction_editor is None


def test_correlation_hint(qapp):
    design, params = _tabs()
    _select(design, "EM", "PM")
    assert params.hint.isHidden()
    params.fraction_editor.fixed.setChecked(False)  # fraction + both mtt free
    assert not params.hint.isHidden()
    _editor(params, "pm1", "mtt").fixed.setChecked(True)
    assert params.hint.isHidden()


def test_refresh_keeps_uncommitted_edits(qapp):
    """Changing the Model tab used to discard edits in the Parameters tab (O.25)."""
    design, params = _tabs()
    _select(design, "EM", "PM")
    _editor(params, "em1", "mtt").val.setText("42")
    params.fraction_editor.fixed.setChecked(False)
    params.fraction_editor.ub.setText("0.8")
    design.rows[0]["frac"].setText("0.4")  # triggers a refresh
    assert _editor(params, "em1", "mtt").val.text() == "42.0"
    assert not params.fraction_editor.fixed.isChecked()
    assert params.fraction_editor.ub.text() == "0.8"
    assert params.fraction_editor.val.text() == "0.4"


def test_controller_calibrates_fraction(qapp, monkeypatch):
    monkeypatch.setattr(file_input, "QMessageBox", type("Stub", (), {"warning": print}))
    tab = FileInputTab(AppState())
    tab.yearly.setChecked(True)
    assert tab._load_input(str(TEMPLATES / "input_template.csv"))
    assert tab._load_target(str(TEMPLATES / "observations_template.csv"))  # 3 samples
    design, params = _tabs(tab.state)
    _select(design, "EM", "PM")
    design.rows[0]["frac"].setText("0.5")
    design.ss_checkbox.setChecked(True)  # steady-state input: no note about it (O.29)
    design.ss_inputs[0].setText("5.0")
    params.fraction_editor.fixed.setChecked(False)
    params.fraction_editor.lb.setText("0.2")
    params.fraction_editor.ub.setText("0.8")
    params.commit()
    state = tab.state
    state.uncertainty_mode, state.sigma_abs = "absolute", [0.3, None]
    state.solver_params["de"].update(maxiter=3, popsize=5)

    ctrl = Controller(state)
    errors, status = [], []
    ctrl.error.connect(errors.append)
    ctrl.status.connect(status.append)
    ctrl.calibrate()
    assert errors == []
    rec = ctrl.ml.params["em1.fraction"]
    assert rec["fixed"] is False and rec["bounds"] == (0.2, 0.8)
    assert 0.2 <= rec["value"] <= 0.8
    assert sum(ctrl.ml.unit_fractions) == pytest.approx(1.0)
    # 3 free parameters (2 mtt + fraction) and 3 observations: no note
    assert status == ["Calibration finished."]

    # 4 free parameters for 3 observations: note in the status message
    state.design_instances[0]["name"] = "DM"  # unit with two parameters
    state.design_instances[0]["prefix"] = "dm1"
    state.params["dm1"] = {"fraction": state.params["em1"]["fraction"]}
    ctrl.calibrate()
    assert errors == []
    assert "more free parameters (4) than observations (3)" in status[-1]
