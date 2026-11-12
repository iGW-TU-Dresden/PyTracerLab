"""Steady-state input in the GUI: explanation in the Model tab and note after a run (O.29)."""

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


def test_model_tab_explains_steady_state(qapp):
    design = ModelDesignTab(AppState(), UNIT_REGISTRY)
    assert not design.ss_checkbox.isChecked()
    assert "assumed to contain no tracer" in design.ss_note.text()
    design.ss_checkbox.setChecked(True)
    assert "older than the input series" in design.ss_note.text()
    assert "no tracer" not in design.ss_note.text()
    assert not hasattr(design, "warmup_value")  # warm-up setting removed


def test_simulation_status_notes_missing_steady_state(qapp, monkeypatch):
    monkeypatch.setattr(file_input, "QMessageBox", type("Stub", (), {"warning": print}))
    tab = FileInputTab(AppState())
    tab.yearly.setChecked(True)
    assert tab._load_input(str(TEMPLATES / "input_template.csv"))
    design = ModelDesignTab(tab.state, UNIT_REGISTRY)
    params = ParametersTab(tab.state, UNIT_REGISTRY)
    design.selection_changed.connect(params.refresh)
    design.rows[0]["combo"].setCurrentIndex(design.rows[0]["combo"].findData("EM"))
    design.rows[0]["frac"].setText("1.0")
    params.commit()

    ctrl = Controller(tab.state)
    errors, status = [], []
    ctrl.error.connect(errors.append)
    ctrl.status.connect(status.append)
    ctrl.simulate()
    assert errors == []
    assert status[-1].startswith("Simulation finished.")
    assert "contains no tracer" in status[-1]

    design.ss_checkbox.setChecked(True)
    design.ss_inputs[0].setText("5.0")
    ctrl.simulate()
    assert errors == [] and status[-1] == "Simulation finished."
