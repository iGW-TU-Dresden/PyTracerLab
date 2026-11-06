"""Tests for the package-level API (``import PyTracerLab as lab``) and tracer data."""

import math
import subprocess
import sys

import pytest

import PyTracerLab as lab


@pytest.mark.parametrize("name", lab.__all__)
def test_public_names_resolve(name):
    assert getattr(lab, name) is not None


def test_package_level_objects_are_the_model_objects():
    from PyTracerLab.model.model import Model
    from PyTracerLab.model.solver import Solver
    from PyTracerLab.model.units import EPMUnit, ExEPMUnit

    assert lab.Model is Model
    assert lab.Solver is Solver
    assert lab.EPMUnit is EPMUnit
    assert lab.ExEPMUnit is ExEPMUnit


def test_legacy_import_still_works():
    import PyTracerLab.model as ism

    assert ism.Model is lab.Model
    assert ism.Solver is lab.Solver


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError):
        lab.does_not_exist  # noqa: B018


def test_import_does_not_load_gui():
    """``import PyTracerLab`` and the model API must not import the GUI or PyQt5."""
    code = (
        "import sys\n"
        "import PyTracerLab as lab\n"
        "assert 'PyQt5' not in sys.modules, 'import'\n"
        "lab.Model, lab.read_series, lab.decay_constant, lab.EPMUnit\n"
        "assert 'PyQt5' not in sys.modules, 'model api'\n"
        "lab.Solver\n"
        "assert 'PyTracerLab.gui' not in sys.modules, 'solver'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_decay_constant():
    assert lab.decay_constant("Tritium") == pytest.approx(math.log(2) / 12.32)
    assert lab.decay_constant("3H") == lab.decay_constant("tritium")
    assert lab.decay_constant("14C", time_unit="month") == pytest.approx(
        math.log(2) / (5700.0 * 12.0)
    )
    # tritiogenic helium uses the decay constant of its parent (tritium)
    assert lab.decay_constant("3He(trit)") == lab.decay_constant("3H")
    with pytest.raises(KeyError, match="Known tracers"):
        lab.decay_constant("Unobtainium")
    with pytest.raises(ValueError, match="time_unit"):
        lab.decay_constant("3H", time_unit="week")


def test_gui_uses_package_tracer_table():
    pytest.importorskip("PyQt5.QtWidgets")  # importing the GUI package needs Qt
    from PyTracerLab.gui.database import Tracers

    assert set(Tracers.tracer_data) == set(lab.TRACERS)
    for name, (half_life, production) in Tracers.tracer_data.items():
        assert half_life == lab.TRACERS[name]["half_life"]
        assert production == lab.TRACERS[name]["production"]
