"""Tests for loading data files in the GUI (file input tab, offscreen Qt)."""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

from PyTracerLab.gui.state import AppState  # noqa: E402
from PyTracerLab.gui.tabs import file_input  # noqa: E402
from PyTracerLab.gui.tabs.file_input import FileInputTab  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
PAPER = EXAMPLES / "paper"


class _MessageBoxStub:
    """Records message boxes instead of showing (and blocking on) them."""

    def __init__(self):
        self.calls = []

    def warning(self, parent, title, text, *args):
        self.calls.append(("warning", title, text))

    def information(self, parent, title, text, *args):
        self.calls.append(("information", title, text))

    def kinds(self):
        return [c[0] for c in self.calls]


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def messages(monkeypatch):
    stub = _MessageBoxStub()
    monkeypatch.setattr(file_input, "QMessageBox", stub)
    return stub


@pytest.fixture
def tab(qapp, messages):
    return FileInputTab(AppState())


def _yearly(tab):
    tab.yearly.setChecked(True)
    assert tab.state.is_monthly is False


def write(tmp_path, text, name="data.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_paper_data_sample_dates_only(tab, messages):
    """Input + observations listed only at sample dates (Listing 1 data)."""
    _yearly(tab)
    assert tab._load_input(str(PAPER / "tritium_input.csv"))
    assert tab._load_target(str(PAPER / "tritium_observations.csv"))
    assert messages.calls == []

    times, vals = tab.state.input_series
    obs_times, obs = tab.state.target_series
    assert vals.ndim == 2 and vals.shape[1] == 1
    assert obs.shape == vals.shape
    assert list(obs_times) == list(times)
    assert np.sum(~np.isnan(obs)) == 35
    i = list(times).index(datetime(1963, 1, 1))
    assert obs[i, 0] == pytest.approx(1294.0)
    assert tab.state.input_data.tracers == ["3H"]
    assert tab.state.target_data is not None
    assert "35 observed" in tab.lbl_tg.text()


def test_monthly_full_grid_files_still_load(tab, messages):
    assert tab._load_input(str(EXAMPLES / "example_input_series_1tracer.csv"))
    assert tab._load_target(str(EXAMPLES / "example_observation_series_1tracer.csv"))
    assert messages.calls == []
    assert tab.state.input_series[1].shape == (600, 1)
    assert tab.state.target_series[1].shape == (600, 1)


def test_two_tracer_file(tab, messages):
    _yearly(tab)
    assert tab._load_input(str(EXAMPLES / "3H_SF6_input.csv"))
    assert tab._load_target(str(EXAMPLES / "3H_SF6_observations.csv"))
    assert tab.state.input_series[1].shape[1] == 2
    assert tab.state.target_series[1].shape == tab.state.input_series[1].shape


def test_semicolon_and_decimal_comma(tab, messages, tmp_path):
    """Regression for O.16: semicolon files loaded empty with the old GUI reader."""
    _yearly(tab)
    path = write(tmp_path, "Datum;3H\n2000;10,5\n2001;9,75\n2002;9\n")
    assert tab._load_input(path)
    np.testing.assert_allclose(tab.state.input_series[1][:, 0], [10.5, 9.75, 9.0])


def test_frequency_mismatch_is_reported(tab, messages):
    # yearly file while "Monthly" is selected
    assert not tab._load_input(str(PAPER / "tritium_input.csv"))
    assert tab.state.input_series is None
    assert messages.kinds() == ["warning"]
    assert "Yearly" in messages.calls[0][2]


def test_unsupported_step_is_reported(tab, messages):
    assert not tab._load_input(str(EXAMPLES / "TracerLPM_benchmark_input_6monthly.csv"))
    assert tab.state.input_series is None
    assert "6 months" in messages.calls[0][2]
    assert "monthly and yearly" in messages.calls[0][2]


def test_unreadable_file_is_reported(tab, messages, tmp_path):
    _yearly(tab)
    path = write(tmp_path, "Date, 3H\n2000, 1\n2001, x\n")
    assert not tab._load_input(path)
    assert tab.state.input_series is None
    assert "line 3" in messages.calls[0][2]


def test_observations_require_input(tab, messages):
    assert not tab._load_target(str(PAPER / "tritium_observations.csv"))
    assert tab.state.target_series is None
    assert messages.calls[0][1] == "No Input Series"


def test_observation_error_keeps_state(tab, messages, tmp_path):
    _yearly(tab)
    tab._load_input(str(PAPER / "tritium_input.csv"))
    path = write(tmp_path, "Date, 3H\n1800, 5\n")  # before the input period
    assert not tab._load_target(path)
    assert tab.state.target_series is None
    assert "outside the period" in messages.calls[0][2]


def test_reloading_input_realigns_observations(tab, messages, tmp_path):
    _yearly(tab)
    tab._load_input(str(PAPER / "tritium_input.csv"))
    tab._load_target(str(PAPER / "tritium_observations.csv"))
    # shorter input series that still covers all observations
    lines = (PAPER / "tritium_input.csv").read_text().splitlines()
    short = [lines[0]] + [ln for ln in lines[1:] if int(ln.split(",")[0]) >= 1950]
    assert tab._load_input(write(tmp_path, "\n".join(short), "short.csv"))
    n = len(short) - 1
    assert tab.state.target_series[1].shape == (n, 1)
    assert np.sum(~np.isnan(tab.state.target_series[1])) == 35
    assert messages.calls == []

    # input series that ends before the last observation -> observations removed
    early = [ln for ln in short if not ln[:4].isdigit() or int(ln[:4]) <= 1980]
    assert tab._load_input(write(tmp_path, "\n".join(early), "early.csv"))
    assert tab.state.target_series is None
    assert tab.state.target_data is None
    assert messages.calls[-1][1] == "Observations Removed"


def test_tracer_count_mismatch_is_reported(tab, messages):
    _yearly(tab)
    tab.cb_t1.setCurrentText("Tritium")
    tab.cb_t2.setCurrentText("Krypton-85")
    assert tab._load_input(str(PAPER / "tritium_input.csv"))  # loaded anyway
    assert messages.kinds() == ["information"]
    assert tab.state.input_series is not None


def test_frequency_change_clears_data(tab, messages):
    _yearly(tab)
    tab._load_input(str(PAPER / "tritium_input.csv"))
    tab._load_target(str(PAPER / "tritium_observations.csv"))
    tab.monthly.setChecked(True)
    assert tab.state.input_series is None
    assert tab.state.target_series is None
    assert tab.state.input_data is None and tab.state.target_data is None
    assert tab._target_path is None
