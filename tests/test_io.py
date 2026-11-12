"""Tests for reading tracer data files (:mod:`PyTracerLab.model.io`)."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from PyTracerLab.model.io import DataFileError, TracerSeries, read_series

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
# example CSV files that are not time series (travel times / TTD weights)
NOT_TIME_SERIES = {"tracer_tracer_sim_TracerLPM.csv", "tts_weights_modflow.csv"}
TIME_SERIES_FILES = sorted(p for p in EXAMPLES.glob("*.csv") if p.name not in NOT_TIME_SERIES)


def write(tmp_path, text, name="data.csv", encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


# --------------------------------------------------------------------------
# backward compatibility: all existing example files still load
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", TIME_SERIES_FILES, ids=lambda p: p.name)
def test_existing_example_files_load(path):
    s = read_series(path)
    assert len(s) > 1
    assert s.step is not None
    assert len(s.times) == len(s)


@pytest.mark.parametrize(
    "path",
    [p for p in TIME_SERIES_FILES if ";" not in p.read_text().splitlines()[1]],
    ids=lambda p: p.name,
)
def test_existing_comma_files_match_genfromtxt(path):
    """The reader returns the same numbers as the np.genfromtxt calls in the notebooks."""
    ref = np.genfromtxt(path, delimiter=",", skip_header=1)[:, 1:]
    s = read_series(path)
    got = s.values.reshape(len(s), -1)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_allclose(got[~np.isnan(got)], ref[~np.isnan(ref)])


def test_semicolon_and_decimal_comma_file():
    s = read_series(EXAMPLES / "tracer_tracer_sim_TracerLPM_forward.csv")
    ref = read_series(EXAMPLES / "tracer_tracer_sim_TracerLPM_forward_format.csv")
    assert s.tracers == ["3H", "3He(trit)", "3Ho", "3H/3Ho", "SF6"]
    np.testing.assert_allclose(s.values, ref.values)


# --------------------------------------------------------------------------
# format variants
# --------------------------------------------------------------------------
def test_basic_yearly_file(tmp_path):
    p = write(tmp_path, "# Date, 3H\n1960, 10\n1961, 20\n1962, nan\n")
    s = read_series(p)
    assert isinstance(s, TracerSeries)
    assert s.tracers == ["3H"]
    assert s.step == "1 year"
    assert s.values.shape == (3,)
    assert s.n_tracers == 1
    assert s.uncertainty is None
    assert list(s.times) == [datetime(1960, 1, 1), datetime(1961, 1, 1), datetime(1962, 1, 1)]
    np.testing.assert_array_equal(s.values[:2], [10.0, 20.0])
    assert np.isnan(s.values[2])


@pytest.mark.parametrize(
    "text",
    [
        "Date,3H,SF6\n2000-01,1.5,2\n2000-02,2.5,3\n",  # no '#'
        "# Date, 3H, SF6\n2000-01, 1.5, 2\n2000-02, 2.5, 3\n",  # '#' and spaces
        "Date;3H;SF6\n2000-01;1,5;2\n2000-02;2,5;3\n",  # semicolon + decimal comma
        "Date\t3H\tSF6\n2000-01\t1.5\t2\n2000-02\t2.5\t3\n",  # tab
        "Date 3H SF6\n2000-01 1.5 2\n2000-02 2.5 3\n",  # whitespace
        '"Date","3H","SF6"\n"2000-01","1.5","2"\n"2000-02","2.5","3"\n',  # quoted
        "# tritium and SF6\n# Date,3H,SF6\n2000-01,1.5,2\n# comment\n2000-02,2.5,3\n",
        "Date,3H,SF6,\n2000-01,1.5,2,\n2000-02,2.5,3,\n",  # trailing delimiter
        "﻿Date,3H,SF6\r\n2000-01,1.5,2\r\n2000-02,2.5,3\r\n",  # BOM + CRLF
    ],
)
def test_format_variants(tmp_path, text):
    s = read_series(write(tmp_path, text))
    assert s.tracers == ["3H", "SF6"]
    assert s.step == "1 month"
    np.testing.assert_allclose(s.values, [[1.5, 2.0], [2.5, 3.0]])


def test_legacy_encoding(tmp_path):
    p = write(tmp_path, "Date;Tritium (µTU);SF6\n2000;1,5;2\n2001;2,5;3\n", encoding="cp1252")
    s = read_series(p)
    assert s.tracers[0] == "Tritium (µTU)"


def test_no_header(tmp_path):
    s = read_series(write(tmp_path, "2000,1,2\n2001,3,4\n"))
    assert s.tracers == ["tracer_1", "tracer_2"]


@pytest.mark.parametrize(
    "dates, step",
    [
        (["1990", "1991", "1992"], "1 year"),
        (["1990-01", "1990-02", "1990-03"], "1 month"),
        (["1990/01", "1990/02", "1990/03"], "1 month"),
        (["1990-01-15", "1990-02-15", "1990-03-15"], "1 month"),
        (["15.01.1990", "15.02.1990", "15.03.1990"], "1 month"),
        (["01.1990", "02.1990", "03.1990"], "1 month"),
        (["1990-01", "1990-07", "1991-01"], "6 months"),
        (["1990.0", "1991.0", "1992.0"], "1 year"),
        (["1990.042", "1990.125", "1990.208"], "1 month"),
        (["1990-01-01 00:00:00", "1991-01-01 00:00:00", "1992-01-01 00:00:00"], "1 year"),
        (["1990-01-01", "1990-01-02", "1990-01-03"], "1 day"),
        (["1990-01-01", "1990-01-08", "1990-01-15"], "7 days"),
    ],
)
def test_date_formats_and_steps(tmp_path, dates, step):
    text = "Date,X\n" + "".join(f"{d},1\n" for d in dates)
    assert read_series(write(tmp_path, text)).step == step


def test_decimal_year_value(tmp_path):
    s = read_series(write(tmp_path, "Date,X\n1990.5,1\n1991.5,1\n"))
    assert s.times[0].year == 1990 and s.times[0].month == 7


@pytest.mark.parametrize("token", ["", "nan", "NaN", "NA", "n/a", "-"])
def test_missing_value_tokens(tmp_path, token):
    s = read_series(write(tmp_path, f"Date,X\n2000,{token}\n2001,1\n"))
    assert np.isnan(s.values[0])


def test_missing_trailing_column(tmp_path):
    s = read_series(write(tmp_path, "Date,3H,SF6\n2000,1\n2001,2\n"))
    assert s.tracers == ["3H", "SF6"]
    assert np.all(np.isnan(s.values[:, 1]))


def test_uncertainty_columns(tmp_path):
    s = read_series(write(tmp_path, "Date,3H,SF6,3H_sd\n2000,10,1,0.5\n2001,20,2,\n"))
    assert s.tracers == ["3H", "SF6"]
    assert s.uncertainty.shape == (2, 2)
    assert s.uncertainty[0, 0] == 0.5
    assert np.isnan(s.uncertainty[1, 0])
    assert np.all(np.isnan(s.uncertainty[:, 1]))


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, match",
    [
        ("", "empty"),
        ("Date,X\nabc,1\n", "no data rows"),
        ("Date,X\n2000,1\n20x1,2\n", r"line 3: could not read the date '20x1'"),
        ("Date,X\n2000,1\n2001,<0.5\n", r"line 3, column 'X': could not read the value '<0.5'"),
        ("Date,X\n2000,1\n2001,2\n2003,3\n", r"irregular time step at lines 3 and 4"),
        ("Date,X\n2001,1\n2000,2\n", "not in increasing order"),
        ("Date,X\n2000,1\n", "at least two time steps"),
        ("Date,X,Y_sd\n2000,1,1\n2001,1,1\n", "no matching tracer column 'Y'"),
        ("Date,X,X_sd\n2000,1,-1\n2001,1,1\n", "must be positive"),
    ],
)
def test_errors(tmp_path, text, match):
    with pytest.raises(DataFileError, match=match):
        read_series(write(tmp_path, text))


def test_error_is_value_error(tmp_path):
    with pytest.raises(ValueError):
        read_series(write(tmp_path, "Date,X\n2000,a\n2001,1\n"))


# --------------------------------------------------------------------------
# observations aligned to the input series
# --------------------------------------------------------------------------
@pytest.fixture
def yearly_input(tmp_path):
    text = "Date,3H\n" + "".join(f"{y},{y - 1990}\n" for y in range(1990, 2000))
    return read_series(write(tmp_path, text, name="input.csv"))


def test_align_sparse_observations(tmp_path, yearly_input):
    p = write(tmp_path, "Date,3H\n1992-06-15,5\n1995,8\n1998.7,3\n", name="obs.csv")
    obs = read_series(p, align_to=yearly_input)
    assert len(obs) == len(yearly_input)
    assert list(obs.times) == list(yearly_input.times)
    expected = np.full(10, np.nan)
    expected[[2, 5, 8]] = [5.0, 8.0, 3.0]
    np.testing.assert_array_equal(obs.values, expected)
    assert obs.step == "1 year"


def test_align_full_grid_observations(tmp_path, yearly_input):
    """The previous format (one row per input time step, nan elsewhere) still works."""
    text = "Date,3H\n" + "".join(f"{y},{'nan' if y % 3 else y}\n" for y in range(1990, 2000))
    obs = read_series(write(tmp_path, text, name="obs.csv"), align_to=yearly_input)
    plain = read_series(write(tmp_path, text, name="obs2.csv"))
    np.testing.assert_array_equal(obs.values, plain.values)


def test_align_monthly_input(tmp_path):
    text = "Date,X\n" + "".join(f"2000-{m:02d},1\n" for m in range(1, 13))
    inp = read_series(write(tmp_path, text, name="input.csv"))
    obs = read_series(write(tmp_path, "Date,X\n2000-03-31,7\n", name="obs.csv"), align_to=inp)
    assert obs.values[2] == 7.0
    assert np.sum(~np.isnan(obs.values)) == 1


def test_align_with_uncertainty_and_partial_tracers(tmp_path):
    inp_text = "Date,3H,SF6\n" + "".join(f"{y},1,1\n" for y in range(1990, 2000))
    inp = read_series(write(tmp_path, inp_text, name="input.csv"))
    obs_text = "Date,3H,SF6,3H_sd\n1991-03,5,,0.3\n1991-09,,2,\n"  # same year, other tracer
    obs = read_series(write(tmp_path, obs_text, name="obs.csv"), align_to=inp)
    np.testing.assert_array_equal(obs.values[1], [5.0, 2.0])
    assert obs.uncertainty[1, 0] == 0.3
    assert np.isnan(obs.uncertainty[1, 1])


def test_align_errors(tmp_path, yearly_input):
    def obs(text):
        return read_series(write(tmp_path, text, name="obs.csv"), align_to=yearly_input)

    with pytest.raises(DataFileError, match=r"lines 2 and 3: two observations of 3H"):
        obs("Date,3H\n1992-03,5\n1992-09,6\n")
    with pytest.raises(DataFileError, match=r"line 2: the date 1985 lies outside"):
        obs("Date,3H\n1985,5\n")
    with pytest.raises(DataFileError, match="2 tracer column"):
        obs("Date,3H,SF6\n1992,5,1\n")
    # rows without any value outside the period are ignored
    assert np.all(np.isnan(obs("Date,3H\n1985,nan\n").values))


def test_sparse_observations_without_align_to_give_hint(tmp_path):
    p = write(tmp_path, "Date,3H\n1992,5\n1995,8\n1996,3\n")
    with pytest.raises(DataFileError, match="align_to"):
        read_series(p)


def test_paper_observation_file_aligns():
    inp = read_series(EXAMPLES / "paper" / "tritium_input.csv")
    obs = read_series(EXAMPLES / "paper" / "tritium_observations.csv", align_to=inp)
    legacy = read_series(EXAMPLES / "TracerLPM_benchmark_observations_yearly.csv")
    np.testing.assert_array_equal(obs.values, legacy.values)


def test_templates_load():
    """The template files in docs/examples/templates load as documented."""
    templates = EXAMPLES / "templates"
    inp = read_series(templates / "input_template.csv")
    assert inp.step == "1 year" and inp.tracers == ["3H"]
    obs = read_series(templates / "observations_template.csv", align_to=inp)
    assert obs.tracers == ["3H"]
    assert np.sum(~np.isnan(obs.values)) == 3
    np.testing.assert_array_equal(np.isnan(obs.values), np.isnan(obs.uncertainty))
