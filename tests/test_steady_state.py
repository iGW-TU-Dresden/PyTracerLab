"""Exact steady-state input before the record instead of a simulated warm-up (O.29, D30)."""

import time
import warnings
from datetime import datetime

import numpy as np
import pytest
from scipy import integrate

import PyTracerLab as lab
from PyTracerLab.model.model import PyTracerLabWarning

LAM = np.log(2.0) / 12.32  # tritium, 1/yr
T = np.arange(1950, 2021)
X = 5.0 + 150.0 * np.exp(-0.5 * ((T - 1964) / 3.0) ** 2)  # bomb-peak-like input


def _units():
    return [
        lab.EMUnit(mtt=30.0),
        lab.DMUnit(mtt=30.0, DP=0.2),
        lab.PMUnit(mtt=150.0),  # beyond the 71-year input series
        lab.EPMUnit(mtt=30.0, eta=1.4),
        lab.ExEPMUnit(mtt=30.0, exp_part=2.0, piston_part=1.0),
        lab.GMUnit(mtt=30.0, alpha=0.5),
        lab.LMUnit(mtt=60.0),
        lab.PEMUnit(mtt=30.0, above=5.0, screen=10.0, below=3.0),
        lab.AgeBinUnit([0.0, 10.0, 40.0, 1000.0], [0.3, 0.3, 0.4]),
    ]


IDS = [type(u).__name__ for u in _units()]


def _sim(unit, x, ss, **kw):
    m = lab.Model(dt=1.0, lambda_=LAM, input_series=x, steady_state_input=ss, **kw)
    m.add_unit(unit, fraction=1.0)
    return m.simulate()


# ------------------------------------------------------------ total response
def _dm_pdf(s, mtt, dp):
    return np.exp(-((1 - s / mtt) ** 2) / (4 * dp * s / mtt)) / (
        s * np.sqrt(4 * np.pi * dp * s / mtt)
    )


@pytest.mark.parametrize("prod", [False, True])
def test_total_response_closed_forms(prod):
    """M_lambda = integral of g(s) exp(-lam s) over all travel times."""

    def check(unit, pdf, pts):
        q = sum(
            integrate.quad(lambda s: pdf(s) * np.exp(-LAM * s), a, b, limit=200)[0]
            for a, b in zip([0.0] + pts, pts + [np.inf])
        )
        expected = 1.0 - q if prod else q
        assert unit.total_response(LAM, prod) == pytest.approx(expected, rel=1e-8)

    check(lab.EMUnit(mtt=30.0), lambda s: np.exp(-s / 30.0) / 30.0, [30.0])
    check(lab.DMUnit(mtt=30.0, DP=0.2), lambda s: _dm_pdf(s, 30.0, 0.2), [30.0])
    check(lab.DMUnit(mtt=8.0, DP=0.01), lambda s: _dm_pdf(s, 8.0, 0.01), [6.0, 8.0, 10.0])
    pm = lab.PMUnit(mtt=150.0)
    assert pm.total_response(LAM, prod) == pytest.approx(
        1 - np.exp(-LAM * 150) if prod else np.exp(-LAM * 150)
    )


@pytest.mark.parametrize("unit", _units(), ids=IDS)
def test_total_response_equals_long_window(unit):
    """Exact units: the sum of the response over a window covering the TTD."""
    if isinstance(unit, (lab.EMUnit, lab.DMUnit)):
        pytest.skip("grid units: checked against the closed forms above")
    h = unit.get_impulse_response(np.arange(0.0, 20000.0), 1.0, LAM)
    assert h.sum() == pytest.approx(unit.total_response(LAM), rel=1e-9)
    assert unit.total_response(0.0) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------- steady state
@pytest.mark.parametrize("unit", _units(), ids=IDS)
@pytest.mark.parametrize("prod", [False, True])
def test_constant_input_gives_exact_steady_state(unit, prod):
    """Input equal to the steady state -> c * M_lambda at every step, also for long TTDs."""
    m = lab.Model(
        dt=1.0, lambda_=LAM, input_series=np.full(80, 6.0), steady_state_input=6.0, production=prod
    )
    m.add_unit(unit, fraction=1.0)
    np.testing.assert_allclose(m.simulate(), 6.0 * unit.total_response(LAM, prod), rtol=1e-12)


@pytest.mark.parametrize("unit", _units(), ids=IDS)
def test_steady_state_equals_long_constant_prefix(unit):
    """Same result as writing the steady state out as a long input before the series."""
    n_pre = 3000
    sim = _sim(unit, X, 5.0)
    ref = _sim(unit, np.r_[np.full(n_pre, 5.0), X], 0.0)[n_pre:]
    # grid units (EM, DM) are normalised on their grid: tiny differences
    rtol = 1e-3 if isinstance(unit, (lab.EMUnit, lab.DMUnit)) else 1e-9
    np.testing.assert_allclose(sim, ref, rtol=rtol, atol=1e-9)


def test_long_half_life_monthly_without_cap():
    """14C, monthly steps, mtt 5000 years: exact and fast (used to be capped at 5000 steps)."""
    lam = np.log(2.0) / (5700.0 * 12.0)  # per month
    unit = lab.EMUnit(mtt=5000.0 * 12.0)
    x = np.full(70 * 12, 100.0)
    t0 = time.perf_counter()
    m = lab.Model(dt=1.0, lambda_=lam, input_series=x, steady_state_input=100.0)
    m.add_unit(unit, fraction=1.0)
    sim = m.simulate()
    assert time.perf_counter() - t0 < 1.0
    np.testing.assert_allclose(sim, 100.0 / (1.0 + lam * 5000.0 * 12.0), rtol=1e-12)


def test_warmup_arguments_have_no_effect():
    m = lab.Model(
        dt=1.0,
        lambda_=LAM,
        input_series=X,
        target_series=X.copy(),
        steady_state_input=5.0,
        n_warmup_half_lives=10,
        n_warmup_steps=500,
    )
    m.add_unit(lab.EMUnit(mtt=10.0), fraction=1.0)
    sim = m.simulate()
    assert m.n_warmup == 0 and len(m.input_series) == len(X) and len(m.target_series) == len(X)
    np.testing.assert_array_equal(sim, _sim(lab.EMUnit(mtt=10.0), X, 5.0))


# ----------------------------------------------------- no steady-state input
def test_no_steady_state_input_means_tracer_free_with_warning():
    times = [datetime(int(y), 1, 1) for y in T]
    m = lab.Model(dt=1.0, lambda_=LAM, input_series=X, time_steps=times)
    m.add_unit(lab.EMUnit(mtt=30.0), fraction=1.0)
    with pytest.warns(PyTracerLabWarning) as rec:
        sim = m.simulate()
    msg = str(rec[0].message)
    assert "recharged before the start of the input series (1950-01) contains no tracer" in msg
    assert "set the steady-state input to 0" in msg
    np.testing.assert_allclose(sim, _sim(lab.EMUnit(mtt=30.0), X, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyTracerLabWarning)
        m.simulate()  # warned once per model
        _sim(lab.EMUnit(mtt=30.0), X, 0.0)  # 0 confirms tracer-free water: no warning


def test_report_names_missing_steady_state(tmp_path):
    m = lab.Model(dt=1.0, lambda_=LAM, input_series=X, target_series=X.copy())
    m.add_unit(lab.EMUnit(mtt=5.0), fraction=1.0)
    with pytest.warns(PyTracerLabWarning):
        text = m.write_report(str(tmp_path / "r.txt"), frequency="1 year")
    assert "Steady-state input: none (water older than the input series contains no tracer)" in text
    assert "Warmup" not in text
