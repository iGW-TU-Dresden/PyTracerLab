"""Units with exact time-step integration: GM, LM, PEM and EPM (WP4, D19–D23).

The closed-form expressions are checked against numerical quadrature, the
limiting cases against each other, and the EPM against the TracerLPM results
of the paper's Example 1 (O.26). Sources of the equations:
``pub/reviews/wp4_equations.md`` (not part of the package).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, special

import PyTracerLab as lab
from PyTracerLab.model.registry import UNIT_REGISTRY

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
LAM = np.log(2.0) / 12.32  # tritium, 1/yr


def _pdf(unit, s):
    """Continuous TTD g(s) of a unit, written out independently of the package."""
    if isinstance(unit, lab.GMUnit):
        a, mtt = unit.alpha, unit.mtt
        return s ** (a - 1) * np.exp(-a * s / mtt) / ((mtt / a) ** a * special.gamma(a))
    if isinstance(unit, lab.LMUnit):
        return np.where(s <= 2 * unit.mtt, 1 / (2 * unit.mtt), 0.0)
    if isinstance(unit, lab.EPMUnit):
        eta, mtt = unit.eta, unit.mtt
        pdf = eta / mtt * np.exp(-eta * s / mtt + eta - 1)
        return np.where(s >= mtt * (1 - 1 / eta), pdf, 0.0)
    if isinstance(unit, lab.PEMUnit):  # Jurgens et al. (2012), Eq. A6
        H = unit.above + unit.screen + unit.below
        n1 = H / (H - unit.above)
        n2 = np.inf if unit.below == 0 else H / unit.below
        tau_aq = unit.mtt_aquifer
        t1, t2 = tau_aq * np.log(n1), tau_aq * np.log(n2)
        pdf = np.exp(-s / tau_aq) / (tau_aq * (1 / n1 - 1 / n2))
        return np.where((s >= t1) & (s <= t2), pdf, 0.0)
    raise TypeError(unit)


UNITS = [
    lab.GMUnit(mtt=8.0, alpha=0.4),
    lab.GMUnit(mtt=8.0, alpha=1.0),
    lab.GMUnit(mtt=25.0, alpha=4.0),
    lab.LMUnit(mtt=2.7),
    lab.LMUnit(mtt=30.0),
    lab.EPMUnit(mtt=15.0, eta=1.5),
    lab.EPMUnit(mtt=7.3, eta=1.0),
    lab.PEMUnit(mtt=12.0, above=10.0, screen=20.0, below=20.0),
    lab.PEMUnit(mtt=12.0, above=0.0, screen=5.0, below=30.0),
    lab.PEMUnit(mtt=12.0, above=5.0, screen=10.0, below=0.0),
]
IDS = [f"{type(u).__name__}-{i}" for i, u in enumerate(UNITS)]


@pytest.mark.parametrize("unit", UNITS, ids=IDS)
@pytest.mark.parametrize("lam", [0.0, LAM])
def test_step_response_equals_quadrature(unit, lam):
    """Each step holds the integral of g(s) exp(-lam s) over the step."""
    dt = 1.0
    tau = np.arange(0.0, 2000.0, dt)
    h = unit.get_impulse_response(tau, dt, lam)
    assert h[0] == 0.0
    for k in (1, 2, 3, 5, 8, 13, 20, 40):
        breaks = [b for b in _breakpoints(unit) if (k - 1) * dt < b < k * dt]
        q = integrate.quad(
            lambda s: _pdf(unit, s) * np.exp(-lam * s),
            (k - 1) * dt,
            k * dt,
            points=breaks or None,
            limit=200,
        )[0]
        assert h[k] == pytest.approx(q, rel=1e-6, abs=1e-12)


def _breakpoints(unit):
    if isinstance(unit, lab.LMUnit):
        return [2 * unit.mtt]
    if isinstance(unit, lab.EPMUnit):
        return [unit.mtt * (1 - 1 / unit.eta)]
    if isinstance(unit, lab.PEMUnit):
        tau_aq = unit.mtt_aquifer
        H = unit.above + unit.screen + unit.below
        pts = [tau_aq * np.log(H / (H - unit.above))]
        if unit.below > 0:
            pts.append(tau_aq * np.log(H / unit.below))
        return pts
    return []


@pytest.mark.parametrize("unit", UNITS, ids=IDS)
def test_unit_area_mean_and_production(unit):
    dt = 0.05
    tau = np.arange(0.0, 1500.0, dt)
    h0 = unit.get_impulse_response(tau, dt, 0.0)
    assert h0.sum() == pytest.approx(1.0, abs=1e-12)
    # step k covers ((k-1) dt, k dt]; its mean lag is about (k - 1/2) dt
    assert np.sum((tau - dt / 2) * h0) == pytest.approx(unit.mtt, rel=2e-3)
    # decay and production responses add up to the response without decay
    h = unit.get_impulse_response(tau, dt, LAM)
    hp = unit.get_impulse_response(tau, dt, LAM, prod=True)
    assert np.allclose(h + hp, h0, atol=1e-15)
    assert np.all(h >= 0) and np.all(hp >= -1e-15)


def test_gm_shape_one_is_exponential_model():
    tau = np.arange(0.0, 500.0, 1.0)
    gm = lab.GMUnit(mtt=10.0, alpha=1.0).get_impulse_response(tau, 1.0, LAM)
    epm = lab.EPMUnit(mtt=10.0, eta=1.0).get_impulse_response(tau, 1.0, LAM)  # exact EM
    assert np.allclose(gm, epm, rtol=1e-12, atol=1e-15)
    em = lab.EMUnit(mtt=10.0).get_impulse_response(tau, 1.0, LAM)  # trapezoid rule
    assert np.abs(gm - em).max() < 1e-3


def test_gm_small_shape_has_finite_response():
    """g(0) is infinite for alpha < 1; the step integrals are finite."""
    tau = np.arange(0.0, 3000.0, 1.0)
    h = lab.GMUnit(mtt=5.0, alpha=0.2).get_impulse_response(tau, 1.0, LAM)
    assert np.all(np.isfinite(h)) and h[1] > 0.5


def test_pem_special_cases():
    """Jurgens et al. (2012): Table 1 (cases 1 and 2), Eqs. 10 and 12."""
    tau = np.arange(0.0, 800.0, 1.0)
    # case 1: whole aquifer screened -> exponential model
    whole = lab.PEMUnit(mtt=20.0, above=0.0, screen=50.0, below=0.0)
    assert whole.mtt_aquifer == pytest.approx(20.0)
    em = lab.EPMUnit(mtt=20.0, eta=1.0).get_impulse_response(tau, 1.0, LAM)
    assert np.allclose(whole.get_impulse_response(tau, 1.0, LAM), em, atol=1e-15)
    # case 3 (TracerLPM's PEM, screen to the base) = EPM with eta = 1 + ln(1 + ratio)
    for ratio in (0.1, 1.0, 3.0):
        pem = lab.PEMUnit(mtt=25.0, above=ratio, screen=1.0, below=0.0)
        assert pem.mtt_aquifer * (1 + np.log(1 + ratio)) == pytest.approx(25.0)  # Eq. 10
        epm = lab.EPMUnit(mtt=25.0, eta=1 + np.log(1 + ratio))  # Eq. 12
        a = pem.get_impulse_response(tau, 1.0, LAM)
        assert np.allclose(a, epm.get_impulse_response(tau, 1.0, LAM), atol=1e-15)
    # case 2: screen from the water table to z2
    pem = lab.PEMUnit(mtt=25.0, above=0.0, screen=30.0, below=20.0)
    n2 = 50.0 / 20.0
    tau_aq = pem.mtt_aquifer
    assert (1 + 1 / (n2 - 1)) * tau_aq * (1 - np.log(n2) / n2 - 1 / n2) == pytest.approx(25.0)
    # only ratios of the thicknesses matter
    scaled = lab.PEMUnit(mtt=25.0, above=0.0, screen=3.0, below=2.0)
    assert np.allclose(
        scaled.get_impulse_response(tau, 1.0, LAM), pem.get_impulse_response(tau, 1.0, LAM)
    )


def test_exepm_equals_epm():
    tau = np.arange(0.0, 500.0, 1.0)
    ex = lab.ExEPMUnit(mtt=12.0, exp_part=2.0, piston_part=1.0)
    epm = lab.EPMUnit(mtt=12.0, eta=1.5)
    assert np.allclose(
        ex.get_impulse_response(tau, 1.0, LAM), epm.get_impulse_response(tau, 1.0, LAM)
    )
    # no piston part: exponential model (used to return a zero response)
    ex0 = lab.ExEPMUnit(mtt=12.0, exp_part=1.0, piston_part=0.0)
    assert ex0.get_impulse_response(tau, 1.0, 0.0).sum() == pytest.approx(1.0)


def _response(unit, mtt):
    unit.set_param_values({"mtt": mtt})
    return unit.get_impulse_response(np.arange(0.0, 200.0, 1.0), 1.0, LAM)


@pytest.mark.parametrize(
    "unit",
    [lab.LMUnit(mtt=1.0), lab.EPMUnit(mtt=1.0, eta=1.6), lab.PEMUnit(1.0, 10.0, 20.0, 20.0)],
    ids=["LM", "EPM", "PEM"],
)
def test_response_changes_smoothly_with_mtt(unit):
    """No plateaus between grid points (D19): the response follows mtt continuously."""
    mtts = np.linspace(2.5, 2.9, 9)
    means = [np.sum(np.arange(200) * _response(unit, m)) for m in mtts]
    steps = np.diff(means)
    assert np.all(steps > 0)  # strictly increasing (decay makes the mean lag < mtt)
    assert steps.max() / steps.min() < 1.5  # no jumps


def test_lm_least_squares_recovers_off_grid_mtt():
    """Gradient-based least squares works because the LM response is smooth in mtt."""
    x = np.genfromtxt(EXAMPLES / "TracerLPM_benchmark_input_yearly.csv", delimiter=",")[:, 1]
    truth = lab.Model(dt=1.0, lambda_=LAM, input_series=x, steady_state_input=6.0)
    truth.add_unit(lab.LMUnit(mtt=7.3), fraction=1.0, prefix="lm")
    obs = np.full(len(x), np.nan)
    idx = np.arange(110, len(x), 4)
    obs[idx] = truth.simulate()[idx]
    m = lab.Model(dt=1.0, lambda_=LAM, input_series=x, target_series=obs, steady_state_input=6.0)
    m.add_unit(lab.LMUnit(mtt=4.0), fraction=1.0, prefix="lm", bounds=[(0.5, 50.0)])
    params, _ = lab.Solver(m).least_squares()
    assert params["lm.mtt"] == pytest.approx(7.3, rel=1e-4)


@pytest.mark.parametrize("case", [1, 2, 3])
def test_epm_matches_tracerlpm_example_1(case):
    """EPM (exact integration) vs. TracerLPM, paper Example 1 (O.26)."""
    mtt, epm_ratio = {1: (15.0, 0.5), 2: (20.0, 0.7), 3: (30.0, 0.8)}[case]
    inp = np.genfromtxt(EXAMPLES / "TracerLPM_benchmark_input_yearly.csv", delimiter=",")
    bm = np.genfromtxt(EXAMPLES / f"TracerLPM_benchmark_simulations_{case}.csv", delimiter=",")
    sel = np.isin(inp[:, 0], bm[:, 0])
    lam = np.log(2.0) / 12.3
    for unit in (
        lab.EPMUnit(mtt=mtt, eta=1.0 + epm_ratio),
        lab.PEMUnit(mtt=mtt, above=np.exp(epm_ratio) - 1.0, screen=1.0, below=0.0),
    ):
        m = lab.Model(
            dt=1.0,
            lambda_=lam,
            input_series=inp[:, 1],
            steady_state_input=6.0,
            n_warmup_half_lives=10,
        )
        m.add_unit(unit, fraction=1.0)
        diff = m.simulate()[sel] - bm[:, 3]
        assert np.abs(diff).max() < 0.5  # TU; was up to 79 TU with the trapezoid rule


def test_pm_response_independent_of_time_step():
    """O.27: the PM output was scaled by 1/dt."""
    for dt in (1.0, 0.5, 1.0 / 12.0):
        m = lab.Model(
            dt=dt, lambda_=LAM, input_series=np.full(int(100 / dt), 10.0), n_warmup_steps=0
        )
        m.add_unit(lab.PMUnit(mtt=10.0), fraction=1.0)
        assert m.simulate()[-1] == pytest.approx(10.0 * np.exp(-LAM * 10.0))


@pytest.mark.parametrize(
    "unit, msg",
    [
        (lab.GMUnit(mtt=10.0, alpha=0.0), "alpha"),
        (lab.GMUnit(mtt=-1.0, alpha=1.0), "mtt"),
        (lab.LMUnit(mtt=0.0), "mtt"),
        (lab.EPMUnit(mtt=10.0, eta=0.9), "eta"),
        (lab.ExEPMUnit(mtt=10.0, exp_part=0.0, piston_part=1.0), "exp_part"),
        (lab.PEMUnit(mtt=10.0, above=1.0, screen=0.0, below=1.0), "screen"),
        (lab.PEMUnit(mtt=10.0, above=-1.0, screen=1.0, below=1.0), "above"),
    ],
)
def test_invalid_parameters_raise(unit, msg):
    with pytest.raises(ValueError, match=msg):
        unit.get_impulse_response(np.arange(0.0, 50.0), 1.0, 0.0)


def test_response_outside_window_comes_from_steady_state():
    """All travel times beyond the input series (PEM screen deep, short series, O.29)."""
    pem = lab.PEMUnit(mtt=500.0, above=40.0, screen=1.0, below=1.0)
    tau = np.arange(0.0, 50.0)
    assert np.abs(pem.get_impulse_response(tau, 1.0, 0.0)).max() == 0.0
    m = lab.Model(dt=1.0, lambda_=LAM, input_series=np.full(50, 100.0), steady_state_input=8.0)
    m.add_unit(pem, fraction=1.0)
    np.testing.assert_allclose(m.simulate(), 8.0 * pem.total_response(LAM))


def test_registry_and_defaults():
    for name, cls in (("GM", lab.GMUnit), ("LM", lab.LMUnit), ("PEM", lab.PEMUnit)):
        assert UNIT_REGISTRY[name] is cls
        unit = cls(**{p["key"]: p["default"] for p in cls.PARAMS})
        assert [p["key"] for p in cls.PARAMS] == list(unit.param_values())
        for p in cls.PARAMS:
            lo, hi = p["bounds"]
            assert lo <= p["default"] <= hi
        h = unit.get_impulse_response(np.arange(0.0, 300.0), 1.0, LAM)
        assert np.all(np.isfinite(h))


def test_pem_geometry_fixed_by_default():
    m = lab.Model(dt=1.0, lambda_=LAM, input_series=np.full(100, 5.0))
    m.add_unit(
        lab.PEMUnit(mtt=10.0, above=10.0, screen=20.0, below=20.0),
        fraction=1.0,
        prefix="pem",
        bounds=[(1.0, 50.0), (0.0, 100.0), (0.1, 100.0), (0.0, 100.0)],
    )
    assert m.param_keys(free_only=True) == ["pem.mtt"]
    m.set_fixed("pem.below", False)
    assert set(m.param_keys(free_only=True)) == {"pem.mtt", "pem.below"}
    # other units: all parameters free by default (unchanged)
    m2 = lab.Model(dt=1.0, lambda_=LAM, input_series=np.full(100, 5.0))
    m2.add_unit(lab.ExEPMUnit(mtt=10.0, exp_part=1.0, piston_part=0.5), fraction=1.0, prefix="epm")
    assert set(m2.param_keys(free_only=True)) == {"epm.mtt", "epm.exp_part", "epm.piston_part"}
