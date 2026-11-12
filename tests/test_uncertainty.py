"""Measurement uncertainty in the solvers: sigma, sigma_rel (WP2)."""

from datetime import datetime

import numpy as np
import pytest
from scipy.optimize import curve_fit
from scipy.stats import norm

from PyTracerLab.model import EMUnit, Model, Solver
from PyTracerLab.model import solver as ms

T = np.arange(1950, 2021)
LAM_3H = 0.693 / 12.32
OBS_ROWS = np.arange(5, len(T), 5)  # every 5th year, starting 1955


def _input(n_tracers=1):
    peak = 5.0 + 150.0 * np.exp(-0.5 * ((T - 1964) / 3.0) ** 2)
    if n_tracers == 1:
        return peak
    # second tracer: stable, numbers about 100 times larger than the first
    ramp = 1000.0 * np.clip((T - 1960) / 60.0, 0.0, None)
    return np.column_stack([peak, ramp])


def _model(n_tracers=1):
    """Model with yearly steps, warm-up and noisy observations every 5 years."""
    x = _input(n_tracers)
    lam = LAM_3H if n_tracers == 1 else [LAM_3H, 0.0]
    ss = 5.0 if n_tracers == 1 else [5.0, 0.0]
    truth = Model(dt=1.0, lambda_=lam, input_series=x, steady_state_input=ss, n_warmup_steps=30)
    truth.add_unit(EMUnit(mtt=12.0), fraction=1.0)
    sim = truth.simulate()
    rng = np.random.default_rng(0)
    obs = np.full_like(sim, np.nan)
    obs[OBS_ROWS] = sim[OBS_ROWS] * (1.0 + 0.05 * rng.standard_normal(sim[OBS_ROWS].shape))
    m = Model(
        dt=1.0,
        lambda_=lam,
        input_series=x,
        target_series=obs,
        steady_state_input=ss,
        n_warmup_steps=30,
        time_steps=[datetime(int(y), 1, 1) for y in T],
    )
    m.add_unit(EMUnit(mtt=5.0), fraction=1.0, bounds=[(1.0, 50.0)])
    return m, obs


# ----------------------------------------------------------------- sigma array
def test_sigma_forms():
    m, obs = _model()
    s = Solver(m)
    observed = ~np.isnan(obs)
    assert s._sigma_matrix(None, None) is None
    for sigma in (2.0, 2, [2.0], np.full(len(T), 2.0)):  # integer sigma: O.19
        sig = s._sigma_matrix(sigma, None)
        assert sig.shape == (len(T), 1)
        assert np.all(sig[observed, 0] == 2.0)
        assert np.all(np.isnan(sig[~observed, 0]))
    # relative uncertainty refers to the observed value
    np.testing.assert_allclose(s._sigma_matrix(None, 0.1), s._sigma_matrix(0.1 * np.abs(obs), None))


def test_sigma_given_wins_relative_fills_gaps():
    m, obs = _model()
    sd = np.full(len(obs), np.nan)
    sd[OBS_ROWS[:3]] = 7.0
    sig = Solver(m)._sigma_matrix(sd, 0.1)[:, 0]
    assert np.all(sig[OBS_ROWS[:3]] == 7.0)
    np.testing.assert_allclose(sig[OBS_ROWS[3:]], 0.1 * np.abs(obs[OBS_ROWS[3:]]))


def test_missing_sigma_is_an_error():
    m, obs = _model()
    sd = np.full(len(obs), np.nan)
    sd[OBS_ROWS[1:]] = 1.0  # first observation (1955) without uncertainty
    with pytest.raises(ValueError, match=r"1 observation\(s\) have no positive .*1955-01-01"):
        Solver(m).dream_sample(n_samples=5, burn_in=0, n_chains=4, n_diff_pairs=1, sigma=sd)


def test_relative_sigma_of_zero_observation_is_an_error():
    m, _ = _model()
    m.target_series = m.target_series.copy()
    m.target_series[OBS_ROWS[0]] = 0.0
    with pytest.raises(ValueError, match=r"1955-01-01 \(sigma = 0\)"):
        Solver(m).least_squares(sigma_rel=0.05)


def test_wrong_sigma_shape_is_an_error():
    s = Solver(_model()[0])
    with pytest.raises(ValueError, match=r"sigma has shape \(7,\)"):
        s._sigma_matrix(np.ones(7), None)
    with pytest.raises(ValueError, match="sigma_rel has shape"):
        s._sigma_matrix(None, np.ones(len(T)))


def test_per_observation_sigma_with_steady_state():
    """sigma refers to the observations; the steady state adds no rows (O.29)."""
    m, obs = _model()
    s = Solver(m)
    sd = np.where(np.isnan(obs), np.nan, 1.0 + 0.1 * np.abs(obs))
    before = s._sigma_matrix(sd, None)
    m.simulate()  # used to prepend a warm-up to target_series
    assert m.n_warmup == 0 and len(m.target_series) == len(obs)
    np.testing.assert_array_equal(s._sigma_matrix(sd, None), before)
    sd[OBS_ROWS[0]] = np.nan
    with pytest.raises(ValueError, match="1955-01-01"):
        s._sigma_matrix(sd, None)


def test_multi_tracer_sigma():
    m, obs = _model(n_tracers=2)
    s = Solver(m)
    n = len(T)
    per_tracer = s._sigma_matrix([1.0, 50.0], None)
    per_obs = s._sigma_matrix(np.column_stack([np.full(n, 1.0), np.full(n, 50.0)]), None)
    np.testing.assert_array_equal(per_tracer, per_obs)
    # relative uncertainty for tracer 1 only
    sig = s._sigma_matrix([np.nan, 50.0], 0.05)
    rows = OBS_ROWS
    np.testing.assert_allclose(sig[rows, 0], 0.05 * np.abs(obs[rows, 0]))
    assert np.all(sig[rows, 1] == 50.0)
    with pytest.raises(ValueError, match=r"\(tracer 1\)"):
        s._sigma_matrix([np.nan, 50.0], None)
    s.differential_evolution(maxiter=5, popsize=5, sigma=[1.0, 50.0], random_state=1)


# ------------------------------------------------------------------- solvers
SOLVER_CALLS = [
    ("differential_evolution", dict(maxiter=20, popsize=8, random_state=1)),
    ("least_squares", dict()),
    ("mcmc_sample", dict(n_samples=50, burn_in=20, random_state=1)),
    ("dream_sample", dict(n_samples=30, burn_in=10, n_chains=4, n_diff_pairs=1, random_state=1)),
]


@pytest.mark.parametrize("method,kwargs", SOLVER_CALLS)
def test_constant_sigma_array_equals_scalar(method, kwargs):
    results = []
    for sigma in (2.0, np.full(len(T), 2.0)):
        m, _ = _model()
        results.append(getattr(Solver(m), method)(sigma=sigma, **kwargs))
    if isinstance(results[0], tuple):
        assert results[0][0] == results[1][0]
        np.testing.assert_array_equal(results[0][1], results[1][1])
    else:
        np.testing.assert_array_equal(results[0]["samples"], results[1]["samples"])
        np.testing.assert_array_equal(results[0]["logpost"], results[1]["logpost"])


def test_relative_sigma_equals_explicit_array():
    m1, obs = _model()
    m2, _ = _model()
    kw = dict(maxiter=20, popsize=8, random_state=1)
    sol_rel, _ = Solver(m1).differential_evolution(sigma_rel=0.1, **kw)
    sol_abs, _ = Solver(m2).differential_evolution(sigma=0.1 * np.abs(obs), **kw)
    assert sol_rel == sol_abs


def test_least_squares_matches_curve_fit_with_per_observation_sigma():
    m, obs = _model()
    rows = OBS_ROWS
    sd = np.full(len(obs), np.nan)
    sd[rows] = 0.5 + 0.1 * np.abs(obs[rows])  # heteroscedastic
    sol, _ = Solver(m).least_squares(sigma=sd)

    def f(_x, mtt):
        ref = Model(
            dt=1.0, lambda_=LAM_3H, input_series=_input(), steady_state_input=5.0, n_warmup_steps=30
        )
        ref.add_unit(EMUnit(mtt=mtt), fraction=1.0)
        return ref.simulate()[rows]

    popt, _ = curve_fit(
        f,
        rows,
        obs[rows],
        p0=[5.0],
        bounds=([1.0], [50.0]),
        sigma=sd[rows],
        absolute_sigma=True,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    assert sol["u0.mtt"] == pytest.approx(popt[0], rel=1e-6)


# --------------------------------------------------------------- likelihood
def _loglik_data():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(20, 2))
    y[::3, 0] = np.nan
    sim = y + rng.normal(scale=0.3, size=y.shape)
    sim[np.isnan(y)] = 0.0
    return y, sim


def test_loglik_known_sigma_is_gaussian():
    y, sim = _loglik_data()
    rng = np.random.default_rng(4)
    sig = np.where(np.isnan(y), np.nan, rng.uniform(0.1, 1.0, size=y.shape))
    mask = ~np.isnan(y)
    expected = norm.logpdf(y[mask], loc=sim[mask], scale=sig[mask]).sum()
    assert Solver._loglik(y, sim, sig) == pytest.approx(expected)
    # failed simulation at an observation
    bad = sim.copy()
    bad[1, 1] = np.nan
    assert Solver._loglik(y, bad, sig) == -np.inf


def test_loglik_without_sigma_marginalises_per_tracer():
    """sigma=None: one unknown standard deviation per tracer (O.21)."""
    y, sim = _loglik_data()
    expected = 0.0
    for k in range(2):
        m = ~np.isnan(y[:, k])
        expected -= 0.5 * m.sum() * np.log(np.sum((sim[m, k] - y[m, k]) ** 2))
    assert Solver._loglik(y, sim, None) == pytest.approx(expected)
    # changing the unit of one tracer does not change the comparison of two simulations
    other = sim + 0.1
    scale = np.array([1.0, 1000.0])
    diff = Solver._loglik(y, sim, None) - Solver._loglik(y, other, None)
    diff_scaled = Solver._loglik(y * scale, sim * scale, None) - Solver._loglik(
        y * scale, other * scale, None
    )
    assert diff == pytest.approx(diff_scaled)


# --------------------------------------------------------------- GUI wrappers
@pytest.mark.parametrize(
    "key,method",
    [
        ("de", "differential_evolution"),
        ("lsq", "least_squares"),
        ("mcmc", "mcmc_sample"),
        ("dream", "dream_sample"),
    ],
)
def test_gui_wrappers_pass_sigma_unchanged(monkeypatch, key, method):
    """Per-observation sigma with NaN entries used to be dropped silently (O.22)."""
    captured = {}

    class Stop(Exception):
        pass

    def fake(self, **kwargs):
        captured.update(kwargs)
        raise Stop

    monkeypatch.setattr(ms.Solver, method, fake)
    m, obs = _model()
    sd = np.where(np.isnan(obs), np.nan, 1.0)
    with pytest.raises(Stop):
        ms.run_solver(m, key, {"sigma": sd, "sigma_rel": 0.05})
    assert captured["sigma"] is sd
    assert captured["sigma_rel"] == 0.05
