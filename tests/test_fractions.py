"""Calibration of the mixing fraction of binary mixtures (WP3, D15-D17)."""

import numpy as np
import pytest

from PyTracerLab.model import EMUnit, EPMUnit, Model, PMUnit, Solver
from PyTracerLab.model.model import PyTracerLabWarning

T = np.arange(1950, 2021)
LAM_3H = 0.693 / 12.32
X = 5.0 + 150.0 * np.exp(-0.5 * ((T - 1964) / 3.0) ** 2)  # bomb-peak-like input
TRUE = {"young.mtt": 8.0, "young.fraction": 0.6}


def _binary(obs=None, f=0.5, mtt_young=4.0):
    """Young EPM + old PM (practically tracer-free), yearly steps with warm-up."""
    m = Model(
        dt=1.0,
        lambda_=LAM_3H,
        input_series=X,
        target_series=obs,
        steady_state_input=5.0,
        n_warmup_steps=60,
    )
    m.add_unit(
        EPMUnit(mtt=mtt_young, eta=1.5), fraction=f, prefix="young", bounds=[(1, 50), (1, 3)]
    )
    m.add_unit(PMUnit(mtt=200.0), fraction=1.0 - f, prefix="old", bounds=[(50, 500)])
    return m


def _observations():
    truth = _binary(f=TRUE["young.fraction"], mtt_young=TRUE["young.mtt"])
    sim = truth.simulate()
    obs = np.full_like(sim, np.nan)
    obs[10::2] = sim[10::2]  # every second year from 1960
    return obs


def _calibration_model():
    m = _binary(obs=_observations())
    m.set_fixed("young.eta")
    m.set_fixed("old.mtt")
    m.set_fixed("young.fraction", False)
    return m


# ------------------------------------------------------------------ registry
def test_fraction_parameter_only_for_two_units():
    m = Model(dt=1.0, lambda_=LAM_3H, input_series=X)
    m.add_unit(EMUnit(mtt=5.0), fraction=1.0, prefix="a")
    assert m.fraction_key() is None

    m.unit_fractions[0] = 0.7  # as before: the user gives all fractions
    m.add_unit(PMUnit(mtt=20.0), fraction=0.3, prefix="b")
    assert m.fraction_key() == "a.fraction"
    rec = m.params["a.fraction"]
    assert rec["value"] == rec["initial"] == pytest.approx(0.7)
    assert rec["bounds"] == (0.0, 1.0)
    assert rec["fixed"] is True
    assert m.param_keys(free_only=True) == ["a.mtt", "b.mtt"]  # unchanged solver vector

    m.add_unit(PMUnit(mtt=30.0), fraction=0.0, prefix="c")
    assert m.fraction_key() is None
    with pytest.raises(KeyError, match="exactly two units"):
        m.set_fixed("a.fraction", False)


def test_third_unit_rejected_while_fraction_free():
    m = _binary()
    m.set_fixed("young.fraction", False)
    with pytest.raises(ValueError, match="only be calibrated for two units"):
        m.add_unit(PMUnit(mtt=30.0), fraction=0.0, prefix="third")
    assert len(m.units) == 2 and m.fraction_key() == "young.fraction"


def test_add_unit_validation():
    m = _binary()
    with pytest.raises(ValueError, match="already used"):
        m.add_unit(PMUnit(mtt=30.0), fraction=0.0, prefix="old")
    with pytest.raises(ValueError, match="between 0 and 1"):
        Model(dt=1.0, lambda_=LAM_3H, input_series=X).add_unit(PMUnit(mtt=3.0), fraction=1.2)


def test_set_fraction_sets_remainder_and_simulation():
    m = _binary(f=0.5)
    m.set_param("young.fraction", 0.25)
    assert m.unit_fractions == pytest.approx([0.25, 0.75])

    young = Model(dt=1.0, lambda_=LAM_3H, input_series=X, steady_state_input=5.0, n_warmup_steps=60)
    young.add_unit(EPMUnit(mtt=4.0, eta=1.5), fraction=1.0)
    old = Model(dt=1.0, lambda_=LAM_3H, input_series=X, steady_state_input=5.0, n_warmup_steps=60)
    old.add_unit(PMUnit(mtt=200.0), fraction=1.0)
    np.testing.assert_allclose(m.simulate(), 0.25 * young.simulate() + 0.75 * old.simulate())


def test_fraction_bounds_validated():
    m = _binary()
    m.set_bounds("young.fraction", (0.2, 0.8))
    assert m.params["young.fraction"]["bounds"] == (0.2, 0.8)
    for bad in ((-0.1, 0.5), (0.2, 1.1), (0.8, 0.2)):
        with pytest.raises(ValueError, match="0 <= lower <= upper <= 1"):
            m.set_bounds("young.fraction", bad)


def test_fractions_outside_unit_interval_rejected():
    m = _binary()
    m.unit_fractions[:] = [1.2, -0.2]
    with pytest.raises(ValueError, match="between 0 and 1"):
        m.simulate()


def test_fixed_fraction_keeps_user_values():
    """Fixed by default: solvers never write the fractions given by the user."""
    m = _binary(obs=_observations(), f=0.6)
    m.unit_fractions[1] = 0.395  # within the old tolerance, must stay as given
    Solver(m).least_squares()
    assert m.unit_fractions == [0.6, 0.395]


# --------------------------------------------------------------- calibration
def test_least_squares_recovers_fraction():
    m = _calibration_model()
    sol, _ = Solver(m).least_squares()
    assert sol["young.fraction"] == pytest.approx(TRUE["young.fraction"], abs=1e-4)
    assert sol["young.mtt"] == pytest.approx(TRUE["young.mtt"], rel=1e-3)
    assert m.unit_fractions == pytest.approx([0.6, 0.4], abs=1e-4)


def test_differential_evolution_recovers_fraction():
    m = _calibration_model()
    sol, _ = Solver(m).differential_evolution(maxiter=200, popsize=15, tol=1e-8, random_state=1)
    assert sol["young.fraction"] == pytest.approx(TRUE["young.fraction"], abs=1e-3)
    assert sol["young.mtt"] == pytest.approx(TRUE["young.mtt"], rel=1e-2)


def test_dream_samples_fraction_within_bounds():
    m = _calibration_model()
    m.set_bounds("young.fraction", (0.3, 0.9))
    res = Solver(m).dream_sample(
        n_samples=150, burn_in=150, n_chains=4, n_diff_pairs=1, sigma=1.0, random_state=2
    )
    f = res["samples"][:, res["param_names"].index("young.fraction")]
    assert np.all((f >= 0.3) & (f <= 0.9))
    assert np.median(f) == pytest.approx(TRUE["young.fraction"], abs=0.05)
    assert sum(m.unit_fractions) == pytest.approx(1.0)


# ------------------------------------------------------- guidance (D17)
def test_warning_when_more_free_parameters_than_observations():
    obs = np.full(len(T), np.nan)
    obs[[30, 60]] = [20.0, 8.0]  # two observations
    m = _binary(obs=obs)
    m.set_fixed("young.eta")
    m.set_fixed("young.fraction", False)  # free: young.mtt, old.mtt, fraction
    with pytest.warns(PyTracerLabWarning, match=r"more free parameters \(3\) than observations"):
        Solver(m).differential_evolution(maxiter=2, popsize=5, random_state=0)


def test_no_warning_when_determined(recwarn):
    Solver(_calibration_model()).differential_evolution(maxiter=2, popsize=5, random_state=0)
    assert not [w for w in recwarn if issubclass(w.category, PyTracerLabWarning)]


# --------------------------------------------------------------------- report
def test_report_shows_fractions(tmp_path):
    m = _calibration_model()
    m.set_param("young.fraction", 0.6)
    m.param_uncert = {"young.fraction": [0.5, 0.6, 0.7]}
    m.param_map = {"young.fraction": 0.6}
    text = m.write_report(str(tmp_path / "r.txt"), frequency="1 year")
    assert "[young] fraction=0.600" in text
    assert "[old] fraction=0.400 (= 1 - young.fraction)" in text
    assert "young.fraction  value=0.6" in text
    assert "fraction        1%-Quantile=0.3, 50%-Quantile (Median)=0.4, 99%-Quantile=0.5" in text


# ------------------------------------------------------------ O.23 warm-up
def test_warmup_ignores_stable_tracers():
    m = Model(
        dt=1.0,
        lambda_=[LAM_3H, 0.0],
        input_series=np.column_stack([X, np.linspace(0, 10, len(T))]),
        steady_state_input=[5.0, 0.0],
        n_warmup_half_lives=2,
    )
    m.add_unit(EMUnit(mtt=5.0), fraction=1.0)
    m.simulate()
    assert m.n_warmup == 2 * int(0.693 / LAM_3H)


def test_warmup_all_stable_needs_steps():
    m = Model(dt=1.0, lambda_=0.0, input_series=X, steady_state_input=1.0)
    m.add_unit(EMUnit(mtt=5.0), fraction=1.0)
    with pytest.raises(ValueError, match="n_warmup_steps"):
        m.simulate()
    m = Model(dt=1.0, lambda_=0.0, input_series=X)  # no warm-up: no error
    m.add_unit(EMUnit(mtt=5.0), fraction=1.0)
    assert m.simulate().shape == X.shape
