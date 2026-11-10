"""DREAM: outlier chains are moved to the best chain during burn-in (O.31)."""

import warnings

import numpy as np
import pytest

import PyTracerLab as lab
from PyTracerLab.model.model import PyTracerLabWarning
from PyTracerLab.model.solver import _outlier_chains


def test_outlier_chains_iqr_rule():
    rng = np.random.default_rng(0)
    hist = rng.normal(-10.0, 0.5, size=(50, 8))
    assert _outlier_chains(hist).size == 0
    hist[:, 5] -= 40.0  # one chain in a region of low posterior density
    np.testing.assert_array_equal(_outlier_chains(hist), [5])


class _TwoModes(lab.Solver):
    """Solver with a fixed, bimodal log-posterior of the single free parameter.

    Narrow modes at 2 (major) and 8 (minor, log density lower by 20): a chain
    started in the minor mode cannot leave it with proposals built from the
    other chains, which are all in the major mode.
    """

    def _simulate_given_free(self, theta):
        return np.atleast_1d(np.asarray(theta, dtype=float))

    def _loglik(self, y_obs, sim, sig):
        x = float(np.ravel(sim)[0])
        return float(
            np.logaddexp(-0.5 * ((x - 2.0) / 0.1) ** 2, -20.0 - 0.5 * ((x - 8.0) / 0.1) ** 2)
        )


def _run(outlier_check):
    target = np.full(10, np.nan)
    target[-1] = 1.0
    m = lab.Model(
        dt=1.0, lambda_=0.05, input_series=np.ones(10), target_series=target, steady_state_input=0.0
    )
    m.add_unit(lab.EMUnit(mtt=2.0), fraction=1.0, bounds=[(0.1, 10.0)])
    start = np.r_[2.0 + 0.01 * np.arange(7), 8.0].reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _TwoModes(m).dream_sample(
            n_samples=200, burn_in=200, start=start, random_state=1, outlier_check=outlier_check
        )


def test_dream_moves_outlier_chain_during_burn_in():
    res = _run(outlier_check=True)
    assert res["n_outlier_resets"] >= 1
    assert np.all(np.abs(res["samples"] - 2.0) < 1.0)  # all chains in the major mode
    assert max(res["gelman_rubin"].values()) < 1.2


def test_no_outlier_check_in_last_tenth_of_burn_in(monkeypatch):
    """Checks at 10 %, ..., 90 % of burn-in (Vrugt et al., 2008: a moved chain
    needs another burn-in before sampling)."""
    import PyTracerLab.model.solver as solver_mod

    calls = []
    original = solver_mod._outlier_chains

    def recording(hist):
        calls.append(len(hist))
        return original(hist)

    monkeypatch.setattr(solver_mod, "_outlier_chains", recording)
    _run(outlier_check=True)  # burn_in = 200
    # history passed = second half of steps 1..k for k = 20, 40, ..., 180
    assert calls == [10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_dream_without_outlier_check_keeps_stuck_chain():
    res = _run(outlier_check=False)
    assert res["n_outlier_resets"] == 0
    stuck = res["samples_chain"][7, :, 0]
    assert np.all(np.abs(stuck - 8.0) < 1.0)
    assert max(res["gelman_rubin"].values()) > 10.0


def test_outlier_rule_needs_four_chains():
    """With 3 chains, Q1 - 2 IQR is always below the lowest chain (O.33)."""
    hist = np.full((50, 3), -10.0)
    hist[:, 0] = -1e6  # one chain stuck far below the others
    assert _outlier_chains(hist).size == 0
    hist = np.full((50, 4), -10.0)
    hist[:, 0] = -1e6
    np.testing.assert_array_equal(_outlier_chains(hist), [0])


def _one_parameter_model():
    target = np.full(10, np.nan)
    target[-1] = 1.0
    m = lab.Model(
        dt=1.0, lambda_=0.05, input_series=np.ones(10), target_series=target, steady_state_input=0.0
    )
    m.add_unit(lab.EMUnit(mtt=2.0), fraction=1.0, bounds=[(0.1, 10.0)])
    return m


def test_default_chains_work_with_default_difference_pairs():
    """Default arguments failed for 1-3 free parameters (3 pairs need 7 chains, O.33)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyTracerLabWarning)
        res = lab.Solver(_one_parameter_model()).dream_sample(
            n_samples=5, burn_in=10, sigma=1.0, random_state=0
        )
    assert res["samples_chain"].shape[0] == 7  # max(4, 2 * 1, 2 * 3 + 1)
    res = lab.Solver(_one_parameter_model()).dream_sample(
        n_samples=5, burn_in=10, n_diff_pairs=1, sigma=1.0, random_state=0
    )
    assert res["samples_chain"].shape[0] == 4  # max(4, 2 * 1, 2 * 1 + 1)


def test_warning_with_three_chains():
    solver = lab.Solver(_one_parameter_model())
    with pytest.warns(PyTracerLabWarning, match="at least 4 chains"):
        solver.dream_sample(
            n_samples=5, burn_in=10, n_chains=3, n_diff_pairs=1, sigma=1.0, random_state=0
        )
    with warnings.catch_warnings():  # no warning without the check
        warnings.simplefilter("error", PyTracerLabWarning)
        solver.dream_sample(
            n_samples=5,
            burn_in=10,
            n_chains=3,
            n_diff_pairs=1,
            sigma=1.0,
            random_state=0,
            outlier_check=False,
        )


def test_too_many_difference_pairs_names_the_needed_chains():
    with pytest.raises(ValueError, match="needs at least 5 chains"):
        lab.Solver(_one_parameter_model()).dream_sample(
            n_samples=5, burn_in=10, n_chains=4, n_diff_pairs=2, sigma=1.0
        )
