"""Age-bin model: discrete, shape-free travel time distribution (WP5, D24-D29)."""

import numpy as np
import pytest
from scipy import integrate, stats

import PyTracerLab as lab
from PyTracerLab.model.model import PyTracerLabWarning

LAM = np.log(2.0) / 12.32  # tritium, 1/yr
T = np.arange(1950, 2021)
X = 5.0 + 150.0 * np.exp(-0.5 * ((T - 1964) / 3.0) ** 2)  # bomb-peak-like input
EDGES = [0.0, 5.0, 15.0, 30.0, 60.0]
TRUE_W = np.array([0.3, 0.2, 0.4, 0.1])


def _model(obs=None, weights=None, edges=EDGES):
    m = lab.Model(dt=1.0, lambda_=LAM, input_series=X, target_series=obs, steady_state_input=5.0)
    m.add_unit(lab.AgeBinUnit(edges, weights), fraction=1.0, prefix="bins")
    return m


def _observations():
    sim = _model(weights=TRUE_W).simulate()
    obs = np.full_like(sim, np.nan)
    obs[10:] = sim[10:]  # yearly from 1960
    return obs


# ------------------------------------------------------------ parametrisation
@pytest.mark.parametrize(
    "w",
    [[0.3, 0.2, 0.4, 0.1], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.2, 0.0, 0.8, 0.0]],
)
def test_weights_coordinates_round_trip(w):
    unit = lab.AgeBinUnit(EDGES, w)
    np.testing.assert_allclose(unit.weights, w, atol=1e-12)
    coords = list(unit.coords_from_weights(w).values())
    np.testing.assert_allclose(unit.weights_from(coords), w, atol=1e-12)


def test_every_point_of_the_box_is_a_histogram():
    unit = lab.AgeBinUnit(np.arange(7.0))  # 6 bins, 5 coordinates
    rng = np.random.default_rng(0)
    u = rng.uniform(size=(5000, 5))
    u[:10] = rng.integers(0, 2, size=(10, 5))  # corners of the box
    w = unit.weights_from(u)
    assert w.shape == (5000, 6)
    assert np.all(w >= 0.0)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)


def test_uniform_coordinates_give_flat_dirichlet():
    """Default uniform prior in the box = every histogram equally probable (D24)."""
    k = 5
    unit = lab.AgeBinUnit(np.arange(k + 1.0))
    w = unit.weights_from(np.random.default_rng(1).uniform(size=(200_000, k - 1)))
    np.testing.assert_allclose(w.mean(axis=0), 1.0 / k, atol=3e-3)
    for j in range(k):  # marginals of the flat Dirichlet: Beta(1, K - 1)
        assert stats.kstest(w[::10, j], stats.beta(1, k - 1).cdf).pvalue > 0.01


def test_two_bins_coordinate_is_the_weight():
    unit = lab.AgeBinUnit([0.0, 10.0, 30.0], [0.3, 0.7])
    assert unit.param_values() == {"u1": pytest.approx(0.3)}
    np.testing.assert_allclose(unit.weights_from([0.42]), [0.42, 0.58])


def test_default_weights_equal_and_derived_quantities():
    unit = lab.AgeBinUnit(EDGES)
    np.testing.assert_allclose(unit.weights, 0.25)
    w = TRUE_W
    unit = lab.AgeBinUnit(EDGES, w)
    np.testing.assert_allclose(unit.centres, [2.5, 10.0, 22.5, 45.0])
    assert unit.mtt == pytest.approx(w @ [2.5, 10.0, 22.5, 45.0])
    np.testing.assert_allclose(unit.cumulative, [0.0, 0.3, 0.5, 0.9, 1.0])
    assert "AgeBinUnit(edges=[0, 5, 15, 30, 60]" in repr(unit)


# ------------------------------------------------------------------ response
@pytest.mark.parametrize("lam", [0.0, LAM])
def test_step_response_equals_quadrature(lam):
    edges = [0.0, 2.3, 7.9, 15.2, 40.5]  # off the time grid
    w = [0.1, 0.35, 0.25, 0.3]
    unit = lab.AgeBinUnit(edges, w)
    h = unit.get_impulse_response(np.arange(0.0, 100.0), 1.0, lam)
    assert h[0] == 0.0

    def pdf(s):
        k = np.searchsorted(edges, s, side="left") - 1
        return w[k] / (edges[k + 1] - edges[k]) if 0 <= k < len(w) else 0.0

    for k in range(1, 45):
        pts = [e for e in edges if k - 1 < e < k]
        q = integrate.quad(lambda s: pdf(s) * np.exp(-lam * s), k - 1, k, points=pts or None)[0]
        assert h[k] == pytest.approx(q, rel=1e-9, abs=1e-14)


def test_unit_area_mean_and_production():
    unit = lab.AgeBinUnit([0.0, 2.3, 7.9, 15.2, 40.5], [0.1, 0.35, 0.25, 0.3])
    dt = 0.05
    tau = np.arange(0.0, 60.0, dt)
    h0 = unit.get_impulse_response(tau, dt, 0.0)
    assert h0.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.sum((tau - dt / 2) * h0) == pytest.approx(unit.mtt, rel=1e-3)
    h = unit.get_impulse_response(tau, dt, LAM)
    hp = unit.get_impulse_response(tau, dt, LAM, prod=True)
    np.testing.assert_allclose(h + hp, h0, atol=1e-15)
    assert np.all(h >= 0) and np.all(hp >= -1e-15)


def test_one_bin_equals_linear_model():
    tau = np.arange(0.0, 200.0)
    lm = lab.LMUnit(mtt=7.3)
    one = lab.AgeBinUnit([0.0, 14.6])
    assert one.param_values() == {} and one.mtt == pytest.approx(7.3)
    for prod in (False, True):
        np.testing.assert_allclose(
            one.get_impulse_response(tau, 1.0, LAM, prod),
            lm.get_impulse_response(tau, 1.0, LAM, prod),
            atol=1e-15,
        )


def test_simulation_is_weighted_sum_of_bins():
    """The output is linear in the weights (one uniform block per bin)."""
    sim = _model(weights=TRUE_W).simulate()
    parts = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = lab.Model(dt=1.0, lambda_=LAM, input_series=X, steady_state_input=5.0)
        m.add_unit(lab.AgeBinUnit([a, b]), fraction=1.0)
        parts.append(m.simulate())
    np.testing.assert_allclose(sim, TRUE_W @ np.array(parts), rtol=1e-12)


# ------------------------------------------------------- simulation window
def test_bins_older_than_the_input_series():
    """A wide old bin beyond the 71-year input gets the steady-state input (O.29)."""
    edges, w = [0.0, 30.0, 1000.0], [0.6, 0.4]
    sim = _model(weights=w, edges=edges).simulate()
    # the same steady state written out as a long constant input before the series
    ref = lab.Model(
        dt=1.0, lambda_=LAM, input_series=np.r_[np.full(1200, 5.0), X], steady_state_input=0.0
    )
    ref.add_unit(lab.AgeBinUnit(edges, w), fraction=1.0)
    np.testing.assert_allclose(sim, ref.simulate()[1200:], rtol=1e-10, atol=1e-12)


# ----------------------------------------------------------------- registry
def test_registry_keys_bounds_and_natural_order():
    m = lab.Model(dt=1.0, lambda_=LAM, input_series=np.full(100, 5.0))
    m.add_unit(lab.AgeBinUnit(np.arange(13.0)), fraction=1.0, prefix="bins")  # 12 bins
    keys = [f"bins.u{i}" for i in range(1, 12)]
    assert m.param_keys(free_only=True) == keys  # u10 after u9, all free
    assert m.get_bounds() == [(0.0, 1.0)] * 11  # registered automatically
    m.set_bounds("bins.u3", (0.1, 0.9))
    for bad in ((-0.1, 0.5), (0.2, 1.5)):
        with pytest.raises(ValueError, match="0 <= lower <= upper <= 1"):
            m.set_bounds("bins.u3", bad)
    with pytest.raises(ValueError, match="0 <= lower <= upper <= 1"):
        lab.Model(dt=1.0, lambda_=LAM, input_series=np.full(100, 5.0)).add_unit(
            lab.AgeBinUnit(EDGES), fraction=1.0, prefix="b", bounds=[(0.0, 2.0)] * 3
        )


def test_parameter_order_of_existing_units_unchanged():
    m = lab.Model(dt=1.0, lambda_=LAM, input_series=np.full(100, 5.0))
    m.add_unit(lab.PEMUnit(mtt=10.0, above=1.0, screen=2.0, below=3.0), 0.5, prefix="pem")
    m.add_unit(lab.DMUnit(mtt=10.0, DP=0.1), 0.5, prefix="dm")
    expected = sorted(
        m.params, key=lambda k: (m.params[k]["unit_index"], m.params[k]["local_name"])
    )
    assert m.param_keys() == expected  # same as the plain string sort used before


def test_binary_mixture_with_old_component():
    m = lab.Model(dt=1.0, lambda_=LAM, input_series=X, steady_state_input=5.0)
    m.add_unit(lab.AgeBinUnit(EDGES), fraction=0.8, prefix="bins")
    m.add_unit(lab.PMUnit(mtt=100.0), fraction=0.2, prefix="old", bounds=[(50.0, 120.0)])
    assert m.fraction_key() == "bins.fraction"
    assert m.param_keys()[:4] == ["bins.fraction", "bins.u1", "bins.u2", "bins.u3"]
    assert np.all(np.isfinite(m.simulate()))


@pytest.mark.parametrize(
    "edges, weights, msg",
    [
        ([5.0], None, "at least two"),
        ([0.0, 5.0, 5.0], None, "strictly increasing"),
        ([-1.0, 5.0], None, ">= 0"),
        ([0.0, np.inf], None, "finite"),
        (EDGES, [0.5, 0.5], "give 4 weights"),
        (EDGES, [0.5, 0.6, -0.1, 0.0], ">= 0"),
        (EDGES, [0.3, 0.3, 0.3, 0.0], "sum to 1"),
    ],
)
def test_invalid_input_rejected(edges, weights, msg):
    with pytest.raises(ValueError, match=msg):
        lab.AgeBinUnit(edges, weights)


def test_invalid_coordinate_rejected():
    unit = lab.AgeBinUnit(EDGES)
    unit.set_param_values({"u2": 1.2})
    with pytest.raises(ValueError, match="'u2' must be between 0 and 1"):
        unit.get_impulse_response(np.arange(0.0, 100.0), 1.0, LAM)


# -------------------------------------------------------------- calibration
def _calibration_model():
    return _model(obs=_observations())  # starts from equal weights


def test_least_squares_recovers_histogram():
    m = _calibration_model()
    lab.Solver(m).least_squares()
    np.testing.assert_allclose(m.units[0].weights, TRUE_W, atol=1e-4)


def test_differential_evolution_recovers_histogram():
    m = _calibration_model()
    lab.Solver(m).differential_evolution(maxiter=300, popsize=15, tol=1e-10, random_state=3)
    np.testing.assert_allclose(m.units[0].weights, TRUE_W, atol=1e-2)


def test_dream_samples_are_histograms():
    m = _calibration_model()
    res = lab.Solver(m).dream_sample(
        n_samples=150, burn_in=300, n_chains=6, n_diff_pairs=1, sigma=1.0, random_state=4
    )
    bins = m.units[0]
    w = bins.weights_from_samples(res, "bins")
    assert w.shape == (res["samples"].shape[0], 4)
    assert np.all(w >= 0) and np.allclose(w.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.median(w, axis=0), TRUE_W, atol=0.1)
    assert np.median(w @ bins.centres) == pytest.approx(TRUE_W @ bins.centres, rel=0.1)


def test_weights_from_samples_with_fixed_coordinate():
    bins = lab.AgeBinUnit(EDGES, TRUE_W)
    u2 = bins.param_values()["u2"]
    res = {"param_names": ["other.mtt", "bins.u1", "bins.u3"], "samples": [[9.0, 0.5, 0.5]]}
    expected = bins.weights_from([[0.5, u2, 0.5]])
    np.testing.assert_allclose(bins.weights_from_samples(res, "bins"), expected)
    with pytest.raises(KeyError, match="no coordinates"):
        bins.weights_from_samples(res, "nope")


def test_warning_when_more_bins_than_observations():
    obs = np.full(len(T), np.nan)
    obs[[30, 50, 70]] = [20.0, 10.0, 8.0]
    m = _model(obs=obs, edges=[0, 5, 10, 20, 30, 45, 60])  # 5 coordinates
    with pytest.warns(PyTracerLabWarning, match=r"more free parameters \(5\) than observations"):
        lab.Solver(m).differential_evolution(maxiter=2, popsize=5, random_state=0)


def test_report_lists_weights(tmp_path):
    m = _model(obs=_observations(), weights=TRUE_W)
    text = m.write_report(str(tmp_path / "report.txt"), frequency="1 year")
    assert "bin weights     0.3000, 0.2000, 0.4000, 0.1000" in text
    assert "younger than    0.0000, 0.3000, 0.5000, 0.9000, 1.0000" in text
    assert f"mean travel time {TRUE_W @ [2.5, 10.0, 22.5, 45.0]:.6g}" in text


def test_exported_at_package_level():
    assert "AgeBinUnit" in lab.__all__
    from PyTracerLab.model import AgeBinUnit

    assert AgeBinUnit is lab.AgeBinUnit
