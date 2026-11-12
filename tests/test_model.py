import numpy as np
import pytest

from PyTracerLab.model import EPMUnit, Model, PMUnit, Solver


def make_simple_model(target_series=True):
    x = np.ones(100)
    y = None
    if target_series:
        y = np.ones(100)
    m = Model(dt=1.0, lambda_=0.01, input_series=x, target_series=y)
    m.add_unit(EPMUnit(mtt=5.0, eta=1.2), fraction=0.6, prefix="epm", bounds=[(1, 50), (1, 3)])
    m.add_unit(PMUnit(mtt=10.0), fraction=0.4, prefix="pm", bounds=[(1, 50)])
    return m


def test_registry_roundtrip():
    m = make_simple_model()
    v = m.get_vector()
    m.set_vector(v)
    v2 = m.get_vector()
    assert np.allclose(v, v2)


def test_simulate_shapes():
    m = make_simple_model(target_series=False)
    sim = m.simulate()
    assert sim.shape == (100,)


def test_solver_fits_synthetic():
    # generate synthetic data with known params
    true = Model(dt=1.0, lambda_=0.01, input_series=np.ones(100))
    true.add_unit(EPMUnit(mtt=8.0, eta=1.5), fraction=0.5, prefix="epm")
    true.add_unit(PMUnit(mtt=15.0), fraction=0.5, prefix="pm")
    obs = true.simulate()

    m = Model(dt=1.0, lambda_=0.01, input_series=np.ones(100), target_series=obs)
    m.add_unit(EPMUnit(mtt=5.0, eta=2.0), fraction=0.5, prefix="epm", bounds=[(1, 20), (1, 5)])
    m.add_unit(PMUnit(mtt=10.0), fraction=0.5, prefix="pm", bounds=[(1, 30)])

    m.set_vector(m.get_vector("initial", free_only=False), which="value", free_only=False)
    sim_init = m.simulate()
    err_init = float(np.mean((obs - sim_init) ** 2))

    solver = Solver(m)
    sol, sim = solver.differential_evolution(maxiter=50, popsize=10)
    # at least should reduce error vs initials
    err_final = float(np.mean((obs - sim) ** 2))
    assert err_final < err_init


def test_solver_respects_fixed():
    m = make_simple_model()
    m.set_fixed("epm.mtt", True)
    fixed_before = m.params["epm.mtt"]["value"]

    solver = Solver(m)
    sol, _ = solver.differential_evolution(maxiter=5, popsize=5)
    assert sol["epm.mtt"] == pytest.approx(fixed_before)


def test_solver_rejects_observations_not_on_input_grid():
    """Sparse observations used to give a silently infinite objective (O.15)."""
    m = Model(dt=1.0, lambda_=0.01, input_series=np.ones(100), target_series=np.ones(5))
    m.add_unit(PMUnit(mtt=10.0), fraction=1.0, bounds=[(1, 30)])
    with pytest.raises(ValueError, match="align_to"):
        Solver(m)


def test_solver_rejects_tracer_column_mismatch():
    m = Model(
        dt=1.0,
        lambda_=[0.01, 0.02],
        input_series=np.ones((100, 2)),
        target_series=np.ones((100, 1)),
    )
    m.add_unit(PMUnit(mtt=10.0), fraction=1.0, bounds=[(1, 30)])
    with pytest.raises(ValueError, match="tracer columns"):
        Solver(m)


def test_report_without_observations(tmp_path):
    """O.28: forward run without observations (used to raise UnboundLocalError)."""
    m = make_simple_model(target_series=False)
    text = m.write_report(str(tmp_path / "r.txt"), frequency="1 year")
    assert "No observations given." in text and "MSE: n/a" in text
    assert "epm.mtt" in text


def test_report_with_single_tracer_name(tmp_path):
    """O.28: a single tracer name as string (used to be split into characters)."""
    m = make_simple_model()
    text = m.write_report(str(tmp_path / "r.txt"), frequency="1 year", tracer="Tritium")
    assert "Tracer: Tritium" in text and "Tracer Tritium" in text
    assert "Tracer r" not in text
