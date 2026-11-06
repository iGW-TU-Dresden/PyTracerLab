"""Run Listing 1 of the PyTracerLab paper exactly as printed in the manuscript."""

import runpy
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1] / "docs" / "examples" / "paper"


def test_listing_1_runs_as_printed(monkeypatch, capsys):
    # the listing reads its data files with relative paths
    monkeypatch.chdir(PAPER_DIR)
    ns = runpy.run_path(str(PAPER_DIR / "listing_1.py"), run_name="__main__")

    opt_param = ns["opt_param"]
    opt_sim = ns["opt_sim"]
    assert set(opt_param) == {"u0.mtt", "u0.eta"}
    assert all(np.isfinite(v) for v in opt_param.values())
    assert 0.5 <= opt_param["u0.mtt"] <= 100.0
    assert 1.0 <= opt_param["u0.eta"] <= 5.0
    assert np.asarray(opt_sim).shape == ns["inp"].values.shape
    assert "u0.mtt" in capsys.readouterr().out
