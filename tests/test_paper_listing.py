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
    assert set(opt_param) == {"gw.mtt", "gw.fraction", "runoff.mtt"}
    assert all(np.isfinite(v) for v in opt_param.values())
    assert opt_param["runoff.mtt"] == 0.0  # fixed
    # TracerLPM Example 3 (Missouri River, data as distributed with TracerLPM):
    # PyTracerLab finds mtt 6.40 yr, fraction 0.557 (TracerLPM: 6.8 yr, 0.56 with
    # its own objective, the sum of the absolute relative errors)
    assert abs(opt_param["gw.mtt"] - 6.40) < 0.1
    assert abs(opt_param["gw.fraction"] - 0.557) < 0.01
    assert np.asarray(opt_sim).shape == ns["inp"].values.shape
    assert "gw.mtt" in capsys.readouterr().out
