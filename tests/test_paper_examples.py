"""Run the example scripts of the PyTracerLab paper (quick settings where available).

The scripts in ``docs/examples/paper`` produce the figures and tables of the
paper. These tests check that they run with the current version and that the
benchmark against TracerLPM (Example 1) still holds.
"""

import csv
import runpy
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

PAPER_DIR = Path(__file__).resolve().parents[1] / "docs" / "examples" / "paper"


def _main(name):
    return runpy.run_path(str(PAPER_DIR / f"{name}.py"), run_name="paper_example")["main"]


def test_example_1_matches_tracerlpm(tmp_path):
    _main("example_1")(out=tmp_path)
    with open(tmp_path / "example_1_differences.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    limits = {"PM": 0.01, "EM": 0.5, "EPM": 0.1, "DM": 12.0}  # TU
    for row in rows:
        assert float(row["max_abs_diff_TU"]) < limits[row["unit"]], row
    assert (tmp_path / "example_1.png").exists()


def test_example_2_runs(tmp_path):
    _main("example_2")(quick=True, out=tmp_path)
    for name in ("fit", "posterior", "free_fraction"):
        assert (tmp_path / f"example_2_{name}.png").exists()
    assert (tmp_path / "example_2_results.csv").read_text(encoding="utf-8").count("\n") > 20


def test_example_3_runs(tmp_path):
    _main("example_3")(quick=True, out=tmp_path)
    text = (tmp_path / "example_3_results.csv").read_text(encoding="utf-8")
    assert text.startswith("quantity,") and "MODFLOW," in text and "age bins median" in text
    assert (tmp_path / "example_3.png").exists()


def test_example_4_matches_tracerlpm(tmp_path):
    _main("example_4")(quick=True, out=tmp_path)
    with open(tmp_path / "example_4_results.csv", encoding="utf-8") as f:
        rows = {r["quantity"]: r for r in csv.DictReader(f)}
    # forward simulation at TracerLPM's best fit (TracerLPM Example 3, 1 July samples)
    assert float(rows["max rel diff to TracerLPM at its best fit (%)"]["gw.mtt"]) < 0.1
    # TracerLPM's objective minimised with PyTracerLab: close to TracerLPM's solution
    mine = rows["PyTracerLab with TracerLPM's objective"]
    assert abs(float(mine["gw.mtt"]) - 6.8) < 0.3
    assert abs(float(mine["gw.fraction"]) - 0.56) < 0.02
    assert float(mine["objective"]) <= float(rows["TracerLPM best fit"]["objective"])
    assert (tmp_path / "example_4.png").exists()


def test_example_5_matches_tracerlpm(tmp_path):
    _main("example_5")(quick=True, out=tmp_path)
    with open(tmp_path / "example_5_results.csv", encoding="utf-8") as f:
        rows = {r["quantity"]: r for r in csv.DictReader(f) if r.get("quantity")}
    # dispersion model (DP 0.8) for mean travel times 1-500 yr vs TracerLPM
    diff = rows["max abs diff DM to TracerLPM (mtt 1-500 yr)"]
    assert float(diff["3H"]) < 0.1  # TU
    assert float(diff["3He(trit)"]) < 0.1  # TU
    assert float(diff["SF6"]) < 0.05  # pptv
    for name in ("tracer_tracer", "posterior"):
        assert (tmp_path / f"example_5_{name}.png").exists()
