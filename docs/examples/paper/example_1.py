"""Example 1 of the PyTracerLab paper: forward simulations compared with TracerLPM.

Four single-unit models (PM, EM, EPM, DM), each with three parameter sets, are
simulated with the yearly tritium input time series of Example 3 of TracerLPM
(Jurgens et al., 2012) and compared with the results of TracerLPM for the same
input and parameters (``example_1_tracerlpm_set*.csv``).

Run from this folder: ``python example_1.py``. The figure and a table of the
differences are written to ``output/``.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import PyTracerLab as lab  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

# parameter sets of TracerLPM (EPM ratio 0.5, 0.7, 0.8 -> eta = 1 + ratio)
SETS = [
    {"mtt": 15.0, "eta": 1.5, "DP": 0.5},
    {"mtt": 20.0, "eta": 1.7, "DP": 0.7},
    {"mtt": 30.0, "eta": 1.8, "DP": 0.8},
]
UNITS = {
    "PM": ("Piston Flow Model", lambda p: lab.PMUnit(mtt=p["mtt"])),
    "EM": ("Exponential Model", lambda p: lab.EMUnit(mtt=p["mtt"])),
    "EPM": ("Exponential Piston Flow Model", lambda p: lab.EPMUnit(mtt=p["mtt"], eta=p["eta"])),
    "DM": ("Dispersion Model", lambda p: lab.DMUnit(mtt=p["mtt"], DP=p["DP"])),
}


def label(name, p):
    """Legend text with the parameter values of one curve."""
    text = rf"$\tau_m$ = {p['mtt']:.0f} yr"
    if name == "EPM":
        text += rf", $\eta$ = {p['eta']:.1f}"
    if name == "DM":
        text += f", DP = {p['DP']:.1f}"
    return text


def simulate(inp, unit):
    """Forward simulation with yearly time steps and 6 TU before 1850 (as in TracerLPM)."""
    m = lab.Model(
        dt=1.0,
        lambda_=lab.decay_constant("Tritium"),
        input_series=inp.values,
        steady_state_input=6.0,
        time_steps=inp.times,
    )
    m.add_unit(unit, fraction=1.0)
    return m.simulate()


def main(out=OUT):
    out = Path(out)
    out.mkdir(exist_ok=True)
    inp = lab.read_series(HERE / "tritium_input.csv")
    ref = [lab.read_series(HERE / f"example_1_tracerlpm_set{i}.csv") for i in (1, 2, 3)]
    sel = np.isin(inp.times, ref[0].times)  # TracerLPM results 1950-2010
    years = np.array([t.year for t in ref[0].times])

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True)
    colors = ["red", "royalblue", "green"]
    dark = ["darkred", "navy", "darkgreen"]
    rows = []
    for ax, (j, (name, (title, make))) in zip(axes.flat, enumerate(UNITS.items())):
        handles = []
        for i, p in enumerate(SETS):
            sim = simulate(inp, make(p))[sel]
            bench = ref[i].values[:, j]
            diff = sim - bench
            rows.append((name, i + 1, np.abs(diff).max(), np.sqrt(np.mean(diff**2))))
            ax.plot(years, sim, c=colors[i], lw=1.5, zorder=100)
            ax.scatter(years, bench, c=dark[i], marker=".", s=20.0, zorder=1000)
            handles.append(Line2D([0], [0], color=colors[i], lw=1.5, label=label(name, p)))
        handles.append(
            Line2D(
                [0],
                [0],
                marker=".",
                color="w",
                markerfacecolor="k",
                markeredgecolor="k",
                markersize=8,
                label="TracerLPM",
            )
        )
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Tritium [TU]")
        ax.grid(True, zorder=0, alpha=0.7)
        ax.set_xlim(years[0], years[-1])
        ax.set_ylim(0.0)
    for ax in axes[1]:
        ax.set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(out / "example_1.png", dpi=400, bbox_inches="tight")

    lines = ["unit,set,max_abs_diff_TU,rmse_TU"]
    lines += [f"{n},{s},{d:.3f},{r:.3f}" for n, s, d, r in rows]
    (out / "example_1_differences.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Differences PyTracerLab - TracerLPM (1950-2010):")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT, help="output folder (default: output/)")
    main(out=parser.parse_args().out)
