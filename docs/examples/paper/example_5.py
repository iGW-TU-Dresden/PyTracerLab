"""Example 5 of the PyTracerLab paper: multi-tracer analysis (tritium, 3He(trit), SF6).

Data follow Example 1 of TracerLPM (Jurgens et al., 2012), as distributed with
TracerLPM: semi-annual input time series of tritium and SF6 (1900-2020) and one
groundwater sample (17 August 2004) with tritium, tritiogenic helium-3 and SF6.
The 3He(trit) and SF6 values are processed values as reported there (3He(trit)
from noble-gas data, SF6 as atmospheric mixing ratio in pptv).

Step 1 (tracer-tracer plots): the concentrations at the sampling date are
simulated for mean travel times of 1-500 yr with several model types (model
comparison) and, for the dispersion model, compared with TracerLPM
(``example_5_tracerlpm.csv``). Step 2: the mean travel time and the dispersion
parameter of the dispersion model are inferred with DREAM, using the
uncertainties given in the observation file.

Run from this folder: ``python example_5.py`` (about 1 min; ``--quick`` for a
test run). Figures and tables are written to ``output/``.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import PyTracerLab as lab  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

JURGENS = {"dm.DP": 0.8, "dm.mtt": 75.3}  # solution of Jurgens et al. (2012), Example 1
MTT = np.geomspace(1.0, 500.0, 91)  # mean travel times of the tracer-tracer plots (yr)
MARKED = {10.0: "o", 30.0: "s", 100.0: "^"}  # mean travel times marked in the model comparison
# model types of the comparison: (label, unit, colour); shape parameters fixed
MODELS = [
    ("PM", lambda: lab.PMUnit(mtt=1.0), "tab:orange"),
    ("EM", lambda: lab.EMUnit(mtt=1.0), "tab:green"),
    (r"EPM ($\eta$ = 1.5)", lambda: lab.EPMUnit(mtt=1.0, eta=1.5), "tab:purple"),
    (r"GM ($\alpha$ = 0.5)", lambda: lab.GMUnit(mtt=1.0, alpha=0.5), "tab:brown"),
    ("DM (DP = 0.8)", lambda: lab.DMUnit(mtt=1.0, DP=0.8), "k"),
]
TRACER_LABELS = {
    "3H": r"Tritium $^3$H [TU]",
    "3He(trit)": r"Tritiogenic helium $^3$He$_\mathrm{trit}$ [TU]",
    "SF6": r"SF$_6$ [pptv]",
}


def make_model(inp, obs=None):
    """Model without units: 3H (decay), 3He(trit) (produced by 3H decay), SF6 (no decay)."""
    lam = lab.decay_constant("Tritium")
    return lab.Model(
        dt=0.5,  # semi-annual time steps, in years
        lambda_=[lam, lam, 0.0],
        input_series=inp.values,  # 3He(trit) column = tritium input (parent)
        target_series=None if obs is None else obs.values,
        production=[False, True, False],
        steady_state_input=[3.0, 3.0, 0.0],  # before 1900: 3 TU tritium, no SF6
        time_steps=inp.times,
    )


def sweep(inp, unit, step, mtt):
    """Concentrations at time step ``step`` for each mean travel time in ``mtt``."""
    m = make_model(inp)
    m.add_unit(unit, fraction=1.0, prefix="u")
    out = np.empty((len(mtt), inp.values.shape[1]))
    for j, t in enumerate(mtt):
        m.set_param("u.mtt", t)
        out[j] = m.simulate()[step]
    return out


def tracer_plot_comparison(ax, curves, o, c1, c2, xlim):
    """Model comparison: one curve per model type, markers at selected mean travel times."""
    for (name, _, colour), r in zip(MODELS, curves):
        lw = 2.0 if name.startswith("DM") else 1.2
        ax.plot(r[:, c1], r[:, c2], c=colour, lw=lw, label=name, zorder=5)
        for t, mk in MARKED.items():
            j = int(np.argmin(np.abs(MTT - t)))
            ax.scatter(
                r[j, c1], r[j, c2], marker=mk, s=22, c=colour, edgecolor="w", lw=0.4, zorder=6
            )
    ax.scatter(
        o[c1], o[c2], marker="X", s=140, c="r", edgecolor="k", zorder=20, label="Observation"
    )
    ax.set(xlim=xlim, ylim=(0, 5.6))


def tracer_plot_benchmark(ax, r, tlpm, o, c1, c2, norm):
    """Dispersion model coloured by mean travel time, TracerLPM, dilution and mixing lines."""
    x, y = r[:, c1], r[:, c2]
    for i in (0, 30, 40, 47, 53, 59, 65, 72, len(MTT) - 1):
        ax.plot(
            [0, x[i]],
            [0, y[i]],
            c="k",
            lw=0.8,
            alpha=0.3,
            ls="--",
            label="Binary mixing with tracer-free water" if i == 0 else None,
        )
    for p in (0.75, 0.5, 0.25):
        ax.plot(
            x * p,
            y * p,
            c="k",
            lw=1.2,
            alpha=0.8 * p,
            label=f"{100 * (1 - p):.0f} % tracer-free water",
        )
    pts = np.column_stack([x, y])
    lc = LineCollection(
        np.stack([pts[:-1], pts[1:]], axis=1), cmap="turbo", norm=norm, lw=6, zorder=10
    )
    lc.set_array(np.sqrt(MTT[:-1] * MTT[1:]))
    ax.plot(x, y, c="k", lw=7, alpha=0.6, zorder=9)
    ax.add_collection(lc)
    (l1,) = ax.plot(tlpm[:, c1], tlpm[:, c2], c="k", lw=1.6, zorder=11)
    (l2,) = ax.plot(tlpm[:, c1], tlpm[:, c2], c="w", ls=":", lw=1.6, zorder=12)
    ax.scatter(
        o[c1], o[c2], marker="X", s=140, c="r", edgecolor="k", zorder=20, label="Observation"
    )
    ax.set(xlim=(0, None), ylim=(0, None))
    return lc, (l1, l2)


def main(quick=False, out=OUT):
    out = Path(out)
    out.mkdir(exist_ok=True)
    inp = lab.read_series(HERE / "example_5_input.csv")
    obs = lab.read_series(HERE / "example_5_observations.csv", align_to=inp)
    step = int(np.where(np.isfinite(obs.values).all(axis=1))[0][0])  # July 2004
    o = obs.values[step]
    tlpm_raw = np.loadtxt(HERE / "example_5_tracerlpm.csv", delimiter=",", skiprows=1)
    tlpm_mtt, tlpm = tlpm_raw[:, 0], tlpm_raw[:, 1:]  # columns 3H, 3He(trit), SF6

    # step 1: tracer-tracer plots
    curves = []
    for _, make, _ in MODELS:  # the dispersion model comes last (timed without start-up costs)
        t0 = time.perf_counter()
        curves.append(sweep(inp, make(), step, MTT))
        t_sweep = time.perf_counter() - t0
    dm = curves[-1]
    diff = sweep(inp, lab.DMUnit(mtt=1.0, DP=0.8), step, tlpm_mtt) - tlpm
    # model comparison: mean travel times for which all three concentrations agree
    # with the observations within one standard deviation
    sd = obs.uncertainty[step]
    fine = np.geomspace(MTT[0], MTT[-1], 200 if quick else 2000)
    fits = []
    for name, make, _ in MODELS:
        z = (np.abs(sweep(inp, make(), step, fine) - o) / sd).max(axis=1)
        inside = fine[z <= 1.0]
        fits.append((name, z.min(), fine[z.argmin()], inside))

    # step 2: DREAM with the dispersion model
    m = make_model(inp, obs)
    m.add_unit(
        lab.DMUnit(mtt=10.0, DP=0.8),
        fraction=1.0,
        bounds=[(1.0, 1000.0), (0.01, 10.0)],
        prefix="dm",
    )
    solver = lab.Solver(m)
    n_s, n_b = (100, 100) if quick else (4000, 4000)
    t0 = time.perf_counter()
    dream = solver.dream_sample(
        n_samples=n_s,
        burn_in=n_b,
        n_chains=8,
        cr=[0.25, 0.5, 0.75, 1.0],
        sigma=obs.uncertainty,
        random_state=42,
    )
    t_dream = time.perf_counter() - t0
    names = dream["param_names"]
    s = dream["samples"]
    rhat = solver.gelman_rubin(dream["samples_chain"])
    q = {k: np.quantile(s[:, i], [0.01, 0.05, 0.5, 0.95, 0.99]) for i, k in enumerate(names)}
    for k, v in dream["posterior_map"].items():
        m.set_param(k, v)
    sim_map = m.simulate()[step]

    rows = ["quantity,3H,3He(trit),SF6"]
    rows.append(
        "max abs diff DM to TracerLPM (mtt 1-500 yr),"
        + ",".join(f"{v:.4f}" for v in np.abs(diff).max(axis=0))
    )
    rows.append("observation," + ",".join(f"{v:g}" for v in o))
    rows.append("sigma," + ",".join(f"{v:g}" for v in obs.uncertainty[step]))
    rows.append("simulated at DREAM MAP," + ",".join(f"{v:.3f}" for v in sim_map))
    rows += ["", "model,smallest max |sim - obs| / sigma,at mtt (yr),mtt range within 1 sigma (yr)"]
    for name, zmin, at, inside in fits:
        within = f"{inside.min():.1f}-{inside.max():.1f}" if inside.size else "none"
        rows.append(f"{name.replace('$', '').replace(chr(92), '')},{zmin:.2f},{at:.1f},{within}")
    rows += ["", "quantity,dm.DP,dm.mtt"]
    rows.append(f"Jurgens et al. (2012),{JURGENS['dm.DP']},{JURGENS['dm.mtt']}")
    rows.append(
        f"DREAM MAP,{dream['posterior_map']['dm.DP']:.4f},{dream['posterior_map']['dm.mtt']:.4f}"
    )
    for p, lab_ in enumerate(("q01", "q05", "median", "q95", "q99")):
        rows.append(f"DREAM {lab_},{q['dm.DP'][p]:.4f},{q['dm.mtt'][p]:.4f}")
    r = dict(zip(names, rhat))
    rows.append(f"DREAM R-hat,{r['dm.DP']:.3f},{r['dm.mtt']:.3f}")
    rows.append(f"DREAM outlier chains moved in burn-in,{dream['n_outlier_resets']},")
    rows.append(f"forward simulation DM (ms),{1000 * t_sweep / len(MTT):.3f},")
    rows.append(f"tracer-tracer sweep DM, {len(MTT)} runs (s),{t_sweep:.3f},")
    rows.append(f"DREAM run time (s),{t_dream:.1f},")
    (out / "example_5_results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))

    # Fig. 7: model comparison (top) and dispersion model vs TracerLPM (bottom)
    fig, ax = plt.subplots(2, 2, figsize=(10, 9), sharey="row")
    pairs = ((0, 2), (1, 2))  # (3H, SF6), (3He(trit), SF6)
    for col, (c1, c2) in enumerate(pairs):
        tracer_plot_comparison(ax[0, col], curves, o, c1, c2, xlim=(0, 8) if c1 == 0 else (0, 45))
    norm = LogNorm(vmin=MTT[0], vmax=MTT[-1])
    for col, (c1, c2) in enumerate(pairs):
        lc, tlpm_line = tracer_plot_benchmark(ax[1, col], dm, tlpm, o, c1, c2, norm)
    for a, (c1, c2) in zip(ax[0], pairs):
        a.set_xlabel(TRACER_LABELS[obs.tracers[c1]])
    for a, (c1, c2) in zip(ax[1], pairs):
        a.set_xlabel(TRACER_LABELS[obs.tracers[c1]])
    for row in ax:
        row[0].set_ylabel(TRACER_LABELS["SF6"])
    for a, letter in zip(ax.ravel(), "abcd"):
        a.grid(True, alpha=0.3)
        a.text(0.02, 0.98, f"({letter})", transform=a.transAxes, va="top", fontweight="bold")
    for a in ax[0]:
        a.set_title("Model comparison")
    for a in ax[1]:
        a.set_title("Dispersion model (DP = 0.8) and TracerLPM")
    handles, labels = ax[0, 1].get_legend_handles_labels()
    for t, mk in MARKED.items():
        handles.append(plt.Line2D([], [], ls="", marker=mk, c="grey", ms=5))
        labels.append(rf"$\tau_m$ = {t:.0f} yr")
    ax[0, 1].legend(handles, labels, fontsize=8, loc="upper right")
    handles, labels = ax[1, 1].get_legend_handles_labels()
    ax[1, 1].legend(handles + [tlpm_line], labels + ["TracerLPM"], fontsize=8, loc="upper right")
    fig.tight_layout()
    cbar = fig.colorbar(lc, ax=ax[1, :], fraction=0.03, pad=0.02)
    cbar.set_label(r"Mean travel time $\tau_m$ [yr]")
    cbar.set_ticks([1, 2, 5, 10, 20, 50, 100, 200, 500])
    cbar.set_ticklabels(["1", "2", "5", "10", "20", "50", "100", "200", "500"])
    cbar.minorticks_off()
    fig.savefig(out / "example_5_tracer_tracer.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    # Fig. 8: joint and marginal posterior distribution
    i_dp, i_mtt = names.index("dm.DP"), names.index("dm.mtt")
    fig = plt.figure(figsize=(4.2, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
    a = fig.add_subplot(gs[1, 0])
    a_x = fig.add_subplot(gs[0, 0], sharex=a)
    a_y = fig.add_subplot(gs[1, 1], sharey=a)
    a.hist2d(s[:, i_dp], s[:, i_mtt], bins=40, cmap="Blues")
    a_x.hist(s[:, i_dp], bins=40, color="steelblue", alpha=0.7)
    a_y.hist(s[:, i_mtt], bins=40, color="steelblue", alpha=0.7, orientation="horizontal")
    a_x.axis("off")
    a_y.axis("off")
    pm = dream["posterior_map"]
    a.scatter(
        pm["dm.DP"],
        pm["dm.mtt"],
        s=60,
        c="r",
        edgecolor="k",
        lw=0.6,
        zorder=10,
        label="MAP (DREAM)",
    )
    a.scatter(
        JURGENS["dm.DP"],
        JURGENS["dm.mtt"],
        s=45,
        marker="v",
        c="w",
        edgecolor="k",
        lw=0.6,
        zorder=11,
        label="Jurgens et al. (2012)",
    )
    a.set(xlabel="Dispersion parameter DP [-]", ylabel=r"Mean travel time $\tau_m$ [yr]")
    a.legend(fontsize=7, loc="lower right")
    fig.savefig(out / "example_5_posterior.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="a few iterations only (test)")
    parser.add_argument("--out", default=OUT, help="output folder (default: output/)")
    args = parser.parse_args()
    main(quick=args.quick, out=args.out)
