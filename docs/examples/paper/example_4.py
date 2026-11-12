"""Example 4 of the PyTracerLab paper: measured tritium time series (Missouri River).

Data and model follow Example 3 of TracerLPM (Jurgens et al., 2012), using the
data as distributed with TracerLPM: yearly tritium input (1850-2020) and 35
measured tritium concentrations in the Missouri River (1963-1997, sampling
dates 1 July). The river water is modelled as a binary mixture of groundwater
(exponential model) and runoff of the current time step (piston flow with zero
travel time); the mean travel time of the groundwater and the mixing fraction
are calibrated.

The yearly input values are used for every month of the year
(``tritium_input_monthly.csv``), which is the step function TracerLPM uses
for yearly input; the monthly time steps place the July samples exactly.

The script compares PyTracerLab with TracerLPM's result for the same data and
model (``example_4_tracerlpm.csv``: TracerLPM simulation at its best fit) and
quantifies the parameter uncertainty with DREAM.

Run from this folder: ``python example_4.py`` (about 1 min; ``--quick`` for a
test run). Figures and tables are written to ``output/``.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import optimize  # noqa: E402

import PyTracerLab as lab  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

# TracerLPM's best fit (BMM-EMM-PFM, PFM age 0; objective: sum of the absolute
# relative errors, 4.377)
TRACERLPM = {"gw.mtt": 6.8, "gw.fraction": 0.56}
SIGMA_REL = 0.2  # assumed relative measurement uncertainty (RMS of the residuals)


def make_model(inp, obs):
    """Model of Listing 1: groundwater (EM) + runoff (PM, travel time 0)."""
    m = lab.Model(
        dt=1 / 12,
        lambda_=lab.decay_constant("Tritium"),
        input_series=inp.values,
        target_series=obs.values,
        steady_state_input=6.0,
        time_steps=inp.times,
    )
    m.add_unit(lab.EMUnit(mtt=5.0), fraction=0.5, prefix="gw", bounds=[(0.5, 50.0)])
    m.add_unit(lab.PMUnit(mtt=0.0), fraction=0.5, prefix="runoff")
    m.set_fixed("runoff.mtt")
    m.set_fixed("gw.fraction", False)
    return m


def simulate_at(m, values):
    for key, val in values.items():
        m.set_param(key, val)
    return m.simulate()


def tracerlpm_objective(m, values, idx, obs):
    """TracerLPM's objective: sum of the absolute relative errors."""
    sim = simulate_at(m, values)
    return float(np.sum(np.abs(sim[idx] - obs[idx]) / obs[idx]))


def main(quick=False, out=OUT):
    out = Path(out)
    out.mkdir(exist_ok=True)
    inp = lab.read_series(HERE / "tritium_input_monthly.csv")
    obs = lab.read_series(HERE / "tritium_observations_monthly.csv", align_to=inp)
    tlpm = lab.read_series(HERE / "example_4_tracerlpm.csv", align_to=inp)
    o = obs.values.ravel()
    idx = np.where(np.isfinite(o))[0]
    m = make_model(inp, obs)
    keys = m.param_keys(free_only=True)

    # 1) forward simulation at TracerLPM's best fit
    sim_tlpm_par = simulate_at(m, TRACERLPM)
    ref = tlpm.values.ravel()[idx]
    diff = sim_tlpm_par[idx] - ref
    # 2) the same objective as TracerLPM, minimised with PyTracerLab simulations
    bounds = m.get_bounds(free_only=True)

    def f(p):
        return tracerlpm_objective(m, dict(zip(keys, p)), idx, o)

    r = optimize.differential_evolution(f, bounds, seed=1, tol=1e-8, maxiter=50 if quick else 1000)
    r = optimize.minimize(f, r.x, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10})
    l1 = dict(zip(keys, r.x))
    # 3) PyTracerLab: weighted least squares (relative uncertainty) with DE
    solver = lab.Solver(m)
    de_x, de_sim = solver.differential_evolution(
        sigma_rel=SIGMA_REL, random_state=42, maxiter=20 if quick else 10000
    )
    # 4) DREAM
    n_s, n_b = (100, 100) if quick else (2000, 2000)
    dream = solver.dream_sample(
        n_samples=n_s,
        burn_in=n_b,
        n_chains=8,
        cr=[0.25, 0.5, 0.75, 1.0],
        sigma_rel=SIGMA_REL,
        return_sim=True,
        random_state=42,
    )
    rhat = solver.gelman_rubin(dream["samples_chain"])
    q = {
        k: np.quantile(dream["samples"][:, i], [0.01, 0.5, 0.99])
        for i, k in enumerate(dream["param_names"])
    }
    corr = np.corrcoef(dream["samples"].T)[0, 1]

    rows = [
        "quantity,gw.mtt,gw.fraction,objective",
        f"TracerLPM best fit,{TRACERLPM['gw.mtt']},{TRACERLPM['gw.fraction']},"
        f"{tracerlpm_objective(m, TRACERLPM, idx, o):.4f}",
        f"PyTracerLab with TracerLPM's objective,{l1['gw.mtt']:.4f},{l1['gw.fraction']:.4f},"
        f"{r.fun:.4f}",
        f"PyTracerLab DE (sigma_rel {SIGMA_REL}),{de_x['gw.mtt']:.4f},{de_x['gw.fraction']:.4f},",
        f"DREAM MAP,{dream['posterior_map']['gw.mtt']:.4f},"
        f"{dream['posterior_map']['gw.fraction']:.4f},",
    ]
    for p, lab_ in ((0, "q01"), (1, "median"), (2, "q99")):
        rows.append(f"DREAM {lab_},{q['gw.mtt'][p]:.4f},{q['gw.fraction'][p]:.4f},")
    rows.append(f"DREAM R-hat,{rhat[0]:.3f},{rhat[1]:.3f},")
    rows.append(f"DREAM correlation,{corr:.3f},,")
    rows.append(f"DREAM outlier chains moved in burn-in,{dream['n_outlier_resets']},,")
    rows.append(f"max abs diff to TracerLPM at its best fit (TU),{np.abs(diff).max():.3f},,")
    rows.append(
        f"max rel diff to TracerLPM at its best fit (%),{100 * np.abs(diff / ref).max():.3f},,"
    )
    (out / "example_4_results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))

    # figure: concentrations and posterior distribution
    t = np.array([d.year + (d.month - 1) / 12 for d in inp.times])
    w = (t >= 1960) & (t < 2000)
    band = np.quantile(dream["sims"].reshape(-1, len(t)), [0.01, 0.5, 0.99], axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    c = "royalblue"
    ax[0].fill_between(
        t[w], band[0, w], band[2, w], color=c, alpha=0.25, lw=0, label="DREAM (1-99 %)"
    )
    ax[0].plot(t[w], de_sim[w], c="red", lw=1.0, label="PyTracerLab (best fit)")
    ax[0].scatter(
        t[idx], ref, marker="x", c="k", s=18, lw=0.8, zorder=50, label="TracerLPM (best fit)"
    )
    ax[0].errorbar(
        t[idx],
        o[idx],
        yerr=SIGMA_REL * o[idx],
        fmt="o",
        mfc="w",
        mec="k",
        ecolor="grey",
        ms=4,
        capsize=2,
        zorder=40,
        label="Observations (±1σ)",
    )
    ax[0].set(
        yscale="log",
        xlim=(1960, 2000),
        xlabel="Time",
        ylabel="Tritium [TU]",
        title="Missouri River: simulated and observed tritium",
    )
    ax[0].grid(True, which="both", alpha=0.5)
    ax[0].legend(fontsize=8)

    s = dream["samples"]
    i_m, i_f = dream["param_names"].index("gw.mtt"), dream["param_names"].index("gw.fraction")
    ax[1].hist2d(s[:, i_m], s[:, i_f], bins=40, cmap="Blues")
    marks = [
        (dream["posterior_map"], dict(marker="o", c="r", s=50, label="MAP (DREAM)")),
        (de_x, dict(marker="s", c="orange", s=25, label="PyTracerLab (best fit)")),
        (TRACERLPM, dict(marker="v", c="w", s=45, label="TracerLPM (best fit)")),
    ]
    for values, style in marks:
        ax[1].scatter(
            values["gw.mtt"], values["gw.fraction"], edgecolor="k", lw=0.6, zorder=10, **style
        )
    ax[1].set(
        xlabel=r"Mean travel time of groundwater $\tau_m$ [yr]",
        ylabel="Fraction of groundwater [-]",
        title="Posterior distribution",
    )
    ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "example_4.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="a few iterations only (test)")
    parser.add_argument("--out", default=OUT, help="output folder (default: output/)")
    args = parser.parse_args()
    main(quick=args.quick, out=args.out)
