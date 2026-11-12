"""Example 2 of the PyTracerLab paper: recovery of known parameters.

Synthetic tritium observations are generated with a binary mixture of two
dispersion model units (20 % with a mean travel time of 2 yr, 80 % with 25 yr)
and the semi-annual tritium input time series of Example 3 of TracerLPM
(Jurgens et al., 2012). Ten observations are perturbed with relative
(observation-specific) Gaussian noise of 5 %. The parameters are then inferred
with all four solvers of PyTracerLab, using a relative measurement uncertainty
of 5 %:

- differential evolution (DE),
- gradient-based least squares (LSQ) from ten random starting points,
- Metropolis-Hastings (MH) and DREAM (Markov chain Monte Carlo).

An additional DREAM run shows the correlation of the parameters if the mixing
fraction is calibrated as well.

Run from this folder: ``python example_2.py`` (about 3 min; ``--quick`` runs
a few iterations only, for testing). Figures and tables are written to
``output/``.
"""

import argparse
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import PyTracerLab as lab  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

DT = 0.5  # semi-annual time steps (yr)
TRUE = {"dm1.mtt": 2.0, "dm1.DP": 0.1, "dm2.mtt": 25.0, "dm2.DP": 0.1, "dm1.fraction": 0.2}
N_OBS = 10
NOISE_REL = 0.05  # relative noise of the observations and assumed uncertainty
SEED = 1234
LABELS = {
    "dm1.mtt": r"$\tau_{m,1}$ [yr]",
    "dm1.DP": r"$DP_1$ [-]",
    "dm2.mtt": r"$\tau_{m,2}$ [yr]",
    "dm2.DP": r"$DP_2$ [-]",
    "dm1.fraction": r"$f_1$ [-]",
}
C_DREAM, C_DE, C_LSQ = "royalblue", "red", "darkorange"


def make_model(inp, target=None, params=None, free_fraction=False):
    """Binary mixture of two dispersion model units (bounds as in the paper)."""
    params = params or {"dm1.mtt": 5.0, "dm1.DP": 0.5, "dm2.mtt": 50.0, "dm2.DP": 0.5}
    m = lab.Model(
        dt=DT,
        lambda_=lab.decay_constant("Tritium"),
        input_series=inp.values,
        target_series=target,
        steady_state_input=6.0,
        time_steps=inp.times,
    )
    bounds = [(1.0, 100.0), (0.001, 5.0)]
    m.add_unit(
        lab.DMUnit(mtt=params["dm1.mtt"], DP=params["dm1.DP"]),
        fraction=TRUE["dm1.fraction"],
        prefix="dm1",
        bounds=bounds,
    )
    m.add_unit(
        lab.DMUnit(mtt=params["dm2.mtt"], DP=params["dm2.DP"]),
        fraction=1.0 - TRUE["dm1.fraction"],
        prefix="dm2",
        bounds=bounds,
    )
    if free_fraction:
        # the first unit is the smaller one; avoids the trivial symmetry of
        # two units with the same bounds
        m.set_fixed("dm1.fraction", False)
        m.set_bounds("dm1.fraction", (0.0, 0.5))
    return m


def set_params(m, values):
    for key, val in values.items():
        if key in m.params:
            m.set_param(key, val)


def total_ttd(m, values, n_steps):
    """Travel time distribution of the mixture (1/yr) at the bin centres."""
    set_params(m, values)
    ttds = m.get_ttds(n_steps=n_steps)
    h = sum(f * d for f, d in zip(ttds["fractions"], ttds["distributions"]))
    return h[1:] / DT


def chi2(sim, obs, idx):
    return float(np.sum(((sim[idx] - obs[idx]) / (NOISE_REL * obs[idx])) ** 2))


def timed(func, *args, **kwargs):
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    return res, time.perf_counter() - t0


def corner(samples, keys, marks, fname, truth=None, bins=30):
    """Pairwise and marginal posterior histograms (lower triangle)."""
    d = len(keys)
    fig, axes = plt.subplots(d, d, figsize=(1.8 * d, 1.8 * d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=bins, color="lightsteelblue", edgecolor="k", lw=0.3)
                ax.set_yticks([])
                if truth is not None:
                    ax.axvline(truth[keys[i]], c="k", ls="--", lw=1.0)
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=bins, cmap="Blues")
                if truth is not None:
                    ax.axvline(truth[keys[j]], c="k", ls="--", lw=1.0)
                    ax.axhline(truth[keys[i]], c="k", ls="--", lw=1.0)
                for values, style in marks:
                    ax.scatter(values[keys[j]], values[keys[i]], **{"zorder": 10, **style})
            if i == d - 1:
                ax.set_xlabel(LABELS[keys[j]])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS[keys[i]])
            elif j > 0:
                ax.set_yticklabels([])
    handles = [plt.Line2D([0], [0], c="k", ls="--", lw=1.0, label="True value")] if truth else []
    handles += [
        plt.Line2D(
            [0],
            [0],
            ls="",
            marker=s["marker"],
            markerfacecolor=s["color"],
            markeredgecolor="k",
            markersize=8,
            label=s["label"],
        )
        for _, s in marks
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout()
    fig.savefig(fname, dpi=400, bbox_inches="tight")
    plt.close(fig)


def summary(res, keys):
    s = res["samples"]
    return {k: np.quantile(s[:, res["param_names"].index(k)], [0.01, 0.5, 0.99]) for k in keys}


def main(quick=False, out=OUT):
    out = Path(out)
    out.mkdir(exist_ok=True)
    inp = lab.read_series(HERE / "tritium_input_semiannual.csv")
    n = len(inp)
    rng = np.random.default_rng(SEED)

    # --- synthetic observations (true model, relative noise) ---
    true_model = make_model(inp, params=TRUE)
    sim_true = true_model.simulate()
    # random sampling dates from 1950 on (legacy generator: the same dates as in
    # earlier versions of this example)
    idx = np.sort(np.random.RandomState(SEED).choice(np.arange(100, n), N_OBS, replace=False))
    obs = np.full(n, np.nan)
    obs[idx] = sim_true[idx] * (1.0 + NOISE_REL * rng.standard_normal(N_OBS))

    m = make_model(inp, target=obs)
    solver = lab.Solver(m)
    keys = m.param_keys(free_only=True)
    sig = {"sigma_rel": NOISE_REL}

    # forward run time (R2.G2)
    n_rep = 50 if quick else 2000
    t0 = time.perf_counter()
    for _ in range(n_rep):
        m.simulate()
    t_forward = (time.perf_counter() - t0) / n_rep

    # --- differential evolution ---
    (de_x, de_sim), t_de = timed(
        solver.differential_evolution,
        maxiter=10 if quick else 1000,
        popsize=5 if quick else 20,
        random_state=42,
        **sig,
    )

    # --- least squares from random starting points ---
    lsq = []
    lo, hi = np.array(m.get_bounds(free_only=True)).T
    t_lsq = 0.0
    for start in rng.uniform(lo, hi, size=(2 if quick else 10, len(keys))):
        for key, val in zip(keys, start):
            m.set_initial(key, val)
            m.set_param(key, val)
        try:
            (x, s), t = timed(solver.least_squares, **sig)
        except (RuntimeError, ValueError) as err:  # no convergence
            print("LSQ start", np.round(start, 2), "failed:", err)
            continue
        t_lsq += t
        lsq.append((chi2(s, obs, idx), x, s, start))
    lsq.sort(key=lambda r: r[0])
    lsq_chi2, lsq_x, lsq_sim, _ = lsq[0]

    # --- MCMC: Metropolis-Hastings and DREAM (same number of model runs) ---
    n_s, n_b = (100, 100) if quick else (2000, 2000)
    n_chains = 8
    mh, t_mh = timed(
        solver.mcmc_sample,
        n_samples=n_chains * n_s,
        burn_in=n_chains * n_b,
        random_state=42,
        **sig,
    )
    dream_kw = dict(
        n_samples=n_s,
        burn_in=n_b,
        n_chains=n_chains,
        cr=[0.25, 0.5, 0.75, 1.0],
        return_sim=True,
        random_state=42,
    )
    dream, t_dream = timed(solver.dream_sample, **dream_kw, **sig)

    # --- DREAM with free mixing fraction (Supplementary Material) ---
    # five free parameters: more chains, a longer burn-in and more samples
    # (R-hat <= 1.02)
    m_f = make_model(inp, target=obs, free_fraction=True)
    dream_f = lab.Solver(m_f).dream_sample(
        **{**dream_kw, "n_chains": 10, "burn_in": n_b * 2, "n_samples": n_s * 3}, **sig
    )
    keys_f = m_f.param_keys(free_only=True)

    # --- tables ---
    rows = ["method,parameter,true,estimate,q01,q99"]
    for k in keys:
        rows.append(f"DE,{k},{TRUE[k]},{de_x[k]:.4f},,")
        rows.append(f"LSQ (best of {len(lsq)}),{k},{TRUE[k]},{lsq_x[k]:.4f},,")
        for name, res in (("MH", mh), ("DREAM", dream)):
            q = summary(res, [k])[k]
            rows.append(f"{name},{k},{TRUE[k]},{q[1]:.4f},{q[0]:.4f},{q[2]:.4f}")
    for k in keys_f:
        q = summary(dream_f, [k])[k]
        rows.append(f"DREAM free fraction,{k},{TRUE[k]},{q[1]:.4f},{q[0]:.4f},{q[2]:.4f}")
    (out / "example_2_results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    conv = ["run,parameter,gelman_rubin,outlier_chains_moved_in_burn_in"]
    for name, res in (("DREAM", dream), ("DREAM free fraction", dream_f)):
        conv += [
            f"{name},{k},{v:.3f},{res['n_outlier_resets']}" for k, v in res["gelman_rubin"].items()
        ]
    (out / "example_2_convergence.csv").write_text("\n".join(conv) + "\n", encoding="utf-8")

    times = [
        "solver,settings,run_time_s",
        f"forward run,{n} time steps (2 DM units),{t_forward:.6f}",
        f"DE,popsize 20 maxiter 1000,{t_de:.2f}",
        f"LSQ,{len(lsq)} starts (total),{t_lsq:.2f}",
        f"MH,{n_chains * (n_s + n_b)} steps,{t_mh:.2f}",
        f"DREAM,{n_chains} chains x {n_s + n_b} steps,{t_dream:.2f}",
    ]
    (out / "example_2_runtimes.csv").write_text("\n".join(times) + "\n", encoding="utf-8")
    lsq_rows = ["chi2," + ",".join(f"start {k}" for k in keys) + "," + ",".join(keys)]
    for c2, x, _, start in lsq:
        lsq_rows.append(
            f"{c2:.3f},"
            + ",".join(f"{v:.3f}" for v in start)
            + ","
            + ",".join(f"{x[k]:.4f}" for k in keys)
        )
    (out / "example_2_lsq_starts.csv").write_text("\n".join(lsq_rows) + "\n", encoding="utf-8")

    # --- Fig.: simulated and observed concentrations, travel time distributions ---
    t_years = np.array([t.year + (t.month - 1) / 12 for t in inp.times])
    w = slice(100, n)  # 1950-2020
    q = np.quantile(dream["sims"].reshape(-1, n), [0.01, 0.5, 0.99], axis=0)
    fig, ax = plt.subplots(2, 1, figsize=(5, 6.5), gridspec_kw={"height_ratios": [2, 1.3]})
    ax[0].fill_between(t_years[w], q[0, w], q[2, w], color=C_DREAM, alpha=0.2, lw=0)
    ax[0].plot(t_years[w], q[1, w], c=C_DREAM, lw=1.5, label="DREAM (median, 1-99 %)")
    ax[0].plot(t_years[w], de_sim[w], c=C_DE, ls=":", lw=2.5, label="Differential evolution")
    ax[0].plot(t_years[w], lsq_sim[w], c=C_LSQ, ls="--", lw=1.2, label="Least squares")
    ax[0].plot(t_years[w], sim_true[w], c="k", lw=1.0, label="True model")
    ax[0].errorbar(
        t_years[idx],
        obs[idx],
        yerr=NOISE_REL * obs[idx],
        fmt="o",
        mfc="w",
        mec="k",
        ecolor="k",
        ms=5,
        capsize=2,
        zorder=100,
        label="Observations (±1σ)",
    )
    ax[0].set(
        xlim=(t_years[100], t_years[-1]),
        ylim=(0, None),
        xlabel="Time",
        ylabel="Tritium [TU]",
        title="Simulated and observed concentrations",
    )
    ax[0].grid(True, alpha=0.7)
    ax[0].legend(fontsize=8)

    n_tt = 71  # 35 yr
    tau = (np.arange(1, n_tt) - 0.5) * DT
    draws = dream["samples"][np.random.default_rng(0).choice(len(dream["samples"]), 1000)]
    tt = np.array([total_ttd(m, dict(zip(dream["param_names"], d)), n_tt) for d in draws])
    qt = np.quantile(tt, [0.01, 0.5, 0.99], axis=0)
    ax[1].fill_between(tau, qt[0], qt[2], color=C_DREAM, alpha=0.2, lw=0)
    ax[1].plot(tau, qt[1], c=C_DREAM, lw=1.5)
    ax[1].plot(tau, total_ttd(m, de_x, n_tt), c=C_DE, ls=":", lw=2.5)
    ax[1].plot(tau, total_ttd(m, lsq_x, n_tt), c=C_LSQ, ls="--", lw=1.2)
    ax[1].plot(tau, total_ttd(true_model, TRUE, n_tt), c="k", lw=1.0)
    ax[1].set(
        xlim=(0, tau[-1]),
        ylim=(0, None),
        xlabel="Travel time [yr]",
        ylabel="Density [1/yr]",
        title="Travel time distributions",
    )
    ax[1].grid(True, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out / "example_2_fit.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    # --- Fig.: posterior distribution ---
    order = ["dm1.mtt", "dm1.DP", "dm2.mtt", "dm2.DP"]
    pick = [dream["param_names"].index(k) for k in order]
    # nested markers: DE and LSQ usually find the same optimum
    marks = [
        (
            dream["posterior_map"],
            {
                "marker": "o",
                "color": "r",
                "s": 90,
                "edgecolor": "k",
                "lw": 0.5,
                "label": "MAP (DREAM)",
                "zorder": 10,
            },
        ),
        (
            de_x,
            {
                "marker": "v",
                "color": "w",
                "s": 40,
                "edgecolor": "k",
                "lw": 0.5,
                "label": "Differential evolution",
                "zorder": 11,
            },
        ),
        (
            lsq_x,
            {
                "marker": "s",
                "color": C_LSQ,
                "s": 10,
                "edgecolor": "k",
                "lw": 0.3,
                "label": "Least squares",
                "zorder": 12,
            },
        ),
    ]
    corner(dream["samples"][:, pick], order, marks, out / "example_2_posterior.png", truth=TRUE)

    # --- SI: free mixing fraction ---
    order_f = ["dm1.fraction"] + order
    pick_f = [dream_f["param_names"].index(k) for k in order_f]
    corner(
        dream_f["samples"][:, pick_f],
        order_f,
        [(dream_f["posterior_map"], marks[0][1])],
        out / "example_2_free_fraction.png",
        truth=TRUE,
    )

    print("\n".join(rows))
    print("\n".join(times))
    print("\n".join(conv))
    print("LSQ chi2 of all starts:", [round(r[0], 2) for r in lsq])
    print(
        "Machine:",
        platform.processor(),
        "| Python",
        platform.python_version(),
        "| NumPy",
        np.__version__,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="a few iterations only (test)")
    parser.add_argument("--out", default=OUT, help="output folder (default: output/)")
    args = parser.parse_args()
    main(quick=args.quick, out=args.out)
