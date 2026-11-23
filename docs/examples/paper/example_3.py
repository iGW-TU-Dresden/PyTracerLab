"""Example 3 of the PyTracerLab paper: river water from a groundwater flow model.

The reference system is a MODFLOW 6 model (Rudolph et al., 2026; details in the
Supplementary Material): river water is a mixture of groundwater (recharge,
77.9 %) and tracer-free water from an injection well (22.1 %). The tracer
input in recharge is synthetic (1970-2019, half-life of tritium, no tracer
before 1970); ten tracer concentrations in river water (every 5 years) are the
observations, and particle tracking gives the true travel time distribution
of the groundwater (``example_3_particle_travel_times.csv``: travel times in
days and flux weights).

Two lumped parameter models are calibrated with DREAM:

- EM + PM: an exponential model unit for the groundwater and a piston flow
  model unit with a large travel time (500 yr) for the tracer-free water; the
  mean travel time and the mixing fraction are calibrated;
- age bins: one age-bin unit for all river water (no assumption on the shape of
  the travel time distribution); the last bin (85-1000 yr) contains old
  groundwater and the tracer-free injected water, so that no second unit is
  needed. Because the tracer input starts in 1970, water older than about
  50 yr carries no tracer in any observation: the weight of the bin 50-85 yr is
  not determined by the data.

Both are compared with the true fractions of river water in travel time
classes (particle tracking, weighted by the recharge flux, and water balance).
Water with travel times below 85 yr is considered young (Supplementary
Material). In the EM + PM model, the mixing fraction of the EM unit is the
fraction of groundwater (recharge) in river water; the fraction of young water
follows from it and the EM.

Run from this folder: ``python example_3.py`` (about 2 min; ``--quick`` for a
test run). Figures and tables are written to ``output/``.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import PyTracerLab as lab  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

F_RCH = 0.779  # fraction of river water that entered the system via recharge
EDGES = [0.0, 5.0, 15.0, 30.0, 50.0, 85.0, 1000.0]  # age bins (yr)
N_YOUNG = EDGES.index(85.0)  # number of classes of young water (< 85 yr)
SIGMA_REL = 0.1  # relative uncertainty (model structure error; data are model output)
CLASSES = ["0-5", "5-15", "15-30", "30-50", "50-85", ">85 or\ntracer-free"]


def make_model(inp, obs):
    return lab.Model(
        dt=1 / 12,
        lambda_=lab.decay_constant("Tritium"),
        input_series=inp.values,
        target_series=obs.values,
        steady_state_input=0.0,  # no tracer before 1970
        time_steps=inp.times,
    )


def true_shares():
    """Fractions of river water in the travel time classes (particle tracking)."""
    data = np.genfromtxt(HERE / "example_3_particle_travel_times.csv", delimiter=",", names=True)
    tt = data["tts"] / 365.25  # days -> years
    w = data["weights"] / data["weights"].sum()
    gw = np.array([w[(tt >= a) & (tt < b)].sum() for a, b in zip(EDGES[:-2], EDGES[1:-1])])
    shares = F_RCH * gw
    return np.append(shares, 1.0 - shares.sum()), tt, w


def em_shares(mtt, fraction):
    """Fractions of river water in the classes for EM (groundwater) + tracer-free water."""
    cdf = 1.0 - np.exp(-np.asarray(EDGES[:-1], dtype=float)[:, None] / np.atleast_1d(mtt))
    young = fraction * np.diff(cdf, axis=0)  # classes below 85 yr
    return np.vstack([young, 1.0 - young.sum(axis=0)]).T  # (n, number of classes)


def nse(sim, obs, idx):
    return 1.0 - np.sum((sim[idx] - obs[idx]) ** 2) / np.sum((obs[idx] - obs[idx].mean()) ** 2)


def main(quick=False, out=OUT):
    out = Path(out)
    out.mkdir(exist_ok=True)
    inp = lab.read_series(HERE / "example_3_input.csv")
    obs = lab.read_series(HERE / "example_3_observations.csv", align_to=inp)
    o = obs.values.ravel()
    idx = np.where(np.isfinite(o))[0]
    n_s, n_b = (100, 100) if quick else (2000, 4000)
    kw = dict(
        n_samples=n_s,
        burn_in=n_b,
        n_chains=8,
        cr=[0.25, 0.5, 0.75, 1.0],
        sigma_rel=SIGMA_REL,
        return_sim=True,
        random_state=42,
    )

    # (a) EM + PM (tracer-free water), mixing fraction calibrated
    m_em = make_model(inp, obs)
    m_em.add_unit(lab.EMUnit(mtt=10.0), fraction=0.5, prefix="em", bounds=[(0.5, 1000.0)])
    m_em.add_unit(lab.PMUnit(mtt=500.0), fraction=0.5, prefix="pm")
    m_em.set_fixed("pm.mtt")
    m_em.set_fixed("em.fraction", False)
    res_em = lab.Solver(m_em).dream_sample(**kw)
    s_em = res_em["samples"]
    p = res_em["param_names"]
    sh_em = em_shares(s_em[:, p.index("em.mtt")], s_em[:, p.index("em.fraction")])

    # (b) one age-bin unit for all river water
    bins = lab.AgeBinUnit(edges=EDGES)
    m_bins = make_model(inp, obs)
    m_bins.add_unit(bins, fraction=1.0, prefix="bins")
    res_bins = lab.Solver(m_bins).dream_sample(**kw)
    sh_bins = bins.weights_from_samples(res_bins, "bins")

    truth, tt, w = true_shares()

    # tables
    rows = ["quantity," + ",".join(c.replace("\n", " ") for c in CLASSES) + ",young (<85 yr)"]
    rows.append("MODFLOW," + ",".join(f"{v:.4f}" for v in truth) + f",{truth[:N_YOUNG].sum():.4f}")
    for name, sh in (("EM + PM", sh_em), ("age bins", sh_bins)):
        young = sh[:, :N_YOUNG].sum(axis=1)
        for q, lab_ in ((0.5, "median"), (0.01, "q01"), (0.99, "q99")):
            vals = np.quantile(sh, q, axis=0)
            rows.append(
                f"{name} {lab_},"
                + ",".join(f"{v:.4f}" for v in vals)
                + f",{np.quantile(young, q):.4f}"
            )
    em_q = {k: np.quantile(s_em[:, i], [0.01, 0.5, 0.99]) for i, k in enumerate(p)}
    sims = {}
    for name, res in (("EM + PM", res_em), ("age bins", res_bins)):
        sims[name] = np.quantile(res["sims"].reshape(-1, len(o)), [0.01, 0.5, 0.99], axis=0)
        rhat = max(res["gelman_rubin"].values())
        rows.append(
            f"{name}: NSE of median {nse(sims[name][1], o, idx):.3f}; max R-hat {rhat:.3f}; "
            f"outlier chains moved {res['n_outlier_resets']}"
        )
    rows.append(
        "EM + PM: em.mtt [q01 median q99] "
        + " ".join(f"{v:.2f}" for v in em_q["em.mtt"])
        + "; em.fraction "
        + " ".join(f"{v:.3f}" for v in em_q["em.fraction"])
    )
    (out / "example_3_results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))

    # figure
    t = np.array([d.year + (d.month - 1) / 12 for d in inp.times])
    colors = {"EM + PM": "red", "age bins": "royalblue"}
    fig, ax = plt.subplots(2, 1, figsize=(5.5, 7), gridspec_kw={"height_ratios": [1.3, 1]})
    for name, q in sims.items():
        ax[0].fill_between(t, q[0], q[2], color=colors[name], alpha=0.15, lw=0)
        ax[0].plot(t, q[1], c=colors[name], lw=1.5, label=f"{name} (NSE {nse(q[1], o, idx):.2f})")
    ax[0].scatter(
        t[idx],
        o[idx],
        marker="o",
        facecolor="w",
        edgecolor="k",
        s=35,
        zorder=10,
        label="MODFLOW (river water)",
    )
    ax[0].set(
        xlim=(t[0], t[-1] + 1 / 12),
        ylim=(0, None),
        xlabel="Time",
        ylabel="Concentration [-]",
        title="Simulated concentrations in river water",
    )
    ax[0].grid(True, alpha=0.7)
    ax[0].legend(fontsize=8)

    x = np.arange(len(CLASSES))
    width = 0.27
    ax[1].bar(
        x - width,
        truth,
        width,
        color="lightgrey",
        edgecolor="k",
        lw=0.5,
        label="MODFLOW (particle tracking)",
    )
    for k, (name, sh) in enumerate((("EM + PM", sh_em), ("age bins", sh_bins))):
        q = np.quantile(sh, [0.01, 0.5, 0.99], axis=0)
        ax[1].bar(
            x + k * width,
            q[1],
            width,
            color=colors[name],
            alpha=0.6,
            edgecolor="k",
            lw=0.5,
            yerr=[q[1] - q[0], q[2] - q[1]],
            capsize=2,
            label=name,
        )
    ax[1].set_xticks(x, CLASSES, fontsize=8)
    ax[1].set(
        xlabel="Travel time class [yr]",
        ylabel="Fraction of river water [-]",
        title="Travel time distribution of river water",
    )
    ax[1].grid(True, axis="y", alpha=0.7)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "example_3.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="a few iterations only (test)")
    parser.add_argument("--out", default=OUT, help="output folder (default: output/)")
    args = parser.parse_args()
    main(quick=args.quick, out=args.out)
