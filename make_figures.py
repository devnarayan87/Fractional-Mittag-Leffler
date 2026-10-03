"""make_figures.py -- regenerate all paper figures from results/*.json"""
import json
import numpy as np
import matplotlib.pyplot as plt
from fmlar_lib import psi_weights, arfima_weights, simulate_fmlar, sample_acf, model_acf, gph_estimate

R = "results"


def load(name):
    return json.load(open(f"{R}/{name}"))


# ------------------------------------------------------------------
# Figure 1: (a) ML kernel vs power-law kernel [MADE MORE PROMINENT],
# (b) ACF benchmark against ARFIMA, (c) constant-memory recovery bias/RMSE
# summary across sample sizes (replaces the old single-trajectory panel,
# which moved to new Figure 3).
# ------------------------------------------------------------------
def figure1():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

    # (a) kernel comparison: FMLAR (Mittag-Leffler) vs ARFIMA (power-law)
    J = 60
    j = np.arange(0, J + 1)
    ax = axes[0]
    for alpha, col in zip([0.55, 0.70, 0.85], ["#d62728", "#1f77b4", "#2ca02c"]):
        psi = psi_weights(alpha, 1.0, J)
        ax.plot(j, psi, color=col, lw=2, label=fr"FMLAR $\alpha$={alpha}")
        d = 1 - alpha
        w = arfima_weights(d, J)
        ax.plot(j, w, color=col, lw=1.2, ls="--", alpha=0.8)
    ax.plot([], [], color="gray", lw=2, label="FMLAR (Mittag-Leffler), solid")
    ax.plot([], [], color="gray", lw=1.2, ls="--", label="ARFIMA (power-law), dashed")
    ax.set_xlabel("lag $j$")
    ax.set_ylabel(r"weight $\psi_j$")
    ax.set_title("(a) Mittag-Leffler vs.\ power-law kernel shape")
    ax.legend(fontsize=8)

    # (b) ACF benchmark
    rng = np.random.default_rng(101)
    X = simulate_fmlar(0.7, 1.0, T=4000, J=300, rng=rng)
    K = 60
    rho_hat = sample_acf(X, K)
    rho_fmlar = model_acf(0.7, 1.0, K, J=300)
    d = 0.3
    w = arfima_weights(d, 300)
    rho_arfima = np.array([np.dot(w[: 300 + 1 - k], w[k:]) / np.dot(w, w) for k in range(1, K + 1)])
    d_hat, _ = gph_estimate(X)
    ax = axes[1]
    ax.plot(range(1, K + 1), rho_hat, "o", ms=3, color="gray", alpha=0.6, label="sample ACF (simulated FMLAR)")
    ax.plot(range(1, K + 1), rho_fmlar, "-", color="#1f77b4", label=r"FMLAR model ACF ($\alpha$=0.7)")
    ax.plot(range(1, K + 1), rho_arfima, "--", color="#d62728", label="tail-equiv. ARFIMA(0,d=0.3,0)")
    ax.set_xlabel("lag $k$")
    ax.set_ylabel(r"$\rho(k)$")
    ax.set_title(f"(b) Sample ACF vs.\ model/ARFIMA benchmark\n" + r"(global GPH $\hat d$" + f"={d_hat:.3f}, true d=0.300)")
    ax.legend(fontsize=8)

    # (c) constant-memory recovery: bias/RMSE across sample sizes
    exp1 = load("exp1_constant_memory.json")
    import pandas as pd
    df = pd.DataFrame(exp1)
    Ts = sorted(df["T"].unique())
    de_rmse_a, w_rmse_a = [], []
    for T in Ts:
        sub = df[df["T"] == T]
        de_rmse_a.append(np.sqrt(np.mean((sub["de_alpha"] - 0.65) ** 2)))
        w_rmse_a.append(np.sqrt(np.mean((sub["whittle_alpha"] - 0.65) ** 2)))
    ax = axes[2]
    ax.plot(Ts, de_rmse_a, "o-", label=r"DE (ACF-shape) $\hat\alpha$ RMSE")
    ax.plot(Ts, w_rmse_a, "s-", label=r"Whittle $\hat\alpha$ RMSE")
    ax.set_xlabel("sample size $T$")
    ax.set_ylabel(r"RMSE($\hat\alpha$)")
    ax.set_title(r"(c) $\hat\alpha$ recovery vs.\ sample size" + "\n(true " + r"$\alpha$=0.65, 8-10 reps/size)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("fmlar_figure1.png", dpi=170)
    plt.close(fig)
    print("Figure 1 saved.")


# ------------------------------------------------------------------
# Figure 2: Nile discharge + break marker, forecast RMSE comparison,
# post-break vs full-sample memory estimates.
# ------------------------------------------------------------------
def figure2():
    import statsmodels.api as sm
    nile = sm.datasets.nile.load_pandas().data
    years = nile["year"].values
    volume = nile["volume"].values

    fit = load("exp4_nile_fit.json")
    fc = load("exp4_nile_forecast.json")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

    ax = axes[0]
    ax.plot(years, volume, color="#1f4e79")
    ax.axvline(1899, color="crimson", ls="--", lw=1.2, label="1899 dam-construction break")
    ax.set_xlabel("year")
    ax.set_ylabel(r"discharge ($10^8\,$m$^3$)")
    ax.set_title("(a) Nile annual discharge at Aswan, 1871-1970")
    ax.legend(fontsize=8)

    ax = axes[1]
    methods = ["naive", "ar1", "arfima", "fmlar"]
    labels = ["Naive", "AR(1)", "ARFIMA(0,d,0)", "FMLAR"]
    rmse = [fc["rmse"][m] for m in methods]
    colors = ["#888888", "#1f77b4", "#d62728", "#2ca02c"]
    ax.bar(labels, rmse, color=colors)
    dm = fc["dm_fmlar_vs_arfima"]
    ax.set_ylabel("10-yr-ahead RMSE")
    ax.set_title("(b) Forecast RMSE, 1961-1970\n" + fr"(DM FMLAR vs.\ ARFIMA: $p$={dm['p_value']:.2f})")

    ax = axes[2]
    full = fit["full"]
    post = fit["post_break"]
    labels2 = ["GPH $\\hat d$", "DE $\\hat\\alpha$", "Whittle $\\hat\\alpha$"]
    full_vals = [full["d_gph"], full["de_alpha"], full["whittle_alpha"]]
    post_vals = [post["d_gph_post"], post["de_alpha_post"], np.nan]
    x = np.arange(len(labels2))
    w = 0.35
    ax.bar(x - w / 2, full_vals, w, label="Full sample (1871-1970)", color="#1f77b4")
    ax.bar(x + w / 2, [post["d_gph_post"], post["de_alpha_post"], np.nan], w,
           label=f"Post-1899 only (n={post['n_post']})", color="#ff7f0e")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels2)
    ax.set_title("(c) Memory estimate: full sample vs.\npost-break subsample")
    ax.legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig("fmlar_figure2.png", dpi=170)
    plt.close(fig)
    print("Figure 2 saved.")


# ------------------------------------------------------------------
# Figure 3 (NEW): multiple alpha(t) trajectories, GPH vs MLP with
# uncertainty bands (mean +/- SE across realizations); plus the second
# real dataset (sunspots) windowed-memory trajectory.
# ------------------------------------------------------------------
def figure3():
    exp2 = load("exp2_time_varying.json")
    sun = load("exp5_sunspot.json")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    axes = axes.ravel()
    titles = {"linear_up": r"linear $\alpha$: 0.55$\to$0.92",
              "linear_down": r"linear $\alpha$: 0.85$\to$0.58",
              "sinusoidal": r"sinusoidal $\alpha$: 0.70$\pm$0.12",
              "step": r"step $\alpha$: 0.60$\to$0.85"}
    for ax, kind in zip(axes[:4], ["linear_up", "linear_down", "sinusoidal", "step"]):
        d = exp2[kind]
        c = np.array(d["centers"])
        true_d = np.array(d["true_d"])
        gph_m, gph_se = np.array(d["gph_mean"]), np.array(d["gph_se"])
        mlp_m, mlp_se = np.array(d["mlp_mean"]), np.array(d["mlp_se"])
        ax.plot(c, true_d, "k-", lw=2, label="true $d(t)$")
        ax.plot(c, gph_m, "o-", color="#1f77b4", ms=3, label="windowed GPH")
        ax.fill_between(c, gph_m - 1.645 * gph_se, gph_m + 1.645 * gph_se, color="#1f77b4", alpha=0.2)
        ax.plot(c, mlp_m, "s-", color="#d62728", ms=3, label="MLP tracker")
        ax.fill_between(c, mlp_m - 1.645 * mlp_se, mlp_m + 1.645 * mlp_se, color="#d62728", alpha=0.2)
        ax.set_title(titles[kind] + f"\nMAE: GPH={d['gph_mae']:.3f}, MLP={d['mlp_mae']:.3f}", fontsize=9.5)
        ax.set_xlabel("t")
        ax.set_ylabel("$d(t)$")
        ax.legend(fontsize=7)

    # Panel 5: sunspot windowed memory trajectory
    ax = axes[4]
    tv = sun["tv_track"]
    ax.plot(tv["years"], tv["d_hat"], "o-", color="#2ca02c", ms=3)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("year")
    ax.set_ylabel(r"windowed GPH $\hat d(t)$")
    ax.set_title("(e) Sunspot series: windowed memory\nestimate over 1700-2008", fontsize=9.5)

    # Panel 6: sunspot forecast RMSE
    ax = axes[5]
    fc = sun["forecast"]
    methods = ["naive", "ar1", "arfima", "fmlar"]
    labels = ["Naive", "AR(1)", "ARFIMA", "FMLAR"]
    rmse = [fc["rmse"][m] for m in methods]
    ax.bar(labels, rmse, color=["#888888", "#1f77b4", "#d62728", "#2ca02c"])
    dm = fc["dm_fmlar_vs_arfima"]
    ax.set_ylabel("20-yr-ahead RMSE")
    ax.set_title(f"(f) Sunspot forecast RMSE\n" + fr"(DM FMLAR vs.\ ARFIMA: $p$={dm['p_value']:.2f})", fontsize=9.5)

    plt.tight_layout()
    plt.savefig("fmlar_figure3.png", dpi=170)
    plt.close(fig)
    print("Figure 3 saved.")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
