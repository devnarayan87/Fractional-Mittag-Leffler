"""
run_experiments.py
---------------------
Runs every empirical experiment needed for the revised paper and writes
results incrementally to JSON files in ./results/ (so a long run can be
monitored / resumed). Designed to be run with `nohup python3
run_experiments.py > log.txt 2>&1 &` and polled.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fmlar_lib import (
    simulate_fmlar, simulate_fmlar_tv, simulate_arfima,
    gph_estimate, windowed_gph, de_acf_fit, whittle_fit,
    psi_weights, arfima_weights, one_step_forecast_from_weights,
    build_mlp_training_corpus, log_periodogram_features,
    diebold_mariano, whittle_negloglik, periodogram,
)
from sklearn.neural_network import MLPRegressor

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save(name, obj):
    with open(os.path.join(RESULTS_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# Experiment 1: constant-memory recovery across sample sizes (bias/RMSE/
# SD/coverage), DE vs Whittle. Addresses suggestions #15, #16.
# ----------------------------------------------------------------------
def whittle_se(x, alpha_hat, theta_hat, sigma2_hat, J=300, n_freq=300, h=1e-3):
    """Numerical-Hessian-based asymptotic SE for (alpha, theta) at the
    Whittle optimum (standard inverse-observed-information approximation)."""
    lam, I = periodogram(x)
    m = min(n_freq, len(I))
    lam_obs, I_obs = lam[:m], I[:m]
    theta0 = np.array([alpha_hat, theta_hat, np.log(sigma2_hat)])

    def f(p):
        return whittle_negloglik(p, I_obs, lam_obs, J=J)

    n = len(theta0)
    H = np.zeros((n, n))
    f0 = f(theta0)
    for i in range(n):
        for j in range(n):
            hi = np.zeros(n); hi[i] = h * max(abs(theta0[i]), 0.1)
            hj = np.zeros(n); hj[j] = h * max(abs(theta0[j]), 0.1)
            fpp = f(theta0 + hi + hj)
            fpm = f(theta0 + hi - hj)
            fmp = f(theta0 - hi + hj)
            fmm = f(theta0 - hi - hj)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * hi[i] * hj[j])
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(n, np.nan)
    return se[0], se[1]  # se_alpha, se_theta


def exp1_constant_memory_recovery(sample_sizes=(1000, 2500, 4000), reps=10,
                                   true_alpha=0.65, true_theta=1.2, seed0=1000):
    rows = []
    for T in sample_sizes:
        for r in range(reps):
            seed = seed0 + 1000 * T + r
            rng = np.random.default_rng(seed)
            J = min(300, T // 3)
            X = simulate_fmlar(true_alpha, true_theta, T=T, J=J, rng=rng)

            a_de, th_de = de_acf_fit(X, K=40, J=min(J, 250), seed=seed)
            a_w, th_w, s2_w, res = whittle_fit(X, n_freq=min(300, T // 4), J=min(J, 250),
                                                init=(a_de, th_de), n_restarts=3, seed=seed)
            se_a, se_th = whittle_se(X, a_w, th_w, s2_w, J=min(J, 250),
                                      n_freq=min(300, T // 4))
            covers_alpha = abs(a_w - true_alpha) <= 1.645 * se_a if np.isfinite(se_a) else None
            covers_theta = abs(th_w - true_theta) <= 1.645 * se_th if np.isfinite(se_th) else None

            row = dict(T=T, rep=r, seed=seed,
                       de_alpha=a_de, de_theta=th_de,
                       whittle_alpha=a_w, whittle_theta=th_w,
                       whittle_se_alpha=se_a, whittle_se_theta=se_th,
                       covers_alpha_90=covers_alpha, covers_theta_90=covers_theta)
            rows.append(row)
            save("exp1_constant_memory.json", rows)
            log(f"exp1 T={T} rep={r} DE=({a_de:.3f},{th_de:.3f}) "
                f"Whittle=({a_w:.3f},{th_w:.3f}) se=({se_a:.3f},{se_th:.3f})")
    return rows


# ----------------------------------------------------------------------
# Experiment 2: multiple time-varying alpha(t) trajectories, GPH vs MLP,
# with uncertainty bands from repeated noise realizations.
# Addresses suggestions #17, #25.
# ----------------------------------------------------------------------
def make_trajectory(kind, T):
    if kind == "linear_up":
        return lambda t: 0.55 + (0.92 - 0.55) * min(max(t / T, 0), 1)
    if kind == "linear_down":
        return lambda t: 0.85 - (0.85 - 0.58) * min(max(t / T, 0), 1)
    if kind == "sinusoidal":
        return lambda t: 0.7 + 0.12 * np.sin(2 * np.pi * t / T)
    if kind == "step":
        return lambda t: 0.60 if t < T / 2 else 0.85
    raise ValueError(kind)


def exp2_time_varying(T=4000, J=250, theta=1.0, window=600, step=200,
                       n_realizations=8, trajectories=("linear_up", "linear_down",
                                                        "sinusoidal", "step"),
                       mlp_seed=999):
    log("Training MLP corpus for exp2...")
    rng_train = np.random.default_rng(mlp_seed)
    Xtr, ytr = build_mlp_training_corpus(n_series=60, T=2000, J=250, rng=rng_train)
    mlp = MLPRegressor(hidden_layer_sizes=(24, 12), activation="relu",
                        solver="adam", alpha=1e-3, max_iter=3000,
                        early_stopping=True, random_state=mlp_seed)
    mlp.fit(Xtr, ytr)
    save("exp2_mlp_training_summary.json",
         {"n_train": len(ytr), "n_iter": int(mlp.n_iter_), "loss": float(mlp.loss_)})
    log(f"MLP trained: n={len(ytr)}, n_iter={mlp.n_iter_}, loss={mlp.loss_:.4f}")

    results = {}
    for kind in trajectories:
        alpha_func = make_trajectory(kind, T)
        gph_runs, mlp_runs, centers_ref, true_d_ref = [], [], None, None
        for rz in range(n_realizations):
            rng = np.random.default_rng(2000 + hash(kind) % 1000 + rz)
            X, alphas = simulate_fmlar_tv(alpha_func, theta, T, J, rng)
            centers, dhat_gph = windowed_gph(X, window, step)
            true_d = 1.0 - alphas[centers]
            feats = np.array([log_periodogram_features(X[c - window // 2: c + window // 2])
                               for c in centers])
            dhat_mlp = mlp.predict(feats)
            gph_runs.append(dhat_gph)
            mlp_runs.append(dhat_mlp)
            if centers_ref is None:
                centers_ref, true_d_ref = centers, true_d
            log(f"exp2 {kind} realization {rz}: GPH MAE={np.mean(np.abs(dhat_gph - true_d)):.3f} "
                f"MLP MAE={np.mean(np.abs(dhat_mlp - true_d)):.3f}")

        gph_arr = np.array(gph_runs)  # (n_realizations, n_windows)
        mlp_arr = np.array(mlp_runs)
        results[kind] = {
            "centers": centers_ref.tolist(),
            "true_d": true_d_ref.tolist(),
            "gph_mean": gph_arr.mean(axis=0).tolist(),
            "gph_se": (gph_arr.std(axis=0, ddof=1) / np.sqrt(n_realizations)).tolist(),
            "mlp_mean": mlp_arr.mean(axis=0).tolist(),
            "mlp_se": (mlp_arr.std(axis=0, ddof=1) / np.sqrt(n_realizations)).tolist(),
            "gph_mae": float(np.mean(np.abs(gph_arr.mean(axis=0) - true_d_ref))),
            "mlp_mae": float(np.mean(np.abs(mlp_arr.mean(axis=0) - true_d_ref))),
            "gph_neg_frac": float(np.mean(gph_arr < 0)),
            "mlp_neg_frac": float(np.mean(mlp_arr < 0)),
        }
        save("exp2_time_varying.json", results)
    return results


# ----------------------------------------------------------------------
# Experiment 3: Ulam-Hyers numeric check (kept similar to original, but
# with a few more repetitions for robustness).
# ----------------------------------------------------------------------
def exp3_ulam_hyers(regimes=(("Slow", 0.55, 0.60), ("Medium", 0.55, 0.75),
                              ("Fast", 0.55, 0.95)),
                     T=1500, J=250, theta=1.0, reps=15, seed0=5000):
    rows = []
    for name, a0, a1 in regimes:
        dalpha = abs(a1 - a0)
        amid = (a0 + a1) / 2
        rmses = []
        for r in range(reps):
            seed = seed0 + r + hash(name) % 100
            rng1 = np.random.default_rng(seed)
            rng2 = np.random.default_rng(seed)  # same seed -> same innovation path
            alpha_func = lambda t, a0=a0, a1=a1, T=T: a0 + (a1 - a0) * min(max(t / T, 0), 1)
            Xtv, _ = simulate_fmlar_tv(alpha_func, theta, T, J, rng1)
            Xref = simulate_fmlar(amid, theta, T=T, J=J, rng=rng2)
            rmse = float(np.sqrt(np.mean((Xtv - Xref) ** 2)))
            rmses.append(rmse)
        rows.append({"regime": name, "delta_alpha": dalpha,
                      "rmse_mean": float(np.mean(rmses)), "rmse_sd": float(np.std(rmses, ddof=1)),
                      "ratio": float(np.mean(rmses) / dalpha)})
        save("exp3_ulam_hyers.json", rows)
        log(f"exp3 {name}: RMSE={np.mean(rmses):.3f}+-{np.std(rmses,ddof=1):.3f}")
    return rows


# ----------------------------------------------------------------------
# Experiment 4: Nile River case study (memory estimation, structural-break
# split, forecasting comparison + Diebold-Mariano test).
# ----------------------------------------------------------------------
def rolling_arfima_forecast(train, d_hat, n_ar=60):
    w = arfima_weights(d_hat, min(400, len(train) - 1))
    return one_step_forecast_from_weights(train, w, n_ar=n_ar)


def nile_analysis():
    nile = sm.datasets.nile.load_pandas().data
    years = nile["year"].values.astype(int)
    x = nile["volume"].values.astype(float)
    xm = x - x.mean()

    d_gph, se_gph = gph_estimate(xm)
    a_de, th_de = de_acf_fit(xm, K=25, J=90, seed=11)
    a_w, th_w, s2_w, res = whittle_fit(xm, n_freq=49, J=90, init=(a_de, th_de),
                                        n_restarts=5, seed=11)

    break_idx = int(np.where(years == 1899)[0][0])
    post = xm[break_idx:]
    d_gph_post, _ = gph_estimate(post)
    a_de_post, th_de_post = de_acf_fit(post, K=15, J=min(60, len(post) - 2), seed=12)

    full_fit = {"d_gph": d_gph, "de_alpha": a_de, "de_theta": th_de,
                "whittle_alpha": a_w, "whittle_theta": th_w, "whittle_sigma2": s2_w}
    break_fit = {"n_post": int(len(post)), "d_gph_post": d_gph_post,
                 "de_alpha_post": a_de_post, "de_theta_post": th_de_post}
    save("exp4_nile_fit.json", {"full": full_fit, "post_break": break_fit})
    log(f"Nile full-sample: GPH d={d_gph:.3f}, DE=({a_de:.3f},{th_de:.3f}), "
        f"Whittle=({a_w:.3f},{th_w:.3f})")
    log(f"Nile post-1899:   GPH d={d_gph_post:.3f}, DE=({a_de_post:.3f},{th_de_post:.3f})")

    # --- forecasting comparison, last 10 years, expanding window ---
    n = len(x)
    test_start = n - 10
    fmlar_psi = psi_weights(a_de, th_de, min(400, test_start - 1))

    naive_f, ar1_f, arfima_f, fmlar_f, actual, dgph_used = [], [], [], [], [], []
    for t in range(test_start, n):
        train = x[:t]
        train_c = train - train.mean()
        naive_f.append(train[-1])
        # AR(1) via Yule-Walker (lag-1 autocorrelation)
        phi1 = np.corrcoef(train_c[:-1], train_c[1:])[0, 1]
        ar1_f.append(train.mean() + phi1 * (train[-1] - train.mean()))
        d_local, _ = gph_estimate(train_c)
        d_local = np.clip(d_local, 0.01, 0.49)
        dgph_used.append(d_local)
        w = arfima_weights(d_local, min(300, t - 1))
        arfima_f.append(train.mean() + one_step_forecast_from_weights(train_c, w, n_ar=min(60, t - 1)))
        fmlar_f.append(train.mean() + one_step_forecast_from_weights(
            train_c, fmlar_psi, n_ar=min(60, t - 1)))
        actual.append(x[t])

    actual = np.array(actual)
    fdict = {"naive": np.array(naive_f), "ar1": np.array(ar1_f),
             "arfima": np.array(arfima_f), "fmlar": np.array(fmlar_f)}
    rmse = {k: float(np.sqrt(np.mean((actual - v) ** 2))) for k, v in fdict.items()}

    dm_stat, dm_p = diebold_mariano(actual - fdict["fmlar"], actual - fdict["arfima"], h=1)
    dm_naive_stat, dm_naive_p = diebold_mariano(actual - fdict["fmlar"], actual - fdict["naive"], h=1)

    forecast_table = {"years": years[test_start:].tolist(), "actual": actual.tolist(),
                       "naive": fdict["naive"].tolist(), "ar1": fdict["ar1"].tolist(),
                       "arfima": fdict["arfima"].tolist(), "fmlar": fdict["fmlar"].tolist(),
                       "d_gph_used": dgph_used, "rmse": rmse,
                       "dm_fmlar_vs_arfima": {"stat": dm_stat, "p_value": dm_p},
                       "dm_fmlar_vs_naive": {"stat": dm_naive_stat, "p_value": dm_naive_p}}
    save("exp4_nile_forecast.json", forecast_table)
    log(f"Nile forecast RMSE: {rmse}")
    log(f"DM (FMLAR vs ARFIMA): stat={dm_stat:.3f}, p={dm_p:.3f}")
    return full_fit, break_fit, forecast_table


# ----------------------------------------------------------------------
# Experiment 5: second real dataset (sunspot numbers, deseasonalized),
# same style of analysis. Addresses #19, #26.
# ----------------------------------------------------------------------
def sunspot_analysis():
    """
    Second real dataset: annual sunspot numbers (Wolf/Zurich series),
    1700-2008, statsmodels' bundled `sunspots` dataset (309 observations,
    about 3x the length of the Nile series). Analyzed on the log1p scale.
    The series has a well-known ~11-year activity cycle (confirmed below
    via its periodogram peak); we use a GPH bandwidth (m ~ sqrt(n) ~ 17)
    well below that cycle's frequency index so the low-frequency long-
    memory estimate is not dominated by the cycle, and flag this
    bandwidth-sensitivity explicitly rather than deseasonalizing (which,
    tested separately, over-differences the series and destroys the very
    persistence being measured). Multi-century solar-cycle-amplitude
    modulation (Maunder Minimum, Modern Maximum) makes this a plausible
    setting for genuinely time-varying memory, examined via a windowed
    GPH trajectory over the full series.
    """
    sun = sm.datasets.sunspots.load_pandas().data
    years = sun["YEAR"].values.astype(int)
    raw = sun["SUNACTIVITY"].values.astype(float)
    logx = np.log1p(raw)
    xm = logx - logx.mean()

    lam, I = periodogram(xm)
    cycle_peak_idx = int(np.argmax(I[:60]))
    m_gph = int(len(xm) ** 0.5)

    d_gph, se_gph = gph_estimate(xm, m=m_gph)
    a_de, th_de = de_acf_fit(xm, K=30, J=150, seed=21)
    a_w, th_w, s2_w, res = whittle_fit(xm, n_freq=100, J=150, init=(a_de, th_de),
                                        n_restarts=5, seed=21)

    fit = {"n": int(len(xm)), "m_gph": m_gph, "cycle_peak_idx": cycle_peak_idx,
           "cycle_peak_period_years": float(2 * np.pi / lam[cycle_peak_idx]),
           "d_gph": d_gph, "se_gph": se_gph, "de_alpha": a_de, "de_theta": th_de,
           "whittle_alpha": a_w, "whittle_theta": th_w, "whittle_sigma2": s2_w}
    log(f"Sunspot (log1p, m={m_gph}, cycle peak at idx {cycle_peak_idx}, "
        f"period {2*np.pi/lam[cycle_peak_idx]:.1f}y): GPH d={d_gph:.3f}+-{se_gph:.3f}, "
        f"DE=({a_de:.3f},{th_de:.3f}), Whittle=({a_w:.3f},{th_w:.3f})")

    # windowed GPH over the whole series to show time-varying memory
    centers, dhat = windowed_gph(xm, window=80, step=20, m=int(80 ** 0.5))
    tv_track = {"years": years[centers].tolist(), "centers": centers.tolist(),
                "d_hat": dhat.tolist()}

    # forecasting comparison, last 20 years, expanding window, log1p scale
    n = len(raw)
    test_start = n - 20
    fmlar_psi = psi_weights(a_de, th_de, min(300, test_start - 1))
    naive_f, ar1_f, arfima_f, fmlar_f, actual = [], [], [], [], []
    for t in range(test_start, n):
        train = logx[:t]
        m_ = train.mean()
        train_c = train - m_
        naive_f.append(raw[t - 1])
        phi1 = np.corrcoef(train_c[:-1], train_c[1:])[0, 1] if len(train_c) > 2 else 0.0
        ar1_pred = m_ + phi1 * (train[-1] - m_)
        d_local, _ = gph_estimate(train_c, m=int(len(train_c) ** 0.5))
        d_local = np.clip(d_local, 0.01, 0.49)
        w = arfima_weights(d_local, min(200, t - 1))
        arfima_pred = m_ + one_step_forecast_from_weights(train_c, w, n_ar=min(60, t - 1))
        fmlar_pred = m_ + one_step_forecast_from_weights(train_c, fmlar_psi, n_ar=min(60, t - 1))
        ar1_f.append(np.expm1(ar1_pred))
        arfima_f.append(np.expm1(arfima_pred))
        fmlar_f.append(np.expm1(fmlar_pred))
        actual.append(raw[t])

    actual = np.array(actual)
    fdict = {"naive": np.array(naive_f), "ar1": np.array(ar1_f),
             "arfima": np.array(arfima_f), "fmlar": np.array(fmlar_f)}
    rmse = {k: float(np.sqrt(np.mean((actual - v) ** 2))) for k, v in fdict.items()}
    dm_stat, dm_p = diebold_mariano(actual - fdict["fmlar"], actual - fdict["arfima"], h=1)

    forecast_table = {"years": years[test_start:].tolist(), "actual": actual.tolist(),
                       "naive": fdict["naive"].tolist(), "ar1": fdict["ar1"].tolist(),
                       "arfima": fdict["arfima"].tolist(), "fmlar": fdict["fmlar"].tolist(),
                       "rmse": rmse, "dm_fmlar_vs_arfima": {"stat": dm_stat, "p_value": dm_p}}

    save("exp5_sunspot.json", {"fit": fit, "tv_track": tv_track, "forecast": forecast_table})
    log(f"Sunspot forecast RMSE: {rmse}; DM stat={dm_stat:.3f} p={dm_p:.3f}")
    return fit, tv_track, forecast_table


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "1"):
        log("=== Experiment 1: constant-memory recovery ===")
        exp1_constant_memory_recovery()
    if which in ("all", "2"):
        log("=== Experiment 2: time-varying trajectories ===")
        exp2_time_varying()
    if which in ("all", "3"):
        log("=== Experiment 3: Ulam-Hyers check ===")
        exp3_ulam_hyers()
    if which in ("all", "4"):
        log("=== Experiment 4: Nile analysis ===")
        nile_analysis()
    if which in ("all", "5"):
        log("=== Experiment 5: Sunspot analysis ===")
        sunspot_analysis()
    log("DONE")
