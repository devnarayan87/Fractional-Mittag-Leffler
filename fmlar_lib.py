"""
fmlar_lib.py
-------------
Core library used to (re)produce every numerical/figure result in the
revised FMLAR paper. Implements:

  - the Mittag-Leffler weight kernel psi_j(alpha,theta) = E_alpha(-theta j^alpha)
  - FMLAR (constant- and time-varying-memory) simulation
  - windowed / global GPH estimation
  - differential-evolution ACF-shape estimation
  - Whittle MLE estimation
  - a shallow MLP local-memory tracker
  - ARFIMA(0,d,0) simulation/forecasting utilities (for benchmarking)
  - a Diebold-Mariano test
"""

import numpy as np
from scipy.special import gammaln, rgamma
from scipy.optimize import differential_evolution, minimize
from scipy import stats


# ----------------------------------------------------------------------
# Mittag-Leffler function (robust hybrid: series for small |z|, optimally
# truncated asymptotic expansion for large |z|; alpha in (0,1)).
# ----------------------------------------------------------------------
def _ml_series(z, alpha, n_terms=200, tol=1e-14):
    sign_z = np.sign(z)
    log_absz = np.log(np.abs(z) + 1e-300)
    result = np.ones_like(z)
    streak = np.zeros_like(z, dtype=int)
    for k in range(1, n_terms):
        log_term = k * log_absz - gammaln(alpha * k + 1.0)
        term = (sign_z ** k) * np.exp(np.clip(log_term, -700, 700))
        result = result + term
        small = np.abs(term) < tol * (np.abs(result) + 1e-300)
        streak = np.where(small, streak + 1, 0)
        if np.all(streak >= 3):
            break
    return result


def _ml_asymptotic(z, alpha, max_terms=30):
    K = max_terms
    terms = np.empty((K,) + z.shape)
    for k in range(1, K + 1):
        terms[k - 1] = (z ** (-k)) * rgamma(1.0 - alpha * k)
    abs_terms = np.abs(terms)
    k_star = np.argmin(abs_terms, axis=0)
    cumsum = np.cumsum(terms, axis=0)
    flat_cumsum = cumsum.reshape(K, -1)
    flat_kstar = k_star.reshape(-1)
    picked = flat_cumsum[flat_kstar, np.arange(flat_kstar.size)]
    return -picked.reshape(z.shape)


def mittag_leffler(z, alpha, threshold=6.0):
    """E_alpha(z) for real z (vectorised), alpha in (0,1]."""
    z = np.asarray(z, dtype=float)
    if alpha >= 1.0 - 1e-12:
        return np.exp(z)
    result = np.empty_like(z)
    small = np.abs(z) <= threshold
    large = ~small
    if np.any(small):
        result[small] = _ml_series(z[small], alpha)
    if np.any(large):
        zl = z[large]
        pos = zl > 0
        out = np.empty_like(zl)
        if np.any(~pos):
            out[~pos] = _ml_asymptotic(zl[~pos], alpha)
        if np.any(pos):
            out[pos] = _ml_series(zl[pos], alpha)
        result[large] = out
    return result


def psi_weights(alpha, theta, J):
    """psi_j(alpha,theta) = E_alpha(-theta*j^alpha), j=0..J, psi_0=1."""
    j = np.arange(0, J + 1, dtype=float)
    z = np.empty_like(j)
    z[0] = 0.0
    z[1:] = -theta * j[1:] ** alpha
    psi = mittag_leffler(z, alpha)
    psi[0] = 1.0
    return psi


# ----------------------------------------------------------------------
# FMLAR simulation (constant and time-varying memory)
# ----------------------------------------------------------------------
def simulate_fmlar(alpha, theta, T, J, rng, burn=None):
    """Constant-memory FMLAR(alpha,theta): X_t = sum_{j=0}^J psi_j eps_{t-j}."""
    burn = burn if burn is not None else J
    psi = psi_weights(alpha, theta, J)
    eps = rng.standard_normal(T + burn)
    X = np.convolve(eps, psi, mode="full")[: T + burn]
    return X[burn:]


def simulate_fmlar_tv(alpha_func, theta, T, J, rng, burn=None):
    """
    Time-varying FMLAR: frozen-kernel construction,
    X_t = sum_{j=0}^{min(t,J)} psi_j(alpha(t),theta) eps_{t-j}.
    `alpha_func(t)` returns alpha at absolute time index t in [0,T).
    Returns X (length T) and the alpha(t) trajectory actually used.
    """
    burn = burn if burn is not None else J
    Ttot = T + burn
    eps = rng.standard_normal(Ttot)
    X = np.zeros(Ttot)
    # Precompute weights only when alpha changes materially (speed-up):
    # here we recompute per t but cache on rounded alpha to 3 decimals.
    cache = {}
    alphas_used = np.empty(Ttot)
    for t in range(Ttot):
        t_rel = t - burn  # relative time to report alpha(t) on the [0,T) scale
        a = alpha_func(max(t_rel, 0))
        alphas_used[t] = a
        key = round(a, 3)
        if key not in cache:
            cache[key] = psi_weights(a, theta, J)
        psi = cache[key]
        jmax = min(t, J)
        X[t] = np.dot(psi[: jmax + 1], eps[t - jmax: t + 1][::-1])
    return X[burn:], alphas_used[burn:]


# ----------------------------------------------------------------------
# GPH estimator (global and windowed)
# ----------------------------------------------------------------------
def periodogram(x):
    n = len(x)
    xm = x - x.mean()
    f = np.fft.rfft(xm)
    I = (np.abs(f) ** 2) / (2 * np.pi * n)
    lam = 2 * np.pi * np.arange(len(f)) / n
    return lam[1:], I[1:]  # drop zero frequency


def gph_estimate(x, m=None):
    """Global GPH log-periodogram regression; returns (d_hat, se).

    Regression: log I(lam_j) = c - d * log(4 sin^2(lam_j/2)) + u_j,
    since f(lam) ~ c |lam|^{-2d} ~ c' (4 sin^2(lam/2))^{-d} near lam=0.
    """
    n = len(x)
    if m is None:
        m = int(n ** 0.5)
    lam, I = periodogram(x)
    lam_m, I_m = lam[:m], I[:m]
    y = np.log(I_m + 1e-300)
    X = np.log(4 * np.sin(lam_m / 2) ** 2)
    Xc = X - X.mean()
    slope = np.sum(Xc * (y - y.mean())) / np.sum(Xc ** 2)
    d_hat = -slope
    resid = (y - y.mean()) - slope * Xc
    se = np.sqrt((np.pi ** 2 / 6) / np.sum(Xc ** 2))
    return d_hat, se


def windowed_gph(x, window, step, m=None):
    """Sliding-window GPH; returns (centers, d_hat array)."""
    n = len(x)
    centers, dhats = [], []
    for start in range(0, n - window + 1, step):
        seg = x[start: start + window]
        d_hat, _ = gph_estimate(seg, m=m)
        centers.append(start + window // 2)
        dhats.append(d_hat)
    return np.array(centers), np.array(dhats)


# ----------------------------------------------------------------------
# Differential-evolution ACF-shape estimator
# ----------------------------------------------------------------------
def sample_acf(x, K):
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x) / n
    rho = np.empty(K)
    for k in range(1, K + 1):
        rho[k - 1] = np.dot(x[:-k], x[k:]) / n / c0
    return rho


def model_acf(alpha, theta, K, J=400):
    psi = psi_weights(alpha, theta, J)
    gamma0 = np.dot(psi, psi)
    rho = np.empty(K)
    for k in range(1, K + 1):
        if k > J:
            rho[k - 1] = 0.0
        else:
            rho[k - 1] = np.dot(psi[: J + 1 - k], psi[k:]) / gamma0
    return rho


def de_acf_fit(x, K=40, J=300, bounds=((0.51, 0.99), (0.05, 5.0)), seed=0):
    rho_hat = sample_acf(x, K)

    def obj(params):
        a, th = params
        rho_m = model_acf(a, th, K, J=J)
        return np.sum((rho_m - rho_hat) ** 2)

    res = differential_evolution(obj, bounds, seed=seed, tol=1e-8, maxiter=300,
                                  polish=True, updating="deferred", workers=1)
    return res.x[0], res.x[1]


# ----------------------------------------------------------------------
# Whittle MLE estimator
# ----------------------------------------------------------------------
def model_spectrum(alpha, theta, lam, J=400):
    """
    Exact (up to MA truncation at lag J) model spectral density (up to the
    sigma^2/(2*pi) scale factor, applied by the caller), evaluated directly
    at frequency vector `lam` via f(lam) = |Psi(e^{-i*lam})|^2, where
    Psi(z) = sum_{j=0}^J psi_j z^j. This is more direct/robust than going
    through the autocovariance-then-FFT route (no zero-padding / frequency
    subsampling mismatch to get wrong).
    """
    psi = psi_weights(alpha, theta, J)
    j = np.arange(J + 1)
    # lam: (n_freq,), j: (J+1,) -> phase matrix (n_freq, J+1)
    phase = np.outer(lam, j)
    Psi = (psi[None, :] * np.exp(-1j * phase)).sum(axis=1)
    return np.abs(Psi) ** 2


def whittle_negloglik(params, I_obs, lam_obs, J=300):
    alpha, theta, log_sigma2 = params
    if not (0.51 < alpha < 0.999) or theta <= 1e-4:
        return 1e10
    sigma2 = np.exp(log_sigma2)
    f_model = (sigma2 / (2 * np.pi)) * model_spectrum(alpha, theta, lam_obs, J=J)
    f_model = np.maximum(f_model, 1e-12)
    return np.sum(np.log(f_model) + I_obs / f_model)


def whittle_fit(x, n_freq=300, J=300, init=None, n_restarts=5, seed=0):
    lam, I = periodogram(x)
    m = min(n_freq, len(I))
    lam_obs, I_obs = lam[:m], I[:m]
    if init is None:
        init = (0.7, 1.0)
    rng = np.random.default_rng(seed)
    best = None
    log_s2_0 = np.log(np.var(x))
    starts = [(init[0], init[1], log_s2_0)]
    for _ in range(n_restarts - 1):
        a0 = np.clip(init[0] + rng.normal(0, 0.05), 0.52, 0.98)
        th0 = max(0.05, init[1] * np.exp(rng.normal(0, 0.3)))
        starts.append((a0, th0, log_s2_0 + rng.normal(0, 0.2)))
    for x0 in starts:
        res = minimize(whittle_negloglik, x0=np.array(x0),
                        args=(I_obs, lam_obs, J), method="Nelder-Mead",
                        options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-5,
                                 "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
    a_hat, th_hat, logs2 = best.x
    return a_hat, th_hat, np.exp(logs2), best


# ----------------------------------------------------------------------
# MLP local-memory tracker
# ----------------------------------------------------------------------
def log_periodogram_features(x, n_feat=10):
    lam, I = periodogram(x)
    feats = np.log(I[:n_feat] + 1e-300)
    return feats


def build_mlp_training_corpus(n_series, T, J, rng, a_start_range=(0.52, 0.75),
                               a_end_range=(0.55, 0.95), theta_range=(0.5, 2.0),
                               window=600, step=300, n_feat=10):
    X_list, y_list = [], []
    for s in range(n_series):
        a0 = rng.uniform(*a_start_range)
        a1 = rng.uniform(*a_end_range)
        th = rng.uniform(*theta_range)

        def alpha_func(t, a0=a0, a1=a1, T=T):
            frac = min(max(t / T, 0.0), 1.0)
            return a0 + (a1 - a0) * frac

        Xs, alphas = simulate_fmlar_tv(alpha_func, th, T, J, rng)
        n = len(Xs)
        for start in range(0, n - window + 1, step):
            seg = Xs[start:start + window]
            feats = log_periodogram_features(seg, n_feat=n_feat)
            center = start + window // 2
            true_alpha = alphas[center]
            true_d = 1.0 - true_alpha
            X_list.append(feats)
            y_list.append(true_d)
    return np.array(X_list), np.array(y_list)


# ----------------------------------------------------------------------
# ARFIMA(0,d,0) utilities (benchmark model)
# ----------------------------------------------------------------------
def arfima_weights(d, J):
    j = np.arange(0, J + 1)
    logw = gammaln(j + d) - gammaln(d) - gammaln(j + 1)
    w = np.exp(logw)
    w[0] = 1.0
    return w


def simulate_arfima(d, T, J, rng, burn=None):
    burn = burn if burn is not None else J
    w = arfima_weights(d, J)
    eps = rng.standard_normal(T + burn)
    X = np.convolve(eps, w, mode="full")[: T + burn]
    return X[burn:]


# ----------------------------------------------------------------------
# One-step-ahead AR(infty)-style forecasting (works for any psi/weight seq)
# ----------------------------------------------------------------------
def ar_inf_coeffs_from_ma(psi, n_ar):
    """
    Invert an MA(infinity) representation with weights psi (psi[0]=1) to
    the AR(infinity) form X_t = sum_{k=1}^n_ar pi_k X_{t-k} + eps_t,
    via the standard recursive psi/pi relation.
    """
    J = len(psi) - 1
    n_ar = min(n_ar, J)
    pi = np.zeros(n_ar + 1)
    pi[0] = 1.0
    for k in range(1, n_ar + 1):
        s = 0.0
        for i in range(1, k + 1):
            if i <= J:
                s += psi[i] * pi[k - i]
        pi[k] = -s
    return -pi[1:]  # coefficients pi_1..pi_n_ar s.t. Xhat_t = sum pi_k X_{t-k}


def one_step_forecast_from_weights(history, psi, n_ar=60):
    """One-step-ahead forecast of the next value given `history` (most
    recent last) and MA weights `psi`, via AR(infinity) inversion."""
    pi = ar_inf_coeffs_from_ma(psi, min(n_ar, len(history)))
    h = history[::-1][: len(pi)]  # most recent first
    return float(np.dot(pi, h))


# ----------------------------------------------------------------------
# Diebold-Mariano test
# ----------------------------------------------------------------------
def diebold_mariano(e1, e2, h=1, power=2):
    """
    Diebold-Mariano test comparing forecast errors e1, e2 (same length).
    loss_i = |e_i|^power. Returns (DM statistic, two-sided p-value).
    Uses a Newey-West-type long-run variance with h-1 lags (h=1 here).
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    # long-run variance via Newey-West with h-1 lags
    gamma0 = np.var(d, ddof=0)
    lrv = gamma0
    for lag in range(1, h):
        w = 1 - lag / h
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        lrv += 2 * w * cov
    se = np.sqrt(lrv / n)
    dm_stat = dbar / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value
