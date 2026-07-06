"""Physics extraction from measurements: cooling time constants and the spatial
decay length of the steady excess-temperature field.

Both reduce to the single nondimensional parameter of the steady 2-D fin equation
∇̃²θ − ĥ·θ = 0 (source-free regions), where ĥ = h·L²/(k_eff·d) and lengths are
normalized by the board's long side L. The far field of a compact source decays
as θ ∝ K₀(r/L_d) ~ e^{-r/L_d}/√r, so ln(θ·√r) is linear in r with slope −1/L_d.
"""

from __future__ import annotations

import numpy as np


def fit_cooling(t_s: np.ndarray, excess: np.ndarray, min_excess_c: float = 1.5) -> dict:
    """Log-linear fit of excess(t) = θ0·exp(−t/τ) on frames with excess > min_excess_c."""
    sel = excess > min_excess_c
    if sel.sum() < 5:
        raise ValueError("too few frames above min_excess for a cooling fit")
    t, y = t_s[sel], np.log(excess[sel])
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "tau_s": float(-1.0 / slope),
        "theta0_c": float(np.exp(intercept)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_used": int(sel.sum()),
        "t_span_s": float(t[-1] - t[0]),
    }


def radial_decay_fit(
    theta: np.ndarray,
    r_min_px: float = 12.0,
    r_max_px: float = 70.0,
    min_theta_c: float = 0.2,
    bin_px: float = 2.0,
) -> dict:
    """Fit the fin-equation far field θ ∝ e^{-r/L_d}/√r around the field's hotspot.

    Uses the median over angles at each radius (robust to secondary sources).
    Returns L_d in pixels plus the binned profile for plotting.
    """
    peak = np.unravel_index(int(np.argmax(theta)), theta.shape)
    rows, cols = np.indices(theta.shape)
    r = np.hypot(rows - peak[0], cols - peak[1]).ravel()
    v = theta.ravel()

    edges = np.arange(0.0, r.max() + bin_px, bin_px)
    centers, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi)
        if sel.sum() >= 8:
            centers.append(0.5 * (lo + hi))
            medians.append(float(np.median(v[sel])))
    centers = np.asarray(centers)
    medians = np.asarray(medians)

    fit_sel = (centers >= r_min_px) & (centers <= r_max_px) & (medians > min_theta_c)
    if fit_sel.sum() < 5:
        raise ValueError("too few radial bins for a decay fit")
    x = centers[fit_sel]
    y = np.log(medians[fit_sel] * np.sqrt(x))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "L_d_px": float(-1.0 / slope),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "peak_rc": [int(peak[0]), int(peak[1])],
        "n_bins": int(fit_sel.sum()),
        "profile_r_px": centers.tolist(),
        "profile_theta_c": medians.tolist(),
        "fit_r_px": x.tolist(),
        "fit_ln_theta_sqrt_r": y.tolist(),
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
    }


def h_hat_from_decay(L_d_px: float, norm_length_px: float = 157.0) -> float:
    """Nondimensional ĥ = 1/L̃_d² with lengths normalized by the board's long side."""
    return float((norm_length_px / L_d_px) ** 2)


def fit_h_hat_pde(
    theta: np.ndarray,
    norm_length_px: float = 157.0,
    smooth_sigma_px: float = 2.0,
    source_quantile: float = 0.75,
    border_px: int = 4,
    min_theta_c: float = 0.5,
) -> dict:
    """Direct least-squares estimate of ĥ from the steady source-free fin equation.

    In source-free regions ∇̃²θ = ĥ·θ (normalized coords), so
    ĥ* = Σ(∇̃²θ·θ)/Σ(θ²) over pixels that are (a) interior, (b) below the
    `source_quantile` of θ (crude source exclusion until components are
    annotated), and (c) warm enough to carry signal. This estimates exactly the
    parameter the fine-tune PDE loss uses — unlike far-field profile fits, it
    stays valid when the decay length is comparable to the board size.
    """
    from scipy.ndimage import gaussian_filter, laplace

    th = gaussian_filter(theta.astype(np.float64), smooth_sigma_px)
    lap = laplace(th) * norm_length_px**2  # ∇̃² with px spacing → normalized coords

    mask = np.zeros(theta.shape, dtype=bool)
    mask[border_px:-border_px, border_px:-border_px] = True
    mask &= th < np.quantile(th, source_quantile)
    mask &= th > min_theta_c

    x, y = th[mask], lap[mask]
    if x.size < 100:
        raise ValueError("too few source-free pixels for the PDE fit")
    h_hat = float((y * x).sum() / (x * x).sum())
    pred = h_hat * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "h_hat": h_hat,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_px": int(x.size),
        "smooth_sigma_px": smooth_sigma_px,
        "source_quantile": source_quantile,
    }
