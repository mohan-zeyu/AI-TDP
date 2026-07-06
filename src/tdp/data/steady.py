"""Steady-state segment detection and canonical (time-averaged) field extraction."""

from __future__ import annotations

import numpy as np


def detect_steady(
    t_s: np.ndarray,
    series: np.ndarray,
    win_s: float = 63.0,
    max_slope_c_per_min: float = 0.2,
    max_range_c: float = 0.3,
) -> np.ndarray:
    """Boolean mask: frame i is steady if, over the trailing `win_s` window, the
    series drifts slower than `max_slope_c_per_min` and spans less than `max_range_c`."""
    steady = np.zeros(len(t_s), dtype=bool)
    for i in range(len(t_s)):
        sel = (t_s >= t_s[i] - win_s) & (t_s <= t_s[i])
        if sel.sum() < 3:
            continue
        tw, yw = t_s[sel], series[sel]
        slope = float(np.polyfit(tw, yw, 1)[0]) * 60.0  # °C/min
        steady[i] = abs(slope) <= max_slope_c_per_min and float(np.ptp(yw)) <= max_range_c
    return steady


def canonical_steady(
    T: np.ndarray,
    t_s: np.ndarray,
    steady: np.ndarray,
    min_spacing_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Mean field and per-pixel std over steady frames subsampled >= min_spacing_s apart.

    The subsampling makes the retained frames approximately independent, so the
    std map doubles as a measured per-pixel repeatability (sensor noise) estimate
    and n_eff is an honest effective sample size.
    """
    picked: list[int] = []
    last_t = -np.inf
    for i in np.flatnonzero(steady):
        if t_s[i] - last_t >= min_spacing_s:
            picked.append(int(i))
            last_t = t_s[i]
    if not picked:
        raise ValueError("no steady frames found")
    sub = T[picked].astype(np.float64)
    return sub.mean(0), sub.std(0, ddof=1) if len(picked) > 1 else np.zeros_like(sub[0]), len(picked), picked
