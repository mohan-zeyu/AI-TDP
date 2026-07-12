"""Shared metrics: every method is scored with exactly these numbers."""

from __future__ import annotations

import numpy as np


def field_metrics(pred: np.ndarray, truth: np.ndarray,
                  trusted: np.ndarray | None = None) -> dict:
    if trusted is None:
        trusted = np.ones_like(truth, dtype=bool)
    err = pred - truth
    pk_t = np.unravel_index(int(np.argmax(truth)), truth.shape)
    pk_p = np.unravel_index(int(np.argmax(pred)), pred.shape)
    return {
        "rmse_trust": float(np.sqrt((err[trusted] ** 2).mean())),
        "mae_trust": float(np.abs(err[trusted]).mean()),
        "max_T_err": float(pred.max() - truth.max()),
        "hotspot_px": float(np.hypot(pk_p[0] - pk_t[0], pk_p[1] - pk_t[1])),
    }
