"""Validation patches: held-out 3×3 spots inside the trusted area, stratified
across the temperature range, kept away from sensor sites and each other.
They are never supervised and never used as input sensors — they drive λ
selection, early stopping, and the honest spatial-accuracy metric."""

from __future__ import annotations

import numpy as np


def make_validation_patches(
    field: np.ndarray,
    trusted: np.ndarray,
    sites_px: np.ndarray,      # (S, 2) rows/cols of sensor sites
    n_patches: int = 12,
    exclude_r: float = 5.0,    # min distance from sensor sites (px)
    min_sep: float = 8.0,      # min distance between patches (px)
    border: int = 3,
) -> list[tuple[int, int]]:
    h, w = field.shape
    ok = trusted.copy()
    ok[:border] = ok[-border:] = False
    ok[:, :border] = ok[:, -border:] = False
    rows, cols = np.nonzero(ok)
    vals = field[rows, cols]

    rr, cc = rows.astype(float), cols.astype(float)
    far_from_sites = np.ones(len(rows), dtype=bool)
    for sr, sc in sites_px:
        far_from_sites &= np.hypot(rr - sr, cc - sc) >= exclude_r

    chosen: list[tuple[int, int]] = []
    for q in np.quantile(vals, np.linspace(0.05, 0.97, n_patches)):
        band_eps = 0.3
        while True:
            cand = far_from_sites & (np.abs(vals - q) <= band_eps)
            for pr, pc in chosen:
                cand &= np.hypot(rr - pr, cc - pc) >= min_sep
            if cand.sum() >= 10 or band_eps > 10.0:
                break
            band_eps *= 1.8
        idx = np.nonzero(cand)[0]
        if len(idx) == 0:
            continue
        # maximin: farthest from everything already chosen (spread the patches)
        if chosen:
            d = np.min(
                [np.hypot(rr[idx] - pr, cc[idx] - pc) for pr, pc in chosen], axis=0)
            pick = idx[int(np.argmax(d))]
        else:
            pick = idx[len(idx) // 2]
        chosen.append((int(rows[pick]), int(cols[pick])))
    return chosen


def patch_pixels(patches: list[tuple[int, int]], shape: tuple[int, int],
                 radius: int = 1) -> np.ndarray:
    """Boolean mask of all pixels belonging to the (2r+1)² patches."""
    mask = np.zeros(shape, dtype=bool)
    for r, c in patches:
        mask[max(0, r - radius):r + radius + 1, max(0, c - radius):c + radius + 1] = True
    return mask
