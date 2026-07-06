"""Locate and extract the fixed-size board window from raw thermal frames.

The board window (157 rows x 103 cols ≈ the Pi 4B's 85x56 mm PCB) is found by
maximizing the enclosed excess-temperature energy with an integral image — a
threshold-free criterion that is stable against the warm halo around the board.

For re-staged scenes (case00 was shot 2 h later, moved/rotated) the window is
instead registered against a reference crop by zero-normalized cross-correlation
of gradient-magnitude maps over the four 90° rotation hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

DEFAULT_SIZE = (157, 103)  # (rows, cols)


def estimate_ambient(frame: np.ndarray, border_px: int = 8) -> float:
    """Ambient estimate = median of the frame's border ring (background around the board)."""
    b = border_px
    ring = np.concatenate(
        [frame[:b].ravel(), frame[-b:].ravel(), frame[b:-b, :b].ravel(), frame[b:-b, -b:].ravel()]
    )
    return float(np.median(ring))


def find_crop(
    frame: np.ndarray,
    size: tuple[int, int] = DEFAULT_SIZE,
    t_amb: float | None = None,
) -> tuple[int, int, float]:
    """Top-left (row0, col0) of the `size` window maximizing enclosed excess energy.

    Returns (row0, col0, captured_fraction) where captured_fraction is the share
    of the frame's total excess-temperature energy inside the window.
    """
    h, w = size
    if frame.shape[0] < h or frame.shape[1] < w:
        raise ValueError(f"frame {frame.shape} smaller than window {size}")
    if t_amb is None:
        t_amb = estimate_ambient(frame)
    excess = np.clip(frame.astype(np.float64) - t_amb, 0.0, None)
    ii = np.pad(excess, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    window_sums = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]
    r, c = np.unravel_index(int(np.argmax(window_sums)), window_sums.shape)
    total = excess.sum()
    frac = float(window_sums[r, c] / total) if total > 0 else 0.0
    return int(r), int(c), frac


def apply_crop(
    frames: np.ndarray, row0: int, col0: int, size: tuple[int, int] = DEFAULT_SIZE
) -> np.ndarray:
    """Crop (H, W) or (N, H, W) frames to the fixed window."""
    h, w = size
    if frames.ndim == 2:
        return frames[row0 : row0 + h, col0 : col0 + w]
    return frames[:, row0 : row0 + h, col0 : col0 + w]


def _gradient_magnitude(frame: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(frame.astype(np.float64))
    return np.hypot(gy, gx)


def _zncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-normalized cross-correlation of `template` slid over `image` ('valid')."""
    t = template - template.mean()
    t_norm = float(np.sqrt((t * t).sum()))
    out_shape = (image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1)
    if t_norm < 1e-12 or min(out_shape) < 1:
        return np.zeros(out_shape)
    num = fftconvolve(image, t[::-1, ::-1], mode="valid")
    ones = np.ones_like(t)
    win_sum = fftconvolve(image, ones, mode="valid")
    win_sumsq = fftconvolve(image * image, ones, mode="valid")
    win_var = np.clip(win_sumsq - win_sum**2 / t.size, 1e-12, None)
    return num / (np.sqrt(win_var) * t_norm)


@dataclass
class Registration:
    rot90: int  # number of CCW 90° rotations applied to the raw frame before cropping
    row0: int
    col0: int
    zncc: float
    candidates: dict[int, tuple[int, int, float]]  # rot90 -> (row0, col0, zncc)


def register_crop(
    frame: np.ndarray,
    ref_patch: np.ndarray,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> Registration:
    """Register a re-staged frame against a reference crop.

    Gradient-magnitude ZNCC template matching over the four rotation hypotheses;
    gradients are level-invariant, so a cold (near-isothermal) board can still be
    matched against a hot reference through its edge/component pattern.
    """
    template = _gradient_magnitude(ref_patch)
    candidates: dict[int, tuple[int, int, float]] = {}
    for k in range(4):
        rotated = np.rot90(frame, k)
        if rotated.shape[0] < size[0] or rotated.shape[1] < size[1]:
            continue
        zmap = _zncc_map(_gradient_magnitude(rotated), template)
        r, c = np.unravel_index(int(np.argmax(zmap)), zmap.shape)
        candidates[k] = (int(r), int(c), float(zmap[r, c]))
    best_k = max(candidates, key=lambda k: candidates[k][2])
    r, c, score = candidates[best_k]
    return Registration(rot90=best_k, row0=r, col0=c, zncc=score, candidates=candidates)
