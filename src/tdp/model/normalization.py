"""The sim-to-real contract: nondimensionalization and coordinate conventions.

Everything the operator sees is nondimensional:
  - excess temperature  θ = (T − T_amb) / ΔT_s, with ΔT_s = max sensor excess
    (floored) — computable from the sensor inputs alone at inference time;
  - coordinates normalized by the board's LONG side: y = (row+0.5)/n_rows ∈ (0,1)
    top→bottom, x = (col+0.5)/n_rows ∈ (0, aspect), so the board occupies
    [0, a]×[0, 1] with a = n_cols/n_rows (Pi 4B crop: a = 103/157 ≈ 0.656);
  - the steady PDE collapses to ∇̃²θ − ĥ·θ + Q̂ = 0 with the single physics
    parameter ĥ = h·L²/(k_eff·d).

v2 checkpoints carry NO dataset statistics (the v1 T_mean/T_std were a leakage
channel); per-frame normalization replaces them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOARD_N_ROWS = 157
BOARD_N_COLS = 103
BOARD_ASPECT = BOARD_N_COLS / BOARD_N_ROWS  # ≈ 0.656
DT_FLOOR_C = 1.0  # °C floor on ΔT_s for real frames (avoids noise blow-up at idle)


@dataclass
class FrameNorm:
    t_amb: float      # °C (real: border median; synthetic: 0 by construction)
    dT_scale: float   # °C per θ-unit
    aspect: float


def dT_scale_from_sensors(sensor_temps_c: np.ndarray, t_amb: float,
                          floor_c: float = DT_FLOOR_C) -> float:
    return float(max(np.max(sensor_temps_c) - t_amb, floor_c))


def theta_from_temp(T_c: np.ndarray | float, norm: FrameNorm):
    return (T_c - norm.t_amb) / norm.dT_scale


def temp_from_theta(theta, norm: FrameNorm):
    return theta * norm.dT_scale + norm.t_amb


def px_to_xy(rows: np.ndarray, cols: np.ndarray,
             n_rows: int = BOARD_N_ROWS, n_cols: int = BOARD_N_COLS) -> np.ndarray:
    """Pixel (row, col) → nondimensional (x, y); long side (rows) normalized to 1."""
    y = (np.asarray(rows, dtype=np.float64) + 0.5) / n_rows
    x = (np.asarray(cols, dtype=np.float64) + 0.5) / n_rows
    return np.stack([x, y], axis=-1)


def xy_to_px(xy: np.ndarray, n_rows: int = BOARD_N_ROWS) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64)
    cols = xy[..., 0] * n_rows - 0.5
    rows = xy[..., 1] * n_rows - 0.5
    return rows, cols


def cond_vector(h_hat: float, gamma: float | None, aspect: float, robin: bool) -> np.ndarray:
    """Condition-token input: [scaled log ĥ, scaled log γ (0 if Dirichlet), aspect, BC flag]."""
    return np.array(
        [
            np.log(h_hat) / 3.0,
            (np.log(gamma) / 3.0) if robin and gamma is not None else 0.0,
            aspect,
            1.0 if robin else 0.0,
        ],
        dtype=np.float32,
    )
