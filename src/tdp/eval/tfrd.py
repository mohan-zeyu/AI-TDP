"""Adapter for the public TFR-HSS benchmark (TFRD dataset, arXiv:2108.08298).

TFRD samples are .mat files with:
  u     — the true temperature field (n×n, Kelvin, ambient 298 K)
  u_obs — the field sampled at the fixed monitoring points (0 elsewhere)
  F     — the heat-source layout
Their task (reconstruct the field from the monitoring points) is exactly our
no-context operator task; the adapter converts a sample into our nondimensional
convention: θ = (T − 298)/s, coordinates normalized by the grid side (aspect 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from tdp.model.normalization import px_to_xy

AMBIENT_K = 298.0
TOL = 1e-14


def read_list(root: Path, list_path: str) -> list[Path]:
    """TFRD lists hold bare names resolved against a sibling directory:
    test/test_N.txt -> test/test_N/<name>, train/train_val.txt -> train/train/<name>."""
    lp = root / list_path
    lines = lp.read_text().split()
    list_dir, stem = lp.parent, lp.stem
    files = []
    for ln in lines:
        for cand in (list_dir / stem / ln, list_dir / "train" / ln,
                     list_dir / ln, root / ln):
            for p in (cand, cand.with_suffix(".mat")):
                if p.is_file():
                    files.append(p)
                    break
            else:
                continue
            break
    if not files:
        raise FileNotFoundError(f"no samples resolved from {lp}")
    return files


def load_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    m = sio.loadmat(str(path))
    return m["u"].astype(np.float64), m["u_obs"].astype(np.float64)


def sensors_from_obs(u_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monitoring-point coordinates (nondim) and temperatures (K)."""
    rows, cols = np.nonzero(u_obs > TOL)
    n = u_obs.shape[0]
    xy = px_to_xy(rows, cols, n_rows=n, n_cols=u_obs.shape[1])
    return xy, u_obs[rows, cols]


def to_model_inputs(u: np.ndarray, u_obs: np.ndarray):
    """(sensors tensor (1,K,3), scale s, grid queries, θ truth) for one sample."""
    xy, temps = sensors_from_obs(u_obs)
    theta_s = temps - AMBIENT_K
    s = float(max(theta_s.max(), 1.0))
    sensors = torch.from_numpy(
        np.concatenate([xy, (theta_s / s)[:, None]], -1).astype(np.float32))[None]
    n = u.shape[0]
    rr, cc = np.meshgrid(np.arange(n), np.arange(u.shape[1]), indexing="ij")
    q_xy = px_to_xy(rr.ravel(), cc.ravel(), n_rows=n, n_cols=u.shape[1])
    theta_true = (u.ravel() - AMBIENT_K) / s
    return sensors, s, q_xy, theta_true


class TFRDDataset(Dataset):
    """Training view: emits the same tuple as tdp.train.pretrain.OperatorDataset
    (empty context, condition token dropped, PDE slots dummy — data loss only)."""

    def __init__(self, files: list[Path], n_query: int = 384, seed: int = 0):
        self.files = files
        self.n_query = n_query
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, i))
        u, u_obs = load_sample(self.files[i])
        xy, temps = sensors_from_obs(u_obs)
        theta_s = temps - AMBIENT_K
        s = float(max(theta_s.max(), 1.0))
        sensors = np.concatenate([xy, (theta_s / s)[:, None]], -1).astype(np.float32)

        n = u.shape[0]
        rows = rng.integers(0, n, self.n_query)
        cols = rng.integers(0, u.shape[1], self.n_query)
        q_xy = px_to_xy(rows, cols, n_rows=n, n_cols=u.shape[1]).astype(np.float32)
        q_theta = ((u[rows, cols] - AMBIENT_K) / s).astype(np.float32)

        return (torch.from_numpy(sensors),
                torch.from_numpy(q_xy),
                torch.from_numpy(q_theta),
                torch.zeros(4, 2),                      # collocation dummy (λ=0)
                torch.zeros(4),
                torch.zeros(4, dtype=torch.float32),    # cond (masked)
                torch.tensor(True),                     # cond dropped
                torch.tensor(1.0),
                torch.zeros(0, 3),                      # no context
                torch.zeros(0, dtype=torch.long))


@torch.no_grad()
def evaluate(model, files: list[Path], device="cpu", chunk=8192,
             limit: int | None = None) -> dict:
    """Paper-comparable metrics over a test list: MAE (K) on the full field,
    max-AE, and MAE at the hottest 1% of pixels (hotspot fidelity)."""
    model.eval()
    maes, maxaes, hot_maes = [], [], []
    for path in files[:limit]:
        u, u_obs = load_sample(path)
        sensors, s, q_xy, theta_true = to_model_inputs(u, u_obs)
        sensors = sensors.to(device)
        preds = []
        for j in range(0, len(q_xy), chunk):
            q = torch.from_numpy(q_xy[j:j + chunk].astype(np.float32))[None].to(device)
            preds.append(model(sensors, q, None)[0].cpu().numpy())
        pred_K = np.concatenate(preds) * s + AMBIENT_K
        true_K = theta_true * s + AMBIENT_K
        err = np.abs(pred_K - true_K)
        maes.append(float(err.mean()))
        maxaes.append(float(err.max()))
        hot = true_K >= np.quantile(true_K, 0.99)
        hot_maes.append(float(err[hot].mean()))
    return {"n_samples": len(maes), "mae_K": float(np.mean(maes)),
            "max_ae_K": float(np.mean(maxaes)), "hotspot_mae_K": float(np.mean(hot_maes))}
