"""Provisional M6 zero-shot gate: pretrained operator on the real board, no fine-tuning.

For each target (session-1 half load, session-2 full load): feed K=8 sensors at
the placeholder board sites + one same-session reference frame as context, and
reconstruct the full field. Compare against thin-plate RBF interpolation given
the identical sensors, and against the no-context model.

PROVISIONAL: sensor sites are the placeholder layout (real tape sites pending
M3) and metrics run over all pixels incl. emissivity-artifact regions (trust
masks pending), so absolute numbers are pessimistic. The gate question is
relative: does the physics-pretrained operator beat plain interpolation?

Usage:  uv run scripts/zero_shot_check.py [--ckpt models/v2/pretrain_v2.pt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator

from tdp.model.normalization import px_to_xy
from tdp.model.operator import load_checkpoint
from tdp.sim.scenarios import PLACEHOLDER_BOARD_LAYOUT

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "real"
QC = ROOT / "reports" / "qc"


def field_from_canonical(path: Path) -> tuple[np.ndarray, float]:
    d = np.load(path)
    return d["mean_T"].astype(np.float64), float(d["t_amb"])


def field_from_tail(path: Path, n_tail: int = 10) -> tuple[np.ndarray, float]:
    d = np.load(path)
    return (d["T"][-n_tail:].astype(np.float64).mean(0),
            float(np.median(d["t_amb"][-n_tail:])))


def grid_xy(shape: tuple[int, int]) -> np.ndarray:
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return px_to_xy(rr.ravel(), cc.ravel(), n_rows=shape[0], n_cols=shape[1])


def sensors_at_sites(field: np.ndarray, amb: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Placeholder sites (u,v) -> (K,2) nondim coords + readings (3x3 median)."""
    h, w = field.shape
    rows = np.clip((PLACEHOLDER_BOARD_LAYOUT[:, 1] * h).astype(int), 1, h - 2)
    cols = np.clip((PLACEHOLDER_BOARD_LAYOUT[:, 0] * w).astype(int), 1, w - 2)
    vals = np.array([np.median(field[r - 1:r + 2, c - 1:c + 2]) for r, c in zip(rows, cols)])
    xy = px_to_xy(rows, cols, n_rows=h, n_cols=w)
    dT = max(vals.max() - amb, 1.0)
    return xy, vals, dT


def context_points(field: np.ndarray, amb: float, n: int = 192, seed: int = 0):
    rng = np.random.default_rng(seed)
    h, w = field.shape
    rows = rng.integers(0, h, n)
    cols = rng.integers(0, w, n)
    xy = px_to_xy(rows, cols, n_rows=h, n_cols=w)
    ex = np.clip(field[rows, cols] - amb, 0, None)
    s_ctx = max(ex.max(), 1e-6)
    pts = np.concatenate([xy, (ex / s_ctx)[:, None]], axis=-1).astype(np.float32)
    return pts, np.zeros(n, dtype=np.int64)


@torch.no_grad()
def predict(model, sens_xy, sens_theta, q_xy, ctx=None, ctx_state=None, chunk=4096):
    sensors = torch.from_numpy(
        np.concatenate([sens_xy, sens_theta[:, None]], -1).astype(np.float32))[None]
    ctx_t = torch.from_numpy(ctx)[None] if ctx is not None else None
    st_t = torch.from_numpy(ctx_state)[None] if ctx_state is not None else None
    outs = []
    for i in range(0, len(q_xy), chunk):
        q = torch.from_numpy(q_xy[i:i + chunk].astype(np.float32))[None]
        outs.append(model(sensors, q, None, context=ctx_t, context_state=st_t)[0].numpy())
    return np.concatenate(outs)


def evaluate_target(name, model, target, amb, ctx_field, ctx_amb):
    h, w = target.shape
    q_xy = grid_xy(target.shape)
    s_xy, s_vals, dT = sensors_at_sites(target, amb)
    theta_s = (s_vals - amb) / dT
    ctx, ctx_state = context_points(ctx_field, ctx_amb)

    pred_ctx = predict(model, s_xy, theta_s, q_xy, ctx, ctx_state).reshape(h, w) * dT + amb
    pred_noctx = predict(model, s_xy, theta_s, q_xy).reshape(h, w) * dT + amb
    rbf = RBFInterpolator(s_xy, s_vals, kernel="thin_plate_spline")(q_xy).reshape(h, w)

    hot = target - amb > 3.0  # crude artifact exclusion (connector shields read ~ambient)
    res = {}
    for tag, pred in [("model+ctx", pred_ctx), ("model", pred_noctx), ("RBF", rbf)]:
        err = pred - target
        pk_t = np.unravel_index(np.argmax(target), target.shape)
        pk_p = np.unravel_index(np.argmax(pred), pred.shape)
        res[tag] = {
            "rmse_all": float(np.sqrt((err ** 2).mean())),
            "rmse_hot": float(np.sqrt((err[hot] ** 2).mean())),
            "max_T_err": float(pred.max() - target.max()),
            "hotspot_dist_px": float(np.hypot(pk_p[0] - pk_t[0], pk_p[1] - pk_t[1])),
        }

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))
    vmin, vmax = target.min(), target.max()
    for ax, (title, img) in zip(axes, [
        ("measured (canonical)", target), ("model + context", pred_ctx),
        ("model, no context", pred_noctx), ("RBF interpolation", rbf),
    ]):
        im = ax.imshow(img, cmap="hot", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}", fontsize=10)
        ax.plot(s_xy[:, 0] * h - 0.5, s_xy[:, 1] * h - 0.5, "c^", ms=5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    im = axes[4].imshow(np.abs(pred_ctx - target), cmap="viridis")
    axes[4].set_title("|model+ctx − measured| (°C)", fontsize=10)
    fig.colorbar(im, ax=axes[4], fraction=0.046)
    fig.suptitle(f"zero-shot (no fine-tuning) — {name}, K=8 placeholder sites", fontsize=12)
    fig.tight_layout()
    fig.savefig(QC / f"zeroshot_{name}.png", dpi=110)
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "models" / "v2" / "pretrain_v2.pt")
    args = ap.parse_args()
    model, ckpt = load_checkpoint(args.ckpt)
    model.eval()
    print(f"checkpoint {args.ckpt.name} (epoch {ckpt.get('epoch')}, "
          f"val ctx {ckpt.get('val_rmse_ctx'):.4f})")

    targets = []
    t4, a4 = field_from_canonical(PROC / "canonical_case04_half_load.npz")
    t1, a1 = field_from_canonical(PROC / "canonical_case01_idle.npz")
    targets.append(("s1_half_load", t4, a4, t1, a1))
    t6, a6 = field_from_canonical(PROC / "canonical_case06_full_load_20min.npz")
    t7, a7 = field_from_tail(PROC / "case07_cooldown_full_20min.npz")
    targets.append(("s2_full_load", t6, a6, t7, a7))

    print(f"\n{'target':14s} {'method':10s} {'RMSE_all':>9s} {'RMSE_hot':>9s} "
          f"{'maxT_err':>9s} {'hotspot_px':>11s}")
    for name, target, amb, ctxf, ctxa in targets:
        res = evaluate_target(name, model, target, amb, ctxf, ctxa)
        for tag, m in res.items():
            print(f"{name:14s} {tag:10s} {m['rmse_all']:9.2f} {m['rmse_hot']:9.2f} "
                  f"{m['max_T_err']:+9.2f} {m['hotspot_dist_px']:11.1f}")
    print(f"\nfigures: reports/qc/zeroshot_*.png   (provisional: placeholder sites, no trust mask)")


if __name__ == "__main__":
    main()
