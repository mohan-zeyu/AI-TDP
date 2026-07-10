"""M7 Tier-1 runner: λ grid on validation patches → final test on the untouched
full-load cases (case02 s1; case06 s2 cross-session).

Usage:  uv run scripts/run_finetune.py [--steps 600]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator

from tdp.data.patches import patch_pixels
from tdp.model.normalization import px_to_xy
from tdp.model.operator import load_checkpoint
from tdp.train.finetune import (
    FinetuneConfig,
    eval_patches,
    finetune,
    load_case,
    make_sensor_tensor,
    predict_field,
    read_sites,
    sites_px_for,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "real"
CFG = ROOT / "configs"
QC = ROOT / "reports" / "qc"

LAMBDA_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]


def grid_xy(shape):
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return px_to_xy(rr.ravel(), cc.ravel(), n_rows=shape[0], n_cols=shape[1])


def test_case(model, card, case, sites, session, patches_rc=None):
    spx = sites_px_for(sites, session, case.shape)
    xy, vals = read_sites(case.field, spx)
    sensors, dT = make_sensor_tensor(xy, vals, case.amb)
    q_xy = grid_xy(case.shape)
    pred = (predict_field(model, card, sensors, case.aspect, q_xy)
            .reshape(case.shape) * dT + case.amb)
    rbf = RBFInterpolator(xy, vals, kernel="thin_plate_spline")(q_xy).reshape(case.shape)

    out = {}
    for tag, p in [("tier1", pred), ("RBF", rbf)]:
        err = p - case.field
        pk_t = np.unravel_index(np.argmax(case.field), case.shape)
        pk_p = np.unravel_index(np.argmax(p), case.shape)
        m = {
            "rmse_trust": float(np.sqrt((err[case.trusted] ** 2).mean())),
            "max_T_err": float(p.max() - case.field.max()),
            "hotspot_px": float(np.hypot(pk_p[0] - pk_t[0], pk_p[1] - pk_t[1])),
        }
        if patches_rc:
            pm = patch_pixels(patches_rc, case.shape)
            m["rmse_patches"] = float(np.sqrt((err[pm] ** 2).mean()))
        out[tag] = m
    return out, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "models" / "v2" / "pretrain_v2.pt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--card-name", default="pi4b_s1",
                    help="board-card file stem under models/boards/")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.ckpt)
    model.eval()
    sites = json.loads((CFG / "sensors_board.json").read_text(encoding="utf-8"))["sites"]
    rects = json.loads((CFG / "source_mask_board.json").read_text(encoding="utf-8"))["rects"]
    patches = json.loads((CFG / "validation_patches.json").read_text(encoding="utf-8"))
    p_s1 = [tuple(rc) for rc in patches["s1"]["patches_rc"]]
    p_s2 = [tuple(rc) for rc in patches["s2"]["patches_rc"]]

    train_cases = [load_case(PROC, "case01_idle"), load_case(PROC, "case04_half_load")]

    print("=== λ grid (selected on s1 validation patches) ===")
    results = {}
    for lam in LAMBDA_GRID:
        cfg = FinetuneConfig(steps=args.steps, lambda_pde=lam)
        card, hist = finetune(model, train_cases, sites, rects, p_s1, cfg,
                              session="s1", verbose=False)
        results[lam] = (hist["best_patch_rmse"], card, hist)
        print(f"λ={lam:g}: patch RMSE {hist['best_patch_rmse']:.3f} °C  "
              f"(ĥ={hist['h_hat']:.2f}, γ={hist['gamma']:.2f}, best@{hist['best_step']})")
    lam_best = min(results, key=lambda k: results[k][0])
    patch_rmse, card, hist = results[lam_best]
    print(f"-> selected λ={lam_best:g}, patch RMSE {patch_rmse:.3f} °C, "
          f"ĥ={hist['h_hat']:.2f} (measured bound ≲3)")

    # zero-shot baseline at the same patches (tokens-free, context-free would be
    # unfair — use the pretrained model with a fresh random card? No: baseline =
    # untrained warm-start card, no optimization steps.)
    cfg0 = FinetuneConfig(steps=1, lambda_pde=0.0)
    card0, _ = finetune(model, train_cases, sites, rects, p_s1, cfg0,
                        session="s1", verbose=False)
    zs_patch = eval_patches(model, card0, train_cases, p_s1, sites, "s1")
    gain = 100 * (1 - patch_rmse / zs_patch)
    print(f"zero-shot(warm-start only) patch RMSE {zs_patch:.3f} °C → "
          f"Tier-1 improvement {gain:.1f}% (acceptance ≥ 20%)")

    print("\n=== final test (untouched full-load cases) ===")
    c02 = load_case(PROC, "case02_full_load")
    c06 = load_case(PROC, "case06_full_load_20min")
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    report = {"lambda": lam_best, "h_hat": hist["h_hat"], "gamma": hist["gamma"],
              "patch_rmse_c": patch_rmse, "zeroshot_patch_rmse_c": zs_patch,
              "tests": {}}
    for row, (case, sess, prc) in enumerate([(c02, "s1", p_s1), (c06, "s2", p_s2)]):
        res, pred = test_case(model, card, case, sites, sess, prc)
        report["tests"][case.case_id] = res
        for tag, m in res.items():
            extra = f"  patches {m['rmse_patches']:.2f}" if "rmse_patches" in m else ""
            print(f"{case.case_id:26s} {tag:6s} trust-RMSE {m['rmse_trust']:.2f} °C  "
                  f"maxT {m['max_T_err']:+.2f}  hotspot {m['hotspot_px']:.1f}px{extra}")
        vmin, vmax = case.field.min(), case.field.max()
        for col, (title, img, cmap, kw) in enumerate([
            (f"{case.case_id} measured", case.field, "hot", dict(vmin=vmin, vmax=vmax)),
            ("Tier-1 board card", pred, "hot", dict(vmin=vmin, vmax=vmax)),
            ("|error| (°C)", np.abs(pred - case.field), "viridis", {}),
        ]):
            im = axes[row, col].imshow(img, cmap=cmap, **kw)
            axes[row, col].set_title(title, fontsize=10)
            fig.colorbar(im, ax=axes[row, col], fraction=0.046)
            for r, c in prc:
                axes[row, col].plot(c, r, "wx", ms=4)
    fig.suptitle(f"M7 Tier-1: frozen backbone + {card.tokens.shape[1]} board tokens "
                 f"(λ={lam_best:g}, ĥ={hist['h_hat']:.2f})", fontsize=12)
    fig.tight_layout()
    fig.savefig(QC / f"finetune_tier1_{args.card_name}.png", dpi=110)

    card_path = ROOT / "models" / "boards" / f"{args.card_name}.pt"
    card.save(card_path, meta={
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": args.ckpt.name, "lambda_pde": lam_best,
        "train_cases": [c.case_id for c in train_cases], "session": "s1",
    })
    (ROOT / "models" / "boards" / f"{args.card_name}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    n_train = sum(p.numel() for p in card.parameters())
    print(f"\nboard card: {card_path.relative_to(ROOT)} "
          f"({n_train} trainable params, {card_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
