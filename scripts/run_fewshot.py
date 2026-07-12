"""Few-shot board-card calibration: how many measurement frames does a new
board need?

Refits the Tier-1 card with only N steady frames per training condition
(N = 1, 2, 4, all), plus the deployment-extreme configs "one half-load frame
only" and "one idle frame only". Fixed recipe (λ=0, 600 steps — no re-tuning
per N), 3 seeds each; evaluation is always against the FULL-data canonical
fields of the untouched full-load cases.

Usage:  uv run scripts/run_fewshot.py [--seeds 3]
Outputs: reports/results/fewshot.json, reports/figures/fig_eval_fewshot.png
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

from tdp.eval.metrics import field_metrics
from tdp.eval.protocols import grid_xy, ours_predict_T
from tdp.model.operator import load_checkpoint
from tdp.train.finetune import (
    FinetuneConfig,
    finetune,
    load_case,
    load_case_subset,
    read_sites,
    sites_px_for,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "real"
CFG = ROOT / "configs"
RES = ROOT / "reports" / "results"
FIG = ROOT / "reports" / "figures"

BLUE, GRAY = "#2a78d6", "#64748b"


def eval_test(model, card, case, sites, session) -> float:
    spx = sites_px_for(sites, session, case.shape)
    xy, vals = read_sites(case.field, spx)
    pred = ours_predict_T(model, xy, vals, case.amb, case.aspect,
                          grid_xy(case.shape), card=card).reshape(case.shape)
    return field_metrics(pred, case.field, case.trusted)["rmse_trust"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    model, _ = load_checkpoint(ROOT / "models/v2/pretrain_v2.pt")
    model.eval()
    sites = json.loads((CFG / "sensors_board.json").read_text("utf-8"))["sites"]
    rects = json.loads((CFG / "source_mask_board.json").read_text("utf-8"))["rects"]
    p_s1 = [tuple(rc) for rc in
            json.loads((CFG / "validation_patches.json").read_text("utf-8"))
            ["s1"]["patches_rc"]]

    c02 = load_case(PROC, "case02_full_load")      # evaluation ground truth:
    c06 = load_case(PROC, "case06_full_load_20min")  # always full-data canonicals

    # (config name, [(case id, n_frames)], x position for the frames sweep)
    configs = [
        ("both_1f", [("case01_idle", 1), ("case04_half_load", 1)], 1),
        ("both_2f", [("case01_idle", 2), ("case04_half_load", 2)], 2),
        ("both_4f", [("case01_idle", 4), ("case04_half_load", 4)], 4),
        ("both_all", [("case01_idle", None), ("case04_half_load", None)], 11),
        ("half_only_1f", [("case04_half_load", 1)], None),
        ("idle_only_1f", [("case01_idle", 1)], None),
    ]

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "recipe": {"lambda_pde": 0.0, "steps": args.steps, "seeds": args.seeds},
           "configs": {}}
    for name, spec, _x in configs:
        rows = {"case02": [], "case06": [], "n_frames_total": []}
        for seed in range(args.seeds):
            train_cases, n_tot = [], 0
            for cid, n in spec:
                case, n_used = load_case_subset(PROC, cid, n, seed=seed * 100 + 7)
                train_cases.append(case)
                n_tot += n_used
            card, _ = finetune(model, train_cases, sites, rects, p_s1,
                               FinetuneConfig(steps=args.steps, lambda_pde=0.0,
                                              seed=seed),
                               session="s1", verbose=False)
            rows["case02"].append(eval_test(model, card, c02, sites, "s1"))
            rows["case06"].append(eval_test(model, card, c06, sites, "s2"))
            rows["n_frames_total"].append(n_tot)
        out["configs"][name] = {
            "n_frames_total": int(np.mean(rows["n_frames_total"])),
            "case02_mean": float(np.mean(rows["case02"])),
            "case02_std": float(np.std(rows["case02"])),
            "case06_mean": float(np.mean(rows["case06"])),
            "case06_std": float(np.std(rows["case06"])),
        }
        c = out["configs"][name]
        print(f"{name:14s} ({c['n_frames_total']:2d} frames): "
              f"case02 {c['case02_mean']:.2f}±{c['case02_std']:.2f} °C · "
              f"case06 {c['case06_mean']:.2f}±{c['case06_std']:.2f} °C")

    # references from the frozen headline run
    heads = json.loads((RES / "results.json").read_text("utf-8"))["real_headline"]
    refs = {cid: {"zeroshot": heads[cid]["zeroshot_ctx"]["rmse_trust"],
                  "rbf": heads[cid]["rbf"]["rmse_trust"]}
            for cid in heads}
    out["references"] = refs

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "fewshot.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- figure: frames sweep + single-condition extremes ----
    sweep = [(x, out["configs"][n]) for n, _s, x in configs if x is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, cid, key in ((axes[0], "case02_full_load", "case02"),
                         (axes[1], "case06_full_load_20min", "case06")):
        xs = [x for x, _ in sweep]
        ys = [c[f"{key}_mean"] for _, c in sweep]
        es = [c[f"{key}_std"] for _, c in sweep]
        ax.errorbar(xs, ys, yerr=es, fmt="-o", ms=5, lw=2, capsize=3, color=BLUE,
                    label="board card (idle + half load)")
        for name, marker, lbl in (("half_only_1f", "s", "1 half-load frame only"),
                                  ("idle_only_1f", "^", "1 idle frame only")):
            c = out["configs"][name]
            ax.errorbar([1], [c[f"{key}_mean"]], yerr=[c[f"{key}_std"]], fmt=marker,
                        ms=8, capsize=3, color=BLUE, mfc="white", label=lbl)
        ax.axhline(refs[cid]["zeroshot"], color=GRAY, ls="--", lw=1.5,
                   label="zero-shot + context")
        ax.axhline(refs[cid]["rbf"], color="#eb6834", ls=":", lw=1.5,
                   label="RBF interpolation")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) if x < 11 else "all" for x in xs])
        ax.set_xlabel("calibration frames per condition")
        ax.set_title(cid, fontsize=10)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("trusted-pixel RMSE (°C)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Few-shot board calibration — untouched full-load tests", fontsize=12)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_eval_fewshot.png", dpi=130)
    print(f"-> {RES / 'fewshot.json'}, reports/figures/fig_eval_fewshot.png")


if __name__ == "__main__":
    main()
