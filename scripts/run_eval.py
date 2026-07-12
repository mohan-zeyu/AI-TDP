"""M8: the full evaluation matrix — one command regenerates every number/figure.

Synthetic: K-curves (±context, ±PDE twin, vs RBF/GP), amplitude OOD, layout OOD.
Real: headline table on the untouched full-load cases (all methods), K-subset
curve, leave-one-site-out, quasi-static cooling check, diode consistency.

Usage:  uv run scripts/run_eval.py [--quick]
Outputs: reports/results/{results.json, RESULTS.md}, reports/figures/fig_eval_*.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from tdp.eval import protocols as P
from tdp.model.operator import ModelConfig, ThermalOperatorV2, load_checkpoint
from tdp.sim.scenarios import ScenarioConfig, generate_boards
from tdp.train.finetune import FinetuneConfig, finetune, load_case

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "real"
CFG = ROOT / "configs"
RES = ROOT / "reports" / "results"
FIG = ROOT / "reports" / "figures"

# fixed method -> (color, label); identity never re-mapped between figures
STYLE = {
    "bicubic":        ("#eda100", "Bicubic"),
    "rbf":            ("#eb6834", "RBF (thin-plate)"),
    "gp":             ("#1baf7a", "Gaussian process"),
    "pinn":           ("#4a3aa7", "Per-case PINN (60 s)"),
    "zeroshot_noctx": ("#a9c7ec", "Ours · zero-shot"),
    "zeroshot_ctx":   ("#6d9ee1", "Ours · zero-shot + context"),
    "tier1":          ("#2a78d6", "Ours · board card"),
    "tier1_twin":     ("#64748b", "λ=0 twin · board card"),
    "tier1_scratch":  ("#e34948", "No-pretrain ablation"),
    "ours_ctx":       ("#2a78d6", "Ours + context"),
    "ours_noctx":     ("#a9c7ec", "Ours, no context"),
    "twin_ctx":       ("#64748b", "λ=0 twin + context"),
    "twin_noctx":     ("#b6bec9", "λ=0 twin, no context"),
    "ours_card":      ("#2a78d6", "Ours · board card"),
}
GRID_KW = dict(alpha=0.3, linewidth=0.6)


class Card:
    def __init__(self, path: Path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.tokens = d["tokens"].float()
        self.log_h, self.log_gamma = float(d["log_h"]), float(d["log_gamma"])

    def cond(self, aspect: float) -> torch.Tensor:
        return torch.tensor([[self.log_h / 3, self.log_gamma / 3, aspect, 1.0]],
                            dtype=torch.float32)


def field_from_tail(path: Path, n_tail=10):
    d = np.load(path)
    return d["T"][-n_tail:].astype(np.float64).mean(0), float(np.median(d["t_amb"][-n_tail:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="skip PINN, OOD sweeps and the from-scratch ablation")
    args = ap.parse_args()
    torch.manual_seed(0)

    main_model, ckpt = load_checkpoint(ROOT / "models/v2/pretrain_v2.pt")
    twin_model, _ = load_checkpoint(ROOT / "models/v2/pretrain_v2_nopde.pt")
    main_model.eval()
    twin_model.eval()
    cards = {"main": Card(ROOT / "models/boards/pi4b_s1.pt"),
             "twin": Card(ROOT / "models/boards/pi4b_s1_nopde.pt")}
    sites = json.loads((CFG / "sensors_board.json").read_text("utf-8"))["sites"]
    rects = json.loads((CFG / "source_mask_board.json").read_text("utf-8"))["rects"]
    patches = json.loads((CFG / "validation_patches.json").read_text("utf-8"))
    p_s1 = [tuple(rc) for rc in patches["s1"]["patches_rc"]]

    scfg = ScenarioConfig(**{k: tuple(v) if isinstance(v, list) else v
                             for k, v in ckpt["scenario_config"].items()})
    print("generating held-out synthetic boards ...")
    val_boards = generate_boards(32, scfg, seed=999_983)

    results: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "quick": args.quick}

    # ---------------- synthetic ----------------
    print("synthetic K-curves (ours/twin ± ctx, RBF, GP) ...")
    trials = 2 if args.quick else 3
    results["synthetic_kcurves"] = P.synthetic_kcurves(
        main_model, twin_model, val_boards, trials=trials)

    if not args.quick:
        print("amplitude OOD ...")
        results["amplitude_ood"] = P.amplitude_ood(main_model, val_boards)
        print("layout OOD (6-7 sources) ...")
        ood_boards = generate_boards(24, replace(scfg, n_sources=(6, 7)), seed=777)
        results["layout_ood"] = P.layout_ood(main_model, ood_boards, val_boards)

    # ---------------- real ----------------
    c01 = load_case(PROC, "case01_idle")
    c02 = load_case(PROC, "case02_full_load")
    c04 = load_case(PROC, "case04_half_load")
    c06 = load_case(PROC, "case06_full_load_20min")
    tail7, amb7 = field_from_tail(PROC / "case07_cooldown_full_20min.npz")

    scratch = None
    if not args.quick:
        print("training the no-pretrain ablation card (random backbone) ...")
        rnd = ThermalOperatorV2(ModelConfig(**ckpt["model_config"])).eval()
        card_s, _ = finetune(rnd, [c01, c04], sites, rects, p_s1,
                             FinetuneConfig(steps=600, lambda_pde=0.0),
                             session="s1", verbose=False)
        scratch = (rnd, card_s)

    print("real headline: case02 (s1) ...")
    budget = 10.0 if args.quick else 60.0
    m02, _ = P.real_headline(main_model, twin_model, cards, c02, "s1", sites, rects,
                             c01.field, c01.amb, pinn_budget=budget, scratch=scratch)
    print("real headline: case06 (s2, cross-session) ...")
    m06, _ = P.real_headline(main_model, twin_model, cards, c06, "s2", sites, rects,
                             tail7, amb7, pinn_budget=budget, scratch=scratch)
    results["real_headline"] = {"case02_full_load": m02, "case06_full_load_20min": m06}

    print("real K-subset curve (case02) ...")
    results["real_k_subsets"] = P.real_k_subsets(main_model, cards["main"], c02, sites)

    print("leave-one-site-out ...")
    results["loso"] = {"case02_full_load": P.loso_sites(main_model, cards["main"],
                                                        c02, sites, "s1"),
                       "case06_full_load_20min": P.loso_sites(main_model, cards["main"],
                                                              c06, sites, "s2")}

    print("quasi-static cooling check (case07) ...")
    trust06 = np.load(PROC / "trust_case06_full_load_20min.npz")["trusted"]
    results["quasi_static"] = P.quasi_static(main_model, cards["main"],
                                             PROC / "case07_cooldown_full_20min.npz",
                                             sites, trust06)

    print("diode consistency (case06, SoC held out of inputs) ...")
    results["diode"] = P.diode_consistency(main_model, cards["main"],
                                           PROC / "case06_full_load_20min.npz", sites,
                                           ROOT / "实验数据（黑胶带版）/tlog1.csv",
                                           PROC / "soc_log.json")

    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (RES / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    make_figures(results)
    write_md(results)
    print(f"\n-> {RES / 'results.json'}, {RES / 'RESULTS.md'}, reports/figures/fig_eval_*.png")


# ---------------------------------------------------------------- outputs
def make_figures(r: dict) -> None:
    # headline bars
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    for ax, (cid, mm) in zip(axes, r["real_headline"].items()):
        order = sorted(mm, key=lambda m: mm[m]["rmse_trust"], reverse=True)
        y = np.arange(len(order))
        for i, m in enumerate(order):
            v = mm[m]["rmse_trust"]
            ax.barh(i, v, height=0.62, color=STYLE[m][0])
            ax.text(v + 0.06, i, f"{v:.2f}", va="center", fontsize=9, color="#334155")
        ax.set_yticks(y, [STYLE[m][1] for m in order], fontsize=9)
        ax.set_xlabel("trusted-pixel RMSE (°C)")
        ax.set_title(cid, fontsize=11)
        ax.grid(axis="x", **GRID_KW)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Untouched full-load tests — identical sensors for every method", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eval_headline.png", dpi=130)
    plt.close(fig)

    # synthetic k-curves
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    kc = r["synthetic_kcurves"]
    for m in ("ours_ctx", "ours_noctx", "twin_ctx", "twin_noctx", "rbf", "gp"):
        ks = sorted(kc[m], key=int)
        ax.plot([int(k) for k in ks], [kc[m][k] for k in ks], "-o", ms=4, lw=2,
                color=STYLE[m][0], label=STYLE[m][1])
    ax.set_xlabel("number of sensors K")
    ax.set_ylabel("RMSE (θ′ units)")
    ax.set_title("Held-out synthetic boards")
    ax.grid(**GRID_KW)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eval_kcurves_synth.png", dpi=130)
    plt.close(fig)

    # real K-subsets
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for m, d in r["real_k_subsets"].items():
        ks = sorted(d, key=int)
        ax.plot([int(k) for k in ks], [d[k] for k in ks], "-o", ms=4, lw=2,
                color=STYLE[m][0], label=STYLE[m][1])
    ax.set_xlabel("sensors used (of 6 sites)")
    ax.set_ylabel("trusted RMSE (°C)")
    ax.set_title("case02: subsets of the real sensor sites")
    ax.grid(**GRID_KW)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eval_ksubsets_real.png", dpi=130)
    plt.close(fig)

    # LOSO
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (cid, d) in zip(axes, r["loso"].items()):
        names = list(d)
        x = np.arange(len(names))
        ax.bar(x - 0.18, [abs(d[s]["ours_card"]) for s in names], width=0.36,
               color=STYLE["ours_card"][0], label=STYLE["ours_card"][1])
        ax.bar(x + 0.18, [abs(d[s]["rbf"]) for s in names], width=0.36,
               color=STYLE["rbf"][0], label=STYLE["rbf"][1])
        ax.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("|error| at held-out site (°C)")
        ax.set_title(f"leave-one-site-out — {cid}", fontsize=10)
        ax.grid(axis="y", **GRID_KW)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eval_loso.png", dpi=130)
    plt.close(fig)

    # quasi-static
    q = r["quasi_static"]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True,
                             height_ratios=[2, 1])
    axes[0].plot(q["t_min"], q["rmse"], "-o", ms=3, lw=2, color=STYLE["tier1"][0],
                 label="trusted RMSE")
    axes[0].plot(q["t_min"], np.abs(q["max_err"]), "-s", ms=3, lw=2,
                 color=STYLE["rbf"][0], label="|max-T error|")
    axes[0].set_ylabel("error (°C)")
    axes[0].set_title("Steady model applied along the cool-down (case07)")
    axes[0].grid(**GRID_KW)
    axes[0].legend(fontsize=8)
    axes[1].plot(q["t_min"], q["mean_excess"], "-", lw=2, color="#64748b")
    axes[1].set_ylabel("mean excess (°C)")
    axes[1].set_xlabel("t since load stop (min)")
    axes[1].grid(**GRID_KW)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eval_quasistatic.png", dpi=130)
    plt.close(fig)

    # diode consistency
    d = r["diode"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(d["t_min"], d["tape_soc"], "-", lw=2, color="#64748b",
            label="IR on SoC tape (measured)")
    ax.plot(d["t_min"], d["pred_soc"], "-", lw=2, color=STYLE["tier1"][0],
            label="ours: SoC predicted from the OTHER sites")
    ax.plot(d["t_min"], d["diode_shifted"], "--", lw=2, color=STYLE["gp"][0],
            label=f"on-die diode ({d['diode_offset_c']:+.1f} °C offset)")
    ax.set_xlabel("t since case06 start (min)")
    ax.set_ylabel("T (°C)")
    ax.set_title("Three views of the hotspot — sensor withheld from the model")
    ax.grid(**GRID_KW)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eval_diode.png", dpi=130)
    plt.close(fig)

    # amplitude OOD
    if "amplitude_ood" in r:
        a = r["amplitude_ood"]
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        xs = sorted(a, key=float)
        ax.plot([float(x) for x in xs], [a[x] for x in xs], "-o", ms=5, lw=2,
                color=STYLE["ours_ctx"][0])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("source-amplitude scale (training max = 1×)")
        ax.set_ylabel("RMSE (θ′ units)")
        ax.set_title("Amplitude extrapolation — flat = the contract works")
        ax.grid(**GRID_KW)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / "fig_eval_amplitude.png", dpi=130)
        plt.close(fig)


def write_md(r: dict) -> None:
    lines = ["# Evaluation results (auto-generated by scripts/run_eval.py)",
             f"\nGenerated {r['generated_at']} · quick={r['quick']}\n",
             "## Untouched full-load tests (trusted-pixel RMSE °C · max-T err · hotspot px)\n",
             "| method | case02 (s1) | case06 (s2, cross-session) |", "|---|---|---|"]
    hm = r["real_headline"]
    methods = list(hm["case02_full_load"])
    for m in methods:
        cells = []
        for cid in hm:
            v = hm[cid][m]
            cells.append(f"{v['rmse_trust']:.2f} · {v['max_T_err']:+.1f} · "
                         f"{v['hotspot_px']:.0f}")
        lines.append(f"| {STYLE[m][1]} | {cells[0]} | {cells[1]} |")

    kc = r["synthetic_kcurves"]
    lines += ["\n## Synthetic K-curves (RMSE θ′)\n",
              "| K | " + " | ".join(STYLE[m][1] for m in kc) + " |",
              "|---|" + "---|" * len(kc)]
    for k in sorted(next(iter(kc.values())), key=int):
        lines.append(f"| {k} | " + " | ".join(f"{kc[m][k]:.3f}" for m in kc) + " |")

    if "amplitude_ood" in r:
        a = r["amplitude_ood"]
        lines += ["\n## Amplitude OOD (ours + context, RMSE θ′)\n",
                  "| scale | " + " | ".join(sorted(a, key=float)) + " |",
                  "|---|" + "---|" * len(a),
                  "| RMSE | " + " | ".join(f"{a[x]:.3f}" for x in sorted(a, key=float)) + " |"]
    if "layout_ood" in r:
        lines += [f"\n## Layout OOD: {json.dumps(r['layout_ood'])}\n"]

    d = r["diode"]
    lines += [f"\n## Diode consistency: junction−tape offset {d['diode_offset_c']:+.1f} °C; "
              "SoC predicted from the other sites tracks both instruments "
              "(fig_eval_diode.png).\n",
              "\nDiscipline: identical seeded sensors for all methods; metrics on "
              "trusted pixels; model selection was on validation patches only; the "
              "full-load cases were never trained on (seen once during development "
              "before this frozen recipe — stated, not hidden).\n"]
    (RES / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
