"""M2 physics extraction from the processed real dataset.

Produces, from data/processed/real/*.npz:
  - steady-state segments + canonical mean fields + per-pixel repeatability maps
    (canonical_<case>.npz) for the idle / half-load / full-load cases,
  - cooling time constants τ from the two cool-down cases,
  - the spatial decay length L_d of the steady excess field → measured ĥ = (L/L_d)²,
  - the black-tape emissivity systematic (ε 0.95 vs camera setting 0.93),
  - physics.json + diagnostic figures under reports/qc/.

Usage:  uv run scripts/analyze_physics.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tdp.data.cooling import (
    fit_cooling,
    fit_cooling_offset,
    fit_h_hat_pde,
    h_hat_from_decay,
    radial_decay_fit,
)
from tdp.data.radiometry import tape_bias_c
from tdp.data.steady import canonical_steady, detect_steady
from tdp.io.manifest import read_manifest, write_manifest

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "real"
QC = ROOT / "reports" / "qc"

STEADY_CASES = ["case01_idle", "case02_full_load", "case04_half_load",
                "case06_full_load_20min"]
COOLING_CASES = ["case03_cooldown_full", "case05_cooldown_half",
                 "case07_cooldown_full_20min"]
NORM_LENGTH_PX = 157.0  # board long side in px (crop rows); nondimensional length unit


def main() -> int:
    manifest = read_manifest(PROC / "manifest.json")
    data = {c["case_id"]: np.load(PROC / Path(c["npz"]).name) for c in manifest["cases"]}
    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # ---- steady segments + canonical fields ------------------------------------
    fig_s, axes_s = plt.subplots(1, len(STEADY_CASES), figsize=(15, 3.6))
    fig_c, axes_c = plt.subplots(2, len(STEADY_CASES), figsize=(13, 8))
    out["steady"] = {}
    for j, cid in enumerate(STEADY_CASES):
        d = data[cid]
        t, mean_series, amb = d["t_rel_s"], d["frame_mean"], d["t_amb"]
        excess = mean_series - amb
        steady = detect_steady(t, mean_series)
        fallback = False
        try:
            mean_T, std_T, n_eff, picked = canonical_steady(d["T"], t, steady)
        except ValueError:
            fallback = True  # short/idle runs may never satisfy the strict criterion
            picked = list(range(max(0, len(t) - 4), len(t)))
            sub = d["T"][picked].astype(np.float64)
            mean_T, std_T, n_eff = sub.mean(0), sub.std(0, ddof=1), len(picked)
        t_amb_can = float(np.mean(amb[picked]))
        np.savez_compressed(
            PROC / f"canonical_{cid}.npz",
            mean_T=mean_T.astype(np.float32),
            std_T=std_T.astype(np.float32),
            n_eff=n_eff,
            frame_indices=np.asarray(picked),
            t_amb=t_amb_can,
        )
        out["steady"][cid] = {
            "n_frames": int(len(t)),
            "n_steady": int(steady.sum()),
            "n_eff": int(n_eff),
            "fallback_last4": fallback,
            "steady_start_s": float(t[np.flatnonzero(steady)[0]]) if steady.any() else None,
            "noise_std_median_c": float(np.median(std_T)),
            "noise_std_p95_c": float(np.percentile(std_T, 95)),
            "t_amb_c": t_amb_can,
            "peak_c": float(mean_T.max()),
        }

        ax = axes_s[j]
        ax.plot(t / 60, mean_series, label="mean T")
        if steady.any():
            ax.fill_between(t / 60, mean_series.min(), mean_series.max(),
                            where=steady, alpha=0.25, color="green", label="steady")
        ax.plot(np.asarray(t)[picked] / 60, np.asarray(mean_series)[picked], "r.", ms=6,
                label=f"picked (n={n_eff})")
        ax.set_title(f"{cid}{' [fallback]' if fallback else ''}")
        ax.set_xlabel("t (min)")
        ax.set_ylabel("°C")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        im0 = axes_c[0, j].imshow(mean_T, cmap="hot")
        axes_c[0, j].set_title(f"{cid} canonical mean (n_eff={n_eff})")
        fig_c.colorbar(im0, ax=axes_c[0, j], fraction=0.046)
        im1 = axes_c[1, j].imshow(std_T, cmap="viridis")
        axes_c[1, j].set_title(f"per-pixel std, median {np.median(std_T):.2f} °C")
        fig_c.colorbar(im1, ax=axes_c[1, j], fraction=0.046)

    fig_s.tight_layout()
    fig_s.savefig(QC / "physics_steady.png", dpi=110)
    fig_c.tight_layout()
    fig_c.savefig(QC / "physics_canonical.png", dpi=110)
    plt.close(fig_s)
    plt.close(fig_c)

    # ---- cooling time constants --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    out["cooling"] = {}
    for cid in COOLING_CASES:
        d = data[cid]
        t = d["t_rel_s"]
        excess = d["frame_mean"] - d["t_amb"]
        fit = fit_cooling_offset(t, excess)
        fit["loglin_tau_s"] = fit_cooling(t, excess)["tau_s"]  # zero-offset diagnostic
        out["cooling"][cid] = fit
        ax.plot(t / 60, excess, ".", ms=3, label=f"{cid}")
        tt = np.linspace(0, t[-1], 200)
        ax.plot(tt / 60, fit["theta_inf_c"] + fit["amp_c"] * np.exp(-tt / fit["tau_s"]),
                "-", lw=1.2,
                label=f"τ={fit['tau_s']:.0f}±{fit['tau_std_s']:.0f}s, "
                      f"θ∞={fit['theta_inf_c']:.1f}°C, R²={fit['r2']:.3f}")
    ax.set_xlabel("t (min)")
    ax.set_ylabel("mean excess (°C)")
    ax.set_title("Cool-down → idle-powered equilibrium: θ(t) = θ∞ + A·e^(−t/τ)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(QC / "physics_cooling.png", dpi=110)
    plt.close(fig)

    taus = [out["cooling"][c]["tau_s"] for c in COOLING_CASES]
    out["cooling"]["tau_spread"] = float((max(taus) - min(taus)) / np.mean(taus))

    # ---- spatial decay length → ĥ_meas -------------------------------------------
    decay_cases = [("case02_full_load", 157.0), ("case04_half_load", 157.0),
                   ("case06_full_load_20min", 178.0)]  # norm = long side in px per session
    fig, axes = plt.subplots(1, len(decay_cases), figsize=(5.5 * len(decay_cases), 4.2))
    out["decay"] = {}
    for j, (cid, norm_px) in enumerate(decay_cases):
        can = np.load(PROC / f"canonical_{cid}.npz")
        theta = can["mean_T"].astype(np.float64) - float(can["t_amb"])
        pde_fit = fit_h_hat_pde(theta, norm_px)
        fit = radial_decay_fit(theta)
        h_hat = h_hat_from_decay(fit["L_d_px"], norm_px)
        out["decay"][cid] = {
            "h_hat_pde": pde_fit["h_hat"],       # primary estimate (source-free ∇̃²θ = ĥθ)
            "pde_fit_r2": pde_fit["r2"],
            "pde_fit_n_px": pde_fit["n_px"],
            "radial_L_d_px": fit["L_d_px"],      # far-field diagnostic; invalid when L_d ≳ board
            "radial_h_hat": h_hat,
            "radial_r2": fit["r2"],
            "peak_rc": fit["peak_rc"],
        }
        r = np.asarray(fit["profile_r_px"])
        th = np.asarray(fit["profile_theta_c"])
        pos = th > 0
        axes[j].plot(r[pos], np.log(th[pos] * np.sqrt(r[pos])), ".", ms=5, label="profile")
        xf = np.asarray(fit["fit_r_px"])
        axes[j].plot(xf, fit["fit_slope"] * xf + fit["fit_intercept"], "-",
                     label=f"far-field fit L_d={fit['L_d_px']:.0f}px (diagnostic)\n"
                           f"PDE fit ĥ={pde_fit['h_hat']:.2f} (R²={pde_fit['r2']:.2f})")
        axes[j].set_xlabel("r from hotspot (px)")
        axes[j].set_ylabel("ln(θ·√r)")
        axes[j].set_title(cid)
        axes[j].legend(fontsize=8)
        axes[j].grid(alpha=0.3)
    fig.suptitle("Fin-equation far field: θ ∝ e^(−r/L_d)/√r")
    fig.tight_layout()
    fig.savefig(QC / "physics_decay.png", dpi=110)
    plt.close(fig)

    # ---- tape emissivity systematic ----------------------------------------------
    out["tape_bias"] = {
        f"{t}C": round(tape_bias_c(t), 3) for t in (33.0, 48.0, 64.0, 79.0)
    }
    out["conclusion"] = {
        "h_hat_meas": None,
        "h_hat_upper_bound": 3.0,
        "recommended_pretrain_logU_range": [0.3, 30.0],
        "rationale": (
            "The steady excess field is a warm dome filling the whole board: no exponential "
            "far field exists within 103x157 px (radial fit degenerate, L_d >~ board size), "
            "and the pointwise Laplacian signal (~1e-3 °C/px²) is below camera noise/emissivity "
            "texture (PDE fit R²≈0.03). Field flatness bounds ĥ ≲ 3; a sharper estimate needs "
            "the M3 source masks (contour-integral form) or a bigger/cooler board."
        ),
    }
    out["notes"] = [
        "case00 (unplugged) was re-staged: 180° rotation confirmed by ZNCC registration, "
        "but the board appears ~5-8% larger (tripod moved closer) → per-pixel bias map "
        "deferred until a re-measurement with untouched tripod (wishlist item 1).",
        "case02 never fully plateaus (residual drift ~0.2 °C/min at 12.5 min); its canonical "
        "field is a quasi-steady tail average (n_eff=2). case06 (session 2, 20 min) reaches a "
        "true plateau with n_eff=17 and peak 82.0 °C.",
        "case04 per-pixel std (median 0.38 °C) includes slow drift across the 6-12.5 min "
        "steady window, so it is a conservative (upper) repeatability estimate; case01/02 "
        "n_eff=2 stds (~0.07 °C) reflect shot-to-shot noise only.",
        "Cooling decays to the IDLE-POWERED equilibrium (θ∞ ≈ 12-13 °C excess), not ambient "
        "— the board stays on after `killall yes`. The offset model θ∞+A·e^(-t/τ) fits at "
        "R²≈0.999 (zero-offset log fits were model-mismatched: 411/565s were artifacts). "
        "Window-dependent single-exp τ (134-149 s over ~4 min vs 251 s over 25 min) reveals "
        "a two-mode system: fast die/package mode ~140 s, slow PCB tail ~250 s.",
        "Session 2 (2026-07-09): camera ~13% closer (PCB 178x118 px, ~0.478 mm/px), board "
        "mounted 180° vs session 1 (crops rotated back via rot90=2 overrides). case08 foil "
        "sub-cases are radiometric reference scenes — their NPZ crops are meaningless by "
        "design (flagged); reflected-temperature analysis reads the raw CSVs.",
        "Connector shields (USB/HDMI/Ethernet) read near-ambient in all load cases while "
        "surrounded by 50-60 °C board — textbook emissivity failure, key story figure.",
    ]

    write_manifest(PROC / "physics.json", out)

    # ---- summary + M2 pass criteria ------------------------------------------------
    print("=== M2 physics summary ===")
    for cid, s in out["steady"].items():
        print(f"{cid}: n_steady={s['n_steady']}/{s['n_frames']} n_eff={s['n_eff']}"
              f"{' FALLBACK' if s['fallback_last4'] else ''}, peak {s['peak_c']:.1f}°C, "
              f"noise σ median {s['noise_std_median_c']:.3f}°C p95 {s['noise_std_p95_c']:.3f}°C")
    for cid in COOLING_CASES:
        f = out["cooling"][cid]
        print(f"{cid}: τ={f['tau_s']:.0f}±{f['tau_std_s']:.0f}s θ∞={f['theta_inf_c']:.1f}°C "
              f"A={f['amp_c']:.1f}°C R²={f['r2']:.4f} span={f['t_span_s']:.0f}s "
              f"(zero-offset diagnostic τ={f['loglin_tau_s']:.0f}s)")
    print(f"τ spread across cooling cases: {out['cooling']['tau_spread'] * 100:.1f}%")
    for cid, dd in out["decay"].items():
        print(f"{cid}: ĥ_meas(PDE)={dd['h_hat_pde']:.2f} (R²={dd['pde_fit_r2']:.3f}, "
              f"n={dd['pde_fit_n_px']}px) | radial diagnostic L_d={dd['radial_L_d_px']:.0f}px "
              f"(R²={dd['radial_r2']:.2f})")
    print(f"tape bias (ε0.95 vs set 0.93): {out['tape_bias']}")
    print(f"conclusion: ĥ upper bound ≈ {out['conclusion']['h_hat_upper_bound']}, "
          f"pretrain range log-U{out['conclusion']['recommended_pretrain_logU_range']}")
    print(f"-> physics.json + physics_*.png written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
