"""M3: build sensor-site / source-mask configs and per-case trust masks.

Sites come from the tape-layout photo (assets/Taged_Chip.png: large sheet over
SoC/RAM/PMIC region, ALL connector shields tape-wrapped, reference chips and
solder mask exposed) mapped into crop fractional coordinates (u along short
side, v along long side, v↓; session-1 orientation). The SoC site snaps to the
canonical hotspot; other component sites snap to local maxima near photo-derived
priors. Trust masks: interior pixels that are NOT low-emissivity artifacts
(reading near-ambient on a hot board).

Also quantifies the built-in shield A/B: session-1 bare shields vs session-2
taped shields at full load.

Outputs: configs/sensors_board.json, configs/source_mask_board.json,
data/processed/real/trust_<case>.npz, printout of the shield A/B numbers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "real"
CFG = ROOT / "configs"

LOAD_CANONICALS = ["case01_idle", "case02_full_load", "case04_half_load",
                   "case06_full_load_20min"]

# (id, u, v, snap_radius_px, trusted sessions, note)
SITES = [
    ("soc",        0.39, 0.61, 8, ["s1", "s2"], "SoC lid, under tape sheet (hotspot)"),
    ("ram",        0.66, 0.61, 6, ["s1", "s2"], "LPDDR4, under tape sheet"),
    ("usb_ctrl",   0.72, 0.30, 6, ["s1", "s2"], "VL805 area (exposed chip in photo)"),
    ("pmic",       0.15, 0.72, 6, ["s1", "s2"], "power circuitry, under tape edge"),
    ("mid_solder", 0.50, 0.42, 0, ["s1", "s2"], "bare solder mask, mid board"),
    ("corner_cold", 0.10, 0.93, 0, ["s1", "s2"], "PCB corner, coldest trusted region"),
    ("eth_shield", 0.16, 0.05, 0, ["s2"], "Ethernet shield — tape-wrapped in s2 only"),
    ("usb_shield", 0.60, 0.04, 0, ["s2"], "USB stack shield — tape-wrapped in s2 only"),
]

SOURCE_RECTS = [  # (id, u0, v0, u1, v1) fractional, dilate before PDE masking
    ("soc",      0.24, 0.49, 0.55, 0.74),
    ("ram",      0.56, 0.52, 0.78, 0.70),
    ("usb_ctrl", 0.62, 0.22, 0.84, 0.38),
    ("pmic",     0.04, 0.62, 0.24, 0.84),
    ("connector_band", 0.00, 0.00, 1.00, 0.13),  # shields/ports: not 2-D fin material
]


def snap_to_local_max(field: np.ndarray, u: float, v: float, radius_px: int):
    h, w = field.shape
    r, c = int(round(v * h)), int(round(u * w))
    if radius_px > 0:
        r0, r1 = max(0, r - radius_px), min(h, r + radius_px + 1)
        c0, c1 = max(0, c - radius_px), min(w, c + radius_px + 1)
        sub = field[r0:r1, c0:c1]
        dr, dc = np.unravel_index(int(np.argmax(sub)), sub.shape)
        r, c = r0 + dr, c0 + dc
    return r, c


def main():
    can = {cid: np.load(PROC / f"canonical_{cid}.npz") for cid in LOAD_CANONICALS}
    f6 = can["case06_full_load_20min"]["mean_T"].astype(np.float64)  # snap reference
    f2 = can["case02_full_load"]["mean_T"].astype(np.float64)

    sites_out = []
    print(f"{'site':12s} {'u':>6s} {'v':>6s} | T@case02(s1) T@case06(s2)")
    for sid, u, v, rad, sessions, note in SITES:
        r6, c6 = snap_to_local_max(f6, u, v, rad)
        u_f, v_f = (c6 + 0.5) / f6.shape[1], (r6 + 0.5) / f6.shape[0]
        r2, c2 = int(round(v_f * f2.shape[0])), int(round(u_f * f2.shape[1]))
        r2, c2 = min(r2, f2.shape[0] - 2), min(c2, f2.shape[1] - 2)
        t2 = float(np.median(f2[r2 - 1:r2 + 2, c2 - 1:c2 + 2]))
        t6 = float(np.median(f6[r6 - 1:r6 + 2, c6 - 1:c6 + 2]))
        print(f"{sid:12s} {u_f:6.3f} {v_f:6.3f} |   {t2:7.1f}      {t6:7.1f}")
        sites_out.append({"id": sid, "u": round(u_f, 4), "v": round(v_f, 4),
                          "trusted_sessions": sessions, "note": note})

    (CFG / "sensors_board.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coords": "fractional crop coords: u = col/width (short side), v = row/height (long side, v down); session-1 orientation",
        "source": "assets/Taged_Chip.png + canonical-field snapping (SoC=hotspot argmax)",
        "sampling": "3x3 median at the site pixel",
        "sites": sites_out,
    }, indent=2), encoding="utf-8")

    (CFG / "source_mask_board.json").write_text(json.dumps({
        "coords": "fractional crop coords (u0,v0,u1,v1); dilate ~4 px before source-free PDE masking",
        "rects": [{"id": i, "u0": a, "v0": b, "u1": c, "v1": d}
                  for i, a, b, c, d in SOURCE_RECTS],
    }, indent=2), encoding="utf-8")

    # trust masks: interior, excluding low-emissivity artifacts (near-ambient
    # pixels on a hot board). Session 2's taped shields pass automatically.
    for cid in LOAD_CANONICALS:
        d = can[cid]
        field = d["mean_T"].astype(np.float64)
        amb = float(d["t_amb"])
        excess = field - amb
        artifact = excess < 0.15 * excess.max()
        trusted = ~artifact
        trusted[:2] = trusted[-2:] = False
        trusted[:, :2] = trusted[:, -2:] = False
        np.savez_compressed(PROC / f"trust_{cid}.npz", trusted=trusted,
                            artifact_frac=float(artifact.mean()))
        print(f"trust_{cid}: trusted {trusted.mean() * 100:.1f}% of pixels")

    # shield A/B: same fractional top band, bare (s1) vs taped (s2), full load
    band = (0.0, 0.13)
    def band_stats(field, amb):
        h = field.shape[0]
        reg = field[int(band[0] * h):int(band[1] * h)]
        return float(np.median(reg)), float(field.mean()), amb
    s1 = band_stats(f2, float(can["case02_full_load"]["t_amb"]))
    s2 = band_stats(f6, float(can["case06_full_load_20min"]["t_amb"]))
    print("\n=== shield A/B (connector band, full load) ===")
    print(f"s1 bare shields : median {s1[0]:.1f} °C on a board averaging {s1[1]:.1f} °C (amb {s1[2]:.1f})")
    print(f"s2 taped shields: median {s2[0]:.1f} °C on a board averaging {s2[1]:.1f} °C (amb {s2[2]:.1f})")
    print(f"-> the same metal, same load class: apparent jump ≈ {s2[0] - s1[0]:+.1f} °C once ε is fixed")


if __name__ == "__main__":
    main()
