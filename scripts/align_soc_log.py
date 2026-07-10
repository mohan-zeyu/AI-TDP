"""Align the Pi's on-die temperature log (tlog1.csv: epoch_s,temp_C @ ~2 s) with
the IR frame timelines; check for the 80 °C soft-throttle signature.

Outputs: reports/qc/soc_log_alignment.png + data/processed/real/soc_log.json.
Fine time offset is found by correlating the warm-up derivative of the diode
log against the IR max-temperature curve (robust to clock error ≤ ±5 min).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "real"
QC = ROOT / "reports" / "qc"
TLOG = ROOT / "实验数据（黑胶带版）" / "tlog1.csv"

CASES = ["case06_full_load_20min", "case07_cooldown_full_20min"]


def main():
    raw = np.loadtxt(TLOG, delimiter=",")
    t_log, T_log = raw[:, 0], raw[:, 1]
    print(f"tlog: {len(t_log)} samples, "
          f"{datetime.fromtimestamp(t_log[0])} .. {datetime.fromtimestamp(t_log[-1])} "
          f"(local), T {T_log.min():.1f}..{T_log.max():.1f} °C")

    ir_t, ir_max = [], []
    for cid in CASES:
        d = np.load(PROC / f"{cid}.npz")
        ir_t.append(d["timestamp_epoch"].astype(np.float64))
        ir_max.append(d["frame_max"].astype(np.float64))
    ir_t = np.concatenate(ir_t)
    ir_max = np.concatenate(ir_max)
    order = np.argsort(ir_t)
    ir_t, ir_max = ir_t[order], ir_max[order]

    # The Pi has no RTC — its clock can be arbitrarily wrong, and logging was
    # stopped together with the load, so the log may not cover the cool-down.
    # Align by curve shape against the case06 warm-up/plateau only.
    d6 = np.load(PROC / f"{CASES[0]}.npz")
    t6 = d6["timestamp_epoch"].astype(np.float64)
    m6 = d6["frame_max"].astype(np.float64)
    ir_z = (m6 - m6.mean()) / m6.std()
    best = None
    for off in np.arange(-16000.0, 16001.0, 2.0):
        ts = t_log + off
        if ts[0] > t6[0] or ts[-1] < t6[-1]:
            continue  # log must cover the case06 window
        log_at_ir = np.interp(t6, ts, T_log)
        s = log_at_ir.std()
        if s < 1e-6:
            continue
        c = float(ir_z @ ((log_at_ir - log_at_ir.mean()) / s) / len(t6))
        if best is None or c > best[0]:
            best = (c, off)
    corr, off = best
    print(f"alignment vs case06: offset {off:+.0f} s ({off / 60:+.1f} min Pi clock error), "
          f"series correlation {corr:.3f}")

    sel = (t_log + off >= ir_t[0] - 300) & (t_log + off <= ir_t[-1] + 300)
    tl, Tl = t_log[sel] + off, T_log[sel]

    above80 = float(np.mean(Tl >= 80.0) * 100)
    tmax = float(Tl.max())
    # surface-vs-junction offset over the case06 steady tail
    tail = (tl >= t6[-1] - 300) & (tl <= t6[-1])
    diode_tail = float(np.median(Tl[tail])) if tail.any() else float("nan")
    ir_tail = float(np.median(ir_max[200:240]))
    print(f"diode max {tmax:.1f} °C · {above80:.1f}% of aligned span ≥ 80 °C")
    print(f"steady tail: diode {diode_tail:.1f} vs IR surface max {ir_tail:.1f} "
          f"→ junction−surface ≈ {diode_tail - ir_tail:+.1f} °C")

    t0 = ir_t[0]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot((tl - t0) / 60, Tl, lw=0.9, label="SoC diode (vcgencmd)")
    ax.plot((ir_t - t0) / 60, ir_max, ".", ms=3, label="IR surface max (on tape)")
    ax.axhline(80, color="r", ls="--", lw=0.8, label="80 °C soft-throttle")
    ax.set_xlabel("t since case06 start (min)")
    ax.set_ylabel("T (°C)")
    ax.set_title(f"SoC diode vs IR surface — offset {off:+.0f}s, corr {corr:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    QC.mkdir(parents=True, exist_ok=True)
    fig.savefig(QC / "soc_log_alignment.png", dpi=110)
    plt.close(fig)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tlog": TLOG.name,
        "n_samples": int(len(t_log)),
        "offset_s": float(off),
        "derivative_corr": corr,
        "diode_max_c": tmax,
        "pct_time_above_80c": above80,
        "junction_minus_surface_c": diode_tail - ir_tail,
        "note": "offset applied as t_pi + offset = t_camera_epoch (local-TZ epochs)",
    }
    (PROC / "soc_log.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("-> soc_log.json + soc_log_alignment.png")


if __name__ == "__main__":
    main()
