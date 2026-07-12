"""Live thermal self-monitoring loop (M9) — run ON the Raspberry Pi.

Each cycle: read the SoC diode (`vcgencmd measure_temp`), convert to a surface
estimate via the measured junction-surface offset, feed it as the single live
sensor (K=1; the board card carries the layout), reconstruct a coarse field,
report max temperature + hotspot location, raise an alarm above the threshold.

On a dev machine use --simulate <tlog.csv> to replay a recorded diode log.

Usage (Pi):  python -m tdp.deploy.monitor_pi --threshold 75
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tdp.model.normalization import BOARD_ASPECT
from tdp.model.operator import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
SOC_UV = (0.394, 0.660)          # fractional SoC site (configs/sensors_board.json)
JUNCTION_MINUS_SURFACE = -1.0    # measured (data/processed/real/soc_log.json)
COARSE = (40, 26)


def read_diode() -> float:
    out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True)
    return float(out.stdout.split("=")[1].split("'")[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "models/v2/pretrain_v2.pt")
    ap.add_argument("--card", type=Path, default=ROOT / "models/boards/pi4b_s1.pt")
    ap.add_argument("--ambient", type=float, default=26.5)
    ap.add_argument("--threshold", type=float, default=75.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--simulate", type=Path, default=None,
                    help="replay a tlog.csv instead of vcgencmd")
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(2)
    model, _ = load_checkpoint(args.ckpt)
    model.eval()
    card = torch.load(args.card, map_location="cpu", weights_only=False)
    tokens = card["tokens"].float()
    cond = torch.tensor([[card["log_h"] / 3, card["log_gamma"] / 3, BOARD_ASPECT, 1.0]])

    gr, gc = COARSE
    u = np.linspace(0.01, 0.99, gc) * BOARD_ASPECT
    v = np.linspace(0.01, 0.99, gr)
    uu, vv = np.meshgrid(u, v)
    q = torch.from_numpy(np.stack([uu.ravel(), vv.ravel()], -1).astype(np.float32))[None]

    replay = None
    if args.simulate:
        replay = iter(np.loadtxt(args.simulate, delimiter=",")[:, 1])
    log_f = open(args.log, "a") if args.log else None

    print(f"monitoring: K=1 diode sensor -> {gr}x{gc} field · alarm > {args.threshold} °C")
    try:
        while True:
            t0 = time.time()
            diode = next(replay) if replay is not None else read_diode()
            surface = diode - JUNCTION_MINUS_SURFACE  # junction−surface=−1 ⇒ +1.0
            theta_s = max(surface - args.ambient, 1.0)
            sensors = torch.tensor([[[SOC_UV[0] * BOARD_ASPECT, SOC_UV[1], 1.0]]])
            with torch.no_grad():
                theta = model(sensors, q, cond, board_tokens=tokens)[0].numpy()
            field = theta.reshape(gr, gc) * theta_s + args.ambient
            i, j = np.unravel_index(int(field.argmax()), field.shape)
            tmax = float(field.max())
            ms = (time.time() - t0) * 1000
            flag = "  ⚠ ALARM" if tmax > args.threshold else ""
            line = (f"diode {diode:5.1f} °C · field max {tmax:5.1f} °C at "
                    f"({i:2d},{j:2d}) · {ms:5.1f} ms{flag}")
            print(line)
            if log_f:
                log_f.write(f"{time.time():.1f},{diode:.1f},{tmax:.1f},{i},{j}\n")
                log_f.flush()
            time.sleep(max(0.0, args.interval - (time.time() - t0)))
    except (KeyboardInterrupt, StopIteration):
        print("stopped.")
    finally:
        if log_f:
            log_f.close()


if __name__ == "__main__":
    main()
