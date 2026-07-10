"""Inference-latency trial (M9): the deployment scenario, timed.

Scenario per inference: K real sensor readings (annotated sites, values from
the canonical half-load field) -> full temperature field of the board.
Grids: 'full' = the native 157x103 crop (16,171 points), 'coarse' = 40x26
(1,040 points — enough to locate a hotspot and trip an alarm).

Runs eager PyTorch (1 thread = Pi-realistic, and all threads) and the
parity-checked TorchScript export; ONNX Runtime if installed (`uv sync
--extra deploy`). The same script runs unmodified on a Raspberry Pi.

Usage:  uv run scripts/bench_latency.py [--iters 50]
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from tdp.deploy.export import DeployedBoardModel, export_torchscript
from tdp.model.normalization import px_to_xy
from tdp.model.operator import load_checkpoint
from tdp.train.finetune import load_case, make_sensor_tensor, read_sites, sites_px_for

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "real"
OUT = ROOT / "reports" / "results"

GRIDS = {"full_157x103": (157, 103), "coarse_40x26": (40, 26)}


def grid_queries(rows: int, cols: int, native=(157, 103)) -> torch.Tensor:
    rr = np.linspace(0, native[0] - 1, rows)
    cc = np.linspace(0, native[1] - 1, cols)
    r, c = np.meshgrid(rr, cc, indexing="ij")
    xy = px_to_xy(r.ravel(), c.ravel(), n_rows=native[0], n_cols=native[1])
    return torch.from_numpy(xy.astype(np.float32))[None]


def bench(fn, iters: int, warmup: int = 8) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean()), "hz_at_p50": float(1000.0 / np.percentile(t, 50))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "models" / "v2" / "pretrain_v2.pt")
    ap.add_argument("--card", type=Path, default=ROOT / "models" / "boards" / "pi4b_s1.pt")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    model, _ = load_checkpoint(args.ckpt)
    model.eval()
    card = torch.load(args.card, map_location="cpu", weights_only=False)
    cond = torch.tensor([[card["log_h"] / 3.0, card["log_gamma"] / 3.0, 103 / 157, 1.0]],
                        dtype=torch.float32)
    deployed = DeployedBoardModel(model, card["tokens"].float(), cond)

    # realistic sensor input: the 6 s1 sites read from the canonical half-load field
    sites = json.loads((ROOT / "configs" / "sensors_board.json").read_text("utf-8"))["sites"]
    case = load_case(PROC, "case04_half_load")
    xy, vals = read_sites(case.field, sites_px_for(sites, "s1", case.shape))
    sensors, dT = make_sensor_tensor(xy, vals, case.amb)
    n_params = sum(p.numel() for p in model.parameters())

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {"machine": platform.machine(), "system": platform.system(),
                 "processor": platform.processor(), "torch": torch.__version__,
                 "default_threads": torch.get_num_threads()},
        "model": {"params": n_params, "K_sensors": int(sensors.shape[1]),
                  "board_tokens": int(card["tokens"].shape[1])},
        "runs": {},
    }
    print(f"host: {platform.system()}/{platform.machine()} · torch {torch.__version__} · "
          f"{n_params / 1e6:.2f}M params · K={sensors.shape[1]} sensors + 8 board tokens")

    ts_dir = ROOT / "models" / "boards"
    for grid_name, (gr, gc) in GRIDS.items():
        q = grid_queries(gr, gc)
        nq = q.shape[1]
        print(f"\n=== grid {grid_name} ({nq} query points) ===")

        variants = {}
        with torch.no_grad():
            for threads in (1, torch.get_num_threads()):
                torch.set_num_threads(threads)
                tag = f"eager_cpu_{threads}thr"
                variants[tag] = bench(lambda: deployed(sensors, q), args.iters)

            ts_path = ts_dir / f"pi4b_s1_{grid_name}.ts"
            export_torchscript(deployed, sensors, q, ts_path)
            traced = torch.jit.load(str(ts_path))
            traced = torch.jit.optimize_for_inference(traced)
            for threads in (1, torch.get_num_threads()):
                torch.set_num_threads(threads)
                variants[f"torchscript_{threads}thr"] = bench(
                    lambda: traced(sensors, q), args.iters)

            try:
                import onnxruntime as ort  # noqa: F401

                onnx_path = ts_dir / f"pi4b_s1_{grid_name}.onnx"
                torch.onnx.export(deployed, (sensors, q), str(onnx_path),
                                  input_names=["sensors", "queries"],
                                  output_names=["theta"], opset_version=17)
                sess = ort.InferenceSession(str(onnx_path),
                                            providers=["CPUExecutionProvider"])
                s_np, q_np = sensors.numpy(), q.numpy()
                ref = deployed(sensors, q).numpy()
                got = sess.run(None, {"sensors": s_np, "queries": q_np})[0]
                assert np.abs(ref - got).max() < 1e-3, "ONNX parity failed"
                variants["onnxruntime"] = bench(
                    lambda: sess.run(None, {"sensors": s_np, "queries": q_np}),
                    args.iters)
            except ImportError:
                print("  (onnxruntime not installed — skipped; `uv sync --extra deploy`)")

            if torch.backends.mps.is_available():
                dev = torch.device("mps")
                d_m = DeployedBoardModel(model.to(dev), card["tokens"].float().to(dev),
                                         cond.to(dev))
                s_m, q_m = sensors.to(dev), q.to(dev)

                def mps_run():
                    d_m(s_m, q_m)
                    torch.mps.synchronize()

                variants["eager_mps"] = bench(mps_run, args.iters)
                model.to("cpu")

        for tag, m in variants.items():
            print(f"  {tag:22s} p50 {m['p50_ms']:8.1f} ms   p95 {m['p95_ms']:8.1f} ms   "
                  f"→ {m['hz_at_p50']:6.1f} Hz")
        results["runs"][grid_name] = variants

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    results["peak_rss_mb"] = round(rss_mb, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\npeak RSS {rss_mb:.0f} MB · results -> reports/results/latency.json")
    print("Raspberry Pi: clone repo on the Pi, `pip install torch numpy pyyaml scipy "
          "matplotlib pillow scikit-learn`, then run this same script (CPU-only).")


if __name__ == "__main__":
    main()
