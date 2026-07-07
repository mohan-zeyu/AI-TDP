"""Pretrain ThermalOperatorV2 on domain-randomized synthetic scenarios (M5).

Colab-compatible single command:
    uv run scripts/run_pretrain.py [--small] [--twin-nopde] [--device auto]
Colab:
    !pip install -e . && python scripts/run_pretrain.py

--small runs the ~minutes smoke configuration (gates the code before GPU time);
--twin-nopde trains the λ_pde=0 ablation twin from the same seed and data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tdp.model.operator import ModelConfig
from tdp.sim.scenarios import ScenarioConfig
from tdp.train.pretrain import TrainConfig, train

ROOT = Path(__file__).resolve().parents[1]

SMALL_OVERRIDES = {"n_train": 192, "n_val": 32, "epochs": 12, "batch_size": 16,
                   "n_query": 256, "n_colloc": 128, "pde_warmup_epochs": 2,
                   "pde_ramp_epochs": 4}


def _tupled(d: dict) -> dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pretrain_v2.yaml")
    ap.add_argument("--small", action="store_true", help="minutes-scale smoke config")
    ap.add_argument("--twin-nopde", action="store_true", help="train the λ=0 ablation twin")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models" / "v2")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scfg = ScenarioConfig(**_tupled(cfg["scenario"]))
    tdict = dict(cfg["train"])
    if args.small:
        tdict.update(SMALL_OVERRIDES)
        if args.small and cfg["scenario"].get("ny", 96) > 64:
            scfg.ny = 64
    if args.epochs is not None:
        tdict["epochs"] = args.epochs
    if args.seed is not None:
        tdict["seed"] = args.seed
    tcfg = TrainConfig(**tdict)
    if args.twin_nopde:
        tcfg.lambda_pde_max = 0.0

    tag = "pretrain_v2" + ("_nopde" if args.twin_nopde else "") + ("_small" if args.small else "")
    mcfg = ModelConfig(**cfg["model"])
    train(mcfg, scfg, tcfg, device_str=args.device, out_dir=args.out_dir, tag=tag)


if __name__ == "__main__":
    main()
