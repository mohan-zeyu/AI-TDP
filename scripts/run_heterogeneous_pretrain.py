"""Continue the full TDP backbone on heterogeneous-material simulations.

This stage teaches the existing 1.06M-parameter operator that regular package
regions can have conductivity unlike the surrounding PCB.  It deliberately
mixes heterogeneous and homogeneous boards and warm-starts from pretrain_v2.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tdp.model.operator import ModelConfig
from tdp.sim.scenarios import ScenarioConfig
from tdp.train.pretrain import TrainConfig, train


ROOT = Path(__file__).resolve().parents[1]


def _tupled(values: dict) -> dict:
    return {key: tuple(value) if isinstance(value, list) else value
            for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pretrain_heterogeneous.yaml",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=ROOT / "models" / "v2" / "pretrain_v2.pt",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-boards", type=int)
    parser.add_argument("--val-boards", type=int)
    parser.add_argument("--tag", default="pretrain_v3_heterogeneous")
    parser.add_argument(
        "--source-adapter-only", action="store_true",
        help="freeze the migrated backbone and train only SourceGeometryAdapter",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models" / "v3")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scenario = ScenarioConfig(**_tupled(raw["scenario"]))
    train_values = _tupled(raw["train"])
    if args.small:
        scenario.ny = 48
        train_values.update({
            "n_train": 48,
            "n_val": 12,
            "epochs": 2,
            "batch_size": 2,
            "n_query": 96,
            "n_colloc": 32,
            "n_boundary": 16,
            "n_interface": 8,
            "n_context": (24, 48),
            "pde_warmup_epochs": 0,
            "pde_ramp_epochs": 1,
        })
    if args.epochs is not None:
        train_values["epochs"] = args.epochs
    if args.batch_size is not None:
        train_values["batch_size"] = args.batch_size
    if args.train_boards is not None:
        train_values["n_train"] = args.train_boards
    if args.val_boards is not None:
        train_values["n_val"] = args.val_boards
    if args.source_adapter_only:
        train_values["source_adapter_only"] = True

    checkpoint = train(
        ModelConfig(**raw["model"]),
        scenario,
        TrainConfig(**train_values),
        device_str=args.device,
        out_dir=args.out_dir,
        tag=args.tag + ("_small" if args.small else ""),
        init_checkpoint=args.init_checkpoint,
    )
    print(f"heterogeneous checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
