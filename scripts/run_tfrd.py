"""Run our operator on the public TFR-HSS benchmark (TFRD).

Modes:
  zeroshot — the Pi-pretrained operator, unmodified, on a TFRD test list
  train    — train our architecture on the TFRD train split (fresh or
             --warm-start from the pretrained checkpoint), then evaluate
             every test list; GPU strongly recommended

Data: download TFRD (Baidu Pan 14BipTer1fkilbRjrQNbKiQ, password tfrd) and
point --data-root at one case folder (e.g. data/tfrd/HSink).

Colab:
  !git clone -b organized <repo> && cd <repo> && pip install -e .
  !python scripts/run_tfrd.py --data-root /content/drive/.../HSink --mode train
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tdp.eval.tfrd import TFRDDataset, evaluate, read_list
from tdp.model.operator import ModelConfig, ThermalOperatorV2, load_checkpoint, save_checkpoint
from tdp.train.pretrain import collate, pick_device

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--mode", choices=("zeroshot", "train"), default="zeroshot")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "models/v2/pretrain_v2.pt")
    ap.add_argument("--warm-start", action="store_true",
                    help="init TFRD training from --ckpt instead of random")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-list", default="train/train_val.txt")
    ap.add_argument("--test-lists", nargs="*",
                    default=[f"test/test_{i}.txt" for i in range(6)])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap samples per test list (quick runs)")
    ap.add_argument("--train-limit", type=int, default=None,
                    help="train on only N samples (seeded subsample; the "
                         "validation split stays fixed for comparability)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    case = args.data_root.name
    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "case": case, "mode": args.mode, "warm_start": args.warm_start,
                 "tests": {}}

    if args.mode == "zeroshot":
        model, _ = load_checkpoint(args.ckpt)
        model = model.to(device)
        tag = f"tfrd_{case}_zeroshot"
    else:
        files = read_list(args.data_root, args.train_list)
        n_val = min(max(len(files) // 20, 10), 200)
        n_val = min(n_val, max(len(files) - 1, 1))  # never empty the train pool
        pool, val_files = files[:-n_val], files[-n_val:]  # val fixed across sizes
        if args.train_limit is not None and args.train_limit < len(pool):
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(pool), args.train_limit, replace=False)
            train_files = [pool[i] for i in sorted(idx)]
        else:
            train_files = pool
        print(f"TFRD {case}: {len(train_files)} train / {len(val_files)} val samples"
              + (f" (train-limit {args.train_limit}, seed {args.seed})"
                 if args.train_limit else ""))

        if args.warm_start:
            model, _ = load_checkpoint(args.ckpt)
        else:
            _, ckpt = load_checkpoint(args.ckpt)  # reuse architecture config only
            model = ThermalOperatorV2(ModelConfig(**ckpt["model_config"]))
        model = model.to(device)

        loader = DataLoader(TFRDDataset(train_files), batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate, num_workers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = float("inf")
        tag = (f"tfrd_{case}" + ("_warm" if args.warm_start else "")
               + (f"_n{args.train_limit}" if args.train_limit else ""))
        ckpt_path = ROOT / "models/v2" / f"{tag}.pt"
        for ep in range(args.epochs):
            model.train()
            tot, nb = 0.0, 0
            for sensors, s_mask, q_xy, q_theta, *_rest, cond, cond_drop, _h, ctx, cs, cm in loader:
                pred = model(sensors.to(device), q_xy.to(device), None,
                             sensor_mask=s_mask.to(device))
                loss = F.mse_loss(pred, q_theta.to(device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
                nb += 1
            sched.step()
            val = evaluate(model, val_files, device=device, limit=50)
            print(f"ep {ep:3d}  train {tot / nb:.5f}  val MAE {val['mae_K']:.3f} K")
            if val["mae_K"] < best:
                best = val["mae_K"]
                save_checkpoint(ckpt_path, model.cpu(), extra={"tfrd_case": case,
                                                               "val_mae_K": best})
                model.to(device)
        print(f"best val MAE {best:.3f} K -> {ckpt_path}")
        model, _ = load_checkpoint(ckpt_path)
        model = model.to(device)

    for tl in args.test_lists:
        try:
            files = read_list(args.data_root, tl)
        except FileNotFoundError:
            continue
        m = evaluate(model, files, device=device, limit=args.limit)
        out["tests"][tl] = m
        print(f"{tl}: MAE {m['mae_K']:.3f} K · max-AE {m['max_ae_K']:.2f} K · "
              f"hotspot MAE {m['hotspot_mae_K']:.3f} K  (n={m['n_samples']})")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"-> reports/results/{tag}.json")


if __name__ == "__main__":
    main()
