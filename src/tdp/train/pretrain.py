"""Domain-randomized pretraining of ThermalOperatorV2 (M5).

Every training item is: one randomized scenario → sampled sensor set (random-K
or jittered board layout, noisy) → per-sample scale s = max sensor θ → the
model regresses θ/s at query points, PDE-regularized at collocation points.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tdp.model.normalization import cond_vector
from tdp.model.operator import ModelConfig, ThermalOperatorV2, save_checkpoint
from tdp.sim.fdm import bilinear
from tdp.sim.scenarios import (
    PLACEHOLDER_BOARD_LAYOUT,
    Scenario,
    ScenarioConfig,
    generate_dataset,
    sample_sensors,
)
from tdp.train.losses import pde_loss, pde_residual, probe_double_backward


@dataclass
class TrainConfig:
    n_train: int = 4096
    n_val: int = 128
    epochs: int = 250
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    k_min: int = 4
    k_max: int = 16
    n_query: int = 384
    n_colloc: int = 192
    colloc_margin: float = 0.02
    lambda_pde_max: float = 1e-3
    pde_warmup_epochs: int = 10
    pde_ramp_epochs: int = 20
    layout_prob: float = 0.2
    noise_theta_max: float = 0.10
    board_aspect_frac: float = 0.25
    seed: int = 0


class OperatorDataset(Dataset):
    """Stochastic view over scenarios: fresh sensors/queries/collocation each epoch."""

    def __init__(self, scenarios: list[Scenario], tcfg: TrainConfig,
                 layout: np.ndarray | None, base_seed: int):
        self.scenarios = scenarios
        self.tcfg = tcfg
        self.layout = layout
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, ep: int) -> None:
        self.epoch = ep

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, i: int):
        t = self.tcfg
        scn = self.scenarios[i]
        rng = np.random.default_rng((self.base_seed, self.epoch, i))

        s_xy, s_val = sample_sensors(
            scn, rng, (t.k_min, t.k_max), layout=self.layout,
            layout_prob=t.layout_prob, noise_theta_max=t.noise_theta_max,
        )
        scale = float(max(s_val.max(), 1e-6))
        sensors = np.concatenate([s_xy, (s_val / scale)[:, None]], axis=-1)

        q_xy = np.stack([rng.uniform(0, scn.aspect, t.n_query),
                         rng.uniform(0, 1, t.n_query)], axis=-1)
        q_theta = bilinear(scn.theta, q_xy, scn.aspect) / scale

        m = t.colloc_margin
        c_xy = np.stack([rng.uniform(m * scn.aspect, (1 - m) * scn.aspect, t.n_colloc),
                         rng.uniform(m, 1 - m, t.n_colloc)], axis=-1)
        q_over_s = scn.q_at(c_xy) / scale

        cond = cond_vector(scn.h_hat, scn.gamma, scn.aspect, scn.robin)
        return (
            torch.from_numpy(sensors.astype(np.float32)),
            torch.from_numpy(q_xy.astype(np.float32)),
            torch.from_numpy(q_theta.astype(np.float32)),
            torch.from_numpy(c_xy.astype(np.float32)),
            torch.from_numpy(q_over_s.astype(np.float32)),
            torch.from_numpy(cond),
            torch.tensor(scn.h_hat, dtype=torch.float32),
        )


def collate(batch):
    max_k = max(item[0].shape[0] for item in batch)
    B = len(batch)
    sensors = torch.zeros(B, max_k, 3)
    mask = torch.ones(B, max_k, dtype=torch.bool)  # True = padding
    for i, item in enumerate(batch):
        k = item[0].shape[0]
        sensors[i, :k] = item[0]
        mask[i, :k] = False
    stack = [torch.stack([item[j] for item in batch]) for j in range(1, 7)]
    q_xy, q_theta, c_xy, q_over_s, cond, h_hat = stack
    return sensors, mask, q_xy, q_theta, c_xy, q_over_s, cond, h_hat


def lambda_schedule(epoch: int, t: TrainConfig) -> float:
    if epoch < t.pde_warmup_epochs:
        return 0.0
    if epoch < t.pde_warmup_epochs + t.pde_ramp_epochs:
        return t.lambda_pde_max * (epoch - t.pde_warmup_epochs) / t.pde_ramp_epochs
    return t.lambda_pde_max


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    se, n = 0.0, 0
    for sensors, mask, q_xy, q_theta, *_rest, cond, _h in loader:
        pred = model(sensors.to(device), q_xy.to(device), cond.to(device),
                     sensor_mask=mask.to(device))
        se += F.mse_loss(pred, q_theta.to(device), reduction="sum").item()
        n += q_theta.numel()
    return (se / n) ** 0.5


def k_scaling_curve(model, scenarios, device, ks=(4, 6, 8, 12, 16),
                    n_scen=16, trials=4, seed=1234) -> dict[int, float]:
    """RMSE (θ' units) vs sensor count on held-out scenarios."""
    model.eval()
    out = {}
    for k in ks:
        errs = []
        for si, scn in enumerate(scenarios[:n_scen]):
            for tr in range(trials):
                rng = np.random.default_rng((seed, k, si, tr))
                xy, val = sample_sensors(scn, rng, (k, k), layout=None,
                                         layout_prob=0.0, noise_theta_max=0.0)
                scale = float(max(val.max(), 1e-6))
                sensors = torch.from_numpy(
                    np.concatenate([xy, (val / scale)[:, None]], -1).astype(np.float32)
                )[None].to(device)
                ny, nx = scn.theta.shape
                q_xy = np.stack(
                    [g.ravel() for g in np.meshgrid(
                        np.linspace(0, scn.aspect, nx), np.linspace(0, 1, ny))],
                    axis=-1,
                )[::4]
                cond = torch.from_numpy(
                    cond_vector(scn.h_hat, scn.gamma, scn.aspect, scn.robin))[None].to(device)
                with torch.no_grad():
                    pred = model(sensors, torch.from_numpy(
                        q_xy.astype(np.float32))[None].to(device), cond).cpu().numpy()[0]
                truth = scn.theta.ravel()[::4] / scale
                errs.append(float(np.sqrt(np.mean((pred - truth) ** 2))))
        out[k] = float(np.mean(errs))
    return out


def train(model_cfg: ModelConfig, scfg: ScenarioConfig, tcfg: TrainConfig,
          device_str: str = "auto", out_dir: Path | str = "models/v2",
          tag: str = "pretrain_v2", layout: np.ndarray | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(device_str)
    torch.manual_seed(tcfg.seed)

    print(f"generating {tcfg.n_train}+{tcfg.n_val} scenarios (ny={scfg.ny}) ...")
    t0 = time.time()
    train_scen = generate_dataset(tcfg.n_train, scfg, seed=tcfg.seed,
                                  board_aspect_frac=tcfg.board_aspect_frac)
    val_scen = generate_dataset(tcfg.n_val, scfg, seed=tcfg.seed + 999_983,
                                board_aspect_frac=tcfg.board_aspect_frac)
    print(f"  done in {time.time() - t0:.1f}s")

    layout = PLACEHOLDER_BOARD_LAYOUT if layout is None else layout
    train_ds = OperatorDataset(train_scen, tcfg, layout, base_seed=tcfg.seed)
    val_ds = OperatorDataset(val_scen, tcfg, layout, base_seed=tcfg.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)

    model = ThermalOperatorV2(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device.type}  params={n_params / 1e6:.2f}M  tag={tag}")

    if tcfg.lambda_pde_max > 0 and not probe_double_backward(model, device):
        print(f"!! double-backward failed on {device.type} -> falling back to cpu")
        device = torch.device("cpu")
        model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
    hist = {"train_data": [], "train_pde": [], "val_rmse": [], "lambda": []}
    best_val, best_ep = float("inf"), -1
    ckpt_path = out_dir / f"{tag}.pt"

    for ep in range(tcfg.epochs):
        lam = lambda_schedule(ep, tcfg)
        train_ds.set_epoch(ep)
        model.train()
        sd = sp = 0.0
        nb = 0
        for sensors, mask, q_xy, q_theta, c_xy, q_over_s, cond, h_hat in train_loader:
            sensors, mask = sensors.to(device), mask.to(device)
            q_xy, q_theta = q_xy.to(device), q_theta.to(device)
            cond, h_hat = cond.to(device), h_hat.to(device)

            pred = model(sensors, q_xy, cond, sensor_mask=mask)
            data_loss = F.mse_loss(pred, q_theta)
            if lam > 0:
                R = pde_residual(model, sensors, mask, cond,
                                 c_xy.to(device), q_over_s.to(device), h_hat)
                p_loss = pde_loss(R)
                loss = data_loss + lam * p_loss
            else:
                p_loss = torch.tensor(0.0)
                loss = data_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sd += data_loss.item()
            sp += p_loss.detach().item()
            nb += 1
        sched.step()

        val_rmse = evaluate(model, val_loader, device)
        hist["train_data"].append(sd / nb)
        hist["train_pde"].append(sp / nb)
        hist["val_rmse"].append(val_rmse)
        hist["lambda"].append(lam)
        if val_rmse < best_val:
            best_val, best_ep = val_rmse, ep
            save_checkpoint(ckpt_path, model, extra={
                "train_config": asdict(tcfg), "scenario_config": asdict(scfg),
                "epoch": ep, "val_rmse": val_rmse, "tag": tag,
            })
        if ep % 5 == 0 or ep == tcfg.epochs - 1:
            print(f"ep {ep:3d}  data {sd / nb:.5f}  pde {sp / nb:.5f}  "
                  f"val_rmse(θ') {val_rmse:.4f}  λ {lam:.1e}")

    kcurve = k_scaling_curve(model, val_scen, device)
    print("K-scaling RMSE(θ'):", {k: round(v, 4) for k, v in kcurve.items()})
    (out_dir / f"{tag}_history.json").write_text(
        json.dumps({"hist": hist, "k_curve": kcurve, "best_val": best_val,
                    "best_epoch": best_ep, "n_params": n_params}, indent=2),
        encoding="utf-8",
    )
    print(f"best val RMSE {best_val:.4f} @ epoch {best_ep} -> {ckpt_path}")
    return ckpt_path
