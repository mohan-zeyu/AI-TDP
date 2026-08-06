"""Domain-randomized pretraining of ThermalOperatorV2 (M5, v2.5 in-context).

Every training item: one board (layout+physics) → pick a target state → sample
live sensors from it; with probability (1 − context_dropout) also hand the model
1–2 reference frames of *other* states of the same board as context tokens.
The model regresses θ/s at query points, PDE-regularized at collocation points.
Context dropout keeps the no-context (v2) mode functional; condition-token
dropout teaches the model to infer physics from context when the oracle vector
is absent.
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
from tdp.model.operator import ModelConfig, ThermalOperatorV2, load_checkpoint, save_checkpoint
from tdp.sim.fdm import bilinear
from tdp.sim.scenarios import (
    PLACEHOLDER_BOARD_LAYOUT,
    Board,
    ScenarioConfig,
    generate_boards,
    sample_context,
    sample_sensors,
)
from tdp.train.losses import (
    boundary_residual,
    interface_flux_loss,
    pde_loss,
    pde_residual,
    probe_double_backward,
)


@dataclass
class TrainConfig:
    n_train: int = 4096          # boards (each with scenario.states_per_board states)
    n_val: int = 128
    epochs: int = 250
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    k_min: int = 4
    k_max: int = 16
    target_k: int = 3            # deployment sensor count emphasized with context
    target_k_prob: float = 0.0   # default preserves the original uniform-K distribution
    k_min_ctx: int = 2           # with context present, K may be tiny — context must carry
    n_query: int = 384
    n_colloc: int = 192
    n_boundary: int = 64
    n_interface: int = 24
    material_dropout: float = 0.15
    source_dropout: float = 0.05
    source_adapter_only: bool = False
    colloc_margin: float = 0.02
    lambda_pde_max: float = 1e-3
    lambda_boundary_max: float = 0.0
    lambda_interface_max: float = 0.0
    pde_warmup_epochs: int = 10
    pde_ramp_epochs: int = 20
    layout_prob: float = 0.2
    noise_theta_max: float = 0.10
    board_aspect_frac: float = 0.25
    context_dropout: float = 0.2       # p(no context) — keeps v2 mode alive
    cond_dropout: float = 0.3          # p(condition token masked) — infer physics from context
    n_context: tuple = (64, 192)       # points per context frame
    context_two_frames_prob: float = 0.5
    context_self_prob: float = 0.2     # p(context pool may include the target state)
    val_context_weight: float = 0.5    # heterogeneous layouts are unidentifiable without context
    seed: int = 0


class OperatorDataset(Dataset):
    """Stochastic view over boards: fresh target state, sensors, context,
    queries and collocation points every epoch."""

    def __init__(self, boards: list[Board], tcfg: TrainConfig,
                 layout: np.ndarray | None, base_seed: int,
                 force_context: bool | None = None):
        self.boards = boards
        self.tcfg = tcfg
        self.layout = layout
        self.base_seed = base_seed
        self.epoch = 0
        self.force_context = force_context  # True/False overrides context_dropout (val)
        # Val datasets (force_context set) use the same K range in both modes so
        # the ctx / no-ctx RMSE columns are directly comparable.
        self.fair_k = force_context is not None

    def set_epoch(self, ep: int) -> None:
        self.epoch = ep

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, i: int):
        t = self.tcfg
        board = self.boards[i]
        rng = np.random.default_rng((self.base_seed, self.epoch, i))
        target = board.states[int(rng.integers(len(board.states)))]

        if self.force_context is None:
            use_ctx = rng.random() >= t.context_dropout
        else:
            use_ctx = self.force_context

        ctx_pts = np.zeros((0, 3), np.float32)
        ctx_state = np.zeros((0,), np.int64)
        if use_ctx:
            pool = [s for s in board.states if s is not target]
            if not pool or rng.random() < t.context_self_prob:
                pool = list(board.states)
            n_frames = 2 if (rng.random() < t.context_two_frames_prob and len(pool) >= 2) else 1
            picks = rng.choice(len(pool), size=n_frames, replace=False)
            parts, states = [], []
            for f, pi in enumerate(picks):
                m = int(rng.integers(t.n_context[0], t.n_context[1] + 1))
                p, st = sample_context(pool[int(pi)], rng, m, frame_idx=f)
                parts.append(p)
                states.append(st)
            ctx_pts = np.concatenate(parts)
            ctx_state = np.concatenate(states)

        k_lo = t.k_min if self.fair_k else (t.k_min_ctx if use_ctx else t.k_min)
        k_range = (k_lo, t.k_max)
        # Deployment is K=3 whether board information arrives as reference
        # context, explicit material geometry, or a learned Board Card.
        if rng.random() < t.target_k_prob:
            k_range = (t.target_k, t.target_k)
        s_xy, s_val = sample_sensors(
            target, rng, k_range, layout=self.layout,
            layout_prob=t.layout_prob, noise_theta_max=t.noise_theta_max,
        )
        scale = float(max(s_val.max(), 1e-6))
        sensors = np.concatenate([s_xy, (s_val / scale)[:, None]], axis=-1)

        q_xy = np.stack([rng.uniform(0, target.aspect, t.n_query),
                         rng.uniform(0, 1, t.n_query)], axis=-1)
        q_theta = bilinear(target.theta, q_xy, target.aspect) / scale

        m = t.colloc_margin
        c_xy = np.stack([rng.uniform(m * target.aspect, (1 - m) * target.aspect, t.n_colloc),
                         rng.uniform(m, 1 - m, t.n_colloc)], axis=-1)
        q_over_s = target.q_at(c_xy) / scale
        conductivity = target.conductivity_at(c_xy)
        sink_multiplier = target.sink_at(c_xy)

        side = rng.integers(0, 4, size=t.n_boundary)
        along = rng.random(t.n_boundary)
        b_xy = np.empty((t.n_boundary, 2), dtype=np.float32)
        b_normal = np.zeros((t.n_boundary, 2), dtype=np.float32)
        left, right, top, bottom = (side == j for j in range(4))
        b_xy[left] = np.stack([np.zeros(left.sum()), along[left]], axis=-1)
        b_xy[right] = np.stack([
            np.full(right.sum(), target.aspect), along[right]
        ], axis=-1)
        b_xy[top] = np.stack([along[top] * target.aspect, np.zeros(top.sum())], axis=-1)
        b_xy[bottom] = np.stack([
            along[bottom] * target.aspect, np.ones(bottom.sum())
        ], axis=-1)
        b_normal[left, 0], b_normal[right, 0] = -1.0, 1.0
        b_normal[top, 1], b_normal[bottom, 1] = -1.0, 1.0
        b_conductivity = target.conductivity_at(b_xy)
        b_gamma = target.boundary_gamma_at(b_xy, side)
        b_robin = np.full(t.n_boundary, target.robin, dtype=bool)
        materials = target.material_descriptors()
        if len(materials) and t.material_dropout > 0:
            keep = rng.random(len(materials)) >= t.material_dropout
            # Do not systematically erase every complex example; the separate
            # homogeneous replay already covers genuinely geometry-free input.
            if not keep.any() and len(keep) > 1:
                keep[int(rng.integers(len(keep)))] = True
            materials = materials[keep]
        sources = target.source_descriptors()
        if len(sources) > 1 and t.source_dropout > 0:
            keep = rng.random(len(sources)) >= t.source_dropout
            if not keep.any():
                keep[int(rng.integers(len(keep)))] = True
            sources = sources[keep]
        boundaries = target.boundary_descriptors()

        cond = cond_vector(target.h_hat, target.gamma, target.aspect, target.robin)
        cond_drop = bool(rng.random() < t.cond_dropout) and use_ctx  # keep cond when no context

        return (
            torch.from_numpy(sensors.astype(np.float32)),
            torch.from_numpy(q_xy.astype(np.float32)),
            torch.from_numpy(q_theta.astype(np.float32)),
            torch.from_numpy(c_xy.astype(np.float32)),
            torch.from_numpy(q_over_s.astype(np.float32)),
            torch.from_numpy(cond),
            torch.tensor(cond_drop),
            torch.tensor(target.h_hat, dtype=torch.float32),
            torch.from_numpy(conductivity.astype(np.float32)),
            torch.from_numpy(sink_multiplier.astype(np.float32)),
            torch.from_numpy(b_xy),
            torch.from_numpy(b_normal),
            torch.from_numpy(b_conductivity.astype(np.float32)),
            torch.from_numpy(b_gamma.astype(np.float32)),
            torch.from_numpy(b_robin),
            torch.from_numpy(ctx_pts),
            torch.from_numpy(ctx_state),
            torch.from_numpy(materials),
            torch.from_numpy(sources),
            torch.from_numpy(boundaries),
        )


def _pad_stack(items: list[torch.Tensor], width: int, dim_feat: int | None):
    """Pad variable-length (L, F) or (L,) tensors to width; mask True = pad."""
    B = len(items)
    if dim_feat is None:
        out = torch.zeros(B, width, dtype=items[0].dtype if items[0].numel() else torch.long)
    else:
        out = torch.zeros(B, width, dim_feat)
    mask = torch.ones(B, width, dtype=torch.bool)
    for i, it in enumerate(items):
        n = it.shape[0]
        if n:
            out[i, :n] = it
            mask[i, :n] = False
    return out, mask


def collate(batch):
    sensors_l = [b[0] for b in batch]
    max_k = max(s.shape[0] for s in sensors_l)
    sensors, s_mask = _pad_stack(sensors_l, max_k, 3)

    fixed = [torch.stack([b[j] for b in batch]) for j in range(1, 15)]
    (q_xy, q_theta, c_xy, q_over_s, cond, cond_drop, h_hat,
     conductivity, sink_multiplier, b_xy, b_normal, b_conductivity,
     b_gamma, b_robin) = fixed

    ctx_l = [b[15] for b in batch]
    max_m = max(c.shape[0] for c in ctx_l)
    if max_m == 0:
        ctx = ctx_state = ctx_mask = None
    else:
        ctx, ctx_mask = _pad_stack(ctx_l, max_m, 3)
        ctx_state, _ = _pad_stack([b[16] for b in batch], max_m, None)

    material_l = [b[17] for b in batch]
    max_r = max(m.shape[0] for m in material_l)
    materials, material_mask = _pad_stack(material_l, max_r, 10)

    source_l = [b[18] for b in batch]
    max_s = max(s.shape[0] for s in source_l)
    sources, source_mask = _pad_stack(source_l, max_s, 8)

    boundary_l = [b[19] for b in batch]
    max_e = max(e.shape[0] for e in boundary_l)
    boundaries, boundary_mask = _pad_stack(boundary_l, max_e, 8)

    return (sensors, s_mask, q_xy, q_theta, c_xy, q_over_s,
            cond, cond_drop, h_hat, conductivity, sink_multiplier,
            b_xy, b_normal, b_conductivity, b_gamma, b_robin,
            ctx, ctx_state, ctx_mask, materials, material_mask,
            sources, source_mask, boundaries, boundary_mask)



def lambda_schedule(epoch: int, t: TrainConfig) -> float:
    return t.lambda_pde_max * physics_ramp(epoch, t)


def physics_ramp(epoch: int, t: TrainConfig) -> float:
    if epoch < t.pde_warmup_epochs:
        return 0.0
    if epoch < t.pde_warmup_epochs + t.pde_ramp_epochs:
        return (epoch - t.pde_warmup_epochs) / t.pde_ramp_epochs
    return 1.0


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    se, n = 0.0, 0
    for batch in loader:
        (sensors, s_mask, q_xy, q_theta, _c, _q, cond, cond_drop, _h, _k, _sink,
         _bxy, _bn, _bk, _bg, _br, ctx, ctx_state, ctx_mask,
         materials, material_mask, sources, source_mask,
         boundaries, boundary_mask) = _to_device(batch, device)
        pred = model(sensors, q_xy, cond, sensor_mask=s_mask,
                     context=ctx, context_state=ctx_state, context_mask=ctx_mask,
                     cond_mask=cond_drop, materials=materials,
                     material_mask=material_mask, sources=sources,
                     source_mask=source_mask, boundaries=boundaries,
                     boundary_mask=boundary_mask)
        se += F.mse_loss(pred, q_theta, reduction="sum").item()
        n += q_theta.numel()
    return (se / n) ** 0.5


def k_scaling_curve(model, boards, device, ks=(2, 3, 4, 6, 8, 12, 16),
                    n_scen=16, trials=4, seed=1234, with_context=False) -> dict[int, float]:
    """RMSE (θ' units) vs sensor count on held-out boards (first state each)."""
    model.eval()
    out = {}
    for k in ks:
        errs = []
        for si, board in enumerate(boards[:n_scen]):
            scn = board.states[0]
            for tr in range(trials):
                rng = np.random.default_rng((seed, k, si, tr))
                xy, val = sample_sensors(scn, rng, (k, k), layout=None,
                                         layout_prob=0.0, noise_theta_max=0.0)
                scale = float(max(val.max(), 1e-6))
                sensors = torch.from_numpy(
                    np.concatenate([xy, (val / scale)[:, None]], -1).astype(np.float32)
                )[None].to(device)
                ctx = ctx_state = None
                if with_context and len(board.states) > 1:
                    p, st = sample_context(board.states[1], rng, 128, frame_idx=0)
                    ctx = torch.from_numpy(p)[None].to(device)
                    ctx_state = torch.from_numpy(st)[None].to(device)
                ny, nx = scn.theta.shape
                q_xy = np.stack(
                    [g.ravel() for g in np.meshgrid(
                        np.linspace(0, scn.aspect, nx), np.linspace(0, 1, ny))],
                    axis=-1,
                )[::4]
                cond = torch.from_numpy(
                    cond_vector(scn.h_hat, scn.gamma, scn.aspect, scn.robin))[None].to(device)
                materials = torch.from_numpy(scn.material_descriptors())[None].to(device)
                sources = torch.from_numpy(scn.source_descriptors())[None].to(device)
                boundaries = torch.from_numpy(scn.boundary_descriptors())[None].to(device)
                with torch.no_grad():
                    pred = model(sensors, torch.from_numpy(
                        q_xy.astype(np.float32))[None].to(device), cond,
                        context=ctx, context_state=ctx_state,
                        materials=materials, sources=sources,
                        boundaries=boundaries).cpu().numpy()[0]
                truth = scn.theta.ravel()[::4] / scale
                errs.append(float(np.sqrt(np.mean((pred - truth) ** 2))))
        out[k] = float(np.mean(errs))
    return out


def train(model_cfg: ModelConfig, scfg: ScenarioConfig, tcfg: TrainConfig,
          device_str: str = "auto", out_dir: Path | str = "models/v2",
          tag: str = "pretrain_v2", layout: np.ndarray | None = None,
          init_checkpoint: Path | str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(device_str)
    torch.manual_seed(tcfg.seed)

    print(f"generating {tcfg.n_train}+{tcfg.n_val} boards x {scfg.states_per_board} states "
          f"(ny={scfg.ny}) ...")
    t0 = time.time()
    train_boards = generate_boards(tcfg.n_train, scfg, seed=tcfg.seed,
                                   board_aspect_frac=tcfg.board_aspect_frac)
    val_boards = generate_boards(tcfg.n_val, scfg, seed=tcfg.seed + 999_983,
                                 board_aspect_frac=tcfg.board_aspect_frac)
    print(f"  done in {time.time() - t0:.1f}s")

    layout = PLACEHOLDER_BOARD_LAYOUT if layout is None else layout
    train_ds = OperatorDataset(train_boards, tcfg, layout, base_seed=tcfg.seed)
    val_ctx = OperatorDataset(val_boards, tcfg, layout, base_seed=tcfg.seed + 1,
                              force_context=True)
    val_noctx = OperatorDataset(val_boards, tcfg, layout, base_seed=tcfg.seed + 1,
                                force_context=False)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader_ctx = DataLoader(val_ctx, batch_size=tcfg.batch_size, shuffle=False,
                                collate_fn=collate, num_workers=0)
    val_loader_noctx = DataLoader(val_noctx, batch_size=tcfg.batch_size, shuffle=False,
                                  collate_fn=collate, num_workers=0)

    init_meta = None
    if init_checkpoint is None:
        model = ThermalOperatorV2(model_cfg)
    else:
        model, init_meta = load_checkpoint(init_checkpoint, map_location="cpu")
        if asdict(model.cfg) != asdict(model_cfg):
            raise ValueError("initial checkpoint model_config does not match requested model config")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if tcfg.source_adapter_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.source_adapter.parameters():
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    n_trainable = sum(parameter.numel() for parameter in trainable)
    print(f"device={device.type}  params={n_params / 1e6:.2f}M  "
          f"trainable={n_trainable / 1e3:.1f}K  tag={tag}")

    if tcfg.lambda_pde_max > 0 and not probe_double_backward(model, device):
        print(f"!! double-backward failed on {device.type} -> falling back to cpu")
        device = torch.device("cpu")
        model = model.to(device)

    opt = torch.optim.AdamW(trainable, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
    hist = {"train_data": [], "train_pde": [], "train_boundary": [],
            "train_interface": [], "val_rmse_ctx": [],
            "val_rmse_noctx": [], "lambda": []}
    best_val, best_ep = float("inf"), -1
    ckpt_path = out_dir / f"{tag}.pt"

    for ep in range(tcfg.epochs):
        lam = lambda_schedule(ep, tcfg)
        ramp = physics_ramp(ep, tcfg)
        lam_boundary = tcfg.lambda_boundary_max * ramp
        lam_interface = tcfg.lambda_interface_max * ramp
        train_ds.set_epoch(ep)
        model.train()
        sd = sp = sb = si = 0.0
        nb = 0
        for batch in train_loader:
            (sensors, s_mask, q_xy, q_theta, c_xy, q_over_s,
             cond, cond_drop, h_hat, conductivity, sink_multiplier,
             b_xy, b_normal, b_conductivity, b_gamma, b_robin,
             ctx, ctx_state, ctx_mask, materials, material_mask,
             sources, source_mask, boundaries, boundary_mask) = _to_device(batch, device)

            def predict(xy, batch_index=None):
                sl = slice(None) if batch_index is None else slice(batch_index, batch_index + 1)
                return model(
                    sensors[sl], xy, cond[sl], sensor_mask=s_mask[sl],
                    context=None if ctx is None else ctx[sl],
                    context_state=None if ctx_state is None else ctx_state[sl],
                    context_mask=None if ctx_mask is None else ctx_mask[sl],
                    cond_mask=cond_drop[sl], materials=materials[sl],
                    material_mask=material_mask[sl],
                    sources=sources[sl], source_mask=source_mask[sl],
                    boundaries=boundaries[sl], boundary_mask=boundary_mask[sl],
                )

            pred = predict(q_xy)
            data_loss = F.mse_loss(pred, q_theta)
            if lam > 0:
                R = pde_residual(
                    predict, c_xy, q_over_s, h_hat, conductivity=conductivity,
                    sink_multiplier=sink_multiplier,
                )
                p_loss = pde_loss(R)
            else:
                p_loss = data_loss.new_zeros(())
            if lam_boundary > 0:
                b_loss = pde_loss(boundary_residual(
                    predict, b_xy, b_normal, b_conductivity, b_gamma, b_robin
                ))
            else:
                b_loss = data_loss.new_zeros(())
            if lam_interface > 0:
                i_loss = interface_flux_loss(
                    predict, materials, material_mask,
                    n_points=tcfg.n_interface,
                )
                if i_loss is None:
                    i_loss = data_loss.new_zeros(())
            else:
                i_loss = data_loss.new_zeros(())
            loss = (data_loss + lam * p_loss + lam_boundary * b_loss
                    + lam_interface * i_loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sd += data_loss.item()
            sp += p_loss.detach().item()
            sb += b_loss.detach().item()
            si += i_loss.detach().item()
            nb += 1
        sched.step()

        v_ctx = evaluate(model, val_loader_ctx, device)
        v_noctx = evaluate(model, val_loader_noctx, device)
        v_sel = (tcfg.val_context_weight * v_ctx
                 + (1.0 - tcfg.val_context_weight) * v_noctx)
        hist["train_data"].append(sd / nb)
        hist["train_pde"].append(sp / nb)
        hist["train_boundary"].append(sb / nb)
        hist["train_interface"].append(si / nb)
        hist["val_rmse_ctx"].append(v_ctx)
        hist["val_rmse_noctx"].append(v_noctx)
        hist["lambda"].append(lam)
        if v_sel < best_val:
            best_val, best_ep = v_sel, ep
            save_checkpoint(ckpt_path, model, extra={
                "train_config": asdict(tcfg), "scenario_config": asdict(scfg),
                "epoch": ep, "val_rmse_ctx": v_ctx, "val_rmse_noctx": v_noctx,
                "tag": tag,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
                "init_epoch": init_meta.get("epoch") if init_meta is not None else None,
            })
        if ep % 5 == 0 or ep == tcfg.epochs - 1:
            print(f"ep {ep:3d}  data {sd / nb:.5f}  pde {sp / nb:.5f}  "
                  f"val ctx {v_ctx:.4f} | no-ctx {v_noctx:.4f}  λ {lam:.1e}")

    # Report the deployable best checkpoint, not the in-memory final epoch.
    model, _ = load_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    kcurve = k_scaling_curve(model, val_boards, device, with_context=False)
    kcurve_ctx = k_scaling_curve(model, val_boards, device, with_context=True)
    print("K-scaling RMSE(θ') no-ctx:", {k: round(v, 4) for k, v in kcurve.items()})
    print("K-scaling RMSE(θ')   ctx:", {k: round(v, 4) for k, v in kcurve_ctx.items()})
    (out_dir / f"{tag}_history.json").write_text(
        json.dumps({"hist": hist, "k_curve": kcurve, "k_curve_ctx": kcurve_ctx,
                    "best_val": best_val, "best_epoch": best_ep,
                    "n_params": n_params, "n_trainable": n_trainable}, indent=2),
        encoding="utf-8",
    )
    print(f"best val (mean ctx/no-ctx) {best_val:.4f} @ epoch {best_ep} -> {ckpt_path}")
    return ckpt_path
