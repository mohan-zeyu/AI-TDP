"""M7 Tier-1 fine-tuning: board-token adaptation with a frozen backbone.

Trainables (~1k params): N board tokens (enter the attention set through the
context-token interface, warm-started from encoded real reference points) plus
the physics scalars log ĥ and log γ (fed through the condition token). The
1.06M-parameter operator stays frozen — forgetting is impossible and no
synthetic replay is needed. The result is a per-board "board card" (a few KB).

Supervision: trust-masked canonical fields (validation patches excluded),
1/σ²-weighted; inputs are noise-resampled per step (measured σ maps); PDE
residual applies source-free outside dilated component rectangles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tdp.data.patches import patch_pixels
from tdp.model.normalization import px_to_xy
from tdp.train.losses import pde_loss, pde_residual

ROOT = Path(__file__).resolve().parents[2].parent  # src/tdp/train -> repo root


@dataclass
class FinetuneConfig:
    n_tokens: int = 8
    steps: int = 600
    lr_tokens: float = 3e-2
    lr_scalars: float = 1e-2
    n_query: int = 512
    n_colloc: int = 192
    lambda_pde: float = 1e-4
    lambda_prior: float = 1e-2   # weak prior: log ĥ ~ N(0, ln 3) → ĥ within the measured bound
    sigma_floor_c: float = 0.10
    sigma_clip_c: float = 0.35   # cap: drift-inflated σ near the hotspot must not
                                 # down-weight the peak into oblivion
    n_query_hot: int = 32        # always-included hottest supervised pixels
    n_query_site: int = 32       # always-included pixels around sensor sites
    jitter_px: int = 1
    eval_every: int = 20
    seed: int = 0


class BoardCard(nn.Module):
    """The deployable per-board artifact: tokens + physics scalars."""

    def __init__(self, n_tokens: int, d_model: int, init_tokens: torch.Tensor | None = None):
        super().__init__()
        if init_tokens is None:
            init_tokens = torch.randn(1, n_tokens, d_model) * 0.02
        self.tokens = nn.Parameter(init_tokens.clone())
        self.log_h = nn.Parameter(torch.tensor(0.0))
        self.log_gamma = nn.Parameter(torch.tensor(0.0))

    def cond(self, aspect: float) -> torch.Tensor:
        return torch.stack([
            self.log_h / 3.0,
            self.log_gamma / 3.0,
            torch.tensor(float(aspect)),
            torch.tensor(1.0),
        ]).unsqueeze(0)  # (1, 4)

    def save(self, path: Path, meta: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tokens": self.tokens.detach().cpu(),
                    "log_h": float(self.log_h), "log_gamma": float(self.log_gamma),
                    "meta": meta or {}}, path)


@dataclass
class RealCase:
    case_id: str
    field: np.ndarray       # canonical mean (H, W) °C
    std: np.ndarray         # per-pixel repeatability map
    amb: float
    trusted: np.ndarray     # bool (H, W)
    shape: tuple[int, int]
    aspect: float


def load_case(proc: Path, cid: str) -> RealCase:
    can = np.load(proc / f"canonical_{cid}.npz")
    trust = np.load(proc / f"trust_{cid}.npz")["trusted"]
    field = can["mean_T"].astype(np.float64)
    h, w = field.shape
    return RealCase(cid, field, can["std_T"].astype(np.float64), float(can["t_amb"]),
                    trust, (h, w), w / h)


def load_case_subset(proc: Path, cid: str, n_frames: int | None,
                     seed: int = 0) -> tuple[RealCase, int]:
    """Few-shot variant: build the training view from only n_frames of the
    steady frames (evaluation elsewhere still uses the full-data canonical).
    n_frames=None or >= available -> all steady frames."""
    can = np.load(proc / f"canonical_{cid}.npz")
    d = np.load(proc / f"{cid}.npz")
    idx = np.asarray(can["frame_indices"])
    if n_frames is not None and n_frames < len(idx):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, n_frames, replace=False))
    T = d["T"][idx].astype(np.float64)
    field = T.mean(0)
    std = T.std(0, ddof=1) if len(idx) > 1 else np.zeros_like(field)
    amb = float(np.mean(d["t_amb"][idx]))
    trust = np.load(proc / f"trust_{cid}.npz")["trusted"]
    h, w = field.shape
    return RealCase(cid, field, std, amb, trust, (h, w), w / h), int(len(idx))


def sites_px_for(sites: list[dict], session: str, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.array([[s["v"] * h, s["u"] * w] for s in sites
                     if session in s["trusted_sessions"]])


def read_sites(field: np.ndarray, spx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(K,2) nondim coords + 3×3-median readings at site pixels."""
    h, w = field.shape
    rows = np.clip(spx[:, 0].round().astype(int), 1, h - 2)
    cols = np.clip(spx[:, 1].round().astype(int), 1, w - 2)
    vals = np.array([np.median(field[r - 1:r + 2, c - 1:c + 2]) for r, c in zip(rows, cols)])
    return px_to_xy(rows, cols, n_rows=h, n_cols=w), vals


def make_sensor_tensor(xy: np.ndarray, vals: np.ndarray, amb: float):
    dT = max(vals.max() - amb, 1.0)
    theta = (vals - amb) / dT
    sensors = torch.from_numpy(
        np.concatenate([xy, theta[:, None]], -1).astype(np.float32))[None]
    return sensors, dT


def source_free_sampler(source_rects: list[dict], aspect: float, dilate: float = 0.03):
    """Rejection sampler for collocation points outside dilated component rects."""
    rects = [(r["u0"] - dilate, r["v0"] - dilate, r["u1"] + dilate, r["v1"] + dilate)
             for r in source_rects]

    def sample(rng: np.random.Generator, n: int) -> np.ndarray:
        out = []
        while sum(len(o) for o in out) < n:
            u = rng.uniform(0.02, 0.98, 4 * n)
            v = rng.uniform(0.02, 0.98, 4 * n)
            ok = np.ones(len(u), dtype=bool)
            for u0, v0, u1, v1 in rects:
                ok &= ~((u >= u0) & (u <= u1) & (v >= v0) & (v <= v1))
            out.append(np.stack([u[ok] * aspect, v[ok]], -1))
        return np.concatenate(out)[:n]

    return sample


@torch.no_grad()
def predict_field(model, card: BoardCard, sensors, aspect: float,
                  q_xy: np.ndarray, chunk: int = 4096) -> np.ndarray:
    outs = []
    for i in range(0, len(q_xy), chunk):
        q = torch.from_numpy(q_xy[i:i + chunk].astype(np.float32))[None]
        outs.append(model(sensors, q, card.cond(aspect),
                          board_tokens=card.tokens)[0].numpy())
    return np.concatenate(outs)


@torch.no_grad()
def eval_patches(model, card, cases: list[RealCase], patches_rc, sites, session) -> float:
    """RMSE (°C) at held-out validation-patch pixels across the given cases."""
    se, n = 0.0, 0
    for case in cases:
        pmask = patch_pixels(patches_rc, case.shape)
        rows, cols = np.nonzero(pmask)
        q_xy = px_to_xy(rows, cols, n_rows=case.shape[0], n_cols=case.shape[1])
        spx = sites_px_for(sites, session, case.shape)
        xy, vals = read_sites(case.field, spx)
        sensors, dT = make_sensor_tensor(xy, vals, case.amb)
        pred = predict_field(model, card, sensors, case.aspect, q_xy) * dT + case.amb
        truth = case.field[rows, cols]
        se += float(((pred - truth) ** 2).sum())
        n += len(truth)
    return (se / n) ** 0.5


def finetune(model, train_cases: list[RealCase], sites: list[dict],
             source_rects: list[dict], patches_rc, cfg: FinetuneConfig,
             session: str = "s1", verbose: bool = True) -> tuple[BoardCard, dict]:
    """Optimize a BoardCard on the train cases; early-stop on patch RMSE."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    # warm start: encode reference points from the train cases, spread-pick N
    pts = []
    for case in train_cases:
        h, w = case.shape
        rows = rng.integers(2, h - 2, 96)
        cols = rng.integers(2, w - 2, 96)
        ex = np.clip(case.field[rows, cols] - case.amb, 0, None)
        s_ctx = max(ex.max(), 1e-6)
        pts.append(np.concatenate(
            [px_to_xy(rows, cols, n_rows=h, n_cols=w), (ex / s_ctx)[:, None]], -1))
    pts_t = torch.from_numpy(np.concatenate(pts).astype(np.float32))
    with torch.no_grad():
        enc = model.sens_enc(pts_t[None])[0]
    init = enc[torch.linspace(0, len(enc) - 1, cfg.n_tokens).long()][None]
    card = BoardCard(cfg.n_tokens, model.cfg.d_model, init_tokens=init)

    opt = torch.optim.Adam([
        {"params": [card.tokens], "lr": cfg.lr_tokens},
        {"params": [card.log_h, card.log_gamma], "lr": cfg.lr_scalars},
    ])

    supervision, hot_pools, site_pools, samplers = [], [], [], []
    for case in train_cases:
        sup = case.trusted & ~patch_pixels(patches_rc, case.shape)
        px = np.stack(np.nonzero(sup), -1)
        supervision.append(px)
        vals = case.field[px[:, 0], px[:, 1]]
        hot_pools.append(px[vals >= np.quantile(vals, 0.98)])  # peak region
        spx_all = sites_px_for(sites, session, case.shape)
        near = np.zeros(len(px), dtype=bool)
        for sr, sc in spx_all:
            near |= np.hypot(px[:, 0] - sr, px[:, 1] - sc) <= 2.0
        site_pools.append(px[near] if near.any() else px[:1])
        samplers.append(source_free_sampler(source_rects, case.aspect))

    hist = {"loss": [], "patch_rmse": []}
    best = (float("inf"), None, -1)
    for step in range(cfg.steps):
        i = int(rng.integers(len(train_cases)))
        case = train_cases[i]
        h, w = case.shape

        noisy = case.field + rng.standard_normal(case.shape) * np.maximum(
            case.std, cfg.sigma_floor_c)
        spx = sites_px_for(sites, session, case.shape)
        spx = spx + rng.integers(-cfg.jitter_px, cfg.jitter_px + 1, spx.shape)
        xy, vals = read_sites(noisy, spx)
        sensors, dT = make_sensor_tensor(xy, vals, case.amb)

        n_rand = cfg.n_query - cfg.n_query_hot - cfg.n_query_site
        sel = np.concatenate([
            supervision[i][rng.integers(0, len(supervision[i]), n_rand)],
            hot_pools[i][rng.integers(0, len(hot_pools[i]), cfg.n_query_hot)],
            site_pools[i][rng.integers(0, len(site_pools[i]), cfg.n_query_site)],
        ])
        q_xy = px_to_xy(sel[:, 0], sel[:, 1], n_rows=h, n_cols=w)
        q_t = torch.from_numpy(
            ((case.field[sel[:, 0], sel[:, 1]] - case.amb) / dT).astype(np.float32))[None]
        sig = np.minimum(case.std[sel[:, 0], sel[:, 1]], cfg.sigma_clip_c)
        wgt = 1.0 / (sig ** 2 + cfg.sigma_floor_c ** 2)
        wgt = torch.from_numpy((wgt / wgt.mean()).astype(np.float32))[None]

        cond = card.cond(case.aspect)
        q = torch.from_numpy(q_xy.astype(np.float32))[None]
        pred = model(sensors, q, cond, board_tokens=card.tokens)
        data_loss = (wgt * (pred - q_t) ** 2).mean()

        loss = data_loss + cfg.lambda_prior * (card.log_h / np.log(3.0)) ** 2
        if cfg.lambda_pde > 0:
            c_xy = torch.from_numpy(samplers[i](rng, cfg.n_colloc).astype(np.float32))[None]
            h_hat = torch.exp(card.log_h)[None]
            R = pde_residual(
                lambda xy_: model(sensors, xy_, cond, board_tokens=card.tokens),
                c_xy, torch.zeros(1, cfg.n_colloc), h_hat)
            loss = loss + cfg.lambda_pde * pde_loss(R)

        opt.zero_grad()
        loss.backward()
        opt.step()
        hist["loss"].append(float(loss.detach()))

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            rmse = eval_patches(model, card, train_cases, patches_rc, sites, session)
            hist["patch_rmse"].append([step, rmse])
            if rmse < best[0]:
                best = (rmse, {k: v.detach().clone() for k, v in card.state_dict().items()},
                        step)
            if verbose and step % (cfg.eval_every * 5) == 0:
                print(f"  step {step:4d}  loss {float(loss):.4f}  patch RMSE {rmse:.3f} °C  "
                      f"ĥ={float(torch.exp(card.log_h)):.2f}")

    card.load_state_dict(best[1])
    hist["best_patch_rmse"], hist["best_step"] = best[0], best[2]
    hist["h_hat"] = float(torch.exp(card.log_h))
    hist["gamma"] = float(torch.exp(card.log_gamma))
    return card, hist
