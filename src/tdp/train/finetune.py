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
import torch.nn.functional as F

from tdp.data.patches import patch_pixels
from tdp.model.normalization import px_to_xy
from tdp.sim.fdm import bilinear
from tdp.train.losses import boundary_residual, interface_flux_loss, pde_loss, pde_residual

ROOT = Path(__file__).resolve().parents[2].parent  # src/tdp/train -> repo root


@dataclass
class FinetuneConfig:
    n_tokens: int = 8
    steps: int = 600
    lr_tokens: float = 3e-2
    lr_scalars: float = 1e-2
    lr_material: float = 8e-3
    lr_source: float = 8e-3
    lr_boundary: float = 5e-3
    n_query: int = 512
    n_colloc: int = 192
    lambda_pde: float = 1e-4
    lambda_boundary: float = 0.0
    lambda_interface: float = 0.0
    lambda_material_prior: float = 2e-3
    lambda_source_prior: float = 2e-3
    lambda_boundary_prior: float = 2e-3
    n_boundary: int = 48
    n_interface: int = 24
    lambda_soc: float = 0.0
    lambda_edge: float = 0.0
    lambda_plateau: float = 0.0
    lambda_peak: float = 0.0
    n_shape: int = 128
    n_edge: int = 48
    shape_region_index: int = 0
    shape_region_type: str = "material"
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
    """Small per-board state; fixed geometry is data, never backbone weights."""

    def __init__(self, n_tokens: int, d_model: int, init_tokens: torch.Tensor | None = None,
                 material_descriptors: torch.Tensor | None = None,
                 source_descriptors: torch.Tensor | None = None,
                 boundary_descriptors: torch.Tensor | None = None):
        super().__init__()
        if init_tokens is None:
            init_tokens = torch.randn(1, n_tokens, d_model) * 0.02
        self.tokens = nn.Parameter(init_tokens.clone())
        self.log_h = nn.Parameter(torch.tensor(0.0))
        self.log_gamma = nn.Parameter(torch.tensor(0.0))

        material = (torch.zeros(0, 10) if material_descriptors is None
                    else material_descriptors.detach().float().reshape(-1, 10))
        self.register_buffer("material_geometry", torch.cat(
            [material[:, :6], material[:, 9:10]], dim=-1
        ))
        self.register_buffer("material_log_k_prior", material[:, 6].clone())
        self.register_buffer("material_log_h_prior", material[:, 7].clone())
        self.material_log_k = nn.Parameter(material[:, 6].clone())
        self.material_log_h = nn.Parameter(material[:, 7].clone())
        rc0 = material[:, 8].clamp_min(1e-4)
        raw_rc0 = torch.log(torch.expm1(rc0))
        self.register_buffer("material_raw_log1p_rc_prior", raw_rc0.clone())
        self.material_raw_log1p_rc = nn.Parameter(raw_rc0)

        source = (torch.zeros(0, 8) if source_descriptors is None
                  else source_descriptors.detach().float().reshape(-1, 8))
        self.register_buffer("source_geometry", torch.cat(
            [source[:, :6], source[:, 7:8]], dim=-1
        ))
        self.register_buffer("source_log_amp_prior", source[:, 6].clone())
        self.source_log_amp = nn.Parameter(source[:, 6].clone())

        boundary = (torch.zeros(0, 8) if boundary_descriptors is None
                    else boundary_descriptors.detach().float().reshape(-1, 8))
        self.register_buffer("boundary_geometry", torch.cat(
            [boundary[:, :6], boundary[:, 7:8]], dim=-1
        ))
        self.register_buffer("boundary_log_gamma_init", boundary[:, 6].clone())
        self.boundary_log_gamma_delta = nn.Parameter(torch.zeros(len(boundary)))

    def cond(self, aspect: float) -> torch.Tensor:
        return torch.stack([
            self.log_h / 3.0,
            self.log_gamma / 3.0,
            self.log_h.new_tensor(float(aspect)),
            self.log_h.new_tensor(1.0),
        ]).unsqueeze(0)  # (1, 4)

    def materials(self) -> torch.Tensor:
        if self.material_geometry.shape[0] == 0:
            return self.material_geometry.new_zeros(1, 0, 10)
        descriptor = torch.cat([
            self.material_geometry[:, :6],
            self.material_log_k[:, None],
            self.material_log_h[:, None],
            F.softplus(self.material_raw_log1p_rc)[:, None],
            self.material_geometry[:, 6:7],
        ], dim=-1)
        return descriptor.unsqueeze(0)

    def boundaries(self) -> torch.Tensor:
        if self.boundary_geometry.shape[0] == 0:
            return self.boundary_geometry.new_zeros(1, 0, 8)
        log_gamma = (self.boundary_log_gamma_init + self.log_gamma
                     + self.boundary_log_gamma_delta)
        descriptor = torch.cat([
            self.boundary_geometry[:, :6], log_gamma[:, None],
            self.boundary_geometry[:, 6:7],
        ], dim=-1)
        return descriptor.unsqueeze(0)

    def sources(self) -> torch.Tensor:
        if self.source_geometry.shape[0] == 0:
            return self.source_geometry.new_zeros(1, 0, 8)
        descriptor = torch.cat([
            self.source_geometry[:, :6],
            self.source_log_amp[:, None],
            self.source_geometry[:, 6:7],
        ], dim=-1)
        return descriptor.unsqueeze(0)

    def material_fields(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable relative in-plane K and vertical H fields."""
        materials = self.materials()
        if materials.shape[1] == 0:
            ones = torch.ones(xy.shape[:2], device=xy.device, dtype=xy.dtype)
            return ones, ones
        center, half = materials[..., :2], materials[..., 2:4].clamp_min(1e-4)
        sin_a, cos_a = materials[..., 4], materials[..., 5]
        delta = xy[:, :, None, :] - center[:, None, :, :]
        ux = ((cos_a[:, None] * delta[..., 0] + sin_a[:, None] * delta[..., 1])
              / half[:, None, :, 0])
        uy = ((-sin_a[:, None] * delta[..., 0] + cos_a[:, None] * delta[..., 1])
              / half[:, None, :, 1])
        rect = torch.maximum(ux.abs(), uy.abs()) - 1.0
        ellipse = torch.sqrt(ux.square() + uy.square() + 1e-8) - 1.0
        shape = materials[..., 9][:, None]
        inside = torch.sigmoid(-(rect * (1 - shape) + ellipse * shape) / 0.04)
        k = torch.ones_like(inside[..., 0])
        h = torch.ones_like(k)
        for region in range(materials.shape[1]):
            gate = inside[..., region]
            k = k * (1.0 - gate) + torch.exp(materials[:, region, 6])[:, None] * gate
            h = h * (1.0 - gate) + torch.exp(materials[:, region, 7])[:, None] * gate
        return k, h

    def save(self, path: Path, meta: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tokens": self.tokens.detach().cpu(),
                    "log_h": float(self.log_h.detach()),
                    "log_gamma": float(self.log_gamma.detach()),
                    "material_descriptors": self.materials()[0].detach().cpu(),
                    "source_descriptors": self.sources()[0].detach().cpu(),
                    "boundary_descriptors": self.boundaries()[0].detach().cpu(),
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


def material_pixel_pool(descriptor: np.ndarray, shape: tuple[int, int],
                        aspect: float, allowed: np.ndarray) -> np.ndarray:
    """Allowed image pixels inside one explicit material region."""
    height, width = shape
    rows, cols = np.indices(shape)
    x = cols / max(width - 1, 1) * aspect
    y = rows / max(height - 1, 1)
    cx, cy, hx, hy, sin_a, cos_a = descriptor[:6]
    dx, dy = x - cx, y - cy
    u = (cos_a * dx + sin_a * dy) / max(hx, 1e-6)
    v = (-sin_a * dx + cos_a * dy) / max(hy, 1e-6)
    if descriptor[9] > 0.5:
        inside = u**2 + v**2 <= 1.0
    else:
        inside = np.maximum(np.abs(u), np.abs(v)) <= 1.0
    return np.stack(np.nonzero(inside & allowed), axis=-1)


def material_edge_pair_pool(descriptor: np.ndarray, shape: tuple[int, int],
                            aspect: float, allowed: np.ndarray,
                            points_per_edge: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Paired points just inside/outside a material interface."""
    height, width = shape
    cx, cy, hx, hy, sin_a, cos_a = descriptor[:6]
    if descriptor[9] > 0.5:
        phi = np.linspace(0.0, 2.0 * np.pi, 4 * points_per_edge, endpoint=False)
        local_x, local_y = hx * np.cos(phi), hy * np.sin(phi)
        nx, ny = np.cos(phi) / max(hx, 1e-6), np.sin(phi) / max(hy, 1e-6)
        norm = np.hypot(nx, ny)
        nx, ny = nx / norm, ny / norm
    else:
        tangent = np.linspace(-1.0, 1.0, points_per_edge)
        local_x = np.concatenate([
            np.full_like(tangent, -hx), np.full_like(tangent, hx),
            tangent * hx, tangent * hx,
        ])
        local_y = np.concatenate([
            tangent * hy, tangent * hy,
            np.full_like(tangent, -hy), np.full_like(tangent, hy),
        ])
        nx = np.concatenate([
            -np.ones_like(tangent), np.ones_like(tangent),
            np.zeros_like(tangent), np.zeros_like(tangent),
        ])
        ny = np.concatenate([
            np.zeros_like(tangent), np.zeros_like(tangent),
            -np.ones_like(tangent), np.ones_like(tangent),
        ])
    bx = cx + cos_a * local_x - sin_a * local_y
    by = cy + sin_a * local_x + cos_a * local_y
    normal_x = cos_a * nx - sin_a * ny
    normal_y = sin_a * nx + cos_a * ny
    epsilon = 2.0 / max(height - 1, 1)  # two image pixels in normalized length
    inside = np.stack([bx - epsilon * normal_x, by - epsilon * normal_y], axis=-1)
    outside = np.stack([bx + epsilon * normal_x, by + epsilon * normal_y], axis=-1)

    def pixels(xy):
        row = np.rint(xy[:, 1] * (height - 1)).astype(int)
        col = np.rint(xy[:, 0] / aspect * (width - 1)).astype(int)
        valid = ((row >= 0) & (row < height) & (col >= 0) & (col < width))
        row = np.clip(row, 0, height - 1)
        col = np.clip(col, 0, width - 1)
        return row, col, valid

    ri, ci, valid_i = pixels(inside)
    ro, co, valid_o = pixels(outside)
    valid = valid_i & valid_o & allowed[ri, ci] & allowed[ro, co]
    return inside[valid].astype(np.float32), outside[valid].astype(np.float32)


@torch.no_grad()
def predict_field(model, card: BoardCard, sensors, aspect: float,
                  q_xy: np.ndarray, chunk: int = 4096) -> np.ndarray:
    device = next(model.parameters()).device
    sensors = sensors.to(device)
    outs = []
    for i in range(0, len(q_xy), chunk):
        q = torch.from_numpy(q_xy[i:i + chunk].astype(np.float32))[None].to(device)
        outs.append(model(sensors, q, card.cond(aspect),
                          board_tokens=card.tokens, materials=card.materials(),
                          sources=card.sources(),
                          boundaries=card.boundaries())[0].detach().cpu().numpy())
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
             session: str = "s1", verbose: bool = True,
             material_descriptors: torch.Tensor | None = None,
             source_descriptors: torch.Tensor | None = None,
             boundary_descriptors: torch.Tensor | None = None) -> tuple[BoardCard, dict]:
    """Optimize a BoardCard on the train cases; early-stop on patch RMSE."""
    model.eval()
    device = next(model.parameters()).device
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
    pts_t = torch.from_numpy(np.concatenate(pts).astype(np.float32)).to(device)
    with torch.no_grad():
        enc = model.sens_enc(pts_t[None])[0]
    init = enc[torch.linspace(0, len(enc) - 1, cfg.n_tokens).long()][None]
    card = BoardCard(
        cfg.n_tokens, model.cfg.d_model, init_tokens=init,
        material_descriptors=material_descriptors,
        source_descriptors=source_descriptors,
        boundary_descriptors=boundary_descriptors,
    ).to(device)

    groups = [
        {"params": [card.tokens], "lr": cfg.lr_tokens},
        {"params": [card.log_h, card.log_gamma], "lr": cfg.lr_scalars},
    ]
    material_params = [card.material_log_k, card.material_log_h,
                       card.material_raw_log1p_rc]
    if any(parameter.numel() for parameter in material_params):
        groups.append({"params": material_params, "lr": cfg.lr_material})
    if card.source_log_amp.numel():
        groups.append({"params": [card.source_log_amp], "lr": cfg.lr_source})
    if card.boundary_log_gamma_delta.numel():
        groups.append({"params": [card.boundary_log_gamma_delta], "lr": cfg.lr_boundary})
    opt = torch.optim.Adam(groups)

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

    use_shape_loss = any(weight > 0 for weight in (
        cfg.lambda_soc, cfg.lambda_edge, cfg.lambda_plateau, cfg.lambda_peak
    ))
    shape_pools: list[np.ndarray] = []
    edge_pools: list[tuple[np.ndarray, np.ndarray]] = []
    if use_shape_loss:
        if cfg.shape_region_type == "source":
            if card.sources().shape[1] <= cfg.shape_region_index:
                raise ValueError("shape loss requested without the configured source region")
            source = card.sources()[0, cfg.shape_region_index].detach()
            # Reuse the geometry-pool utilities, whose 10-value descriptor has
            # the same first six geometry values and stores shape at index 9.
            shape_descriptor = torch.cat([
                source[:6], source.new_zeros(3), source[7:8]
            ]).cpu().numpy()
        elif cfg.shape_region_type == "material":
            if card.materials().shape[1] <= cfg.shape_region_index:
                raise ValueError("shape loss requested without the configured material region")
            shape_descriptor = card.materials()[
                0, cfg.shape_region_index
            ].detach().cpu().numpy()
        else:
            raise ValueError(f"unknown shape_region_type {cfg.shape_region_type!r}")
        for case, sup_px in zip(train_cases, supervision):
            allowed = np.zeros(case.shape, dtype=bool)
            allowed[sup_px[:, 0], sup_px[:, 1]] = True
            shape_pool = material_pixel_pool(
                shape_descriptor, case.shape, case.aspect, allowed
            )
            edge_pool = material_edge_pair_pool(
                shape_descriptor, case.shape, case.aspect, allowed
            )
            if len(shape_pool) == 0:
                raise ValueError(f"material region has no supervised pixels in {case.case_id}")
            if cfg.lambda_edge > 0 and len(edge_pool[0]) == 0:
                raise ValueError(f"material edge has no supervised pairs in {case.case_id}")
            shape_pools.append(shape_pool)
            edge_pools.append(edge_pool)

    hist = {"loss": [], "data_loss": [], "soc_loss": [], "edge_loss": [],
            "plateau_loss": [], "peak_loss": [], "patch_rmse": []}
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
        sensors = sensors.to(device)

        n_rand = cfg.n_query - cfg.n_query_hot - cfg.n_query_site
        sel = np.concatenate([
            supervision[i][rng.integers(0, len(supervision[i]), n_rand)],
            hot_pools[i][rng.integers(0, len(hot_pools[i]), cfg.n_query_hot)],
            site_pools[i][rng.integers(0, len(site_pools[i]), cfg.n_query_site)],
        ])
        q_xy = px_to_xy(sel[:, 0], sel[:, 1], n_rows=h, n_cols=w)
        q_t = torch.from_numpy(
            ((case.field[sel[:, 0], sel[:, 1]] - case.amb) / dT).astype(np.float32))[None].to(device)
        sig = np.minimum(case.std[sel[:, 0], sel[:, 1]], cfg.sigma_clip_c)
        wgt = 1.0 / (sig ** 2 + cfg.sigma_floor_c ** 2)
        wgt = torch.from_numpy((wgt / wgt.mean()).astype(np.float32))[None].to(device)

        cond = card.cond(case.aspect)
        q = torch.from_numpy(q_xy.astype(np.float32))[None].to(device)

        def predict(xy_, batch_index=None):
            return model(
                sensors, xy_, cond, board_tokens=card.tokens,
                materials=card.materials(), sources=card.sources(),
                boundaries=card.boundaries(),
            )

        pred = predict(q)
        data_loss = (wgt * (pred - q_t) ** 2).mean()

        loss = data_loss + cfg.lambda_prior * (card.log_h / np.log(3.0)) ** 2
        soc_loss = data_loss.new_zeros(())
        edge_loss = data_loss.new_zeros(())
        plateau_loss = data_loss.new_zeros(())
        peak_loss = data_loss.new_zeros(())
        if use_shape_loss:
            shape_sel = shape_pools[i][rng.integers(
                0, len(shape_pools[i]), cfg.n_shape
            )]
            shape_xy = px_to_xy(
                shape_sel[:, 0], shape_sel[:, 1], n_rows=h, n_cols=w
            )
            shape_q = torch.from_numpy(shape_xy.astype(np.float32))[None].to(device)
            shape_truth = torch.from_numpy((
                (case.field[shape_sel[:, 0], shape_sel[:, 1]] - case.amb) / dT
            ).astype(np.float32))[None].to(device)
            shape_pred = predict(shape_q)
            soc_loss = F.mse_loss(shape_pred, shape_truth)
            plateau_loss = F.mse_loss(
                shape_pred - shape_pred.mean(dim=1, keepdim=True),
                shape_truth - shape_truth.mean(dim=1, keepdim=True),
            )
            beta = 20.0
            pred_peak = (torch.logsumexp(beta * shape_pred, dim=1)
                         - np.log(shape_pred.shape[1])) / beta
            truth_peak = (torch.logsumexp(beta * shape_truth, dim=1)
                          - np.log(shape_truth.shape[1])) / beta
            peak_loss = F.mse_loss(pred_peak, truth_peak)
            if cfg.lambda_edge > 0:
                inside_pool, outside_pool = edge_pools[i]
                edge_sel = rng.integers(0, len(inside_pool), cfg.n_edge)
                inside_xy, outside_xy = inside_pool[edge_sel], outside_pool[edge_sel]
                edge_xy = np.concatenate([inside_xy, outside_xy], axis=0)
                edge_q = torch.from_numpy(edge_xy.astype(np.float32))[None].to(device)
                edge_pred = predict(edge_q)
                pred_inside, pred_outside = edge_pred.chunk(2, dim=1)
                truth_inside = bilinear(case.field, inside_xy, case.aspect)
                truth_outside = bilinear(case.field, outside_xy, case.aspect)
                truth_jump = torch.from_numpy((
                    (truth_outside - truth_inside) / dT
                ).astype(np.float32))[None].to(device)
                edge_loss = F.mse_loss(pred_outside - pred_inside, truth_jump)
            loss = (loss + cfg.lambda_soc * soc_loss
                    + cfg.lambda_edge * edge_loss
                    + cfg.lambda_plateau * plateau_loss
                    + cfg.lambda_peak * peak_loss)
        if card.material_log_k.numel():
            material_prior = (
                (card.material_log_k - card.material_log_k_prior).square().mean()
                + (card.material_log_h - card.material_log_h_prior).square().mean()
                + 0.25 * (card.material_raw_log1p_rc
                          - card.material_raw_log1p_rc_prior).square().mean()
            )
            loss = loss + cfg.lambda_material_prior * material_prior
        if card.source_log_amp.numel():
            source_prior = (
                card.source_log_amp - card.source_log_amp_prior
            ).square().mean()
            loss = loss + cfg.lambda_source_prior * source_prior
        if card.boundary_log_gamma_delta.numel():
            loss = loss + cfg.lambda_boundary_prior * card.boundary_log_gamma_delta.square().mean()
        if cfg.lambda_pde > 0:
            c_xy = torch.from_numpy(samplers[i](rng, cfg.n_colloc).astype(np.float32))[None].to(device)
            h_hat = torch.exp(card.log_h)[None]
            conductivity, sink_multiplier = card.material_fields(c_xy)
            R = pde_residual(
                predict, c_xy, torch.zeros(1, cfg.n_colloc, device=device), h_hat,
                conductivity=conductivity, sink_multiplier=sink_multiplier)
            loss = loss + cfg.lambda_pde * pde_loss(R)
        if cfg.lambda_boundary > 0 and card.boundaries().shape[1]:
            descriptors = card.boundaries()[0]
            picks = torch.randint(len(descriptors), (cfg.n_boundary,), device=device)
            selected = descriptors[picks]
            along = torch.rand(cfg.n_boundary, 1, device=device)
            b_xy = (selected[:, :2] + along * (selected[:, 2:4] - selected[:, :2]))[None]
            b_normal = selected[:, 4:6][None]
            b_gamma = torch.exp(selected[:, 6])[None]
            b_robin = (selected[:, 7] > 0.5)[None]
            b_k, _ = card.material_fields(b_xy)
            b_residual = boundary_residual(
                predict, b_xy, b_normal, b_k, b_gamma, b_robin
            )
            loss = loss + cfg.lambda_boundary * pde_loss(b_residual)
        if cfg.lambda_interface > 0 and card.materials().shape[1]:
            interface_loss = interface_flux_loss(
                predict, card.materials(), n_points=cfg.n_interface
            )
            loss = loss + cfg.lambda_interface * interface_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        hist["loss"].append(float(loss.detach()))
        hist["data_loss"].append(float(data_loss.detach()))
        hist["soc_loss"].append(float(soc_loss.detach()))
        hist["edge_loss"].append(float(edge_loss.detach()))
        hist["plateau_loss"].append(float(plateau_loss.detach()))
        hist["peak_loss"].append(float(peak_loss.detach()))

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            rmse = eval_patches(model, card, train_cases, patches_rc, sites, session)
            hist["patch_rmse"].append([step, rmse])
            if rmse < best[0]:
                best = (rmse, {k: v.detach().clone() for k, v in card.state_dict().items()},
                        step)
            if verbose and step % (cfg.eval_every * 5) == 0:
                print(f"  step {step:4d}  loss {float(loss.detach()):.4f}  "
                      f"patch RMSE {rmse:.3f} C  h_hat={float(torch.exp(card.log_h).detach()):.2f}")

    card.load_state_dict(best[1])
    hist["best_patch_rmse"], hist["best_step"] = best[0], best[2]
    hist["h_hat"] = float(torch.exp(card.log_h).detach())
    hist["gamma"] = float(torch.exp(card.log_gamma).detach())
    hist["material_descriptors"] = card.materials()[0].detach().cpu().tolist()
    hist["source_descriptors"] = card.sources()[0].detach().cpu().tolist()
    hist["boundary_descriptors"] = card.boundaries()[0].detach().cpu().tolist()
    return card, hist
