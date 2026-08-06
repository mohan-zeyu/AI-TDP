"""ThermalOperatorV2 — the v1 attention operator (ported verbatim from
AI_TDP.ipynb) extended with (a) a condition token [log ĥ, log γ, aspect, BC] and
(b) v2.5 in-context board conditioning: reference-frame points enter the
attention set as extra tokens, distinguished by learned type embeddings
(live sensor / context frame 0 / context frame 1 / condition token).

Empty context ≡ v2 behaviour. Permutation invariance over sensors and variable
K are untouched. I/O is fully nondimensional (see tdp.model.normalization).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_self: int = 2
    n_cross: int = 3
    n_fourier: int = 32
    fourier_sigma: float = 5.0
    cond_dim: int = 4
    material_dim: int = 10
    material_hidden: int = 64
    source_dim: int = 8
    source_hidden: int = 64
    boundary_dim: int = 8
    boundary_hidden: int = 48
    n_token_types: int = 4  # 0 live sensor · 1 context frame A · 2 context frame B · 3 cond


class FourierFeatures(nn.Module):
    """Random Fourier features: (x, y) → [sin(2π Bx), cos(2π Bx)] (fixed buffer)."""

    def __init__(self, input_dim=2, n_features=32, sigma=5.0):
        super().__init__()
        self.register_buffer("B", torch.randn(input_dim, n_features) * sigma)

    def forward(self, x):
        xB = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(xB), torch.cos(xB)], dim=-1)


class SensorEncoder(nn.Module):
    def __init__(self, d_model, n_fourier, sigma):
        super().__init__()
        self.ff = FourierFeatures(2, n_fourier, sigma)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_fourier + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, sensors):  # (B, K, 3) = (x, y, θ)
        xy = sensors[..., :2]
        th = sensors[..., 2:3]
        return self.mlp(torch.cat([self.ff(xy), th], dim=-1))


class QueryEncoder(nn.Module):
    def __init__(self, d_model, n_fourier, sigma):
        super().__init__()
        self.ff = FourierFeatures(2, n_fourier, sigma)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_fourier, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, q):
        return self.mlp(self.ff(q))


class MaterialGeometryAdapter(nn.Module):
    """Encode query position relative to regular material boundaries."""

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.edge_width = 0.08
        self.mlp = nn.Sequential(
            nn.Linear(11, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # Exact backward compatibility: material input initially changes
        # nothing when an old checkpoint is migrated to the new architecture.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, queries, materials, material_mask=None):
        if materials is None or materials.shape[1] == 0:
            return queries.new_zeros((*queries.shape[:2], self.mlp[-1].out_features))

        center = materials[..., 0:2]
        half = materials[..., 2:4].clamp_min(1e-4)
        sin_a, cos_a = materials[..., 4], materials[..., 5]
        log_k, log_h = materials[..., 6], materials[..., 7]
        log_rc, shape = materials[..., 8], materials[..., 9].clamp(0.0, 1.0)
        delta = queries[:, :, None, :] - center[:, None, :, :]
        local_x = cos_a[:, None, :] * delta[..., 0] + sin_a[:, None, :] * delta[..., 1]
        local_y = -sin_a[:, None, :] * delta[..., 0] + cos_a[:, None, :] * delta[..., 1]
        ux = local_x / half[:, None, :, 0]
        uy = local_y / half[:, None, :, 1]
        ax, ay = ux.abs(), uy.abs()
        rectangle_signed = torch.maximum(ax, ay) - 1.0
        ellipse_signed = torch.sqrt(ux.square() + uy.square() + 1e-8) - 1.0
        signed = rectangle_signed * (1.0 - shape[:, None, :]) + ellipse_signed * shape[:, None, :]
        inside = torch.sigmoid(-signed / self.edge_width)
        edge = torch.exp(-signed.abs() / self.edge_width)
        features = torch.stack([
            ux.clamp(-3.0, 3.0), uy.clamp(-3.0, 3.0),
            ax.clamp(0.0, 3.0), ay.clamp(0.0, 3.0),
            signed.clamp(-1.0, 3.0), inside, edge,
            log_k[:, None, :].expand_as(signed),
            log_h[:, None, :].expand_as(signed),
            log_rc[:, None, :].expand_as(signed),
            shape[:, None, :].expand_as(signed),
        ], dim=-1)
        encoded = self.mlp(features)
        if material_mask is not None:
            encoded = encoded.masked_fill(material_mask[:, None, :, None], 0.0)
        return encoded.sum(dim=2)


class SourceGeometryAdapter(nn.Module):
    """Encode heat-source geometry independently from material geometry.

    A source descriptor is ``[cx,cy,hx,hy,sin(a),cos(a),log_amp,shape]``.
    ``shape=0`` denotes a smooth rectangle and ``shape=1`` an ellipse/Gaussian.
    Amplitudes are relative within one operating state; the absolute thermal
    scale still comes from the live sensor normalization.
    """

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.edge_width = 0.08
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # A v3.1 checkpoint can be migrated without changing its predictions.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, queries, sources, source_mask=None):
        if sources is None or sources.shape[1] == 0:
            return queries.new_zeros((*queries.shape[:2], self.mlp[-1].out_features))

        center = sources[..., 0:2]
        half = sources[..., 2:4].clamp_min(1e-4)
        sin_a, cos_a = sources[..., 4], sources[..., 5]
        log_amp, shape = sources[..., 6], sources[..., 7].clamp(0.0, 1.0)
        delta = queries[:, :, None, :] - center[:, None, :, :]
        local_x = cos_a[:, None, :] * delta[..., 0] + sin_a[:, None, :] * delta[..., 1]
        local_y = -sin_a[:, None, :] * delta[..., 0] + cos_a[:, None, :] * delta[..., 1]
        ux = local_x / half[:, None, :, 0]
        uy = local_y / half[:, None, :, 1]
        ax, ay = ux.abs(), uy.abs()
        rectangle_signed = torch.maximum(ax, ay) - 1.0
        ellipse_signed = torch.sqrt(ux.square() + uy.square() + 1e-8) - 1.0
        signed = rectangle_signed * (1.0 - shape[:, None, :]) + ellipse_signed * shape[:, None, :]
        inside = torch.sigmoid(-signed / self.edge_width)
        edge = torch.exp(-signed.abs() / self.edge_width)
        features = torch.stack([
            ux.clamp(-3.0, 3.0), uy.clamp(-3.0, 3.0),
            ax.clamp(0.0, 3.0), ay.clamp(0.0, 3.0),
            signed.clamp(-1.0, 3.0), inside, edge,
            log_amp[:, None, :].expand_as(signed),
            shape[:, None, :].expand_as(signed),
        ], dim=-1)
        encoded = self.mlp(features)
        if source_mask is not None:
            encoded = encoded.masked_fill(source_mask[:, None, :, None], 0.0)
        return encoded.sum(dim=2)


class BoundaryGeometryAdapter(nn.Module):
    """Shared encoding of arbitrary straight Robin/Dirichlet edge segments.

    Each descriptor is ``[x0,y0,x1,y1,nx,ny,log_gamma,is_robin]``.  A polygon
    boundary can therefore be represented by its line segments without adding
    board-specific weights to the universal operator.
    """

    def __init__(self, d_model: int, hidden: int = 48):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, queries, boundaries, boundary_mask=None):
        if boundaries is None or boundaries.shape[1] == 0:
            return queries.new_zeros((*queries.shape[:2], self.mlp[-1].out_features))
        p0, p1 = boundaries[..., :2], boundaries[..., 2:4]
        normal = boundaries[..., 4:6]
        log_gamma, robin = boundaries[..., 6], boundaries[..., 7]
        tangent = p1 - p0
        length = tangent.square().sum(dim=-1).sqrt().clamp_min(1e-4)
        tangent_unit = tangent / length[..., None]
        delta = queries[:, :, None, :] - p0[:, None, :, :]
        along = (delta * tangent_unit[:, None, :, :]).sum(dim=-1)
        normal_distance = (delta * normal[:, None, :, :]).sum(dim=-1)
        centered = (along / length[:, None, :] - 0.5) * 2.0
        segment_gate = torch.sigmoid((1.0 - centered.abs()) / 0.08)
        features = torch.stack([
            normal_distance.clamp(-1.5, 1.5),
            centered.clamp(-3.0, 3.0),
            segment_gate,
            (-normal_distance.abs() / 0.12).exp() * segment_gate,
            normal[:, None, :, 0].expand_as(along),
            normal[:, None, :, 1].expand_as(along),
            log_gamma[:, None, :].expand_as(along),
            robin[:, None, :].expand_as(along),
            length[:, None, :].expand_as(along),
        ], dim=-1)
        encoded = self.mlp(features)
        if boundary_mask is not None:
            encoded = encoded.masked_fill(boundary_mask[:, None, :, None], 0.0)
        return encoded.sum(dim=2)


class AttentionBlock(nn.Module):
    """Pre-norm transformer block; self-attn if cross=False else cross-attn."""

    def __init__(self, d_model, n_heads, ff_mult=4, cross=False):
        super().__init__()
        self.cross = cross
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model) if cross else None
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x, ctx=None, mask=None):
        q = self.ln_q(x)
        if self.cross:
            kv = self.ln_kv(ctx)
            a, _ = self.attn(q, kv, kv, key_padding_mask=mask)
        else:
            a, _ = self.attn(q, q, q, key_padding_mask=mask)
        x = x + a
        return x + self.ffn(self.ln2(x))


class ThermalOperatorV2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.sens_enc = SensorEncoder(cfg.d_model, cfg.n_fourier, cfg.fourier_sigma)
        self.query_enc = QueryEncoder(cfg.d_model, cfg.n_fourier, cfg.fourier_sigma)
        self.material_adapter = MaterialGeometryAdapter(
            cfg.d_model, cfg.material_hidden
        )
        self.source_adapter = SourceGeometryAdapter(
            cfg.d_model, cfg.source_hidden
        )
        self.boundary_adapter = BoundaryGeometryAdapter(
            cfg.d_model, cfg.boundary_hidden
        )
        self.cond_embed = nn.Linear(cfg.cond_dim, cfg.d_model)
        self.type_embed = nn.Embedding(cfg.n_token_types, cfg.d_model)
        self.self_blocks = nn.ModuleList(
            [AttentionBlock(cfg.d_model, cfg.n_heads, cross=False) for _ in range(cfg.n_self)]
        )
        self.cross_blocks = nn.ModuleList(
            [AttentionBlock(cfg.d_model, cfg.n_heads, cross=True) for _ in range(cfg.n_cross)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, sensors, queries, cond=None, sensor_mask=None,
                context=None, context_state=None, context_mask=None,
                cond_mask=None, board_tokens=None, materials=None,
                material_mask=None, sources=None, source_mask=None,
                boundaries=None, boundary_mask=None):
        """sensors (B,K,3) · queries (B,Q,2) · cond (B,4) or None.

        v2.5 extras (all optional; None ≡ v2 behaviour):
          context (B,M,3) reference-frame points (x, y, θ_ref/s_ctx),
          context_state (B,M) long 0/1 = which reference frame,
          context_mask (B,M) bool True=pad,
          cond_mask (B,) bool True=drop the condition token for that sample,
          board_tokens (B,N,d_model) learned per-board memory (M7 Tier-1
            adaptation) — enters like encoded context (context type embedding).
        Masks: True = padded/ignored (nn.MultiheadAttention convention).
        """
        B = sensors.shape[0]
        dev = sensors.device
        tokens = [self.sens_enc(sensors) + self.type_embed.weight[0]]
        masks = [sensor_mask if sensor_mask is not None
                 else torch.zeros(B, sensors.shape[1], dtype=torch.bool, device=dev)]

        if context is not None and context.shape[1] > 0:
            ce = self.sens_enc(context)
            state = (context_state if context_state is not None
                     else torch.zeros(B, context.shape[1], dtype=torch.long, device=dev))
            ce = ce + self.type_embed(state.clamp(0, 1) + 1)  # types 1, 2
            tokens.append(ce)
            masks.append(context_mask if context_mask is not None
                         else torch.zeros(B, context.shape[1], dtype=torch.bool, device=dev))

        if board_tokens is not None and board_tokens.shape[1] > 0:
            tokens.append(board_tokens + self.type_embed.weight[1])
            masks.append(torch.zeros(B, board_tokens.shape[1], dtype=torch.bool, device=dev))

        if cond is not None:
            ct = self.cond_embed(cond).unsqueeze(1) + self.type_embed.weight[3]
            tokens.append(ct)
            masks.append(cond_mask.unsqueeze(1) if cond_mask is not None
                         else torch.zeros(B, 1, dtype=torch.bool, device=dev))

        s = torch.cat(tokens, dim=1)
        full_mask = torch.cat(masks, dim=1)
        for blk in self.self_blocks:
            s = blk(s, mask=full_mask)
        q = self.query_enc(queries)
        q = q + self.material_adapter(queries, materials, material_mask)
        q = q + self.source_adapter(queries, sources, source_mask)
        q = q + self.boundary_adapter(queries, boundaries, boundary_mask)
        for blk in self.cross_blocks:
            q = blk(q, ctx=s, mask=full_mask)
        return self.head(q).squeeze(-1)  # (B, Q) — θ units

    @torch.no_grad()
    def board_embedding(self, context, context_state=None, context_mask=None):
        """Pooled board token from context points alone (visualization/analysis)."""
        B = context.shape[0]
        dev = context.device
        state = (context_state if context_state is not None
                 else torch.zeros(B, context.shape[1], dtype=torch.long, device=dev))
        s = self.sens_enc(context) + self.type_embed(state.clamp(0, 1) + 1)
        mask = (context_mask if context_mask is not None
                else torch.zeros(B, context.shape[1], dtype=torch.bool, device=dev))
        for blk in self.self_blocks:
            s = blk(s, mask=mask)
        w = (~mask).float().unsqueeze(-1)
        return (s * w).sum(1) / w.sum(1).clamp(min=1.0)  # (B, d_model)


def save_checkpoint(path: Path | str, model: ThermalOperatorV2, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "model_config": asdict(model.cfg),
         **(extra or {})},
        path,
    )


def load_checkpoint(path: Path | str, map_location="cpu") -> tuple[ThermalOperatorV2, dict]:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = ThermalOperatorV2(ModelConfig(**ckpt["model_config"]))
    incompatible = model.load_state_dict(ckpt["model_state"], strict=False)
    allowed_missing = {
        name for name in model.state_dict()
        if name.startswith(("material_adapter.", "source_adapter.", "boundary_adapter."))
    }
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint incompatibility: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model, ckpt
