"""ThermalOperatorV2 — the v1 attention operator (ported verbatim from
AI_TDP.ipynb) plus a single condition token carrying [log ĥ, log γ, aspect, BC].

The token is appended to the encoded sensor set (always valid in the padding
mask), so permutation invariance over sensors and variable K are untouched.
I/O is fully nondimensional (see tdp.model.normalization).
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
        self.cond_embed = nn.Linear(cfg.cond_dim, cfg.d_model)
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

    def forward(self, sensors, queries, cond, sensor_mask=None):
        """sensors (B,K,3) · queries (B,Q,2) · cond (B,4) · sensor_mask (B,K) True=pad."""
        s = self.sens_enc(sensors)
        c = self.cond_embed(cond).unsqueeze(1)  # (B, 1, d)
        s = torch.cat([s, c], dim=1)
        if sensor_mask is not None:
            pad = torch.zeros(sensor_mask.shape[0], 1, dtype=torch.bool,
                              device=sensor_mask.device)
            sensor_mask = torch.cat([sensor_mask, pad], dim=1)  # token always valid
        for blk in self.self_blocks:
            s = blk(s, mask=sensor_mask)
        q = self.query_enc(queries)
        for blk in self.cross_blocks:
            q = blk(q, ctx=s, mask=sensor_mask)
        return self.head(q).squeeze(-1)  # (B, Q) — θ units


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
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt
