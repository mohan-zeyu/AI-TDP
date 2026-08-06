"""Tier-1 fine-tuning building blocks."""

import numpy as np
import torch

from tdp.data.patches import make_validation_patches, patch_pixels
from tdp.model.operator import ModelConfig, ThermalOperatorV2
from tdp.train.finetune import BoardCard, material_edge_pair_pool, material_pixel_pool


def test_board_tokens_change_prediction_and_grads_flow():
    torch.manual_seed(0)
    model = ThermalOperatorV2(ModelConfig(d_model=32, n_fourier=8)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    card = BoardCard(4, 32)
    sensors = torch.randn(1, 5, 3)
    queries = torch.rand(1, 7, 2)
    a = model(sensors, queries, card.cond(0.656), board_tokens=card.tokens)
    b = model(sensors, queries, card.cond(0.656))
    assert not torch.allclose(a, b)
    loss = (a**2).mean()
    loss.backward()
    assert card.tokens.grad is not None and card.tokens.grad.abs().sum() > 0
    assert card.log_h.grad is not None
    n_train = sum(p.numel() for p in card.parameters())
    assert n_train < 2000  # the whole point


def test_validation_patches_respect_constraints():
    rng = np.random.default_rng(0)
    field = 30 + 40 * rng.random((80, 60))
    trusted = np.ones((80, 60), dtype=bool)
    trusted[10:20, 10:20] = False
    sites = np.array([[40.0, 30.0], [10.0, 50.0]])
    patches = make_validation_patches(field, trusted, sites, n_patches=10)
    assert len(patches) >= 8
    for r, c in patches:
        assert trusted[r, c]
        assert all(np.hypot(r - sr, c - sc) >= 5.0 for sr, sc in sites)
    mask = patch_pixels(patches, field.shape)
    assert mask.sum() <= len(patches) * 9


def test_board_card_keeps_geometry_fixed_and_material_values_trainable():
    material = torch.tensor([[
        0.30, 0.60, 0.08, 0.07, 0.0, 1.0,
        np.log(1.8), 0.0, np.log1p(0.01), 0.0,
    ]], dtype=torch.float32)
    boundary = torch.tensor([[
        0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0,
    ]], dtype=torch.float32)
    source = torch.tensor([[
        0.31, 0.59, 0.04, 0.03, 0.0, 1.0, np.log(0.8), 0.0,
    ]], dtype=torch.float32)
    card = BoardCard(24, 128, material_descriptors=material,
                     source_descriptors=source,
                     boundary_descriptors=boundary)
    assert not card.material_geometry.requires_grad
    assert not card.boundary_geometry.requires_grad
    assert not card.source_geometry.requires_grad
    assert card.materials().shape == (1, 1, 10)
    assert card.boundaries().shape == (1, 1, 8)
    assert card.sources().shape == (1, 1, 8)
    loss = (card.materials()[..., 6:9].sum()
            + card.sources()[..., 6].sum()
            + card.boundaries()[..., 6].sum())
    loss.backward()
    assert card.material_log_k.grad is not None
    assert card.material_log_h.grad is not None
    assert card.material_raw_log1p_rc.grad is not None
    assert card.source_log_amp.grad is not None
    assert card.boundary_log_gamma_delta.grad is not None


def test_explicit_material_shape_and_edge_pools_respect_allowed_mask():
    descriptor = np.array([
        0.30, 0.60, 0.08, 0.07, 0.0, 1.0,
        np.log(1.8), 0.0, np.log1p(0.01), 0.0,
    ])
    shape, aspect = (101, 67), 0.66
    allowed = np.ones(shape, dtype=bool)
    allowed[:5] = False
    pixels = material_pixel_pool(descriptor, shape, aspect, allowed)
    inside, outside = material_edge_pair_pool(
        descriptor, shape, aspect, allowed, points_per_edge=24
    )
    assert len(pixels) > 20
    assert len(inside) == len(outside) > 20
    assert np.all((inside[:, 0] >= 0) & (inside[:, 0] <= aspect))
    assert np.all((outside[:, 1] >= 0) & (outside[:, 1] <= 1))
