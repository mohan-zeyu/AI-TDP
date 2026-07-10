"""Tier-1 fine-tuning building blocks."""

import numpy as np
import torch

from tdp.data.patches import make_validation_patches, patch_pixels
from tdp.model.operator import ModelConfig, ThermalOperatorV2
from tdp.train.finetune import BoardCard


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
