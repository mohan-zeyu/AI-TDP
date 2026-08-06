"""v2.5 in-context conditioning: multi-state boards, context tokens, masking."""

import numpy as np
import torch

from tdp.model.operator import ModelConfig, ThermalOperatorV2
from tdp.sim.fdm import solve_steady
from tdp.sim.scenarios import ScenarioConfig, random_board, sample_context
from tdp.train.pretrain import OperatorDataset, TrainConfig, collate


def test_board_states_share_layout_differ_in_amps():
    rng = np.random.default_rng(3)
    board = random_board(rng, ScenarioConfig(ny=48, states_per_board=3))
    assert len(board.states) == 3
    s0, s1 = board.states[0], board.states[1]
    assert s0.layout is s1.layout
    assert not np.allclose(s0.amps, s1.amps)
    assert s0.source_descriptors().shape[1] == 8
    assert not np.allclose(s0.source_descriptors()[:, 6],
                           s1.source_descriptors()[:, 6])
    assert not np.allclose(s0.theta, s1.theta)


def test_lu_reuse_matches_fresh_solve():
    rng = np.random.default_rng(4)
    board = random_board(rng, ScenarioConfig(ny=48, states_per_board=2))
    st = board.states[0]
    lay = board.layout
    nx = st.theta.shape[1]
    from tdp.sim.fdm import grid_coords

    X, Y = grid_coords(st.theta.shape[0], nx, lay.aspect)
    q = np.zeros_like(X)
    for a, s in zip(st.amps, lay.sources):
        q += a * s.q_unit(X, Y)
    fresh = solve_steady(q, lay.aspect, lay.h_hat,
                         bc="robin" if lay.robin else "dirichlet",
                         gamma=lay.gamma if lay.robin else 0.0)
    np.testing.assert_allclose(st.theta, fresh, atol=1e-4)


def test_masked_context_equals_no_context():
    torch.manual_seed(0)
    model = ThermalOperatorV2(ModelConfig(d_model=32, n_fourier=8)).eval()
    B, K, Q, M = 2, 5, 7, 6
    sensors = torch.randn(B, K, 3)
    queries = torch.rand(B, Q, 2)
    cond = torch.randn(B, 4)
    ctx = torch.randn(B, M, 3)
    ctx_state = torch.zeros(B, M, dtype=torch.long)
    all_masked = torch.ones(B, M, dtype=torch.bool)
    with torch.no_grad():
        plain = model(sensors, queries, cond)
        masked = model(sensors, queries, cond, context=ctx,
                       context_state=ctx_state, context_mask=all_masked)
    torch.testing.assert_close(plain, masked, atol=1e-5, rtol=1e-4)


def test_context_changes_prediction():
    torch.manual_seed(0)
    model = ThermalOperatorV2(ModelConfig(d_model=32, n_fourier=8)).eval()
    sensors = torch.randn(1, 4, 3)
    queries = torch.rand(1, 9, 2)
    cond = torch.randn(1, 4)
    ctx = torch.randn(1, 12, 3)
    st = torch.zeros(1, 12, dtype=torch.long)
    with torch.no_grad():
        a = model(sensors, queries, cond)
        b = model(sensors, queries, cond, context=ctx, context_state=st)
    assert not torch.allclose(a, b)


def test_source_adapter_is_backward_compatible_and_trainable():
    torch.manual_seed(7)
    model = ThermalOperatorV2(ModelConfig(d_model=32, n_fourier=8))
    sensors = torch.randn(1, 3, 3)
    queries = torch.rand(1, 11, 2)
    cond = torch.randn(1, 4)
    sources = torch.tensor([[[
        0.35, 0.55, 0.08, 0.05, 0.0, 1.0, 0.0, 0.0,
    ]]])
    plain = model(sensors, queries, cond)
    conditioned = model(sensors, queries, cond, sources=sources)
    torch.testing.assert_close(plain, conditioned)
    conditioned.square().mean().backward()
    final = model.source_adapter.mlp[-1]
    assert final.weight.grad is not None and final.weight.grad.abs().sum() > 0


def test_dataset_and_collate_shapes():
    rng_cfg = ScenarioConfig(ny=32, states_per_board=3)
    boards = [random_board(np.random.default_rng(i), rng_cfg) for i in range(4)]
    tcfg = TrainConfig(n_query=16, n_colloc=8, n_context=(8, 12),
                       context_dropout=0.5, k_min_ctx=2)
    ds = OperatorDataset(boards, tcfg, layout=None, base_seed=0)
    batch = collate([ds[i] for i in range(len(boards))])
    (sensors, s_mask, q_xy, q_theta, c_xy, q_over_s,
     cond, cond_drop, h_hat, conductivity, sink_multiplier,
     b_xy, b_normal, b_conductivity, b_gamma, b_robin,
     ctx, ctx_state, ctx_mask, materials, material_mask,
     sources, source_mask, boundaries, boundary_mask) = batch
    B = len(boards)
    assert sensors.shape[0] == B and sensors.shape[2] == 3
    assert s_mask.dtype == torch.bool
    assert q_xy.shape == (B, 16, 2) and q_theta.shape == (B, 16)
    assert cond.shape == (B, 4) and h_hat.shape == (B,)
    assert conductivity.shape == (B, 8)
    assert sink_multiplier.shape == (B, 8)
    assert b_xy.shape == (B, 64, 2)
    assert b_gamma.shape == b_robin.shape == (B, 64)
    assert materials.shape[0] == B and materials.shape[2] == 10
    assert sources.shape[0] == B and sources.shape[2] == 8
    assert source_mask.dtype == torch.bool
    assert boundaries.shape[0] == B and boundaries.shape[2] == 8
    if ctx is not None:
        assert ctx.shape[0] == B and ctx.shape[2] == 3
        assert ctx_state.shape == ctx.shape[:2] == ctx_mask.shape
    # forward pass with the batch runs
    model = ThermalOperatorV2(ModelConfig(d_model=32, n_fourier=8))
    out = model(sensors, q_xy, cond, sensor_mask=s_mask, context=ctx,
                context_state=ctx_state, context_mask=ctx_mask, cond_mask=cond_drop,
                materials=materials, material_mask=material_mask,
                sources=sources, source_mask=source_mask,
                boundaries=boundaries, boundary_mask=boundary_mask)
    assert out.shape == (B, 16)


def test_sample_context_normalized():
    rng = np.random.default_rng(5)
    board = random_board(rng, ScenarioConfig(ny=32, states_per_board=1))
    pts, state = sample_context(board.states[0], rng, 20, frame_idx=1)
    assert pts.shape == (20, 3)
    assert state.tolist() == [1] * 20
    assert pts[:, 2].max() <= 1.0 + 1e-6
