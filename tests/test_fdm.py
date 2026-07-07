"""FDM solver validation: manufactured solution, convergence order, physics balances."""

import numpy as np
import pytest

from tdp.sim.fdm import bilinear, grid_coords, solve_steady


def dirichlet_error(ny: int, aspect: float = 0.7, h_hat: float = 4.0) -> float:
    """Manufactured solution θ* = sin(πx/a)·sin(πy) → exact Q̂ = (ĥ + π²(1/a²+1))·θ*."""
    nx = max(int(round(ny * aspect)), 12)
    X, Y = grid_coords(ny, nx, aspect)
    theta_exact = np.sin(np.pi * X / aspect) * np.sin(np.pi * Y)
    q = (h_hat + np.pi**2 * (1 / aspect**2 + 1)) * theta_exact
    theta = solve_steady(q, aspect, h_hat, bc="dirichlet")
    return float(np.abs(theta - theta_exact).max())


def test_dirichlet_manufactured_convergence_order():
    e1, e2 = dirichlet_error(33), dirichlet_error(65)
    assert e1 < 5e-3
    order = np.log2(e1 / e2)
    assert 1.6 < order < 2.4  # O(h²)


def test_robin_large_gamma_approaches_dirichlet():
    ny, aspect, h_hat = 65, 0.7, 4.0
    nx = max(int(round(ny * aspect)), 12)
    X, Y = grid_coords(ny, nx, aspect)
    q = 200.0 * np.exp(-((X - 0.35) ** 2 + (Y - 0.5) ** 2) / (2 * 0.08**2))
    th_d = solve_steady(q, aspect, h_hat, bc="dirichlet")
    th_r = solve_steady(q, aspect, h_hat, bc="robin", gamma=1e5)
    assert np.abs(th_d - th_r).max() < 0.02 * th_d.max()


def test_robin_insulated_global_energy_balance():
    """γ=0 (insulated edges): all heat leaves via ĥθ → ∫Q̂ = ĥ∫θ."""
    ny, aspect, h_hat = 97, 0.66, 2.5
    nx = max(int(round(ny * aspect)), 12)
    X, Y = grid_coords(ny, nx, aspect)
    q = 150.0 * np.exp(-((X - 0.3) ** 2 + (Y - 0.6) ** 2) / (2 * 0.1**2))
    theta = solve_steady(q, aspect, h_hat, bc="robin", gamma=0.0)
    dx, dy = aspect / (nx - 1), 1.0 / (ny - 1)
    # trapezoid weights on the tensor grid
    wx = np.full(nx, dx); wx[[0, -1]] /= 2
    wy = np.full(ny, dy); wy[[0, -1]] /= 2
    W = np.outer(wy, wx)
    assert abs((q * W).sum() - h_hat * (theta * W).sum()) < 0.01 * (q * W).sum()


def test_theta_positive_and_bounded():
    rng = np.random.default_rng(0)
    from tdp.sim.scenarios import ScenarioConfig, random_scenario

    for _ in range(3):
        scn = random_scenario(rng, ScenarioConfig(ny=48))
        assert scn.theta.min() > -1e-8
        assert np.isfinite(scn.theta).all()
        assert scn.theta.max() > 0


def test_bilinear_matches_grid_nodes():
    ny, nx, aspect = 33, 23, 0.7
    rng = np.random.default_rng(1)
    grid = rng.normal(size=(ny, nx))
    X, Y = grid_coords(ny, nx, aspect)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    np.testing.assert_allclose(bilinear(grid, pts, aspect), grid.ravel(), atol=1e-9)
