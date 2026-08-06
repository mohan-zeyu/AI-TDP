"""FDM solver validation: manufactured solution, convergence order, physics balances."""

import numpy as np
import pytest

from tdp.sim.fdm import bilinear, grid_coords, solve_steady, solve_steady_variable_k


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


@pytest.mark.parametrize("bc,gamma", [("dirichlet", 1.0), ("robin", 0.8)])
def test_variable_k_uniform_field_matches_legacy_solver(bc, gamma):
    ny, aspect, h_hat = 49, 0.66, 2.0
    nx = max(int(round(ny * aspect)), 12)
    X, Y = grid_coords(ny, nx, aspect)
    q = 100.0 * np.exp(-((X - 0.3) ** 2 + (Y - 0.55) ** 2) / (2 * 0.08**2))
    expected = solve_steady(q, aspect, h_hat, bc=bc, gamma=gamma)
    actual = solve_steady_variable_k(
        q, np.ones_like(q), aspect, h_hat, bc=bc, gamma=gamma
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_variable_k_regular_inclusion_changes_hotspot_shape():
    """A square material/source region must survive as a non-radial response."""
    ny, aspect, h_hat = 81, 0.66, 2.0
    nx = max(int(round(ny * aspect)), 12)
    X, Y = grid_coords(ny, nx, aspect)
    inside = (np.abs(X - 0.31) <= 0.10) & (np.abs(Y - 0.55) <= 0.15)
    q = np.where(inside, 120.0, 0.0)
    conductivity = np.where(inside, 0.20, 1.0)
    theta = solve_steady_variable_k(
        q, conductivity, aspect, h_hat, bc="robin", gamma=1.0
    )
    assert np.isfinite(theta).all()
    assert theta.min() > -1e-8
    # The 90%-of-peak set should inherit the rectangular region's anisotropy.
    hot = theta >= 0.90 * theta.max()
    rows, cols = np.nonzero(hot)
    assert np.ptp(rows) > 1.25 * np.ptp(cols)


def test_random_heterogeneous_board_has_material_map_and_finite_states():
    from tdp.sim.scenarios import ScenarioConfig, random_board

    cfg = ScenarioConfig(
        ny=40,
        heterogeneous_prob=1.0,
        n_material_regions=(2, 2),
        states_per_board=2,
    )
    board = random_board(np.random.default_rng(23), cfg)
    assert len(board.layout.materials) == 2
    xy = np.stack(np.meshgrid(
        np.linspace(0, board.layout.aspect, 40),
        np.linspace(0, 1, 40),
    ), axis=-1)
    conductivity = board.layout.conductivity_at(xy)
    assert np.any(np.abs(conductivity - 1.0) > 1e-6)
    assert all(np.isfinite(state.theta).all() for state in board.states)


def test_interface_contact_resistance_raises_enclosed_hotspot_temperature():
    ny, aspect = 61, 0.66
    nx = max(int(round(ny * aspect)), 12)
    X, Y = grid_coords(ny, nx, aspect)
    inside = (np.abs(X - 0.30) < 0.09) & (np.abs(Y - 0.55) < 0.14)
    q = np.where(inside, 100.0, 0.0)
    rc_x = (inside[:, :-1] != inside[:, 1:]).astype(float) * 0.02
    rc_y = (inside[:-1, :] != inside[1:, :]).astype(float) * 0.02
    no_contact = solve_steady_variable_k(
        q, np.ones_like(q), aspect, 2.0, bc="dirichlet"
    )
    with_contact = solve_steady_variable_k(
        q, np.ones_like(q), aspect, 2.0, bc="dirichlet",
        contact_resistance=(rc_x, rc_y),
    )
    assert with_contact.max() > 1.05 * no_contact.max()


def test_four_independent_robin_edges_break_left_right_symmetry():
    ny, aspect = 55, 0.72
    nx = max(int(round(ny * aspect)), 12)
    X, Y = grid_coords(ny, nx, aspect)
    q = 100.0 * np.exp(-((X - aspect / 2) ** 2 + (Y - 0.5) ** 2) / 0.02)
    gamma = {
        "left": np.full(ny, 20.0),
        "right": np.full(ny, 0.1),
        "top": np.ones(nx),
        "bottom": np.ones(nx),
    }
    theta = solve_steady_variable_k(
        q, np.ones_like(q), aspect, 2.0, bc="robin", gamma=gamma
    )
    assert np.isfinite(theta).all()
    assert not np.allclose(theta, theta[:, ::-1], rtol=1e-3, atol=1e-3)
