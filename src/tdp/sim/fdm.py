"""Nondimensional steady-heat FDM solver on rectangular domains.

Solves  ∇²θ − ĥ·θ + Q̂ = 0  on [0, a] × [0, 1] with either
  - Dirichlet θ = 0 on the boundary, or
  - Robin −∂θ/∂n = γ·θ (convective edge) via ghost-node elimination.

Assembly is fully vectorized (COO triplets), so a fresh system per randomized
scenario costs milliseconds at the default 96-row grid.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def grid_coords(ny: int, nx: int, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """Node coordinates: X, Y arrays of shape (ny, nx); y ∈ [0,1], x ∈ [0,aspect]."""
    x = np.linspace(0.0, aspect, nx)
    y = np.linspace(0.0, 1.0, ny)
    return np.meshgrid(x, y)


def factorize_steady(
    ny: int,
    nx: int,
    aspect: float,
    h_hat: float,
    bc: str = "robin",
    gamma: float = 1.0,
):
    """Assemble and LU-factorize the steady operator once; reuse across source
    fields of the same board (multi-state generation shares one factorization)."""
    dx = aspect / (nx - 1)
    dy = 1.0 / (ny - 1)
    cx, cy = 1.0 / dx**2, 1.0 / dy**2
    N = ny * nx
    idx = np.arange(N).reshape(ny, nx)

    diag = np.full((ny, nx), -2.0 * cx - 2.0 * cy - h_hat)
    rows_list, cols_list, vals_list = [], [], []

    # (row_slice_of_source, col_slice_of_source, neighbor_shift, coeff,
    #  edge_row_slice, edge_col_slice, opposite_shift, face_spacing)
    # For nodes NOT on a face: plain neighbor coefficient. For nodes ON a face
    # (Robin): ghost elimination adds coeff to the OPPOSITE neighbor and
    # −2γ·spacing·coeff to the diagonal.
    faces = [
        # left neighbor (i-1): missing on col 0
        ((slice(None), slice(1, None)), (0, -1), cx, (slice(None), slice(0, 1)), (0, +1), dx),
        # right neighbor (i+1): missing on col nx-1
        ((slice(None), slice(0, -1)), (0, +1), cx, (slice(None), slice(nx - 1, nx)), (0, -1), dx),
        # up neighbor (j-1): missing on row 0
        ((slice(1, None), slice(None)), (-1, 0), cy, (slice(0, 1), slice(None)), (+1, 0), dy),
        # down neighbor (j+1): missing on row ny-1
        ((slice(0, -1), slice(None)), (+1, 0), cy, (slice(ny - 1, ny), slice(None)), (-1, 0), dy),
    ]

    for interior_sl, shift, coeff, edge_sl, opp_shift, spacing in faces:
        src = idx[interior_sl]
        nbr = np.roll(idx, (-shift[0], -shift[1]), axis=(0, 1))[interior_sl]
        rows_list.append(src.ravel())
        cols_list.append(nbr.ravel())
        vals_list.append(np.full(src.size, coeff))

        if bc == "robin":
            e_src = idx[edge_sl]
            e_opp = np.roll(idx, (-opp_shift[0], -opp_shift[1]), axis=(0, 1))[edge_sl]
            rows_list.append(e_src.ravel())
            cols_list.append(e_opp.ravel())
            vals_list.append(np.full(e_src.size, coeff))
            diag[edge_sl] += -2.0 * gamma * spacing * coeff

    if bc == "dirichlet":
        boundary = np.zeros((ny, nx), dtype=bool)
        boundary[0, :] = boundary[-1, :] = True
        boundary[:, 0] = boundary[:, -1] = True
        diag[boundary] = 1.0
        keep = ~boundary.ravel()[np.concatenate(rows_list)]
        rows = np.concatenate(rows_list)[keep]
        cols = np.concatenate(cols_list)[keep]
        vals = np.concatenate(vals_list)[keep]
    elif bc == "robin":
        boundary = None
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)
    else:
        raise ValueError(f"unknown bc {bc!r}")

    rows = np.concatenate([rows, np.arange(N)])
    cols = np.concatenate([cols, np.arange(N)])
    vals = np.concatenate([vals, diag.ravel()])
    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
    return splu(A), boundary


def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Face conductivity for a discontinuous, cell-wise material field."""
    return 2.0 * a * b / np.maximum(a + b, np.finfo(np.float64).tiny)


def factorize_steady_variable_k(
    conductivity: np.ndarray,
    aspect: float,
    h_hat: float | np.ndarray,
    bc: str = "robin",
    gamma: float | dict[str, np.ndarray] = 1.0,
    contact_resistance: tuple[np.ndarray, np.ndarray] | None = None,
):
    """Factorize ``div(k grad(theta)) - h_hat*theta`` on a tensor grid.

    ``conductivity`` is relative in-plane conductivity at grid nodes.  The
    reference PCB material has k=1.  Harmonic means are used on cell faces,
    which is the conservative choice at abrupt material interfaces and avoids
    the excessive heat leakage produced by arithmetic averaging.

    The Robin discretization reduces exactly to :func:`factorize_steady` for
    k=1.  ``h_hat`` may be scalar or a spatial field, allowing a later model to
    distinguish in-plane conduction from material-dependent vertical loss.
    """
    k = np.asarray(conductivity, dtype=np.float64)
    if k.ndim != 2 or min(k.shape) < 3:
        raise ValueError("conductivity must be a 2-D grid with both dimensions >= 3")
    if not np.isfinite(k).all() or np.any(k <= 0):
        raise ValueError("conductivity must contain finite positive values")

    ny, nx = k.shape
    dx = aspect / (nx - 1)
    dy = 1.0 / (ny - 1)
    N = ny * nx
    idx = np.arange(N).reshape(ny, nx)
    sink = np.broadcast_to(np.asarray(h_hat, dtype=np.float64), k.shape)
    if not np.isfinite(sink).all() or np.any(sink < 0):
        raise ValueError("h_hat must contain finite non-negative values")

    diag = -sink.copy()
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    vals_list: list[np.ndarray] = []

    def add_links(left: np.ndarray, right: np.ndarray, coeff: np.ndarray) -> None:
        rows_list.extend([left.ravel(), right.ravel()])
        cols_list.extend([right.ravel(), left.ravel()])
        vals_list.extend([coeff.ravel(), coeff.ravel()])

    # Conservative interior fluxes.  Each face contributes +g*T_neighbor and
    # -g*T_self to its two adjacent node equations.
    if contact_resistance is None:
        rc_x = np.zeros((ny, nx - 1), dtype=np.float64)
        rc_y = np.zeros((ny - 1, nx), dtype=np.float64)
    else:
        rc_x, rc_y = (np.asarray(v, dtype=np.float64) for v in contact_resistance)
        if rc_x.shape != (ny, nx - 1) or rc_y.shape != (ny - 1, nx):
            raise ValueError("contact_resistance face arrays have incompatible shapes")
        if np.any(rc_x < 0) or np.any(rc_y < 0):
            raise ValueError("contact resistance must be non-negative")

    # Series resistance across a face: half a cell of each material plus the
    # grid-independent dimensionless interface resistance Rc. Rc=0 reduces
    # exactly to the harmonic mean and refinement does not redefine Rc.
    gx = 1.0 / (dx * (0.5 * dx / k[:, :-1] + rc_x + 0.5 * dx / k[:, 1:]))
    add_links(idx[:, :-1], idx[:, 1:], gx)
    diag[:, :-1] -= gx
    diag[:, 1:] -= gx

    gy = 1.0 / (dy * (0.5 * dy / k[:-1, :] + rc_y + 0.5 * dy / k[1:, :]))
    add_links(idx[:-1, :], idx[1:, :], gy)
    diag[:-1, :] -= gy
    diag[1:, :] -= gy

    if bc == "robin":
        # Ghost-node elimination doubles the inward face flux at a boundary.
        # These are the additional copies; the first copies are in add_links.
        if isinstance(gamma, dict):
            edge_gamma = (
                np.broadcast_to(gamma["left"], (ny,)),
                np.broadcast_to(gamma["right"], (ny,)),
                np.broadcast_to(gamma["top"], (nx,)),
                np.broadcast_to(gamma["bottom"], (nx,)),
            )
        else:
            edge_gamma = tuple(np.full(n, float(gamma)) for n in (ny, ny, nx, nx))
        faces = (
            (idx[:, 0], idx[:, 1], gx[:, 0], dx, edge_gamma[0]),
            (idx[:, -1], idx[:, -2], gx[:, -1], dx, edge_gamma[1]),
            (idx[0, :], idx[1, :], gy[0, :], dy, edge_gamma[2]),
            (idx[-1, :], idx[-2, :], gy[-1, :], dy, edge_gamma[3]),
        )
        boundary = None
        for edge, inward, conductance, spacing, gamma_edge in faces:
            rows_list.append(edge.ravel())
            cols_list.append(inward.ravel())
            vals_list.append(conductance.ravel())
            edge_rows, edge_cols = np.unravel_index(edge, (ny, nx))
            diag[edge_rows, edge_cols] -= conductance
            # Physical Robin condition: -n·k∇theta = gamma*theta.  The edge
            # transfer coefficient is independent of the local in-plane k.
            # For k=1 this reduces to the legacy -2*gamma/spacing stencil.
            diag[edge_rows, edge_cols] -= 2.0 * gamma_edge / spacing
    elif bc == "dirichlet":
        boundary = np.zeros((ny, nx), dtype=bool)
        boundary[0, :] = boundary[-1, :] = True
        boundary[:, 0] = boundary[:, -1] = True
    else:
        raise ValueError(f"unknown bc {bc!r}")

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    if boundary is not None:
        keep = ~boundary.ravel()[rows]
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        diag[boundary] = 1.0

    rows = np.concatenate([rows, np.arange(N)])
    cols = np.concatenate([cols, np.arange(N)])
    vals = np.concatenate([vals, diag.ravel()])
    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
    return splu(A), boundary


def solve_with(lu, boundary: np.ndarray | None, q_hat: np.ndarray) -> np.ndarray:
    """Solve a factorized system for one gridded source field (ny, nx)."""
    ny, nx = q_hat.shape
    if boundary is None:  # robin
        rhs = (-q_hat).ravel()
    else:  # dirichlet
        rhs = np.where(boundary, 0.0, -q_hat).ravel()
    return lu.solve(rhs).reshape(ny, nx)


def solve_steady(
    q_hat: np.ndarray,
    aspect: float,
    h_hat: float,
    bc: str = "robin",
    gamma: float = 1.0,
) -> np.ndarray:
    """Assemble + solve in one call (convenience for tests and one-off fields)."""
    ny, nx = q_hat.shape
    lu, boundary = factorize_steady(ny, nx, aspect, h_hat, bc, gamma)
    return solve_with(lu, boundary, q_hat)


def solve_steady_variable_k(
    q_hat: np.ndarray,
    conductivity: np.ndarray,
    aspect: float,
    h_hat: float | np.ndarray,
    bc: str = "robin",
    gamma: float | dict[str, np.ndarray] = 1.0,
    contact_resistance: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Solve one heterogeneous-material steady field."""
    q_hat = np.asarray(q_hat)
    if q_hat.shape != np.asarray(conductivity).shape:
        raise ValueError("q_hat and conductivity must have identical shapes")
    lu, boundary = factorize_steady_variable_k(
        conductivity, aspect, h_hat, bc=bc, gamma=gamma,
        contact_resistance=contact_resistance,
    )
    return solve_with(lu, boundary, q_hat)


def bilinear(grid: np.ndarray, xy: np.ndarray, aspect: float) -> np.ndarray:
    """Bilinear interpolation of a (ny, nx) grid at nondimensional points (…, 2)."""
    ny, nx = grid.shape
    gx = np.clip(xy[..., 0] / aspect, 0.0, 1.0) * (nx - 1)
    gy = np.clip(xy[..., 1], 0.0, 1.0) * (ny - 1)
    i0 = np.clip(np.floor(gx).astype(np.int64), 0, nx - 2)
    j0 = np.clip(np.floor(gy).astype(np.int64), 0, ny - 2)
    fx, fy = gx - i0, gy - j0
    return ((1 - fx) * (1 - fy) * grid[j0, i0] + fx * (1 - fy) * grid[j0, i0 + 1]
            + (1 - fx) * fy * grid[j0 + 1, i0] + fx * fy * grid[j0 + 1, i0 + 1])
