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


def solve_steady(
    q_hat: np.ndarray,
    aspect: float,
    h_hat: float,
    bc: str = "robin",
    gamma: float = 1.0,
) -> np.ndarray:
    """Solve the nondimensional steady equation for a gridded source q_hat (ny, nx)."""
    ny, nx = q_hat.shape
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
        rhs = np.where(boundary, 0.0, -q_hat).ravel()
    elif bc == "robin":
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)
        rhs = (-q_hat).ravel()
    else:
        raise ValueError(f"unknown bc {bc!r}")

    rows = np.concatenate([rows, np.arange(N)])
    cols = np.concatenate([cols, np.arange(N)])
    vals = np.concatenate([vals, diag.ravel()])
    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
    theta = splu(A).solve(rhs)
    return theta.reshape(ny, nx)


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
