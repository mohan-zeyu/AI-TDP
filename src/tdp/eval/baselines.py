"""Baselines. Every baseline receives exactly the same sensor sets as our model.

- bicubic: scipy griddata (cubic inside the hull, nearest outside) — the naive floor
- rbf:     thin-plate-spline RBF — the standard classical answer
- gp:      Gaussian process (C·RBF + White, fitted lengthscale) — the strong classical answer
- pinn:    a fresh coordinate-MLP PINN optimized per case under a wall-clock
           budget — physics-informed but *without* pretraining
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


def bicubic_interp(s_xy: np.ndarray, s_vals: np.ndarray, q_xy: np.ndarray) -> np.ndarray:
    from scipy.interpolate import griddata

    cubic = griddata(s_xy, s_vals, q_xy, method="cubic")
    nearest = griddata(s_xy, s_vals, q_xy, method="nearest")
    return np.where(np.isnan(cubic), nearest, cubic)


def rbf_interp(s_xy: np.ndarray, s_vals: np.ndarray, q_xy: np.ndarray) -> np.ndarray:
    from scipy.interpolate import RBFInterpolator

    if len(s_vals) < 3:  # thin-plate needs >= 3 points in 2-D
        return RBFInterpolator(s_xy, s_vals, kernel="linear", degree=0)(q_xy)
    return RBFInterpolator(s_xy, s_vals, kernel="thin_plate_spline")(q_xy)


def gp_interp(s_xy: np.ndarray, s_vals: np.ndarray, q_xy: np.ndarray) -> np.ndarray:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    kernel = (ConstantKernel(1.0, (1e-2, 1e3))
              * RBF(0.3, (5e-2, 2.0))
              + WhiteKernel(1e-3, (1e-6, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
                                  random_state=0)
    gp.fit(s_xy, s_vals)
    return gp.predict(q_xy)


class _PinnMLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy).squeeze(-1)


def pinn_percase(
    s_xy: np.ndarray,
    s_vals: np.ndarray,
    q_xy: np.ndarray,
    aspect: float,
    colloc_sampler,          # rng, n -> (n, 2) source-free points
    budget_s: float = 60.0,
    h_hat: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Per-case PINN, time-budget-matched to our board-card fit (~1 min).

    Trains a Tanh MLP on the K sensor readings plus the source-free residual
    ∇²θ − ĥθ = 0 outside the annotated component rectangles — i.e. the same
    physics information our fine-tune uses, but no pretrained prior.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mu, sd = float(s_vals.mean()), float(max(s_vals.std(), 1.0))
    xy_t = torch.from_numpy(s_xy.astype(np.float32))
    y_t = torch.from_numpy(((s_vals - mu) / sd).astype(np.float32))

    net = _PinnMLP()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0 = time.time()
    while time.time() - t0 < budget_s:
        c_np = colloc_sampler(rng, 256).astype(np.float32)
        c = torch.from_numpy(c_np).requires_grad_(True)
        theta = net(c)
        g = torch.autograd.grad(theta.sum(), c, create_graph=True)[0]
        txx = torch.autograd.grad(g[:, 0].sum(), c, create_graph=True)[0][:, 0]
        tyy = torch.autograd.grad(g[:, 1].sum(), c, create_graph=True)[0][:, 1]
        pde = ((txx + tyy - h_hat * theta) ** 2).mean() / (1.0 + h_hat) ** 2
        data = ((net(xy_t) - y_t) ** 2).mean()
        loss = 10.0 * data + 1e-3 * pde
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(torch.from_numpy(q_xy.astype(np.float32))).numpy()
    return pred * sd + mu
