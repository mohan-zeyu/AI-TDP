"""Loss terms: data MSE + autograd PDE residual in nondimensional coordinates.

Residual of the normalized field θ' = θ/s:  R = ∇̃²θ' − ĥ·θ' + Q̂/s.
Each sample's residual is weighted by 1/(1+ĥ) so gradient magnitudes stay
comparable across the randomized ĥ range.
"""

from __future__ import annotations

import torch


def pde_residual(predict_fn, colloc, q_over_s, h_hat):
    """Relative residual at collocation points: R / (|∇²θ'| + |ĥθ'| + |Q̂/s| + 1).

    The self-normalizing denominator (detached) keeps the loss O(1) across the
    randomized ĥ range and across training stages, so λ_pde acts as a genuine
    light-touch regularizer rather than silently rebalancing the objective.
    predict_fn: coords (B,M,2) → θ (B,M) with the sample's full conditioning
    (sensors/context/cond) closed over. colloc (B,M,2) · q_over_s (B,M) · h_hat (B,).
    """
    colloc = colloc.detach().requires_grad_(True)
    theta = predict_fn(colloc)  # (B, M)
    g1 = torch.autograd.grad(theta.sum(), colloc, create_graph=True)[0]  # (B, M, 2)
    txx = torch.autograd.grad(g1[..., 0].sum(), colloc, create_graph=True)[0][..., 0]
    tyy = torch.autograd.grad(g1[..., 1].sum(), colloc, create_graph=True)[0][..., 1]
    lap = txx + tyy
    sink = h_hat[:, None] * theta
    residual = lap - sink + q_over_s
    denom = (lap.abs() + sink.abs() + q_over_s.abs() + 1.0).detach()
    return residual / denom


def pde_loss(residual: torch.Tensor) -> torch.Tensor:
    return (residual**2).mean()


def probe_double_backward(model, device) -> bool:
    """Check that second-order autograd works on this device (MPS is flaky)."""
    try:
        sensors = torch.zeros(1, 3, 3, device=device)
        cond = torch.zeros(1, model.cfg.cond_dim, device=device)
        colloc = torch.rand(1, 4, 2, device=device)
        q = torch.zeros(1, 4, device=device)
        h = torch.ones(1, device=device)
        r = pde_residual(lambda xy: model(sensors, xy, cond), colloc, q, h)
        loss = pde_loss(r)
        torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad][:1],
                            allow_unused=True)
        return True
    except Exception:
        return False
