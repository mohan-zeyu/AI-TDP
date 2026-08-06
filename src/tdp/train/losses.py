"""Loss terms: data MSE + autograd PDE residual in nondimensional coordinates.

Residual of the normalized field θ' = θ/s:  R = ∇̃²θ' − ĥ·θ' + Q̂/s.
Each sample's residual is weighted by 1/(1+ĥ) so gradient magnitudes stay
comparable across the randomized ĥ range.
"""

from __future__ import annotations

import torch


def pde_residual(
    predict_fn,
    colloc,
    q_over_s,
    h_hat,
    conductivity=None,
    sink_multiplier=None,
):
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
    diffusion = lap if conductivity is None else conductivity * lap
    sink_coefficient = h_hat[:, None]
    if sink_multiplier is not None:
        sink_coefficient = sink_coefficient * sink_multiplier
    sink = sink_coefficient * theta
    residual = diffusion - sink + q_over_s
    denom = (diffusion.abs() + sink.abs() + q_over_s.abs() + 1.0).detach()
    return residual / denom


def pde_loss(residual: torch.Tensor) -> torch.Tensor:
    return (residual**2).mean()


def boundary_residual(
    predict_fn,
    points,
    normals,
    conductivity,
    gamma,
    robin,
):
    """Dirichlet or physical Robin residual on the outer PCB boundary.

    Robin convention: ``-n dot k grad(theta) = gamma*theta``.  ``robin`` is
    shape (B,) and selects this residual; Dirichlet samples use theta=0.
    """
    points = points.detach().requires_grad_(True)
    theta = predict_fn(points)
    grad = torch.autograd.grad(theta.sum(), points, create_graph=True)[0]
    normal_grad = (grad * normals).sum(dim=-1)
    flux = -conductivity * normal_grad
    if gamma.ndim == 1:
        gamma = gamma[:, None]
    if robin.ndim == 1:
        robin = robin[:, None]
    robin_r = flux - gamma * theta
    residual = torch.where(robin, robin_r, theta)
    denom = torch.where(
        robin,
        (flux.abs() + (gamma * theta).abs() + 1.0).detach(),
        (theta.abs() + 1.0).detach(),
    )
    return residual / denom


def interface_flux_loss(
    predict_fn,
    materials,
    material_mask=None,
    n_points: int = 24,
    eps: float = 0.004,
):
    """Flux-continuity loss at ideal-contact rectangular interfaces.

    The single neural field is already temperature-continuous.  Gradients are
    evaluated just inside and outside each interface so a piecewise flux can be
    represented despite the smooth coordinate network.
    """
    if materials is None or materials.shape[1] == 0 or n_points <= 0:
        return None
    batch, n_regions, _ = materials.shape
    if material_mask is None:
        material_mask = torch.zeros(
            batch, n_regions, dtype=torch.bool, device=materials.device
        )

    inside_parts, outside_parts, normal_parts, kin_parts, rc_parts = [], [], [], [], []
    for b in range(batch):
        valid = torch.nonzero(~material_mask[b], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        pick = valid[torch.randint(valid.numel(), (n_points,), device=materials.device)]
        region = materials[b, pick]
        cx, cy = region[:, 0], region[:, 1]
        hx, hy = region[:, 2], region[:, 3]
        sin_a, cos_a = region[:, 4], region[:, 5]
        edge = torch.randint(4, (n_points,), device=materials.device)
        tangent = torch.rand(n_points, device=materials.device) * 2.0 - 1.0
        rect_x = torch.where(
            edge == 0, -hx,
            torch.where(edge == 1, hx, tangent * hx),
        )
        rect_y = torch.where(
            edge == 2, -hy,
            torch.where(edge == 3, hy, tangent * hy),
        )
        rect_nx = torch.where(edge == 0, -torch.ones_like(hx),
                              torch.where(edge == 1, torch.ones_like(hx),
                                          torch.zeros_like(hx)))
        rect_ny = torch.where(edge == 2, -torch.ones_like(hy),
                              torch.where(edge == 3, torch.ones_like(hy),
                                          torch.zeros_like(hy)))
        phi = torch.rand(n_points, device=materials.device) * (2.0 * torch.pi)
        ellipse_x, ellipse_y = hx * torch.cos(phi), hy * torch.sin(phi)
        ellipse_nx = torch.cos(phi) / hx.clamp_min(1e-4)
        ellipse_ny = torch.sin(phi) / hy.clamp_min(1e-4)
        ellipse_norm = torch.sqrt(ellipse_nx.square() + ellipse_ny.square()).clamp_min(1e-6)
        ellipse_nx, ellipse_ny = ellipse_nx / ellipse_norm, ellipse_ny / ellipse_norm
        shape = region[:, 9].clamp(0.0, 1.0)
        local_x = rect_x * (1.0 - shape) + ellipse_x * shape
        local_y = rect_y * (1.0 - shape) + ellipse_y * shape
        nx_local = rect_nx * (1.0 - shape) + ellipse_nx * shape
        ny_local = rect_ny * (1.0 - shape) + ellipse_ny * shape
        bx = cx + cos_a * local_x - sin_a * local_y
        by = cy + sin_a * local_x + cos_a * local_y
        nx = cos_a * nx_local - sin_a * ny_local
        ny = sin_a * nx_local + cos_a * ny_local
        boundary = torch.stack([bx, by], dim=-1)
        normal = torch.stack([nx, ny], dim=-1)
        inside_parts.append(boundary - eps * normal)
        outside_parts.append(boundary + eps * normal)
        normal_parts.append(normal)
        kin_parts.append(torch.exp(region[:, 6]))
        # Stored as log(1 + eta) so zero contact resistance remains finite.
        # Keep the list parallel to the geometry lists.
        rc_parts.append(torch.expm1(region[:, 8]).clamp_min(0.0))

    if not inside_parts:
        return materials.sum() * 0.0
    inside = torch.cat(inside_parts).unsqueeze(0).detach().requires_grad_(True)
    outside = torch.cat(outside_parts).unsqueeze(0).detach().requires_grad_(True)
    normals = torch.cat(normal_parts).unsqueeze(0)
    k_inside = torch.cat(kin_parts).unsqueeze(0)
    rc_interface = torch.cat(rc_parts).unsqueeze(0)

    # The sampled points come from potentially different batch items.  Evaluate
    # per original sample to retain its sensors/context/material conditioning.
    losses = []
    cursor = 0
    for b in range(batch):
        count = n_points if (~material_mask[b]).any() else 0
        if count == 0:
            continue
        pin = inside[:, cursor:cursor + count]
        pout = outside[:, cursor:cursor + count]
        normal = normals[:, cursor:cursor + count]
        kin = k_inside[:, cursor:cursor + count]
        rc = rc_interface[:, cursor:cursor + count]
        tin = predict_fn(pin, batch_index=b)
        tout = predict_fn(pout, batch_index=b)
        gin = torch.autograd.grad(tin.sum(), pin, create_graph=True)[0]
        gout = torch.autograd.grad(tout.sum(), pout, create_graph=True)[0]
        fin = kin * (gin * normal).sum(dim=-1)
        fout = (gout * normal).sum(dim=-1)  # background k=1
        residual = (fin - fout) / (fin.abs() + fout.abs() + 1.0).detach()
        jump = tin - tout + rc * fin
        jump = jump / (tin.abs() + tout.abs() + (rc * fin).abs() + 1.0).detach()
        losses.append((residual.square() + jump.square()).mean())
        cursor += count
    return torch.stack(losses).mean() if losses else materials.sum() * 0.0


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
