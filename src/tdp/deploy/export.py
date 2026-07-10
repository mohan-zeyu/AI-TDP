"""Deployment packaging: bake a board card into a fixed-signature module and
export it (TorchScript now; ONNX optional via the `deploy` extra).

The deployed model's contract is intentionally minimal:
    forward(sensors (1, K, 3), queries (1, Q, 2)) -> θ (1, Q)
with the board tokens and the condition vector stored as buffers. Temperature
denormalization (θ·ΔT_s + T_amb) is two scalar ops done by the caller.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from tdp.model.operator import ThermalOperatorV2


class DeployedBoardModel(nn.Module):
    def __init__(self, model: ThermalOperatorV2, board_tokens: torch.Tensor,
                 cond: torch.Tensor):
        super().__init__()
        self.model = model.eval()
        self.register_buffer("board_tokens", board_tokens.detach().clone())  # (1, N, d)
        self.register_buffer("cond", cond.detach().clone())                  # (1, 4)

    def forward(self, sensors: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        return self.model(sensors, queries, self.cond, board_tokens=self.board_tokens)


def load_deployed(ckpt_path: Path | str, card_path: Path | str) -> DeployedBoardModel:
    from tdp.model.operator import load_checkpoint

    model, _ = load_checkpoint(ckpt_path)
    card = torch.load(Path(card_path), map_location="cpu", weights_only=False)
    tokens = card["tokens"]
    cond = torch.tensor([[card["log_h"] / 3.0, card["log_gamma"] / 3.0, 0.0, 1.0]])
    # aspect is frame-dependent; caller sets it before export
    return DeployedBoardModel(model, tokens, cond)


def export_torchscript(deployed: DeployedBoardModel, sensors: torch.Tensor,
                       queries: torch.Tensor, out: Path | str,
                       parity_tol: float = 1e-4) -> Path:
    """Trace at the deployment shapes and verify parity against eager."""
    deployed.eval()
    with torch.no_grad():
        ref = deployed(sensors, queries)
        traced = torch.jit.trace(deployed, (sensors, queries))
        got = traced(sensors, queries)
    err = float((ref - got).abs().max())
    if err > parity_tol:
        raise RuntimeError(f"TorchScript parity failed: max |Δ| = {err:.2e}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    return out
