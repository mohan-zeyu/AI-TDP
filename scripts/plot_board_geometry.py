"""Visual QC for fixed Board Card material geometry on a training field."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def corners(region: dict, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    aspect = width / height
    half_x = float(region["half_u"]) * aspect
    half_y = float(region["half_v"])
    local = np.array([[-half_x, -half_y], [half_x, -half_y],
                      [half_x, half_y], [-half_x, half_y]])
    angle = np.deg2rad(float(region.get("angle_deg", 0.0)))
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    xy = local @ rotation.T + np.array([
        float(region["center_u"]) * aspect, float(region["center_v"])
    ])
    uv = np.stack([xy[:, 0] / aspect, xy[:, 1]], axis=-1)
    return np.stack([uv[:, 0] * (width - 1), uv[:, 1] * (height - 1)], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="case14_half_a")
    parser.add_argument("--board", default="orange_pi_5_pro")
    parser.add_argument("--geometry", type=Path,
                        default=ROOT / "configs" / "board_material_geometry_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(ROOT / "data" / "processed" / "real_v3" / f"{args.state}.npz") as data:
        field = data["mean_T"]
    spec = json.loads(args.geometry.read_text(encoding="utf-8"))["boards"][args.board]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2))
    for axis in axes:
        image = axis.imshow(field, cmap="inferno")
        for index, region in enumerate(spec["materials"]):
            polygon = corners(region, field.shape)
            axis.add_patch(Polygon(polygon, closed=True, fill=False,
                                   edgecolor="cyan", linewidth=1.8))
            center = polygon.mean(axis=0)
            axis.text(center[0], center[1], region["id"], color="cyan",
                      fontsize=8, ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046)
    axes[0].set_title(f"{args.state}: Board Card material geometry")
    axes[1].set_xlim(15, 95)
    axes[1].set_ylim(155, 65)
    axes[1].set_title("component-region detail")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
