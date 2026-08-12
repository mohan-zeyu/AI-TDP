"""Plot the corrected Pi 4B material/source partition and fitted Robin boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def rectangle_from_descriptor(row: np.ndarray, aspect: float, width: int, height: int):
    center_u = float(row[0]) / aspect
    center_v = float(row[1])
    half_u = float(row[2]) / aspect
    half_v = float(row[3])
    return (
        (center_u - half_u) * width,
        (center_v - half_v) * height,
        2.0 * half_u * width,
        2.0 * half_v * height,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--card", type=Path,
        default=ROOT / "benchmark" / "results" /
        "v31_pi4b_corrected_fullgpu_pmic_k3_t24_600" / "pi4b_paint_card_k3.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports" / "paper_zh" /
        "figure_material_boundary_corrected_pi4b.png",
    )
    args = parser.parse_args()

    payload = torch.load(args.card, map_location="cpu", weights_only=False)
    materials = np.asarray(payload["material_descriptors"], dtype=float)
    sources = np.asarray(payload["source_descriptors"], dtype=float)
    boundaries = np.asarray(payload["boundary_descriptors"], dtype=float)

    height, width = 166, 110
    aspect = width / height
    material_labels = ["主芯片封装", "RAM封装（低发热）", "USB控制器"]

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 160,
        "savefig.dpi": 320,
    })

    k_map = np.ones((height, width), dtype=float)
    material_boxes = []
    for row, label in zip(materials, material_labels):
        x, y, box_width, box_height = rectangle_from_descriptor(
            row, aspect, width, height
        )
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(width, int(round(x + box_width)))
        y1 = min(height, int(round(y + box_height)))
        k_rel = float(np.exp(row[6]))
        k_map[y0:y1, x0:x1] = k_rel
        material_boxes.append((x, y, box_width, box_height, label, k_rel))

    fig = plt.figure(figsize=(11.2, 5.35), facecolor="white")
    ax_left = fig.add_axes([0.075, 0.20, 0.34, 0.72])
    cax_left = fig.add_axes([0.425, 0.26, 0.018, 0.58])
    ax_right = fig.add_axes([0.565, 0.20, 0.34, 0.72])
    cax_right = fig.add_axes([0.915, 0.26, 0.018, 0.58])

    k_min = min(0.95, float(k_map.min()))
    k_max = max(1.80, float(k_map.max()))
    k_norm = colors.Normalize(vmin=k_min, vmax=k_max)
    shown = ax_left.imshow(
        k_map, cmap="YlOrRd", norm=k_norm, origin="upper",
        interpolation="nearest", aspect="auto"
    )
    for x, y, box_width, box_height, label, k_rel in material_boxes:
        ax_left.add_patch(Rectangle(
            (x, y), box_width, box_height,
            fill=False, edgecolor="#172033", linewidth=1.25,
        ))
        ax_left.text(
            x + box_width / 2, y + box_height / 2,
            f"{label}\n$K_i/K_0={k_rel:.2f}$",
            ha="center", va="center", fontsize=8.0, color="#172033",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white",
                  "alpha": 0.82, "ec": "none"},
        )

    source_boxes = []
    for index, row in enumerate(sources):
        x, y, box_width, box_height = rectangle_from_descriptor(
            row, aspect, width, height
        )
        source_boxes.append((x, y, box_width, box_height))
        ax_left.add_patch(Rectangle(
            (x, y), box_width, box_height,
            fill=False, edgecolor="#d62728", linewidth=1.35,
            linestyle=(0, (4, 2)),
        ))

    pmic_x = np.mean([box[0] + box[2] / 2 for box in source_boxes[1:]])
    pmic_y = np.mean([box[1] + box[3] / 2 for box in source_boxes[1:]])
    ax_left.annotate(
        "PMIC三个局部热源", xy=(pmic_x, pmic_y), xytext=(68, 145),
        textcoords="data", color="#b91c1c", fontsize=8.2,
        arrowprops={"arrowstyle": "->", "color": "#b91c1c", "lw": 0.9},
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.82,
              "ec": "none"},
    )

    legend_handles = [
        Line2D([0], [0], color="#172033", lw=1.3, label="材料区域"),
        Line2D([0], [0], color="#d62728", lw=1.3, linestyle=(0, (4, 2)),
               label="局部热源"),
    ]
    ax_left.legend(handles=legend_handles, loc="upper left", framealpha=0.90,
                   fontsize=8.0)
    ax_left.set_xlabel("横向位置/像素")
    ax_left.set_ylabel("纵向位置/像素")
    ax_left.text(
        0.5, -0.16, "（a）相对等效导热系数与局部热源分区",
        transform=ax_left.transAxes, ha="center", va="top", fontsize=10.8,
    )
    colorbar_left = fig.colorbar(shown, cax=cax_left)
    colorbar_left.set_label(r"相对等效导热系数 $K/K_0$")

    ax_right.set_xlim(-0.08, 1.08)
    ax_right.set_ylim(1.08, -0.08)
    ax_right.set_aspect("equal")
    ax_right.add_patch(Rectangle(
        (0, 0), 1, 1, facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.0
    ))
    gamma = np.exp(boundaries[:, 6])
    gamma_norm = colors.Normalize(vmin=float(gamma.min()), vmax=float(gamma.max()))
    gamma_cmap = plt.get_cmap("viridis")
    for index, row in enumerate(boundaries):
        x0, y0, x1, y1 = row[:4]
        x0, x1 = x0 / aspect, x1 / aspect
        value = float(np.exp(row[6]))
        ax_right.plot(
            [x0, x1], [y0, y1], linewidth=7, solid_capstyle="butt",
            color=gamma_cmap(gamma_norm(value)),
        )
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        if index == 0:
            offset = (0.065, 0.0)
        elif index in (1, 2):
            offset = (0.0, 0.055 if index == 1 else -0.055)
        else:
            offset = (-0.095, 0.0)
        ax_right.text(
            mid_x + offset[0], mid_y + offset[1],
            rf"$\gamma_{{{index + 1}}}={value:.2f}$",
            fontsize=7.7, ha="center", va="center", color="#263448",
            bbox={"fc": "white", "alpha": 0.72, "ec": "none", "pad": 0.8},
        )
    ax_right.text(
        0.5, 0.5, r"$-\mathbf{n}\!\cdot\!K\nabla T=\gamma(T-T_0)$",
        ha="center", va="center", fontsize=12.0, color="#334155",
    )
    ax_right.set_xlabel("归一化横坐标")
    ax_right.set_ylabel("归一化纵坐标")
    ax_right.text(
        0.5, -0.16, "（b）分段Robin边界换热系数",
        transform=ax_right.transAxes, ha="center", va="top", fontsize=10.8,
    )
    gamma_mappable = plt.cm.ScalarMappable(norm=gamma_norm, cmap=gamma_cmap)
    colorbar_right = fig.colorbar(gamma_mappable, cax=cax_right)
    colorbar_right.set_label(r"相对边界系数 $\gamma$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
