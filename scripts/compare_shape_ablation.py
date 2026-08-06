"""Compare baseline, SoC-only, and full explicit-shape Board Cards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def hot_mask(field: np.ndarray, fraction: float) -> np.ndarray:
    background = float(np.median(field[:20, :20]))
    return field >= background + fraction * (float(field.max()) - background)


def mask_stats(prediction: np.ndarray, truth: np.ndarray, fraction: float) -> dict:
    pred_mask, truth_mask = hot_mask(prediction, fraction), hot_mask(truth, fraction)

    def bounds(mask):
        rows, cols = np.nonzero(mask)
        return {"height_px": int(np.ptp(rows) + 1),
                "width_px": int(np.ptp(cols) + 1), "area_px": int(mask.sum())}

    return {"iou": float((pred_mask & truth_mask).sum() / (pred_mask | truth_mask).sum()),
            "truth": bounds(truth_mask), "prediction": bounds(pred_mask)}


def load_run(path: Path):
    with np.load(path) as archive:
        truth = archive["target"].mean(axis=0)
        prediction = archive["prediction"].mean(axis=0)
    report = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
    return truth, prediction, report["new_card_test"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--soc", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--baseline-label", default="A: baseline")
    parser.add_argument("--soc-label", default="B: + SoC loss")
    parser.add_argument("--full-label", default="C: + edge/plateau/peak")
    parser.add_argument(
        "--title",
        default="Explicit hotspot-shape loss ablation: cyan=90%, green=95%",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {"A_baseline": args.baseline, "B_soc_only": args.soc,
             "C_full_shape": args.full}
    loaded = {name: load_run(path) for name, path in paths.items()}
    truth = loaded["A_baseline"][0]
    for candidate_truth, _, _ in loaded.values():
        np.testing.assert_allclose(candidate_truth, truth)

    output = {}
    for name, (_, prediction, metrics) in loaded.items():
        output[name] = {
            "hotspot_shape": {str(level): mask_stats(prediction, truth, level)
                              for level in (0.90, 0.95)},
            "soc_roi": metrics["soc_roi"],
            "trusted_region": metrics["trusted_region"],
            "full_crop": metrics["full_crop"],
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shape_ablation_metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = [("held-out truth", truth)] + [
        (label, loaded[key][1]) for key, label in (
            ("A_baseline", args.baseline_label),
            ("B_soc_only", args.soc_label),
            ("C_full_shape", args.full_label),
        )
    ]
    low, high = float(truth.min()), float(truth.max())
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2))
    for axis, (title, field) in zip(axes, fields):
        image = axis.imshow(field, cmap="inferno", vmin=low, vmax=high)
        for level, color in ((0.90, "cyan"), (0.95, "lime")):
            background = float(np.median(field[:20, :20]))
            threshold = background + level * (float(field.max()) - background)
            axis.contour(field, levels=[threshold], colors=[color], linewidths=1.4)
        axis.set_xlim(20, 75)
        axis.set_ylim(145, 85)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output_dir / "shape_ablation_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
