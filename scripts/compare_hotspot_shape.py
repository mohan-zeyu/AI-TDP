"""Compare high-temperature plateau geometry between two prediction runs."""

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
    pred_mask = hot_mask(prediction, fraction)
    truth_mask = hot_mask(truth, fraction)

    def bounds(mask: np.ndarray) -> dict:
        rows, cols = np.nonzero(mask)
        return {
            "height_px": int(np.ptp(rows) + 1),
            "width_px": int(np.ptp(cols) + 1),
            "area_px": int(mask.sum()),
        }

    return {
        "iou": float((pred_mask & truth_mask).sum() / (pred_mask | truth_mask).sum()),
        "truth": bounds(truth_mask),
        "prediction": bounds(pred_mask),
    }


def load_mean(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        return archive["target"].mean(axis=0), archive["prediction"].mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    truth, baseline = load_mean(args.baseline)
    candidate_truth, candidate = load_mean(args.candidate)
    np.testing.assert_allclose(candidate_truth, truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "baseline": {str(level): mask_stats(baseline, truth, level)
                     for level in (0.90, 0.95)},
        "heterogeneous": {str(level): mask_stats(candidate, truth, level)
                          for level in (0.90, 0.95)},
    }
    (args.output_dir / "hotspot_shape_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    fields = [("case17 truth", truth), (args.baseline_label, baseline),
              (args.candidate_label, candidate)]
    low, high = float(truth.min()), float(truth.max())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for axis, (title, field) in zip(axes, fields):
        image = axis.imshow(field, cmap="inferno", vmin=low, vmax=high)
        for level, color in ((0.90, "cyan"), (0.95, "lime")):
            threshold = float(np.median(field[:20, :20])) + level * (
                float(field.max()) - float(np.median(field[:20, :20]))
            )
            axis.contour(field, levels=[threshold], colors=[color], linewidths=1.4)
        axis.set_xlim(20, 75)
        axis.set_ylim(145, 85)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle("SoC hotspot shape: cyan=90%, green=95% temperature-rise contour")
    fig.tight_layout()
    fig.savefig(args.output_dir / "hotspot_shape_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
