"""Compare any number of v3 sensor-layout runs on one held-out target."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def hot_mask(field: np.ndarray, fraction: float) -> np.ndarray:
    background = float(np.median(field[:20, :20]))
    return field >= background + fraction * (float(field.max()) - background)


def iou(prediction: np.ndarray, truth: np.ndarray, fraction: float) -> float:
    pred_mask = hot_mask(prediction, fraction)
    truth_mask = hot_mask(truth, fraction)
    return float((pred_mask & truth_mask).sum() / (pred_mask | truth_mask).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=NPZ",
        help="repeat for each layout",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    truth = None
    for item in args.run:
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        with np.load(path) as archive:
            candidate_truth = archive["target"].mean(axis=0)
            prediction = archive["prediction"].mean(axis=0)
        if truth is None:
            truth = candidate_truth
        else:
            np.testing.assert_allclose(candidate_truth, truth)
        report = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
        metrics = report["new_card_test"]
        runs.append({
            "label": label,
            "path": str(path),
            "prediction": prediction,
            "trusted_mae_c": metrics["trusted_region"]["mae_c"],
            "trusted_rmse_c": metrics["trusted_region"]["rmse_c"],
            "soc_rmse_c": metrics["soc_roi"]["rmse_c"],
            "peak_error_c": metrics["soc_roi"]["peak_error_c"],
            "hotspot_error_px": metrics["soc_roi"]["hotspot_error_px"],
            "full_rmse_c": metrics["full_crop"]["rmse_c"],
            "iou_90": iou(prediction, truth, 0.90),
            "iou_95": iou(prediction, truth, 0.95),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [{k: v for k, v in run.items() if k != "prediction"} for run in runs]
    (args.output_dir / "k3_layout_metrics.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "k3_layout_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=serializable[0].keys())
        writer.writeheader()
        writer.writerows(serializable)

    fields = [("case17 truth", truth)] + [(run["label"], run["prediction"]) for run in runs]
    columns = 3
    rows = math.ceil(len(fields) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13.5, 4.3 * rows), squeeze=False)
    low, high = float(truth.min()), float(truth.max())
    for axis, (title, field) in zip(axes.flat, fields):
        image = axis.imshow(field, cmap="inferno", vmin=low, vmax=high)
        for level, color in ((0.90, "cyan"), (0.95, "lime")):
            threshold = float(np.median(field[:20, :20])) + level * (
                float(field.max()) - float(np.median(field[:20, :20]))
            )
            axis.contour(field, levels=[threshold], colors=[color], linewidths=1.3)
        axis.set_xlim(15, 85)
        axis.set_ylim(150, 80)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes.flat[len(fields):]:
        axis.axis("off")
    figure.suptitle("K=3 sensor-layout comparison: cyan=90%, green=95% temperature-rise contour")
    figure.tight_layout()
    figure.savefig(args.output_dir / "k3_layout_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(serializable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
