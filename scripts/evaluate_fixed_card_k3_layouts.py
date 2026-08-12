"""Evaluate one frozen board card with several K=3 sensor layouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_v3_transfer import (
    ROOT,
    load_saved_card,
    load_test_state,
    metric_set,
    predict_stack,
    sensor_roi,
)
from tdp.model.operator import load_checkpoint


plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


LAYOUTS = [
    ("soc_gradient_cold", "主芯片+邻近梯度点+冷角点", ["soc", "ram", "corner_cold"]),
    ("soc_pmic_cold", "主芯片+PMIC+冷角点", ["soc", "pmic", "corner_cold"]),
    ("soc_usb_cold", "主芯片+USB控制器+冷角点", ["soc", "usb_ctrl", "corner_cold"]),
    ("soc_usb_pmic", "主芯片+USB控制器+PMIC", ["soc", "usb_ctrl", "pmic"]),
    ("soc_gradient_usb", "主芯片+邻近梯度点+USB控制器", ["soc", "ram", "usb_ctrl"]),
    ("gradient_usb_cold", "邻近梯度点+USB控制器+冷角点（无主芯片）", ["ram", "usb_ctrl", "corner_cold"]),
]


def hot_mask(field: np.ndarray, fraction: float) -> np.ndarray:
    background = float(np.median(field[:20, :20]))
    threshold = background + fraction * (float(field.max()) - background)
    return field >= threshold


def hot_iou(prediction: np.ndarray, truth: np.ndarray, fraction: float) -> float:
    pred_mask = hot_mask(prediction, fraction)
    truth_mask = hot_mask(truth, fraction)
    union = pred_mask | truth_mask
    return float((pred_mask & truth_mask).sum() / max(int(union.sum()), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "models" / "v3" / "pretrain_v31_material_physics_512x12.pt",
    )
    parser.add_argument(
        "--card", type=Path,
        default=ROOT / "models" / "boards" / "pi4b_corrected_geometry_k3_t24.pt",
    )
    parser.add_argument(
        "--sensors", type=Path, default=ROOT / "configs" / "sensors_v3.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports" / "model_comparison" /
        "v31_corrected_fixed_card_k3_layouts",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    model, checkpoint_meta = load_checkpoint(args.checkpoint)
    model.to(device).eval()
    card = load_saved_card(args.card, model, device)

    sensor_config = json.loads(args.sensors.read_text(encoding="utf-8"))
    by_id = {site["id"]: site for site in sensor_config["layouts"]["pi4b_s3"]}
    test = load_test_state("case17_full")
    target = test["T"].astype(np.float32)
    trusted = test["trusted"].astype(bool)
    soc_mask = sensor_roi(target.shape[1:], by_id["soc"])
    truth_mean = target.mean(axis=0)

    rows = []
    predictions = {}
    for layout_id, label, site_ids in LAYOUTS:
        sites = []
        for site_id in site_ids:
            site = dict(by_id[site_id])
            site["trusted_sessions"] = ["pi4b_paint"]
            sites.append(site)
        prediction = predict_stack(model, card, test, sites, "pi4b_paint")
        pred_mean = prediction.mean(axis=0)
        predictions[layout_id] = prediction
        trusted_metrics = metric_set(prediction, target, trusted)
        soc_metrics = metric_set(prediction, target, soc_mask)
        full_metrics = metric_set(prediction, target, np.ones_like(trusted))
        rows.append({
            "layout_id": layout_id,
            "label": label,
            "sensor_ids": site_ids,
            "trusted_mae_c": trusted_metrics["mae_c"],
            "trusted_rmse_c": trusted_metrics["rmse_c"],
            "soc_rmse_c": soc_metrics["rmse_c"],
            "peak_error_c": soc_metrics["peak_error_c"],
            "hotspot_error_px": soc_metrics["hotspot_error_px"],
            "full_rmse_c": full_metrics["rmse_c"],
            "iou_90": hot_iou(pred_mean, truth_mean, 0.90),
            "iou_95": hot_iou(pred_mean, truth_mean, 0.95),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluation": "fixed corrected Board Card; sensor inputs changed at inference only",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_meta.get("epoch"),
        "card": str(args.card),
        "test_state": "case17_full",
        "test_frames": int(len(target)),
        "layouts": rows,
    }
    (args.output_dir / "fixed_card_k3_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "fixed_card_k3_predictions.npz",
        target=target,
        trusted=trusted,
        **predictions,
    )

    fields = [("case17 truth", truth_mean)] + [
        (row["label"], predictions[row["layout_id"]].mean(axis=0)) for row in rows
    ]
    figure, axes = plt.subplots(3, 3, figsize=(14.2, 12.2), squeeze=False)
    low, high = float(truth_mean.min()), float(truth_mean.max())
    for axis, (title, field) in zip(axes.flat, fields):
        image = axis.imshow(field, cmap="inferno", vmin=low, vmax=high)
        for level, color in ((0.90, "cyan"), (0.95, "lime")):
            mask = hot_mask(field, level)
            axis.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.2)
        axis.set_xlim(15, 90)
        axis.set_ylim(155, 80)
        axis.set_title(title, fontsize=10)
        figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes.flat[len(fields):]:
        axis.axis("off")
    figure.suptitle(
        "固定修正后Board Card的K=3选点对比（青色90%，绿色95%温升轮廓）",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "fixed_card_k3_comparison.png", dpi=200)
    plt.close(figure)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
