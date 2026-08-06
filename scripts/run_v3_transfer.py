"""Train and evaluate a board card on the log-aligned v3 measurements.

The default protocol uses only the Pi 4B matte-paint half-load plateau for
adaptation and keeps the full-load plateau entirely untouched for testing.
The frozen, full 4096-board pretraining checkpoint is used by default.  For the
Pi profile, the legacy tape-board card is evaluated on the same K=3 inputs as a
cross-surface baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from tdp.data.patches import make_validation_patches
from tdp.model.normalization import px_to_xy
from tdp.model.operator import load_checkpoint
from tdp.train.finetune import (
    BoardCard,
    FinetuneConfig,
    RealCase,
    finetune,
    make_sensor_tensor,
    predict_field,
    read_sites,
    sites_px_for,
)


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "real_v3"


def load_real_case(state_id: str) -> RealCase:
    with np.load(PROC / f"{state_id}.npz") as archive:
        field = archive["mean_T"].astype(np.float64)
        std = archive["std_T"].astype(np.float64)
        trusted = archive["trusted"].astype(bool)
        ambient = float(np.median(archive["t_amb"]))
    height, width = field.shape
    return RealCase(
        state_id,
        field,
        np.maximum(std, 0.10),
        ambient,
        trusted,
        (height, width),
        width / height,
    )


def load_test_state(state_id: str) -> dict[str, np.ndarray]:
    with np.load(PROC / f"{state_id}.npz") as archive:
        return {key: archive[key] for key in ("T", "t_amb", "timestamp_epoch", "trusted")}


def load_saved_card(path: Path, model, device: torch.device) -> BoardCard:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tokens = payload["tokens"].to(dtype=torch.float32)
    card = BoardCard(
        tokens.shape[1], model.cfg.d_model, init_tokens=tokens,
        material_descriptors=payload.get("material_descriptors"),
        source_descriptors=payload.get("source_descriptors"),
        boundary_descriptors=payload.get("boundary_descriptors"),
    ).to(device)
    with torch.no_grad():
        card.log_h.fill_(float(payload["log_h"]))
        # New cards persist already-composed per-segment gamma values.  Legacy
        # cards have only the global scalar.
        card.log_gamma.fill_(
            0.0 if "boundary_descriptors" in payload else float(payload["log_gamma"])
        )
    return card


def board_geometry_descriptors(path: Path, board_id: str, aspect: float):
    """Convert board-card JSON in fractional (u,v) coordinates to model tensors."""
    if not path.exists():
        return None, None, None
    spec = json.loads(path.read_text(encoding="utf-8"))["boards"].get(board_id)
    if spec is None:
        return None, None, None
    materials = []
    for region in spec.get("materials", []):
        angle = np.deg2rad(float(region.get("angle_deg", 0.0)))
        materials.append([
            float(region["center_u"]) * aspect,
            float(region["center_v"]),
            float(region["half_u"]) * aspect,
            float(region["half_v"]),
            float(np.sin(angle)), float(np.cos(angle)),
            float(np.log(region.get("k_rel_init", 1.0))),
            float(np.log(region.get("h_rel_init", 1.0))),
            float(np.log1p(region.get("rc_init", 0.0))),
            float(region.get("shape", "rect") == "ellipse"),
        ])
    sources = []
    for region in spec.get("sources", []):
        angle = np.deg2rad(float(region.get("angle_deg", 0.0)))
        sources.append([
            float(region["center_u"]) * aspect,
            float(region["center_v"]),
            float(region["half_u"]) * aspect,
            float(region["half_v"]),
            float(np.sin(angle)), float(np.cos(angle)),
            float(np.log(max(region.get("amplitude_init", 1.0), 1e-4))),
            float(region.get("shape", "rect") in ("ellipse", "gauss")),
        ])
    boundaries = []
    side_geometry = {
        "left": lambda a, b: (0.0, a, 0.0, b, -1.0, 0.0),
        "right": lambda a, b: (aspect, a, aspect, b, 1.0, 0.0),
        "top": lambda a, b: (a * aspect, 0.0, b * aspect, 0.0, 0.0, -1.0),
        "bottom": lambda a, b: (a * aspect, 1.0, b * aspect, 1.0, 0.0, 1.0),
    }
    for segment in spec.get("boundaries", []):
        geometry = side_geometry[segment["side"]](
            float(segment.get("start", 0.0)), float(segment.get("end", 1.0))
        )
        boundaries.append([
            *geometry,
            float(np.log(segment.get("gamma_init", 1.0))),
            float(segment.get("type", "robin") == "robin"),
        ])
    material_tensor = torch.tensor(materials, dtype=torch.float32).reshape(-1, 10)
    source_tensor = torch.tensor(sources, dtype=torch.float32).reshape(-1, 8)
    boundary_tensor = torch.tensor(boundaries, dtype=torch.float32).reshape(-1, 8)
    return material_tensor, source_tensor, boundary_tensor


def predict_stack(model, card: BoardCard, test: dict[str, np.ndarray], sites: list[dict],
                  session: str) -> np.ndarray:
    fields = test["T"]
    height, width = fields.shape[1:]
    rows, cols = np.indices((height, width))
    queries = px_to_xy(rows.ravel(), cols.ravel(), n_rows=height, n_cols=width)
    site_pixels = sites_px_for(sites, session, (height, width))
    predictions = []
    for field, ambient in zip(fields, test["t_amb"]):
        xy, values = read_sites(field, site_pixels)
        sensors, scale = make_sensor_tensor(xy, values, float(ambient))
        theta = predict_field(model, card, sensors, width / height, queries)
        predictions.append(theta.reshape(height, width) * scale + float(ambient))
    return np.asarray(predictions, dtype=np.float32)


def metric_set(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if mask.ndim == 2:
        mask = np.repeat(mask[None], len(target), axis=0)
    error = prediction - target
    valid = error[mask]
    width = prediction.shape[-1]
    peak_errors, hotspot_errors = [], []
    for pred, true, valid_mask in zip(prediction, target, mask):
        pred_masked = np.where(valid_mask, pred, -np.inf)
        true_masked = np.where(valid_mask, true, -np.inf)
        pred_index = int(pred_masked.argmax())
        true_index = int(true_masked.argmax())
        peak_errors.append(abs(float(pred_masked.flat[pred_index] - true_masked.flat[true_index])))
        hotspot_errors.append(float(np.hypot(
            pred_index % width - true_index % width,
            pred_index // width - true_index // width,
        )))
    grad_pred_y, grad_pred_x = np.gradient(prediction, axis=(1, 2))
    grad_true_y, grad_true_x = np.gradient(target, axis=(1, 2))
    grad_error = np.hypot(grad_pred_y - grad_true_y, grad_pred_x - grad_true_x)
    return {
        "mae_c": float(np.abs(valid).mean()),
        "rmse_c": float(np.sqrt(np.mean(valid ** 2))),
        "max_ae_c": float(np.abs(valid).max()),
        "peak_error_c": float(np.mean(peak_errors)),
        "hotspot_error_px": float(np.mean(hotspot_errors)),
        "gradient_mae_c_per_px": float(grad_error[mask].mean()),
    }


def sensor_roi(shape: tuple[int, int], site: dict, radius: int = 12) -> np.ndarray:
    height, width = shape
    row = int(round(float(site["v"]) * height))
    col = int(round(float(site["u"]) * width))
    rows, cols = np.indices(shape)
    return (rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="pi4b_paint")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "measurements_v3.yaml")
    parser.add_argument("--sensors", type=Path, default=ROOT / "configs" / "sensors_v3.json")
    parser.add_argument(
        "--board-geometry", type=Path,
        default=ROOT / "configs" / "board_material_geometry_v1.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "v2" / "pretrain_v2.pt")
    parser.add_argument("--legacy-card", type=Path, default=ROOT / "models" / "boards" / "pi4b_s1.pt")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument(
        "--shape-loss", choices=("none", "soc", "full"), default="none",
        help="explicit material-region loss ablation",
    )
    parser.add_argument(
        "--shape-region", choices=("material", "source"), default="material",
        help="geometry used by the explicit hotspot-shape loss",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.protocol not in config["protocols"]:
        raise ValueError(f"unknown protocol {args.protocol!r}")
    protocol = config["protocols"][args.protocol]
    profile_id = protocol.get("profile", args.protocol)
    profile = config["profiles"][profile_id]
    if not profile["dense_supervision"]:
        raise ValueError(f"{args.protocol} is marked unsafe for dense supervision")

    sensor_config = json.loads(args.sensors.read_text(encoding="utf-8"))
    layout = sensor_config["layouts"][protocol["sensor_layout"]]
    by_id = {site["id"]: site for site in layout}
    sites = []
    for sensor_id in protocol["sensor_ids"]:
        site = dict(by_id[sensor_id])
        site["trusted_sessions"] = [profile_id]
        sites.append(site)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    model, checkpoint_meta = load_checkpoint(args.checkpoint)
    model.to(device).eval()

    train_cases = [load_real_case(state_id) for state_id in protocol["train_states"]]
    reference = train_cases[-1]
    site_pixels = sites_px_for(sites, profile_id, reference.shape)
    patches = make_validation_patches(
        reference.field,
        reference.trusted,
        site_pixels,
        n_patches=12,
        exclude_r=4.0,
        min_sep=7.0,
    )
    source_mask_files = {
        "raspberry_pi_4b": ROOT / "configs" / "source_mask_board.json",
        "orange_pi_5_pro": ROOT / "configs" / "source_mask_opi5pro.json",
    }
    source_mask_path = source_mask_files.get(profile["board"])
    source_rects = (json.loads(source_mask_path.read_text(encoding="utf-8"))["rects"]
                    if source_mask_path is not None else [])
    material_descriptors, source_descriptors, boundary_descriptors = board_geometry_descriptors(
        args.board_geometry, profile["board"], reference.aspect
    )

    shape_weights = {
        "none": {"lambda_soc": 0.0, "lambda_edge": 0.0,
                 "lambda_plateau": 0.0, "lambda_peak": 0.0},
        "soc": {"lambda_soc": 2.0, "lambda_edge": 0.0,
                "lambda_plateau": 0.0, "lambda_peak": 0.0},
        "full": {"lambda_soc": 2.0, "lambda_edge": 0.1,
                 "lambda_plateau": 0.1, "lambda_peak": 0.05},
    }[args.shape_loss]
    tune_config = FinetuneConfig(
        n_tokens=args.tokens,
        steps=args.steps,
        n_query=512,
        n_query_hot=96,
        n_query_site=32,
        lambda_pde=5e-5,
        lambda_boundary=5e-5,
        lambda_interface=2e-4,
        n_colloc=64,
        n_boundary=32,
        n_interface=16,
        **shape_weights,
        shape_region_type=args.shape_region,
        eval_every=20,
        seed=31,
    )
    print(
        f"device={device} checkpoint_epoch={checkpoint_meta.get('epoch')} "
        f"train={protocol['train_states']} test={protocol['test_state']} "
        f"sensors={protocol['sensor_ids']}"
    )
    card, history = finetune(
        model,
        train_cases,
        sites,
        source_rects,
        patches,
        tune_config,
        session=profile_id,
        verbose=True,
        material_descriptors=material_descriptors,
        source_descriptors=source_descriptors,
        boundary_descriptors=boundary_descriptors,
    )

    test = load_test_state(protocol["test_state"])
    prediction = predict_stack(model, card, test, sites, profile_id)
    target = test["T"].astype(np.float32)
    trust = test["trusted"].astype(bool)
    soc_site = next(site for site in sites if site["id"] == "soc")
    soc_mask = sensor_roi(target.shape[1:], soc_site)
    region_masks = {}
    height, width = target.shape[1:]
    region_rows, region_cols = np.indices((height, width))
    for region_name, region in protocol.get("evaluation_regions", {}).items():
        center_row = float(region["v"]) * height
        center_col = float(region["u"]) * width
        radius = float(region["radius_px"])
        region_masks[region_name] = (
            (region_rows - center_row) ** 2 + (region_cols - center_col) ** 2 <= radius ** 2
        )
    report = {
        "protocol": {
            "name": args.protocol,
            "profile": profile_id,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": None,
            "pretrain_epoch": checkpoint_meta.get("epoch"),
            "train_states": protocol["train_states"],
            "test_state": protocol["test_state"],
            "test_frames": int(len(target)),
            "sensor_ids": protocol["sensor_ids"],
            "steps": args.steps,
            "tokens": args.tokens,
            "material_regions": 0 if material_descriptors is None else len(material_descriptors),
            "source_regions": 0 if source_descriptors is None else len(source_descriptors),
            "boundary_segments": 0 if boundary_descriptors is None else len(boundary_descriptors),
            "shape_loss": args.shape_loss,
            "shape_region": args.shape_region,
            "shape_loss_weights": shape_weights,
            "device": str(device),
            "test_is_untouched": True,
            "test_semantics": protocol.get("test_semantics", "held_out_session"),
        },
        "training": history,
        "new_card_test": {
            "trusted_region": metric_set(prediction, target, trust),
            "soc_roi": metric_set(prediction, target, soc_mask),
            "evaluation_regions": {
                name: metric_set(prediction, target, mask)
                for name, mask in region_masks.items()
            },
            "full_crop": metric_set(prediction, target, np.ones_like(trust)),
        },
    }

    legacy_prediction = None
    if profile_id == "pi4b_paint" and args.legacy_card.exists():
        legacy_card = load_saved_card(args.legacy_card, model, device)
        legacy_prediction = predict_stack(model, legacy_card, test, sites, profile_id)
        report["legacy_card_test"] = {
            "card": str(args.legacy_card),
            "note": "old tape-board card evaluated cross-surface with identical K=3 readings",
            "trusted_region": metric_set(legacy_prediction, target, trust),
            "soc_roi": metric_set(legacy_prediction, target, soc_mask),
            "full_crop": metric_set(legacy_prediction, target, np.ones_like(trust)),
        }

    import hashlib

    report["protocol"]["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    output_dir = args.output_dir or (
        ROOT / "benchmark" / "results" /
        f"v3_{args.protocol}_{args.shape_loss}_k{len(sites)}_{args.steps}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    card.save(output_dir / f"{args.protocol}_card_k{len(sites)}.pt", meta=report["protocol"])
    arrays = {"prediction": prediction, "target": target, "trusted": trust,
              "timestamp_epoch": test["timestamp_epoch"]}
    if legacy_prediction is not None:
        arrays["legacy_prediction"] = legacy_prediction
    np.savez_compressed(output_dir / "test_predictions.npz", **arrays)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    true_mean = target.mean(axis=0)
    pred_mean = prediction.mean(axis=0)
    columns = 4 if legacy_prediction is not None else 3
    fig, axes = plt.subplots(1, columns, figsize=(4.8 * columns, 4.4))
    low, high = float(min(true_mean.min(), pred_mean.min())), float(max(true_mean.max(), pred_mean.max()))
    truth_title = (
        "cooldown truth"
        if protocol.get("test_semantics") == "dynamic_cooldown_stress_test"
        else "held-out truth"
    )
    panels = [(truth_title, true_mean), ("new v3 board card", pred_mean)]
    if legacy_prediction is not None:
        panels.append(("legacy tape card", legacy_prediction.mean(axis=0)))
    for axis, (title, field) in zip(axes, panels):
        image = axis.imshow(field, cmap="inferno", vmin=low, vmax=high)
        axis.scatter(site_pixels[:, 1], site_pixels[:, 0], marker="x", c="cyan", s=55, lw=1.8)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046)
    error_axis = axes[-1]
    error = np.abs(pred_mean - true_mean)
    image = error_axis.imshow(error, cmap="magma", vmin=0)
    error_axis.set_title("new-card absolute error")
    fig.colorbar(image, ax=error_axis, fraction=0.046)
    fig.suptitle(
        f"{args.protocol}: train {', '.join(protocol['train_states'])}; "
        f"untouched test {protocol['test_state']}; K={len(sites)}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "test_preview.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report["new_card_test"], ensure_ascii=False, indent=2))
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
