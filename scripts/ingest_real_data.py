"""Ingest the raw IR session (实验数据（黑胶带版）) into cropped, analysis-ready NPZ stacks.

Per case: parse + integrity-self-check every HM CSV, locate the fixed 157x103 board
window (energy criterion; ZNCC registration for the re-staged case00), crop all
frames, and write data/processed/real/<case_id>.npz plus a QC figure. A global
manifest.json records provenance, crop geometry, and per-case statistics.

Usage:  uv run scripts/ingest_real_data.py [--config configs/crop.yaml]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

import tdp
from tdp.data.crop import apply_crop, estimate_ambient, find_crop, register_crop
from tdp.io.hm_csv import HMParseError, read_hm_csv
from tdp.io.manifest import write_manifest

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "实验数据（黑胶带版）"
OUT_DIR = ROOT / "data" / "processed" / "real"
QC_DIR = ROOT / "reports" / "qc"

# (raw folder name, ascii id, english label, kind)
# kind: steady -> detect window on last (hottest) frame; cooling -> first frame;
#       calib  -> ZNCC registration against the hot reference (scene was re-staged)
CASES = [
    ("case00_不插电", "case00_unpowered", "unplugged, emissivity-calibration set", "calib"),
    ("case01_待机", "case01_idle", "idle", "steady"),
    ("case02_满载", "case02_full_load", "full CPU load", "steady"),
    ("case03_满载冷却过程", "case03_cooldown_full", "cooling after full load", "cooling"),
    ("case04_半负载", "case04_half_load", "half CPU load", "steady"),
    ("case05_半负载冷却过程", "case05_cooldown_half", "cooling after half load", "cooling"),
]


def load_case_frames(case_dir: Path) -> tuple[list, list[str]]:
    files = sorted((case_dir / "data").glob("HM*.csv")) or sorted(case_dir.rglob("HM*.csv"))
    frames, errors = [], []
    for f in files:
        try:
            frames.append(read_hm_csv(f))
        except HMParseError as exc:
            errors.append(str(exc))
    return frames, errors


def qc_figure(
    case_id: str,
    det_raw: np.ndarray,
    rot90: int,
    row0: int,
    col0: int,
    size: tuple[int, int],
    cropped_stack: np.ndarray,
    t_rel: np.ndarray,
    jpeg_path: Path | None,
    out_png: Path,
) -> None:
    h, w = size
    rotated = np.rot90(det_raw, rot90)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    im0 = axes[0].imshow(rotated, cmap="hot")
    axes[0].add_patch(Rectangle((col0 - 0.5, row0 - 0.5), w, h, ec="cyan", fc="none", lw=1.5))
    axes[0].set_title(f"{case_id}: raw (rot90={rot90}) + window")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    det_crop = rotated[row0 : row0 + h, col0 : col0 + w]
    im1 = axes[1].imshow(det_crop, cmap="hot")
    axes[1].set_title(f"cropped {h}x{w} (detection frame)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    if jpeg_path is not None and jpeg_path.exists():
        axes[2].imshow(np.asarray(Image.open(jpeg_path)))
        axes[2].set_title("camera JPEG (same frame)")
    else:
        axes[2].text(0.5, 0.5, "no JPEG", ha="center", va="center")
    axes[2].axis("off")

    axes[3].plot(t_rel / 60.0, cropped_stack.max(axis=(1, 2)), label="max")
    axes[3].plot(t_rel / 60.0, cropped_stack.mean(axis=(1, 2)), label="mean")
    axes[3].set_xlabel("t (min)")
    axes[3].set_ylabel("T (°C)")
    axes[3].set_title("cropped-window temperature")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "crop.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    size = tuple(cfg["size"])
    border_px = int(cfg.get("border_px", 8))
    max_dev = int(cfg.get("max_deviation_px", 3))
    overrides = cfg.get("overrides") or {}

    print(f"raw dir: {RAW_DIR}")
    all_frames: dict[str, list] = {}
    all_errors: dict[str, list[str]] = {}
    for zh, cid, _label, _kind in CASES:
        frames, errors = load_case_frames(RAW_DIR / zh)
        all_frames[cid], all_errors[cid] = frames, errors
        print(f"  {cid}: parsed {len(frames)} frames, {len(errors)} errors")
        for e in errors:
            print(f"    PARSE ERROR: {e}")

    # Reference window from the hot case
    ref_case = cfg["reference"]["case"]
    ref_idx = int(cfg["reference"]["frame"])
    ref_frame = all_frames[ref_case][ref_idx].temps
    ref_amb = estimate_ambient(ref_frame, border_px)
    ref_r, ref_c, ref_frac = find_crop(ref_frame, size, ref_amb)
    ref_patch = apply_crop(ref_frame, ref_r, ref_c, size)
    print(f"reference window ({ref_case}[{ref_idx}]): row0={ref_r} col0={ref_c} "
          f"captures {ref_frac * 100:.1f}% of excess energy")

    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tdp_version": tdp.__version__,
        "raw_dir": RAW_DIR.name,
        "crop": {
            "size": list(size),
            "reference": {"case": ref_case, "frame": ref_idx, "row0": ref_r, "col0": ref_c},
            "max_deviation_px": max_dev,
        },
        "cases": [],
    }

    n_total = n_ok = 0
    for zh, cid, label, kind in CASES:
        frames = all_frames[cid]
        errors = all_errors[cid]
        n_total += len(frames) + len(errors)
        n_ok += len(frames)
        if not frames:
            print(f"!! {cid}: no frames, skipping")
            continue

        det_idx = 0 if kind == "cooling" else len(frames) - 1
        det_frame = frames[det_idx].temps
        rot90, needs_review, zncc = 0, False, None

        if cid in overrides:
            ov = overrides[cid]
            row0, col0, rot90 = int(ov["row0"]), int(ov["col0"]), int(ov.get("rot90", 0))
            method = "override"
        elif kind == "calib":
            reg = register_crop(det_frame, ref_patch, size)
            rot90, row0, col0, zncc = reg.rot90, reg.row0, reg.col0, reg.zncc
            needs_review = True
            method = "zncc_registration"
            print(f"  {cid}: registration rot90={rot90} row0={row0} col0={col0} "
                  f"zncc={zncc:.3f} (candidates: "
                  + ", ".join(f"k={k}:{v[2]:.3f}" for k, v in sorted(reg.candidates.items()))
                  + ")")
        else:
            det_r, det_c, frac = find_crop(det_frame, size, estimate_ambient(det_frame, border_px))
            deviation = max(abs(det_r - ref_r), abs(det_c - ref_c))
            if deviation <= max_dev:
                # Same fixed scene: force the shared reference window so pixels
                # correspond 1:1 across cases (sensor coords are cross-case).
                row0, col0 = ref_r, ref_c
                method = f"reference (detected ({det_r},{det_c}), dev {deviation}px)"
            else:
                row0, col0 = det_r, det_c
                method = "excess_energy"
                needs_review = True
                print(f"  !! {cid}: window ({row0},{col0}) deviates {deviation}px "
                      f"from reference ({ref_r},{ref_c}) — flagged for review")

        raw_stack = np.stack([np.rot90(f.temps, rot90) for f in frames])
        cropped = apply_crop(raw_stack, row0, col0, size).astype(np.float32)
        t_amb = np.array([estimate_ambient(f.temps, border_px) for f in frames], np.float32)
        epochs = np.array([f.timestamp.timestamp() for f in frames], np.int64)
        t_rel = (epochs - epochs[0]).astype(np.float64)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        npz_path = OUT_DIR / f"{cid}.npz"
        np.savez_compressed(
            npz_path,
            T=cropped,
            t_rel_s=t_rel,
            timestamp_epoch=epochs,
            t_amb=t_amb,
            frame_min=cropped.min(axis=(1, 2)),
            frame_max=cropped.max(axis=(1, 2)),
            frame_mean=cropped.mean(axis=(1, 2)),
            crop_row0=row0,
            crop_col0=col0,
            rot90=rot90,
        )

        det_stem = frames[det_idx].path.stem
        jpeg = frames[det_idx].path.parent.parent / "photo" / f"{det_stem}.jpeg"
        if not jpeg.exists():
            jpeg = jpeg.with_suffix(".jpg")
        qc_figure(cid, det_frame, rot90, row0, col0, size, cropped, t_rel,
                  jpeg if jpeg.exists() else None, QC_DIR / f"{cid}_crop.png")

        manifest["cases"].append({
            "case_id": cid,
            "source_dir_zh": zh,
            "label": label,
            "kind": kind,
            "n_frames": len(frames),
            "parse_errors": errors,
            "encodings": sorted({f.encoding for f in frames}),
            "crop": {"row0": row0, "col0": col0, "rot90": rot90, "method": method,
                     "zncc": zncc, "needs_review": needs_review},
            "time": {"start": frames[0].timestamp.isoformat(),
                     "end": frames[-1].timestamp.isoformat(),
                     "duration_s": float(t_rel[-1]),
                     "median_dt_s": float(np.median(np.diff(t_rel))) if len(t_rel) > 1 else None},
            "t_amb_median_c": float(np.median(t_amb)),
            "crop_max_c": float(cropped.max()),
            "crop_min_c": float(cropped.min()),
            "camera_meta_first_frame": frames[0].meta,
            "npz": npz_path.relative_to(ROOT).as_posix(),
            "files": [f.path.name for f in frames],
        })
        print(f"  {cid}: {len(frames)} frames -> {npz_path.name} "
              f"[({row0},{col0}) rot90={rot90} {method}], "
              f"T_crop {cropped.min():.1f}..{cropped.max():.1f} °C, "
              f"amb~{np.median(t_amb):.1f} °C")

    write_manifest(OUT_DIR / "manifest.json", manifest)
    print(f"\nself-check: {n_ok}/{n_total} frames parsed & verified")
    print(f"manifest: {(OUT_DIR / 'manifest.json').relative_to(ROOT)}")
    print(f"QC figures: {QC_DIR.relative_to(ROOT)}/")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
