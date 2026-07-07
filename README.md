# TDP — Thermal-field reconstruction from sparse trusted sensors

**Physics-constrained reconstruction of chip/board temperature fields from K sparse
point measurements, toward on-device thermal self-monitoring.**

Dense, accurate temperature maps drive dynamic thermal management (throttling,
reliability) — but the two available instruments each deliver only half:

- **IR thermography** is dense but *biased*: PCB surfaces mix emissivities from
  ~0.05 (connector shields) to ~0.95 (solder mask). In our own measurements the
  USB/HDMI shields read near-ambient while sitting on a 60 °C board — and inside
  a closed server there is no optical access at all.
- **Contact / on-die sensors** are accurate but *sparse* (a handful per board).

TDP trains a small (~1M-param) **set-invariant attention operator** that maps any
K sensor readings → the full temperature field, regularized by the steady heat
equation. It is pretrained on domain-randomized FDM scenarios, then fine-tuned
per board on IR measurements calibrated with black tape (ε≈0.95): *a general
physical prior, cheaply calibrated to each board*. The same network is small
enough to run on the monitored device itself.

This is the TFR-HSS problem class (Chen et al., *Sci China Inf Sci* 2022,
[arXiv:2108.08298](https://arxiv.org/abs/2108.08298)); see `docs/story_zh.md`
for the full (Chinese) competition narrative and literature anchors.

## Method

1. **Nondimensional contract** (`tdp.model.normalization`): θ = (T − T_amb)/ΔT_s
   with ΔT_s from the sensor inputs alone; coordinates normalized by the board's
   long side; the steady PDE collapses to ∇̃²θ − ĥθ + Q̂ = 0 with a single
   physics parameter ĥ = hL²/(k_eff·d). Linearity ⇒ amplitude extrapolation
   ("severe conditions") holds by construction.
2. **Operator** (`tdp.model.operator`): sensor set → Fourier features + self-attention
   (permutation/count invariant) → cross-attention from query coordinates → θ(x,y),
   plus one condition token [log ĥ, log γ, aspect, BC].
3. **Pretraining** (`tdp.train.pretrain`): 4096 randomized rectangular scenarios
   (aspect, ĥ ∈ log-U[0.3,30] — range measured from our board, BC type, 1–5
   analytic sources), random-K + board-layout sensor curriculum with measured
   noise levels, MSE + self-normalized autograd PDE residual (λ ramped to 1e-3).
   A λ=0 twin provides the physics ablation.
4. **Per-board fine-tuning** (M7, upcoming): trust-masked dense IR supervision
   with tape pixels held out for validation, learnable ĥ checked against the
   measured bound, 50% synthetic replay against forgetting.

## Real dataset

One session (2026-07-04) of a bare **Raspberry Pi 4B** under a HIKMICRO camera
(192×256 px, ~6.3 s cadence, ε set 0.93, 0.1 m): unplugged / idle / full load /
cooling / half load / cooling — 336 CSV+JPEG frames. The ingestion pipeline
parses (mixed GBK & UTF-8-BOM encodings!), self-checks every frame against the
camera's own stats block, crops the fixed 157×103 board window (≈0.543 mm/px),
and extracts physics: full-load hotspot **79.2 °C**, cooling τ ≈ 411–565 s,
per-pixel repeatability 0.07–0.38 °C, tape-emissivity systematic ≤ +0.92 °C.
Raw data lives on the `data` branch under `实验数据（黑胶带版）/`.

## Quickstart

```bash
uv sync                                    # installs the tdp package (editable)
uv run pytest                              # parser + FDM + normalization tests
uv run scripts/ingest_real_data.py         # raw CSVs -> data/processed/real/*.npz + QC PNGs
uv run scripts/analyze_physics.py          # steady/cooling/h-hat extraction -> physics.json
uv run scripts/run_pretrain.py --small     # ~3 min smoke pretrain (any device)
uv run scripts/run_pretrain.py             # full pretrain (GPU recommended)
uv run scripts/run_pretrain.py --twin-nopde  # λ=0 ablation twin
```

Colab: clone the repo, `pip install -e .`, then run the same
`python scripts/run_pretrain.py` — device autodetects; checkpoints land in `models/v2/`.

## Repository layout

```
src/tdp/          io (HM CSV parser, manifest) · data (crop, steady, cooling, radiometry)
                  sim (nondim FDM, scenario randomization) · model (operator, normalization)
                  train (losses, pretraining) · eval, deploy (upcoming)
scripts/          ingest_real_data · analyze_physics · run_pretrain · (finetune/eval/bench upcoming)
configs/          crop.yaml · pretrain_v2.yaml · (sensors_board.json after annotation)
docs/             story_zh.md (competition narrative) · m3_data_requirements.md · data_card.md
reports/qc/       crop QC + physics figures        models/v1|v2/   checkpoints
legacy/           v1 per-scenario PINN baselines (kept runnable)    AI_TDP.ipynb (v1 provenance)
```

## Status

M0 scaffold ✅ · M1 ingestion (336/336 verified) ✅ · M2 physics extraction ✅ ·
M3 sensor annotation (awaiting tape positions + logs) · M4 story docs (draft) ·
M5 pretraining (code + smoke ✅, full run pending) · M6 zero-shot gate → M10 (see plan).

## License

MIT (see `LICENSE`).
