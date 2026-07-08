# Training Protocol — Pretraining, Fine-tuning, and Testing (v2)

The authoritative spec for how the operator is trained and evaluated. Code:
`src/tdp/{sim,model,train}`, `scripts/run_pretrain.py`, (upcoming) `scripts/run_finetune.py`,
`scripts/run_eval.py`. Configs: `configs/pretrain_v2.yaml`, (upcoming) `configs/finetune_board.yaml`.

```
                 ┌─ SYNTHETIC (M5) ────────────────────────────┐
 FDM scenarios ──► sample K sensors ──► operator ──► θ̂(x,y)    │ loss = MSE(θ̂, θ)
 (randomized)    │  (+ noise, layouts)   + cond token           │      + λ·PDE residual
                 └──────────────────────────────────────────────┘
                                   │ pretrained weights
                                   ▼
                 ┌─ REAL BOARD (M6 gate → M7) ─────────────────┐
 IR frames ──────► K sensors from trusted sites ─► operator ───► θ̂ everywhere
 (idle+half)     │ supervision: dense trusted px (− patches)   │ + source-free PDE (learnable ĥ, γ)
                 │ 50% synthetic replay                        │ early stop on validation patches
                 └──────────────────────────────────────────────┘
                                   │ fine-tuned board model
                                   ▼
                 TEST (once): full-load case ─ sensors in, field out, scored on trusted px
```

## 1. The nondimensional contract (shared by every stage)

- Excess temperature **θ = (T − T_amb) / ΔT_s**; `T_amb` = frame border median
  (real) or 0 (synthetic); **ΔT_s = max sensor excess**, floored at 1 °C (real).
  ΔT_s is computable from the model's own inputs → valid at inference.
- Coordinates normalized by the board's long side: domain [0, a] × [0, 1];
  Pi 4B crop a = 103/157 ≈ 0.656 (≈ 0.543 mm/px).
- Steady physics: **∇̃²θ − ĥ·θ + Q̂ = 0**, ĥ = h·L²/(k_eff·d). One parameter.
- Because the PDE is linear with homogeneous BCs, scaling the source scales θ
  linearly and ΔT_s divides it out: **amplitude ("severity") extrapolation is
  exact by construction**; only nonlinear effects (h(ΔT), source redistribution)
  remain — they are what the full-load test measures.
- Checkpoints carry model config only — never dataset statistics.

## 2. Pretraining (M5) — learn the *family* of board fields

### 2.1 Scenario distribution (`configs/pretrain_v2.yaml`)

| Parameter | Distribution | Rationale |
|---|---|---|
| aspect a | U[0.5, 1.0]; 25% pinned to 0.656 | generalize across board shapes; board aspect always in-distribution |
| ĥ | log-U[0.3, 30] | M2 measured bound ĥ ≲ 3 on the Pi 4B; range centers the bound, covers v1's 5 |
| BC | 70% Robin (−∂θ/∂n = γθ, γ log-U[0.1,10]), 30% Dirichlet | real board edges are convective; Dirichlet keeps v1 compatibility |
| sources | 1–5; 70% rotated anisotropic Gaussians (σ_maj U[0.04,0.15], ratio U[0.4,1]), 30% smooth rectangles (half-extent U[0.03,0.15], tanh edge 0.012) | components are blobs and rectangles; smooth edges keep the autograd residual meaningful |
| amplitudes | relative U[0.2, 1] | multi-source contrast |
| grid | ny=96, nx=round(96·a), fresh sparse-LU solve per scenario (~ms) | resolution ≥ real crop scale |
| dataset | 4096 train / 128 val, seeds fixed | laptop/Colab-friendly |

### 2.2 Sensor curriculum (per training item, re-sampled every epoch)

- 80%: random placement, K ~ U{4..16}.
- 20%: the board sensor-site layout (`configs/sensors_board.json` once annotated;
  placeholder in `tdp.sim.scenarios` until then), jittered ±2 px equivalent —
  closes the layout gap before fine-tuning ever starts.
- Additive Gaussian reading noise, σ ~ U[0, 0.1] in normalized units — brackets
  the measured per-pixel repeatability (0.07–0.38 °C against ΔT_s ≥ several °C).
- Per-sample scale s = max noisy sensor value; inputs (x, y, θ/s), targets θ/s.
- Resampling rule: if no sensor sees ≥ 25% of the field peak, resample (≤3
  tries). Rationale: an all-blind sensor set makes the amplitude unidentifiable
  and destabilizes normalized targets; real layouts always cover the hot units.

### 2.3 Losses

- Data: `L_data = MSE(θ̂, θ/s)` at 384 random query points.
- Physics: relative residual at 192 collocation points (margin 0.02):
  `R = (∇̃²θ̂ − ĥθ̂ + Q̂/s) / (|∇̃²θ̂| + |ĥθ̂| + |Q̂/s| + 1)` (denominator detached).
  Self-normalization keeps L_pde = mean R² at O(1) across the ĥ range and all
  training stages, so λ is a genuine light-touch regularizer (measured: λ·L_pde
  ≈ 6·10⁻⁴ vs L_data ≈ 0.1 — v1's un-normalized version silently reached parity
  with the data loss).
- Schedule: λ = 0 for 10 epochs, linear ramp to 1e-3 over 20, then constant.
- Total: `L = L_data + λ(t)·L_pde`.

### 2.4 Optimization & artifacts

AdamW lr 1e-3, wd 1e-5, cosine over 250 epochs, grad-clip 1.0, batch 32
(variable-K padded with key-padding mask; condition token always valid).
Best-val checkpoint → `models/v2/pretrain_v2.pt` = {model_state, model_config,
train/scenario configs, val RMSE}; history + K-curve → `*_history.json`.

### 2.5 The ablation twin

`--twin-nopde` trains λ=0 from the same seed and data → the ±physics comparison
is a controlled twin, not a historical notebook.

### 2.6 Acceptance (gate to M6) & how to run

- FDM O(h²) manufactured-solution test green (`tests/test_fdm.py`); ✅
- smoke run (`--small`, ~3 min) K-curve monotone; ✅ (0.32→0.25 for K=4→16)
- full run: val RMSE(θ′) stable, K-curve monotone, amplitude-invariance exact.
- Local smoke: `uv run scripts/run_pretrain.py --small` (MPS double-backward
  verified working). Full: same command on Colab GPU (`pip install -e .` first),
  then the twin.

## 3. Zero-shot gate (M6) — before any fine-tuning

Inputs: pretrained model, canonical half-load field, annotated sensor sites.
Feed K tape-site sensor readings (real values), predict the full field, score at
the validation patches, compare against RBF interpolation given the same K
readings. **Decision rule:** model ≥ RBF → proceed to M7; model ≪ RBF → the
normalization/coverage is broken; fix before touching fine-tuning. Either
outcome is reported.

## 4. Fine-tuning (M7) — calibrate the prior to *this* board

### 4.1 Data

- Train: case01 (idle) + case04 (half load) steady frames, subsampled ≥30 s
  apart (n_eff honest), plus per-pixel-noise resampling of the canonical fields
  as augmentation.
- Never used in training: case02 (full load, test), late-cooling frames
  (quasi-static validation), validation patches (below).

### 4.2 The three pixel roles (per frame)

| Role | What | Where from |
|---|---|---|
| **Input sensors** (K ≈ 5–8) | (x, y, T) fed to the model | designated trusted sites from annotation — SoC tape site, RAM, PMIC, solder-mask, cold corner |
| **Supervision** (dense) | targets for L_data, weight 1/σ² (per-pixel std map) | trusted mask = tape regions (**core supervision — the sheets over the processors**) + solder mask; minus shiny metal; minus validation patches |
| **Validation patches** (~12) | never an input, never supervised; drive λ selection, early stopping, spatial metrics | stratified over the trusted area: on-tape hot, on-tape mid, solder-mask mid, cold corner, near-edge |

Trust mask construction: annotated tape-region polygons + solder-mask default,
minus annotated shiny-metal regions (connectors; bare SoC in config A); refined
by the session-2 cold-isothermal bias map when available.

### 4.3 One training step

1. Pick a real frame (50%) or a fresh synthetic scenario (50%, replay —
   anti-forgetting; we own the simulator, replay strictly dominates freezing).
2. Real branch: ambient = border median; pick K sensors from the sites (jitter
   ±1 px, use 3×3 median values); ΔT_s from those readings; normalize.
3. Queries: ~512 pixels sampled from the supervision mask ∝ 1/σ²-weighted MSE.
4. PDE: collocation points outside dilated component rectangles
   (`configs/source_mask_board.json`); residual uses **learnable log ĥ and
   log γ** (fed through the condition token; init from M2: log ĥ₀ = 0, weak
   prior `λ_prior·(log ĥ − log ĥ₀)²` with λ_prior small — M2 only bounded ĥ ≲ 3).
5. Optimizer: AdamW, model lr 1e-4 → 1e-5 cosine; (log ĥ, log γ) lr 1e-2;
   300–1000 steps; early stop on validation-patch RMSE.
6. λ_pde ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2} — selected on validation patches only;
   λ = 0 in the grid means the physics is allowed to lose.

### 4.4 Acceptance (M7)

- validation-patch RMSE ≥ 20% better than zero-shot;
- fitted ĥ consistent with the measured bound (≲3) and the post-annotation
  source-masked re-estimate;
- synthetic-val regression < 10% (replay working).

## 5. Testing & reporting (M8)

- **Spatial:** validation-patch RMSE/MAE vs RBF (thin-plate), GP (RBF+White,
  fitted), bicubic, and time-budget-matched per-scenario PINN
  (`legacy/chip_thermal_pinn.py`) — all given the *identical* sensor sets.
- **Severity extrapolation (headline, touched once):** sensors read from case02
  full-load frames at the same sites → predict → score on the full trusted mask:
  field RMSE/MAE, **max-temperature error**, **hotspot localization error (px)**.
- **Ablations:** ±PDE (twin), ±pretraining (fine-tune from scratch), ±replay.
- **Independent channels:** SoC diode log (junction ≠ surface — compare with a
  fitted constant offset, never equated); session-2 contact-probe spot checks.
- **Session-2 A/B:** bare config = raw-IR-fails demo (bare SoC lid vs diode) and
  the measured per-pixel emissivity-error field (taped − bare at matched steady
  states); taped config = the main training/eval data, superseding session 1.
- Everything regenerates from one command (`scripts/run_eval.py` →
  `reports/figures/`).

## 6. Failure modes & planned fallbacks

| Symptom | Interpretation | Fallback |
|---|---|---|
| M6: zero-shot ≪ RBF | normalization/coverage broken | fix contract before M7; check ĥ range & aspect handling |
| λ=0 wins the grid | PDE invalid at this fidelity | report ablation honestly; physics still carries pretraining |
| patches on tape err ≫ patches on solder mask | trust mask or tape ε wrong | re-anchor with contact probe; tighten mask |
| fitted ĥ → bound edges | prior mis-set or residual mis-scaled | re-estimate with session-2 source masks (contour-integral form) |
| full-load error ≫ validation error | severity extrapolation genuinely hard | report as the finding; analyze h(ΔT) correction as future work |

## Symbols

θ excess temperature (normalized) · s / ΔT_s per-frame scale · ĥ = hL²/(k_eff d)
· γ edge Robin coefficient · a aspect ratio · K sensor count · λ PDE-loss weight
· σ per-pixel repeatability map · V validation patches.
