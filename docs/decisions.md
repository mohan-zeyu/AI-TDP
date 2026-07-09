# Decision Log

Dated, append-only. Each entry: decision → why → where it lives.

**2026-07-06 — Story: sparse trusted sensors → full field, self-monitoring outlook.**
Merges data-center urgency (motivation), board experiment (validation), on-device
inference (demo). Matches the existing attention operator; TFR-HSS literature
anchor. → `docs/story_zh.md`, README.

**2026-07-06 — Nondimensional contract; no dataset stats in checkpoints.**
θ = (T−T_amb)/ΔT_s (inputs-only computable), coords by long side, single ĥ.
v1's global T_mean/T_std were a leakage channel. → `tdp/model/normalization.py`,
`docs/training_protocol.md` §1.

**2026-07-07 — Shared crop window (61,42) across the load session.**
Per-case detections agreed within 2 px; forcing the reference guarantees 1:1
pixel correspondence across cases (sensor coordinates are cross-case).
→ `scripts/ingest_real_data.py`.

**2026-07-07 — case00 usable for orientation only.**
Re-staged (180°, ~6% scale) → no pixel-aligned bias map from session 1; replaced
by session-2 cold isothermal. → `data_card.md`, `physics.json` notes.

**2026-07-07 — ĥ reported as a bound (≲3), not a value; pretrain range log-U[0.3,30].**
Warm dome fills the board (no far field); pointwise Laplacian below noise. The
two degenerate estimators are reported as diagnostics. → `analyze_physics.py`,
`configs/pretrain_v2.yaml`.

**2026-07-07 — τ discrepancy (411 vs 565 s) reported as physics.**
Consistent with temperature-dependent natural convection + windows ~0.5τ;
longer cool-downs in session 2. → `physics.json`, `data_card.md`.

**2026-07-07 — Self-normalized PDE residual.**
Raw residual made λ·L_pde ≈ L_data (silent rebalancing); relative residual keeps
it a light regularizer at any ĥ. → `tdp/train/losses.py`,
`docs/training_protocol.md` §2.3.

**2026-07-07 — Pretraining on Colab; MPS verified for smoke.**
User's compute choice; double-backward works on MPS, so local smoke gates the
code before GPU time. → `scripts/run_pretrain.py`.

**2026-07-08 — Session 1 recognized as the TAPED configuration.**
Large sheets over the processors: hotspot 79.2 °C is an on-tape trusted reading;
trusted region = whole sheet. Bare BCM2711 lid is shiny → bare IR under-reads the
key unit (the story's core artifact). → `data_card.md`, `story_zh.md`.

**2026-07-08 — Session 2 = bare/taped A/B design, bare first.**
Bare: problem demo + (via matched steady states) measured per-pixel
emissivity-error field. Tape added mid-session without moving anything; board
fixed to surface; SoC diode logged in both configs. → `m3_data_requirements.md`
Part B.

**2026-07-08 — Held-out unit = validation patches, not the tape region.**
With sheet coverage, excluding all tape would starve the hotspot supervision.
Three pixel roles: input sensors / dense trusted supervision (incl. tape) /
~12 stratified validation patches (never input, never supervised).
→ `docs/training_protocol.md` §4.2; corrected in README/story/plan.

**2026-07-08 — Tape protocol tiered (5 core + optional).**
User: per-board calibration must stay light; effort on units that matter. Tape
is lab-only instrumentation; count affects validation statistics only.
→ `m3_data_requirements.md` B1.

**2026-07-10 — v2.5 in-context board conditioning adopted, before the full pretrain.**
User wanted an explicit board representation for cross-board transfer
(contrastive-embedding instinct). Chosen mechanism: boards get multiple
operating states (shared layout + LU, re-drawn amplitudes); 1–2 reference
frames enter as typed context tokens; context/cond dropout keeps v2 mode
intact; fair-K dual validation; K-curve down to K=2. Contrastive kept as
optional auxiliary (discriminative ≠ task-sufficient; global vector bottlenecks
layout). Timing: architecture change is nearly free before the expensive
pretrain. Smoke: ctx 0.296 vs no-ctx 0.349 val RMSE @12 ep.
→ `docs/board_transfer_architecture.md`, `tdp/{sim,model,train}`.
