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

**2026-07-11 — Physics-ablation finding (λ=0 twin, same seed/data): the PDE
residual is not the load-bearing element; the physics enters via the data.**
Synthetic in-distribution: tied (val 0.0773 vs 0.0768; K-curve +1–7% for PDE).
Real zero-shot: PDE better in 3/4 settings, decisively in the sensor-starved
regime (no-ctx hotspot 4 px vs 34 px). Real after Tier-1: within test noise
(case02 3.08 vs 3.01 °C; patches favor PDE 1.57 vs 1.69 °C). Interpretation for
the report: the pipeline's "physics-informed" content lives in the FDM-generated
training distribution, the nondimensional contract, and measured parameter
bounds — the residual term adds a robustness margin where information is
scarce, not a headline accuracy gain. Stated plainly; the twin table is the
evidence. → M8 ablation column, `models/boards/pi4b_s1_nopde_report.json`.

**2026-07-12 — Shipped checkpoint stays pretrain_v2 (PDE), adjudicated with 5 seeds.**
User asked why not the λ=0 twin ("better in our tests"). Single-seed test edge
was seed noise: 5 card refits per backbone give case02 3.010±0.157 (main) vs
2.999±0.048 (twin) — tie; case06 3.150±0.144 vs 3.189±0.152 — ordering FLIPPED
vs the single-seed run; selection metric (patches) favors main 1.650±0.079 vs
1.703±0.038. Plus main's clear robustness edge in sensor-starved regimes
(hotspot 4 vs 34 px, no-ctx). Choosing by test-case numbers would be test-set
selection — the discipline exists precisely for this moment.

**2026-07-11 — Fine-tuning = board-token adaptation, not full-weight training.**
User challenged full fine-tuning; with v2.5 the board is an explicit input, so
adaptation moves to representation space: freeze the 1.06M backbone, learn ~8
board tokens (+ log ĥ, log γ) ≈ 1k params, warm-started from encoded context.
Matches n_eff ≈ 30 real frames; forgetting impossible → replay unnecessary;
deployable "board card" = few KB. Head-only unfreeze = fallback; full FT with
replay = M8 upper-bound ablation only. Eval discipline unchanged (validation
patches, λ grid incl. 0, untouched test cases). → `training_protocol.md` §4.3.

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
