# Data Card — IR measurement sessions

## Session 1 — 2026-07-04 (black-tape version, `实验数据（黑胶带版）/`)

| Item | Value | Source |
|---|---|---|
| Board | Raspberry Pi 4B, no heatsink/fan; **large black-tape sheets over the central processor region (SoC + adjacent)** during all captures | user photo 2026-07-08 |
| Camera | HIKMICRO (exact model **pending A5**), 192×256 px | file format |
| Cadence / distance / ε setting | ~6.3 s · 0.1 m · 0.93 (reflected temp set 25.0 °C) | CSV metadata |
| Cases | case00 unplugged (12 f, 20:39, **re-staged**) · case01 idle (12) · case02 full load (120) · case03 cooling (36) · case04 half load (120) · case05 cooling (36) | folders |
| Board window | 157×103 px at (row 61, col 42), shared across cases 01–05 (detected deviation ≤2 px); ≈0.543 mm/px | ingest |
| Ambient (border median) | 25.9–26.7 °C across session | ingest |
| Full-load hotspot | 79.2 °C at crop (95, 40) (SoC) — read **on tape**, i.e. a trusted value (known bias +0.5–0.9 °C) | canonical field |
| Cooling constants | τ = 411 s (post-full) / 565 s (post-half), R² ≈ 0.98, spans only ~0.5 τ | physics.json |
| Repeatability | per-pixel σ median 0.07 °C (short window) – 0.38 °C (incl. drift) | physics.json |
| ĥ = hL²/(k_eff d) | upper bound ≲ 3 (no far field on board; Laplacian below noise) | physics.json |
| Tape systematic (ε 0.95 vs set 0.93) | +0.17 °C @33 °C … +0.92 °C @79 °C | radiometry model |

**Known limitations:** case00 shot 2 h later with tripod moved (180° rotation +
~6% scale) → per-pixel bias map deferred; full load still drifting ~0.2 °C/min at
12.5 min (canonical field = quasi-steady tail, n_eff = 2); cooling windows cover
only ~half a time constant; steady frames are near-duplicates (n_eff 2–11).

**Pending records (A1–A6, see `docs/m3_data_requirements.md`):** tape positions &
tape type · SoC temperature logs + clock sync · workload commands · airflow &
mounting orientation · camera model/accuracy · competition deadline.

## Session 2 — planned (weekend re-measurement)

Protocol in `docs/m3_data_requirements.md` Part B: 8–10 tape spots (incl. one
connector shield), cold isothermal set with untouched tripod (per-pixel bias
map), ≥20 min loads and cool-downs, SoC log at 2 s with sync anchor, session log
sheet. Optional: USB-C power readings, contact-probe spot checks, USB-load case.
