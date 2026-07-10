# Data Card — IR measurement sessions

## Session 1 — 2026-07-04 (black-tape version, `实验数据（黑胶带版）/`)

| Item | Value | Source |
|---|---|---|
| Board | Raspberry Pi 4B, no heatsink/fan; **large black-tape sheets over the central processor region (SoC + adjacent)** during all captures | user photo 2026-07-08 |
| Camera | HIKMICRO (exact model **pending A5**), 192×256 px | file format |
| Cadence / distance / ε setting | ~6.3 s · 0.1 m · 0.93 (reflected temp set 25.0 °C) | CSV metadata |
| Cases | case00 unplugged (12 f, 20:39, **re-staged**) · case01 idle (12) · case02 full load (120) · case03 cooling (36) · case04 half load (120) · case05 cooling (36) | folders |
| Workload | full load = 4× `yes > /dev/null &` (all 4 cores busy-loop; GPU idle); half load = 2× (assumed — to confirm); stopped via `killall yes` (clean step-off for τ fits). Caveat: surface 79.2 °C ⇒ junction ≈ 80 °C soft-throttle threshold — full-load source may have been DVFS-regulated near the plateau; check the SoC log for saw-toothing at ~80 °C | group member, 2026-07-08 |
| Board window | 157×103 px at (row 61, col 42), shared across cases 01–05 (detected deviation ≤2 px); ≈0.543 mm/px | ingest |
| Ambient (border median) | 25.9–26.7 °C across session | ingest |
| Full-load hotspot | 79.2 °C at crop (95, 40) (SoC) — read **on tape**, i.e. a trusted value (known bias +0.5–0.9 °C) | canonical field |
| Cooling constants | ~~τ = 411/565 s (zero-offset fits)~~ **superseded** — model mismatch; see Session 2 row: offset model, θ∞ ≈ 13 °C, two modes | physics.json |
| Repeatability | per-pixel σ median 0.07 °C (short window) – 0.38 °C (incl. drift) | physics.json |
| ĥ = hL²/(k_eff d) | upper bound ≲ 3 (no far field on board; Laplacian below noise) | physics.json |
| Tape systematic (ε 0.95 vs set 0.93) | +0.17 °C @33 °C … +0.92 °C @79 °C | radiometry model |

**Known limitations:** case00 shot 2 h later with tripod moved (180° rotation +
~6% scale) → per-pixel bias map deferred; full load still drifting ~0.2 °C/min at
12.5 min (canonical field = quasi-steady tail, n_eff = 2); cooling windows cover
only ~half a time constant; steady frames are near-duplicates (n_eff 2–11).

**Delivered since:** tape-layout photo ✓ · SoC log ✓ · workload commands ✓.
**Still pending (nice-to-have for the report):** tape type/layers · confirm half
load = 2× `yes` · OS version/bitness · power supply model · airflow & mounting
orientation · camera model/accuracy spec · competition deadline & format.

## Session 2 — 2026-07-09 (delivered early: extended runs + foil study)

| Item | Value |
|---|---|
| Cases | case06 full load 20 min (240 f) · case07 cooling 20 min (240 f) · case08 unplugged foil-background: smooth/rough × light on/off (16 f total) |
| Camera | re-staged ~13% closer: PCB = **178×118 px** (~0.478 mm/px); board mounted 180° vs session 1 (crops rotated back, window (52,33) via override) |
| Integrity | 496/496 frames parse + metadata self-check |
| Full-load plateau | **real** this time: peak 82.0 °C canonical, n_eff = 17 (vs 2 in session 1) |
| Cooling physics | decays to **idle-powered equilibrium θ∞ ≈ 13 °C excess** (board stays on after `killall yes`); offset fits R² ≈ 0.999; window-dependent τ ⇒ two thermal modes (fast ~140 s die/package, slow ~250 s PCB tail). Session-1 zero-offset τ values (411/565 s) were model-mismatch artifacts — superseded. |
| case08 purpose | reflected-temperature / emissivity radiometry (foil scenes); crops meaningless by design — analysis reads raw CSVs |
| SoC diode log (`tlog1.csv`, 2 s) | Pi clock was 2 h 25 min off (no RTC) — aligned by curve shape vs case06 (corr 0.943). **Diode max 82.7 °C; 36% of plateau ≥ 80 °C (soft throttle active); junction − IR-on-tape = −1.0 °C** |
| Tape layout (`assets/Taged_Chip.png`) | large sheet over SoC/RAM/PMIC + **all connector shields wrapped** → 8 annotated sites, trust masks, shield A/B (+14 °C vs session 1's bare shields) |

## Session 3 — optional remaining wishlist

Sessions 1–2 covered the essentials (user considers the A/B done: case06/07 =
taped arm, case08 = bare/foil radiometry arm). Still valuable if ever convenient:
a *loaded* bare-board run at the same tripod (pixel-aligned emissivity-error
field), a cold isothermal set aligned with a load session, contact-probe spot
checks, USB-C power readings, a second board (the transfer headline figure).

Original protocol (historical):

Protocol in `docs/m3_data_requirements.md` Part B — **A/B design**: config A bare
(cold isothermal + full load + cooling; demonstrates the bare-SoC-lid emissivity
failure), then tape applied without moving board/tripod, config B taped
(mirrors session 1 + tiered extra patches; the training/eval configuration).
Matched steady states give a measured per-pixel emissivity-error field; ≥20 min
loads and cool-downs; SoC log at 2 s in both configs with a sync anchor; session
log sheet; visible-light exports per config for annotation. Optional: USB-C
power readings, contact-probe spot checks, USB-load case.
