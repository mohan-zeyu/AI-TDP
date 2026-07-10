# Measurement Targets — what each item is actually for

> **STATUS: HISTORICAL (fulfilled 2026-07-10).** The targets below were met by
> sessions 1–2; results live in `story_zh.md` §3–4 and `data_card.md`.

Companion to `docs/m3_data_requirements.md` (the *how*). This file is the *why*:
each measurement exists because one specific claim, figure, or pipeline stage
consumes it. If a row's target doesn't matter to you, cut that row — nothing
else silently depends on it. The last section gives the minimum viable set.

Claims the project makes (for reference):
- **C1 accuracy** — from K sparse trusted points, the model reconstructs the field
  (validated at held-out locations, vs interpolation baselines).
- **C2 severity extrapolation** — fine-tuned on mild conditions, correct at full load.
- **C3 physics consistency** — the learned physical parameter matches independent
  measurements; the PDE prior demonstrably helps.
- **C4 the premise** — raw IR genuinely misleads where emissivity is low; sparse
  *trusted* sensing + reconstruction is the fix.
- **C5 deployability** — small model, real-time on the device itself.

## Thermal captures

| Capture | What it physically measures | Consumed by | If skipped / shortened |
|---|---|---|---|
| **Half load, ≥20 min to plateau** | the board's steady field at the *training* condition, with enough plateau time for truly-steady frames (honest labels + noise map) | fine-tune training (C1); per-pixel σ weighting | drifting labels, tiny effective sample — session 1's weakness repeats |
| **Full load, ≥20 min** | the steady field at the *severest* condition — never used in training | the one-shot extrapolation test (C2), headline number | test target carries a drift caveat (session 1: only 2 quasi-steady frames) |
| **Cooling after each load, ≥20 min (~2.5 τ)** | the board's thermal time constant τ, twice independently | τ(full) ≈ τ(half) is the physics cross-check (C3); ĥ prior support; late frames = quasi-static validation | τ stays an extrapolated guess; session 1's 31% τ discrepancy stays unexplained |
| **Idle, ~10 min** | a low-amplitude operating point | training-condition diversity (C1); exercises the ΔT_s floor | mild loss of condition range — cheapest row to cut |
| **Cold isothermal (board off ≥3 h, same tripod)** | with the board uniformly at room temp, every apparent temperature variation = pure emissivity/reflection artifact → per-pixel artifact map | trust mask refinement (which pixels supervise fine-tuning), bias correction (C4) | trust mask is hand-drawn from the photo instead of measured — coarser, still workable |
| **One bare-board full load** *(optional, high value)* | the same steady state with tape absent: shows the shiny SoC lid under-reading in raw IR; (taped − bare) at matched states = *measured* emissivity-error field | the strongest C4 figure (raw IR vs diode vs reconstruction at the key unit) | C4 rests on the connector artifact + literature values instead of a direct measurement |

## Records & auxiliary channels

| Item | What it provides | Consumed by | If skipped |
|---|---|---|---|
| **Tape-layout photo (ruler in frame) / visible-light export** | pixel-accurate outline of trusted (tape) regions and component positions | `configs/sensors_board.json` + trust/source masks — the fine-tune literally cannot be specified without knowing where the tape is | I guess regions from thermal contrast; misclassified pixels poison supervision |
| **SoC diode log (2 s) + one time anchor** | an *independent instrument* reading a temperature the camera also infers | breaks the circularity "the same camera both supervises and validates" (C3); live input channel for the on-device demo (C5); ties bare/taped runs together | validation rests entirely on IR-at-tape — the first question a sharp judge will ask |
| **Workload commands (what "half" means, GPU on/off)** | condition labels + reproducibility | data card, report methods section; interpreting source amplitudes | conditions are folder-name-level labels only; extrapolation claim gets fuzzier |
| **Environment & mounting (airflow, orientation, surface)** | validity conditions for the constant-h convection model | report's assumptions section; explains anomalies if any | a judge asking "was there a draft?" has no answer on record |
| **Camera model + accuracy spec** | the absolute-accuracy denominator (±2 °C class) | uncertainty budget — without it no absolute error bar can be stated | relative claims only |
| **Session log sheet (one line per event)** | the timeline that aligns frames ↔ logs ↔ actions | ingestion labeling; forensics when a curve looks odd | any anomaly becomes undiagnosable after the fact |
| **Aluminum-foil apparent temperature** *(optional)* | the reflected-temperature term for radiometric correction | makes low-ε bias correction quantitative instead of assumed (C4) | reflected temp assumed = room temp (reasonable, unverified) |
| **USB-C power meter at each load** *(optional)* | electrical power in = total source amplitude ground truth | power-in vs temperature-field consistency (C3); source term sanity | Q̂ amplitudes remain relative, not absolute |
| **One contact-probe reading at a tape patch** *(optional)* | absolute anchor from a third instrument | upgrades "tape = truth" from assumption to verified (C3/C4) | tape trust rests on ε≈0.95 literature value (fine, weaker) |
| **USB-traffic load case** *(optional)* | a steady state where the heat source *moves* (VL805 instead of SoC) | source-layout generalization test (C1 stretch) | generalization claim limited to amplitude, not layout |

## What session 1 (2026-07-04) already covers — dedup guidance

Session 1 is the **primary training/test dataset**, not a discarded draft: half
load (plateaued, 41 steady frames) + idle = fine-tune training; full load = the
untouched severity test; cool-downs = τ (truncated) + quasi-static validation.
It becomes fully usable the moment two things are handed over — the tape-layout
photo and the SoC log files. No new measurement is a prerequisite for M3–M7.

Consequences for the new session (dedup):
- **drop idle** (already have it);
- **drop the second cooling** — one ≥20′ cooling after full load fixes τ;
- **half load = optional replication** — session 1's half load stays the primary
  training condition; a fresh one only buys a cleaner within-session replication;
- **keep**: cold isothermal (irreplaceable, 2 min), full load ≥20′ (clean test
  target), one cooling ≥20′, optional bare full-load (the C4 demo);
- new session additionally unlocks the **cross-session robustness test**
  (train on one day, test on the other) — a claim no single session can make.

## Minimum viable set

If you cut everything cuttable, the pipeline still runs end-to-end with exactly:
**half load 20′ + full load 20′ + one cooling 20′ + tape-layout photo + SoC log.**

Everything else upgrades one named claim: cold isothermal → measured trust mask
(C4); second cooling → the τ cross-check (C3); bare full-load → the premise
demonstrated at the SoC itself (C4); contact probe / power meter / foil →
absolute anchors (C3/C4); idle / USB case → condition and layout range (C1).

Cut by deciding which claim you're willing to weaken — not by effort estimates:
the entire optional tier above costs less than one extra cooling run.
