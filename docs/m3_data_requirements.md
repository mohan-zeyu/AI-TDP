# M3 Data Handover & Re-measurement Protocol

What to provide / measure, in what form, and why. Items are marked **[required]**,
**[recommended]**, or **[optional]**. Part D lists things explicitly NOT needed —
do not spend time on them.

---

## Part A — Records about the existing 2026-07-04 session (no new measurement)

### A1. Black-tape positions **[required]**
- For each tape patch that was on the board during the session:
  - **which component / where** (component name is enough: e.g. "SoC lid", "RAM",
    "USB controller", "PMIC area", "bare solder mask center-right"), and
  - **approximate patch size** (mm, rough).
- Accepted forms (any ONE): a visible-light photo of the taped board (roughly
  perpendicular, phone camera is fine) / a list of component names / pixel
  coordinates if you noted them (say which frame: raw 192×256 or cropped 103×157,
  and which orientation).
- Precision: ±2–3 px is fine — I snap to the patch center on the thermal image
  and sample a 3×3 median.
- Also note: tape type (ordinary black electrical tape? brand if known), single
  or double layer. (Affects assumed ε = 0.95 ± 0.02 and the contact-resistance caveat.)

### A2. SoC temperature logs **[required if they exist — do not reconstruct]**
- The raw log files from that day, any format (`vcgencmd measure_temp` loop,
  `/sys/class/thermal/thermal_zone0/temp` dumps, …), plus the sampling interval.
- Time sync: one anchor is enough — e.g. "load was started at camera time ≈18:08",
  or the Pi's timezone + whether its clock was NTP-synced. If nothing is known,
  I align by correlating the log's warm-up curve with the IR max-T curve (±5 s is
  achievable and sufficient).
- If no logs survive from that session: skip — the new session (Part B) covers it.

### A3. Workload definition **[required]**
- Exact commands used for **full load** and **half load** (tool + thread count,
  e.g. `stress-ng --cpu 4` vs `--cpu 2`), and what "half" meant (half the cores?
  duty cycling?).
- Whether the GPU was loaded; OS (Raspberry Pi OS version, 32/64-bit);
  any case/heatsink/fan on the board (from the images it looks bare — confirm);
  power supply used (official 5V/3A? phone charger?).
- Exact start/stop times are **[optional]** — folder timestamps already bracket them.

### A4. Environment & mounting **[required, 5 lines of text]**
- Board orientation: flat on desk / standing vertically? On standoffs? What
  surface/material underneath?
- Air: still room air, or AC/fan/open window nearby?
- Rough room temperature if you noted it (IR border median says ≈26.5 °C — confirm plausible).
- Camera-to-board distance ≈0.1 m per metadata — confirm.
- Was anything warm (person, laptop, lamp) in front of / reflected by the board
  during captures?

### A5. Camera identity **[required, 2 lines]**
- HIKMICRO model name (files are 192×256 — e.g. Pocket-series?), and if the manual
  is handy: stated accuracy (typically ±2 °C or ±2 %) and NETD.
- Export software used for the CSVs (name + version if visible).

### A6. Competition logistics **[required, 2 lines]**
- Which competition/track, the deadline, report format/page limit, and whether a
  live demo is possible at the defense. (This schedules M4–M10.)

---

## Part B — New measurement session (one evening; fixes case00 and strengthens everything)

**Context (2026-07-08):** session 1 was measured with large tape sheets already
covering the central processors (SoC + adjacent region) — the "taped"
configuration. The new session is a **two-configuration A/B design**:

- **Config A — bare board [strongly recommended, ~45 min]**: the BCM2711 heat
  spreader is shiny metal, so bare raw IR under-reads the *most important unit*.
  A gives (1) the problem-demonstration figure (bare SoC lid reads cold while
  the on-die diode says ~80 °C) and (2), subtracted from B at the matched
  steady state, a directly measured **per-pixel emissivity-error field**.
- **Config B — taped like session 1 [required, ~1.5 h]**: trusted ground-truth
  regions over the processors; this is the configuration all training/validation
  continues on. Mirror the session-1 tape coverage, plus the tiered extra spots
  from B1 (solder mask, cold corner; connector patch if it sticks).

**Order: A first, then apply tape without moving anything** — tape can be added
mid-session, but not removed cleanly. Sequence:
cold capture (bare) → full load ≥20 min (bare) → cooling ≥20 min (bare) →
apply tape carefully → idle → half load ≥20 min → cooling ≥20 min →
full load ≥20 min → cooling ≥20 min (all taped).
A taped cold capture is a bonus if the board can cool again afterward
(e.g. next morning, tripod untouched).

**Golden rules:** (1) once the tripod is set, do not touch it for the entire
session, including the cold capture and tape application; (2) **fix the board
to the surface first** (corner tape/standoffs — it currently sits loose on the
foam mat), so applying patches cannot shift it; (3) same camera settings as
session 1 (ε = 0.93, distance 0.1 m); (4) plain-text log of times of every
action; (5) nobody stands in front of the board during captures (reflections);
(6) log the SoC diode in BOTH configs — it proves the A/B states match and
measures the tape's own thermal perturbation.

### B1. Taping plan — do this BEFORE the session **[tiered]**
Patches ≥ 4×4 mm, flat and well-adhered (air gaps ruin contact), same roll for all.
Note the roles: tape spots are one-session *validation instruments* (held out of
training, removed afterward) — they are NOT deployment sensors, and their count
does not affect how light the per-board fine-tune is. More spots = more
leave-one-out validation points, nothing else.

**Core [required, 5]** — spans hot/medium/cold and the units that matter:
1. SoC lid (~5×5 mm, one patch only — tape insulates slightly),
2. RAM package,
3. PMIC / power-circuitry area (left edge),
4. bare solder-mask area mid-board,
5. far corner of the PCB (coldest region — validates the field, not because we
   care about it operationally).

**Demo patch [recommended, 1, only if it sticks well]:**
6. one shiny connector shield (USB or Ethernet). Not because port temperature
   matters — it is where IR lies most (~27 °C shown on a ~55 °C board), so this
   single patch enables the "raw IR vs tape vs reconstruction" money figure and
   tests the model exactly where IR supervision is masked out. Skip if adhesion
   on curved metal is poor.

**Statistics fillers [optional, up to 3]:** any additional spread-out spots
(second bare area, USB controller VL805, GPIO-side edge) purely to fatten
leave-one-out statistics.

Then photograph the taped board with a ruler in frame (positions + scale in one shot).

### B0. Visible-light reference **[required, 2 minutes]**
HIKMICRO Analyzer shows a 可见光 (visible-light) image per capture — export it
for at least one frame per configuration, or take a perpendicular phone photo of
the board in each configuration from the tripod position. This is what the
tape-region masks (`configs/source_mask_board.json` / trust masks) get annotated
from. Also drop the session-1 visible photo (the one showing the tape layout)
into `docs/session1_records/`.

### B2. Cold isothermal capture **[required — the single most valuable new data]**
- Board fully off and cold: powered off ≥3 h, ideally overnight, in the measurement room.
- Room stable (no AC blowing at the scene), no warm objects in view.
- Capture 10–20 frames over ~2 min.
- **[recommended]** Lay a small piece of crumpled aluminum foil flat in the scene
  and note its apparent temperature — that is the reflected-temperature estimate.
- **[recommended]** Note actual room temperature with any thermometer.

### B3. Case sequence and durations (tripod untouched throughout)
| # | Case | Duration | Note |
|---|---|---|---|
| 1 | cold isothermal (B2) | ~2 min | board off & cold |
| 2 | boot + idle | ≥10 min | from power-on |
| 3 | half load | **≥20 min** | last session's 12.5 min didn't fully plateau |
| 4 | cooling | **≥20 min** | τ ≈ 7–9 min, so 20 min ≈ 2.5τ (last time only 3.7 min) |
| 5 | full load | **≥20 min** | |
| 6 | cooling | **≥20 min** | |

If time-boxed, priority order: B2 > half-load 20′ > full-load 20′ > one 20′ cooling > idle.
Frame cadence: the camera's ~6 s auto cadence is fine everywhere.
Use the same folder structure as last time (`caseXX_名称/data`, `caseXX_名称/photo`),
any parent folder name — just tell me what it is.

### B4. SoC logging + time sync **[required during B3]**
Run on the Pi for the whole session:
```bash
while true; do
  echo "$(date +%s),$(vcgencmd measure_temp | grep -o '[0-9.]*')" >> ~/tlog.csv
  sleep 2
done
```
Sync anchor (any ONE): photograph the Pi showing `date` next to the camera's
clock display, or write down the camera time at the moment you launch each load
command. NTP-synced Pi + noted timezone also works.

### B5. Session log sheet **[required, plain text]**
One line per event with camera-clock time: tripod set, cold capture start/end,
power-on, load start/stop, anything unusual (person walked by, door opened).

---

## Part C — Optional extras (only if genuinely easy)

- **USB-C power meter** readings at idle / half / full (**cheap, high value**:
  gives source-amplitude ground truth, enables the power-in vs temperature-field story).
- **One contact reading** (thermocouple / DS18B20 / kitchen probe) at 1–2 tape
  spots at steady state — turns "trusted sensor" from an assumption into a verification.
- **One extra load case with a different source layout**, e.g. sustained USB
  traffic (heats the VL805 USB controller instead of the SoC) — 10 extra minutes,
  tests reconstruction when the heat source *moves*.
- **A second, different board** (any SBC) mini-session — the "general prior +
  cheap per-board calibration" transfer figure. Only if convenient.

## Part D — Explicitly NOT needed (do not spend time)

- Precise humidity, precise camera distance (±1 cm is fine), lens-transmission
  parameters, professional blackbody calibration, emissivity tables for every
  material, exact governor/DVFS traces, sub-second time sync.

## Where to put things

- Session-1 records (photos, logs, notes): anywhere under `docs/` or just tell me
  the paths — I'll organize them into `docs/session1_records/`.
- New session raw data: its own top-level folder (same per-case structure as before).
- Everything else (annotation into configs, alignment, ingestion) is my job.
