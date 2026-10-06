# pehuen-species-interactions

Analysis of species distribution and temporal interactions in Bosque Pehuén, from
the camera-trap campaigns published by `camera-traps` (otoño 2025, primavera 2025,
otoño 2026). Written in R using `camtrapR` and `overlap`.

This project is a **consumer** of the camera-trap canonical table. It reads the
published contract and tables, verifies them, and derives nothing the producer has
already decided. What that means in practice is in "The handshake" below.

---

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate pehuen-analysis
Rscript setup_packages.R      # camtrapR, overlap, activity, nanoparquet (CRAN only)
```

On the Windows box `library(sf)` fails under `conda activate` in Git Bash; run
everything through `conda run -n pehuen-analysis Rscript ...` instead.

### 2. Paths

There are none to edit. Every path is derived from this project's location: the
producer is expected at `../../camera-traps` and the platform at
`../../plataforma-territorial`, which is the FMA monorepo layout. `.here` in the
project root anchors `here()` to this directory and must not be deleted. For a
checkout laid out differently set `FMA_MONOREPO` to the repository root.

### 3. Line endings, on a checkout older than 2026-10-05

`contract_load()` recomputes the SHA-256 of the producer's `estaciones.geojson` from the
bytes on disk, so the checkout must hold the bytes that were hashed. `camera-traps/.gitattributes`
pins those files to LF — but it governs *checkout*, so files already in the working tree from
before it existed keep their CRLF and `git status` stays clean, because git normalizes on
comparison. The symptom is a refusal with exit 2 on a repository that looks pristine. Measured on
Windows 2026-10-06: the local file hashed `46f140dd…` against a published `10c5680f…`. One-time
fix, from the monorepo root:

```bash
rm -f camera-traps/data/campaigns/estaciones.geojson camera-traps/data/campaigns/*/deployments.csv
git checkout -- camera-traps/data/campaigns/
```

---

## The handshake

Every script starts by verifying the camera-trap contract
(`camera-traps/data/CANONICAL_STATE.json`), following `MANUAL-SALUD-DATOS.md`
Fase 10 in the producer's docs. `R/00_contract.R` owns it.

- **`01_load_data.R`** calls `contract_load()` before opening any file. Absent
  contract, unparseable contract, a schema version other than the one this project
  reads (`CONTRACT_SCHEMA_VERSION`, currently 5), a requested campaign the
  contract does not describe, or a station registry that is not the one the contract
  hashes, all **refuse**: a `REFUSED (...)` message naming the mismatch and **exit
  status 2**. An R error exits 1 and reads as a crash; a refusal is a verdict and must
  look like one.
- After a successful load, `01` writes `data/contract_stamp.json`: the registry hash
  and the declared block of every campaign it read, verbatim.
- **`02`–`06`** call `contract_assert_current()` first. It compares the stamp against
  the contract as published *now* and refuses if the registry hash or any field of any
  campaign moved, naming the field (`otono_2025.n_animal_rows: data/ built from 707,
  published now 712`). A campaign re-ingested or a registry re-published upstream
  therefore cannot keep feeding stale numbers or coordinates into a figure; the fix is
  always "re-run `01`".
- The gate has its own tests: `Rscript tests/test_contract.R` (base R, 33
  assertions, including subprocess checks that a refusal exits 2).

What the scripts deliberately do **not** do (manual 10F.3): parse station labels,
repair or shift timestamps, translate species names, decide whether a frame holds an
animal, decide what counts as an independent event, or compute camera-days. All of
those arrive decided, in `observations.parquet` or `deployments.csv`.

### Inputs the contract does not cover

The contract describes `observations.parquet` column by column, hashes
`deployments.csv` and, since schema 5 (2026-10-05), hashes the station registry. One
input is still outside it:

| input | published by | existence | content |
|---|---|---|---|
| `campaigns/estaciones.geojson` | camera-traps | refused if absent | **verified**: SHA-256 against `stations_sha256` |
| `plataforma-territorial/data/boundary.geojson` | the platform | refused if absent | **not verified here** |

**The registry** is verified byte-for-byte by `contract_load()`, before `01` opens
anything. It is looked up *next to the contract* (`registry_path()`), so a contract
pointed elsewhere with `FMA_CANONICAL_STATE` brings its own registry and the two cannot
describe different states. A mismatch, or a contract that publishes no hash, refuses
with exit 2 and names `python setup/build_station_registry.py --check` and
`python -m camtrap.canonical_state --publish`. Every coordinate and `altitude_m` this
project uses is therefore the one camera-traps last published, which is what the
altitude covariate and solar-time anchoring need.

**The boundary** belongs to a producer with no contract. A missing file refuses with
exit 2 rather than dying inside `st_read` (an R error exits 1, which reads as a
crash); its content is unverified.

What *is* enforced from here: a station that appears in the canonical table but not in
the registry **refuses**. It used to warn and null the station, which meant records
could leave the analysis with a success exit code.

---

## Running the analysis

**`01_load_data.R` is a required first step, on every machine and after every clone.**
`data/` is not tracked: the `.rds` files and `contract_stamp.json` are gitignored, so a fresh
checkout has no data at all until you build it. That is deliberate. An `.rds` in git is an
opaque second copy of the canonical table, one commit away from disagreeing with it, and
nothing in the repository would say which upstream state it came from. Rebuilding takes
seconds and is always current by construction.

Run in order. `01` writes `data/`; the rest read it and refuse if it is missing or stale.

```bash
Rscript R/01_load_data.R              # REQUIRED FIRST. contract, parquet, deployments, GeoJSON -> data/
Rscript R/02_detection_summary.R      # episodes, rate per 100 camera-days, naive occupancy — by season
Rscript R/03_activity_patterns.R      # 24 h kernel density activity curves (pooled), both frames
Rscript R/04_temporal_overlap.R       # Δ1/Δ4 pairwise overlap + CI + Monterroso category, both frames
Rscript R/05_spatial_distribution.R   # presence maps + episode bubble maps, natives by season
Rscript R/06_seasonal_detection_maps.R  # bubble maps per season PERIOD, chronological
```

Tests:

```bash
Rscript tests/test_contract.R    # 33 assertions — the consumer gate
Rscript tests/test_seasons.R     # 41 assertions — boundaries, the year straddle,
                                 # and effort conservation against the real field record
Rscript tests/test_timeofday.R   # 35 assertions — anchors solar_rad() to a direct
                                 # activity::transtime() call, proves the reference
                                 # anchor is pinned, and bounds the UTC-offset assumption
Rscript tests/test_overlap.R     # 30 assertions — anchors estimate_overlap() to a direct
                                 # overlap::overlapEst() call, guards the CI choice, and
                                 # checks camtrapR derives the same radians this project does
Rscript tests/test_detection_history.R  # 33 assertions — 1/0/NA cell semantics, the half-open
                                 # window, effort conserved against 02's rate denominator,
                                 # and every episode is a 1 (all species, all seasons)
```

### Figures have stable paths (`R/00_figures.R`)

camtrapR stamps the run date into the filenames it chooses — `Presence_Puma_2026-09-15.png`
— and the name is not configurable. `figures/` is tracked, so every run used to add a
fresh set beside the previous one; 24 superseded PNGs were deleted by hand on 2026-09-08
and 24 more on 2026-09-15. `stabilize_dated_pngs()` renames them back to undated names
after each camtrapR block, so a figure keeps one path for its whole life and a re-run
shows as a change to it rather than a new file plus a stranded one. Where this project
chooses the filename (04's per-pair plots) there is simply no date in it.

The date is not lost: a figure's date is its commit's date, and in a working tree its
mtime. What the stamp actually recorded was which run produced it, and that question is
answered properly by the contract gate — a figure whose data moved refuses to regenerate
rather than sitting on disk under an older name.

Running `02`–`06` before `01` is not an error you have to remember to avoid. They refuse:

```
REFUSED (data/ is not current):
  - data/contract_stamp.json not found: data/ has never been built by
    R/01_load_data.R under a verified contract. Run it first.
```

**What is tracked:** `figures/` and `data/overlap_stats.csv`, the numeric overlap results.
Those are readable outputs you would put in a report, not intermediates. The line is whether a
person can read it, not whether it is derived.

### Citations live in the manuscript, not in the figures

Removed from figure text on 2026-10-06. The numbers, the estimator symbol and the
category labels all stay on the figures — only the attribution went, because it belongs
in the paper's Methods and reads better there than squeezed onto a plot margin.

| Figure | Text removed |
|---|---|
| `figures/overlap_pairs/*.png` (footnote) | `— Monterroso et al. 2014; estimator per Ridout & Linkie 2009` |
| `figures/04_overlap_summary.png` (subtitle) | `(Ridout & Linkie 2009)` and `Categories from Monterroso et al. (2014).` |
| `figures/04_overlap_frames.png` (caption) | `categoría de Monterroso` → `categoría de solapamiento` |

So two references must be cited in the write-up wherever these figures appear, and they
are no longer visible on the figures themselves:

- **Ridout & Linkie (2009)** — the Δ1/Δ4 estimator and the rule that picks between them
  (Δ4 when `min(n_A, n_B) ≥ 50`, Δ1 otherwise).
- **Monterroso et al. (2014)** — the Low / Moderate / High overlap bands at 0.50 and 0.75,
  and the rule that a pair earns a single label only when its whole CI sits in one band.

Neither is lost from the repository: both are stated in full in the header of
`R/04_temporal_overlap.R` and triaged in `docs/methods-menu-interactions.md`. The
`category` column of `data/overlap_stats.csv` is unchanged and still carries the
classification itself.

This was done together with a clipping fix (below), and it is most of that fix: the
per-pair footnote was being cut at both ends *because* of the citation clause, losing the
leading `O` of "Overlap:" and the closing parenthesis — on the one line that carries the
published Δ and its CI.

### No figure may clip its own content

All 38 figures are checked by measuring ink on the outermost pixel row and column: text
cut off at the device boundary leaves marks there. Three real defects were found and
fixed on 2026-10-06, and the cause was different in each case:

- **The per-pair footnote** overflowed an 8-inch canvas. It hit 6 of 10 pairs — exactly
  those with the long category labels (`Moderate–High`, `Low–Moderate`) — so it was
  length-driven, and dropping the citation clause was enough.
- **`theme_void()` sets `plot.margin` to zero on all four sides**, so titles and captions
  on the map figures were drawn hard against the device edge with their descenders cut.
  Raising the canvas height does *not* help — the caption sits at the bottom edge whatever
  the height is. The margin is now set explicitly in both `map_theme` definitions
  (`05`, `06`) and on `presence_by_species`, which has its own inline theme.
- **`04_overlap_summary.png`** uses `expand = c(0, 0)`, which puts the panel edge exactly
  at 1.00, so the centred `1.00` tick label hung half its width into a 5.5pt margin a few
  pixels too narrow. Right margin widened to 16pt.

None of this moved a number: `data/overlap_stats.csv` stayed byte-identical through the
whole pass.

### Seasons, and why campaign is not one (`R/00_seasons.R`)

A campaign is the interval between two field visits — five to eight months here — and
it is named for the season the cards were **retrieved** in, not the season they
recorded:

| campaign | field window | seasons inside it |
|---|---|---|
| `otono_2025` | 2024-10-09 → 2025-06-11 | primavera, verano, otoño |
| `primavera_2025` | 2025-05-14 → 2026-01-14 | otoño, **invierno**, primavera, verano |
| `otono_2026` | 2025-11-13 → 2026-05-15 | verano, otoño |

The windows are contiguous, so the array is **one continuous record from 2024-10-09 to
2026-05-15** cut into three retrieval intervals. Until 2026-09-15 every figure faceted
by campaign, which meant a panel labelled "Otoño 2026" showed summer and autumn pooled.
Two things were being read as ecology and were not: liebre's apparent collapse to 2
episodes in otoño 2026 (that window holds no winter and no spring, and 109 of liebre's
129 episodes are winter or spring), and a "missing Invierno" that does not exist —
winter is the **best-sampled season in the record**, 96 episodes, inside the
`primavera_2025` window.

Stratifying on the timestamp gives **seven season periods** with two otoños, two
primaveras and two veranos, so a pattern can be asked to repeat.

- `season_start(x)` → the `Date` a period begins on. **This is the join key**:
  unambiguous across the new year, sorts chronologically, and cannot be confused with a
  campaign slug (`primavera_2025` names a campaign that recorded winter).
- `season_of(x)` → `Primavera | Verano | Otoño | Invierno`, for pooling across years.
- `season_label(x)` → an ordered factor for figures. `Verano 2025-26` carries both
  years because the period straddles January. Display only; never join on it.
- `season_effort(deployments)` → one row per (campaign, station, period) with the
  station-days of that deployment falling inside it. It carries `media_status` and
  `valid_effort` through untouched; **which** statuses belong in which denominator
  stays script 02's decision. The split conserves days exactly — 13,598 in, 13,598
  out — asserted in `tests/test_seasons.R` rather than assumed.

Campaign survives as provenance. It is still what the producer publishes, still what
decides admissibility, still carried in every intermediate table. It is no longer an
axis anything is plotted against.

The boundary rule is one table, `SEASON_BOUNDARIES`. Moving to solstice/equinox dates —
under discussion, and arguably right at 38°S where photoperiod is what the seasons
proxy for, and what `activity::transtime()` anchors on — changes that table and no
caller. Astronomical dates drift between years, so that version needs a per-year table
rather than (month, day).

### Two frames of reference (`R/00_timeofday.R`)

A detection's position on the 24-hour circle is computed in two frames, and the
difference between them is a result rather than a correction.

| | what it is | who uses it |
|---|---|---|
| `time_rad` | the camera's own wall clock | every figure published before 2026-10-05; all camtrapR panels |
| `time_solar_rad` | the same detection relative to that day's sunrise and sunset | 03's and 04's solar figures, ggplot only |

Measured at the site, over 2025:

| | sunrise | sunset | day length |
|---|---|---|---|
| 2025-06-20 | 08:07 | 17:30 | 9.4 h |
| 2025-12-20 | 05:17 | 20:14 | 15.0 h |
| **annual swing** | **2.9 h** | **2.8 h** | **5.6 h** |

A species holding a fixed schedule relative to sunrise is therefore smeared across
nearly three hours of pooled clock time. Rowcliffe et al. (2014) name the
consequence — flattened peaks and **overestimated activity level** — and exempt only
the tropics and short studies. This record is 19 months at 39.4°S.

This is piece 3 of a decision taken upstream, not a new idea. `camera-traps`'
`DATA-HEALTH-MANUAL.md` §5.3 forbids adjusting camera clocks for civil time
*because* "the defensible frame for an activity analysis is solar"; `V2-REVIEW.md`
lists "the sun-anchored sensitivity run in pehuén (piece 3)" as deliberately out of
that review's scope.

The transformation is `activity::transtime()` and the sun times are
`activity::get_suntimes()` — nothing here reimplements solar geometry, and
`tests/test_timeofday.R` anchors our value to a direct call of both.

**What it moved.** Mean |Δ| over the ten pairs is 0.055, max 0.179 (Puma × Jabalí),
and **four of ten Monterroso categories changed** — all four by widening, not by
reversing: three pairs land on a compound label spanning more bands, which is the
honest consequence of 12–18-episode samples. `Guiña × Zorro culpeo` does not move at
all (Δ = 0.001). The headline reading survives: natives keep **Low** overlap with
perro in both frames.

**Three things this frame does not fix, and they belong in Methods.**

- `SOLAR_OFFSET_HOURS <- -4` is a declared assumption. The upstream rule is
  *horario de invierno, no DST correction ever*, which makes −4 right; but §5.3 also
  records that these clocks were adjusted once, and the per-deployment offset
  (piece 2) is not published. Measured bound: a −3 offset instead of −4 moves any
  detection **at most 11.4 minutes**, asserted in the test suite.
- `get_suntimes()` is documented as **approximate** and computes geometric sunrise at
  sea level. It does not know the Andean horizon, so at a station with a ridge to the
  east, real first light is later. The error is per-station and unmodelled.
- **One site coordinate, not 27** — measured, not assumed. Sunrise differs by
  **15.7 seconds** between the array's two extreme corners against a 2.9-hour annual
  swing, so per-station coordinates would be false precision at about 1:660. The
  module therefore reads no station file and does not depend on `estaciones.geojson`
  or on the contract that describes it.

**The camtrapR panels stay on the clock and cannot be moved.** `activityDensity()`
and `activityOverlap()` take `recordDateTimeCol` and derive the hour internally;
there is no radians entry point. The solar figures are ggplot-only, which is safe
because the two layers were proved to agree to 1e-16 on 2026-09-15.
`tests/test_overlap.R` now asserts both halves: camtrapR agrees with `time_rad`, and
**disagrees** with `time_solar_rad`, so nobody can "fix" the limitation by writing
solar times into `DateTimeOriginal` without the suite failing.

### Units and admissibility (`R/00_admissibility.R`)

Two questions, two rules, both explicit at the call site:

- `admissible(records, "place")` keeps every identified record. A frame with a
  broken clock still proves the animal was there. Used by `presence()`.
- `admissible(records, "time")` keeps records with a trustworthy timestamp
  (`valid_date & valid_time_of_day`). Required for anything using date, hour or
  season. Used by `episodes()`.

**Counts are episodes, never images.** A camera fires 2–3 frames per trigger, so an
image count measures burst length. `episodes()` reads the producer's
`episode_30min` column (one id per independent detection, 30-minute rule measured
from the last retained detection, never across a clock-segment boundary) and derives
nothing. `record_table.rds`, the camtrapR input, is one row per episode.

### Effort (`data/deployments.rds`, from the producer's `deployments.csv`)

One row per (campaign, station) from the field record, with `media_status`:

| status | means | in a stills-based rate denominator, and surveyed in a detection history | in a naive-occupancy denominator |
|---|---|:---:|:---:|
| `in_canonical` | stills are in the table | yes, if `valid_effort` | yes |
| `video_only_offline` | camera was sampling; media is video outside the pipeline | no | yes |
| `card_failure` | recorded nothing | no | no |
| `unexplained`, `no_field_dates` | no usable effort; surfaced, never absorbed | no | no |

The table is one function, `effort_admissible(deployments, "detections" | "sampling")` in
`R/00_admissibility.R` — the effort-side twin of `admissible(records, ...)`. Script 02 and
`R/00_detection_history.R` both call it; neither restates it.

### Detection histories (`R/00_detection_history.R`)

`detection_history(records, deployments, species, season)` returns the station × occasion
grid an occupancy model needs: `y` (1 detected / 0 surveyed-not-detected / **NA not
surveyed**), `effort` (surveyed days per cell) and `occasions` (start, end). It feeds
`unmarked::unmarkedFrameOccu(y = h$y, obsCovs = list(effort = h$effort))` directly.

- **`season` is required.** A single-season model assumes closure — each station used or
  unused for the whole run of occasions — and the 19-month record does not satisfy it. A
  call without a season refuses. Occasions start on the season's first day; B2 stacks one
  grid per season.
- **`OCCASION_DAYS` is 14, measured.** Occasion length barely moves ψ, because the chance of
  detecting a species at least once in a season is nearly fixed (culpeo, Invierno 2025: 81 %
  at 7 d, 77 % at 14 d). 14 d is where per-occasion p for culpeo and liebre reaches
  ~0.2–0.35; 7 d is the sensitivity run. The measurement, and a plain-language account of ψ,
  p, occasions and closure with worked examples, is in `docs/methods-menu-interactions.md`
  §B0.1.
- **What it supports.** Culpeo in all six usable seasons, liebre in about four, perro only
  with a low-p caveat. Puma, guiña and jabalí are detected at 0–4 stations per season and
  are not estimable by occupancy.

### Overlap (script 04)

Estimator per pair from the smaller sample (Ridout & Linkie 2009): Δ4 when
`min(n_A, n_B) ≥ 50`, Δ1 otherwise; 1000 bootstrap resamples for the 95% CI,
reported as the **`basic0`** interval (`CI_TYPE`). Classification per Monterroso
et al. (2014) applied to the **whole CI**: Low (< 0.50), Moderate (0.50–0.75), High
(≥ 0.75); a CI straddling a threshold gets a compound label. Estimator, estimate, CI
and category are in `data/overlap_stats.csv` and on every figure.

#### Which confidence interval, and why

`bootCI()` returns five, all built from two quantities — `bias = mean(bt) - t0` and
`merr = sd(bt) * 1.96`:

| row | formula | bias-corrected? |
|---|---|---|
| `norm` | `t0 - bias ± merr` | yes |
| `norm0` | `t0 ± merr` | **no** — symmetric on the estimate |
| `perc` | `quantile(bt)` | **no** — raw bootstrap quantiles |
| `basic` | `2*t0 - perc[2:1]` | yes, by reflection |
| `basic0` | `perc - bias` | the percentile interval, shifted |

The `0` suffix means bias correction **removed** — it marks the intervals that belong
with the *uncorrected* point estimate `t0`, not intervals that skip a correction they
ought to make. `?bootCI` is explicit: *"the bootstrap estimates are biased, so 'perc'
should be corrected… If you use the uncorrected estimator, t0, you should use 'basic0'
or 'norm0'."* We report `t0` — `overlapEst()`'s value, the one camtrapR prints inside
each plot — so the interval must be `basic0` or `norm0`. `norm0` returned
`[0.700, 1.0028]` for Guiña × Zorro culpeo, above the maximum the coefficient can
take; `basic0` stays inside [0, 1] on all ten pairs. Hence **`CI_TYPE <- "basic0"`**.

**What the bias is.** `mean(bootstrap replicates) − point estimate`: how far the
resampling distribution sits from the estimate it was generated around. For a
coefficient of overlapping it is a *shrinkage toward the middle* — low estimates pushed
up, high ones pulled down — because Δ is bounded in [0, 1] (noise can only move a
near-1 overlap down and a near-0 overlap up) and because Δ integrates the **minimum**
of two curves, a concave operation, so noise in either curve lowers the expected
minimum. Measured here: Guiña × Perro +0.048, Zorro culpeo × Perro +0.034,
Puma × Liebre −0.049, Guiña × Zorro culpeo −0.094.

**It is not simply "our n is small", and the obvious reading is wrong.** Across the ten
pairs |bias| correlates −0.48 with the smaller sample, and the one pair with a
non-tiny smaller sample (Zorro culpeo × Liebre, n = 129) has a bias of −0.0007 — which
invites the conclusion that more episodes would dissolve the problem. They would not.
Holding the distribution shape fixed and varying n on synthetic data:

| n | bias | bootstrap SD |
|---:|---:|---:|
| 20 | +0.032 | 0.114 |
| 50 | +0.036 | 0.071 |
| 120 | +0.036 | 0.044 |
| 400 | +0.026 | 0.025 |

The **SD** collapses with n, as it must. The **bias barely moves**. What it tracks is
the shape of the distributions relative to the smoothing: a kernel estimate of two
narrow, well-separated clusters overstates their overlap at every n, because the
bandwidth rule keeps smoothing them together. Our n = 129 pair has a near-zero bias
because it is a broad, high-overlap pair where smoothing distorts little — not because
129 is large enough for the problem to disappear.

`basic0` is not bounded by construction; it is a shift of the percentile interval, so a
large enough bias near the boundary could push it past 1. It does not on this data, and
`tests/test_overlap.R` asserts every published CI lies in [0, 1] as a data check that
will fail loudly if a future campaign changes that.

**The estimate and its interval are computed from detection TIMES.** From 2026-07-28
to 2026-09-15 `estimate_overlap()` passed `densityFit()` output — density values —
into `overlapEst(A, B)` and `bootstrap(A, B)`, whose arguments are times in radians.
Both functions accepted them and fitted fresh kernels to the density values, so
every published estimate, CI and category was wrong: mean absolute error 0.21 over
the ten pairs, maximum 0.54, and **all ten categories changed** when it was fixed.
`tests/test_overlap.R` now asserts our estimate equals a direct
`overlap::overlapEst()` call, which is the check that was missing — the number
camtrapR prints inside each per-pair plot was correct all along, and nothing
compared the two.

### Seasonal maps (script 06)

Only species with at least `MIN_EPISODES` (30) independent episodes get a figure.
On the current data: Zorro culpeo (161), Liebre (129), Perro (46). Puma (12), Guiña
(14) and Jabalí (18) are skipped, and the script says so.

One panel per **season period**, in chronological order, so a spatial pattern can be
asked whether it repeats across the two otoños — the one thing 19 months of
single-site record can least afford to pool away. The panel set comes from the field
record, not from the detections: a period the array sampled and the species was never
seen in gets a panel of × marks, because an absence is a result. Each strip carries
its own effort, since the periods differ by a factor of eight (292 camera-days at 9
stations in Primavera 2024 against 2,249 at 27 in Verano 2025-26). A bubble is a
count, not a rate.

`05_spatial_native_by_season.png` keeps the pooled four-season view for cross-species
comparison. Two questions, two figures, one season rule.

---

## Adding a future campaign

1. In `camera-traps`: `python timestamps.py --campaign <slug>`, then
   `python -m camtrap.canonical_state --publish`.
2. Add the slug to `CAMPAIGNS` in `R/01_load_data.R` and a display name to
   `CAMPAIGN_LABELS` in `R/00_contract.R`.
3. Re-run all scripts. Nothing else changes: season panels come from the field record
   via `season_effort()`, and a new campaign simply extends the set of periods. A
   campaign that adds no new calendar months adds no new panels — which is the point.

---

## Focal species

| Spanish       | Latin                   | Guild    |
|:--------------|:------------------------|:---------|
| Puma          | Puma concolor           | Native   |
| Guiña         | Leopardus guigna        | Native   |
| Zorro culpeo  | Lycalopex culpaeus      | Native   |
| Jabalí        | Sus scrofa              | Invasive |
| Liebre        | Lepus europaeus         | Invasive |
| Perro         | Canis lupus familiaris  | Invasive |

Stations are `CT01`..`CT27`, one spelling everywhere, from
`camera-traps/data/campaigns/estaciones.geojson`.

---

## Planned / candidate analyses

`docs/methods-menu-interactions.md` is a critical, sourced menu of alternative
spatial and temporal interaction methods, each triaged against this array's sample
sizes. Its "Open items" list is the analysis backlog.

---

## Project status

- **Last Updated:** 2026-10-06
- **What Changed (2026-10-06, 3 of 3):** **The detection history exists, and its occasion length
  was measured before it was chosen.** New `R/00_detection_history.R` builds the station × occasion
  grid occupancy needs — 1 / 0 / **NA for unsurveyed** (MacKenzie et al. 2003), surveyed days per
  cell as `effort` — for **one season at a time; a call without a season refuses**, because a
  single-season model assumes closure and the 19-month record does not hold it. Occasion length
  was set by fitting a null ψ(.)p(.) per species × season at 1–14 days: ψ barely moves (culpeo
  Invierno 2025: 0.53–0.56 from 3 to 14 d) because the chance of detecting a species at least
  once per season is nearly fixed (81 % at 7 d, 77 % at 14 d), so **`OCCASION_DAYS = 14`**, 7 d
  as sensitivity. The same fits settle scope: culpeo estimable in all six usable seasons,
  liebre in ~4, perro only with a low-p caveat, **puma, guiña and jabalí not estimable by
  occupancy** (0–4 detecting stations per season). Liebre's low ψ despite many photos is
  concentration — 71 % of its episodes at CT09, CT20, CT19 — which a constant-p model reads
  as low ψ and probably underestimates. A plain-language account of ψ, p, occasions,
  cumulative detection and closure, with these cases worked through and references, is the
  new `docs/methods-menu-interactions.md` §B0.1. The surveyed-day rule moved out of 02 into
  `effort_admissible()` in `R/00_admissibility.R`, so 02 and the new module share one rule;
  02's printed output and all three PNGs were byte-identical before and after. 33 new
  assertions; all five suites pass. Two stale doc lines corrected (this block's 1 of 3, and
  the methods menu's `stations_sha256` blocker, which schema 5 closed on 2026-10-05).
- **What Changed (2026-10-06, 2 of 3):** **Citations came off the figures, and no figure
  clips its own content any more.** Attribution moved to the manuscript — see "Citations live
  in the manuscript, not in the figures" above for the exact strings removed and the two
  references (Ridout & Linkie 2009; Monterroso et al. 2014) that must now be cited in the
  write-up. That was also most of the clipping fix: the per-pair footnote was being cut at
  **both** ends by the citation clause, losing the leading `O` of "Overlap:" and the closing
  parenthesis on the one line carrying the published Δ and its CI, in 6 of 10 pairs. Two other
  causes, both different: `theme_void()` zeroes `plot.margin` on all four sides, so the map
  captions were drawn hard against the device edge (raising canvas height does not help, and
  was tried and reverted); and `04_overlap_summary.png` uses `expand = c(0, 0)`, so the centred
  `1.00` tick label hung past a 5.5pt margin. All **38 of 38** figures now carry no ink on their
  outermost pixel row or column, measured. `data/overlap_stats.csv` stayed byte-identical
  throughout and all four suites pass.
- **What Changed (2026-10-06, 1 of 3):** **The Windows box reached schema 5, and the chain reproduces
  there exactly.** `data/overlap_stats.csv` came back **byte-identical to HEAD** after a full
  rebuild and re-run on Windows, so the solar-frame work done on Linux on 2026-10-05 is not
  platform-dependent. All four suites pass and all six scripts exit 0. Two failures were fixed,
  neither in the analysis: `tests/test_contract.R` compared raw path strings, which cannot hold on
  Windows because `tempfile()` returns backslashes while `dirname()` normalizes to forward slashes
  (`registry_path()` was never wrong); and `tests/test_overlap.R` now fails closed when
  `record_table.rds` lacks `time_rad`/`time_solar_rad` — a cache from 2026-09-15 made
  `rt_rad$time_solar_rad` NULL, so `max()` over `numeric(0)` returned `-Inf` and a stale cache read
  as a broken frame. The gate itself had been refusing on Windows for a line-ending reason, now
  documented under Setup §3. **Figures are committed from Windows renders** (this is the work
  machine): both platforms render at identical pixel dimensions with correct accents and en-dashes,
  and the 2.5× byte difference is font rasterization, not resolution — `04_overlap_summary.png`
  rendered here is byte-identical to the 2026-09-15 commit, so there are two stable platform
  renders rather than drift. One real defect found and **not yet fixed** (*fixed later the same
  day — see 2 of 3*): that figure clips its own
  subtitle at the right edge in both renders, which is a layout bug in `R/04_temporal_overlap.R`;
  the other figures have not been checked for it.
- **What Changed (2026-10-05, 3 of 3):** **The analysis gained a second frame of reference.**
  New `R/00_timeofday.R` owns where a detection sits on the 24-hour circle, in both the
  camera's clock frame and a sun-anchored solar frame (`activity::transtime()`, double
  anchoring on sunrise and sunset — Nouvellet et al. 2012; Vazquez et al. 2019). At 39.4°S
  over 19 months sunrise moves 2.9 h, so pooled clock time flattens exactly the peaks this
  analysis is about; Rowcliffe et al. (2014) exempt only the tropics and short studies. This
  is piece 3 of the upstream clock decision (`DATA-HEALTH-MANUAL.md` §5.3), not a new idea.
  `03` and `04` report both frames and `data/overlap_stats.csv` gains a `frame` column — 20
  rows where it had 10. **The ten clock-frame rows are byte-identical to the committed ones**
  (item 2 of 3 below says the file was byte-identical after the schema bump; it still is, row
  for row, in the clock block — this change adds a second block beside it and moves nothing).
  The bootstrap is re-seeded per frame so a published CI cannot depend on whether the solar
  frame ran before or after it. Mean |Δ| 0.055, max 0.179 (Puma × Jabalí); four of ten
  Monterroso categories changed, all by widening onto compound labels, which is the
  12–18-episode samples showing through rather than a new biological claim. The headline
  survives: natives keep **Low** overlap with perro in both frames. Two new figures,
  `03_activity_frames.png` and `04_overlap_frames.png`. `tests/test_timeofday.R` (35
  assertions) anchors the value to a direct package call, proves the reference anchor is
  pinned — unpinned, `transtime()` moved the same detection **42 minutes** depending on how
  many rows the caller passed, and 04 calls it on species subsets — and bounds the UTC-offset
  assumption at **11.4 minutes**. `tests/test_overlap.R` (30, was 25) now asserts camtrapR
  agrees with `time_rad` AND disagrees with `time_solar_rad`, so the camtrapR limitation
  cannot be "fixed" by overwriting `DateTimeOriginal` without the suite failing. Three limits
  for Methods are in "Two frames of reference" above.
- **What Changed (2026-10-05, 2 of 3):** **The station registry is verified.** Contract
  schema 4 → 5. `contract_load()` checks the SHA-256 of `estaciones.geojson` against the
  published `stations_sha256` and refuses with exit 2, and the stamp carries the hash so
  02–06 refuse when the registry is re-published. Four refusals were proved on doctored
  scratch copies (hash altered, hash removed, one latitude moved 1e-5°, stale stamp). The
  full chain re-ran with **`overlap_stats.csv` byte-identical** to HEAD and all six `.rds`
  identical by content digest. 33 contract assertions (was 25). On this Linux machine the
  `pehuen-analysis` env lacked `camtrapR` and `nanoparquet`; both are now installed
  (`r-fs` and `r-httpuv` from conda-forge, since they need libuv).
- **What Changed (2026-10-05, 1 of 3):** Orientation pass over the spatial and temporal analyses
  actually built. No code changed. One stale word corrected: the overlap section above said
  the 95% CI is the **percentile** interval; `CI_TYPE` has been `basic0` since 2026-09-15 and
  the subsection below it already argued for `basic0`. The account of what is built, what is
  blocked and on what is in [[2026-10-05-pehuen-spatial-temporal-stocktake]].
- **What Changed (2026-09-15, 2 of 2):** **Every published overlap number was wrong.**
  `estimate_overlap()` passed `densityFit()` output into `overlapEst(A, B)` and
  `bootstrap(A, B)`, whose arguments are detection times in radians; both fitted fresh
  kernels to the density values instead. Found because the per-pair PNGs carry two
  numbers — camtrapR's, computed correctly inside the panel, and ours in the footnote —
  and they disagreed (`Dhat4=0.81` against `Δ4 = 0.879`). Mean absolute error 0.21 over
  the ten pairs, maximum 0.54 (Guiña × Perro, 0.757 published against 0.221 true), and
  **all ten Monterroso categories changed.** The correction reverses the headline
  reading: the natives show **low** temporal overlap with dog and boar — Guiña × Perro
  and Zorro culpeo × Perro are now Low, Puma × Jabalí and Zorro culpeo × Jabalí
  Low–Moderate — which is temporal segregation from invasives, the pattern the project
  exists to test. Native × native and native × liebre stay high. Read it against the n:
  the biggest movers are the 12–18-episode pairs, and three pairs now land on
  "Low–High", a CI too wide to say anything. `tests/test_overlap.R` (24 assertions)
  anchors our estimate to a direct `overlap::overlapEst()` call so this cannot recur,
  and the CI moved from `norm0` (which returned an upper bound of 1.0028, above the
  coefficient's maximum) to `basic0`, the interval `?bootCI` prescribes when the
  reported estimate is the uncorrected `t0` — see "Which confidence interval, and why"
  above for the bias this corrects and why it is not merely a small-n artefact.
- **What Changed (2026-09-15, 1 of 2):** **The stratifier was wrong.** Every figure faceted by campaign, and
  a campaign is the five- to eight-month interval between field visits, named for the
  season the cards were retrieved in. The three windows are contiguous, so the array is
  one continuous record from 2024-10-09 to 2026-05-15. New `R/00_seasons.R` owns the
  boundary rule, the period key (`season_start`, a Date) and the split of deployment
  effort across periods; `tests/test_seasons.R` (42 assertions) proves the split
  conserves station-days exactly — 13,598 in, 13,598 out — so switching the axis moved
  no effort. Scripts 02, 05 and 06 re-stratified; 06's own `assign_season()` deleted
  rather than copied. Two claims that were being read as ecology are retired: liebre's
  "collapse" in otoño 2026 is that window holding no winter and no spring, and the
  "missing Invierno" of the methods menu does not exist — winter is the best-sampled
  season in the record (96 episodes), inside the `primavera_2025` window. 03 and 04 are
  untouched: they pool, so the campaign label never reached them.
  Seasonal naive occupancy costs something and the figure says so: a season cannot be
  read off a broken clock, so Fig C is time-admissible and drops five station-species
  presences the campaign version kept (CT08 guiña, CT10 jabalí, CT13 liebre, CT18
  perro, CT18 puma). The pooled presence maps in 05 still carry them.
- **Prior (2026-09-08):** The consumer handshake is implemented (`R/00_contract.R`,
  `tests/test_contract.R`): every script verifies the published contract, refuses
  with exit 2, and downstream scripts refuse if `data/` is behind the contract.
  Loader moved to schema 4 and the producer's `estaciones.geojson`; the R copy of the
  30-minute episode rule is retired onto `episode_30min` (zero numbers moved: 380
  episodes both ways); effort now comes from the producer's `deployments.csv`;
  images-where-episodes-were-claimed fixed in 03 and 05; otoño 2026 now appears in
  every campaign facet. All figures re-rendered. **`data/` is no longer tracked** —
  the `.rds` files and the contract stamp are gitignored and `01_load_data.R` is a
  required first step, so the repository never carries an undated second copy of the
  canonical table.
  A second pass audited the ingest procedurally: the contract is true of the
  producer's files on disk, registry and contract agree on 27 stations, no figure is
  older than the data, and every deployment has an explanatory `media_status` (no
  `unexplained`, no `no_field_dates`). Two guard rails were added — the two GeoJSONs
  now refuse with exit 2 instead of crashing with exit 1, and a station missing from
  the registry refuses instead of warning and exiting 0.
- **Integration Status:** `In Progress [REMAINING: Rota power check, B1 occupancy, overlap
  prose rewrite]` — the seasonal foundation and the detection history are in and all six
  scripts run clean against schema 5. Target is a peer-reviewed paper, which sets the
  remaining sequence: `R/00_detection_history.R` is done (2026-10-06), so next is
  `R/07_power_cooccurrence.R` to settle whether the Rota multi-species co-occupancy
  model is estimable here — `docs/methods-menu-interactions.md` §B3 says no on a
  400-site citation, `References/FMA_camera_trap_methods_synthesis.md.pdf` ranks it
  first, and a simulation at this array's own n is worth more than either. The registry
  blocker is closed: station coordinates and `altitude_m` are verified since schema 5,
  so occupancy with an altitude covariate and solar-time anchoring
  (`activity::transtime`) are unblocked on the data side. One item is still upstream:
  `review_outcome` empty → `not_applicable`. It would not move this project's numbers
  (no animal row carries an empty value), and as the contract stands it would not move
  the stamp either (`V2-REVIEW.md` §0-septies).
- **Blockers/Notes:** **Sample size is the binding constraint and the season split makes
  it explicit.** Of 42 species × period cells, none reaches 100 episodes, eight are in
  20–99 (culpeo in five periods, liebre in three), and 30 are below 10. Pooled over the
  whole record only culpeo (161) and liebre (129) clear 100; perro is 46; jabalí, guiña
  and puma are 18, 14 and 12. Detections are also concentrated — CT03 and CT09 hold 37 %
  of all 380 episodes — and the largest shared-station count for any species pair is 9.
  That bounds the interaction question: Niedballa's ≥50-records bar is cleared by
  culpeo × liebre and culpeo × perro, and by nothing else.
  Effort is uneven across periods by a factor of eight (292 camera-days at 9 stations in
  Primavera 2024, the array ramping up, against 2,249 at 27 in Verano 2025-26). Panel
  strips carry their own denominator for that reason.
  Two references added 2026-09-10 bear on method and are not yet acted on: Smith (2025)
  defends independence filters against Peral et al. (2022) but conditions that defence
  on inspecting the raw retained detections, and Edwards et al. (2020) shows
  camera-derived activity curves depend on what the camera is pointed at. Both belong in
  a Methods section before any activity claim.
- **Prior blockers (2026-09-08), now partly void:** that entry reported "six of ten pairs
  changed Monterroso category, Guiña × Zorro culpeo crossing Moderate → High" after the
  CT03 recovery. **Both sides of that comparison were computed by the broken
  `estimate_overlap()`**, so the claim is not meaningful and is retained only as a record
  of what was believed. What was true then and stays true: `overlap_stats.csv` and
  `04_overlap_summary.png` committed on 2026-08-20 predated the CT03 recovery, their
  per-species n summing to 327 against the record table's 380. The written
  interpretation of the overlap results must be re-read against the corrected table.
  Detection-rate and occupancy denominators also changed definition (field-record
  effort instead of days-with-photos), so every script-02 figure moved by
  construction rather than by data.
