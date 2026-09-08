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

---

## The handshake

Every script starts by verifying the camera-trap contract
(`camera-traps/data/CANONICAL_STATE.json`), following `MANUAL-SALUD-DATOS.md`
Fase 10 in the producer's docs. `R/00_contract.R` owns it.

- **`01_load_data.R`** calls `contract_load()` before opening any file. Absent
  contract, unparseable contract, a schema version other than the one this project
  reads (`CONTRACT_SCHEMA_VERSION`, currently 4), or a requested campaign the
  contract does not describe, all **refuse**: a `REFUSED (...)` message naming the
  mismatch and **exit status 2**. An R error exits 1 and reads as a crash; a refusal
  is a verdict and must look like one.
- After a successful load, `01` writes `data/contract_stamp.json`: the declared
  block of every campaign it read, verbatim.
- **`02`–`06`** call `contract_assert_current()` first. It compares the stamp against
  the contract as published *now* and refuses if any field of any campaign moved,
  naming the field (`otono_2025.n_animal_rows: data/ built from 707, published now
  712`). A campaign re-ingested upstream therefore cannot keep feeding stale numbers
  into a figure; the fix is always "re-run `01`".
- The gate has its own tests: `Rscript tests/test_contract.R` (base R, 25
  assertions, including a subprocess check that a refusal exits 2).

What the scripts deliberately do **not** do (manual 10F.3): parse station labels,
repair or shift timestamps, translate species names, decide whether a frame holds an
animal, decide what counts as an independent event, or compute camera-days. All of
those arrive decided, in `observations.parquet` or `deployments.csv`.

### Inputs the contract does not cover

The contract describes `observations.parquet` column by column and hashes
`deployments.csv`. Two inputs are outside it, and it is worth knowing which:

| input | published by | existence | content |
|---|---|---|---|
| `campaigns/estaciones.geojson` | camera-traps | refused if absent | **not verified here** |
| `plataforma-territorial/data/boundary.geojson` | the platform | refused if absent | **not verified here** |

Both are read once, by `01`. A missing one refuses with exit 2 and the command that
regenerates it, rather than dying inside `st_read` — an R error exits 1, which reads
as a crash. Their *content* is another matter: nothing in `CANONICAL_STATE.json`
would let this project notice a moved coordinate or a changed `altitude_m`.
camera-traps does guard the registry against drift from `estaciones.csv`, which owns
station identity, but that check lives upstream and is not visible from here.
Closing it properly means adding a `stations_sha256` to the published state, which is
a schema bump on both sides; it is logged in the producer's `V2-REVIEW.md` §0-septies.

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
Rscript R/02_detection_summary.R      # episodes, rate per 100 camera-days, naive occupancy
Rscript R/03_activity_patterns.R      # 24 h kernel density activity curves
Rscript R/04_temporal_overlap.R       # Δ1/Δ4 pairwise overlap + CI + Monterroso category
Rscript R/05_spatial_distribution.R   # presence maps + episode bubble maps
Rscript R/06_seasonal_detection_maps.R
```

Running `02`–`06` before `01` is not an error you have to remember to avoid. They refuse:

```
REFUSED (data/ is not current):
  - data/contract_stamp.json not found: data/ has never been built by
    R/01_load_data.R under a verified contract. Run it first.
```

**What is tracked:** `figures/` and `data/overlap_stats.csv`, the numeric overlap results.
Those are readable outputs you would put in a report, not intermediates. The line is whether a
person can read it, not whether it is derived.

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

| status | means | in a stills-based rate denominator | in an occupancy denominator |
|---|---|:---:|:---:|
| `in_canonical` | stills are in the table | yes, if `valid_effort` | yes |
| `video_only_offline` | camera was sampling; media is video outside the pipeline | no | yes |
| `card_failure` | recorded nothing | no | no |
| `unexplained`, `no_field_dates` | no usable effort; surfaced, never absorbed | no | no |

Script 02 applies exactly that table.

### Overlap (script 04)

Estimator per pair from the smaller sample (Ridout & Linkie 2009): Δ4 when
`min(n_A, n_B) ≥ 50`, Δ1 otherwise; 1000 bootstrap resamples for the 95% CI.
Classification per Monterroso et al. (2014) applied to the **whole CI**: Low
(< 0.50), Moderate (0.50–0.75), High (≥ 0.75); a CI straddling a threshold gets a
compound label. Estimator, estimate, CI and category are in
`data/overlap_stats.csv` and on every figure.

### Seasonal maps (script 06)

Only species with at least `MIN_EPISODES` (30) independent episodes get a figure.
On the current data: Zorro culpeo, Liebre, Perro. Puma, Guiña and Jabalí have
9–18 episodes and are skipped, and the script says so.

---

## Adding a future campaign

1. In `camera-traps`: `python timestamps.py --campaign <slug>`, then
   `python -m camtrap.canonical_state --publish`.
2. Add the slug to `CAMPAIGNS` in `R/01_load_data.R` and a display name to
   `CAMPAIGN_LABELS` in `R/00_contract.R`.
3. Re-run all scripts. Facets and grids are built from the stamp, so nothing else
   changes.

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

- **Last Updated:** 2026-09-08
- **What Changed:** The consumer handshake is implemented (`R/00_contract.R`,
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
- **Integration Status:** `Ready` — consumer side of the camera-trap boundary is
  closed; all six scripts run clean against schema 4. One item is upstream:
  `estaciones.geojson` is not covered by the contract, so station coordinates and
  `altitude_m` are unverified from here (`V2-REVIEW.md` §0-septies).
- **Blockers/Notes:** `data/overlap_stats.csv` and `04_overlap_summary.png` committed
  on 2026-08-20 predated that day's CT03 recovery and were stale until this
  re-render — their per-species n sums to 327 against the committed record table's
  380. **Six of ten pairs changed Monterroso category**, Guiña × Zorro culpeo
  crossing Moderate → High. No estimator switched and retiring the R episode rule
  moved zero rows, so the delta is the recovery, not this session. The written
  interpretation of the overlap results must be re-read against the new table.
  Detection-rate and occupancy denominators also changed definition (field-record
  effort instead of days-with-photos), so every script-02 figure moved by
  construction rather than by data.
