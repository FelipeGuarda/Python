# pehuen-species-interactions

Prototype analysis of species distribution and temporal interactions in Bosque
Pehuen, based on camera-trap data from the Otoño 2025 and Primavera 2025
campaigns.  Written in R using `camtrapR` and `overlap`.

---

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate pehuen-analysis
```

If any package fails to resolve from conda-forge, run the CRAN fallback:

```bash
conda activate pehuen-analysis
Rscript setup_packages.R
```

### 2. Verify data paths

Open `R/01_load_data.R` and confirm `PATH_OTONO`, `PATH_PV`, `PATH_OT26`,
`PATH_GEOJSON`, and `PATH_BOUNDARY` point to the correct locations on this
machine.

> **Important — corrected vs. reviewed CSVs.** The R loader reads
> `new_labeled_data_corrected.csv`, **not** `new_labeled_data_reviewed.csv`.
> The corrected CSV is produced upstream by `camera-traps/timestamps.py`
> (camera-clock-reset detection + repair). If `_corrected.csv` is missing
> for a campaign, regenerate it before running any analysis:
> ```bash
> cd C:/Users/USUARIO/Dev/Python/camera-traps
> python timestamps.py --campaign <name>
> ```
> See `camera-traps/README.md` → "Step 4b — Timestamp quality" for the
> upstream protocol and anchor-file schema.

---

## Running the analysis

Run scripts in order — each script saves `.rds` files that the next one reads.

```bash
conda activate pehuen-analysis

Rscript R/01_load_data.R          # reads CSVs + GeoJSON, saves data/*.rds
Rscript R/02_detection_summary.R  # bar charts: counts, rates, occupancy
Rscript R/03_activity_patterns.R  # 24h kernel density activity curves
Rscript R/04_temporal_overlap.R   # Δ1/Δ4 pairwise overlap + CI + Monterroso 2014 category
Rscript R/05_spatial_distribution.R  # detection bubble maps
```

All figures are written to `figures/`. Script 04 also persists a numeric
results table to `data/overlap_stats.csv` (estimator, overlap estimate,
95% bootstrap CI, Monterroso category per pair).

**Overlap estimator (Ridout & Linkie 2009).** Script 04 selects the estimator
per pair from the smaller sample: `Δ4` when `min(n_A, n_B) ≥ 50`, `Δ1`
otherwise (crossover threshold from the `overlap` package documentation).
The estimator applied to each pair is written to `data/overlap_stats.csv`
(column `estimator`), to the per-pair PNG footnote, and encoded by point
shape in the summary figure (Δ4 filled / Δ1 open).

**Overlap classification (Monterroso et al. 2014).** Script 04 assigns each
pair a Low / Moderate / High label from its 95% bootstrap CI:

- **Low**: overlap < 0.50
- **Moderate**: 0.50 ≤ overlap < 0.75
- **High**: overlap ≥ 0.75

A pair earns a clean single-band label only when its entire CI sits inside
one band. When the CI straddles a threshold, the pair gets a compound
label (e.g. `Moderate–High`) so the report doesn't overstate confidence.
Both the per-pair PNGs (`figures/overlap_pairs/`) and the summary figure
(`figures/04_overlap_summary.png`) show the category alongside the overlap
estimate and the CI.

**Event independence.** `record_table.rds` (consumed by activity / overlap
analyses) is pre-filtered to independent events with a 30-minute minimum
inter-detection interval per (station × species × campaign), following
O'Brien et al. (2003). See `MIN_DELTA_TIME_MIN` in `R/01_load_data.R`.
Date-based analyses that use `records_all.rds` still see raw triggers.

---

## Planned / candidate analyses

`docs/methods-menu-interactions.md` is a critical, sourced menu of alternative
spatial and temporal interaction methods (activity-level estimation, occupancy
with altitude covariates, the Niedballa spatiotemporal-avoidance framework,
güiña SECR, and methods deliberately excluded with citations), each triaged
against this array's sample-size constraints. Its "Open items" list is the
current analysis backlog — start there before adding new analysis code.

---

## Adding a future campaign

1. **Upstream first** — ensure the new campaign exists in `camera-traps/data/campaigns/<name>/`
   with both `new_labeled_data_reviewed.csv` and `deployment_anchors.csv`, and
   has been processed through `python timestamps.py --campaign <name>` to
   produce `new_labeled_data_corrected.csv`.
2. In `R/01_load_data.R`, add `PATH_NEW_CAMPAIGN <- ".../new_labeled_data_corrected.csv"`
   in the Paths block.
3. Call `read_campaign_csv(PATH_NEW_CAMPAIGN, "<Campaign_Label>")`.
4. Apply the appropriate station ID standardisation block (copy an existing
   block and adapt the regex for the new format).
5. Add the new dataframe to the `bind_rows()` call.
6. Re-run all scripts.

## Timestamp validity flags

`records_all.rds` carries three columns originating in the upstream
`timestamps.py` pipeline:

- `valid_date` — TRUE if the date is trustworthy. FALSE for rows whose
  station had a clock reset with no field anchor (currently CT-15 / CT-16 /
  CT-19 Otoño 2025, TC-16 Primavera and PV). Such rows have `datetime = NA`
  and are dropped by the existing `filter(!is.na(datetime))`.
- `valid_time_of_day` — TRUE if the time-of-day is trustworthy. FALSE for
  rows repaired via `last_real_proxy` anchor (currently CT_18 Otoño 2026:
  ~65 focal-species rows — dates approximate, time-of-day rotated by an
  unknown constant).
- `repair_method` — provenance string (`none`, `offset_from_last_real_proxy`,
  `unrepairable_pending_anchor`, etc.).

`record_table.rds` (used by camtrapR for activity / overlap analyses) is
pre-filtered to `valid_time_of_day == TRUE` in `01_load_data.R`. Custom
analyses that bypass `record_table` and read `records_all.rds` directly
must add their own `filter(valid_time_of_day)` if they depend on
time-of-day. Date-based analyses (detection rate, occupancy, spatial maps)
should use `filter(valid_date)` instead.

---

## Station ID mapping

| tc# | GeoJSON id | Otoño CSV    | Primavera CSV | Notes          |
|:---:|:----------:|:------------:|:-------------:|:--------------|
| 1   | TC-01      | CT01         | TC1_M7.2      | sd validated  |
| 2   | TC-02      | CT02         | —             | Otoño only    |
| 3   | TC-03      | CT03         | —             | Otoño only    |
| 4   | TC-04      | CT04         | TC4_M5.2      | sd validated  |
| 5   | TC-05      | CT05         | —             | Otoño only    |
| 6   | TC-06      | CT06         | TC6_M1.2      | sd validated  |
| 7   | TC-07      | CT07         | —             | Otoño only    |
| 8   | TC-08      | CT08         | —             | Otoño only    |
| 9   | TC-09      | CT09         | TC9_M2.2      | sd validated  |
| 10  | TC-10      | CT10         | TC10_M3.2     | sd validated  |
| 11  | TC-11      | CT11         | TC11_M15.2    | sd validated  |
| 12  | TC-12      | CT12         | TC12_M17.2    | sd validated  |
| 13  | TC-13      | CT13         | TC13_M16.2    | sd validated  |
| 14  | TC-14      | CT14         | —             | Otoño only    |
| 15  | TC-15      | CT15         | TC15_M12.2    | sd validated  |
| 16  | TC-16      | CT16         | TC16_M13.2    | sd validated  |
| 17  | TC-17      | CT17         | —             | Otoño only    |
| 18  | TC-18      | CT18         | TC18_M15.2    | sd validated  |
| 19  | TC-19      | CT19         | TC19_M16.2    | sd validated  |
| 20  | TC-20      | CT20         | TC20_M17.2    | sd validated  |
| 21–26 | TC-21…TC-26 | —         | —             | GeoJSON only  |

`100EK113` (Primavera): anomalous entry, filtered out in 01_load_data.R.

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

---

## Project status

- **Last Updated:** 2026-07-28
- **What Changed:** Applied the 30-minute independence filter to
  `record_table.rds` (O'Brien et al. 2003); switched pairwise-overlap
  estimator dispatch to Δ1 vs Δ4 per Ridout & Linkie 2009 (crossover at
  `min(n_A, n_B) < 50`) with 1000 bootstrap resamples for the 95% CI;
  `data/overlap_stats.csv` gains an `estimator` column and drops
  `small_sample` (now encoded by estimator directly).
- **Integration Status:** `In Progress` — scripts 01–06 functional with the
  above corrections applied. [REMAINING: effort matrix (deployment
  start/end dates) needed to unblock formal occupancy models and honest
  detection rates; raw camera-installation file exists but needs cleanup
  with field collaborator before use.]
- **Blockers/Notes:** True per-station deployment start/end dates still not
  compiled — `camera-traps/build_camera_operation.py` and downstream
  `R/00_camera_operation.R` are designed but deferred pending cleanup of
  the source installation file with Felipe's colleague. See
  `docs/methods-menu-interactions.md` § I "Open items".
