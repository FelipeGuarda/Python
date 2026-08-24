# 01_load_data.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Read the CANONICAL observation tables published by camera-traps, join camera
#   coordinates from the GeoJSON, optionally filter to a species subset, and save
#   clean R objects for every downstream analysis script.
#
# INPUT FILES
#   - camera-traps/data/campaigns/{otono_2025, primavera_2025, otono_2026}/observations.parquet
#   - camera-traps/data/CANONICAL_STATE.json   (the published contract; checked on load)
#   - plataforma-territorial/data/camera_trap_stations.geojson
#   - plataforma-territorial/data/boundary.geojson
#
# OUTPUT FILES  (written to data/ inside the project)
#   - records_all.rds      records for the active SPECIES_FILTER, all campaigns
#   - stations_sf.rds      spatial dataframe with camera locations
#   - boundary_sf.rds      reserve boundary
#   - record_table.rds     camtrapR record table
#   - stations_ct.rds      camtrapR CTtable
#
# REWRITTEN 2026-08-20 — what changed and why it matters to the results
#   This script used to read three `new_labeled_data_corrected.csv` files. Those
#   carry the REVIEWED ROWS ONLY and, more seriously, an UNRESOLVED
#   `observationType`: every row reads `animal`, including the 815 across the three
#   campaigns where the reviewer had written in `observationComments` that the frame
#   holds no animal. The old filter here survived that by accident — it also required
#   a non-empty `scientificName`, and those rows are blank there — but it was luck,
#   not a control.
#
#   Worse, and not survivable by luck: the "spring" campaign read here was
#   `pv_2025_2026`, which is NOT a campaign. It is a second review pass over
#   primavera_2025's cards, made in April and superseded by primavera's own review in
#   August. `primavera_2025` was never read at all. Of the 606 image keys the two
#   share, 128 carried a different species.
#
#   The canonical parquet fixes both: `observation_type` is resolved (see
#   `resolve_review()` in camtrap/observations.py), the row set is every still in the
#   gated export, and `valid_effort` marks stations whose operating period is unknown
#   and which must therefore leave effort DENOMINATORS as well as numerators.
#
# WHAT THIS SCRIPT NO LONGER DOES, DELIBERATELY
#   - Parse station IDs. Three campaigns used three grammars ("CT01", "TC10_M3.2",
#     "CT_18") and each had its own block here. `camtrap/stations.py` owns that, and
#     it is fail-closed: an unrecognised station stops the ingest rather than becoming
#     a dropped row. The parquet arrives with `camera_num` already resolved.
#   - Cross-validate the SD-card code against the GeoJSON. That check confirmed a
#     station label had been parsed correctly. No label is parsed here any more, so
#     the check has no subject.
#   - Filter out `"No reconocible"` by string. The canonical table types those rows
#     `unknown`, not `animal`, so they never reach the animal filter.
#
# SPECIES FILTER
#   Set SPECIES_FILTER to a character vector of Latin names to restrict the output.
#   Set to NULL to retain ALL identified species (for plataforma / full dataset).
#
# HOW TO RE-RUN FOR A NEW CAMPAIGN
#   1. Re-ingest it in camera-traps:  python timestamps.py --campaign <name>
#   2. Re-publish the contract:       python -m camtrap.canonical_state --publish
#   3. Add its slug to CAMPAIGNS below. Nothing else in this file changes.
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)        # reproducible relative paths (auto-detects project root)
library(readr)       # fast CSV reading with consistent type inference
library(dplyr)       # data manipulation
library(stringr)     # string parsing for station ID extraction
library(lubridate)   # datetime parsing
library(sf)          # reading GeoJSON and spatial operations
library(jsonlite)    # reading CANONICAL_STATE.json (the published data contract)


# Parquet reader. `nanoparquet` is preferred — it is tiny and has no Arrow
# dependency; `arrow` is accepted if already installed. One of the two is REQUIRED:
#   Rscript -e 'install.packages("nanoparquet", repos="https://cloud.r-project.org")'
# We deliberately do NOT fall back to a CSV export of the canonical table. A second
# published file would be a second source of truth, which is the exact failure this
# rewrite exists to remove.
.read_parquet <- if (requireNamespace("nanoparquet", quietly = TRUE)) {
  function(path) as.data.frame(nanoparquet::read_parquet(path))
} else if (requireNamespace("arrow", quietly = TRUE)) {
  function(path) as.data.frame(arrow::read_parquet(path))
} else {
  stop(
    "No parquet reader available. Install one:\n",
    "  install.packages(\"nanoparquet\", repos = \"https://cloud.r-project.org\")\n",
    "camera-traps publishes observations.parquet; this script no longer reads CSVs.",
    call. = FALSE
  )
}

# Announce to `here` where the project root is relative to this script.
# This writes a tiny `.here` file in the project root on first run.
here::i_am("R/01_load_data.R")

# Owns which records are admissible for which question, and the unit of analysis.
# Sourced AFTER here::i_am(): before it, here() resolves to the monorepo root rather
# than this project, and the path silently misses.
source(here::here("R", "00_admissibility.R"))


# ── 1. Paths ─────────────────────────────────────────────────────────────────
# TODO(portability): these are absolute Windows paths and so this script cannot run
# on the Linux box. Parameterising them is tracked as V2-REVIEW 1.11.

CAMERA_TRAPS <- "C:/Users/USUARIO/Dev/Python/camera-traps"
PLATAFORMA   <- "C:/Users/USUARIO/Dev/Python/plataforma-territorial"

# Campaign slugs, in chronological order of retrieval.
#
# `pv_2025_2026` is ABSENT ON PURPOSE and must not be added back. It is a second
# review pass over primavera_2025, not a campaign; while this script read it in
# primavera's place the spring data was the superseded April labels. Its files were
# deleted on 2026-08-20 after being measured to hold no unique records.
CAMPAIGNS <- c("otono_2025", "primavera_2025", "otono_2026")

# Human-readable labels used in every figure. Keys must match CAMPAIGNS.
CAMPAIGN_LABELS <- c(
  otono_2025     = "Otono_2025",
  primavera_2025 = "Primavera_2025",
  otono_2026     = "Otono_2026"
)

PATH_STATE    <- file.path(CAMERA_TRAPS, "data", "CANONICAL_STATE.json")
PATH_GEOJSON  <- file.path(PLATAFORMA, "data", "camera_trap_stations.geojson")
PATH_BOUNDARY <- file.path(PLATAFORMA, "data", "boundary.geojson")

# Output directory
dir.create(here("data"), showWarnings = FALSE)


# ── 2. Species configuration ──────────────────────────────────────────────────
# FOCAL_SPECIES maps Latin names (as in scientificName column) to figure labels.
# NATIVE_SPECIES / INVASIVE_SPECIES drive colour coding in all figures.
#
# SPECIES_FILTER controls what ends up in records_all.rds:
#   - Set to names(FOCAL_SPECIES) for the pehuen research analysis (focal 6).
#   - Set to NULL to retain ALL identified species (plataforma / full dataset).

FOCAL_SPECIES <- c(
  "Puma concolor"          = "Puma",
  "Leopardus guigna"       = "Guina",
  "Lycalopex culpaeus"     = "Zorro culpeo",
  "Sus scrofa"             = "Jabali",
  "Lepus europaeus"        = "Liebre",
  "Canis lupus familiaris" = "Perro"
)

NATIVE_SPECIES    <- c("Puma concolor", "Leopardus guigna", "Lycalopex culpaeus")
INVASIVE_SPECIES  <- c("Sus scrofa", "Lepus europaeus", "Canis lupus familiaris")

# ── CHANGE THIS to NULL to keep all identified species ────────────────────────
SPECIES_FILTER <- names(FOCAL_SPECIES)


# ── 2b. Independence threshold for record_table ───────────────────────────────
# O'Brien et al. (2003) 30-minute convention: consecutive triggers of the same
# species at the same station within this window are collapsed to one event.
# Applied to record_table (activity / overlap analyses); records_all keeps raw
# triggers for date-based analyses that do not depend on event independence.
MIN_DELTA_TIME_MIN <- EPISODE_GAP_MINUTES   # from R/00_admissibility.R


# ── 3. Load and parse the station coordinates (GeoJSON) ───────────────────────
# The GeoJSON is GENERATED from camera-traps/data/campaigns/estaciones.csv, which owns
# station identity; it is not hand-maintained. It holds, per physical camera trap:
#   id       → canonical station label "CT01", "CT02", … (was "TC-01" until 2026-08-24;
#              one spelling is now used in the field, the pipeline and the platform)
#   tc       → integer station number (1–27); this is the JOIN KEY
#   geometry → WGS-84 point coordinates (lon, lat)
# `sd_card` was dropped on 2026-08-24: it was the M## grid-module tag from the old
# folder names, not an SD card, not unique (M15 was both CT11 and CT18), and the
# Primavera cross-validation that once read it no longer exists.

stations_sf <- st_read(PATH_GEOJSON, quiet = TRUE) %>%
  # Rename `tc` to `tc_num` to make its role as join key explicit
  rename(tc_num = tc) %>%
  # Keep only the columns we need downstream
  select(id, tc_num, altitude_m, geometry)

message(sprintf("Loaded %d camera stations from GeoJSON.", nrow(stations_sf)))


# ── 4. Verify the published data contract ────────────────────────────────────
# camera-traps publishes CANONICAL_STATE.json alongside the parquets: schema version,
# column list, and per-campaign row/station/animal counts. We check it before reading
# anything, because on 2026-08-19 those tables went from 3,359 rows to 35,807 and not
# one consumer raised an error. A contract nobody verifies is a comment.

state <- jsonlite::fromJSON(PATH_STATE, simplifyVector = TRUE)

EXPECTED_SCHEMA_VERSION <- 2L
if (as.integer(state$schema_version) != EXPECTED_SCHEMA_VERSION) {
  stop(sprintf(
    paste0("CANONICAL_STATE.json declares schema_version %s but this script was written ",
           "against %d.\nThe canonical table has changed shape. Read ",
           "camera-traps/camtrap/observations.py (CANONICAL_COLUMNS) and update this ",
           "script deliberately -- do not just bump the number."),
    state$schema_version, EXPECTED_SCHEMA_VERSION
  ), call. = FALSE)
}

missing_campaigns <- setdiff(CAMPAIGNS, names(state$campaigns))
if (length(missing_campaigns) > 0) {
  stop("Campaigns requested but not present in CANONICAL_STATE.json: ",
       paste(missing_campaigns, collapse = ", "),
       "\nRe-ingest them in camera-traps, then re-publish the contract.", call. = FALSE)
}

message(sprintf(
  "Canonical contract: schema_version %s, %s rows total, %s stations.",
  state$schema_version, format(state$n_rows_total, big.mark = ","),
  state$n_stations_total
))


# ── 5. Read the canonical tables ─────────────────────────────────────────────
# One function, all campaigns. There is nothing per-campaign left to special-case:
# station resolution, clock repair, the review resolution and the Spanish->Latin
# lookup all happened upstream, and the parquet arrives with the answers.

read_canonical <- function(campaign) {
  path <- file.path(CAMERA_TRAPS, "data", "campaigns", campaign, "observations.parquet")
  if (!file.exists(path)) {
    stop(sprintf("Missing canonical table: %s\nRun: cd %s && python timestamps.py --campaign %s",
                 path, CAMERA_TRAPS, campaign), call. = FALSE)
  }
  raw <- .read_parquet(path)

  needed <- c("campaign", "camera_num", "station_canonical", "datetime", "valid_date",
              "valid_time_of_day", "valid_effort", "repair_method", "observation_type",
              "species_latin", "review_outcome", "review_resolution")
  absent <- setdiff(needed, names(raw))
  if (length(absent) > 0) {
    stop(sprintf("%s: canonical table is missing column(s): %s",
                 campaign, paste(absent, collapse = ", ")), call. = FALSE)
  }

  # Row count must match the published contract exactly. If someone re-ingested a
  # campaign without re-publishing, we want to hear about it here and not in a figure.
  declared <- as.integer(state$campaigns[[campaign]]$n_rows)
  if (nrow(raw) != declared) {
    stop(sprintf(
      paste0("%s: parquet holds %d rows but CANONICAL_STATE.json declares %d.\n",
             "The table was rebuilt without re-publishing the contract. In camera-traps: ",
             "python -m camtrap.canonical_state --publish"),
      campaign, nrow(raw), declared), call. = FALSE)
  }

  clean <- raw %>%
    # (a) Identified animals only. `observation_type` here is the RESOLVED type -- the
    #     reviewer's verdict, not the classifier's guess -- so this filter now removes
    #     the human, vehicle, blank and unknown rows correctly. `species_latin` is ""
    #     rather than NA on non-animal rows, hence both tests.
    filter(
      observation_type == "animal",
      !is.na(species_latin),
      species_latin != ""
    ) %>%
    mutate(
      # tz = "UTC" is a LABEL here, not a conversion, and it must stay explicit.
      # Camera clocks are set to Chile local time and the canonical table stores that
      # reading verbatim, so the hour is already the local hour an animal was active;
      # tagging it UTC stops R shifting it. tz = "" would be silently
      # machine-dependent: this conda R has no tzdata so it is a no-op here, but on a
      # box WITH tzdata it converts by the local offset and moves every
      # activity-pattern figure by 3-4 hours.
      datetime = as.POSIXct(datetime, tz = "UTC"),
      campaign = unname(CAMPAIGN_LABELS[campaign]),
      tc_num   = as.integer(camera_num)
    ) %>%
    select(
      campaign, tc_num, station_canonical, datetime,
      valid_date, valid_time_of_day, valid_effort, repair_method,
      species_latin, review_outcome, review_resolution
    )

  message(sprintf(
    "  [%s] %d rows in table; %d identified-animal records (%d species).",
    campaign, nrow(raw), nrow(clean), dplyr::n_distinct(clean$species_latin)
  ))
  clean
}

message("\nReading canonical tables...")
records_raw <- bind_rows(lapply(CAMPAIGNS, read_canonical))


# ── 6. Effort validity ───────────────────────────────────────────────────────
# valid_effort is STATION-level: FALSE means this camera's operating period is
# unknown, so its trap-nights are unknowable and it must leave the effort DENOMINATOR
# as well as the numerator. Every row of such a station carries FALSE, including rows
# whose own timestamp is fine. This script had no access to the flag before
# 2026-08-20 and so could not have excluded those stations from an effort calculation.
#
# We keep the rows and surface the count: no analysis here divides by trap-nights yet.
# Any future occupancy or detection-rate figure MUST filter on valid_effort == TRUE.
n_no_effort <- sum(!records_raw$valid_effort, na.rm = TRUE)
if (n_no_effort > 0) {
  message(sprintf(
    paste0("  NOTE: %d records sit at stations with valid_effort == FALSE. Fine for ",
           "presence and activity; NOT usable in any trap-night denominator."),
    n_no_effort
  ))
}


# ── 7. Join camera coordinates ───────────────────────────────────────────────
# tc_num comes from the canonical table (camera_num), already resolved and validated
# upstream, so this is a plain lookup. left_join, not inner_join, so an unmatched
# station surfaces as NA instead of vanishing.

stations_lookup_full <- st_drop_geometry(stations_sf) %>%
  select(tc_num, id, altitude_m)

records_joined <- records_raw %>%
  left_join(stations_lookup_full, by = "tc_num") %>%
  rename(station_id = id)

unmatched <- filter(records_joined, is.na(station_id))
if (nrow(unmatched) > 0) {
  warning(sprintf(
    "%d records have camera numbers absent from the GeoJSON: %s. The station registry is behind the campaign data.",
    nrow(unmatched), paste(sort(unique(unmatched$tc_num)), collapse = ", ")
  ))
}


# ── 9. Combine campaigns and filter to target species ────────────────────────
# When SPECIES_FILTER is a character vector, only those Latin names are kept.
# When SPECIES_FILTER is NULL, all identified species are retained.

records_all <- records_joined %>%
  # Apply species filter (or keep all if NULL)
  { if (!is.null(SPECIES_FILTER)) filter(., species_latin %in% SPECIES_FILTER) else . } %>%
  # Add human-readable species label (NA for species outside FOCAL_SPECIES)
  mutate(
    species_label = ifelse(
      species_latin %in% names(FOCAL_SPECIES),
      FOCAL_SPECIES[species_latin],
      species_latin
    ),
    guild = case_when(
      species_latin %in% NATIVE_SPECIES   ~ "Native",
      species_latin %in% INVASIVE_SPECIES ~ "Invasive",
      TRUE                                ~ "Other"
    ),
    # Derive date and time-of-day fields used in activity analyses. These are NA for
    # records whose clock could not be repaired, which is correct — an unknown hour
    # must not be silently imputed.
    date      = as.Date(datetime),
    hour      = hour(datetime),
    # Time of day as a fraction of 24 hours, then converted to radians (0 to 2π)
    # This is the format expected by the `overlap` package
    time_rad  = (hour(datetime) * 3600 + minute(datetime) * 60 + second(datetime)) /
                86400 * 2 * pi,
    # ── ADMISSIBILITY, not a filter.
    # This used to be `filter(!is.na(datetime))` at the end of this pipeline, which
    # imposed the strictest rule on every downstream script whether or not it asked.
    # Presence/absence needs a station, not a clock: puma is recorded at 8 stations
    # and the spatial maps showed 6, because CT03's and CT18's clocks failed. The
    # record is kept; the flag says what it may be used for. See R/00_admissibility.R.
    #
    # valid_date and valid_time_of_day come from the canonical table and are NOT
    # re-derived here — camera-traps owns clock repair.
    time_admissible = !is.na(datetime) & valid_date & valid_time_of_day
  )

message(sprintf(
  "\nFinal dataset: %d records across %d stations and %d campaigns. (SPECIES_FILTER: %s)",
  nrow(records_all),
  n_distinct(records_all$station_id),
  n_distinct(records_all$campaign),
  if (is.null(SPECIES_FILTER)) "ALL" else paste(SPECIES_FILTER, collapse = ", ")
))

# What each kind of question may use. Printed rather than assumed, because the gap
# between the two is exactly the defect this structure exists to prevent.
n_place <- nrow(admissible(records_all, "place", quiet = TRUE))
n_time  <- nrow(admissible(records_all, "time",  quiet = TRUE))
st_place <- n_distinct(admissible(records_all, "place", quiet = TRUE)$station_id)
st_time  <- n_distinct(admissible(records_all, "time",  quiet = TRUE)$station_id)
message(sprintf(
  "  admissible for PLACE (presence/absence): %d records, %d stations\n  admissible for TIME  (activity/overlap) : %d records, %d stations",
  n_place, st_place, n_time, st_time))
if (st_place > st_time) {
  message(sprintf(
    "  NOTE: %d station(s) appear ONLY in place-based analyses — their clocks could not be repaired: %s",
    st_place - st_time,
    paste(setdiff(unique(admissible(records_all, "place", quiet = TRUE)$station_id),
                  unique(admissible(records_all, "time",  quiet = TRUE)$station_id)),
          collapse = ", ")))
}
n_ep <- nrow(episodes(records_all, quiet = TRUE))
message(sprintf(
  "  independent episodes (%d-min rule)      : %d  <- the unit for any COUNT; records are images",
  EPISODE_GAP_MINUTES, n_ep))
print(table(records_all$species_label, records_all$campaign))


# ── 10. Save core outputs ─────────────────────────────────────────────────────

saveRDS(records_all,  here("data", "records_all.rds"))
saveRDS(stations_sf,  here("data", "stations_sf.rds"))
saveRDS(st_read(PATH_BOUNDARY, quiet = TRUE), here("data", "boundary_sf.rds"))

message("\nSaved: data/records_all.rds, data/stations_sf.rds, data/boundary_sf.rds")


# ── 11. Build camtrapR-compatible tables ──────────────────────────────────────
# camtrapR functions (activityDensity, activityOverlap, detectionMaps) require
# data in a specific format.  We build these tables here so downstream scripts
# can use camtrapR directly without reformatting.
#
# RECORD TABLE — one row per detection event.
#   Required columns:
#     Station          — station ID; must match CTtable$Station
#     Species          — species label (human-readable, used in figure legends)
#     DateTimeOriginal — POSIXct timestamp
#     Date             — calendar date
#     Time             — time as character "HH:MM:SS"
#   We also carry Campaign as an optional grouping column.
#
# CAMERA TRAP TABLE (CTtable) — one row per station.
#   Required columns:
#     Station   — station ID (must match record_table$Station)
#     Longitude — decimal degrees, WGS-84
#     Latitude  — decimal degrees, WGS-84

# IMPORTANT: record_table is consumed by camtrapR's activityDensity() and
# activityOverlap(), both of which use time-of-day to compute kernel density
# estimates. Rows with valid_time_of_day == FALSE (e.g. CT-18 Otoño 2026,
# repaired via last_real_proxy anchor) carry approximate dates but rotated
# time-of-day — they MUST be excluded from time-of-day analyses.
#
# We also apply the 30-minute independence filter here (see
# MIN_DELTA_TIME_MIN in section 2b): consecutive triggers of the same species
# at the same station within that window collapse to one event.
# `filter_independent_events` walks each (station, species, campaign) group in
# datetime order and keeps a trigger only if it is at least
# `min_delta_min` past the previous *kept* trigger (O'Brien et al. 2003
# "against last independent record" convention).
# MOVED 2026-08-20 to R/00_admissibility.R as `independent()` / `keep_after_min_gap()`.
# It lived here while record_table was the only consumer; the spatial scripts now need
# the same rule, and a second copy is how two figures come to disagree about what an
# independent detection is. This wrapper keeps the local name.
filter_independent_events <- function(df, min_delta_min) {
  independent(df, gap_minutes = min_delta_min)
}

record_table <- records_all %>%
  filter(valid_time_of_day == TRUE) %>%
  filter_independent_events(min_delta_min = MIN_DELTA_TIME_MIN) %>%
  transmute(
    Station          = station_id,
    Species          = species_label,
    DateTimeOriginal = datetime,
    Date             = date,
    Time             = format(datetime, "%H:%M:%S"),
    # Time of day in radians — precomputed here so overlap analyses
    # (04_temporal_overlap.R) source their numeric AND visual layers from the
    # same independence-filtered rows. If we recomputed time_rad from
    # records_all downstream, the numeric n would silently reflect raw
    # triggers while the plot reflected independent events — the bug the
    # single-source pattern here prevents.
    time_rad         = time_rad,
    Campaign         = campaign
  )

message(sprintf(
  "record_table: %d rows after filtering to valid_time_of_day == TRUE and to independent events (%d-min minimum gap; vs %d in records_all).",
  nrow(record_table), MIN_DELTA_TIME_MIN, nrow(records_all)
))

# Extract WGS-84 coordinates from the sf geometry column.
# st_coordinates() returns a matrix with columns X (longitude) and Y (latitude).
coords <- st_coordinates(stations_sf)

stations_ct <- stations_sf %>%
  st_drop_geometry() %>%
  rename(Station = id) %>%
  mutate(
    Longitude = coords[, "X"],
    Latitude  = coords[, "Y"]
  ) %>%
  select(Station, Longitude, Latitude, altitude_m)

saveRDS(record_table, here("data", "record_table.rds"))
saveRDS(stations_ct,  here("data", "stations_ct.rds"))

message("Saved: data/record_table.rds  (camtrapR format)")
message("Saved: data/stations_ct.rds   (camtrapR CTtable format)")
message("Run 02_detection_summary.R next.")
