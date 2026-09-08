# 01_load_data.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Verify the camera-trap contract, read the CANONICAL observation tables and
#   deployment windows published by camera-traps, join station coordinates, filter to
#   the focal species, and save clean R objects for every downstream script.
#
# INPUT FILES  (all published by camera-traps unless noted)
#   - data/CANONICAL_STATE.json                        the contract; verified FIRST
#   - data/campaigns/<campaign>/observations.parquet   the canonical table
#   - data/campaigns/<campaign>/deployments.csv        field windows and effort
#   - data/campaigns/estaciones.geojson                station coordinates
#   - plataforma-territorial/data/boundary.geojson     reserve boundary (platform's)
#
# OUTPUT FILES  (data/ inside the project)
#   - records_all.rds        one row per IMAGE, focal species, all campaigns
#   - deployments.rds        one row per (campaign, station): window, effort, media
#   - stations_sf.rds        sf points, one per station
#   - boundary_sf.rds        reserve boundary
#   - record_table.rds       camtrapR record table, one row per EPISODE
#   - stations_ct.rds        camtrapR CTtable
#   - contract_stamp.json    what the above were built from; downstream scripts
#                            refuse to run if the published contract has moved
#
# WHAT THIS SCRIPT DOES NOT DECIDE, DELIBERATELY (manual 10F.3)
#   Station identity, clock repair, the review verdict, the Spanish->Latin lookup,
#   which frames are one detection event, and how many days a camera operated. Every
#   one arrives in the table or in deployments.csv with the answer. Between
#   2026-04 and 2026-08 this file made four of those decisions itself, each with its
#   own grammar per campaign, and each was eventually measured wrong against the
#   producer's. See R/00_admissibility.R and R/00_contract.R for the history.
#
# HOW TO RE-RUN FOR A NEW CAMPAIGN
#   1. Re-ingest it in camera-traps:  python timestamps.py --campaign <name>
#   2. Re-publish the contract:       python -m camtrap.canonical_state --publish
#   3. Add its slug to CAMPAIGNS below, and a display name to CAMPAIGN_LABELS in
#      R/00_contract.R. Nothing else changes.
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)        # reproducible relative paths, anchored by .here in the project
library(dplyr)
library(lubridate)   # hour(), minute(), second()
library(sf)          # GeoJSON
library(jsonlite)    # used by R/00_contract.R

# Parquet reader. `nanoparquet` is preferred (tiny, no Arrow runtime); `arrow` is
# accepted if already installed. There is deliberately NO CSV fallback: a second
# published file would be a second source of truth, which is the failure the
# canonical table exists to remove.
.read_parquet <- if (requireNamespace("nanoparquet", quietly = TRUE)) {
  function(path) as.data.frame(nanoparquet::read_parquet(path))
} else if (requireNamespace("arrow", quietly = TRUE)) {
  function(path) as.data.frame(arrow::read_parquet(path))
} else {
  stop(
    "No parquet reader available. Install one:\n",
    "  install.packages(\"nanoparquet\", repos = \"https://cloud.r-project.org\")",
    call. = FALSE
  )
}

# `.here` in the project root anchors here() to THIS directory. Without it rprojroot
# walks up to the monorepo's .git and every here("data", ...) resolves to a top-level
# data/ that does not exist. The 00_* modules must be sourced after this line.
here::i_am("R/01_load_data.R")
source(here::here("R", "00_contract.R"))
source(here::here("R", "00_admissibility.R"))


# ── 1. Campaigns and species ─────────────────────────────────────────────────

# Slugs as the producer names them, in order of retrieval. The slug is the campaign's
# identity in every table this script writes; figure text goes through
# campaign_label(). `pv_2025_2026` must never be added: it was a second review pass
# over primavera_2025, not a campaign, and its files were deleted on 2026-08-20.
CAMPAIGNS <- c("otono_2025", "primavera_2025", "otono_2026")

# Latin names as they appear in `species_latin`, mapped to figure labels.
FOCAL_SPECIES <- c(
  "Puma concolor"          = "Puma",
  "Leopardus guigna"       = "Guina",
  "Lycalopex culpaeus"     = "Zorro culpeo",
  "Sus scrofa"             = "Jabali",
  "Lepus europaeus"        = "Liebre",
  "Canis lupus familiaris" = "Perro"
)
NATIVE_SPECIES   <- c("Puma concolor", "Leopardus guigna", "Lycalopex culpaeus")
INVASIVE_SPECIES <- c("Sus scrofa", "Lepus europaeus", "Canis lupus familiaris")

# Set to NULL to keep every identified species (full dataset) instead of the focal 6.
SPECIES_FILTER <- names(FOCAL_SPECIES)


# ── 2. The handshake: verify the contract before opening anything ────────────
# Absent, unreadable, wrong schema or missing campaign all REFUSE here with exit
# status 2. Nothing below runs against an unverified contract.

state <- contract_load(CAMPAIGNS)

CAMPAIGNS_DIR <- file.path(producer_dir(), "data", "campaigns")
PATH_GEOJSON  <- file.path(CAMPAIGNS_DIR, "estaciones.geojson")
PATH_BOUNDARY <- file.path(monorepo_root(), "plataforma-territorial", "data", "boundary.geojson")

dir.create(here("data"), showWarnings = FALSE)


# ── 3. Stations ──────────────────────────────────────────────────────────────
# Generated by camera-traps from estaciones.csv, which owns station identity; not
# hand-maintained. `id` is the canonical label "CT01".."CT27" and is the JOIN KEY
# against the table's `station_canonical` -- the same string on both sides, so there
# is no integer to derive and nothing to parse.

stations_sf <- st_read(PATH_GEOJSON, quiet = TRUE) %>%
  select(id, altitude_m, geometry)

message(sprintf("Loaded %d stations from %s", nrow(stations_sf), basename(PATH_GEOJSON)))


# ── 4. The canonical tables ──────────────────────────────────────────────────

# Columns this project reads. A missing one stops the load; extra ones are ignored.
NEEDED <- c("campaign", "station_canonical", "datetime", "valid_date",
            "valid_time_of_day", "valid_effort", "repair_method", "observation_type",
            "species_latin", "review_outcome", "review_resolution", EPISODE_COLUMN)

read_canonical <- function(campaign) {
  path <- file.path(CAMPAIGNS_DIR, campaign, "observations.parquet")
  if (!file.exists(path)) {
    stop(sprintf("Missing canonical table: %s\nRun: cd %s && python timestamps.py --campaign %s",
                 path, producer_dir(), campaign), call. = FALSE)
  }
  raw <- .read_parquet(path)

  absent <- setdiff(NEEDED, names(raw))
  if (length(absent)) {
    stop(sprintf("%s: canonical table is missing column(s): %s",
                 campaign, paste(absent, collapse = ", ")), call. = FALSE)
  }

  # The one on-disk check kept here: the parquet must hold the rows the contract
  # declares, or someone re-ingested without re-publishing.
  declared <- as.integer(state$campaigns[[campaign]]$n_rows)
  if (nrow(raw) != declared) {
    refuse(sprintf(
      paste0("%s: parquet holds %d rows but the contract declares %d. The table was ",
             "rebuilt without re-publishing. In camera-traps: ",
             "python -m camtrap.canonical_state --publish"),
      campaign, nrow(raw), declared))
  }

  # Station-level effort validity for EVERY station in the table, read before the
  # animal filter so that stations with zero focal detections are still known.
  # valid_effort is set by the producer's clock diagnosis and is constant per station.
  station_effort <- raw %>%
    distinct(campaign, station_id = station_canonical, valid_effort)

  clean <- raw %>%
    # `observation_type` is the RESOLVED type (the reviewer's verdict, not the
    # classifier's guess). `species_latin` is "" on non-animal rows, hence both tests.
    filter(observation_type == "animal", !is.na(species_latin), species_latin != "") %>%
    mutate(
      # tz = "UTC" is a LABEL, not a conversion. Camera clocks read Chile local time
      # and the table stores that reading verbatim, so the hour is already the hour
      # the animal was active. tz = "" would be machine-dependent: a no-op on an R
      # without tzdata, a 3-4 h shift on one with it.
      datetime   = as.POSIXct(datetime, tz = "UTC"),
      station_id = station_canonical
    ) %>%
    select(campaign, station_id, datetime, valid_date, valid_time_of_day, valid_effort,
           repair_method, species_latin, review_outcome, review_resolution,
           all_of(EPISODE_COLUMN))

  message(sprintf("  [%s] %d rows in table; %d identified-animal records (%d species).",
                  campaign, nrow(raw), nrow(clean), n_distinct(clean$species_latin)))
  list(records = clean, station_effort = station_effort)
}

message("\nReading canonical tables...")
loaded         <- lapply(CAMPAIGNS, read_canonical)
records_raw    <- bind_rows(lapply(loaded, `[[`, "records"))
station_effort <- bind_rows(lapply(loaded, `[[`, "station_effort"))


# ── 5. Deployment windows and effort ─────────────────────────────────────────
# One row per (campaign, station) from the FIELD RECORD, so a window exists whether
# or not the camera's clock survived. `media_status` says why a station has no
# stills, and it decides a DENOMINATOR:
#   in_canonical        stills are in the table. The only rows a stills-based rate
#                       may divide by.
#   video_only_offline  the camera WAS sampling; its media is video outside this
#                       pipeline. Belongs in an occupancy/presence denominator, must
#                       be excluded from any stills-based rate.
#   card_failure        recorded nothing. No effort for any question.
#   unexplained / no_field_dates   not an effort figure; surfaced, never absorbed.
# `valid_effort` (producer's clock diagnosis) is joined in so a rate can restrict its
# denominator to the stations whose numerator it can actually see.

read_deployments <- function(campaign) {
  path <- file.path(CAMPAIGNS_DIR, campaign, "deployments.csv")
  if (!file.exists(path)) {
    stop(sprintf("Missing deployments: %s\nIn camera-traps: python -m camtrap.deployments",
                 path), call. = FALSE)
  }
  read.csv(path, stringsAsFactors = FALSE, colClasses = c(
    campaign = "character", station_id = "character", field_start = "character",
    field_end = "character", has_media = "character", media_status = "character",
    note = "character")) %>%
    mutate(
      # tz given explicitly for the same reason as the datetime label above: a
      # calendar date must not depend on the machine's TZ setting.
      field_start = as.Date(field_start, format = "%Y-%m-%d", tz = "UTC"),
      field_end   = as.Date(field_end,   format = "%Y-%m-%d", tz = "UTC"),
      field_days  = as.integer(field_days),
      has_media   = tolower(has_media) == "true"
    )
}

deployments <- bind_rows(lapply(CAMPAIGNS, read_deployments)) %>%
  left_join(station_effort, by = c("campaign", "station_id"))

unexplained <- filter(deployments, media_status %in% c("unexplained", "no_field_dates"))
if (nrow(unexplained) > 0) {
  warning(sprintf(
    "%d deployment(s) with no usable effort (%s). They are in deployments.rds and out of every denominator.",
    nrow(unexplained),
    paste(unique(paste(unexplained$campaign, unexplained$station_id)), collapse = ", ")))
}

message(sprintf("Deployments: %d station-campaigns; camera-days with stills: %s",
                nrow(deployments),
                format(sum(deployments$field_days[deployments$media_status == "in_canonical"]),
                       big.mark = ",")))


# ── 6. Join coordinates ──────────────────────────────────────────────────────
# left_join, so a station the registry does not know surfaces as NA rather than
# vanishing; admissible(., "place") drops NA stations and says so.

records_joined <- records_raw %>%
  left_join(st_drop_geometry(stations_sf) %>% select(id, altitude_m),
            by = c("station_id" = "id"))

unmatched <- filter(records_joined, !station_id %in% stations_sf$id)
if (nrow(unmatched) > 0) {
  warning(sprintf(
    "%d records at station(s) absent from %s: %s. The registry is behind the campaign data.",
    nrow(unmatched), basename(PATH_GEOJSON),
    paste(sort(unique(unmatched$station_id)), collapse = ", ")))
  records_joined$station_id[!records_joined$station_id %in% stations_sf$id] <- NA_character_
}


# ── 7. Species filter, labels, admissibility flag ────────────────────────────

records_all <- records_joined %>%
  { if (!is.null(SPECIES_FILTER)) filter(., species_latin %in% SPECIES_FILTER) else . } %>%
  mutate(
    species_label = ifelse(species_latin %in% names(FOCAL_SPECIES),
                           FOCAL_SPECIES[species_latin], species_latin),
    guild = case_when(
      species_latin %in% NATIVE_SPECIES   ~ "Native",
      species_latin %in% INVASIVE_SPECIES ~ "Invasive",
      TRUE                                ~ "Other"
    ),
    # NA where the clock could not be repaired; an unknown hour is never imputed.
    date     = as.Date(datetime),
    hour     = hour(datetime),
    # Time of day in radians (0..2*pi), the `overlap` package's input.
    time_rad = (hour(datetime) * 3600 + minute(datetime) * 60 + second(datetime)) /
               86400 * 2 * pi,
    # A FLAG, not a filter. Presence needs a station, not a clock; activity needs
    # both. Each script asks for the rule it needs through R/00_admissibility.R.
    time_admissible = !is.na(datetime) & valid_date & valid_time_of_day
  )

message(sprintf(
  "\nFinal dataset: %d records across %d stations and %d campaigns. (SPECIES_FILTER: %s)",
  nrow(records_all), n_distinct(records_all$station_id), n_distinct(records_all$campaign),
  if (is.null(SPECIES_FILTER)) "ALL" else paste(SPECIES_FILTER, collapse = ", ")))

n_no_effort <- sum(!records_all$valid_effort, na.rm = TRUE)
if (n_no_effort > 0) {
  message(sprintf(
    "  NOTE: %d records sit at stations with valid_effort == FALSE. Fine for presence and activity; out of every trap-night denominator.",
    n_no_effort))
}

place <- admissible(records_all, "place", quiet = TRUE)
timed <- admissible(records_all, "time",  quiet = TRUE)
message(sprintf(
  "  admissible for PLACE (presence/absence): %d records, %d stations\n  admissible for TIME  (activity/overlap) : %d records, %d stations",
  nrow(place), n_distinct(place$station_id), nrow(timed), n_distinct(timed$station_id)))
only_place <- setdiff(unique(place$station_id), unique(timed$station_id))
if (length(only_place)) {
  message(sprintf("  NOTE: station(s) in place-based analyses only (clock unrepairable): %s",
                  paste(sort(only_place), collapse = ", ")))
}
n_ep <- nrow(episodes(records_all, quiet = TRUE))
message(sprintf("  independent episodes (%s)             : %d  <- the unit for any COUNT; records are images",
                EPISODE_COLUMN, n_ep))
print(table(records_all$species_label, records_all$campaign))


# ── 8. camtrapR tables ───────────────────────────────────────────────────────
# RECORD TABLE: one row per EPISODE (the producer's rule), time-admissible only,
# because activityDensity() and activityOverlap() read the hour. time_rad is carried
# so 04_temporal_overlap.R's numeric and visual layers use the same rows.

record_table <- episodes(records_all, quiet = TRUE) %>%
  transmute(
    Station          = station_id,
    Species          = species_label,
    DateTimeOriginal = datetime,
    Date             = date,
    Time             = format(datetime, "%H:%M:%S"),
    time_rad         = time_rad,
    Campaign         = campaign
  )

coords <- st_coordinates(stations_sf)
stations_ct <- stations_sf %>%
  st_drop_geometry() %>%
  transmute(Station = id, Longitude = coords[, "X"], Latitude = coords[, "Y"], altitude_m)

message(sprintf("record_table: %d episodes (vs %d images in records_all).",
                nrow(record_table), nrow(records_all)))


# ── 9. Save, then stamp ──────────────────────────────────────────────────────
# The stamp is written LAST. If anything above fails, no stamp is written and the
# downstream scripts keep refusing, which is the correct state for a half-built data/.

saveRDS(records_all,  here("data", "records_all.rds"))
saveRDS(deployments,  here("data", "deployments.rds"))
saveRDS(stations_sf,  here("data", "stations_sf.rds"))
saveRDS(st_read(PATH_BOUNDARY, quiet = TRUE), here("data", "boundary_sf.rds"))
saveRDS(record_table, here("data", "record_table.rds"))
saveRDS(stations_ct,  here("data", "stations_ct.rds"))

contract_stamp_write(state, CAMPAIGNS)

message("\nSaved data/: records_all, deployments, stations_sf, boundary_sf, record_table, stations_ct, contract_stamp.json")
message("Run 02_detection_summary.R next.")
