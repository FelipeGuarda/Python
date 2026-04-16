# 01_load_data.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Read the reviewed observation CSVs from both camera-trap campaigns,
#   standardize station identifiers, join with GPS coordinates from the GeoJSON,
#   optionally filter to a species subset, and save clean R data objects for all
#   downstream analysis scripts.
#
# INPUT FILES
#   - Otoño 2025 reviewed CSV            (Synology drive — update PATH_OTONO)
#   - Primavera-verano 2025-2026 CSV     (camera-traps repo — update PATH_PV)
#   - camera_trap_stations.geojson
#
# OUTPUT FILES  (written to data/ inside the project)
#   - records_all.rds      records for the active SPECIES_FILTER, both campaigns
#   - stations_sf.rds      spatial dataframe with camera locations
#
# SPECIES FILTER
#   Set SPECIES_FILTER to a character vector of Latin names to restrict the
#   output to those species (e.g. for the pehuen research analysis).
#   Set to NULL to retain ALL identified species (for plataforma / full dataset).
#
# HOW TO RE-RUN FOR A NEW CAMPAIGN
#   1. Add the new CSV path in the "Paths" section below.
#   2. Add a new read + standardise block (copy the Otoño or PV block).
#   3. Add the new dataframe to the bind_rows() call at the bottom.
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)        # reproducible relative paths (auto-detects project root)
library(readr)       # fast CSV reading with consistent type inference
library(dplyr)       # data manipulation
library(stringr)     # string parsing for station ID extraction
library(lubridate)   # datetime parsing
library(sf)          # reading GeoJSON and spatial operations

# Announce to `here` where the project root is relative to this script.
# This writes a tiny `.here` file in the project root on first run.
here::i_am("R/01_load_data.R")


# ── 1. Paths — update these when adding campaigns ────────────────────────────

PATH_OTONO <- "C:/Users/USUARIO/SynologyDrive/2. Camaras trampa (SC)/SynologyDrive/DATOS_GRILLA CÁMARAS TRAMPA/2. CAMPAÑAS DE RECOLECCION DE IMAGENES/Otoño 2025/Fotos/new_labeled_data_reviewed.csv"

PATH_PV <- "C:/Users/USUARIO/Dev/Python/camera-traps/data/campaigns/pv_2025_2026/new_labeled_data_reviewed.csv"

PATH_GEOJSON <- "C:/Users/USUARIO/Dev/Python/plataforma-territorial/data/camera_trap_stations.geojson"

PATH_BOUNDARY <- "C:/Users/USUARIO/Dev/Python/plataforma-territorial/data/boundary.geojson"

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


# ── 3. Load and parse the station coordinates (GeoJSON) ───────────────────────
# The GeoJSON holds the canonical information for each physical camera trap:
#   id       → canonical station label "TC-01", "TC-02", …
#   tc       → integer station number (1–26); this is the JOIN KEY
#   sd_card  → SD card code used to cross-validate Primavera station names
#   geometry → WGS-84 point coordinates (lon, lat)

stations_sf <- st_read(PATH_GEOJSON, quiet = TRUE) %>%
  # Rename `tc` to `tc_num` to make its role as join key explicit
  rename(tc_num = tc) %>%
  # Keep only the columns we need downstream
  select(id, tc_num, sd_card, altitude_m, geometry)

message(sprintf("Loaded %d camera stations from GeoJSON.", nrow(stations_sf)))


# ── 4. Helper function: read and minimally clean one campaign CSV ─────────────
# Both campaign CSVs share identical column names (verified), so we use one
# function for both.  This function:
#   a) reads the CSV
#   b) keeps only animal observations that have an identified species
#   c) parses the timestamp into a proper datetime object
#   d) selects and renames only the columns needed for analysis

read_campaign_csv <- function(path, campaign_label) {

  # Force datetime-like columns to character so we control parsing below.
  raw <- read_csv(path, show_col_types = FALSE,
                  col_types = cols(timestamp = col_character(),
                                   DateTime  = col_character()))

  clean <- raw %>%
    # (a) Keep only rows where a real animal species was identified.
    #     observationType == "animal" excludes setup/blank images.
    #     scientificName not empty or "No reconocible" excludes unidentified obs.
    filter(
      observationType == "animal",
      !is.na(scientificName),
      scientificName != "",
      scientificName != "No reconocible"
    ) %>%
    # (b) Parse the datetime.
    #     Older campaigns populate `timestamp`; newer campaigns (pv_2025_2026)
    #     leave it empty and put the value in `DateTime` instead.
    #     We coalesce the two, falling back to DateTime when timestamp is blank.
    #     We do NOT specify tz here: camera clocks are set to Chile local time,
    #     so the numeric hour values are already correct for activity analysis
    #     on any platform without timezone database lookups.
    mutate(
      datetime_str = coalesce(
        na_if(str_trim(as.character(timestamp)), ""),
        na_if(str_trim(as.character(DateTime)),  "")
      ),
      datetime = ymd_hms(datetime_str, quiet = TRUE)
    ) %>%
    # (c) Add a campaign tag for grouping in multi-season analyses.
    mutate(campaign = campaign_label) %>%
    # (d) Select and rename to the columns used throughout the analysis.
    select(
      campaign,
      station_raw  = Deployments,   # original station ID string (kept for traceability)
      datetime,
      species_latin = scientificName,
      count,
      review_outcome = reviewOutcome
    )

  message(sprintf(
    "  [%s] %d animal records read; %d after filtering to identified species.",
    campaign_label, nrow(raw), nrow(clean)
  ))

  clean
}


# ── 5. Read Otoño 2025 ────────────────────────────────────────────────────────

message("Reading Otono 2025...")
otono_raw <- read_campaign_csv(PATH_OTONO, campaign_label = "Otono_2025")

# STATION ID PARSING — Otoño
# Format: "CT01", "CT02", ..., "CT20"
# Rule: strip the literal prefix "CT" and parse the remaining digits as integer.
# Example: "CT07" → 7 → matches tc_num = 7 → GeoJSON station TC-07.

otono_raw <- otono_raw %>%
  mutate(
    tc_num = as.integer(str_replace(station_raw, "^CT", ""))
  )


# ── 6. Read Primavera-verano 2025-2026 ───────────────────────────────────────

message("Reading Primavera-verano 2025-2026...")
pv_raw <- read_campaign_csv(PATH_PV, campaign_label = "PrimaveraVerano_2025_2026")

# STATION ID PARSING — Primavera-verano 2025-2026
# Format: "TC{n}_{sd_card}.2"  e.g. "TC1_M7.2", "TC10_M3.2"
# (Same format as the previous Primavera 2025 campaign; no anomalous entries.)
#
# Step 1: Extract the station integer — the digits between "TC" and "_".
#         Regex: ^TC(\d+)_  captures the number after TC up to the underscore.
# Step 2: Extract the SD card code — the text between "_" and ".2".
#         This will be used in Step 7 to validate the join.

pv_raw <- pv_raw %>%
  mutate(
    # Step 1: Parse tc_num from the station label.
    tc_num    = as.integer(str_match(station_raw, "^TC(\\d+)_")[, 2]),
    # Step 2: Parse the SD card code embedded in the label (e.g. "M3" from "TC10_M3.2").
    sd_parsed = str_match(station_raw, "_(M[^.]+)\\.")[, 2]
  )


# ── 7. Validate Primavera-verano station IDs against the GeoJSON ──────────────
# For each unique station, check that:
#   (a) tc_num parsed from the label matches a real station in the GeoJSON, AND
#   (b) the SD card code embedded in the label matches the GeoJSON `sd_card` field.
#
# This is a double-check: the tc_num alone is the join key, but the SD card gives
# us independent confirmation that the mapping is correct.

stations_lookup <- st_drop_geometry(stations_sf) %>%
  select(tc_num, id, sd_card)

validation <- pv_raw %>%
  distinct(station_raw, tc_num, sd_parsed) %>%
  left_join(stations_lookup, by = "tc_num") %>%
  mutate(sd_match = (sd_parsed == sd_card))

mismatches <- filter(validation, !sd_match | is.na(sd_match))

if (nrow(mismatches) > 0) {
  stop(
    "Station ID validation FAILED for Primavera-verano. The following stations have ",
    "an SD card code in their label that does not match the GeoJSON:\n",
    paste(capture.output(print(mismatches)), collapse = "\n"),
    "\nDo not proceed until this is resolved."
  )
} else {
  message(sprintf(
    "  PV validation passed: all %d station IDs confirmed by both tc_num and sd_card.",
    nrow(validation)
  ))
}


# ── 8. Join both campaigns to GeoJSON coordinates ────────────────────────────
# Now that tc_num is trusted in both datasets, we join to the GeoJSON to get:
#   - station_id: the canonical "TC-XX" identifier used in all figures
#   - altitude_m
#
# We use a left_join (not inner_join) so that any tc_num without a GeoJSON match
# shows up as NA rather than silently disappearing. We check for NAs afterwards.

join_to_stations <- function(df, stations_lookup) {
  df %>%
    left_join(stations_lookup, by = "tc_num") %>%
    rename(station_id = id)
}

stations_lookup_full <- st_drop_geometry(stations_sf) %>%
  select(tc_num, id, altitude_m)

otono <- join_to_stations(otono_raw, stations_lookup_full)
pv    <- join_to_stations(pv_raw,   stations_lookup_full)

# Verify: no unmatched tc_nums (would produce NA station_id)
for (df_name in c("otono", "pv")) {
  df <- get(df_name)
  unmatched <- filter(df, is.na(station_id))
  if (nrow(unmatched) > 0) {
    warning(sprintf(
      "%d records in %s have tc_num values not found in the GeoJSON: %s",
      nrow(unmatched), df_name,
      paste(unique(unmatched$tc_num), collapse = ", ")
    ))
  }
}


# ── 9. Combine campaigns and filter to target species ────────────────────────
# When SPECIES_FILTER is a character vector, only those Latin names are kept.
# When SPECIES_FILTER is NULL, all identified species are retained.

records_all <- bind_rows(otono, pv) %>%
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
    # Derive date and time-of-day fields used in activity analyses
    date      = as.Date(datetime),
    hour      = hour(datetime),
    # Time of day as a fraction of 24 hours, then converted to radians (0 to 2π)
    # This is the format expected by the `overlap` package
    time_rad  = (hour(datetime) * 3600 + minute(datetime) * 60 + second(datetime)) /
                86400 * 2 * pi
  ) %>%
  # Drop rows where datetime parsing failed (would produce NA)
  filter(!is.na(datetime))

message(sprintf(
  "\nFinal dataset: %d records across %d stations and %d campaigns. (SPECIES_FILTER: %s)",
  nrow(records_all),
  n_distinct(records_all$station_id),
  n_distinct(records_all$campaign),
  if (is.null(SPECIES_FILTER)) "ALL" else paste(SPECIES_FILTER, collapse = ", ")
))
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

record_table <- records_all %>%
  transmute(
    Station          = station_id,
    Species          = species_label,
    DateTimeOriginal = datetime,
    Date             = date,
    Time             = format(datetime, "%H:%M:%S"),
    Campaign         = campaign
  )

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
