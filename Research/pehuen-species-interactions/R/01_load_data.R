# 01_load_data.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Read the reviewed observation CSVs from both camera-trap campaigns,
#   standardize station identifiers, join with GPS coordinates from the GeoJSON,
#   filter to the focal species, and save clean R data objects for all
#   downstream analysis scripts.
#
# INPUT FILES
#   - Otoño 2025 reviewed CSV   (on Synology drive — update PATH_OTONO below)
#   - Primavera 2025 reviewed CSV
#   - camera_trap_stations.geojson
#
# OUTPUT FILES  (written to data/ inside the project)
#   - records_all.rds      all focal-species records, both campaigns combined
#   - stations_sf.rds      spatial dataframe with camera locations
#
# HOW TO RE-RUN FOR A NEW CAMPAIGN
#   1. Add the new CSV path in the "Paths" section below.
#   2. Add a new read + standardise block (copy the Otoño or Primavera block).
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

PATH_PRIMAVERA <- "C:/Users/USUARIO/Dev/Python/camera-traps/data/campaigns/primavera_2025/new_labeled_data_reviewed.csv"

PATH_GEOJSON <- "C:/Users/USUARIO/Dev/Python/plataforma-territorial/data/camera_trap_stations.geojson"

PATH_BOUNDARY <- "C:/Users/USUARIO/Dev/Python/plataforma-territorial/data/boundary.geojson"

# Output directory
dir.create(here("data"), showWarnings = FALSE)


# ── 2. Focal species ──────────────────────────────────────────────────────────
# These are the six species the analysis focuses on.
# Keys are the Latin names as stored in the `scientificName` CSV column.
# Values are human-readable labels used in all figures.

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

  raw <- read_csv(path, show_col_types = FALSE)

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
    # (b) Parse the timestamp column.
    #     The field contains strings like " 2025-01-29 18:48:38" (note possible
    #     leading whitespace from the Timelapse2 export).
    mutate(
      datetime = ymd_hms(str_trim(timestamp), tz = "America/Santiago")
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


# ── 6. Read Primavera 2025 ────────────────────────────────────────────────────

message("Reading Primavera 2025...")
primavera_raw <- read_campaign_csv(PATH_PRIMAVERA, campaign_label = "Primavera_2025")

# STATION ID PARSING — Primavera
# Format: "TC{n}_{sd_card}.2"  e.g. "TC1_M7.2", "TC10_M3.2"
# There is also one anomalous entry "100EK113" with no matching station.
#
# Step 1: Drop the anomalous entry that cannot be matched.
# Step 2: Extract the station integer — the digits between "TC" and "_".
#         Regex: ^TC(\d+)_  captures the number after TC up to the underscore.
# Step 3: Extract the SD card code — the text between "_" and ".2".
#         This will be used in Step 7 to validate the join.

primavera_raw <- primavera_raw %>%
  # Step 1: Drop anomalous entry (no matching camera in GeoJSON).
  filter(station_raw != "100EK113") %>%
  mutate(
    # Step 2: Parse tc_num from the station label.
    tc_num    = as.integer(str_match(station_raw, "^TC(\\d+)_")[, 2]),
    # Step 3: Parse the SD card code embedded in the label (e.g. "M7" from "TC1_M7.2").
    sd_parsed = str_match(station_raw, "_(M[^.]+)\\.")[, 2]
  )


# ── 7. Validate Primavera station IDs against the GeoJSON ─────────────────────
# For each unique Primavera station, check that:
#   (a) tc_num parsed from the label matches a real station in the GeoJSON, AND
#   (b) the SD card code embedded in the label matches the GeoJSON `sd_card` field.
#
# This is a double-check: the tc_num alone is the join key, but the SD card gives
# us independent confirmation that the mapping is correct.
# A mismatch here would mean the station label encodes a tc_num that points to
# the wrong physical camera.

stations_lookup <- st_drop_geometry(stations_sf) %>%
  select(tc_num, id, sd_card)

validation <- primavera_raw %>%
  distinct(station_raw, tc_num, sd_parsed) %>%
  left_join(stations_lookup, by = "tc_num") %>%
  mutate(sd_match = (sd_parsed == sd_card))

mismatches <- filter(validation, !sd_match | is.na(sd_match))

if (nrow(mismatches) > 0) {
  stop(
    "Station ID validation FAILED for Primavera. The following stations have ",
    "an SD card code in their label that does not match the GeoJSON:\n",
    paste(capture.output(print(mismatches)), collapse = "\n"),
    "\nDo not proceed until this is resolved."
  )
} else {
  message(sprintf(
    "  Primavera validation passed: all %d station IDs confirmed by both tc_num and sd_card.",
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

otono     <- join_to_stations(otono_raw,     stations_lookup_full)
primavera <- join_to_stations(primavera_raw, stations_lookup_full)

# Verify: no unmatched tc_nums (would produce NA station_id)
for (df_name in c("otono", "primavera")) {
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


# ── 9. Combine campaigns and filter to focal species ─────────────────────────

records_all <- bind_rows(otono, primavera) %>%
  # Keep only the six focal species
  filter(species_latin %in% names(FOCAL_SPECIES)) %>%
  # Add the human-readable species label
  mutate(
    species_label = FOCAL_SPECIES[species_latin],
    guild = case_when(
      species_latin %in% NATIVE_SPECIES   ~ "Native",
      species_latin %in% INVASIVE_SPECIES ~ "Invasive"
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
  "\nFinal dataset: %d focal-species records across %d stations and %d campaigns.",
  nrow(records_all),
  n_distinct(records_all$station_id),
  n_distinct(records_all$campaign)
))
print(table(records_all$species_label, records_all$campaign))


# ── 10. Save outputs ──────────────────────────────────────────────────────────

saveRDS(records_all,  here("data", "records_all.rds"))
saveRDS(stations_sf,  here("data", "stations_sf.rds"))
saveRDS(st_read(PATH_BOUNDARY, quiet = TRUE), here("data", "boundary_sf.rds"))

message("\nSaved: data/records_all.rds, data/stations_sf.rds, data/boundary_sf.rds")
message("Run 02_detection_summary.R next.")
