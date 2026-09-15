# 06_seasonal_detection_maps.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   For each species with >= MIN_EPISODES independent episodes, produce a single
#   figure showing bubble detection maps for every season period the array has
#   recorded, in chronological order.
#
#   PERIODS, NOT POOLED SEASONS (2026-09-15). This script used to pool the two
#   otoños into one "Otoño" panel, which is the one thing a single-site study with
#   19 months of record can least afford to discard: whether a spatial pattern
#   REPEATS. Seven panels now, one per season period, ordered by date. The season
#   rule itself moved to R/00_seasons.R — it lived here, in this file's own
#   assign_season(), and a second copy would have been the first step toward two
#   figures disagreeing about what winter is. 05_spatial_distribution.R keeps the
#   pooled four-season view for cross-species comparison.
#
#   ADMISSIBILITY. A season needs a trustworthy date, so this script uses the time
#   rule from R/00_admissibility.R and nothing else. Until 2026-09-08 it also clipped
#   to a hard date window with tz = "America/Santiago": the window's stated purpose
#   (misconfigured 2017 clocks) is now handled upstream -- those rows arrive with
#   valid_date = FALSE -- and its end date silently cut otoño 2026 six weeks short.
#   The tz was a latent 3-4 h shift on any R with tzdata installed.
#
#   Bubble size is fixed on a shared scale across all panels of a figure so counts
#   are directly comparable within it.  Stations with zero detections in a period
#   appear as faint × marks.
#
#   THE PANELS ARE NOT EQUAL EFFORT and the figure says so: Primavera 2024 is 292
#   camera-days at 9 stations against 2,249 at 27 in Verano 2025-26.  A bubble is a
#   count, not a rate.  The header of this file used to read "Invierno — no field
#   deployment yet"; winter is in fact the best-sampled season in the record (96
#   episodes), it was simply recorded inside the primavera_2025 campaign window.
#
# INPUT   data/records_all.rds
#         data/deployments.rds    (the panel set comes from the field record)
#         data/stations_sf.rds
#         data/boundary_sf.rds
# OUTPUT  figures/06_seasonal_<species_slug>.png  (one file per species)
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(sf)
library(tidyr)

here::i_am("R/06_seasonal_detection_maps.R")
source(here::here("R", "00_contract.R"))
source(here::here("R", "00_admissibility.R"))
source(here::here("R", "00_seasons.R"))
dir.create(here("figures"), showWarnings = FALSE)

contract_assert_current()


# ── 1. Load data ─────────────────────────────────────────────────────────────

records     <- readRDS(here("data", "records_all.rds"))
deployments <- readRDS(here("data", "deployments.rds"))
stations_sf <- readRDS(here("data", "stations_sf.rds"))
boundary_sf <- readRDS(here("data", "boundary_sf.rds"))

# Species with fewer independent episodes than this get no seasonal figure: four
# panels of a handful of bubbles read as a pattern that is not there. Episodes, not
# images (2026-09-08; it was 30 images, which let a burst-heavy species qualify).
MIN_EPISODES <- 30L


# ── 2. Season periods ────────────────────────────────────────────────────────
# season_start is the join key and sorts chronologically; season_label renders it for
# the reader. Both come from R/00_seasons.R; neither is decided here. The panel set
# comes from the FIELD RECORD, not from the detections, so a period the array
# sampled and the species was never seen in still gets a panel of × marks — an
# absence is a result and must be visible.

records_clean <- admissible(records, "time") %>%
  mutate(season_start = season_start(datetime))

period_effort <- season_effort(deployments) %>%
  group_by(season_start) %>%
  summarise(camera_days = sum(effort_days),
            n_stations  = n_distinct(station_id), .groups = "drop") %>%
  arrange(season_start) %>%
  mutate(period = factor(as.character(season_label(season_start)),
                         levels = as.character(season_label(sort(season_start)))),
         strip  = sprintf("%s\n%s días-cámara · %d estaciones",
                          as.character(period), format(camera_days, big.mark = ","), n_stations))

PERIOD_LEVELS <- levels(period_effort$period)

records_clean <- records_clean %>%
  mutate(period = factor(as.character(season_label(season_start)), levels = PERIOD_LEVELS))


# ── 3. Species filter (>= MIN_EPISODES independent episodes) ─────────────────

qualifying <- episode_counts(records_clean, by = "species_label", quiet = TRUE) %>%
  rename(total = n_episodes) %>%
  filter(total >= MIN_EPISODES) %>%
  arrange(desc(total))

skipped <- episode_counts(records_clean, by = "species_label", quiet = TRUE) %>%
  filter(n_episodes < MIN_EPISODES)

message(sprintf(
  "Qualifying species (%d, >= %d episodes): %s",
  nrow(qualifying), MIN_EPISODES,
  paste(sprintf("%s (%d)", qualifying$species_label, qualifying$total), collapse = ", ")
))
if (nrow(skipped)) {
  message(sprintf("  Skipped (< %d episodes): %s", MIN_EPISODES,
                  paste(sprintf("%s (%d)", skipped$species_label, skipped$n_episodes), collapse = ", ")))
}


# ── 4. Shared visual constants ────────────────────────────────────────────────

SPECIES_COLORS <- c(
  "Puma"         = "#1b2a69",
  "Guina"        = "#2c7bb6",
  "Zorro culpeo" = "#74add1",
  "Jabali"       = "#d73027",
  "Liebre"       = "#fc8d59",
  "Perro"        = "#fee090"
)

map_theme <- theme_void(base_size = 11) +
  theme(
    legend.position   = "bottom",
    strip.background  = element_blank(),
    strip.text        = element_text(face = "bold", size = 12),
    plot.title        = element_text(face = "bold", size = 14),
    plot.subtitle     = element_text(size = 9, colour = "grey40"),
    plot.caption      = element_text(size = 8, colour = "grey50", hjust = 0),
    legend.title      = element_text(size = 9),
    legend.text       = element_text(size = 8),
    panel.spacing     = unit(0.8, "lines")
  )


# ── 5. Per-species figures ────────────────────────────────────────────────────

for (sp in qualifying$species_label) {

  sp_records <- records_clean %>% filter(species_label == sp)
  sp_color   <- SPECIES_COLORS[[sp]]
  sp_slug    <- tolower(gsub(" ", "_", sp))

  # Episodes per station × period. `sp_records` is already time-admissible, so
  # nothing further is excluded here.
  det_counts <- episode_counts(sp_records, by = c("station_id", "period"), quiet = TRUE) %>%
    rename(n_detections = n_episodes)

  # Full grid: every station × every period the array sampled (zeros elsewhere)
  full_grid <- expand.grid(
    station_id = stations_sf$id,
    period     = PERIOD_LEVELS,
    stringsAsFactors = FALSE
  ) %>%
    left_join(mutate(det_counts, period = as.character(period)),
              by = c("station_id", "period")) %>%
    mutate(
      n_detections = replace_na(n_detections, 0),
      period       = factor(period, levels = PERIOD_LEVELS)
    )

  # Attach sf geometry
  det_sf <- stations_sf %>%
    rename(station_id = id) %>%
    select(station_id, geometry) %>%
    right_join(full_grid, by = "station_id") %>%
    st_as_sf()

  # Max detections for fixed scale (computed across all seasons)
  max_det <- max(det_sf$n_detections)

  # Periods the array sampled and this species was never seen in. A named absence,
  # not a blank panel: every period here carries the effort printed in its strip.
  periods_with_data <- det_sf %>%
    st_drop_geometry() %>%
    filter(n_detections > 0) %>%
    pull(period) %>%
    unique() %>%
    as.character()

  periods_no_data <- setdiff(PERIOD_LEVELS, periods_with_data)
  caption_txt <- paste0(
    "Panel = periodo estacional, en orden cronológico; el esfuerzo de cada uno va en su título. ",
    "Una burbuja es un conteo, no una tasa.",
    if (length(periods_no_data) > 0) {
      paste0("\nSin registros en: ", paste(periods_no_data, collapse = ", "), ".")
    } else ""
  )

  fig <- ggplot() +
    geom_sf(data = boundary_sf,
            fill = "#f0f4e8", colour = "grey60", linewidth = 0.5) +
    geom_sf(data = stations_sf,
            colour = "grey70", size = 0.8, shape = 4) +
    geom_sf(
      data  = filter(det_sf, n_detections > 0),
      aes(size = n_detections),
      colour = sp_color,
      alpha  = 0.75
    ) +
    scale_size_continuous(
      range  = c(2, 12),
      limits = c(1, max_det),
      name   = "Detecciones"
    ) +
    facet_wrap(~period, nrow = 1,
               labeller = labeller(period = setNames(period_effort$strip,
                                                     as.character(period_effort$period)))) +
    labs(
      title    = sp,
      subtitle = sprintf("Eventos independientes (%d min) por periodo estacional.  × = estación sin detección.", EPISODE_GAP_MINUTES),
      caption  = caption_txt
    ) +
    map_theme

  out_path <- here("figures", sprintf("06_seasonal_%s.png", sp_slug))
  ggsave(out_path, fig, width = 20, height = 5.5, dpi = 300)
  message(sprintf("Saved %s", out_path))
}

message("Done. All seasonal detection maps saved to figures/.")
