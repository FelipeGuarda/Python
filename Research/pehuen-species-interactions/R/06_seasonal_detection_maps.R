# 06_seasonal_detection_maps.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   For each species with >= MIN_EPISODES independent episodes, produce a single
#   figure showing bubble detection maps for all four Southern-Hemisphere seasons
#   side by side.
#
#   Seasons (Southern Hemisphere):
#     Primavera  Sep–Nov
#     Verano     Dec–Feb
#     Otoño      Mar–May
#     Invierno   Jun–Aug
#
#   ADMISSIBILITY. A season needs a trustworthy date, so this script uses the time
#   rule from R/00_admissibility.R and nothing else. Until 2026-09-08 it also clipped
#   to a hard date window with tz = "America/Santiago": the window's stated purpose
#   (misconfigured 2017 clocks) is now handled upstream -- those rows arrive with
#   valid_date = FALSE -- and its end date silently cut otoño 2026 six weeks short.
#   The tz was a latent 3-4 h shift on any R with tzdata installed.
#
#   Bubble size is fixed on a shared scale across all four season panels so
#   counts are directly comparable within a figure.  Stations with zero
#   detections in a season appear as faint × marks.  Seasons with no data
#   (e.g. Invierno — no field deployment yet) show only the station grid.
#
# INPUT   data/records_all.rds
#         data/stations_sf.rds
#         data/boundary_sf.rds
# OUTPUT  figures/06_seasonal_<species_slug>.png  (one file per species)
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(sf)
library(lubridate)
library(tidyr)

here::i_am("R/06_seasonal_detection_maps.R")
source(here::here("R", "00_contract.R"))
source(here::here("R", "00_admissibility.R"))
dir.create(here("figures"), showWarnings = FALSE)

contract_assert_current()


# ── 1. Load data ─────────────────────────────────────────────────────────────

records     <- readRDS(here("data", "records_all.rds"))
stations_sf <- readRDS(here("data", "stations_sf.rds"))
boundary_sf <- readRDS(here("data", "boundary_sf.rds"))

# Species with fewer independent episodes than this get no seasonal figure: four
# panels of a handful of bubbles read as a pattern that is not there. Episodes, not
# images (2026-09-08; it was 30 images, which let a burst-heavy species qualify).
MIN_EPISODES <- 30L


# ── 2. Season assignment ─────────────────────────────────────────────────────

assign_season <- function(month) {
  dplyr::case_when(
    month %in% c(9, 10, 11) ~ "Primavera",
    month %in% c(12, 1, 2)  ~ "Verano",
    month %in% c(3, 4, 5)   ~ "Otoño",
    month %in% c(6, 7, 8)   ~ "Invierno"
  )
}

SEASON_LEVELS <- c("Primavera", "Verano", "Otoño", "Invierno")

records_clean <- admissible(records, "time") %>%
  mutate(season = factor(assign_season(month(datetime)), levels = SEASON_LEVELS))


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

  # Episodes per station × season. `sp_records` is already time-admissible, so
  # nothing further is excluded here.
  det_counts <- episode_counts(sp_records, by = c("station_id", "season"), quiet = TRUE) %>%
    rename(n_detections = n_episodes)

  # Full grid: every station × every season (zeros for missing combos)
  full_grid <- expand.grid(
    station_id = stations_sf$id,
    season     = factor(SEASON_LEVELS, levels = SEASON_LEVELS),
    stringsAsFactors = FALSE
  ) %>%
    left_join(det_counts, by = c("station_id", "season")) %>%
    mutate(
      n_detections = replace_na(n_detections, 0),
      season       = factor(season, levels = SEASON_LEVELS)
    )

  # Attach sf geometry
  det_sf <- stations_sf %>%
    rename(station_id = id) %>%
    select(station_id, geometry) %>%
    right_join(full_grid, by = "station_id") %>%
    st_as_sf()

  # Max detections for fixed scale (computed across all seasons)
  max_det <- max(det_sf$n_detections)

  # Seasons present in data (for caption)
  seasons_with_data <- det_sf %>%
    st_drop_geometry() %>%
    filter(n_detections > 0) %>%
    pull(season) %>%
    unique() %>%
    as.character() %>%
    sort()

  seasons_no_data <- setdiff(SEASON_LEVELS, seasons_with_data)
  caption_txt <- if (length(seasons_no_data) > 0) {
    paste0("Sin registros en: ", paste(seasons_no_data, collapse = ", "), ".")
  } else {
    NULL
  }

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
    facet_wrap(~season, nrow = 1) +
    labs(
      title    = sp,
      subtitle = sprintf("Eventos independientes (%d min) por temporada.  × = estación sin detección.", EPISODE_GAP_MINUTES),
      caption  = caption_txt
    ) +
    map_theme

  out_path <- here("figures", sprintf("06_seasonal_%s.png", sp_slug))
  ggsave(out_path, fig, width = 16, height = 5, dpi = 300)
  message(sprintf("Saved %s", out_path))
}

message("Done. All seasonal detection maps saved to figures/.")
