# 05_spatial_distribution.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Map the spatial distribution of detections for each focal species across
#   the camera-trap grid.  Bubble size = total detections at each station.
#
#   Two outputs:
#     Fig A — One map per species (faceted), both campaigns combined
#     Fig B — Side-by-side campaign comparison for native carnivores
#
# INPUT   data/records_all.rds   (produced by 01_load_data.R)
#         data/stations_sf.rds
#         data/boundary_sf.rds
# OUTPUT  figures/05_spatial_all_species.png
#         figures/05_spatial_native_by_campaign.png
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(patchwork)
library(sf)

here::i_am("R/05_spatial_distribution.R")
dir.create(here("figures"), showWarnings = FALSE)


# ── 1. Load data ─────────────────────────────────────────────────────────────

records     <- readRDS(here("data", "records_all.rds"))
stations_sf <- readRDS(here("data", "stations_sf.rds"))
boundary_sf <- readRDS(here("data", "boundary_sf.rds"))

SPECIES_ORDER <- c("Puma", "Guina", "Zorro culpeo", "Jabali", "Liebre", "Perro")
NATIVE_LABELS <- c("Puma", "Guina", "Zorro culpeo")

SPECIES_COLORS <- c(
  "Puma"         = "#1b2a69",
  "Guina"        = "#2c7bb6",
  "Zorro culpeo" = "#74add1",
  "Jabali"       = "#d73027",
  "Liebre"       = "#fc8d59",
  "Perro"        = "#fee090"
)


# ── 2. Aggregate detections per (station × species) ──────────────────────────
# Join record counts to the spatial station layer.
# Stations with zero detections of a given species are filled with 0 so that
# all cameras appear on every species map (not just the ones where it was seen).

# Step 2a: count detections per station per species
det_by_station <- records %>%
  count(station_id, species_label, name = "n_detections")

# Step 2b: create the full grid of (station × species) combinations
all_combinations <- expand.grid(
  station_id    = unique(stations_sf$id),
  species_label = SPECIES_ORDER,
  stringsAsFactors = FALSE
)

# Step 2c: left-join counts into the full grid so every station appears
det_full <- all_combinations %>%
  left_join(det_by_station, by = c("station_id", "species_label")) %>%
  mutate(n_detections = replace(n_detections, is.na(n_detections), 0))

# Step 2d: attach geometry by joining on station_id → GeoJSON `id`
det_sf <- stations_sf %>%
  rename(station_id = id) %>%
  right_join(det_full, by = "station_id") %>%
  mutate(species_label = factor(species_label, levels = SPECIES_ORDER))


# ── 3. Shared map theme ───────────────────────────────────────────────────────

map_theme <- theme_void(base_size = 12) +
  theme(
    legend.position  = "bottom",
    strip.background = element_blank(),
    strip.text       = element_text(face = "bold", size = 11),
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(size = 10, colour = "grey40")
  )


# ── 4. Figure A — all six species, both campaigns combined ───────────────────
# One facet per species.  Bubble size proportional to total detections.
# Stations with zero detections are shown as tiny grey dots.

fig_all <- ggplot() +
  # Reserve boundary as background polygon
  geom_sf(data = boundary_sf, fill = "#f0f4e8", colour = "grey60", linewidth = 0.5) +
  # All camera stations as a faint reference grid
  geom_sf(data = stations_sf, colour = "grey70", size = 0.8, shape = 4) +
  # Detection bubbles (size = n_detections; 0 shown as a tiny point)
  geom_sf(
    data = filter(det_sf, n_detections > 0),
    aes(size = n_detections, colour = species_label),
    alpha = 0.8
  ) +
  scale_size_continuous(range = c(2, 12), name = "Detections") +
  scale_colour_manual(values = SPECIES_COLORS, guide = "none") +
  facet_wrap(~species_label, ncol = 3) +
  labs(
    title    = "Spatial distribution of detections — focal species",
    subtitle = "Both campaigns combined. X marks = all camera stations."
  ) +
  map_theme

ggsave(here("figures", "05_spatial_all_species.png"),
       fig_all, width = 12, height = 8, dpi = 300)
message("Saved figures/05_spatial_all_species.png")


# ── 5. Figure B — native carnivores, split by campaign ───────────────────────
# Allows visual comparison of whether spatial patterns shift between seasons.

# Aggregate per station × species × campaign
det_by_campaign <- records %>%
  filter(species_label %in% NATIVE_LABELS) %>%
  count(station_id, species_label, campaign, name = "n_detections")

# Full grid for native species × campaign combinations
native_combinations <- expand.grid(
  station_id    = unique(stations_sf$id),
  species_label = NATIVE_LABELS,
  campaign      = c("Otono_2025", "Primavera_2025"),
  stringsAsFactors = FALSE
)

det_native_sf <- native_combinations %>%
  left_join(det_by_campaign,
            by = c("station_id", "species_label", "campaign")) %>%
  mutate(n_detections = replace(n_detections, is.na(n_detections), 0)) %>%
  left_join(
    stations_sf %>% rename(station_id = id) %>% select(station_id, geometry),
    by = "station_id"
  ) %>%
  st_as_sf() %>%
  mutate(
    species_label = factor(species_label, levels = NATIVE_LABELS),
    campaign_lbl  = ifelse(campaign == "Otono_2025", "Otoño 2025", "Primavera 2025")
  )

fig_native <- ggplot() +
  geom_sf(data = boundary_sf, fill = "#f0f4e8", colour = "grey60", linewidth = 0.5) +
  geom_sf(data = stations_sf, colour = "grey70", size = 0.8, shape = 4) +
  geom_sf(
    data    = filter(det_native_sf, n_detections > 0),
    aes(size = n_detections, colour = species_label),
    alpha   = 0.8
  ) +
  scale_size_continuous(range = c(2, 12), name = "Detections") +
  scale_colour_manual(values = SPECIES_COLORS[NATIVE_LABELS], name = NULL) +
  facet_grid(campaign_lbl ~ species_label) +
  labs(
    title    = "Spatial distribution — native carnivores by campaign",
    subtitle = "Bubble size = total detections. X marks = all camera stations."
  ) +
  map_theme

ggsave(here("figures", "05_spatial_native_by_campaign.png"),
       fig_native, width = 12, height = 8, dpi = 300)
message("Saved figures/05_spatial_native_by_campaign.png")
message("All scripts complete. Figures are in the figures/ directory.")
