# 05_spatial_distribution.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Map the spatial distribution of detections across the camera-trap grid.
#
#   Two complementary approaches:
#     A) Presence/absence maps via camtrapR::detectionMaps() — one map per
#        species showing which stations detected it, plus a species richness
#        map.  Saved to figures/detection_maps/.
#
#     B) Episode-count bubble maps via sf + ggplot2 — camtrapR's
#        detectionMaps() shows presence/absence but cannot scale bubble size
#        by count.  For that we use ggplot2 directly.
#
# TWO UNITS, TWO ADMISSIBILITY RULES (see R/00_admissibility.R)
#   Panel A asks WHERE a species was seen, so it uses presence() over every
#   identified record — including records whose camera clock failed, because the
#   station is known with certainty even when the timestamp is not. Before
#   2026-08-20 this panel inherited a time filter from 01_load_data.R and showed
#   puma at 6 stations when the data holds 8 (CT03 and CT18 were lost).
#
#   Panel B asks HOW OFTEN, so it uses episodes() — the 30-minute independent-event
#   rule — and therefore only records with a trustworthy clock. It previously counted
#   IMAGES, which is a burst-length artefact: a camera fires 2-3 frames per trigger,
#   so Bos taurus reads 579 images against 19 episodes.
#
#   The two panels therefore legitimately disagree about which stations appear. That
#   is not an inconsistency to reconcile — it is the difference between "was it here"
#   and "how often", and the captions say so.
#        Fig B1 — all six species, both campaigns combined (faceted)
#        Fig B2 — native carnivores, split by campaign (grid: campaign × species)
#
# INPUT   data/records_all.rds   (produced by 01_load_data.R)
#         data/record_table.rds  (camtrapR format)
#         data/stations_sf.rds   (sf spatial dataframe)
#         data/stations_ct.rds   (camtrapR CTtable format)
#         data/boundary_sf.rds   (reserve boundary polygon)
# OUTPUT  figures/detection_maps/  (camtrapR presence/absence maps)
#         figures/05_spatial_all_species.png
#         figures/05_spatial_native_by_campaign.png
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(sf)
library(camtrapR)   # detectionMaps() — richness panel only

here::i_am("R/05_spatial_distribution.R")
dir.create(here("figures"), showWarnings = FALSE)
dir.create(here("figures", "detection_maps"), showWarnings = FALSE)


# ── 1. Load data ─────────────────────────────────────────────────────────────

source(here::here("R", "00_admissibility.R"))

records      <- readRDS(here("data", "records_all.rds"))
record_table <- readRDS(here("data", "record_table.rds"))
stations_sf  <- readRDS(here("data", "stations_sf.rds"))
stations_ct  <- readRDS(here("data", "stations_ct.rds"))
boundary_sf  <- readRDS(here("data", "boundary_sf.rds"))

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


# ── 2. Presence / absence maps — camtrapR::detectionMaps() ───────────────────
# detectionMaps() creates one map per species showing which camera stations
# detected it (filled symbol) vs. did not detect it (open symbol).  It also
# produces a species richness map (total number of species detected per station).
#
# Key arguments:
#   CTtable           — camera trap station table with Station, Longitude, Latitude
#   recordTable       — camtrapR-format record table
#   Xcol / Ycol       — column names for longitude and latitude in CTtable
#   stationCol        — column name for station ID (must match in both tables)
#   speciesCol        — column name for species (must match in both tables)
#   writePNG = TRUE   — save PNGs to disk
#   plotR    = FALSE  — do not open interactive graphics windows
#   plotDirectory     — destination folder for PNGs
#   richnessPlot      — also produce a species richness map (TRUE by default)

# camtrapR::detectionMaps() cannot be used for the presence panel: its recordTable
# requires DateTimeOriginal, so it structurally cannot represent a record whose clock
# failed — the very records this panel must include. We build presence directly.
message("Generating presence/absence maps (all place-admissible records)...")

pres <- presence(records)

pres_grid <- expand.grid(
  station_id    = unique(stations_sf$id),
  species_label = SPECIES_ORDER,
  stringsAsFactors = FALSE
) %>%
  mutate(detected = paste(station_id, species_label) %in%
                    paste(pres$station_id, pres$species_label))

pres_sf <- stations_sf %>%
  rename(station_id = id) %>%
  right_join(pres_grid, by = "station_id") %>%
  mutate(species_label = factor(species_label, levels = SPECIES_ORDER))

dir.create(here("figures", "detection_maps"), showWarnings = FALSE, recursive = TRUE)

fig_presence <- ggplot() +
  geom_sf(data = boundary_sf, fill = "grey97", colour = "grey60", linewidth = 0.3) +
  geom_sf(data = filter(pres_sf, !detected), shape = 1, size = 1.8,
          colour = "grey55", stroke = 0.4) +
  geom_sf(data = filter(pres_sf, detected), shape = 19, size = 2.6,
          colour = "#1b5e20") +
  facet_wrap(~ species_label) +
  labs(
    title    = "Presencia por estacion",
    subtitle = paste0("Circulo lleno = detectada. Incluye registros sin reloj ",
                      "utilizable: la estacion es conocida aunque la hora no lo sea."),
    caption  = "Unidad: presencia (conjunto), no frecuencia. Ver figura de burbujas para frecuencia."
  ) +
  theme_void(base_size = 9) +
  theme(legend.position = "none")

ggsave(here("figures", "detection_maps", "presence_by_species.png"),
       fig_presence, width = 9, height = 6.5, dpi = 200)

# Richness stays on camtrapR, which is fine: it counts species per station and the
# record_table it reads is time-admissible. Noted as a known narrowing rather than
# silently different from the panel above.
detectionMaps(
  CTtable      = stations_ct,
  recordTable  = record_table,
  Xcol         = "Longitude",
  Ycol         = "Latitude",
  stationCol   = "Station",
  speciesCol   = "Species",
  writePNG     = TRUE,
  plotR        = FALSE,
  plotDirectory = here("figures", "detection_maps"),
  richnessPlot = TRUE
)

message(sprintf(
  "  presence: %d station-species pairs across %d stations (vs %d stations in the time-admissible set)",
  nrow(pres), n_distinct(pres$station_id),
  n_distinct(admissible(records, "time", quiet = TRUE)$station_id)))


# ── 3. Aggregate detections per (station × species) for bubble maps ───────────
# For the count-based bubble maps we need the full station × species grid,
# including zeros (stations where a species was not detected).
# We build this in three steps:
#   (a) Count detections per station per species from the records.
#   (b) Create the full grid of all (station × species) combinations.
#   (c) Left-join counts into the grid; missing combinations get n_detections = 0.
#   (d) Attach sf geometry by joining on station_id → GeoJSON id.

# (a) EPISODE counts, not image counts. A camera fires 2-3 frames per trigger, so
#     counting images ranks stations partly by how long an animal lingered in frame.
det_by_station <- episode_counts(records, by = c("station_id", "species_label")) %>%
  rename(n_detections = n_episodes)

# (b) Full grid
all_combinations <- expand.grid(
  station_id    = unique(stations_sf$id),
  species_label = SPECIES_ORDER,
  stringsAsFactors = FALSE
)

# (c) Join counts; zero-fill missing combinations
det_full <- all_combinations %>%
  left_join(det_by_station, by = c("station_id", "species_label")) %>%
  mutate(n_detections = replace(n_detections, is.na(n_detections), 0))

# (d) Attach geometry
det_sf <- stations_sf %>%
  rename(station_id = id) %>%
  right_join(det_full, by = "station_id") %>%
  mutate(species_label = factor(species_label, levels = SPECIES_ORDER))


# ── 4. Shared map theme ───────────────────────────────────────────────────────

map_theme <- theme_void(base_size = 12) +
  theme(
    legend.position  = "bottom",
    strip.background = element_blank(),
    strip.text       = element_text(face = "bold", size = 11),
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(size = 10, colour = "grey40")
  )


# ── 5. Figure B1 — all six species, both campaigns combined ──────────────────
# One facet per species.  Bubble size is proportional to total detections.
# All camera stations appear on every facet so the grid is always visible.

fig_all <- ggplot() +
  # Reserve boundary as background polygon
  geom_sf(data = boundary_sf,
          fill = "#f0f4e8", colour = "grey60", linewidth = 0.5) +
  # All camera stations as a faint reference grid (X marks)
  geom_sf(data = stations_sf,
          colour = "grey70", size = 0.8, shape = 4) +
  # Bubbles only where n_detections > 0
  geom_sf(
    data  = filter(det_sf, n_detections > 0),
    aes(size = n_detections, colour = species_label),
    alpha = 0.8
  ) +
  scale_size_continuous(range = c(2, 12), name = "Detections") +
  scale_colour_manual(values = SPECIES_COLORS, guide = "none") +
  facet_wrap(~species_label, ncol = 3) +
  labs(
    title    = "Spatial distribution of detections — focal species",
    subtitle = "Both campaigns combined.  X marks = all camera stations.",
    caption  = "Unidad: eventos independientes (30 min), no imagenes. Excluye camaras sin reloj reparable, por lo que puede mostrar MENOS estaciones que el mapa de presencia -- ver figures/detection_maps/presence_by_species.png."
  ) +
  map_theme

ggsave(here("figures", "05_spatial_all_species.png"),
       fig_all, width = 12, height = 8, dpi = 300)
message("Saved figures/05_spatial_all_species.png")


# ── 6. Figure B2 — native carnivores split by campaign ───────────────────────
# Allows visual comparison of whether spatial detection patterns shift between
# Otoño 2025 and Primavera 2025.

# Aggregate per station × species × campaign
det_by_campaign <- records %>%
  filter(species_label %in% NATIVE_LABELS) %>%
  count(station_id, species_label, campaign, name = "n_detections")

# Full grid for native species × both campaigns
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
    campaign_lbl  = ifelse(campaign == "Otono_2025", "Oto\u00f1o 2025", "Primavera 2025")
  )

fig_native <- ggplot() +
  geom_sf(data = boundary_sf,
          fill = "#f0f4e8", colour = "grey60", linewidth = 0.5) +
  geom_sf(data = stations_sf,
          colour = "grey70", size = 0.8, shape = 4) +
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
    subtitle = "Bubble size = independent events (30-min rule).  X marks = all camera stations.",
    caption  = "Unidad: eventos independientes (30 min), no imagenes. Excluye camaras sin reloj reparable, por lo que puede mostrar MENOS estaciones que el mapa de presencia -- ver figures/detection_maps/presence_by_species.png."
  ) +
  map_theme

ggsave(here("figures", "05_spatial_native_by_campaign.png"),
       fig_native, width = 12, height = 8, dpi = 300)
message("Saved figures/05_spatial_native_by_campaign.png")
message("All scripts complete. Figures are in the figures/ directory.")
