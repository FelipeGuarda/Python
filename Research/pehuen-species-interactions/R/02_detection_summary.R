# 02_detection_summary.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Compute basic detection metrics for the focal species and produce summary
#   bar charts:
#     Fig A — total detections per species, split by campaign
#     Fig B — Detection Rate (detections per 100 trap-nights) per species
#     Fig C — Naive occupancy (% of active stations where species was detected)
#
# INPUT   data/records_all.rds   (produced by 01_load_data.R)
# OUTPUT  figures/02_detections_per_species.png
#         figures/02_detection_rate.png
#         figures/02_naive_occupancy.png
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)  # combine multiple ggplot panels into one figure
library(scales)     # comma-formatted axis labels

here::i_am("R/02_detection_summary.R")
dir.create(here("figures"), showWarnings = FALSE)


# ── 1. Load data ─────────────────────────────────────────────────────────────

records <- readRDS(here("data", "records_all.rds"))

# Ordered species factor: native carnivores first, then invasive species.
# This order will be used consistently in all figures in this script.
SPECIES_ORDER <- c("Puma", "Guina", "Zorro culpeo", "Jabali", "Liebre", "Perro")
GUILD_COLORS  <- c("Native" = "#2c7bb6", "Invasive" = "#d7191c")

records <- records %>%
  mutate(species_label = factor(species_label, levels = SPECIES_ORDER))


# ── 2. Trap-night calculation ────────────────────────────────────────────────
# A trap-night = one camera active for one full day.
# We approximate it from the data: for each (campaign, station) pair, count
# the number of distinct calendar days that appear in the records.
# NOTE: This is a lower bound — cameras may have been active on days with no
# animal detections.  A more precise calculation would need deployment metadata
# (start/end dates per station), which is not available in the current CSV.
# This is flagged for improvement when proper deployment metadata is available.

trap_nights <- records %>%
  group_by(campaign, station_id) %>%
  summarise(n_days = n_distinct(date), .groups = "drop") %>%
  group_by(campaign) %>%
  summarise(trap_nights = sum(n_days), .groups = "drop")

message("Trap-nights per campaign:")
print(trap_nights)


# ── 3. Detections per species per campaign ────────────────────────────────────
# Count one detection per record row (each row = one observation event in the
# reviewed CSV).  Detections ≠ independent events — independence filtering is
# not applied here since the goal is a presence/activity overview.

detections <- records %>%
  count(campaign, species_label, guild, name = "n_detections") %>%
  # Add trap-nights to compute the rate
  left_join(trap_nights, by = "campaign")


# ── 4. Figure A — Raw detection counts ───────────────────────────────────────

fig_A <- detections %>%
  ggplot(aes(x = species_label, y = n_detections, fill = guild)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~campaign, ncol = 1, labeller = labeller(
    campaign = c(Otono_2025 = "Otoño 2025", Primavera_2025 = "Primavera 2025")
  )) +
  scale_fill_manual(values = GUILD_COLORS, name = "Guild") +
  scale_y_continuous(labels = comma) +
  labs(
    title    = "Total detections per focal species",
    subtitle = "Each row = one reviewed observation event",
    x        = NULL,
    y        = "Number of detections"
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(here("figures", "02_detections_per_species.png"),
       fig_A, width = 8, height = 7, dpi = 300)
message("Saved figures/02_detections_per_species.png")


# ── 5. Figure B — Detection Rate (per 100 trap-nights) ───────────────────────
# Detection Rate (DR) = (n_detections / trap_nights) * 100
# This normalises counts by camera effort, making campaigns comparable.

fig_B <- detections %>%
  mutate(detection_rate = n_detections / trap_nights * 100) %>%
  ggplot(aes(x = species_label, y = detection_rate, fill = guild)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~campaign, ncol = 1, labeller = labeller(
    campaign = c(Otono_2025 = "Otoño 2025", Primavera_2025 = "Primavera 2025")
  )) +
  scale_fill_manual(values = GUILD_COLORS, name = "Guild") +
  labs(
    title    = "Detection rate per focal species",
    subtitle = "Detections per 100 trap-nights (approximated from active days with records)",
    x        = NULL,
    y        = "Detections / 100 trap-nights"
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(here("figures", "02_detection_rate.png"),
       fig_B, width = 8, height = 7, dpi = 300)
message("Saved figures/02_detection_rate.png")


# ── 6. Figure C — Naive occupancy ────────────────────────────────────────────
# Naive occupancy = proportion of active stations where a species was detected
# at least once.  "Naive" because it does not account for imperfect detection.

# How many distinct stations were active per campaign?
n_stations_active <- records %>%
  group_by(campaign) %>%
  summarise(n_stations_total = n_distinct(station_id), .groups = "drop")

# How many of those stations had at least one detection of each focal species?
occupancy <- records %>%
  group_by(campaign, species_label, guild) %>%
  summarise(n_stations_detected = n_distinct(station_id), .groups = "drop") %>%
  left_join(n_stations_active, by = "campaign") %>%
  mutate(
    naive_occupancy = n_stations_detected / n_stations_total,
    species_label   = factor(species_label, levels = SPECIES_ORDER)
  )

fig_C <- occupancy %>%
  ggplot(aes(x = species_label, y = naive_occupancy, fill = guild)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~campaign, ncol = 1, labeller = labeller(
    campaign = c(Otono_2025 = "Otoño 2025", Primavera_2025 = "Primavera 2025")
  )) +
  scale_fill_manual(values = GUILD_COLORS, name = "Guild") +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  labs(
    title    = "Naive occupancy per focal species",
    subtitle = "Proportion of active stations with at least one detection",
    x        = NULL,
    y        = "Naive occupancy"
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(here("figures", "02_naive_occupancy.png"),
       fig_C, width = 8, height = 7, dpi = 300)
message("Saved figures/02_naive_occupancy.png")
message("Run 03_activity_patterns.R next.")
