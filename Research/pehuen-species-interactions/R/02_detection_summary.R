# 02_detection_summary.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Basic detection metrics for the focal species, per campaign:
#     Fig A — independent episodes per species
#     Fig B — detection rate: episodes per 100 camera-days
#     Fig C — naive occupancy: share of sampling stations where the species was seen
#
# UNITS AND DENOMINATORS (R/00_admissibility.R, and deployments.rds from 01)
#   Counts are EPISODES, never images. Effort comes from the producer's
#   deployments.csv (field record), not from "days with a photo", which was a lower
#   bound that this script used until 2026-09-08 and which rewarded busy cameras.
#
#   Two questions, two denominators, both read off `media_status`:
#     rate       divides stills-based episodes by camera-days of stations whose stills
#                are in the canonical table AND whose clock diagnosis allows effort
#                (valid_effort). A station without a usable clock has no episodes, so
#                putting its days under the line would deflate every rate.
#     occupancy  divides stations-with-presence by stations that were SAMPLING:
#                in_canonical plus video_only_offline. The latter were recording; their
#                detections are just not readable here. Presence needs a station, not
#                a clock, so clock-failed stations stay on both sides of that ratio.
#
# INPUT   data/records_all.rds, data/deployments.rds   (01_load_data.R)
# OUTPUT  figures/02_detections_per_species.png
#         figures/02_detection_rate.png
#         figures/02_naive_occupancy.png
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries and the handshake ───────────────────────────────────────────

library(here)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

here::i_am("R/02_detection_summary.R")
source(here::here("R", "00_contract.R"))
source(here::here("R", "00_admissibility.R"))
dir.create(here("figures"), showWarnings = FALSE)

stamp     <- contract_assert_current()
CAMPAIGNS <- names(stamp$campaigns)


# ── 1. Load ──────────────────────────────────────────────────────────────────

SPECIES_ORDER <- c("Puma", "Guina", "Zorro culpeo", "Jabali", "Liebre", "Perro")
GUILD_COLORS  <- c("Native" = "#2c7bb6", "Invasive" = "#d7191c")

records <- readRDS(here("data", "records_all.rds")) %>%
  mutate(species_label = factor(species_label, levels = SPECIES_ORDER),
         campaign      = factor(campaign, levels = CAMPAIGNS))

deployments <- readRDS(here("data", "deployments.rds")) %>%
  mutate(campaign = factor(campaign, levels = CAMPAIGNS))

campaign_facets <- facet_wrap(~campaign, ncol = 1,
                              labeller = labeller(campaign = campaign_label))


# ── 2. Effort, from the field record ─────────────────────────────────────────

effort <- deployments %>%
  group_by(campaign) %>%
  summarise(
    camera_days        = sum(field_days[media_status == "in_canonical" & valid_effort %in% TRUE]),
    n_stations_sampling = n_distinct(station_id[media_status %in% c("in_canonical", "video_only_offline")]),
    .groups = "drop"
  )

message("Effort per campaign (camera-days with stills and a valid clock; stations sampling):")
print(effort)


# ── 3. Episodes per species per campaign ─────────────────────────────────────

rate_stations <- deployments %>%
  filter(media_status == "in_canonical", valid_effort %in% TRUE) %>%
  select(campaign, station_id)

detections <- records %>%
  semi_join(rate_stations, by = c("campaign", "station_id")) %>%
  episode_counts(by = c("campaign", "species_label", "guild")) %>%
  left_join(effort, by = "campaign") %>%
  mutate(rate_per_100 = n_episodes / camera_days * 100)


# ── 4. Figure A — episodes ───────────────────────────────────────────────────

fig_A <- ggplot(detections, aes(x = species_label, y = n_episodes, fill = guild)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  campaign_facets +
  scale_fill_manual(values = GUILD_COLORS, name = "Guild") +
  scale_y_continuous(labels = comma) +
  labs(
    title    = "Independent detections per focal species",
    subtitle = sprintf("Unit: episodes (%d-min rule, decided at ingest), not images", EPISODE_GAP_MINUTES),
    x = NULL, y = "Episodes"
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(here("figures", "02_detections_per_species.png"), fig_A, width = 8, height = 9, dpi = 300)
message("Saved figures/02_detections_per_species.png")


# ── 5. Figure B — detection rate per 100 camera-days ─────────────────────────

fig_B <- ggplot(detections, aes(x = species_label, y = rate_per_100, fill = guild)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  campaign_facets +
  scale_fill_manual(values = GUILD_COLORS, name = "Guild") +
  labs(
    title    = "Detection rate per focal species",
    subtitle = "Episodes per 100 camera-days; effort from the field record (deployments.csv)",
    caption  = "Denominator: stations with stills in the canonical table and a valid clock diagnosis.",
    x = NULL, y = "Episodes / 100 camera-days"
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "bottom", plot.caption = element_text(hjust = 0, colour = "grey30"))

ggsave(here("figures", "02_detection_rate.png"), fig_B, width = 8, height = 9, dpi = 300)
message("Saved figures/02_detection_rate.png")


# ── 6. Figure C — naive occupancy ────────────────────────────────────────────
# presence() is place-admissible: a station whose clock failed still counts as
# occupied, and it is also in the denominator because it was sampling.

occupancy <- presence(records) %>%
  count(campaign, species_label, guild, name = "n_stations_detected") %>%
  mutate(campaign = factor(campaign, levels = CAMPAIGNS)) %>%
  left_join(effort, by = "campaign") %>%
  mutate(naive_occupancy = n_stations_detected / n_stations_sampling,
         species_label   = factor(species_label, levels = SPECIES_ORDER))

fig_C <- ggplot(occupancy, aes(x = species_label, y = naive_occupancy, fill = guild)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  campaign_facets +
  scale_fill_manual(values = GUILD_COLORS, name = "Guild") +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  labs(
    title    = "Naive occupancy per focal species",
    subtitle = "Share of sampling stations with at least one detection",
    caption  = "Denominator: stations with stills or offline video in the campaign (field record). Not corrected for imperfect detection.",
    x = NULL, y = "Naive occupancy"
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "bottom", plot.caption = element_text(hjust = 0, colour = "grey30"))

ggsave(here("figures", "02_naive_occupancy.png"), fig_C, width = 8, height = 9, dpi = 300)
message("Saved figures/02_naive_occupancy.png")

message("\nEpisodes / rate / occupancy per campaign and species:")
summary_tbl <- detections %>%
  select(campaign, species_label, n_episodes, camera_days, rate_per_100) %>%
  left_join(occupancy %>% select(campaign, species_label, n_stations_detected,
                                 n_stations_sampling, naive_occupancy),
            by = c("campaign", "species_label")) %>%
  arrange(campaign, species_label) %>%
  mutate(rate_per_100 = round(rate_per_100, 2), naive_occupancy = round(naive_occupancy, 2))
print(as.data.frame(summary_tbl), row.names = FALSE)
message("Run 03_activity_patterns.R next.")
