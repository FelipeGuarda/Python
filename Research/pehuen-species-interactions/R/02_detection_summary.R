# 02_detection_summary.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Basic detection metrics for the focal species, per SEASON PERIOD:
#     Fig A — independent episodes per species
#     Fig B — detection rate: episodes per 100 camera-days
#     Fig C — naive occupancy: share of sampling stations where the species was seen
#
# WHY SEASON AND NOT CAMPAIGN (2026-09-15)
#   These three figures faceted by campaign until now, and a campaign is a five- to
#   eight-month interval between field visits that mixes seasons — see the window
#   table in R/00_seasons.R. A rate "per campaign" therefore averaged over summer and
#   autumn and called the result autumn. Campaign survives as provenance: it is still
#   the unit the producer publishes, it still decides which station-days are
#   admissible, and it is still carried in every intermediate table. It is no longer
#   an axis anything is plotted against.
#
#   The seven periods are not equally sampled and the figures say so in the strip
#   labels. Primavera 2024 is 292 camera-days at 9 stations — the array ramping up —
#   against 2,249 days at 27 stations in Verano 2025-26. A rate from 292 days is a
#   real number with wide uncertainty, not a comparable one.
#
# UNITS AND DENOMINATORS (R/00_admissibility.R, R/00_seasons.R, deployments.rds)
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
#                detections are just not readable here.
#
#   ONE THING THE SEASONAL SPLIT COSTS. Naive occupancy per campaign used presence(),
#   which is place-admissible: a frame with a broken clock still proves the animal was
#   there. A season cannot be read off a broken clock, so seasonal occupancy is built
#   from episodes() instead and is TIME-admissible. Fig C therefore drops five
#   station-species presences the campaign version kept, measured 2026-09-15: CT08
#   guiña, CT10 jabalí, CT13 liebre, CT18 perro and CT18 puma. The denominator still
#   counts every sampling station, so the ratio is conservative in the honest
#   direction. Stated on the figure. 05_spatial_distribution.R keeps the place rule
#   for its pooled maps, which is where those five records still appear.
#
# INPUT   data/records_all.rds, data/deployments.rds   (01_load_data.R)
# OUTPUT  figures/02_detections_per_species.png
#         figures/02_detection_rate.png
#         figures/02_naive_occupancy.png
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries and the handshake ───────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(scales)

here::i_am("R/02_detection_summary.R")
source(here::here("R", "00_contract.R"))
source(here::here("R", "00_admissibility.R"))
source(here::here("R", "00_seasons.R"))
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


# ── 2. Effort, from the field record, split across seasons ───────────────────
# season_effort() divides each deployment window among the periods it spans and
# carries media_status through; the two denominator rules below are this script's,
# unchanged from the campaign version. The split conserves days exactly
# (tests/test_seasons.R), so switching the axis moved no effort.

season_days <- season_effort(deployments)

effort <- season_days %>%
  group_by(season_start) %>%
  summarise(
    camera_days         = sum(effort_days[media_status == "in_canonical" & valid_effort %in% TRUE]),
    n_stations_sampling = n_distinct(station_id[media_status %in% c("in_canonical", "video_only_offline")]),
    .groups = "drop"
  ) %>%
  mutate(season = season_label(season_start))

message("Effort per season (camera-days with stills and a valid clock; stations sampling):")
print(as.data.frame(effort), row.names = FALSE)

# Strip labels carry the effort behind each panel. A rate read without its
# denominator is the error this whole re-stratification exists to stop, and the
# seven periods differ by a factor of eight.
season_strip <- setNames(
  sprintf("%s\n%s camera-days · %d estaciones",
          as.character(effort$season), format(effort$camera_days, big.mark = ","),
          effort$n_stations_sampling),
  as.character(effort$season)
)
season_facets <- facet_wrap(~season, ncol = 4,
                            labeller = labeller(season = season_strip))


# ── 3. Episodes per species per season ───────────────────────────────────────
# A record's season comes from its own timestamp; its admissibility comes from the
# deployment it sits in. Both are needed, so the join is on all three keys.

rate_units <- season_days %>%
  filter(media_status == "in_canonical", valid_effort %in% TRUE) %>%
  select(campaign, station_id, season_start)

episodes_seasoned <- records %>%
  episodes() %>%
  mutate(season_start = season_start(datetime))

# An episode whose timestamp falls outside its own deployment's window would be a
# clock repair that left the field record behind. Counted rather than dropped in
# silence: it belongs in no denominator and would quietly shrink every rate.
outside <- anti_join(episodes_seasoned, rate_units,
                     by = c("campaign", "station_id", "season_start"))
if (nrow(outside)) {
  message(sprintf(
    "  NOTE: %d of %d episodes sit outside a rate-admissible station-season and are out of Fig A/B: %s.",
    nrow(outside), nrow(episodes_seasoned),
    paste(sort(unique(paste0(outside$station_id, "/", outside$campaign))), collapse = ", ")))
}

detections <- episodes_seasoned %>%
  semi_join(rate_units, by = c("campaign", "station_id", "season_start")) %>%
  count(season_start, species_label, guild, name = "n_episodes") %>%
  left_join(effort, by = "season_start") %>%
  mutate(rate_per_100 = n_episodes / camera_days * 100)


# ── 4. Figure A — episodes ───────────────────────────────────────────────────

fig_A <- ggplot(detections, aes(x = species_label, y = n_episodes, fill = guild)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  season_facets +
  scale_fill_manual(values = GUILD_COLORS, name = "Guild") +
  scale_y_continuous(labels = comma) +
  labs(
    title    = "Independent detections per focal species",
    subtitle = sprintf("Unit: episodes (%d-min rule, decided at ingest), not images", EPISODE_GAP_MINUTES),
    x = NULL, y = "Episodes"
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(here("figures", "02_detections_per_species.png"), fig_A, width = 14, height = 8, dpi = 300)
message("Saved figures/02_detections_per_species.png")


# ── 5. Figure B — detection rate per 100 camera-days ─────────────────────────

fig_B <- ggplot(detections, aes(x = species_label, y = rate_per_100, fill = guild)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  season_facets +
  scale_fill_manual(values = GUILD_COLORS, name = "Guild") +
  labs(
    title    = "Detection rate per focal species",
    subtitle = "Episodes per 100 camera-days; effort from the field record (deployments.csv)",
    caption  = "Denominator: stations with stills in the canonical table and a valid clock diagnosis.",
    x = NULL, y = "Episodes / 100 camera-days"
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "bottom", plot.caption = element_text(hjust = 0, colour = "grey30"))

ggsave(here("figures", "02_detection_rate.png"), fig_B, width = 14, height = 8, dpi = 300)
message("Saved figures/02_detection_rate.png")


# ── 6. Figure C — naive occupancy ────────────────────────────────────────────
# Time-admissible, unlike the campaign version this replaced: a season cannot be read
# off a broken clock, so this is presence-among-episodes rather than presence(). The
# denominator still counts every station that was sampling, clock-failed ones
# included, so the ratio understates rather than overstates. See the header.

occupancy <- episodes_seasoned %>%
  distinct(season_start, station_id, species_label, guild) %>%
  count(season_start, species_label, guild, name = "n_stations_detected") %>%
  left_join(effort, by = "season_start") %>%
  mutate(naive_occupancy = n_stations_detected / n_stations_sampling,
         species_label   = factor(species_label, levels = SPECIES_ORDER))

fig_C <- ggplot(occupancy, aes(x = species_label, y = naive_occupancy, fill = guild)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  season_facets +
  scale_fill_manual(values = GUILD_COLORS, name = "Guild") +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  labs(
    title    = "Naive occupancy per focal species",
    subtitle = "Share of sampling stations with at least one detection, by season",
    caption  = paste("Denominator: stations sampling in the season — stills or offline video in the field record.",
                     "Numerator is time-admissible only: a season needs a working clock, so records with a failed",
                     "clock prove presence but not when. Not corrected for imperfect detection.", sep = "\n"),
    x = NULL, y = "Naive occupancy"
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "bottom", plot.caption = element_text(hjust = 0, colour = "grey30"))

ggsave(here("figures", "02_naive_occupancy.png"), fig_C, width = 14, height = 8, dpi = 300)
message("Saved figures/02_naive_occupancy.png")

message("\nEpisodes / rate / occupancy per season and species:")
summary_tbl <- detections %>%
  select(season_start, season, species_label, n_episodes, camera_days, rate_per_100) %>%
  left_join(occupancy %>% select(season_start, species_label, n_stations_detected,
                                 n_stations_sampling, naive_occupancy),
            by = c("season_start", "species_label")) %>%
  arrange(season_start, species_label) %>%
  select(-season_start) %>%
  mutate(rate_per_100 = round(rate_per_100, 2), naive_occupancy = round(naive_occupancy, 2))
print(as.data.frame(summary_tbl), row.names = FALSE)
message("Run 03_activity_patterns.R next.")
