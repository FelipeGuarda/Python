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
#   Two questions, two denominators, both decided by effort_admissible() in
#   R/00_admissibility.R off `media_status`:
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
#         figures/02_deployment_timeline.png   (Fig D — operation per station)
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
# carries media_status through; the two denominator rules are effort_admissible()'s
# (R/00_admissibility.R), unchanged from the campaign version. The split conserves
# days exactly (tests/test_seasons.R), so switching the axis moved no effort.

season_days <- season_effort(deployments)

effort <- season_days %>%
  mutate(for_rate     = effort_admissible(season_days, "detections"),
         for_sampling = effort_admissible(season_days, "sampling")) %>%
  group_by(season_start) %>%
  summarise(
    camera_days         = sum(effort_days[for_rate]),
    n_stations_sampling = n_distinct(station_id[for_sampling]),
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
  filter(effort_admissible(season_days, "detections")) %>%
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

# ── 7. Figure D — deployment timeline ────────────────────────────────────────
# One row per station, one bar per deployment, coloured by the same status that
# effort_admissible() reads. It shows in one picture why effort differs eightfold
# between periods (the array ramping up) and where the place-only records come from
# (whole deployments whose clock could not be repaired). Season-period boundaries
# are drawn so a gap can be read against the strata every other figure uses.

DEPLOY_STATUS_COLORS <- c(
  "Fotos, reloj válido"       = "#2c7bb6",
  "Fotos, reloj no reparable" = "#fdae61",
  "Sólo video"                = "grey60",
  "Tarjeta fallida"           = "#d7191c"
)

timeline <- deployments %>%
  mutate(
    status = case_when(
      media_status == "in_canonical" & valid_effort  ~ "Fotos, reloj válido",
      media_status == "in_canonical"                 ~ "Fotos, reloj no reparable",
      media_status == "video_only_offline"           ~ "Sólo video",
      media_status == "card_failure"                 ~ "Tarjeta fallida"
    ),
    status     = factor(status, levels = names(DEPLOY_STATUS_COLORS)),
    station_id = factor(station_id, levels = rev(sort(unique(station_id))))
  )
stopifnot(!anyNA(timeline$status))   # a new media_status must be named, not dropped

record_days    <- seq(min(timeline$field_start), max(timeline$field_end), by = "day")
period_starts  <- sort(unique(season_start(record_days)))
period_breaks  <- period_starts[period_starts > min(record_days)]
period_mids    <- data.frame(
  start = pmax(period_starts, min(record_days)),
  end   = c(period_breaks, max(record_days))
) %>%
  mutate(mid = start + (end - start) / 2, label = as.character(season_label(start)))

fig_D <- ggplot(timeline) +
  geom_vline(xintercept = period_breaks, colour = "grey75", linewidth = 0.3) +
  geom_segment(aes(x = field_start, xend = field_end, y = station_id, yend = station_id,
                   colour = status), linewidth = 3.2) +
  # A white tick at each field visit, so back-to-back deployments of the same
  # status read as two bars rather than one.
  geom_point(data = filter(timeline, field_start > min(field_start)),
             aes(x = field_start, y = station_id), shape = "|", size = 4, colour = "white") +
  geom_text(data = period_mids, aes(x = mid, y = Inf, label = label),
            vjust = -0.6, size = 3.2, colour = "grey30") +
  scale_colour_manual(values = DEPLOY_STATUS_COLORS, name = NULL, drop = FALSE) +
  scale_x_date(date_breaks = "2 months", date_labels = "%b\n%Y") +
  coord_cartesian(clip = "off") +
  labs(
    title    = "Operación de cada estación en el tiempo",
    subtitle = "Una barra por despliegue (entre dos visitas de terreno); líneas grises = inicio de cada temporada",
    caption  = paste("Fuente: deployments.csv del productor. \"Reloj no reparable\": las fotos prueban presencia",
                     "pero no fecha ni hora, y esos días no entran al denominador de las tasas.", sep = "\n"),
    x = NULL, y = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title      = element_text(margin = margin(b = 4)),
        plot.subtitle   = element_text(margin = margin(b = 22)),
        plot.caption    = element_text(hjust = 0, colour = "grey30"))

ggsave(here("figures", "02_deployment_timeline.png"), fig_D, width = 12, height = 8, dpi = 300)
message("Saved figures/02_deployment_timeline.png")

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
