# 07_seasonal_activity.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Answers one question: does a species' daily schedule change between the warm
#   and the cold half of the year? (Question iii, "variación entre temporadas".)
#
# THE TWO FRAMES ANSWER DIFFERENT QUESTIONS — READ THEM SEPARATELY
#   clock  At 39.4°S the night is ~5.6 h longer at the June solstice than at the
#          December one. A nocturnal species with a perfectly rigid schedule
#          relative to sunset therefore DOES look different between halves in clock
#          time. A clock-frame difference is mostly day length, not behaviour.
#   solar  R/00_timeofday.R pins sunrise and sunset to the same reference day for
#          every detection, so day length is removed from the time axis. A
#          difference that survives here is a change in schedule beyond what day
#          length explains: temperature, food, breeding.
#
# THE GROUPING IS THE PROJECT'S SEASON RULE, NOT A NEW ONE
#   warm = Primavera + Verano, cold = Otoño + Invierno, by season_of() from
#   R/00_seasons.R (calendar months today). If that rule moves to solstices and
#   equinoxes, this script follows without an edit. Years are pooled: per season
#   and year no species reaches ACTIVITY_MIN_EPISODES in both halves.
#
#   Halves, not the four seasons, because of sample size (2026-11-06, 415 episodes):
#     culpeo 35 / 54 / 69 / 27   -> halves 89 / 96
#     liebre 58 /  8 / 12 / 53   -> halves 66 / 65
#     perro  14 / 21 /  9 /  5   -> halves 35 / 14  (cold half below the floor)
#   Which species qualify is computed below, not hard-coded.
#
# TWO NUMBERS PER COMPARISON
#   Δ with its bootstrap CI (estimate_overlap(), R/00_activity.R): how similar the
#   two curves are. Monterroso categories are NOT applied: they were defined for
#   two species, not for one species against itself.
#   p (activity::compareCkern, Rowcliffe et al. 2014): a randomisation test of
#   whether both halves could come from one distribution. Its observed Δ4 uses the
#   activity package's own kernel and differs from ours in the third decimal; the
#   p-value is the only thing read from it.
#
# STATION COMPOSITION IS A CONFOUND, AND IT IS REPORTED
#   If a species is recorded at different stations in each half, a shift can be
#   place rather than season. The table carries the stations and the share of the
#   busiest station in each half, so a reader can see when that is the case.
#
# INPUT   data/record_table.rds
# OUTPUT  data/seasonal_activity_stats.csv
#         figures/07_seasonal_activity.png          (species × frame, warm vs cold)
#         figures/07_seasonal_activity_culpeo.png   (culpeo, four seasons, solar;
#                                                    descriptive only)
# ─────────────────────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
suppressPackageStartupMessages(library(activity))   # compareCkern(), fitact()

here::i_am("R/07_seasonal_activity.R")
source(here::here("R", "00_contract.R"))
source(here::here("R", "00_admissibility.R"))   # EPISODE_GAP_MINUTES
source(here::here("R", "00_seasons.R"))
source(here::here("R", "00_timeofday.R"))
source(here::here("R", "00_activity.R"))
dir.create(here("figures"), showWarnings = FALSE)

contract_assert_current()
pbapply::pboptions(type = "none")   # compareCkern() draws a progress bar per test

COMPARE_REPS <- 999

HALF_OF_SEASON <- c(Primavera = "warm", Verano = "warm",
                    "Otoño"   = "cold", Invierno = "cold")
HALF_LABELS <- c(warm = "Cálido (primavera–verano)", cold = "Frío (otoño–invierno)")
HALF_COLORS <- c(warm = "#d95f02", cold = "#1f78b4")
FRAME_LABELS <- c(clock = "Reloj de la cámara", solar = "Hora solar")


# ── 1. Episodes, each with its season and half ───────────────────────────────

record_table <- readRDS(here("data", "record_table.rds")) %>%
  mutate(season = season_of(DateTimeOriginal),
         half   = unname(HALF_OF_SEASON[as.character(season)]))
stopifnot(!anyNA(record_table$half))

half_n <- record_table %>% count(Species, half) %>%
  tidyr::pivot_wider(names_from = half, values_from = n, values_fill = 0)
eligible <- half_n %>%
  filter(warm >= ACTIVITY_MIN_EPISODES, cold >= ACTIVITY_MIN_EPISODES) %>%
  pull(Species)
skipped <- half_n %>% filter(!Species %in% eligible)

message(sprintf("Compared (>= %d episodes in each half): %s", ACTIVITY_MIN_EPISODES,
                paste(eligible, collapse = ", ")))
message(sprintf("Not compared: %s", paste(sprintf("%s (%d/%d)", skipped$Species,
                skipped$warm, skipped$cold), collapse = ", ")))


# ── 2. Δ, CI and randomisation test per species × frame ──────────────────────

station_mix <- function(stations) {
  tab <- sort(table(stations), decreasing = TRUE)
  list(n_stations = length(tab),
       top        = sprintf("%s (%.0f %%)", names(tab)[1], 100 * tab[[1]] / sum(tab)))
}

compare_halves <- function(sp, frame) {
  d <- record_table[record_table$Species == sp, ]
  t_warm <- d[[FRAME_COLUMN[[frame]]]][d$half == "warm"]
  t_cold <- d[[FRAME_COLUMN[[frame]]]][d$half == "cold"]

  set.seed(BOOT_SEED)   # per comparison, so no result depends on loop order
  fit <- estimate_overlap(t_warm, t_cold)
  set.seed(BOOT_SEED)
  test <- compareCkern(fitact(t_warm, sample = "none"),
                       fitact(t_cold, sample = "none"), reps = COMPARE_REPS)

  mw <- station_mix(d$Station[d$half == "warm"])
  mc <- station_mix(d$Station[d$half == "cold"])
  data.frame(
    species = sp, frame = frame,
    n_warm = length(t_warm), n_cold = length(t_cold),
    estimator = fit$estimator, overlap = fit$estimate,
    ci_low = fit$ci_low, ci_high = fit$ci_high,
    p_same_distribution = unname(test["pNull"]),
    stations_warm = mw$n_stations, top_station_warm = mw$top,
    stations_cold = mc$n_stations, top_station_cold = mc$top,
    stringsAsFactors = FALSE
  )
}

stats <- bind_rows(lapply(eligible, function(sp)
  bind_rows(lapply(names(FRAME_COLUMN), function(fr) compare_halves(sp, fr)))))

print(stats %>% select(species, frame, n_warm, n_cold, estimator,
                       overlap, ci_low, ci_high, p_same_distribution))

stats$station_mix_differs <- sub(" .*", "", stats$top_station_warm) !=
                             sub(" .*", "", stats$top_station_cold)
write.csv(stats, here("data", "seasonal_activity_stats.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
message("Saved data/seasonal_activity_stats.csv")


# ── 3. Figure: warm vs cold, species × frame ─────────────────────────────────

curves <- bind_rows(lapply(eligible, function(sp)
  bind_rows(lapply(names(FRAME_COLUMN), function(fr)
    bind_rows(lapply(names(HALF_LABELS), function(h) {
      sub <- record_table[record_table$half == h, ]
      activity_density(sub, sp, fr) %>% mutate(half = h)
    }))))))

# A species whose busiest station differs between halves gets a flag in its panel:
# its warm/cold difference may be place, not season (see the header).
panel_lbl <- stats %>%
  mutate(label = sprintf("Δ = %.2f [%.2f–%.2f]\np = %.3f%s",
                         overlap, ci_low, ci_high, p_same_distribution,
                         ifelse(station_mix_differs,
                                sprintf("\n¡Otras estaciones! %s vs %s",
                                        top_station_warm, top_station_cold), "")),
         species_n = sprintf("%s (%d / %d)", species, n_warm, n_cold))

curves <- curves %>%
  left_join(panel_lbl %>% select(species, frame, species_n),
            by = c("species_label" = "species", "frame")) %>%
  mutate(frame_lbl = factor(FRAME_LABELS[frame], levels = FRAME_LABELS))
panel_lbl <- panel_lbl %>%
  mutate(frame_lbl = factor(FRAME_LABELS[frame], levels = FRAME_LABELS))

sun_lines <- data.frame(frame_lbl = factor(FRAME_LABELS[["solar"]], levels = FRAME_LABELS),
                        x = SOLAR_MNANCHOR * 12 / pi)

fig <- ggplot(curves, aes(x = hour, y = density, colour = half)) +
  geom_vline(data = sun_lines, aes(xintercept = x), colour = "darkorange",
             linewidth = 0.5, linetype = "dashed") +
  geom_line(linewidth = 1.1) +
  geom_text(data = panel_lbl, aes(x = 12, y = Inf, label = label),
            inherit.aes = FALSE, vjust = 1.3, size = 3.2, colour = "grey20") +
  facet_grid(species_n ~ frame_lbl, scales = "free_y") +
  scale_colour_manual(values = HALF_COLORS, labels = HALF_LABELS, name = NULL) +
  scale_x_continuous(breaks = c(0, 6, 12, 18, 24)) +
  labs(
    title    = "¿Cambia el horario entre la mitad cálida y la fría del año?",
    subtitle = paste0("Reloj: la diferencia incluye el largo de la noche (5,6 h más en junio).\n",
                      "Hora solar: el largo del día está descontado; lo que queda es conducta."),
    caption  = sprintf(paste0(
      "Episodios independientes (regla de %d min). Δ = solapamiento (%s), IC 95 %% bootstrap; ",
      "p = prueba de aleatorización (compareCkern, %d repeticiones).\n",
      "Sin comparar (< %d episodios en una mitad): %s. Años agrupados. %s"),
      EPISODE_GAP_MINUTES, paste(unique(stats$estimator), collapse = "/"), COMPARE_REPS,
      ACTIVITY_MIN_EPISODES,
      paste(sprintf("%s (%d/%d)", skipped$Species, skipped$warm, skipped$cold), collapse = ", "),
      time_frame_label("solar")),
    x = "Hora", y = "Densidad de actividad"
  ) +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom",
        strip.background = element_blank(),
        strip.text.y = element_text(face = "italic"),
        strip.text.x = element_text(face = "bold"),
        plot.caption = element_text(hjust = 0, colour = "grey30", size = 8))

ggsave(here("figures", "07_seasonal_activity.png"), fig, width = 10, height = 7.6, dpi = 300)
message("Saved figures/07_seasonal_activity.png")


# ── 4. Figure: culpeo, four seasons, solar frame (descriptive) ───────────────
# The only species with >= 27 episodes in every season. No test: the winter curve
# sits just under the floor, so this is for looking, not for a claim.

culpeo <- record_table[record_table$Species == "Zorro culpeo", ]
season_n <- table(culpeo$season)
culpeo_curves <- bind_rows(lapply(SEASON_LEVELS, function(s)
  activity_density(culpeo[culpeo$season == s, ], "Zorro culpeo", "solar") %>%
    mutate(season = sprintf("%s (n = %d)", s, season_n[[s]]))))
culpeo_curves$season <- factor(culpeo_curves$season,
                               levels = sprintf("%s (n = %d)", SEASON_LEVELS, season_n[SEASON_LEVELS]))

fig_c <- ggplot(culpeo_curves, aes(x = hour, y = density, colour = season)) +
  geom_vline(xintercept = SOLAR_MNANCHOR * 12 / pi, colour = "darkorange",
             linewidth = 0.5, linetype = "dashed") +
  geom_line(linewidth = 1) +
  scale_colour_manual(values = c("#66a61e", "#d95f02", "#7570b3", "#1f78b4"), name = NULL) +
  scale_x_continuous(breaks = c(0, 6, 12, 18, 24)) +
  labs(title = "Zorro culpeo por estación del año, hora solar",
       subtitle = sprintf("Descriptivo: invierno queda bajo el mínimo de %d episodios. Años agrupados.",
                          ACTIVITY_MIN_EPISODES),
       caption = time_frame_label("solar"),
       x = "Hora", y = "Densidad de actividad") +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom",
        plot.caption = element_text(hjust = 0, colour = "grey30", size = 8))

ggsave(here("figures", "07_seasonal_activity_culpeo.png"), fig_c, width = 9, height = 5.5, dpi = 300)
message("Saved figures/07_seasonal_activity_culpeo.png")
