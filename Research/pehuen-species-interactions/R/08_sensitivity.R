# 08_sensitivity.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Answers one question: do the main temporal results survive reasonable
#   alternative analysis choices? Each alternative is a VARIANT of the episode
#   table; every variant goes through the same overlap estimator as the published
#   run, and the output says which results move and which do not.
#
#   Variants today:
#     base            the published table (record_table.rds, 30-min rule)
#     busiest_two     without CT03 and CT09, the two stations with a third of all
#                     episodes (culpeo concentrates at CT03, liebre at CT09). A LEVERAGE
#                     check: it asks whether two sites carry the result, not whether
#                     they are biased.
#     no_rangers      without CT09 and CT19, the two stations near the park rangers'
#                     homes. A BIAS check: both sit at a similar distance from the
#                     houses, so both are treated as exposed to domestic animals and
#                     people, or neither would be (Felipe, 2026-11-06).
#     gap_15, gap_60  independence threshold of 15 and 60 minutes
#   Plus leave-one-station-out over all 27 registry stations: point estimates only
#   (overlap_point()), so the station that moves each pair most is named rather than
#   chosen in advance.
#   Not done: season boundaries at solstices/equinoxes are out of scope for this
#   research (decision 2026-11-12); calendar-month boundaries are final.
#
# WHY THE THRESHOLD VARIANTS ARE COMPUTED HERE, NOT BY THE PRODUCER (2026-11-06)
#   camera-traps owns the episode rule (camtrap/episodes.py) and publishes exactly
#   one: `episode_30min`. That is the definition behind every count anyone
#   publishes, and it stays the only one. A sensitivity threshold is not a second
#   definition. It perturbs the existing one, and its output is a robustness
#   statement, never a count in a figure. Asking the producer for
#   `episode_15min` / `episode_60min` would make the producer's schema carry this
#   study's research choices. Felipe's decision: sensitivity variants live in the
#   consumer that needs them.
#
#   The rule moved upstream in the first place because three consumer copies had
#   drifted apart without anyone noticing (a 33 % undercount). This copy is
#   guarded against that in three ways:
#     1. It lives only in this file. It is not in R/00_admissibility.R, so no
#        other script can pick it up, and nothing it builds is saved as a table
#        or used for a published count.
#     2. It checks itself on every run: at 30 minutes it must reproduce
#        `episode_30min` exactly, the same rows grouped into the same episodes.
#        If it does not, the script stops.
#     3. Known limit: the producer also forbids an episode to continue across a
#        clock reset, and the reset points are not in the published table. At
#        30 minutes that makes no difference (check 2 proves it on this data).
#        At 60 minutes it could merge two episodes across a reset, so gap_60
#        could undercount by at most the 30 → 60 difference itself.
#
# INPUT   data/records_all.rds, data/record_table.rds, data/deployments.rds,
#         data/overlap_stats.csv (only for the species pairs and the base check)
# OUTPUT  data/sensitivity_overlap.csv     (variant × frame × pair)
#         data/sensitivity_episodes.csv    (gap × species counts; liebre autumn rates)
#         data/sensitivity_loo_station.csv (station left out × frame × pair, point only)
#         figures/08_sensitivity_overlap.png
#         figures/08_sensitivity_loo_station.png
# ─────────────────────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)

here::i_am("R/08_sensitivity.R")
source(here::here("R", "00_contract.R"))
source(here::here("R", "00_admissibility.R"))
source(here::here("R", "00_seasons.R"))
source(here::here("R", "00_activity.R"))
dir.create(here("figures"), showWarnings = FALSE)

contract_assert_current()

BUSIEST_STATIONS <- c("CT03", "CT09")   # a third of all episodes (leverage)
RANGER_STATIONS  <- c("CT09", "CT19")   # near the park rangers' homes (bias)
SENSITIVITY_GAPS  <- c(15L, 60L)

records_all  <- readRDS(here("data", "records_all.rds"))
record_table <- readRDS(here("data", "record_table.rds"))
deployments  <- readRDS(here("data", "deployments.rds"))
published    <- read.csv(here("data", "overlap_stats.csv"), fileEncoding = "UTF-8")
PAIRS        <- unique(published[, c("sp1", "sp2")])


# ── 1. Episodes at another threshold (sensitivity only; see header) ──────────
# Gap measured from the last RETAINED detection, per campaign × station × species:
# the producer's rule minus the clock-segment constraint. Returns an episode id per
# row of the time-admissible records.

episode_ids_at_gap <- function(r, gap_minutes) {
  ord <- order(r$campaign, r$station_id, r$species_label, r$datetime)
  key <- paste(r$campaign, r$station_id, r$species_label)[ord]
  t   <- as.numeric(r$datetime)[ord]
  id  <- integer(length(t)); n <- 0L; anchor <- NA_real_
  for (i in seq_along(t)) {
    if (i == 1L || key[i] != key[i - 1L] || t[i] - anchor >= gap_minutes * 60) {
      n <- n + 1L; anchor <- t[i]
    }
    id[i] <- n
  }
  out <- integer(length(t)); out[ord] <- id
  out
}

# The first frame of each episode, in record_table's shape (the camtrapR convention
# that 01_load_data.R uses: the episode sits where its first trigger sits).
as_record_table <- function(r, ids) {
  r$.id <- ids
  r <- r[order(r$.id, r$datetime), ]
  r <- r[!duplicated(r$.id), ]
  data.frame(Station = r$station_id, Species = as.character(r$species_label),
             DateTimeOriginal = r$datetime, time_rad = r$time_rad,
             time_solar_rad = r$time_solar_rad, campaign = r$campaign,
             stringsAsFactors = FALSE)
}

timed <- admissible(records_all, "time", quiet = TRUE)

# Guard 2: at the canonical threshold the recomputation must BE the canonical rule.
ids_30 <- episode_ids_at_gap(timed, EPISODE_GAP_MINUTES)
canon  <- timed[[EPISODE_COLUMN]]
same_partition <- nrow(unique(data.frame(ids_30, canon))) == length(unique(ids_30)) &&
                  length(unique(ids_30)) == length(unique(canon))
if (!same_partition) {
  stop(sprintf(paste0("episode_ids_at_gap(%d) does not reproduce %s (%d vs %d episodes). ",
                      "The sensitivity copy has drifted from the producer's rule; ",
                      "fix it before reading any threshold variant."),
               EPISODE_GAP_MINUTES, EPISODE_COLUMN, length(unique(ids_30)), length(unique(canon))))
}
message(sprintf("Guard: the %d-min recomputation reproduces %s exactly (%d episodes).",
                EPISODE_GAP_MINUTES, EPISODE_COLUMN, length(unique(canon))))


# ── 2. Variants ──────────────────────────────────────────────────────────────

VARIANTS <- c(
  list(base        = record_table,
       busiest_two = record_table[!record_table$Station %in% BUSIEST_STATIONS, ],
       no_rangers  = record_table[!record_table$Station %in% RANGER_STATIONS, ]),
  setNames(lapply(SENSITIVITY_GAPS, function(g)
             as_record_table(timed, episode_ids_at_gap(timed, g))),
           paste0("gap_", SENSITIVITY_GAPS))
)
VARIANT_LABELS <- c(base = "Publicado (30 min)",
                    busiest_two = "Sin las 2 más cargadas (CT03, CT09)",
                    no_rangers = "Sin las de guardaparques (CT09, CT19)",
                    gap_15 = "Umbral 15 min", gap_60 = "Umbral 60 min")


# ── 3. Overlap per variant × frame × pair ────────────────────────────────────
# Seeded per variant × pair, so a variant's CI never depends on which variants ran
# before it. `base` is recomputed under the same seeding instead of read from
# overlap_stats.csv, so a difference between variants is data, not Monte Carlo.

overlap_rows <- bind_rows(lapply(names(VARIANTS), function(v) {
  rt <- VARIANTS[[v]]
  bind_rows(lapply(names(FRAME_COLUMN), function(fr) {
    times <- split(rt[[FRAME_COLUMN[[fr]]]], rt$Species)
    bind_rows(lapply(seq_len(nrow(PAIRS)), function(i) {
      a <- times[[PAIRS$sp1[i]]]; b <- times[[PAIRS$sp2[i]]]
      if (length(a) < 2 || length(b) < 2) return(NULL)
      set.seed(BOOT_SEED)
      fit <- estimate_overlap(a, b)
      data.frame(variant = v, frame = fr, sp1 = PAIRS$sp1[i], sp2 = PAIRS$sp2[i],
                 n1 = fit$n_A, n2 = fit$n_B, estimator = fit$estimator,
                 estimate = fit$estimate, ci_low = fit$ci_low, ci_high = fit$ci_high,
                 category = classify_overlap(fit$ci_low, fit$ci_high),
                 stringsAsFactors = FALSE)
    }))
  }))
}))

# The point estimate is deterministic, so `base` must equal the published one.
check_base <- overlap_rows %>% filter(variant == "base") %>%
  inner_join(published, by = c("frame", "sp1", "sp2"), suffix = c("", "_pub"))
stopifnot(nrow(check_base) == nrow(published),
          isTRUE(all.equal(check_base$estimate, check_base$estimate_pub, tolerance = 1e-9)))

base_cat <- overlap_rows %>% filter(variant == "base") %>%
  select(frame, sp1, sp2, estimate_base = estimate, category_base = category)
overlap_out <- overlap_rows %>%
  left_join(base_cat, by = c("frame", "sp1", "sp2")) %>%
  mutate(delta_vs_base = estimate - estimate_base,
         category_moved = category != category_base)

write.csv(overlap_out, here("data", "sensitivity_overlap.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
message("Saved data/sensitivity_overlap.csv")

moved <- overlap_out %>% filter(variant != "base", category_moved)
message(sprintf("Category changes vs the published run: %d of %d variant rows.",
                nrow(moved), sum(overlap_out$variant != "base")))
if (nrow(moved)) print(moved %>% select(variant, frame, sp1, sp2, n1, n2,
                                        estimate, category_base, category))


# ── 4. Episode counts and the liebre autumn rates, per threshold ─────────────
# Effort does not depend on the threshold, so the denominators are the published
# ones: season_effort() and effort_admissible() exactly as 02 uses them, and only
# episodes inside a rate-admissible station-season are counted, as in 02.

season_days <- season_effort(deployments)
rate_units  <- season_days[effort_admissible(season_days, "detections"),
                           c("campaign", "station_id", "season_start")]
effort <- season_days[effort_admissible(season_days, "detections"), ] %>%
  group_by(season_start) %>% summarise(camera_days = sum(effort_days), .groups = "drop")
AUTUMNS <- as.Date(c("2025-03-01", "2026-03-01"))

gap_tables <- c(list(gap_30 = VARIANTS$base), VARIANTS[paste0("gap_", SENSITIVITY_GAPS)])
episode_summary <- bind_rows(lapply(names(gap_tables), function(g) {
  rt <- gap_tables[[g]]
  rt$campaign     <- if ("campaign" %in% names(rt)) rt$campaign else rt$Campaign
  rt$season_start <- season_start(rt$DateTimeOriginal)
  counts <- as.data.frame(table(Species = rt$Species), stringsAsFactors = FALSE)
  liebre <- rt %>% filter(Species == "Liebre", season_start %in% AUTUMNS) %>%
    semi_join(rate_units, by = c("campaign", "Station" = "station_id", "season_start")) %>%
    count(season_start) %>% right_join(effort %>% filter(season_start %in% AUTUMNS),
                                       by = "season_start") %>%
    mutate(n = coalesce(n, 0L), rate = n / camera_days * 100) %>% arrange(season_start)
  data.frame(gap_minutes = as.integer(sub("gap_", "", g)),
             total = nrow(rt),
             t(setNames(counts$Freq, counts$Species)),
             liebre_otono_2025_rate = liebre$rate[1],
             liebre_otono_2026_rate = liebre$rate[2],
             check.names = FALSE)
}))
episode_summary$liebre_autumn_ratio <- with(episode_summary,
                                            liebre_otono_2025_rate / liebre_otono_2026_rate)
print(episode_summary)
write.csv(episode_summary, here("data", "sensitivity_episodes.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
message("Saved data/sensitivity_episodes.csv")


# ── 5. Figure: every pair, every variant ─────────────────────────────────────

plot_df <- overlap_out %>%
  mutate(pair = factor(paste(sp1, "×", sp2)),
         variant = factor(VARIANT_LABELS[variant], levels = VARIANT_LABELS),
         frame = factor(c(clock = "Reloj de la cámara", solar = "Hora solar")[frame],
                        levels = c("Reloj de la cámara", "Hora solar")))
pair_order <- plot_df %>% filter(variant == VARIANT_LABELS[["base"]],
                                 frame == "Reloj de la cámara") %>%
  arrange(estimate) %>% pull(pair) %>% as.character()
plot_df$pair <- factor(plot_df$pair, levels = pair_order)

fig <- ggplot(plot_df, aes(x = estimate, y = pair, colour = variant)) +
  annotate("rect", xmin = -Inf, xmax = 0.50, ymin = -Inf, ymax = Inf, fill = "grey95") +
  annotate("rect", xmin = 0.75, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "grey95") +
  geom_vline(xintercept = c(0.50, 0.75), linetype = "dashed", colour = "grey60") +
  geom_pointrange(aes(xmin = ci_low, xmax = ci_high),
                  position = position_dodge(width = 0.7), size = 0.25) +
  facet_wrap(~frame) +
  scale_colour_manual(values = c("black", "#1b9e77", "#e7298a", "#7570b3", "#d95f02"), name = NULL) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(title = "¿Resiste el solapamiento a otras decisiones de análisis?",
       subtitle = sprintf("%d de %d combinaciones cambian de categoría de Monterroso respecto de lo publicado.",
                          nrow(moved), sum(overlap_out$variant != "base")),
       caption = paste0("Punto = Δ; barra = IC 95 % bootstrap. Bandas: < 0,50 bajo, ≥ 0,75 alto. ",
                        "Umbrales recalculados aquí, sin la restricción de tramos de reloj\n",
                        "(verificado idéntico al productor a 30 min). CT03 y CT09: un tercio de los eventos ",
                        "(apalancamiento). CT09 y CT19: junto a casas de guardaparques (sesgo)."),
       x = "Coeficiente de solapamiento (Δ)", y = NULL) +
  theme_classic(base_size = 11) +
  theme(legend.position = "bottom", strip.background = element_blank(),
        strip.text = element_text(face = "bold"),
        plot.caption = element_text(hjust = 0, colour = "grey30", size = 8))

ggsave(here("figures", "08_sensitivity_overlap.png"), fig, width = 11, height = 7, dpi = 300)
message("Saved figures/08_sensitivity_overlap.png")


# ── 6. Leave one station out, all 27 registry stations ───────────────────────
# Point estimates only. A station with no dated focal episode cannot move anything
# and returns the base value; it is kept so the table covers the whole array.

STATIONS <- sort(readRDS(here("data", "stations_ct.rds"))$Station)
base_point <- overlap_out %>% filter(variant == "base") %>%
  select(frame, sp1, sp2, estimate_base = estimate)

loo <- bind_rows(lapply(STATIONS, function(st) {
  rt <- record_table[record_table$Station != st, ]
  bind_rows(lapply(names(FRAME_COLUMN), function(fr) {
    times <- split(rt[[FRAME_COLUMN[[fr]]]], rt$Species)
    bind_rows(lapply(seq_len(nrow(PAIRS)), function(i) {
      a <- times[[PAIRS$sp1[i]]]; b <- times[[PAIRS$sp2[i]]]
      if (length(a) < 2 || length(b) < 2) return(NULL)
      data.frame(station_left_out = st, frame = fr, sp1 = PAIRS$sp1[i], sp2 = PAIRS$sp2[i],
                 n1 = length(a), n2 = length(b),
                 estimator = overlap_estimator(length(a), length(b)),
                 estimate = overlap_point(a, b), stringsAsFactors = FALSE)
    }))
  }))
})) %>%
  left_join(base_point, by = c("frame", "sp1", "sp2")) %>%
  mutate(delta_vs_base = estimate - estimate_base,
         # Whether the POINT crosses a Monterroso cut (0.50 / 0.75). Categories proper
         # need a CI; this only says which station could push a pair across a line.
         band_base = cut(estimate_base, c(-Inf, 0.50, 0.75, Inf), right = FALSE),
         band_loo  = cut(estimate,      c(-Inf, 0.50, 0.75, Inf), right = FALSE),
         crosses_cut = as.character(band_base) != as.character(band_loo)) %>%
  select(-band_base, -band_loo)

write.csv(loo, here("data", "sensitivity_loo_station.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
message(sprintf("Saved data/sensitivity_loo_station.csv (%d stations × %d frames × %d pairs)",
                length(STATIONS), length(FRAME_COLUMN), nrow(PAIRS)))

most_influential <- loo %>% group_by(frame, sp1, sp2) %>%
  slice_max(abs(delta_vs_base), n = 1, with_ties = FALSE) %>% ungroup() %>%
  arrange(frame, desc(abs(delta_vs_base)))
message("Most influential station per pair:")
print(as.data.frame(most_influential %>%
  transmute(frame, pair = paste(sp1, "×", sp2), station_left_out,
            base = round(estimate_base, 3), without = round(estimate, 3),
            delta = round(delta_vs_base, 3), crosses_cut)), row.names = FALSE)

loo_plot <- loo %>%
  mutate(pair = factor(paste(sp1, "×", sp2), levels = pair_order),
         frame = factor(c(clock = "Reloj de la cámara", solar = "Hora solar")[frame],
                        levels = c("Reloj de la cámara", "Hora solar")))
lim <- max(abs(loo_plot$delta_vs_base))

fig_loo <- ggplot(loo_plot, aes(x = station_left_out, y = pair, fill = delta_vs_base)) +
  geom_tile(colour = "white") +
  geom_point(data = loo_plot %>% filter(crosses_cut), shape = 4, size = 1.6) +
  facet_wrap(~frame, ncol = 1) +
  scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b",
                       limits = c(-lim, lim), name = "Δ sin la estación\n− Δ publicado") +
  labs(title = "¿Qué estación mueve cada par? Dejando fuera una estación a la vez",
       subtitle = "Sólo estimación puntual (sin bootstrap). × = el punto cruza 0,50 o 0,75.",
       caption = "Estaciones sin eventos focales con hora (CT17, CT23, CT27, entre otras) no mueven nada y quedan en blanco.",
       x = "Estación dejada fuera", y = NULL) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5),
        panel.grid = element_blank(), strip.text = element_text(face = "bold"),
        plot.caption = element_text(hjust = 0, colour = "grey30", size = 8))

ggsave(here("figures", "08_sensitivity_loo_station.png"), fig_loo, width = 11, height = 8, dpi = 300)
message("Saved figures/08_sensitivity_loo_station.png")
