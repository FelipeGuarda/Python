# 03_activity_patterns.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Estimate and visualise the 24-hour activity patterns of the focal species,
#   reproducing the approach in Figure 2 of the reference paper:
#   "Circular daily activity patterns of native carnivores" — kernel density
#   curves on a 24-hour axis, grouped by guild.
#
#   Two complementary outputs:
#     A) Per-species density plots via camtrapR::activityDensity() — one figure
#        per species, saved to figures/activity_individual/.  These use camtrapR's
#        built-in kernel density estimator (von Mises, same underlying method).
#
#     B) Multi-species overlaid figures via overlap::densityFit() + ggplot2 —
#        camtrapR's activityDensity() plots only one species at a time, so for
#        the Fig-2 equivalent with all native carnivores on one panel we compute
#        the densities manually and draw in ggplot2.
#
#   BOTH outputs read record_table.rds: one row per EPISODE, time-admissible. Until
#   2026-09-08 the ggplot curves were fitted on IMAGES while the camtrapR panels
#   beside them used episodes, so the two halves of this script disagreed on the
#   unit. A burst of 3 frames is one animal, not three.
#
# INPUT   data/record_table.rds   (camtrapR format, produced by 01_load_data.R)
# OUTPUT  figures/activity_individual/activity_density_<Species>.png
#           (camtrapR stamps a date into the name it chooses; R/00_figures.R
#            renames it back off so the figure keeps one stable path)
#         figures/03_activity_native_carnivores.png  (Fig 2 equivalent, ggplot2)
#         figures/03_activity_invasive_species.png
#         figures/03_activity_all_species.png        (all six, faceted)
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(overlap)     # densityFit() for multi-species ggplot2 figures
library(camtrapR)    # activityDensity() for per-species individual plots

here::i_am("R/03_activity_patterns.R")
source(here::here("R", "00_contract.R"))
source(here::here("R", "00_admissibility.R"))
source(here::here("R", "00_figures.R"))
source(here::here("R", "00_timeofday.R"))
source(here::here("R", "00_activity.R"))
dir.create(here("figures"), showWarnings = FALSE)
dir.create(here("figures", "activity_individual"), showWarnings = FALSE)

contract_assert_current()


# ── 1. Load data ─────────────────────────────────────────────────────────────
# Episodes, time-admissible, with time_rad precomputed in 01 from the first frame.

record_table <- readRDS(here("data", "record_table.rds"))

SPECIES_ORDER   <- c("Puma", "Guina", "Zorro culpeo", "Jabali", "Liebre", "Perro")
NATIVE_LABELS   <- c("Puma", "Guina", "Zorro culpeo")
INVASIVE_LABELS <- c("Jabali", "Liebre", "Perro")

# Colour palette: one colour per species, consistent across all figures
SPECIES_COLORS <- c(
  "Puma"         = "#1b2a69",
  "Guina"        = "#2c7bb6",
  "Zorro culpeo" = "#74add1",
  "Jabali"       = "#d73027",
  "Liebre"       = "#fc8d59",
  "Perro"        = "#fee090"
)


# ── 2. Per-species density plots — camtrapR::activityDensity() ────────────────
# activityDensity() estimates and plots the activity pattern of one species.
# Internally it uses the same von Mises kernel density as the `overlap` package.
#
# Key arguments:
#   recordTable        — the camtrapR-format record table from 01_load_data.R
#   allSpecies = FALSE — process one species at a time (we loop below)
#   species            — the species label to plot (must match $Species column)
#   writePNG = TRUE    — save the figure to disk
#   plotR    = FALSE   — do not open an interactive graphics window
#   plotDirectory      — destination folder for the saved PNGs
#   speciesCol         — name of the species column (default "Species")
#   recordDateTimeCol  — name of the datetime column (default "DateTimeOriginal")
#
# We skip any species with fewer than 10 records (density estimate unreliable).

message("Generating per-species activity density plots (camtrapR)...")

for (sp in SPECIES_ORDER) {
  n_records <- sum(record_table$Species == sp)

  if (n_records < 10) {
    warning(sprintf("Only %d records for %s — skipping individual density plot.", n_records, sp))
    next
  }

  activityDensity(
    recordTable       = record_table,
    allSpecies        = FALSE,
    species           = sp,
    writePNG          = TRUE,
    plotR             = FALSE,
    plotDirectory     = here("figures", "activity_individual"),
    speciesCol        = "Species",
    recordDateTimeCol = "DateTimeOriginal"
  )

  message(sprintf("  Saved: %s.png", sp))
}

n_fixed <- stabilize_dated_pngs(here("figures", "activity_individual"))
message(sprintf("Saved per-species plots to figures/activity_individual/ (%d renamed off camtrapR's dated names)", n_fixed))


# ── 3. Kernel densities for multi-species ggplot2 figures ─────────────────────
# activity_density() lives in R/00_activity.R, with the smoothing rule it applies.


density_df <- bind_rows(lapply(SPECIES_ORDER, function(sp)
                 activity_density(record_table, sp, "clock"))) %>%
  mutate(species_label = factor(species_label, levels = SPECIES_ORDER))

# The same curves in the solar frame. Kept separate rather than merged into
# density_df: figures 5-7 below are the published clock-frame series and must not
# silently acquire a second set of lines.
density_solar_df <- bind_rows(lapply(SPECIES_ORDER, function(sp)
                      activity_density(record_table, sp, "solar"))) %>%
  mutate(species_label = factor(species_label, levels = SPECIES_ORDER))


# ── 4. Shared ggplot2 helper for multi-species overlay panels ─────────────────
# Dawn and dusk bands (05:00–07:00 and 18:00–20:00) mark the crepuscular window.
# All other styling is shared so native and invasive panels look identical.

plot_activity <- function(df, title_text, frame = "clock") {
  # The twilight annotation is NOT the same object in the two frames, and drawing
  # the clock version on a solar panel would be a false statement about the data.
  #   clock  sunrise wanders 2.9 h across the year, so the best that can be drawn
  #          is an approximate band.
  #   solar  the transformation puts sunrise and sunset on the site's annual mean
  #          for EVERY day, so they are exact lines, not bands. That is the whole
  #          point of the frame and the figure should show it.
  twilight <- if (identical(frame, "solar")) {
    sun_h <- SOLAR_MNANCHOR * 12 / pi
    list(geom_vline(xintercept = sun_h, colour = "darkorange", linewidth = 0.6))
  } else {
    list(annotate("rect", xmin = 5,  xmax = 7,  ymin = -Inf, ymax = Inf,
                  fill = "orange", alpha = 0.08),
         annotate("rect", xmin = 18, xmax = 20, ymin = -Inf, ymax = Inf,
                  fill = "orange", alpha = 0.08))
  }

  ggplot(df, aes(x = hour, y = density, colour = species_label)) +
    geom_line(linewidth = 1.1) +
    twilight +
    scale_x_continuous(
      breaks = c(0, 3, 6, 9, 12, 15, 18, 21, 24),
      labels = c("00:00", "03:00", "06:00", "09:00", "12:00",
                 "15:00", "18:00", "21:00", "24:00"),
      limits = c(0, 24)
    ) +
    scale_colour_manual(values = SPECIES_COLORS, name = NULL) +
    labs(
      title    = title_text,
      subtitle = sprintf("Kernel density (von Mises) on independent episodes (%d-min rule); %s",
                         EPISODE_GAP_MINUTES,
                         if (identical(frame, "solar"))
                           "líneas = amanecer y ocaso (exactos en este marco)"
                         else "shaded bands = approx. dawn/dusk"),
      x        = time_frame_label(frame),
      y        = "Activity density"
    ) +
    theme_classic(base_size = 13) +
    theme(
      legend.position = "bottom",
      axis.text.x     = element_text(angle = 30, hjust = 1)
    )
}


# ── 5. Figure: native carnivores overlaid (Fig 2 equivalent) ─────────────────

fig_native <- plot_activity(
  filter(density_df, species_label %in% NATIVE_LABELS),
  "Daily activity patterns — native carnivores",
  "clock"
)

ggsave(here("figures", "03_activity_native_carnivores.png"),
       fig_native, width = 9, height = 5, dpi = 300)
message("Saved figures/03_activity_native_carnivores.png")


# ── 6. Figure: invasive species overlaid ─────────────────────────────────────

fig_invasive <- plot_activity(
  filter(density_df, species_label %in% INVASIVE_LABELS),
  "Daily activity patterns — invasive species",
  "clock"
)

ggsave(here("figures", "03_activity_invasive_species.png"),
       fig_invasive, width = 9, height = 5, dpi = 300)
message("Saved figures/03_activity_invasive_species.png")


# ── 7. Figure: all six species, faceted ───────────────────────────────────────
# One panel per species arranged 2 columns × 3 rows so curve shapes can be
# compared directly without colour confusion.

fig_all <- ggplot(density_df, aes(x = hour, y = density)) +
  geom_line(aes(colour = species_label), linewidth = 1.1, show.legend = FALSE) +
  annotate("rect", xmin = 5,  xmax = 7,  ymin = -Inf, ymax = Inf,
           fill = "orange", alpha = 0.08) +
  annotate("rect", xmin = 18, xmax = 20, ymin = -Inf, ymax = Inf,
           fill = "orange", alpha = 0.08) +
  facet_wrap(~species_label, ncol = 3) +
  scale_x_continuous(breaks = c(0, 6, 12, 18, 24),
                     labels = c("0", "6", "12", "18", "24")) +
  scale_colour_manual(values = SPECIES_COLORS) +
  labs(
    title = "Daily activity patterns — all focal species",
    x     = "Hour of day",
    y     = "Activity density"
  ) +
  theme_classic(base_size = 12) +
  theme(strip.background = element_blank(),
        strip.text       = element_text(face = "italic"))

ggsave(here("figures", "03_activity_all_species.png"),
       fig_all, width = 11, height = 6, dpi = 300)
message("Saved figures/03_activity_all_species.png")


# ── 8. Figure: the same curves in both frames of reference ───────────────────
# Rows are species, columns are the frame. Reading across a row shows what pooling
# 19 months of clock time costs: sunrise moves 2.9 h over the year at 39.4°S, so a
# peak held relative to sunrise is smeared across nearly three hours of clock time,
# and the curve is flatter than the behaviour. Rowcliffe et al. (2014) name the
# consequence — flattened peaks, OVERESTIMATED activity level.
#
# Only species the A2 rule allows a curve for. Of the six, puma (12), guiña (14) and
# jabalí (18) are in the "plot but do not interpret" band or below; putting them in a
# figure whose subject is a difference between two curves would invite reading a
# difference that is sampling noise. The figure says which are missing and why.

FRAME_MIN_EPISODES <- ACTIVITY_MIN_EPISODES   # R/00_activity.R

frame_n <- record_table %>% count(Species, name = "n")
frame_species <- frame_n %>% filter(n >= FRAME_MIN_EPISODES) %>% pull(Species)
frame_skipped <- frame_n %>% filter(n <  FRAME_MIN_EPISODES)

both_frames <- bind_rows(density_df, density_solar_df) %>%
  filter(species_label %in% frame_species) %>%
  left_join(frame_n, by = c("species_label" = "Species")) %>%
  mutate(
    species_n = sprintf("%s (n = %d)", species_label, n),
    frame_lbl = factor(ifelse(frame == "clock",
                              "Reloj de la cámara", "Hora solar"),
                       levels = c("Reloj de la cámara", "Hora solar"))
  )

sun_h <- SOLAR_MNANCHOR * 12 / pi
sun_lines <- data.frame(
  frame_lbl = factor("Hora solar",
                     levels = levels(both_frames$frame_lbl)),
  x = sun_h
)

fig_frames <- ggplot(both_frames, aes(x = hour, y = density)) +
  geom_vline(data = sun_lines, aes(xintercept = x),
             colour = "darkorange", linewidth = 0.6) +
  geom_line(aes(colour = species_label), linewidth = 1.1, show.legend = FALSE) +
  facet_grid(species_n ~ frame_lbl, scales = "free_y") +
  scale_x_continuous(breaks = c(0, 6, 12, 18, 24),
                     labels = c("0", "6", "12", "18", "24")) +
  scale_colour_manual(values = SPECIES_COLORS) +
  labs(
    title    = "La misma actividad, en dos marcos de referencia",
    subtitle = paste0("A 39,4°S el amanecer se corre 2,9 h en el año, de modo que una hora fija\n",
                      "del reloj no es una hora fija del día. Las líneas naranjas son el amanecer\n",
                      "y el ocaso: exactos en el marco solar, sólo aproximables en el del reloj."),
    caption  = sprintf(
      "%s\nSin curva: %s — bajo %d episodios independientes (regla A2 del menú de métodos).",
      time_frame_label("solar"),
      paste(sprintf("%s (%d)", frame_skipped$Species, frame_skipped$n), collapse = ", "),
      FRAME_MIN_EPISODES),
    x        = "Hora",
    y        = "Densidad de actividad"
  ) +
  theme_classic(base_size = 12) +
  theme(strip.background = element_blank(),
        strip.text.y     = element_text(face = "italic"),
        strip.text.x     = element_text(face = "bold"),
        plot.caption     = element_text(hjust = 0, colour = "grey30", size = 9))

ggsave(here("figures", "03_activity_frames.png"),
       fig_frames, width = 10, height = 7.6, dpi = 300)
message(sprintf("Saved figures/03_activity_frames.png (%d species; skipped %s)",
                length(frame_species),
                paste(frame_skipped$Species, collapse = ", ")))
message("Run 04_temporal_overlap.R next.")
