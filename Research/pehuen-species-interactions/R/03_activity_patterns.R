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
# INPUT   data/records_all.rds    (produced by 01_load_data.R)
#         data/record_table.rds   (camtrapR format, produced by 01_load_data.R)
# OUTPUT  figures/activity_individual/<Species>.png  (one per species, camtrapR)
#         figures/03_activity_native_carnivores.png  (Fig 2 equivalent, ggplot2)
#         figures/03_activity_invasive_species.png
#         figures/03_activity_all_species.png        (all six, faceted)
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(patchwork)
library(overlap)     # densityFit() for multi-species ggplot2 figures
library(camtrapR)    # activityDensity() for per-species individual plots

here::i_am("R/03_activity_patterns.R")
dir.create(here("figures"), showWarnings = FALSE)
dir.create(here("figures", "activity_individual"), showWarnings = FALSE)


# ── 1. Load data ─────────────────────────────────────────────────────────────

record_table <- readRDS(here("data", "record_table.rds"))  # camtrapR format
records      <- readRDS(here("data", "records_all.rds"))   # for ggplot2 multi-species

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

message("Saved per-species plots to figures/activity_individual/")


# ── 3. Compute kernel densities for multi-species ggplot2 figures ─────────────
# camtrapR::activityDensity() overlays only one species per figure.  For the
# Fig-2 equivalent with multiple species on a single 24-hour axis, we compute
# the kernel densities manually using overlap::densityFit(), which returns a
# density vector at 512 equally-spaced points from 0 to 2π.  We then convert
# the radian grid back to hours (0–24) for a readable x-axis.

activity_density <- function(records_df, species_lbl) {
  # (a) Extract the time_rad vector for this species.
  #     time_rad was computed in 01_load_data.R:
  #       (hour*3600 + min*60 + sec) / 86400 * 2π
  times <- records_df %>%
    filter(species_label == species_lbl) %>%
    pull(time_rad)

  if (length(times) < 10) {
    warning(sprintf("Only %d records for %s — density may be unreliable.", length(times), species_lbl))
  }

  # (b) Fit von Mises kernel density at 512 points spanning the full circle.
  #     bw = 1.5 is the default bandwidth (in radians) used in the reference paper.
  fit <- densityFit(times, xgrid = seq(0, 2 * pi, length.out = 512), bw = 1.5)

  # (c) Convert radians back to hours for the x-axis.
  data.frame(
    species_label = species_lbl,
    hour          = seq(0, 24, length.out = 512),
    density       = fit
  )
}

density_list <- lapply(SPECIES_ORDER, function(sp) activity_density(records, sp))
density_df   <- bind_rows(density_list) %>%
  mutate(species_label = factor(species_label, levels = SPECIES_ORDER))


# ── 4. Shared ggplot2 helper for multi-species overlay panels ─────────────────
# Dawn and dusk bands (05:00–07:00 and 18:00–20:00) mark the crepuscular window.
# All other styling is shared so native and invasive panels look identical.

plot_activity <- function(df, title_text) {
  ggplot(df, aes(x = hour, y = density, colour = species_label)) +
    geom_line(linewidth = 1.1) +
    # Shade approximate twilight windows
    annotate("rect", xmin = 5,  xmax = 7,  ymin = -Inf, ymax = Inf,
             fill = "orange", alpha = 0.08) +
    annotate("rect", xmin = 18, xmax = 20, ymin = -Inf, ymax = Inf,
             fill = "orange", alpha = 0.08) +
    scale_x_continuous(
      breaks = c(0, 3, 6, 9, 12, 15, 18, 21, 24),
      labels = c("00:00", "03:00", "06:00", "09:00", "12:00",
                 "15:00", "18:00", "21:00", "24:00"),
      limits = c(0, 24)
    ) +
    scale_colour_manual(values = SPECIES_COLORS, name = NULL) +
    labs(
      title    = title_text,
      subtitle = "Kernel density (von Mises); shaded bands = approx. dawn/dusk",
      x        = "Time of day",
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
  "Daily activity patterns — native carnivores"
)

ggsave(here("figures", "03_activity_native_carnivores.png"),
       fig_native, width = 9, height = 5, dpi = 300)
message("Saved figures/03_activity_native_carnivores.png")


# ── 6. Figure: invasive species overlaid ─────────────────────────────────────

fig_invasive <- plot_activity(
  filter(density_df, species_label %in% INVASIVE_LABELS),
  "Daily activity patterns — invasive species"
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
message("Run 04_temporal_overlap.R next.")
