# 03_activity_patterns.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Estimate and visualise the 24-hour activity patterns of the focal species,
#   reproducing the approach in Figure 2 of the reference paper:
#   "Circular daily activity patterns of native carnivores" — one kernel density
#   curve per species overlaid on a 24-hour clock.
#
#   Method: von Mises kernel density estimation on circular time data using the
#   `overlap` package (Ridout & Linkie 2009).  Time is expressed as radians
#   (0 = midnight, π = noon, 2π = midnight again).
#
# INPUT   data/records_all.rds   (produced by 01_load_data.R)
# OUTPUT  figures/03_activity_native_carnivores.png  (Fig 2 equivalent)
#         figures/03_activity_invasive_species.png
#         figures/03_activity_all_species.png        (all six in one panel)
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(patchwork)
library(overlap)    # kernel density on circular time data

here::i_am("R/03_activity_patterns.R")
dir.create(here("figures"), showWarnings = FALSE)


# ── 1. Load data ─────────────────────────────────────────────────────────────

records <- readRDS(here("data", "records_all.rds"))

SPECIES_ORDER  <- c("Puma", "Guina", "Zorro culpeo", "Jabali", "Liebre", "Perro")
NATIVE_LABELS  <- c("Puma", "Guina", "Zorro culpeo")
INVASIVE_LABELS<- c("Jabali", "Liebre", "Perro")

# Colour palette: one colour per species, consistent across all figures
SPECIES_COLORS <- c(
  "Puma"         = "#1b2a69",
  "Guina"        = "#2c7bb6",
  "Zorro culpeo" = "#74add1",
  "Jabali"       = "#d73027",
  "Liebre"       = "#fc8d59",
  "Perro"        = "#fee090"
)


# ── 2. Helper: build kernel density dataframe for one species ─────────────────
# The `overlap` package's densityPlot() draws to base graphics.  Here we
# replicate the calculation manually so we can plot in ggplot2.
#
# Steps:
#   (a) Extract the time_rad vector for the species (already in records_all).
#   (b) Fit a von Mises kernel density with `overlap::densityFit()`.
#       This returns density values at 512 equally-spaced points from 0 to 2π.
#   (c) Convert those 512 points back to hours (0–24) for a readable x-axis.

activity_density <- function(records_df, species_lbl) {
  times <- records_df %>%
    filter(species_label == species_lbl) %>%
    pull(time_rad)

  if (length(times) < 10) {
    warning(sprintf("Only %d records for %s — density estimate may be unreliable.",
                    length(times), species_lbl))
  }

  # Fit kernel density at 512 points spanning the full 24-hour circle
  fit <- densityFit(times, xgrid = seq(0, 2 * pi, length.out = 512), bw = 1.5)

  # Convert radians back to hours for the x-axis
  data.frame(
    species_label = species_lbl,
    hour          = seq(0, 24, length.out = 512),
    density       = fit
  )
}


# ── 3. Compute densities for all six focal species ────────────────────────────

density_list <- lapply(SPECIES_ORDER, function(sp) activity_density(records, sp))
density_df   <- bind_rows(density_list) %>%
  mutate(
    species_label = factor(species_label, levels = SPECIES_ORDER),
    guild = ifelse(species_label %in% NATIVE_LABELS, "Native", "Invasive")
  )


# ── 4. Helper: build one activity-pattern ggplot ─────────────────────────────
# Shared plotting function so native and invasive panels look identical.

plot_activity <- function(df, title_text) {
  ggplot(df, aes(x = hour, y = density, colour = species_label)) +
    geom_line(linewidth = 1.1) +
    # Shade dawn and dusk bands (approximate twilight hours)
    annotate("rect", xmin = 5, xmax = 7,  ymin = -Inf, ymax = Inf,
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
      legend.position  = "bottom",
      axis.text.x      = element_text(angle = 30, hjust = 1)
    )
}


# ── 5. Figure: native carnivores (Fig 2 equivalent) ──────────────────────────

fig_native <- plot_activity(
  filter(density_df, species_label %in% NATIVE_LABELS),
  "Daily activity patterns — native carnivores"
)

ggsave(here("figures", "03_activity_native_carnivores.png"),
       fig_native, width = 9, height = 5, dpi = 300)
message("Saved figures/03_activity_native_carnivores.png")


# ── 6. Figure: invasive species ──────────────────────────────────────────────

fig_invasive <- plot_activity(
  filter(density_df, species_label %in% INVASIVE_LABELS),
  "Daily activity patterns — invasive species"
)

ggsave(here("figures", "03_activity_invasive_species.png"),
       fig_invasive, width = 9, height = 5, dpi = 300)
message("Saved figures/03_activity_invasive_species.png")


# ── 7. Figure: all six species in one faceted panel ──────────────────────────
# One panel per species, arranged 2 columns × 3 rows so shapes can be compared
# without colour confusion.

fig_all <- ggplot(density_df, aes(x = hour, y = density)) +
  geom_line(aes(colour = species_label), linewidth = 1.1, show.legend = FALSE) +
  annotate("rect", xmin = 5, xmax = 7, ymin = -Inf, ymax = Inf,
           fill = "orange", alpha = 0.08) +
  annotate("rect", xmin = 18, xmax = 20, ymin = -Inf, ymax = Inf,
           fill = "orange", alpha = 0.08) +
  facet_wrap(~species_label, ncol = 3) +
  scale_x_continuous(breaks = c(0, 6, 12, 18, 24),
                     labels = c("0", "6", "12", "18", "24")) +
  scale_colour_manual(values = SPECIES_COLORS) +
  labs(
    title    = "Daily activity patterns — all focal species",
    x        = "Hour of day",
    y        = "Activity density"
  ) +
  theme_classic(base_size = 12) +
  theme(strip.background = element_blank(),
        strip.text = element_text(face = "italic"))

ggsave(here("figures", "03_activity_all_species.png"),
       fig_all, width = 11, height = 6, dpi = 300)
message("Saved figures/03_activity_all_species.png")
message("Run 04_temporal_overlap.R next.")
