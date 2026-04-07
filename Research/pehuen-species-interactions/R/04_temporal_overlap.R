# 04_temporal_overlap.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Estimate pairwise temporal overlap between focal species pairs using the
#   Dhat4 estimator (Ridout & Linkie 2009), reproducing the approach in
#   Figure 3 of the reference paper.
#
#   For each species pair we compute:
#     - Dhat4 overlap coefficient (0 = no overlap, 1 = identical patterns)
#     - 95% bootstrap confidence interval (1000 resamples)
#     - A visual overlay of the two kernel density curves
#
#   Species pairs analysed:
#     Native predators vs. their invasive prey/competitors:
#       Puma        × Jabali
#       Puma        × Liebre
#       Guina       × Liebre
#       Guina       × Perro
#       Zorro       × Jabali
#       Zorro       × Liebre
#       Zorro       × Perro
#     Native-vs-native (to show niche partitioning within guild):
#       Puma        × Guina
#       Puma        × Zorro
#       Guina       × Zorro
#
# INPUT   data/records_all.rds   (produced by 01_load_data.R)
# OUTPUT  figures/04_overlap_summary.png        (Dhat4 dot-plot with CI — Fig 3 equivalent)
#         figures/04_overlap_<sp1>_<sp2>.png    (one overlay curve per pair)
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(patchwork)
library(overlap)    # overlapEst(), bootEst(), bootCI()

here::i_am("R/04_temporal_overlap.R")
dir.create(here("figures"), showWarnings = FALSE)

set.seed(42)  # reproducible bootstrap


# ── 1. Load data ─────────────────────────────────────────────────────────────

records <- readRDS(here("data", "records_all.rds"))

SPECIES_COLORS <- c(
  "Puma"         = "#1b2a69",
  "Guina"        = "#2c7bb6",
  "Zorro culpeo" = "#74add1",
  "Jabali"       = "#d73027",
  "Liebre"       = "#fc8d59",
  "Perro"        = "#fee090"
)

# Build a named list of time_rad vectors, one per species.
# These are the raw circular time data (radians) used by all overlap functions.
times_by_species <- records %>%
  split(.$species_label) %>%
  lapply(function(df) df$time_rad)


# ── 2. Define species pairs to analyse ───────────────────────────────────────

PAIRS <- list(
  # Predator–prey / native–invasive
  c("Puma",         "Jabali"),
  c("Puma",         "Liebre"),
  c("Guina",        "Liebre"),
  c("Guina",        "Perro"),
  c("Zorro culpeo", "Jabali"),
  c("Zorro culpeo", "Liebre"),
  c("Zorro culpeo", "Perro"),
  # Within native guild (niche partitioning)
  c("Puma",         "Guina"),
  c("Puma",         "Zorro culpeo"),
  c("Guina",        "Zorro culpeo")
)


# ── 3. Compute Dhat4 + bootstrap CI for each pair ────────────────────────────
# Dhat4 is the recommended estimator when both sample sizes exceed ~75 records
# (Ridout & Linkie 2009).  For smaller samples, Dhat1 is more conservative;
# we report Dhat4 but flag pairs where either sample is < 75 records.
#
# Bootstrap procedure:
#   (a) Resample each species' time vector with replacement (1000 times).
#   (b) Compute Dhat4 on each resample pair.
#   (c) Derive the 0.025 and 0.975 quantiles as the 95% CI.

N_BOOT <- 1000

overlap_results <- lapply(PAIRS, function(pair) {
  sp1 <- pair[1]
  sp2 <- pair[2]
  t1  <- times_by_species[[sp1]]
  t2  <- times_by_species[[sp2]]

  n1 <- length(t1)
  n2 <- length(t2)

  if (n1 == 0 || n2 == 0) {
    warning(sprintf("No records for %s or %s — skipping pair.", sp1, sp2))
    return(NULL)
  }

  # Point estimate
  dhat4 <- overlapEst(t1, t2, type = "Dhat4")

  # Bootstrap CI
  boot   <- bootEst(t1, t2, nb = N_BOOT, type = "Dhat4")
  ci     <- bootCI(dhat4, boot, conf = 0.95)

  small_sample_flag <- (n1 < 75 || n2 < 75)

  data.frame(
    sp1         = sp1,
    sp2         = sp2,
    n1          = n1,
    n2          = n2,
    dhat4       = dhat4,
    ci_low      = ci["norm0", "lower"],
    ci_high     = ci["norm0", "upper"],
    pair_label  = paste(sp1, "×", sp2),
    guild_type  = ifelse(
      sp1 %in% c("Puma", "Guina", "Zorro culpeo") &
      sp2 %in% c("Puma", "Guina", "Zorro culpeo"),
      "Native vs. Native", "Native vs. Invasive"
    ),
    small_sample = small_sample_flag,
    stringsAsFactors = FALSE
  )
})

overlap_df <- bind_rows(overlap_results)

# Print to console for quick inspection
message("\nOverlap coefficients (Dhat4):")
print(overlap_df %>% select(pair_label, n1, n2, dhat4, ci_low, ci_high, small_sample))


# ── 4. Figure: Dhat4 dot-plot with CI (Fig 3 equivalent) ─────────────────────
# One row per species pair, sorted by Dhat4 descending within each guild type.
# Pairs with small-sample warnings are shown with a hollow point.

overlap_df <- overlap_df %>%
  arrange(guild_type, desc(dhat4)) %>%
  mutate(pair_label = factor(pair_label, levels = rev(unique(pair_label))))

fig_summary <- ggplot(overlap_df,
                      aes(x = dhat4, y = pair_label, colour = guild_type)) +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0.3, linewidth = 0.8) +
  geom_point(aes(shape = small_sample), size = 3.5) +
  scale_shape_manual(
    values = c(`FALSE` = 16, `TRUE` = 1),
    labels = c(`FALSE` = "n >= 75 (both)", `TRUE` = "n < 75 (one or both)"),
    name   = "Sample size"
  ) +
  scale_colour_manual(
    values = c("Native vs. Native" = "#2c7bb6", "Native vs. Invasive" = "#d73027"),
    name   = NULL
  ) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey50") +
  labs(
    title    = "Temporal overlap between focal species pairs",
    subtitle = paste0("Dhat4 estimator with 95% bootstrap CI (", N_BOOT, " resamples)"),
    x        = expression(Delta[4] ~ "(temporal overlap coefficient)"),
    y        = NULL
  ) +
  facet_wrap(~guild_type, ncol = 1, scales = "free_y") +
  theme_classic(base_size = 13) +
  theme(
    legend.position  = "bottom",
    strip.background = element_blank(),
    strip.text       = element_text(face = "bold")
  )

ggsave(here("figures", "04_overlap_summary.png"),
       fig_summary, width = 8, height = 8, dpi = 300)
message("Saved figures/04_overlap_summary.png")


# ── 5. Individual overlap overlay plots for each pair ────────────────────────
# For each pair: two kernel density curves on the same clock face, with the
# overlapping area shaded. Equivalent to the per-pair panels in Fig 3.

plot_overlap_pair <- function(t1, t2, sp1_lbl, sp2_lbl) {
  # Evaluate density at 512 equally-spaced points
  grid  <- seq(0, 2 * pi, length.out = 512)
  hours <- seq(0, 24, length.out = 512)

  d1 <- densityFit(t1, xgrid = grid, bw = 1.5)
  d2 <- densityFit(t2, xgrid = grid, bw = 1.5)

  df_lines <- data.frame(
    hour    = rep(hours, 2),
    density = c(d1, d2),
    species = factor(rep(c(sp1_lbl, sp2_lbl), each = 512),
                     levels = c(sp1_lbl, sp2_lbl))
  )

  # Overlap polygon: min(d1, d2) at each point
  df_overlap <- data.frame(
    hour    = hours,
    density = pmin(d1, d2)
  )

  ggplot() +
    geom_ribbon(data = df_overlap,
                aes(x = hour, ymin = 0, ymax = density),
                fill = "grey70", alpha = 0.5) +
    geom_line(data = df_lines,
              aes(x = hour, y = density, colour = species), linewidth = 1.2) +
    scale_colour_manual(
      values = setNames(SPECIES_COLORS[c(sp1_lbl, sp2_lbl)], c(sp1_lbl, sp2_lbl)),
      name   = NULL
    ) +
    scale_x_continuous(breaks = c(0, 6, 12, 18, 24),
                       labels = c("00:00", "06:00", "12:00", "18:00", "24:00")) +
    labs(x = "Time of day", y = "Activity density") +
    theme_classic(base_size = 12) +
    theme(legend.position = "bottom")
}

# Save one PNG per pair
for (i in seq_len(nrow(overlap_df))) {
  row  <- overlap_df[i, ]
  sp1  <- row$sp1
  sp2  <- row$sp2
  t1   <- times_by_species[[sp1]]
  t2   <- times_by_species[[sp2]]

  dhat_label <- sprintf("Delta[4] == %.3f~(95%%~CI:~%.3f-%.3f)",
                        row$dhat4, row$ci_low, row$ci_high)

  p <- plot_overlap_pair(t1, t2, sp1, sp2) +
    labs(
      title    = paste("Temporal overlap:", sp1, "×", sp2),
      subtitle = parse(text = dhat_label)
    )

  fname <- here("figures", sprintf("04_overlap_%s_%s.png",
                                   gsub(" ", "_", sp1), gsub(" ", "_", sp2)))
  ggsave(fname, p, width = 7, height = 4.5, dpi = 300)
}

message(sprintf("Saved %d individual overlap figures to figures/", nrow(overlap_df)))
message("Run 05_spatial_distribution.R next.")
