# 04_temporal_overlap.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Estimate pairwise temporal overlap between focal species pairs using the
#   Dhat4 estimator (Ridout & Linkie 2009), and classify each pair as
#   Low / Moderate / High overlap following Monterroso et al. (2014):
#
#       Low       Dhat4 <  0.50
#       Moderate  0.50 ≤ Dhat4 < 0.75
#       High      Dhat4 ≥  0.75
#
#   The classification is applied to the 95% bootstrap CI, not just the
#   point estimate: a pair is "significantly" in a given band only when its
#   entire CI sits inside it. When the CI straddles a threshold, we report a
#   compound label (e.g. "Moderate–High") so we don't overstate confidence.
#
#   Two complementary outputs:
#     A) Per-pair overlay plots — one PNG per species pair with the
#        overlapping kernel density curves, Dhat4 on the title, and the
#        overlap category + CI annotated in an outer strip below the plot.
#
#     B) Summary dot-plot — Dhat4 + 95% CI for all pairs, with dashed lines
#        at 0.50 and 0.75, subtle band shading (Low / Moderate / High), and
#        the overlap category appended to each pair label.
#
#   Species pairs analysed:
#     Native predators vs. invasive species:
#       Puma        × Jabali,  Puma        × Liebre
#       Guina       × Liebre,  Guina       × Perro
#       Zorro       × Jabali,  Zorro       × Liebre,  Zorro × Perro
#     Native-vs-native (niche partitioning within guild):
#       Puma × Guina,  Puma × Zorro,  Guina × Zorro
#
# INPUT   data/records_all.rds   (produced by 01_load_data.R)
#         data/record_table.rds  (camtrapR format, produced by 01_load_data.R)
# OUTPUT  figures/overlap_pairs/activity_overlap_<sp1>-<sp2>_<date>.png
#         figures/04_overlap_summary.png           (Dhat4 dot-plot with CI)
#         data/overlap_stats.csv                    (numeric results table)
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(overlap)    # overlapEst(), bootstrap(), bootCI()
library(camtrapR)   # activityOverlap() for per-pair overlay plots

here::i_am("R/04_temporal_overlap.R")
dir.create(here("figures"), showWarnings = FALSE)
dir.create(here("figures", "overlap_pairs"), showWarnings = FALSE)

set.seed(42)  # reproducible bootstrap


# ── Constants + Monterroso classification ────────────────────────────────────
N_BOOT <- 1000        # bootstrap resamples for Dhat4 CI
N_GRID <- 512         # grid resolution for kernel density fitting
GRID   <- seq(0, 2 * pi, length.out = N_GRID)

# Monterroso et al. (2014) overlap categories. A pair earns a clean single-
# band label only when its entire 95% CI is inside one band; a CI that
# straddles a threshold gets a compound label so the report doesn't
# overstate confidence in the classification.
LOW_MOD_THRESHOLD  <- 0.50
MOD_HIGH_THRESHOLD <- 0.75

classify_overlap <- function(ci_low, ci_high) {
  if (ci_high < LOW_MOD_THRESHOLD)                                return("Low")
  if (ci_low  >= MOD_HIGH_THRESHOLD)                              return("High")
  if (ci_low  >= LOW_MOD_THRESHOLD & ci_high < MOD_HIGH_THRESHOLD) return("Moderate")
  if (ci_high < MOD_HIGH_THRESHOLD)                               return("Low–Moderate")
  if (ci_low  >= LOW_MOD_THRESHOLD)                               return("Moderate–High")
  "Low–High"   # CI spans the full [0.50, 0.75] band
}

CATEGORY_LEVELS <- c("Low", "Low–Moderate", "Moderate", "Moderate–High", "High", "Low–High")


# ── 1. Load data ─────────────────────────────────────────────────────────────

record_table <- readRDS(here("data", "record_table.rds"))  # camtrapR format
records      <- readRDS(here("data", "records_all.rds"))   # for bootstrap computation

# Build a named list of time_rad vectors — used by the overlap package directly.
times_by_species <- records %>%
  split(.$species_label) %>%
  lapply(function(df) df$time_rad)


# ── 2. Define species pairs ───────────────────────────────────────────────────

PAIRS <- list(
  # Native × Invasive
  c("Puma",         "Jabali"),
  c("Puma",         "Liebre"),
  c("Guina",        "Liebre"),
  c("Guina",        "Perro"),
  c("Zorro culpeo", "Jabali"),
  c("Zorro culpeo", "Liebre"),
  c("Zorro culpeo", "Perro"),
  # Native × Native (niche partitioning)
  c("Puma",         "Guina"),
  c("Puma",         "Zorro culpeo"),
  c("Guina",        "Zorro culpeo")
)


# ── 3. Compute stats for each pair — Dhat4 + CI + Monterroso category ───────
# All numeric work is done in one loop so the per-pair plots (step 4) can be
# annotated from the same source of truth.
#
#   (a) Kernel densities (densityFit + getBandWidth from `overlap`).
#   (b) Point estimate + 95% bootstrap CI on Dhat4 (overlapEst + bootstrap + bootCI).
#   (c) Monterroso classification derived from the CI (not just the point).
#
# We flag pairs where either sample is < 75 records: for small samples, Dhat1
# is the more conservative estimator (Ridout & Linkie 2009), but we report
# Dhat4 to stay consistent with the reference paper.

message("Computing overlap statistics + Monterroso classification...")

overlap_results <- lapply(PAIRS, function(pair) {
  sp1 <- pair[1]
  sp2 <- pair[2]
  t1  <- times_by_species[[sp1]]
  t2  <- times_by_species[[sp2]]

  n1 <- length(t1)
  n2 <- length(t2)

  if (n1 == 0 || n2 == 0) return(NULL)

  # (a) Kernel densities
  bw1 <- getBandWidth(t1)
  bw2 <- getBandWidth(t2)
  f1  <- densityFit(t1, grid = GRID, bw = bw1)
  f2  <- densityFit(t2, grid = GRID, bw = bw2)

  # (b) Point estimate + bootstrap CI on Dhat4
  dhat4 <- overlapEst(f1, f2, type = "Dhat4")
  boot  <- bootstrap(f1, f2, nb = N_BOOT, type = "Dhat4")
  ci    <- bootCI(dhat4, boot, conf = 0.95)
  ci_low  <- ci["norm0", "lower"]
  ci_high <- ci["norm0", "upper"]

  # (c) Monterroso classification from the CI
  category <- classify_overlap(ci_low, ci_high)

  data.frame(
    sp1          = sp1,
    sp2          = sp2,
    n1           = n1,
    n2           = n2,
    dhat4        = dhat4,
    ci_low       = ci_low,
    ci_high      = ci_high,
    category     = category,
    pair_label   = paste(sp1, "×", sp2),
    guild_type   = ifelse(
      sp1 %in% c("Puma", "Guina", "Zorro culpeo") &
      sp2 %in% c("Puma", "Guina", "Zorro culpeo"),
      "Native vs. Native", "Native vs. Invasive"
    ),
    small_sample = (n1 < 75 || n2 < 75),
    stringsAsFactors = FALSE
  )
})

overlap_df <- bind_rows(overlap_results)

message("\nOverlap coefficients + Monterroso category:")
print(overlap_df %>%
      select(pair_label, n1, n2, dhat4, ci_low, ci_high, category, small_sample))


# ── 4. Per-pair overlay plots — activityOverlap + category annotation ────────
# activityOverlap() draws two kernel density curves on a shared 24-hour axis,
# shades the overlapping area, and prints the Dhat4 coefficient on the title.
# We open the PNG device manually so we can add the Monterroso category and
# CI in the outer bottom margin (activityOverlap does not expose an
# annotation slot). We also override `main` because camtrapR builds the
# default title from the argument NAMES (sp1/sp2) rather than their values.
#
# Filename convention: activity_overlap_<sp1>-<sp2>_<YYYY-MM-DD>.png

message("Generating per-pair overlap plots with Monterroso category...")

for (pair in PAIRS) {
  sp1 <- pair[1]
  sp2 <- pair[2]
  row <- overlap_df[overlap_df$sp1 == sp1 & overlap_df$sp2 == sp2, ]

  if (nrow(row) == 0) {
    warning(sprintf("No records for %s or %s — skipping pair.", sp1, sp2))
    next
  }

  png_path <- here("figures", "overlap_pairs",
                   sprintf("activity_overlap_%s-%s_%s.png",
                           sp1, sp2, Sys.Date()))

  png(png_path, width = 8, height = 6, units = "in", res = 300)
  par(oma = c(3, 0, 0, 0))   # outer bottom margin for the annotation strip
  activityOverlap(
    recordTable       = record_table,
    speciesA          = sp1,
    speciesB          = sp2,
    writePNG          = FALSE,
    plotR             = TRUE,
    overlapEstimator  = "Dhat4",
    speciesCol        = "Species",
    recordDateTimeCol = "DateTimeOriginal",
    main              = paste("Activity overlap:", sp1, "and", sp2)
  )
  mtext(
    sprintf("Overlap: %s   (Dhat4 = %.3f, 95%% CI [%.2f, %.2f]) — Monterroso et al. 2014",
            row$category, row$dhat4, row$ci_low, row$ci_high),
    side = 1, line = 1, outer = TRUE, cex = 0.9, col = "grey20"
  )
  dev.off()

  message(sprintf("  %s × %s  Dhat4 = %.3f  CI [%.2f, %.2f]  → %s",
                  sp1, sp2, row$dhat4, row$ci_low, row$ci_high, row$category))
}

message("Saved per-pair overlap plots to figures/overlap_pairs/")


# ── 5. Figure: Dhat4 summary dot-plot with Monterroso bands ─────────────────
# One row per species pair, ordered by Dhat4 descending within each guild.
# Design elements:
#   • Two vertical dashed lines at 0.50 and 0.75 mark the Monterroso cutoffs.
#   • Subtle background band shading distinguishes Low / Moderate / High.
#   • Point shape encodes sample-size flag (open = n < 75 for at least one sp).
#   • The Monterroso category is appended in square brackets to each pair label.

overlap_df <- overlap_df %>%
  mutate(pair_label_cat = sprintf("%s   [%s]", pair_label, category)) %>%
  arrange(guild_type, desc(dhat4)) %>%
  mutate(pair_label_cat = factor(pair_label_cat,
                                 levels = rev(unique(pair_label_cat))))

# Band shading — rendered as rectangles behind the errorbars.
bands <- data.frame(
  xmin = c(0.00, LOW_MOD_THRESHOLD, MOD_HIGH_THRESHOLD),
  xmax = c(LOW_MOD_THRESHOLD, MOD_HIGH_THRESHOLD, 1.00),
  fill = c("#f7cac9", "#fef3bd", "#c9e4c5")  # light red / yellow / green
)

fig_summary <- ggplot(overlap_df,
                      aes(x = dhat4, y = pair_label_cat, colour = guild_type)) +
  geom_rect(data = bands, inherit.aes = FALSE,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = fill),
            alpha = 0.35) +
  scale_fill_identity() +
  geom_vline(xintercept = c(LOW_MOD_THRESHOLD, MOD_HIGH_THRESHOLD),
             linetype = "dashed", colour = "grey40") +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high),
                 height = 0.3, linewidth = 0.8) +
  geom_point(aes(shape = small_sample), size = 3.5) +
  scale_shape_manual(
    values = c(`FALSE` = 16, `TRUE` = 1),
    labels = c(`FALSE` = "n ≥ 75 (both)", `TRUE` = "n < 75 (one or both)"),
    name   = "Sample size"
  ) +
  scale_colour_manual(
    values = c("Native vs. Native" = "#2c7bb6", "Native vs. Invasive" = "#d73027"),
    name   = NULL
  ) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     expand = c(0, 0)) +
  labs(
    title    = "Temporal overlap between focal species pairs",
    subtitle = paste0("Dhat4 with 95% bootstrap CI (", N_BOOT,
                      " resamples). Categories from Monterroso et al. (2014)."),
    caption  = "Bands: Low (Dhat4 < 0.50) · Moderate (0.50–0.75) · High (≥ 0.75). Category assigned only when entire CI sits in one band.",
    x        = expression(Delta[4] ~ "(temporal overlap coefficient)"),
    y        = NULL
  ) +
  facet_wrap(~guild_type, ncol = 1, scales = "free_y") +
  theme_classic(base_size = 13) +
  theme(
    legend.position  = "bottom",
    strip.background = element_blank(),
    strip.text       = element_text(face = "bold"),
    plot.caption     = element_text(hjust = 0, colour = "grey30", size = 10),
    panel.grid.major.y = element_line(colour = "grey92")
  )

ggsave(here("figures", "04_overlap_summary.png"),
       fig_summary, width = 11, height = 8, dpi = 300)
message("Saved figures/04_overlap_summary.png")


# ── 6. Numeric results table ─────────────────────────────────────────────────
# Persist the stats table so it can be re-read from other scripts or dropped
# into the annual report as a table.

stats_out <- overlap_df %>%
  select(sp1, sp2, guild_type, n1, n2,
         dhat4, ci_low, ci_high, category, small_sample)

write.csv(stats_out,
          here("data", "overlap_stats.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
message("Saved data/overlap_stats.csv")
message("Run 05_spatial_distribution.R next.")
