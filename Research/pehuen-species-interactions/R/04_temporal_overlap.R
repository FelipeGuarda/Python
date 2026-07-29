# 04_temporal_overlap.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Estimate pairwise temporal overlap between focal species pairs and
#   classify each pair as Low / Moderate / High overlap following
#   Monterroso et al. (2014):
#
#       Low       overlap <  0.50
#       Moderate  0.50 ≤ overlap < 0.75
#       High      overlap ≥  0.75
#
#   The estimator (Δ1 vs Δ4) is chosen per pair from the smaller sample
#   size, per Ridout & Linkie (2009): Δ4 when min(n_A, n_B) ≥ 50, Δ1
#   otherwise. See `estimate_overlap()` in section "Overlap estimator"
#   below. The estimator applied to each pair is written to
#   `data/overlap_stats.csv` and to the per-pair PNG footnote so results
#   are always self-describing.
#
#   The classification is applied to the 95% bootstrap CI, not just the
#   point estimate: a pair is "significantly" in a given band only when its
#   entire CI sits inside it. When the CI straddles a threshold, we report a
#   compound label (e.g. "Moderate–High") so we don't overstate confidence.
#
#   Two complementary outputs:
#     A) Per-pair overlay plots — one PNG per species pair with the
#        overlapping kernel density curves, the estimator + point estimate
#        on the title, and the overlap category + CI annotated in an outer
#        strip below the plot.
#
#     B) Summary dot-plot — overlap estimate + 95% CI for all pairs, with
#        dashed lines at 0.50 and 0.75, subtle band shading
#        (Low / Moderate / High), point shape encoding the estimator
#        (Δ4 filled / Δ1 open), and the overlap category appended to each
#        pair label.
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
#         data/record_table.rds  (camtrapR format, produced by 01_load_data.R;
#                                 already 30-min-independence-filtered)
# OUTPUT  figures/overlap_pairs/activity_overlap_<sp1>-<sp2>_<date>.png
#         figures/04_overlap_summary.png            (overlap dot-plot with CI)
#         data/overlap_stats.csv                     (numeric results table)
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
N_BOOT <- 1000        # bootstrap resamples for the overlap-estimate CI
N_GRID <- 512         # grid resolution for kernel density fitting
GRID   <- seq(0, 2 * pi, length.out = N_GRID)

# Estimator dispatch (Ridout & Linkie 2009). Δ4 is appropriate when the
# smaller sample has ≥ SMALLER_N_DHAT4_MIN observations; below that we
# switch to Δ1, which has better small-sample behaviour. The `overlap`
# package documentation places the crossover at 50; the vignette places it
# nearer 75 with a grey zone in between — we take the conservative
# published-doc threshold. Δ5 is never used (unstable, can exceed 1).
SMALLER_N_DHAT4_MIN <- 50

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


# ── Overlap estimator (picks Δ1 vs Δ4 from the smaller sample) ───────────────
# One helper owns the estimator-selection decision. Callers pass two vectors
# of detection times (radians) and receive the point estimate, 95% bootstrap
# CI, the sample sizes, and — critically — the estimator that was applied.
# Downstream code reads `estimator` from the result; nothing else re-derives
# the rule.
estimate_overlap <- function(times_A, times_B, n_boot = N_BOOT) {
  n_A <- length(times_A)
  n_B <- length(times_B)
  estimator <- if (min(n_A, n_B) < SMALLER_N_DHAT4_MIN) "Dhat1" else "Dhat4"

  bw_A <- getBandWidth(times_A)
  bw_B <- getBandWidth(times_B)
  f_A  <- densityFit(times_A, grid = GRID, bw = bw_A)
  f_B  <- densityFit(times_B, grid = GRID, bw = bw_B)

  point <- overlapEst(f_A, f_B, type = estimator)
  boot  <- bootstrap(f_A, f_B, nb = n_boot, type = estimator)
  ci    <- bootCI(point, boot, conf = 0.95)

  list(
    estimate  = unname(point),
    estimator = estimator,
    ci_low    = unname(ci["norm0", "lower"]),
    ci_high   = unname(ci["norm0", "upper"]),
    n_A       = n_A,
    n_B       = n_B
  )
}


# ── 1. Load data ─────────────────────────────────────────────────────────────
# Both the numeric layer (estimate_overlap) and the visual layer
# (activityOverlap) source from record_table so n and shape agree. record_table
# is already independence-filtered upstream (01_load_data.R, 30-min minimum
# gap per station × species × campaign).

record_table <- readRDS(here("data", "record_table.rds"))  # camtrapR format

# Named list of time-of-day (radians) vectors — direct input to the overlap package.
times_by_species <- split(record_table$time_rad, record_table$Species)


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


# ── 3. Compute stats for each pair — overlap + CI + Monterroso category ─────
# All numeric work delegates to `estimate_overlap()`, which owns the Δ1 vs Δ4
# decision (see § "Overlap estimator" above). Per-pair plots (§4) and the
# summary figure (§5) read the `estimator` column instead of hardcoding a
# type — no downstream code re-derives the rule.

message("Computing overlap statistics + Monterroso classification...")

overlap_results <- lapply(PAIRS, function(pair) {
  sp1 <- pair[1]
  sp2 <- pair[2]
  t1  <- times_by_species[[sp1]]
  t2  <- times_by_species[[sp2]]

  if (length(t1) == 0 || length(t2) == 0) return(NULL)

  fit <- estimate_overlap(t1, t2)

  data.frame(
    sp1        = sp1,
    sp2        = sp2,
    n1         = fit$n_A,
    n2         = fit$n_B,
    estimator  = fit$estimator,
    estimate   = fit$estimate,
    ci_low     = fit$ci_low,
    ci_high    = fit$ci_high,
    category   = classify_overlap(fit$ci_low, fit$ci_high),
    pair_label = paste(sp1, "×", sp2),
    guild_type = ifelse(
      sp1 %in% c("Puma", "Guina", "Zorro culpeo") &
      sp2 %in% c("Puma", "Guina", "Zorro culpeo"),
      "Native vs. Native", "Native vs. Invasive"
    ),
    stringsAsFactors = FALSE
  )
})

overlap_df <- bind_rows(overlap_results)

message("\nOverlap coefficients + Monterroso category:")
print(overlap_df %>%
      select(pair_label, n1, n2, estimator, estimate, ci_low, ci_high, category))


# ── 4. Per-pair overlay plots — activityOverlap + category annotation ────────
# activityOverlap() draws two kernel density curves on a shared 24-hour axis,
# shades the overlapping area, and prints the overlap coefficient on the
# title. We open the PNG device manually so we can add the Monterroso
# category and CI in the outer bottom margin (activityOverlap does not
# expose an annotation slot). We also override `main` because camtrapR
# builds the default title from the argument NAMES (sp1/sp2) rather than
# their values. The estimator (Δ1 or Δ4) is read from `row$estimator` and
# passed both to activityOverlap()'s `overlapEstimator=` and to the
# footnote — no local re-derivation of the threshold rule.
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

  estimator_label <- if (row$estimator == "Dhat4") "Δ4" else "Δ1"

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
    overlapEstimator  = row$estimator,
    speciesCol        = "Species",
    recordDateTimeCol = "DateTimeOriginal",
    main              = paste("Activity overlap:", sp1, "and", sp2)
  )
  mtext(
    sprintf("Overlap: %s   (%s = %.3f, 95%% CI [%.2f, %.2f]) — Monterroso et al. 2014; estimator per Ridout & Linkie 2009",
            row$category, estimator_label, row$estimate, row$ci_low, row$ci_high),
    side = 1, line = 1, outer = TRUE, cex = 0.9, col = "grey20"
  )
  dev.off()

  message(sprintf("  %s x %s  %s = %.3f  CI [%.2f, %.2f]  -> %s",
                  sp1, sp2, row$estimator, row$estimate,
                  row$ci_low, row$ci_high, row$category))
}

message("Saved per-pair overlap plots to figures/overlap_pairs/")


# ── 5. Figure: overlap summary dot-plot with Monterroso bands ────────────────
# One row per species pair, ordered by overlap estimate (descending) within
# each guild. Design elements:
#   • Two vertical dashed lines at 0.50 and 0.75 mark the Monterroso cutoffs.
#   • Subtle background band shading distinguishes Low / Moderate / High.
#   • Point shape encodes the estimator used (Δ4 filled; Δ1 open, i.e. the
#     pair had a smaller-sample count below SMALLER_N_DHAT4_MIN and was
#     switched per Ridout & Linkie 2009). This is the same information the
#     old n<75 open-circle flag carried, but read directly off the
#     estimator-selection decision rather than a parallel derived flag.
#   • The Monterroso category is appended in square brackets to each pair
#     label.

overlap_df <- overlap_df %>%
  mutate(pair_label_cat = sprintf("%s   [%s]", pair_label, category)) %>%
  arrange(guild_type, desc(estimate)) %>%
  mutate(pair_label_cat = factor(pair_label_cat,
                                 levels = rev(unique(pair_label_cat))))

# Band shading — rendered as rectangles behind the errorbars.
bands <- data.frame(
  xmin = c(0.00, LOW_MOD_THRESHOLD, MOD_HIGH_THRESHOLD),
  xmax = c(LOW_MOD_THRESHOLD, MOD_HIGH_THRESHOLD, 1.00),
  fill = c("#f7cac9", "#fef3bd", "#c9e4c5")  # light red / yellow / green
)

fig_summary <- ggplot(overlap_df,
                      aes(x = estimate, y = pair_label_cat, colour = guild_type)) +
  geom_rect(data = bands, inherit.aes = FALSE,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = fill),
            alpha = 0.35) +
  scale_fill_identity() +
  geom_vline(xintercept = c(LOW_MOD_THRESHOLD, MOD_HIGH_THRESHOLD),
             linetype = "dashed", colour = "grey40") +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high),
                 height = 0.3, linewidth = 0.8) +
  geom_point(aes(shape = estimator), size = 3.5) +
  scale_shape_manual(
    values = c(Dhat4 = 16, Dhat1 = 1),
    labels = c(Dhat4 = "Δ4 (min n ≥ 50)", Dhat1 = "Δ1 (min n < 50)"),
    name   = "Estimator"
  ) +
  scale_colour_manual(
    values = c("Native vs. Native" = "#2c7bb6", "Native vs. Invasive" = "#d73027"),
    name   = NULL
  ) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     expand = c(0, 0)) +
  labs(
    title    = "Temporal overlap between focal species pairs",
    subtitle = paste0("Δ1/Δ4 selected per pair from the smaller sample (Ridout & Linkie 2009); ",
                      N_BOOT, " bootstrap resamples for 95% CI. ",
                      "Categories from Monterroso et al. (2014)."),
    caption  = "Bands: Low (< 0.50) · Moderate (0.50–0.75) · High (≥ 0.75). Category assigned only when entire CI sits in one band.",
    x        = "Temporal overlap coefficient (Δ1 or Δ4)",
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
         estimator, estimate, ci_low, ci_high, category)

write.csv(stats_out,
          here("data", "overlap_stats.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
message("Saved data/overlap_stats.csv")
message("Run 05_spatial_distribution.R next.")
