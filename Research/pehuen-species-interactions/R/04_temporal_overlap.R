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
# INPUT   data/record_table.rds  (camtrapR format, produced by 01_load_data.R;
#                                 one row per episode, the producer's rule)
# OUTPUT  figures/overlap_pairs/activity_overlap_<sp1>-<sp2>.png
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
source(here::here("R", "00_contract.R"))
dir.create(here("figures"), showWarnings = FALSE)
dir.create(here("figures", "overlap_pairs"), showWarnings = FALSE)

contract_assert_current()

# Re-applied at the start of EACH frame's loop, not once for the script. Seeding
# once would make every clock-frame CI depend on whether the solar frame ran before
# or after it — a published number silently hostage to the order of a vector.
BOOT_SEED <- 42L
set.seed(BOOT_SEED)


# ── Constants + Monterroso classification ────────────────────────────────────
N_BOOT <- 1000        # bootstrap resamples for the overlap-estimate CI

# WHICH OF bootCI()'s FIVE INTERVALS TO REPORT, AND WHY IT IS THIS ONE
#
# The five are all built from two quantities (bootCI source, overlap 0.3.x):
#
#     bias <- mean(bt) - t0                 merr <- sd(bt) * qnorm(0.975)
#
#     norm   = t0 - bias ± merr     bias-corrected normal
#     norm0  = t0 ± merr            NOT corrected — symmetric on the estimate
#     perc   = quantile(bt)         NOT corrected — the raw bootstrap quantiles
#     basic  = 2*t0 - perc[2:1]     bias-corrected by reflection about t0
#     basic0 = perc - bias          the percentile interval shifted by the bias
#
# Note what the `0` suffix means: bias correction REMOVED, not applied. It marks the
# intervals that belong with the UNCORRECTED point estimate t0, not intervals that
# skip a correction they should have made.
#
# WHAT "BIAS" MEANS HERE
#   `bias` is mean(bootstrap replicates) - point estimate: how far the resampling
#   distribution sits from the estimate it was generated around. For a coefficient of
#   overlapping it is a SHRINKAGE TOWARD THE MIDDLE, not a uniform pull in one
#   direction. Measured on this data, 1000 resamples, 2026-09-15:
#
#       pair                   min n     t0    boot mean    bias
#       Guiña × Perro             14   0.221      0.268    +0.048
#       Zorro culpeo × Perro      46   0.304      0.338    +0.034
#       Puma × Guiña              12   0.586      0.555    -0.031
#       Puma × Liebre             12   0.717      0.668    -0.049
#       Zorro culpeo × Liebre    129   0.810      0.809    -0.0007
#       Guiña × Zorro culpeo      14   0.851      0.757    -0.094
#
#   Low estimates are pushed up, high estimates pulled down. Two structural reasons:
#   Δ is bounded in [0, 1], so noise in the fitted densities can only move a near-1
#   overlap downward and a near-0 overlap upward; and Δ integrates the MINIMUM of two
#   density curves, a concave operation, so independent noise in either curve lowers
#   the expected minimum (Jensen).
#
#   IT IS NOT MERELY A SMALL-SAMPLE EFFECT, and it is worth being precise because the
#   obvious reading is wrong. Across the ten real pairs |bias| does correlate -0.48
#   with the smaller sample size, and the only pair with a non-tiny smaller sample
#   (Zorro culpeo × Liebre, n = 129) has a bias of -0.0007. That pattern invites the
#   conclusion that the bias is just thin data and would vanish with more episodes.
#   It would not. On synthetic pairs of two narrow, well-separated clusters, holding
#   the shape fixed and varying n (300 resamples each):
#
#       n =  20    bias +0.032    sd 0.114
#       n =  50    bias +0.036    sd 0.071
#       n = 120    bias +0.036    sd 0.044
#       n = 400    bias +0.026    sd 0.025
#
#   The SD collapses with n, as it must. The BIAS barely moves. What it tracks is the
#   shape of the distributions relative to the smoothing: a kernel estimate of two
#   narrow separated clusters systematically overstates their overlap at every n,
#   because the bandwidth rule keeps smoothing them together. Our n = 129 pair has a
#   near-zero bias because it is a broad, high-overlap pair where smoothing distorts
#   little — not because 129 is large enough to make the problem go away.
#
#   The practical consequence is the same either way: with six of ten pairs resting on
#   12-18 episodes AND several of them narrow and separated, this is a correction that
#   has to be applied rather than argued away.
#
# WHY basic0
#   ?bootCI is explicit: "in general, the bootstrap estimates are biased, so 'perc'
#   should be corrected... 'basic' and 'norm' are appropriate if you are using the
#   bias-corrected estimator, t1. If you use the uncorrected estimator, t0, you should
#   use 'basic0' or 'norm0'."
#
#   We report t0 — `overlapEst()`'s value, the same number camtrapR prints inside each
#   per-pair plot — so the interval must be `basic0` or `norm0`. Between those two,
#   measured on this data: `norm0` is symmetric about t0 and returned [0.700, 1.0028]
#   for Guiña × Zorro culpeo, an upper bound above the maximum the statistic can take;
#   `basic0` stays inside [0, 1] for all ten pairs. So `basic0` is both the documented
#   choice for the estimator we report and the one that does not produce an impossible
#   bound on this data.
#
#   `perc` was used briefly on 2026-09-15 on my recommendation, which misread the
#   suffix convention above: I described `norm0` as bias-corrected when it is not, and
#   `perc` as the only bounded option when `basic0` is bounded here too. Corrected the
#   same day. Against `perc`, `basic0` moves exactly one category — Puma × Liebre from
#   "Low–High" (uninformative) to "Moderate–High" — and where n is large the two
#   coincide exactly (Zorro culpeo × Liebre is [0.72, 0.89] either way), which is the
#   bias going to zero.
#
#   NOT guaranteed bounded by construction. `basic0` is a shift of the percentile
#   interval, so a large enough bias near the boundary could push it past 1. It does
#   not on this data, and tests/test_overlap.R asserts that every published CI lies in
#   [0, 1] — a data check that will fail loudly if a future campaign changes that,
#   rather than a property claimed and never verified.
CI_TYPE <- "basic0"

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



# ── Overlap estimator (picks Δ1 vs Δ4 from the smaller sample) ───────────────
# One helper owns the estimator-selection decision. Callers pass two vectors
# of detection times (radians) and receive the point estimate, 95% bootstrap
# CI, the sample sizes, and — critically — the estimator that was applied.
# Downstream code reads `estimator` from the result; nothing else re-derives
# the rule.
#
# THE ARGUMENTS ARE TIMES, NOT DENSITIES (fixed 2026-09-15)
#   Until this date the three calls below were handed `densityFit()` output —
#   512 density values in [0.02, 0.35] — in the A and B slots, which take
#   detection times in radians. `overlapEst()` cannot tell the difference: it
#   fitted fresh kernels to those density values and returned the overlap of
#   *those*, and `bootstrap()` resampled them, so the CI and every Monterroso
#   category rested on the same mistake. Because both species' density values
#   occupy one narrow numeric band, the error was systematically toward
#   agreement — mean absolute error 0.21 over the ten pairs, maximum 0.54
#   (Guiña × Perro, published 0.757 against a true 0.221), and all ten
#   categories were wrong. It was invisible because the number camtrapR prints
#   inside each per-pair plot is computed correctly, and nothing compared the two.
#   tests/test_overlap.R now does, on every run.
#
#   The bandwidth and grid scaffolding went with it. `overlapEst()` fits its own
#   kernels with the published per-estimator bandwidth adjustments
#   (adjust = c(0.8, 1, 4) for Δ1/Δ4/Δ5, Ridout & Linkie 2009) — the manual path
#   was bypassing exactly the adjustment the estimator is defined with.
estimate_overlap <- function(times_A, times_B, n_boot = N_BOOT) {
  n_A <- length(times_A)
  n_B <- length(times_B)
  estimator <- if (min(n_A, n_B) < SMALLER_N_DHAT4_MIN) "Dhat1" else "Dhat4"

  point <- overlapEst(times_A, times_B, type = estimator)
  boot  <- bootstrap(times_A, times_B, nb = n_boot, type = estimator)
  ci    <- bootCI(point, boot, conf = 0.95)

  list(
    estimate  = unname(point),
    estimator = estimator,
    ci_low    = unname(ci[CI_TYPE, "lower"]),
    ci_high   = unname(ci[CI_TYPE, "upper"]),
    n_A       = n_A,
    n_B       = n_B
  )
}


# ── 1. Load data ─────────────────────────────────────────────────────────────
# Both the numeric layer (estimate_overlap) and the visual layer
# (activityOverlap) source from record_table so n and shape agree. record_table
# is one row per episode, using the rule camera-traps decided at ingest.

record_table <- readRDS(here("data", "record_table.rds"))  # camtrapR format
source(here::here("R", "00_timeofday.R"))   # for time_frame_label() only

# Named lists of time-of-day (radians) vectors — direct input to the overlap package.
# Two frames of reference, built in 01_load_data.R by R/00_timeofday.R:
#
#   clock  the camera's wall time. Every overlap coefficient this project has ever
#          published is in this frame.
#   solar  the same detections relative to that day's sunrise and sunset. Sunrise
#          moves 2.9 h across the year at 39.4°S, so two species that hold fixed
#          schedules relative to it can look differently synchronised in clock time
#          purely because the record spans 19 months.
#
# Neither frame is a correction of the other and the solar one does not supersede the
# published table: the DIFFERENCE between them is the result. A pair whose coefficient
# barely moves is robust to the smearing; one that moves a Monterroso category was
# being read partly as photoperiod.
TIMES_BY_FRAME <- list(
  clock = split(record_table$time_rad,       record_table$Species),
  solar = split(record_table$time_solar_rad, record_table$Species)
)


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

pairs_for_frame <- function(frame) {
  times_by_species <- TIMES_BY_FRAME[[frame]]
  set.seed(BOOT_SEED)   # see BOOT_SEED above: per frame, never once per script

  bind_rows(lapply(PAIRS, function(pair) {
  sp1 <- pair[1]
  sp2 <- pair[2]
  t1  <- times_by_species[[sp1]]
  t2  <- times_by_species[[sp2]]

  if (length(t1) == 0 || length(t2) == 0) return(NULL)

  fit <- estimate_overlap(t1, t2)

  data.frame(
    frame      = frame,
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
  }))
}

# Clock first, always. estimate_overlap() is frame-agnostic by design — it takes
# radians and does not know or care where they came from — so the only thing the
# frame changes is the input vector.
overlap_all <- bind_rows(lapply(c("clock", "solar"), pairs_for_frame))

# §4 and §5 are the published clock-frame figures and are deliberately unchanged.
overlap_df <- overlap_all %>% filter(frame == "clock")

message("\nOverlap coefficients + Monterroso category (clock frame):")
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

  # No date in the name: this project chooses this filename, so the figure keeps one
  # stable path for its whole life and a re-run shows as a change to it rather than a
  # new file beside the old one (R/00_figures.R explains why that matters here).
  png_path <- here("figures", "overlap_pairs",
                   sprintf("activity_overlap_%s-%s.png", sp1, sp2))

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


# ── 5b. Figure: what the frame of reference is worth ─────────────────────────
# A dumbbell, because the quantity of interest is the MOVEMENT between two frames,
# not either value on its own. One row per pair; the open mark is the clock frame
# (every coefficient this project has published), the filled mark is the solar frame,
# and the segment between them is the answer.
#
# Guild stays on the facets and the Monterroso bands keep the colours they have in
# 04_overlap_summary.png, so frame is encoded by SHAPE. Recolouring by frame would
# make the same red mean "invasive" in one figure of this family and "solar" in the
# next — identity must follow the entity, not the panel.
#
# Pairs whose Monterroso category changes are labelled, because that is the only
# movement with a consequence for the written interpretation.

frame_wide <- overlap_all %>%
  select(pair_label, guild_type, estimator, frame, estimate, category) %>%
  tidyr::pivot_wider(names_from = frame,
                     values_from = c(estimate, category)) %>%
  mutate(
    delta          = estimate_solar - estimate_clock,
    category_moved = category_clock != category_solar,
    note           = ifelse(category_moved,
                            sprintf("%s → %s", category_clock, category_solar), NA),
    # A pair whose two marks coincide draws as ONE dot, which reads as a missing
    # series rather than as the result it is. Say it instead of drawing it.
    still          = abs(delta) < 0.01
  ) %>%
  arrange(guild_type, desc(estimate_clock)) %>%
  mutate(pair_label = factor(pair_label, levels = rev(unique(pair_label))))

frame_long <- overlap_all %>%
  mutate(pair_label = factor(pair_label, levels = levels(frame_wide$pair_label)),
         frame = factor(frame, levels = c("clock", "solar")))

fig_frames <- ggplot(frame_wide, aes(y = pair_label)) +
  geom_rect(data = bands, inherit.aes = FALSE,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = fill),
            alpha = 0.35) +
  scale_fill_identity() +
  geom_vline(xintercept = c(LOW_MOD_THRESHOLD, MOD_HIGH_THRESHOLD),
             linetype = "dashed", colour = "grey40") +
  geom_segment(aes(x = estimate_clock, xend = estimate_solar,
                   y = pair_label, yend = pair_label),
               colour = "grey45", linewidth = 1) +
  geom_point(data = frame_long, aes(x = estimate, shape = frame),
             size = 3.2, colour = "grey15", fill = "white", stroke = 0.9) +
  # Labels in a fixed column clear of the plotting area, not beside each dumbbell:
  # trailing the longer mark put them across the 0.75 cutoff line on three rows.
  geom_text(data = subset(frame_wide, category_moved),
            aes(x = 1.03, label = note),
            hjust = 0, size = 3.1, colour = "grey25") +
  geom_text(data = subset(frame_wide, still),
            aes(x = 1.03, label = "sin movimiento"),
            hjust = 0, size = 3.1, colour = "grey45", fontface = "italic") +
  scale_shape_manual(
    values = c(clock = 21, solar = 19),
    labels = c(clock = "Reloj de la cámara", solar = "Hora solar"),
    name   = NULL
  ) +
  scale_x_continuous(limits = c(0, 1.25), breaks = seq(0, 1, 0.25),
                     expand = c(0, 0)) +
  labs(
    title    = "¿Cuánto del solapamiento era fotoperiodo?",
    subtitle = paste0("El mismo par, estimado en los dos marcos de referencia. ",
                      "El amanecer se corre 2,9 h en el año a 39,4°S."),
    caption  = paste0(time_frame_label("solar"),
                      ". Sólo se rotula el par cuya categoría de Monterroso cambia."),
    x        = "Coeficiente de solapamiento (Δ1 o Δ4)",
    y        = NULL
  ) +
  # facet_grid, not facet_wrap: `space = "free_y"` sizes each panel to its own row
  # count. With facet_wrap the three native-native pairs got the same panel height
  # as the seven native-invasive ones and half the figure was blank.
  facet_grid(guild_type ~ ., scales = "free_y", space = "free_y") +
  theme_classic(base_size = 13) +
  theme(
    legend.position  = "bottom",
    strip.background = element_blank(),
    strip.text.y     = element_text(face = "bold", angle = 0),
    plot.caption     = element_text(hjust = 0, colour = "grey30", size = 9),
    panel.grid.major.y = element_line(colour = "grey92")
  )

ggsave(here("figures", "04_overlap_frames.png"),
       fig_frames, width = 12, height = 6, dpi = 300)
message("Saved figures/04_overlap_frames.png")

message("\nFrame of reference — what it moved:")
print(frame_wide %>%
      select(pair_label, estimate_clock, estimate_solar, delta,
             category_clock, category_solar) %>%
      as.data.frame(), digits = 3)
message(sprintf(
  "  mean |delta| = %.3f, max |delta| = %.3f, Monterroso categories changed: %d of %d",
  mean(abs(frame_wide$delta)), max(abs(frame_wide$delta)),
  sum(frame_wide$category_moved), nrow(frame_wide)))


# ── 6. Numeric results table ─────────────────────────────────────────────────
# Persist the stats table so it can be re-read from other scripts or dropped
# into the annual report as a table. BOTH frames, with `frame` as the first column:
# the clock-frame rows are the published series and must not move when the solar
# frame is added beside them.

# Row order: the clock block keeps the order it has always had (guild, then
# descending overlap — §5 used to impose it as a side effect of sorting for the
# figure, which meant the file's shape depended on a plotting step). It is explicit
# here, and the SAME pair order is applied to the solar block, so row i of one block
# and row i of the other are the same pair and the two are readable side by side.
pair_order <- overlap_all %>%
  filter(frame == "clock") %>%
  arrange(guild_type, desc(estimate)) %>%
  mutate(rank = row_number()) %>%
  select(sp1, sp2, rank)

stats_out <- overlap_all %>%
  left_join(pair_order, by = c("sp1", "sp2")) %>%
  arrange(factor(frame, levels = c("clock", "solar")), rank) %>%
  select(frame, sp1, sp2, guild_type, n1, n2,
         estimator, estimate, ci_low, ci_high, category)

write.csv(stats_out,
          here("data", "overlap_stats.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
message("Saved data/overlap_stats.csv")
message("Run 05_spatial_distribution.R next.")
