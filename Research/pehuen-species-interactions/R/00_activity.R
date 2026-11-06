# 00_activity.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Owns ONE body of knowledge: how a sample of detection times (radians, either
#   frame) becomes a daily activity curve, and how two such samples are compared.
#   That is the circular-kernel layer: the smoothing every curve is drawn with, the
#   minimum sample a curve may be interpreted from, the Δ1/Δ4 estimator choice, the
#   bootstrap interval reported, and the Monterroso categories.
#
# WHY THIS FILE EXISTS (2026-11-06)
#   These decisions lived inside 03_activity_patterns.R and 04_temporal_overlap.R,
#   which run their whole analysis on source(). Nothing else could reuse them, and
#   tests/test_overlap.R had to lift estimate_overlap() out of 04's text with
#   parse(). 07_seasonal_activity.R and 08_sensitivity.R compare activity samples
#   too; without this module they would have restated the rules a third and fourth
#   time. Moving them changed no number: overlap_stats.csv and every 03/04 figure
#   were regenerated and compared against the pre-move run.
#
# FRAME-AGNOSTIC BY DESIGN
#   Every function takes radians and does not know which frame produced them.
#   R/00_timeofday.R owns the conversion; activity_density() only selects the column.
#
# REQUIRES    the `overlap` package (attached by the caller or not; calls are
#             namespaced). No dplyr.
# SOURCED BY  03, 04, 07, 08, tests/test_overlap.R.
# ─────────────────────────────────────────────────────────────────────────────


# ── Sample size: below this, a curve is drawn but not interpreted ─────────────
# The A2 rule of the methods menu. Was FRAME_MIN_EPISODES in 03 (2026-10-05); the
# seasonal and sensitivity scripts apply the same floor to each group they compare.
ACTIVITY_MIN_EPISODES <- 30


# ── Which record_table column holds each frame ───────────────────────────────
# Both columns are built in 01_load_data.R by R/00_timeofday.R; this only names them.
FRAME_COLUMN <- c(clock = "time_rad", solar = "time_solar_rad")


# ── Bootstrap seed ───────────────────────────────────────────────────────────
# Re-applied at the start of EACH frame's loop, not once for the script. Seeding
# once would make every clock-frame CI depend on whether the solar frame ran before
# or after it — a published number silently hostage to the order of a vector.
BOOT_SEED <- 42L


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
overlap_estimator <- function(n_A, n_B) {
  if (min(n_A, n_B) < SMALLER_N_DHAT4_MIN) "Dhat1" else "Dhat4"
}

# The point estimate alone, under the same estimator rule. Deterministic and cheap:
# for runs that repeat the estimate many times (leave-one-station-out in 08), where a
# bootstrap per run would cost minutes and answer nothing the point does not.
overlap_point <- function(times_A, times_B) {
  unname(overlap::overlapEst(times_A, times_B,
                             type = overlap_estimator(length(times_A), length(times_B))))
}

estimate_overlap <- function(times_A, times_B, n_boot = N_BOOT) {
  n_A <- length(times_A)
  n_B <- length(times_B)
  estimator <- overlap_estimator(n_A, n_B)

  point <- overlap::overlapEst(times_A, times_B, type = estimator)
  boot  <- overlap::bootstrap(times_A, times_B, nb = n_boot, type = estimator)
  ci    <- overlap::bootCI(point, boot, conf = 0.95)

  list(
    estimate  = unname(point),
    estimator = estimator,
    ci_low    = unname(ci[CI_TYPE, "lower"]),
    ci_high   = unname(ci[CI_TYPE, "upper"]),
    n_A       = n_A,
    n_B       = n_B
  )
}


# ── 3. Compute kernel densities for multi-species ggplot2 figures ─────────────
# camtrapR::activityDensity() overlays only one species per figure.  For the
# Fig-2 equivalent with multiple species on a single 24-hour axis, we compute
# the kernel densities manually using overlap::densityFit(), which returns a
# density vector at 512 equally-spaced points from 0 to 2π.  We then convert
# the radian grid back to hours (0–24) for a readable x-axis.

activity_density <- function(record_table, species_lbl, frame = "clock") {
  # (a) The episodes' position on the 24-hour circle, in the requested frame of
  #     reference. Both columns are built in 01_load_data.R by R/00_timeofday.R,
  #     which owns both conversions; this function selects, it does not derive.
  if (!frame %in% names(FRAME_COLUMN))
    stop(sprintf("activity_density(): unknown frame '%s'.", frame))
  time_col <- FRAME_COLUMN[[frame]]
  times <- record_table[[time_col]][record_table$Species == species_lbl]

  if (length(times) < 10) {
    warning(sprintf("Only %d episodes for %s — density may be unreliable.", length(times), species_lbl))
  }

  # (b) Fit von Mises kernel density at 512 points spanning the full circle.
  #
  #     ONE SMOOTHING FOR EVERY FIGURE IN THE PROJECT (2026-09-15). This was
  #     `bw = 1.5`, hardcoded for all six species and commented as "the default
  #     bandwidth (in radians)". It was neither. `bw` here is the von Mises
  #     CONCENTRATION parameter — higher means LESS smoothing — and the package's
  #     own data-driven values for these species run 4.7 (guiña) to 22.4 (liebre),
  #     so 1.5 was three to fifteen times more smoothing than anything else in the
  #     project used.
  #
  #     That mattered because camtrapR draws the other curves. `overlap::densityPlot`,
  #     which camtrapR::activityDensity() calls for the per-species panels in section
  #     2 of THIS script, and which activityOverlap() calls for 04's overlap figures,
  #     computes `bw <- getBandWidth(A, kmax = 3) / adjust` with adjust = 1. Calling
  #     getBandWidth() here is that same expression, so the ggplot overlays, the
  #     camtrapR panels and the overlap plots now draw one curve per species instead
  #     of three.
  fit <- overlap::densityFit(times, grid = seq(0, 2 * pi, length.out = 512),
                    bw = overlap::getBandWidth(times, kmax = 3))

  # (c) Convert radians back to hours for the x-axis — and rescale the density with
  #     it. densityFit() returns density per RADIAN; plotted against an hours axis
  #     it would integrate to 24/(2*pi) = 3.82, not to 1, and the y-axis would read
  #     3.82x higher than camtrapR's panels for the identical curve. densityPlot()
  #     applies exactly this factor internally via its xscale = 24 argument.
  #     Verified 2026-09-15: after this, ours and camtrapR's curves agree to 1e-16.
  data.frame(
    species_label = species_lbl,
    frame         = frame,
    hour          = seq(0, 24, length.out = 512),
    density       = fit * (2 * pi / 24)
  )
}
