# tests/test_overlap.R
# ─────────────────────────────────────────────────────────────────────────────
# Tests for estimate_overlap() in R/04_temporal_overlap.R. Base R plus `overlap`,
# same style as the other two test files. Run:
#
#     conda run -n pehuen-analysis Rscript tests/test_overlap.R
#
# WHY THIS FILE EXISTS
#   From 2026-07-28 to 2026-09-15 estimate_overlap() passed densityFit() output —
#   density values — into overlapEst(A, B), whose A and B are detection times in
#   radians. The function re-fitted kernels to those density values and returned
#   the overlap of those; bootstrap() resampled them for the CI. Mean absolute
#   error over the ten published pairs was 0.21, maximum 0.54, and all ten
#   Monterroso categories were wrong.
#
#   It was invisible for seven weeks for one reason: the estimate camtrapR prints
#   INSIDE each per-pair plot is computed correctly, straight from the times, and
#   nothing ever compared it against the one in our footnote. A reader looking at
#   activity_overlap_Zorro culpeo-Liebre.png could see "Dhat4=0.81" in the panel
#   and "Δ4 = 0.879" below it, and the code had no opinion about that.
#
#   So the anchor assertion here is exactly that comparison: our helper must agree
#   with a direct overlap::overlapEst() call to floating-point tolerance. Any
#   future refactor that reintroduces a private density path fails this file
#   rather than shipping a plausible wrong number.
#
# SOURCING
#   04_temporal_overlap.R runs its whole analysis on source(), so this file cannot
#   source it. It re-declares the two constants it needs and defines the function
#   under test by extracting it from the script text — which means the test reads
#   the SHIPPING definition, not a copy that could drift from it.
# ─────────────────────────────────────────────────────────────────────────────

library(here)
here::i_am("tests/test_overlap.R")
suppressPackageStartupMessages(library(overlap))

.failures <- 0L
check <- function(label, ok) {
  if (isTRUE(ok)) {
    cat(sprintf("  ok   %s\n", label))
  } else {
    .failures <<- .failures + 1L
    cat(sprintf("  FAIL %s\n", label))
  }
}

# Pull the real definitions out of the script rather than restating them: a copy
# here would be a second place the estimator rule lives, which is the class of
# defect this file exists to catch.
src <- readLines(here::here("R", "04_temporal_overlap.R"))
grab_line <- function(pattern) src[grep(pattern, src)[1]]
grab_block <- function(start_pattern, end_pattern) {
  i <- grep(start_pattern, src)[1]
  j <- grep(end_pattern, src)
  j <- j[j > i][1]
  paste(src[i:j], collapse = "\n")
}
eval(parse(text = grab_line("^N_BOOT <-")))
eval(parse(text = grab_line("^CI_TYPE <-")))
eval(parse(text = grab_line("^SMALLER_N_DHAT4_MIN <-")))
eval(parse(text = grab_block("^estimate_overlap <- function", "^\\}$")))

check("estimate_overlap was extracted from the script", is.function(estimate_overlap))
check("SMALLER_N_DHAT4_MIN came with it", identical(SMALLER_N_DHAT4_MIN, 50))
check("CI_TYPE is a bootCI row name", CI_TYPE %in% c("norm", "norm0", "basic", "basic0", "perc"))
# ?bootCI: "'basic' and 'norm' are appropriate if you are using the bias-corrected
# estimator, t1. If you use the uncorrected estimator, t0, you should use 'basic0' or
# 'norm0'." We report overlapEst()'s t0, so only those two are admissible here. This
# guard encodes the package's own rule so a future edit cannot quietly pair an
# uncorrected estimate with an interval built for the corrected one.
check("CI_TYPE is one of the two intervals valid for an uncorrected estimate",
      CI_TYPE %in% c("basic0", "norm0"))


# Synthetic times: two von Mises-ish clusters with a known, moderate overlap.
# Fixed, so the numbers below do not depend on the canonical table.
set.seed(7)
rad <- function(h) h / 24 * 2 * pi
big_A <- (rad(rnorm(120, 22, 2)) %% (2 * pi))
big_B <- (rad(rnorm(120,  2, 2)) %% (2 * pi))
small_A <- (rad(rnorm(20, 22, 2)) %% (2 * pi))
small_B <- (rad(rnorm(30,  2, 2)) %% (2 * pi))


cat("estimator dispatch (Ridout & Linkie 2009)\n")
set.seed(1); big <- estimate_overlap(big_A, big_B, n_boot = 100)
check("both samples >= 50 selects Dhat4", big$estimator == "Dhat4")
set.seed(1); small <- estimate_overlap(small_A, small_B, n_boot = 100)
check("smaller sample < 50 selects Dhat1", small$estimator == "Dhat1")
set.seed(1); mixed <- estimate_overlap(big_A, small_B, n_boot = 100)
check("the SMALLER sample decides, not the larger", mixed$estimator == "Dhat1")
check("n_A and n_B are the input lengths",
      big$n_A == length(big_A) && big$n_B == length(big_B))


cat("the anchor: our estimate IS the package's estimate\n")
# This is the assertion the 2026-09-15 defect would have failed. Passing density
# fits instead of times returned 0.879 where the package returns 0.810.
set.seed(1); got4 <- estimate_overlap(big_A, big_B, n_boot = 100)$estimate
want4 <- unname(overlapEst(big_A, big_B, type = "Dhat4"))
check(sprintf("Dhat4 matches overlapEst() exactly (%.6f)", want4),
      isTRUE(all.equal(got4, want4, tolerance = 1e-12)))

set.seed(1); got1 <- estimate_overlap(small_A, small_B, n_boot = 100)$estimate
want1 <- unname(overlapEst(small_A, small_B, type = "Dhat1"))
check(sprintf("Dhat1 matches overlapEst() exactly (%.6f)", want1),
      isTRUE(all.equal(got1, want1, tolerance = 1e-12)))

# The specific wrong path, asserted to be wrong, so nobody reinstates it thinking
# it was equivalent. densityFit output is not a time vector.
GRID_512 <- seq(0, 2 * pi, length.out = 512)
f_A <- densityFit(big_A, grid = GRID_512, bw = getBandWidth(big_A))
f_B <- densityFit(big_B, grid = GRID_512, bw = getBandWidth(big_B))
check("passing density fits gives a materially DIFFERENT number",
      abs(unname(overlapEst(f_A, f_B, type = "Dhat4")) - want4) > 0.01)
check("...because density values are not radians",
      max(f_A) < 1 && max(big_A) > 5)


cat("the estimate and its interval are coherent\n")
set.seed(1); r <- estimate_overlap(big_A, big_B, n_boot = 200)
check("estimate is a coefficient in [0, 1]", r$estimate >= 0 && r$estimate <= 1)
check("ci_low <= ci_high", r$ci_low <= r$ci_high)
# basic0 = perc - bias re-centres the bootstrap quantiles on t0, so t0 falls inside
# whenever mean(bt) falls inside its own 2.5-97.5% quantiles -- true of any
# non-pathological bootstrap distribution. Unlike a [0,1] bound, this IS a property of
# the construction, so it is asserted on synthetic data.
check("the CI contains the point estimate",
      r$ci_low <= r$estimate && r$estimate <= r$ci_high)

# The documented formula, checked rather than trusted: basic0 is the percentile
# interval shifted by the bootstrap bias.
set.seed(99)
t0 <- unname(overlapEst(big_A, big_B, type = "Dhat4"))
bt <- bootstrap(big_A, big_B, nb = 200, type = "Dhat4")
ci <- bootCI(t0, bt, conf = 0.95)
bias <- mean(bt[is.finite(bt)]) - t0
check("basic0 == perc - bias, as ?bootCI documents",
      isTRUE(all.equal(unname(ci["basic0", ]), unname(ci["perc", ]) - bias,
                       tolerance = 1e-12)))
check("norm0 is symmetric about the estimate — it is NOT bias-corrected",
      isTRUE(all.equal(mean(unname(ci["norm0", ])), t0, tolerance = 1e-12)))
# What DOES shrink with n is the spread, not the bias. Asserting the bias shrinks
# would be asserting something false: measured on this same synthetic shape it runs
# +0.032 at n = 20 and +0.026 at n = 400 while the SD falls from 0.114 to 0.025. The
# bias tracks the shape of the distributions relative to the smoothing; the SD tracks
# the sample size. Only the second is a property of n.
set.seed(7)
sd_by_n <- vapply(c(20L, 120L), function(n) {
  A <- (rad(rnorm(n, 22, 2)) %% (2 * pi)); B <- (rad(rnorm(n, 2, 2)) %% (2 * pi))
  sd(bootstrap(A, B, nb = 150, type = "Dhat4"))
}, numeric(1))
check(sprintf("bootstrap SD shrinks with n (%.3f at 20 -> %.3f at 120)",
              sd_by_n[1], sd_by_n[2]),
      sd_by_n[2] < sd_by_n[1])

cat("identical inputs overlap perfectly\n")
set.seed(1); same <- estimate_overlap(big_A, big_A, n_boot = 100)
check("a species against itself estimates ~1", same$estimate > 0.99)


cat("time_rad: this project's radians against camtrapR's own\n")
# F003. 01_load_data.R computes time_rad as (h*3600 + m*60 + s)/86400 * 2*pi;
# camtrapR derives its own radians from DateTimeOriginal inside activityDensity() and
# activityOverlap(), and every figure those two draw uses ITS version while every
# number this project publishes uses OURS. They agreed, and nothing checked it —
# the same shape as the defect that made all ten overlap coefficients wrong.
#
# activityDensity() returns its Time.rad vector invisibly, so this compares the two
# derivations end to end without restating camtrapR's formula here. A third copy of
# the rule in a test file would be the very thing being guarded against.
rt_path <- here::here("data", "record_table.rds")
if (file.exists(rt_path)) {
  suppressPackageStartupMessages(library(camtrapR))
  rt_rad <- readRDS(rt_path)

  # Fail closed on a data/ that predates the columns under test. Without this, a
  # record_table.rds written before the solar frame existed makes rt_rad$time_solar_rad
  # NULL, the comparison below runs over numeric(0), and max() returns -Inf — which
  # fails the assertion for the wrong reason and reads as a broken frame rather than a
  # stale cache. Measured on 2026-10-06 against a 2026-09-15 cache.
  missing_cols <- setdiff(c("time_rad", "time_solar_rad"), names(rt_rad))
  if (length(missing_cols) > 0) {
    stop(sprintf(paste0(
      "REFUSED: data/record_table.rds is stale -- missing column(s): %s.\n",
      "  It was written before these columns existed; the frame comparisons below\n",
      "  cannot run against it. Rebuild:\n",
      "    conda run -n pehuen-analysis Rscript R/01_load_data.R"),
      paste(missing_cols, collapse = ", ")), call. = FALSE)
  }

  png(tempfile())  # activityDensity needs a device; we want the return value only
  worst <- max(vapply(c("Zorro culpeo", "Liebre", "Puma"), function(sp) {
    cam  <- activityDensity(recordTable = rt_rad, species = sp, allSpecies = FALSE,
                            writePNG = FALSE, plotR = TRUE, speciesCol = "Species",
                            recordDateTimeCol = "DateTimeOriginal")
    ours <- rt_rad$time_rad[rt_rad$Species == sp]
    max(abs(sort(cam) - sort(ours)))
  }, numeric(1)))
  invisible(dev.off())
  check(sprintf("camtrapR derives the same radians we do IN THE CLOCK FRAME (worst %.1e)", worst),
        worst < 1e-9)

  # The complement, and it is the assertion the solar frame makes necessary.
  # camtrapR recomputes from DateTimeOriginal, so it is permanently on the clock —
  # there is no radians entry point and the solar frame CANNOT be handed to it.
  # If someone ever "fixes" that by writing solar times into DateTimeOriginal, the
  # check above would still pass while every camtrapR panel silently changed
  # meaning. This fails instead: the two columns must be measurably different
  # frames, and 03/04's solar figures are ggplot-only for exactly this reason.
  worst_solar <- max(vapply(c("Zorro culpeo", "Liebre", "Puma"), function(sp) {
    cam  <- activityDensity(recordTable = rt_rad, species = sp, allSpecies = FALSE,
                            writePNG = FALSE, plotR = TRUE, speciesCol = "Species",
                            recordDateTimeCol = "DateTimeOriginal")
    sol  <- rt_rad$time_solar_rad[rt_rad$Species == sp]
    max(abs(sort(cam) - sort(sol)))
  }, numeric(1)))
  check(sprintf("...and NOT the solar ones -- the frames are distinct (worst %.3f rad)",
                worst_solar),
        worst_solar > 1e-3)
  dev.off()
} else {
  cat("  skip data/record_table.rds absent -- run R/01_load_data.R\n")
}


cat("against the real record table\n")
if (file.exists(rt_path)) {
  rt <- readRDS(rt_path)
  tb <- split(rt$time_rad, rt$Species)
  t1 <- tb[["Zorro culpeo"]]; t2 <- tb[["Liebre"]]
  set.seed(1); real <- estimate_overlap(t1, t2, n_boot = 100)
  check("Zorro culpeo x Liebre selects Dhat4 (161 / 129)", real$estimator == "Dhat4")
  check("...and agrees with the number camtrapR prints in the plot panel",
        isTRUE(all.equal(real$estimate,
                         unname(overlapEst(t1, t2, type = "Dhat4")),
                         tolerance = 1e-12)))
  # The published CSV must not drift from the helper that wrote it.
  csv_path <- here::here("data", "overlap_stats.csv")
  if (file.exists(csv_path)) {
    csv <- read.csv(csv_path, fileEncoding = "UTF-8")
    # The file carries both frames since 2026-10-05, so every lookup must say which.
    # `real` above was computed from rt$time_rad, i.e. the clock frame.
    check("overlap_stats.csv declares a frame for every row",
          "frame" %in% names(csv) && !any(is.na(csv$frame)) &&
          setequal(unique(csv$frame), c("clock", "solar")))
    check("both frames cover the same ten pairs",
          {
            k <- function(f) sort(paste(csv$sp1[csv$frame == f], csv$sp2[csv$frame == f]))
            length(k("clock")) == 10 && identical(k("clock"), k("solar"))
          })
    row <- csv[csv$frame == "clock" & csv$sp1 == "Zorro culpeo" & csv$sp2 == "Liebre", ]
    check("overlap_stats.csv carries that same estimate (clock frame)",
          nrow(row) == 1 && isTRUE(all.equal(row$estimate, real$estimate, tolerance = 1e-6)))
    check("the frame changes the number -- it is not a duplicated block",
          {
            cl <- csv[csv$frame == "clock", ]
            so <- csv[csv$frame == "solar", ]
            cl <- cl[order(cl$sp1, cl$sp2), ]; so <- so[order(so$sp1, so$sp2), ]
            max(abs(cl$estimate - so$estimate)) > 1e-3
          })
    check("n is a property of the records, not of the frame",
          {
            cl <- csv[csv$frame == "clock", ]; so <- csv[csv$frame == "solar", ]
            cl <- cl[order(cl$sp1, cl$sp2), ]; so <- so[order(so$sp1, so$sp2), ]
            identical(cl$n1, so$n1) && identical(cl$n2, so$n2) &&
              identical(cl$estimator, so$estimator)
          })
    check("every published CI lies inside [0, 1]",
          all(csv$ci_low >= 0) && all(csv$ci_high <= 1))
    check("every published CI contains its point estimate",
          all(csv$ci_low <= csv$estimate & csv$estimate <= csv$ci_high))
  } else {
    cat("  skip data/overlap_stats.csv absent -- run R/04_temporal_overlap.R\n")
  }
} else {
  cat("  skip data/record_table.rds absent -- run R/01_load_data.R\n")
}

if (.failures > 0) {
  cat(sprintf("\n%d test(s) FAILED\n", .failures))
  quit(save = "no", status = 1L)
}
cat("\nall tests passed\n")
