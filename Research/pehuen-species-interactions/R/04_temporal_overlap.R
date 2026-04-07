# 04_temporal_overlap.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Estimate pairwise temporal overlap between focal species pairs using the
#   Dhat4 estimator (Ridout & Linkie 2009), reproducing the approach in
#   Figure 3 of the reference paper.
#
#   Two complementary outputs:
#     A) Per-pair overlay plots via camtrapR::activityOverlap() — one figure
#        per species pair with overlapping kernel density curves and the Dhat4
#        coefficient printed on the plot.  Saved to figures/overlap_pairs/.
#
#     B) Summary dot-plot via overlap + ggplot2 — camtrapR does not return a
#        data frame with bootstrap CIs, so we compute Dhat4 + 95% CI ourselves
#        and plot all pairs on a single summary figure (Fig 3 equivalent).
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
# OUTPUT  figures/overlap_pairs/<sp1>_<sp2>.png   (one per pair, camtrapR)
#         figures/04_overlap_summary.png           (Dhat4 dot-plot with CI)
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. Libraries ─────────────────────────────────────────────────────────────

library(here)
library(dplyr)
library(ggplot2)
library(overlap)    # overlapEst(), bootEst(), bootCI() for bootstrap summary
library(camtrapR)   # activityOverlap() for per-pair overlay plots

here::i_am("R/04_temporal_overlap.R")
dir.create(here("figures"), showWarnings = FALSE)
dir.create(here("figures", "overlap_pairs"), showWarnings = FALSE)

set.seed(42)  # reproducible bootstrap


# ── 1. Load data ─────────────────────────────────────────────────────────────

record_table <- readRDS(here("data", "record_table.rds"))  # camtrapR format
records      <- readRDS(here("data", "records_all.rds"))   # for bootstrap computation

SPECIES_COLORS <- c(
  "Puma"         = "#1b2a69",
  "Guina"        = "#2c7bb6",
  "Zorro culpeo" = "#74add1",
  "Jabali"       = "#d73027",
  "Liebre"       = "#fc8d59",
  "Perro"        = "#fee090"
)

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


# ── 3. Per-pair overlay plots — camtrapR::activityOverlap() ──────────────────
# activityOverlap() draws two kernel density curves on a shared 24-hour axis,
# shades the overlapping area, and prints the Dhat4 coefficient on the plot.
# It uses the `overlap` package internally — same estimator as our summary step.
#
# Key arguments:
#   recordTable        — the camtrapR-format record table
#   speciesA / speciesB— the two species to compare (must match $Species column)
#   overlapEstimator   — which Dhat estimator to use (we use Dhat4 throughout)
#   writePNG = TRUE    — save the figure to disk
#   plotR    = FALSE   — do not open an interactive graphics window
#   plotDirectory      — destination folder for saved PNGs
#
# The function invisibly returns a named vector of overlap estimates
# (Dhat1, Dhat4, Dhat5) — we capture Dhat4 here so it can feed into the
# summary table in step 4.

message("Generating per-pair overlap plots (camtrapR)...")

camtrapr_dhat4 <- list()  # collect point estimates returned by camtrapR

for (pair in PAIRS) {
  sp1 <- pair[1]
  sp2 <- pair[2]

  n1 <- sum(record_table$Species == sp1)
  n2 <- sum(record_table$Species == sp2)

  if (n1 == 0 || n2 == 0) {
    warning(sprintf("No records for %s or %s — skipping pair.", sp1, sp2))
    next
  }

  # activityOverlap() saves one PNG named "<speciesA>_<speciesB>.png"
  estimates <- activityOverlap(
    recordTable       = record_table,
    speciesA          = sp1,
    speciesB          = sp2,
    writePNG          = TRUE,
    plotR             = FALSE,
    plotDirectory     = here("figures", "overlap_pairs"),
    overlapEstimator  = "Dhat4",
    speciesCol        = "Species",
    recordDateTimeCol = "DateTimeOriginal"
  )

  # Store the Dhat4 point estimate for comparison with our bootstrap results
  pair_key <- paste(sp1, sp2, sep = "_")
  camtrapr_dhat4[[pair_key]] <- estimates["Dhat4"]

  message(sprintf("  %s × %s  Dhat4 = %.3f", sp1, sp2, estimates["Dhat4"]))
}

message("Saved per-pair overlap plots to figures/overlap_pairs/")


# ── 4. Bootstrap CI for each pair — overlap package ──────────────────────────
# camtrapR's activityOverlap() does not compute confidence intervals.
# We compute them here using the bootstrap procedure from the overlap package:
#   (a) bootEst() resamples each species' time vector 1000 times with
#       replacement and computes Dhat4 on each resample pair.
#   (b) bootCI() derives the 0.025 and 0.975 quantiles as the 95% CI.
#
# The Dhat4 point estimates here should match the camtrapR values above
# (they use the same estimator).  Any tiny numerical difference is due to
# floating-point precision; both are correct.
#
# We flag pairs where either sample is < 75 records: for small samples, Dhat1
# is the more conservative estimator (Ridout & Linkie 2009), but we report
# Dhat4 to stay consistent with the reference paper.

N_BOOT <- 1000

overlap_results <- lapply(PAIRS, function(pair) {
  sp1 <- pair[1]
  sp2 <- pair[2]
  t1  <- times_by_species[[sp1]]
  t2  <- times_by_species[[sp2]]

  n1 <- length(t1)
  n2 <- length(t2)

  if (n1 == 0 || n2 == 0) return(NULL)

  # (a) Point estimate
  dhat4 <- overlapEst(t1, t2, type = "Dhat4")

  # (b) Bootstrap CI
  boot <- bootEst(t1, t2, nb = N_BOOT, type = "Dhat4")
  ci   <- bootCI(dhat4, boot, conf = 0.95)

  data.frame(
    sp1          = sp1,
    sp2          = sp2,
    n1           = n1,
    n2           = n2,
    dhat4        = dhat4,
    ci_low       = ci["norm0", "lower"],
    ci_high      = ci["norm0", "upper"],
    pair_label   = paste(sp1, "\u00d7", sp2),   # × character
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

# Print summary to console for quick inspection
message("\nOverlap coefficients (Dhat4 + 95% bootstrap CI):")
print(overlap_df %>% select(pair_label, n1, n2, dhat4, ci_low, ci_high, small_sample))


# ── 5. Figure: Dhat4 summary dot-plot (Fig 3 equivalent) ─────────────────────
# One row per species pair, ordered by Dhat4 descending within each guild type.
# Pairs with small-sample warnings are shown as hollow points.
# A vertical dashed line at 0.5 marks the "moderate overlap" threshold.

overlap_df <- overlap_df %>%
  arrange(guild_type, desc(dhat4)) %>%
  mutate(pair_label = factor(pair_label, levels = rev(unique(pair_label))))

fig_summary <- ggplot(overlap_df,
                      aes(x = dhat4, y = pair_label, colour = guild_type)) +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high),
                 height = 0.3, linewidth = 0.8) +
  geom_point(aes(shape = small_sample), size = 3.5) +
  scale_shape_manual(
    values = c(`FALSE` = 16, `TRUE` = 1),
    labels = c(`FALSE` = "n \u2265 75 (both)", `TRUE` = "n < 75 (one or both)"),
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
message("Run 05_spatial_distribution.R next.")
