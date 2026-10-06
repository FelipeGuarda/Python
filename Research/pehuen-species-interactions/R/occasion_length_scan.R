# occasion_length_scan.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   The evidence behind OCCASION_DAYS (R/00_detection_history.R) and behind the
#   occupancy scope in docs/methods-menu-interactions.md §B0.1: a null single-season
#   psi(.)p(.) fitted per species x season period x occasion length. One-off evidence,
#   not a step in the 01-07 chain, hence unnumbered. Re-run it if the field record,
#   the effort rule or the season boundaries change, and re-read §B0.1 against it.
#
# WHAT IT IS AND IS NOT
#   Maximum likelihood on the standard single-season likelihood (MacKenzie et al.
#   2002) with NA occasions skipped (MacKenzie et al. 2003) -- written out here
#   because `unmarked` is not in the environment yet. No standard errors, no
#   covariates, one p for every station. That is enough to choose an occasion length
#   and to see which species x season cells are identifiable at all; it is not a
#   result to cite. B1 refits with unmarked::occu(), SEs and an effort covariate.
#
#   `boundary` flags a fit that ran to psi -> 1: too few detecting stations to tell
#   absent from missed. Those p values mean nothing. 0.95 is a reading aid, not a test.
#
# INPUT   data/records_all.rds, data/deployments.rds   (01_load_data.R)
# OUTPUT  data/occasion_length_scan.csv   (tracked: a readable result, like
#         overlap_stats.csv). One row per species x season x occasion length.
# ─────────────────────────────────────────────────────────────────────────────

library(here)
here::i_am("R/occasion_length_scan.R")
source(here::here("R", "00_contract.R"))
source(here::here("R", "00_detection_history.R"))

contract_assert_current()

LENGTHS_DAYS <- c(1L, 3L, 5L, 7L, 10L, 14L)
BOUNDARY_PSI <- 0.95

records     <- readRDS(here("data", "records_all.rds"))
deployments <- readRDS(here("data", "deployments.rds"))
seasons     <- sort(unique(season_effort(deployments)$season_start))
species     <- sort(unique(records$species_label))


# Negative log-likelihood of psi(.)p(.) on the logit scale. A station's history
# contributes psi * prod(p^y (1-p)^(1-y)) over its surveyed occasions, plus (1 - psi)
# when it was never detected; NA occasions drop out of the product.
null_nll <- function(par, y) {
  psi <- plogis(par[1]); p <- plogis(par[2])
  -sum(apply(y, 1, function(h) {
    h <- h[!is.na(h)]
    log(psi * prod(p^h * (1 - p)^(1 - h)) + (1 - psi) * (sum(h) == 0))
  }))
}

rows <- list()
for (s in as.list(seasons)) {
  for (L in LENGTHS_DAYS) {
    for (sp in species) {
      h  <- detection_history(records, deployments, sp, s, occasion_days = L, quiet = TRUE)
      y  <- h$y
      k  <- mean(rowSums(!is.na(y)))
      detecting <- sum(rowSums(y, na.rm = TRUE) > 0)
      psi <- p <- NA_real_
      if (detecting > 0) {
        fit <- optim(c(0, -1), null_nll, y = y, method = "BFGS")
        psi <- plogis(fit$par[1]); p <- plogis(fit$par[2])
      }
      rows[[length(rows) + 1L]] <- data.frame(
        season_start     = s,
        season           = as.character(season_label(s)),
        species          = sp,
        occasion_days    = L,
        stations         = nrow(y),
        occasions_mean   = round(k, 1),
        detections       = sum(y, na.rm = TRUE),
        stations_detected = detecting,
        naive_occupancy  = round(detecting / nrow(y), 3),
        psi              = round(psi, 3),
        p                = round(p, 3),
        # Chance a used station is detected at least once in the season.
        p_cumulative     = round(1 - (1 - p)^k, 3),
        boundary         = !is.na(psi) & psi >= BOUNDARY_PSI,
        stringsAsFactors = FALSE
      )
    }
  }
}
scan <- do.call(rbind, rows)

out <- here("data", "occasion_length_scan.csv")
write.csv(scan, out, row.names = FALSE)
message(sprintf("Saved %s: %d fits, %d at the psi -> 1 boundary, %d with no detection.",
                "data/occasion_length_scan.csv", sum(!is.na(scan$psi)),
                sum(scan$boundary), sum(is.na(scan$psi))))

wide <- reshape(scan[!scan$boundary & !is.na(scan$psi) & scan$occasion_days %in% c(5, 7, 10, 14),
                     c("species", "season_start", "season", "occasion_days", "p", "psi")],
                idvar = c("species", "season_start", "season"),
                timevar = "occasion_days", direction = "wide")
wide <- wide[order(wide$species, wide$season_start), setdiff(names(wide), "season_start")]
message("\np per occasion, and psi, at 5 / 7 / 10 / 14 days (boundary fits left out):")
print(wide, row.names = FALSE)
