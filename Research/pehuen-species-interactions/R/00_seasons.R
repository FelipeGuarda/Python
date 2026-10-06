# 00_seasons.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Owns ONE decision: where an austral season begins, and how a station's field
#   window divides among the seasons it spans. Nothing else in this project may
#   decide what season a date belongs to.
#
# WHY THIS FILE EXISTS
#   Every figure in this project used to facet by CAMPAIGN, and a campaign is not a
#   season. Measured from deployments.rds on 2026-09-15:
#
#     otono_2025      2024-10-09 → 2025-06-11   primavera, verano, otoño
#     primavera_2025  2025-05-14 → 2026-01-14   otoño, INVIERNO, primavera, verano
#     otono_2026      2025-11-13 → 2026-05-15   verano, otoño
#
#   A campaign is the interval between two field visits — five to eight months here —
#   and it is named for the season the cards were RETRIEVED in, not the season they
#   recorded. The three windows are contiguous, so the array is one continuous record
#   from 2024-10-09 to 2026-05-15 that was cut into three retrieval intervals. Facets
#   labelled "Otoño 2026" were showing summer and autumn pooled.
#
#   Two things followed from that, and both were being read as ecology:
#     - Liebre appears to collapse to 2 episodes in otoño 2026. It does not: that
#       window contains no winter and no spring, and 109 of liebre's 129 episodes are
#       winter or spring. The collapse is the window, not the hare.
#     - 06_seasonal_detection_maps.R stated "Invierno — no field deployment yet", and
#       the methods menu built §A3 and §B2 on a missing winter. Winter is sampled: 95
#       of 96 winter episodes sit inside the primavera_2025 window.
#
# THE KEY IS A DATE, NOT A NAME
#   A season period is identified by `season_start`, the Date its first day falls on.
#   Join on it, sort on it, group by it. It is unambiguous across the year boundary
#   (Verano 2024-25 starts 2024-12-01), it sorts chronologically for free, and it
#   cannot be confused with a campaign slug — `primavera_2025` names a campaign that
#   recorded winter, and a season period that did not exist then.
#
#   `season_label()` renders a period for a human and returns an ORDERED factor;
#   never join on it. Level sets follow the data they were built from.
#
# THE BOUNDARY RULE IS ONE TABLE
#   SEASON_BOUNDARIES below is the whole rule. Moving from calendar months to
#   solstice/equinox dates — under discussion, and defensible at 38°S where
#   photoperiod is what the seasons are a proxy for, and what activity::transtime()
#   anchors on — is an edit to that table's `day` column and nothing else. No
#   function signature changes, no caller changes, and `season_effort()`'s day
#   arithmetic is boundary-agnostic. Astronomical boundaries drift a day or two
#   between years, so that version needs a per-year table rather than (month, day);
#   the shape is the same and the callers still do not change.
#
# EFFORT IS SPLIT, NEVER RE-DECIDED
#   `season_effort()` divides each deployment's field window across the periods it
#   spans and carries `media_status` and `valid_effort` through untouched. WHICH
#   statuses belong in which denominator is 02_detection_summary.R's decision (see
#   the table in the README) and is not restated here.
#
# REQUIRES    nothing beyond base R.
# SOURCED BY  02, 05, 06, 00_detection_history. Source it after here::i_am().
# ─────────────────────────────────────────────────────────────────────────────


# The first day of each austral season, as (month, day). THIS TABLE IS THE RULE.
# `month` must stay distinct across the four rows: season_of() identifies a period by
# the month its start falls in, which holds for calendar and astronomical boundaries
# alike.
SEASON_BOUNDARIES <- data.frame(
  season = c("Verano", "Otoño", "Invierno", "Primavera"),
  month  = c(12L,      3L,      6L,         9L),
  day    = c(1L,       1L,      1L,         1L),
  stringsAsFactors = FALSE
)

# Display order within a year, starting at spring. Kept identical to the order
# 06_seasonal_detection_maps.R used before this file existed, so figures that pool
# across years read the same as they did.
SEASON_LEVELS <- c("Primavera", "Verano", "Otoño", "Invierno")


# ── The period a date belongs to ─────────────────────────────────────────────

# Date/POSIXct vector -> Date vector: the first day of the season period containing
# each element. NA in, NA out. This is the join key for everything seasonal.
season_start <- function(x) {
  d   <- as.Date(x)
  out <- structure(rep(NA_real_, length(d)), class = "Date")
  ok  <- !is.na(d)
  if (!any(ok)) return(out)

  # One year of slack each side so findInterval() can never fall off the front.
  yrs <- seq(min(as.integer(format(d[ok], "%Y"))) - 1L,
             max(as.integer(format(d[ok], "%Y"))) + 1L)
  # format= is explicit so as.Date() does not try candidate formats through
  # strptime(), which reads the process timezone. This R has no zone database and
  # falls back to the OS zone; a boundary must not depend on which machine ran it.
  bounds <- sort(as.Date(unlist(lapply(yrs, function(y) {
    sprintf("%04d-%02d-%02d", y, SEASON_BOUNDARIES$month, SEASON_BOUNDARIES$day)
  })), format = "%Y-%m-%d"))

  out[ok] <- bounds[findInterval(d[ok], bounds)]
  out
}


# Date/POSIXct vector -> factor over SEASON_LEVELS. The season without its year;
# use it to pool across years, and season_start() when the year matters.
season_of <- function(x) {
  starts <- season_start(x)
  months <- as.integer(format(starts, "%m"))
  factor(SEASON_BOUNDARIES$season[match(months, SEASON_BOUNDARIES$month)],
         levels = SEASON_LEVELS)
}


# Date/POSIXct vector -> ordered factor of labels, chronological. Accepts any date
# inside a period or the period's start, so season_label(datetime) and
# season_label(season_start) agree.
#
# Verano is labelled "Verano 2025-26" because it straddles the new year: a bare
# "Verano 2025" would be read as January 2025 by half the readers and December 2025
# by the other half.
season_label <- function(x) {
  starts <- season_start(x)
  years  <- as.integer(format(starts, "%Y"))
  names  <- as.character(season_of(starts))

  txt <- ifelse(
    names == "Verano",
    sprintf("Verano %d-%02d", years, (years + 1L) %% 100L),
    sprintf("%s %d", names, years)
  )
  txt[is.na(starts)] <- NA_character_

  ord <- unique(starts[!is.na(starts)])
  factor(txt, levels = txt[match(sort(ord), starts)], ordered = TRUE)
}


# ── Effort, split across the periods a deployment spans ──────────────────────

# deployments.rds -> one row per (campaign, station, season period) with the days of
# that deployment's field window that fall inside the period.
#
# The producer's convention is half-open: field_days == field_end - field_start, so
# the last day is not counted. The split uses the same convention, which makes
# sum(effort_days) == field_days exact for every deployment — asserted in
# tests/test_seasons.R rather than trusted.
#
# A deployment with no usable window (no dates, or end <= start) contributes no rows.
# It is not an error: media_status already says why, and 02 reports those separately.
season_effort <- function(deployments) {
  needed <- c("campaign", "station_id", "field_start", "field_end",
              "media_status", "valid_effort")
  missing <- setdiff(needed, names(deployments))
  if (length(missing)) {
    stop("`deployments` is missing column(s): ", paste(missing, collapse = ", "),
         ". Re-run R/01_load_data.R.", call. = FALSE)
  }

  empty <- data.frame(
    campaign     = character(), station_id = character(),
    season_start = structure(numeric(), class = "Date"),
    effort_days  = integer(),   media_status = character(),
    valid_effort = logical(),   stringsAsFactors = FALSE
  )
  if (nrow(deployments) == 0) return(empty)

  parts <- lapply(seq_len(nrow(deployments)), function(i) {
    row <- deployments[i, , drop = FALSE]
    a <- as.Date(row$field_start); b <- as.Date(row$field_end)
    if (is.na(a) || is.na(b) || b <= a) return(NULL)

    # Counting the days themselves, rather than differencing boundaries, is why this
    # cannot disagree with field_days: every day is assigned to exactly one period.
    days  <- seq(a, b - 1L, by = "day")
    tally <- table(season_start(days))

    data.frame(
      campaign     = row$campaign,
      station_id   = row$station_id,
      season_start = as.Date(names(tally)),
      effort_days  = as.integer(tally),
      media_status = row$media_status,
      valid_effort = row$valid_effort,
      stringsAsFactors = FALSE
    )
  })

  out <- do.call(rbind, c(list(empty), parts))
  out[order(out$season_start, out$station_id), , drop = FALSE]
}
