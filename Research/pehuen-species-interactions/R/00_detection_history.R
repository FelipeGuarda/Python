# 00_detection_history.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Owns ONE decision: how the field record becomes a station x occasion detection
#   history for an occupancy model -- which station-days count as surveyed, how they
#   group into occasions, and when a cell is 1, 0 or NA. Nothing else in this project
#   may build an occasion grid; B1 (single-season occupancy), B2 (dynamic occupancy)
#   and 07 (the Rota power check) all read it from here.
#
# WHAT A CELL MEANS
#   1   the species was recorded at least once in that occasion at that station
#   0   the station was surveyed on at least one day of the occasion and the species
#       was not recorded
#   NA  the station was not surveyed on any day of the occasion. This is NOT a zero:
#       MacKenzie et al. (2003, "Missing observations") enter unsurveyed occasions as
#       missing, and a zero there would be an absence nobody observed.
#
#   "Surveyed" is effort_admissible(deployments, "detections") (R/00_admissibility.R):
#   stills in the canonical table and a clock that allows effort. Video-only stations
#   were recording but their media is not readable here, and a broken clock cannot
#   place a detection in an occasion, so both are NA rather than 0. Measured
#   2026-10-06: 0 of 380 time-admissible episodes fall in a deployment this rule
#   excludes, so the NA rule costs no detection today; tests/test_detection_history.R
#   asserts that, so a future campaign that breaks it fails by name.
#
# A SEASON IS REQUIRED
#   A single-season occupancy model assumes CLOSURE: each station is used or unused
#   for the whole run of occasions (MacKenzie et al. 2002; Burton et al. 2015 call it
#   hard to defend over weeks or months). The record is 19 months long. A grid over
#   all of it fed to occu() would read a station culpeo used in winter and left in
#   summer as one closed state, and nothing in the output would show it. So there is
#   no whole-record default: `season` is a season_start Date (R/00_seasons.R) and a
#   call without one refuses. Occasions start on the season's first day and never
#   cross its last; B2 builds one grid per season and stacks them.
#
# OCCASION LENGTH -- 14 DAYS, MEASURED (2026-10-06)
#   A null psi(.)p(.) fit at 1/3/5/7/10/14 days per species x season (methods menu
#   §B0.1) showed: psi barely moves with occasion length (culpeo Invierno 2025: 0.53,
#   0.52, 0.54, 0.52, 0.56 from 3 to 14 d), because the chance of detecting a species
#   at least once over the season is nearly fixed (culpeo: 81% at 7 d, 77% at 14 d).
#   Length re-packages the same detections; it does not add information. 14 d is the
#   length at which per-occasion p for culpeo and liebre reaches ~0.2-0.35, inside the
#   1-15 day range Burton et al. (2015) report (median 5), leaving ~6.5 occasions per
#   season. 7 d is the sensitivity run. A season's last occasion is usually partial
#   (a 92-day season is six 14-day occasions plus 8 days); `effort` carries the
#   surveyed days per cell so a model can use them as a detection covariate.
#
# WHY NOT camtrapR::detectionHistory()
#   The methods menu names cameraOperation() -> detectionHistory(). Both are built
#   for one setup-to-retrieval window per camera, and have no notion of a season
#   window or of our effort rule; reproducing season-anchored occasions through them
#   needs a campaign-as-camera workaround and a column-subset of the operation
#   matrix, with seven column/timezone arguments that every caller would have to get
#   right. The grid is a few lines of day arithmetic and is asserted cell by cell in
#   tests/test_detection_history.R instead. The output has the shape unmarked wants:
#   unmarkedFrameOccu(y = h$y, obsCovs = list(effort = h$effort)).
#
# REQUIRES    nothing beyond base R. Sources R/00_admissibility.R and
#             R/00_seasons.R itself, so a caller needs only this file.
# SOURCED BY  (B1, B2, 07 -- not yet written). Source it after here::i_am().
# ─────────────────────────────────────────────────────────────────────────────

source(here::here("R", "00_admissibility.R"))
source(here::here("R", "00_seasons.R"))

OCCASION_DAYS <- 14L


# records_all.rds, deployments.rds, one species_label, one season_start Date ->
#   list(y         station x occasion matrix of 1 / 0 / NA,
#        effort    station x occasion matrix of surveyed days (NA where y is NA),
#        occasions data.frame(occasion, start, end) -- `end` is the occasion's last
#                  day, inclusive)
# Rows are every station surveyed on at least one day of the season, sorted; a
# station with no surveyed day in the season carries no information and has no row.
detection_history <- function(records, deployments, species, season,
                              occasion_days = OCCASION_DAYS, quiet = FALSE) {
  if (missing(season) || is.null(season) || length(season) != 1L || is.na(season)) {
    stop("`season` is required: a single-season occupancy model assumes closure, ",
         "which the 19-month record does not satisfy. Pass one season_start Date ",
         "(R/00_seasons.R), e.g. as.Date(\"2025-06-01\").", call. = FALSE)
  }
  season <- as.Date(season)
  if (season_start(season) != season) {
    stop(sprintf("`season` %s is not the first day of a season period; did you mean %s?",
                 season, season_start(season)), call. = FALSE)
  }
  occasion_days <- as.integer(occasion_days)
  if (length(occasion_days) != 1L || is.na(occasion_days) || occasion_days < 1L) {
    stop("`occasion_days` must be one positive whole number.", call. = FALSE)
  }
  known <- sort(unique(records$species_label))
  if (length(species) != 1L || !species %in% known) {
    stop(sprintf("`species` must be one of: %s.", paste(known, collapse = ", ")),
         call. = FALSE)
  }

  # The season's days, found by the season rule itself rather than by restating where
  # the next season begins -- an astronomical boundary table would change that.
  span <- seq(season, season + 100L, by = "day")
  season_days <- span[season_start(span) == season]

  # Surveyed station-days. The producer's window is half-open (field_days ==
  # field_end - field_start; R/00_seasons.R), so the last day is not a survey day.
  # Consecutive campaigns share their changeover day, hence unique().
  dep <- deployments[effort_admissible(deployments, "detections") &
                       !is.na(deployments$field_start) & !is.na(deployments$field_end) &
                       deployments$field_end > deployments$field_start, , drop = FALSE]
  surveyed <- unique(do.call(rbind, c(
    list(data.frame(station_id = character(), day = as.Date(character()))),
    lapply(seq_len(nrow(dep)), function(i) {
      days <- seq(dep$field_start[i], dep$field_end[i] - 1L, by = "day")
      days <- days[days %in% season_days]
      if (length(days)) data.frame(station_id = dep$station_id[i], day = days)
    }))))

  n_occ    <- ceiling(length(season_days) / occasion_days)
  occ_of   <- function(day) as.integer(day - season) %/% occasion_days + 1L
  starts   <- season + (seq_len(n_occ) - 1L) * occasion_days
  occasions <- data.frame(
    occasion = seq_len(n_occ),
    start    = starts,
    end      = pmin(starts + occasion_days - 1L, max(season_days))
  )

  stations <- sort(unique(surveyed$station_id))
  dims     <- list(stations, paste0("o", seq_len(n_occ)))
  effort   <- unclass(table(factor(surveyed$station_id, levels = stations),
                            factor(occ_of(surveyed$day), levels = seq_len(n_occ))))
  effort   <- matrix(as.integer(effort), nrow(effort), ncol(effort), dimnames = dims)
  effort[effort == 0L] <- NA_integer_
  y <- ifelse(is.na(effort), NA_integer_, 0L)
  dimnames(y) <- dims

  # Detections: one episode is enough to make a cell 1, so the unit does not change
  # the history; episodes() is used because it is the time-admissible unit.
  e <- episodes(records[records$species_label == species, , drop = FALSE], quiet = quiet)
  e_day <- as.Date(e$datetime)
  e <- e[e_day %in% season_days, , drop = FALSE]
  e_day <- e_day[e_day %in% season_days]
  on_surveyed_day <- paste(e$station_id, e_day) %in% paste(surveyed$station_id, surveyed$day)
  if (any(!on_surveyed_day) && !quiet) {
    message(sprintf(
      "  detection_history(%s, %s): %d episode(s) on a day that is not surveyed are left out: %s.",
      species, season, sum(!on_surveyed_day),
      paste(sort(unique(paste0(e$station_id[!on_surveyed_day], "/",
                               e_day[!on_surveyed_day]))), collapse = ", ")))
  }
  if (any(on_surveyed_day)) {
    y[cbind(match(e$station_id[on_surveyed_day], stations),
            occ_of(e_day[on_surveyed_day]))] <- 1L
  }

  list(y = y, effort = effort, occasions = occasions)
}
