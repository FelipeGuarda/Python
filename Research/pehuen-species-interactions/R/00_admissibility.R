# 00_admissibility.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Owns ONE decision: which records are admissible for which kind of question,
#   and what the unit of analysis is.
#
# WHY THIS FILE EXISTS
#   `01_load_data.R` used to end with `filter(!is.na(datetime))`, and every
#   downstream script inherited that filter without asking for it. That is correct
#   for activity and overlap analysis, which need a trustworthy clock — and WRONG
#   for presence/absence, which needs only a resolved station.
#
#   The cost was measured on 2026-08-20: puma is recorded at 8 stations across the
#   three campaigns, and the spatial maps showed 6. CT03 (7 images) and CT18 (2)
#   vanished because their cameras' clocks failed. The camera identity was never in
#   doubt for any of those frames — only the timestamp was.
#
#   A second, independent error sat next to it: the maps counted IMAGES. A camera
#   fires 2–3 frames per trigger, so an image count is a burst-length artefact, not
#   an abundance signal. Bos taurus is 579 images and 19 episodes; Lepus europaeus is
#   389 images and 129 episodes. Ranking stations or species by image count ranks
#   them partly by how long each animal lingered in frame.
#
# THE RULE
#   Two questions, two admissibility rules, both explicit at the call site:
#
#     admissible(records, "place")  keeps every identified record. A frame with a
#                                  broken clock still proves the animal was there.
#     admissible(records, "time")   keeps records with a trustworthy timestamp.
#                                  Required for anything using date, hour or season.
#
#   Two units, never mixed:
#
#     presence(records)   one row per (campaign, station, species) — the set of
#                         places a species was seen. Uses "place".
#     episodes(records)   one row per independent detection event, 30-minute rule.
#                         Uses "time", because independence is undefined without a
#                         clock. THIS is the unit for any count.
#
#   Raw `records` is one row per IMAGE and should not be counted directly. If you
#   find yourself writing count(records, ...) ask whether you meant episodes().
#
# REQUIRES    dplyr (for `independent()`). Source AFTER library(dplyr).
# SOURCED BY  01, 02, 03, 05, 06. In 01, source it after here::i_am().
# ─────────────────────────────────────────────────────────────────────────────

# The 30-minute independence rule. DUPLICATED DECISION, deliberately named:
# `camera-traps/Anual-reports/2025/py/01_data_prep.py` holds `EPISODE_GAP` for the
# annual report. Two languages, so no shared constant is possible — but if one moves
# the other must move with it, and the report is the one to match.
EPISODE_GAP_MINUTES <- 30


admissible <- function(records, for_question = c("time", "place"), quiet = FALSE) {
  for_question <- match.arg(for_question)

  if (!"time_admissible" %in% names(records)) {
    stop("`records` has no `time_admissible` column — it predates ",
         "R/00_admissibility.R. Re-run R/01_load_data.R.", call. = FALSE)
  }

  out <- switch(
    for_question,
    # Every identified record. `station_id` must be resolved; a record with no
    # station cannot answer a question about place either.
    place = records[!is.na(records$station_id), , drop = FALSE],
    # A trustworthy clock. `time_admissible` is set in 01_load_data.R from the
    # canonical table's own validity flags, not re-derived here.
    time  = records[records$time_admissible & !is.na(records$datetime), , drop = FALSE]
  )

  if (!quiet) {
    dropped <- nrow(records) - nrow(out)
    if (dropped > 0) {
      lost_st <- setdiff(unique(records$station_id[!is.na(records$station_id)]),
                         unique(out$station_id))
      message(sprintf(
        "  admissible(for='%s'): %d of %d records kept (%d dropped%s).",
        for_question, nrow(out), nrow(records), dropped,
        if (length(lost_st)) sprintf(", removing station(s) %s entirely",
                                     paste(sort(lost_st), collapse = ", ")) else ""
      ))
    }
  }
  out
}


presence <- function(records, quiet = FALSE) {
  r <- admissible(records, "place", quiet = quiet)
  # distinct(), not count(): presence is a set. Whether an animal appeared once or
  # two hundred times at a station is a different question, and episodes() answers it.
  out <- unique(r[, c("campaign", "station_id", "species_label", "guild"), drop = FALSE])
  out <- out[order(out$campaign, out$station_id, out$species_label), , drop = FALSE]
  rownames(out) <- NULL
  out
}


# THE independence rule, in one place.
#
# The gap is measured from the last RETAINED detection, not from the immediately
# previous one. That distinction is not cosmetic: detections at 0, 20 and 40 minutes
# are TWO independent events (0 and 40) under this rule and ONE if you compare each
# record only against its predecessor. This is the standard camtrapR definition and
# the one `record_table` has always used.
keep_after_min_gap <- function(datetimes, min_delta_min) {
  n <- length(datetimes)
  if (n == 0) return(logical(0))
  keep <- logical(n)
  keep[1] <- TRUE
  last_kept <- datetimes[1]
  if (n >= 2) {
    for (i in seq(2, n)) {
      if (as.numeric(difftime(datetimes[i], last_kept, units = "mins")) >= min_delta_min) {
        keep[i] <- TRUE
        last_kept <- datetimes[i]
      }
    }
  }
  keep
}


# Grouping includes CAMPAIGN as well as station and species. Two campaigns at one
# station are separate deployments months apart; independence within one says nothing
# about the other.
independent <- function(df, gap_minutes = EPISODE_GAP_MINUTES) {
  df %>%
    dplyr::arrange(station_id, species_label, campaign, datetime) %>%
    dplyr::group_by(station_id, species_label, campaign) %>%
    dplyr::mutate(.keep_event = keep_after_min_gap(datetime, gap_minutes)) %>%
    dplyr::ungroup() %>%
    dplyr::filter(.keep_event) %>%
    dplyr::select(-.keep_event)
}


episodes <- function(records, gap_minutes = EPISODE_GAP_MINUTES, quiet = FALSE) {
  r <- admissible(records, "time", quiet = quiet)
  if (nrow(r) == 0) return(r[0, , drop = FALSE])
  out <- independent(r, gap_minutes)
  rownames(out) <- NULL
  out
}


episode_counts <- function(records, by = c("station_id", "species_label"), ...) {
  e <- episodes(records, ...)
  if (nrow(e) == 0) {
    empty <- as.data.frame(setNames(rep(list(character()), length(by)), by))
    empty$n_episodes <- integer()
    return(empty)
  }
  e$.one <- 1L
  agg <- aggregate(list(n_episodes = e$.one),
                   by = e[, by, drop = FALSE], FUN = length)
  agg[order(-agg$n_episodes), , drop = FALSE]
}
