# 00_accompaniment.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Owns ONE decision: whether an animal's visit to a camera happened in the
#   company of people -- a person or a vehicle at the same station, in the same
#   campaign, in the frame or within a time window of the visit.
#
# WHY IT EXISTS
#   Almost every Perro at Bosque Pehuén is the rangers' dog, and many Caballo are
#   ridden or led. Counted as wildlife detections they inflate "invasive" activity
#   with the team's own movements. This file says which visits were the team's.
#
# TWO KINDS OF EVIDENCE, kept apart in the output
#   in the frame  `with_human` / `with_vehicle`, the co-presence tags the reviewer
#                 ticks on the animal's own image. The producer labels each image
#                 once (animal > person > vehicle), so a dog walking beside its
#                 handler is labelled Perro and the handler exists ONLY as a tag.
#                 NA = that campaign/species was not checked, not "no person".
#   nearby        a frame labelled human/vehicle at the same station-campaign within
#                 ACCOMPANIMENT_WINDOW_MIN of the episode's span. Catches the person
#                 who triggered a separate frame; cannot see the one in the frame.
#   Until the tags are published, `accompanied` rests on proximity alone and is a
#   LOWER BOUND.
#
# THE WINDOW IS NOT THE EPISODE GAP
#   Both are 30 minutes today. EPISODE_GAP_MINUTES (the producer's) decides when two
#   detections of one animal are one visit; ACCOMPANIMENT_WINDOW_MIN decides how far
#   a person may be from that visit and still count as company. Different decisions,
#   so different names: changing one must never silently change the other.
#
# REQUIRES    R/00_admissibility.R sourced first (admissible(), EPISODE_COLUMN).
# SOURCED BY  09_domestic_accompaniment.R
# ─────────────────────────────────────────────────────────────────────────────

ACCOMPANIMENT_WINDOW_MIN <- 30


# One row per time-admissible episode of `species_latin`, from `records` shaped like
# records_identified.rds, against `people` shaped like human_activity.rds.
#
#   episode, campaign, station_id, start, end, hour   the visit, first to last frame
#   in_frame_human, in_frame_vehicle   any frame tagged; NA if none of its frames was checked
#   gap_min       minutes from the span to the nearest person/vehicle frame at that
#                 station-campaign (0 if inside it); NA if the team never appears there
#   near_human, near_vehicle           a frame of that kind within window_min
#   accompanied   any of the four is TRUE. FALSE means "no evidence", not "alone".
accompanied <- function(records, people, species_latin,
                        window_min = ACCOMPANIMENT_WINDOW_MIN) {
  r <- admissible(records[records$species_latin %in% species_latin, , drop = FALSE],
                  "time", quiet = TRUE)
  p <- admissible(people, "time", quiet = TRUE)
  if (nrow(r) == 0) return(.empty_accompaniment())

  spans <- lapply(split(r, r[[EPISODE_COLUMN]]), function(e) {
    data.frame(
      episode          = e[[EPISODE_COLUMN]][1],
      campaign         = e$campaign[1],
      station_id       = e$station_id[1],
      start            = min(e$datetime),
      end              = max(e$datetime),
      in_frame_human   = .any_tag(e$with_human),
      in_frame_vehicle = .any_tag(e$with_vehicle),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, spans)
  out$hour <- as.POSIXlt(out$start)$hour

  near <- t(vapply(seq_len(nrow(out)), function(i) {
    here <- p[p$campaign == out$campaign[i] & p$station_id == out$station_id[i], ]
    if (nrow(here) == 0) return(c(gap = NA_real_, human = 0, vehicle = 0))
    # Distance from a frame to an interval: zero inside it, else to the nearer end.
    d <- pmax(as.numeric(difftime(out$start[i], here$datetime, units = "mins")),
              as.numeric(difftime(here$datetime, out$end[i], units = "mins")), 0)
    inside <- d <= window_min
    c(gap = min(d),
      human   = any(inside & here$subject == "human"),
      vehicle = any(inside & here$subject == "vehicle"))
  }, numeric(3)))
  out$gap_min      <- near[, "gap"]
  out$near_human   <- near[, "human"] == 1
  out$near_vehicle <- near[, "vehicle"] == 1
  out$accompanied  <- out$in_frame_human %in% TRUE | out$in_frame_vehicle %in% TRUE |
                      out$near_human | out$near_vehicle

  out <- out[order(out$campaign, out$station_id, out$start), , drop = FALSE]
  rownames(out) <- NULL
  out
}


# TRUE if any frame carries the tag; FALSE if checked and none does; NA if no frame
# was checked. One checked frame is enough to speak for the episode.
.any_tag <- function(x) {
  if (any(x %in% TRUE)) return(TRUE)
  if (all(is.na(x))) return(NA)
  FALSE
}

.empty_accompaniment <- function() {
  data.frame(episode = character(), campaign = character(), station_id = character(),
             start = as.POSIXct(character(), tz = "UTC"),
             end = as.POSIXct(character(), tz = "UTC"),
             in_frame_human = logical(), in_frame_vehicle = logical(),
             hour = integer(), gap_min = numeric(), near_human = logical(),
             near_vehicle = logical(), accompanied = logical(),
             stringsAsFactors = FALSE)
}
