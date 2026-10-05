# 00_timeofday.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Owns ONE decision: where on the 24-hour circle a detection sits — in either of
#   the two frames of reference this project uses, and what separates them.
#   Nothing else may convert a timestamp into an angle.
#
#     clock frame   the camera's own wall-clock reading, as a position on a fixed
#                   24-hour circle. What every figure in this project used until now.
#     solar frame   the same detection re-expressed relative to that day's sunrise
#                   and sunset at this site. The frame the animal actually uses.
#
# WHY THIS FILE EXISTS
#   Bosque Pehuén is at 39.4°S and the record spans 19 months. Measured at the site
#   with the constants below, over 2025:
#
#     2025-06-20   sunrise 08:07   sunset 17:30   day length  9.4 h
#     2025-12-20   sunrise 05:17   sunset 20:14   day length 15.0 h
#     annual swing      2.9 h           2.8 h              5.6 h
#
#   A species holding a rigid schedule relative to sunrise therefore appears, in
#   pooled clock time, as a peak smeared across nearly three hours. Rowcliffe et al.
#   (2014) state the consequence directly: the progression of sunrise and sunset
#   flattens peaks and OVERESTIMATES activity level; the problem is negligible in the
#   tropics and dramatic over long periods at higher latitudes. Ours is the second
#   case, not the first.
#
#   This is not a new idea in this ecosystem. It is piece 3 of the clock decision
#   taken upstream on 2026-08-14 and recorded in camera-traps/docs/DATA-HEALTH-MANUAL.md
#   §5.3: cameras are never adjusted for civil time BECAUSE "the defensible frame for
#   an activity analysis is solar, and solar time is derived from the instant and the
#   location — both of which survive a constant offset". camera-traps/docs/V2-REVIEW.md
#   lists "the sun-anchored sensitivity run in pehuén (piece 3)" as deliberately out of
#   that review's scope. This file is that piece.
#
# WHY BOTH FRAMES LIVE HERE
#   The hours-to-radians rule used to sit in 01_load_data.R. Leaving it there and
#   adding only the solar half would have put the two frames' definitions in two files,
#   and the whole point of the exercise is comparing them. One module owns "what is an
#   angle here", in both frames, or the comparison is between two things nobody owns.
#
# ONE SITE COORDINATE, NOT TWENTY-SEVEN — MEASURED, NOT ASSUMED
#   Sunrise differs by 15.7 SECONDS between the two extreme corners of the array
#   (lat spread 0.035°, lon spread 0.038°, from camera-traps' estaciones.csv) against
#   an annual swing of 2.9 hours. Per-station coordinates would be false precision at a
#   ratio of about 1:660. So this module reads no station file, holds no station
#   identity, and does not depend on estaciones.geojson or on the contract that
#   describes it. That independence is deliberate: it keeps the solar frame usable
#   without waiting on anything upstream.
#
# THE CIVIL OFFSET IS A DECLARED ASSUMPTION, NOT A MEASURED FACT
#   SOLAR_OFFSET_HOURS says what the camera clock means. The upstream rule is
#   "horario de invierno, no DST correction ever" (§5.3), which makes -4 the right
#   reading. But §5.3 also records that these clocks were adjusted once historically,
#   and repaired segments are anchored to field wall-clock readings taken against
#   phones that DO follow DST. The per-deployment offset — piece 2 of the same
#   decision — is NOT published, so this consumer cannot derive the true instant.
#   -4 is therefore an assumption with a known failure mode. It is one constant, and
#   tests/test_timeofday.R measures what a one-hour error in it is worth so the
#   assumption is bounded rather than trusted. If that bound ever matters, the fix is
#   upstream (publish piece 2), not here.
#
# THE REFERENCE ANCHOR IS PINNED, AND THIS IS THE SUBTLE PART
#   activity::transtime(type = "average") defaults mnanchor to the mean of the anchors
#   it is HANDED. That makes the transformation depend on which rows the caller passed:
#   the same detection moved 42 MINUTES between a full-table call and a 50-row subset
#   when measured on 2026-10-05. 04_temporal_overlap.R computes every pair on a species
#   subset, so each of the ten pairs would have sat on a slightly different clock, with
#   nothing comparing them — the same family of defect as the densityFit() bug and the
#   two time_rad derivations. SOLAR_MNANCHOR pins the reference to the site's full
#   annual cycle, so solar_rad() is a pure function of its argument and subsetting is
#   safe. Do not remove the pin.
#
# REQUIRES    activity (get_suntimes, transtime). Base R otherwise.
# SOURCED BY  01 (builds both columns), 03, 04. Source it after here::i_am().
# ─────────────────────────────────────────────────────────────────────────────

library(activity)   # get_suntimes(), transtime()


# ── The configuration surface — the whole of it ──────────────────────────────

# What the camera clock means. See "THE CIVIL OFFSET" above before changing it.
SOLAR_OFFSET_HOURS <- -4

# Any point inside the array; the array's own extent is worth 16 s of sunrise.
SITE_LAT <- -39.44
SITE_LON <- -71.74

# transtime()'s transformation. "average" rescales the day onto the site's mean
# sunrise and sunset, so the solar axis still reads as local hours; "equinoctial"
# rescales onto a 12/12 day (sunrise 06:00, solar noon 12:00, sunset 18:00), which is
# the presentation Vazquez et al. use for comparison across latitudes and seasons.
# "single" anchors on sunrise alone (Nouvellet et al. 2012). Changing this constant
# changes every solar figure and no caller.
SOLAR_ANCHOR <- "average"

# REFERENCES
#   Nouvellet, P., Rasmussen, G.S.A., Macdonald, D.W. & Courchamp, F. (2012) Noisy
#     clocks and silent sunrises: measurement methods of daily activity pattern.
#     Journal of Zoology 286: 179-184.                           [single anchoring]
#   Vazquez, C., Rowcliffe, J.M., Spoelstra, K. & Jansen, P.A. (2019) Comparing diel
#     activity patterns of wildlife across latitudes and seasons: time transformation
#     using day length. Methods in Ecology and Evolution.        [double anchoring]
#   Rowcliffe, J.M. et al. (2014) Quantifying levels of animal activity using camera
#     trap data. Methods in Ecology and Evolution 5: 1170-1179.  [why it matters]


# ── The pinned reference anchor ──────────────────────────────────────────────
# Mean sunrise and sunset across a complete annual cycle at this site, in radians.
# Computed from a fixed non-leap reference year so the value is a function of the
# constants above and not of the data, the machine, or the run date.
SOLAR_REFERENCE_YEAR <- 2025L

SOLAR_MNANCHOR <- local({
  days <- seq(as.Date(sprintf("%d-01-01", SOLAR_REFERENCE_YEAR)),
              as.Date(sprintf("%d-12-31", SOLAR_REFERENCE_YEAR)), by = "day")
  st <- get_suntimes(days, SITE_LAT, SITE_LON, SOLAR_OFFSET_HOURS)
  c(mean(st$sunrise), mean(st$sunset)) * pi / 12
})


# ── Public ───────────────────────────────────────────────────────────────────

#' Position on the 24-hour circle in the camera's own wall time.
#'
#' NA in, NA out: an unrepairable clock has no hour, and an unknown hour is never
#' imputed (the same rule 01_load_data.R applies to `date`).
clock_rad <- function(datetime) {
  lt <- as.POSIXlt(datetime)
  (lt$hour * 3600 + lt$min * 60 + lt$sec) / 86400 * 2 * pi
}

#' Position on the 24-hour circle relative to that day's sunrise and sunset.
#'
#' The transformation and the sun times both come from `activity`; nothing here
#' reimplements solar geometry. Note that get_suntimes() is documented as
#' APPROXIMATE and computes geometric sunrise at sea level — it does not know the
#' Andean horizon, so first light at a station with a ridge to the east is later
#' than the number. Both limits belong in Methods.
solar_rad <- function(datetime, type = SOLAR_ANCHOR) {
  st <- get_suntimes(as.Date(datetime), SITE_LAT, SITE_LON, SOLAR_OFFSET_HOURS)
  anchor <- as.matrix(st[, c("sunrise", "sunset")]) * pi / 12
  .quietly_radian(
    if (identical(type, "single")) {
      transtime(clock_rad(datetime), anchor[, 1],
                mnanchor = SOLAR_MNANCHOR[1], type = "single")
    } else {
      transtime(clock_rad(datetime), anchor,
                mnanchor = if (identical(type, "equinoctial")) NULL else SOLAR_MNANCHOR,
                type = type)
    })
}

# transtime() guesses whether its input is radians or proportion-of-day by testing
# max(dat) < 1, and warns when it is. That test is wrong for legitimate input: a
# handful of detections all before 03:49 are radians below 1, and so is an all-NA
# column, which is the ordinary case for a station whose clock was unrepairable.
# 04 calls this on species subsets, so the false positive would fire on most runs and
# train the reader to ignore warnings from this module. Matched on the message text
# and nothing else -- any other warning transtime() raises still surfaces.
.quietly_radian <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    if (grepl("expecting radian data", conditionMessage(w), fixed = TRUE) ||
        grepl("no non-missing arguments to max", conditionMessage(w), fixed = TRUE)) {
      invokeRestart("muffleWarning")
    }
  })
}

#' Hours between sunrise and sunset. For figure strips that must say how much day
#' a period actually had.
daylength_hours <- function(date) {
  get_suntimes(as.Date(date), SITE_LAT, SITE_LON, SOLAR_OFFSET_HOURS)$daylength
}

#' What to call the axis, so the transformation and its description cannot drift
#' apart. `frame` is "clock" or "solar".
time_frame_label <- function(frame) {
  switch(frame,
    clock = "Hora del reloj de la cámara (horario de invierno, UTC-4)",
    solar = switch(SOLAR_ANCHOR,
      average     = "Hora solar (anclada a amanecer y ocaso; escala: día medio del sitio)",
      equinoctial = "Hora solar equinoccial (amanecer 06:00, mediodía 12:00, ocaso 18:00)",
      single      = "Hora solar (anclada al amanecer)"),
    stop(sprintf("time_frame_label(): unknown frame '%s'; expected \"clock\" or \"solar\".",
                 frame))
  )
}
