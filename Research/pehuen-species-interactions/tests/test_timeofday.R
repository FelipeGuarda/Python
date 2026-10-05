# tests/test_timeofday.R
# ─────────────────────────────────────────────────────────────────────────────
# Tests for R/00_timeofday.R. Base R only, same as the other three suites. Run:
#
#     conda run -n pehuen-analysis Rscript tests/test_timeofday.R
#
# Four things are worth the file.
#
#   1. THE ANCHOR. solar_rad() must equal a direct activity::transtime() call on
#      activity::get_suntimes() output. The solar geometry is the package's, not
#      ours, and the test must not reimplement it -- a second copy of the rule
#      inside the test is exactly the defect this project keeps finding (two
#      time_rad derivations with nothing comparing them; densityFit() output fed to
#      overlapEst() for seven weeks because camtrapR printed its own correct number
#      and nothing compared the two).
#
#   2. THE PIN. transtime(type="average") defaults its reference to the mean of the
#      anchors it is handed, so an unpinned call makes the transformation depend on
#      which rows the caller passed. 04 computes every pair on a species subset.
#      The test proves subsetting cannot move a detection.
#
#   3. THE PHYSICS. Near the equinoxes the solar frame must nearly coincide with the
#      clock frame; at the solstices it must depart by the sunrise offset. If the
#      transform were inert or inverted, this is what would catch it.
#
#   4. THE OFFSET BOUND. SOLAR_OFFSET_HOURS is a declared assumption (see the module
#      header). This measures what a one-hour error in it is worth, so the assumption
#      is bounded rather than trusted.
# ─────────────────────────────────────────────────────────────────────────────

library(here)
here::i_am("tests/test_timeofday.R")
source(here::here("R", "00_timeofday.R"))

.failures <- 0L
check <- function(label, ok) {
  if (isTRUE(ok)) {
    cat(sprintf("  ok   %s\n", label))
  } else {
    .failures <<- .failures + 1L
    cat(sprintf("  FAIL %s\n", label))
  }
}
P <- function(s) as.POSIXct(s, tz = "UTC")
TOL <- 1e-12


cat("clock_rad -- the conversion 01_load_data.R used to own\n")
check("midnight is 0",        abs(clock_rad(P("2025-07-01 00:00:00")) - 0) < TOL)
check("06:00 is pi/2",        abs(clock_rad(P("2025-07-01 06:00:00")) - pi/2) < TOL)
check("noon is pi",           abs(clock_rad(P("2025-07-01 12:00:00")) - pi) < TOL)
check("18:00 is 3*pi/2",      abs(clock_rad(P("2025-07-01 18:00:00")) - 3*pi/2) < TOL)
check("seconds are carried",
      abs(clock_rad(P("2025-07-01 00:00:01")) - (2*pi/86400)) < TOL)
check("it matches the lubridate expression it replaced",
      {
        dt <- P(c("2025-06-20 08:30:15", "2025-12-20 21:05:00", "2025-01-01 00:00:00"))
        old <- (as.integer(format(dt, "%H")) * 3600 +
                as.integer(format(dt, "%M")) * 60 +
                as.integer(format(dt, "%S"))) / 86400 * 2 * pi
        max(abs(clock_rad(dt) - old)) < TOL
      })
check("NA in, NA out",        is.na(clock_rad(as.POSIXct(NA))))
check("length is preserved",  length(clock_rad(P(c("2025-07-01 01:00:00", NA)))) == 2L)
check("the date is irrelevant to the angle",
      abs(clock_rad(P("2024-02-29 09:17:33")) - clock_rad(P("2026-11-03 09:17:33"))) < TOL)


cat("solar_rad -- THE ANCHOR: our value IS the package's value\n")
dt <- P(c("2024-10-09 05:12:00", "2025-06-20 08:30:15", "2025-12-20 21:05:00",
          "2026-05-15 19:44:02", "2025-03-21 06:00:00"))
direct <- local({
  st  <- get_suntimes(as.Date(dt), SITE_LAT, SITE_LON, SOLAR_OFFSET_HOURS)
  anc <- as.matrix(st[, c("sunrise", "sunset")]) * pi / 12
  transtime(clock_rad(dt), anc, mnanchor = SOLAR_MNANCHOR, type = "average")
})
check("solar_rad == transtime(get_suntimes(...)) exactly",
      max(abs(solar_rad(dt, "average") - direct)) < TOL)
check("the equinoctial branch is the package's too",
      {
        st  <- get_suntimes(as.Date(dt), SITE_LAT, SITE_LON, SOLAR_OFFSET_HOURS)
        anc <- as.matrix(st[, c("sunrise", "sunset")]) * pi / 12
        max(abs(solar_rad(dt, "equinoctial") -
                transtime(clock_rad(dt), anc, type = "equinoctial"))) < TOL
      })
check("the single branch is the package's too",
      {
        st  <- get_suntimes(as.Date(dt), SITE_LAT, SITE_LON, SOLAR_OFFSET_HOURS)
        anc <- as.matrix(st[, c("sunrise", "sunset")]) * pi / 12
        max(abs(solar_rad(dt, "single") -
                transtime(clock_rad(dt), anc[, 1],
                          mnanchor = SOLAR_MNANCHOR[1], type = "single"))) < TOL
      })


cat("solar_rad -- THE PIN: a subset cannot move a detection\n")
span <- seq(as.Date("2024-10-09"), as.Date("2026-05-15"), by = "day")
many <- as.POSIXct(paste(span, "06:30:00"), tz = "UTC")
check("row 1 is identical whether 1 or 585 rows are passed",
      abs(solar_rad(many)[1] - solar_rad(many[1])) < TOL)
check("a 50-row subset agrees with the full table on all 50",
      max(abs(solar_rad(many)[1:50] - solar_rad(many[1:50]))) < TOL)
check("the reference anchor is a constant, not a function of the data",
      length(SOLAR_MNANCHOR) == 2L && all(is.finite(SOLAR_MNANCHOR)))
check("the pin is what does it -- unpinned, the same row moves >10 min",
      {
        st  <- get_suntimes(as.Date(many), SITE_LAT, SITE_LON, SOLAR_OFFSET_HOURS)
        anc <- as.matrix(st[, c("sunrise", "sunset")]) * pi / 12
        unpinned_full   <- transtime(clock_rad(many), anc, type = "average")[1]
        unpinned_subset <- transtime(clock_rad(many[1:50]), anc[1:50, ], type = "average")[1]
        abs(unpinned_full - unpinned_subset) * 86400 / (2*pi) / 60 > 10
      })


cat("solar_rad -- THE PHYSICS\n")
# At an equinox sunrise is near 06:00 and sunset near 18:00, which is also where the
# site's annual mean sits, so the transform is near the identity.
eq <- P(paste(c("2025-03-21", "2025-09-22"), "10:00:00"))
check("near the equinoxes solar is within 30 min of clock",
      max(abs(solar_rad(eq) - clock_rad(eq))) * 86400 / (2*pi) / 60 < 30)
# At the winter solstice sunrise is 08:07, more than an hour after the annual mean of
# 06:42, so a morning detection must be pulled EARLIER on the solar circle.
ws <- P("2025-06-20 09:00:00")
check("at the winter solstice a morning detection moves earlier",
      solar_rad(ws) < clock_rad(ws))
check("...and by more than half an hour",
      (clock_rad(ws) - solar_rad(ws)) * 86400 / (2*pi) / 60 > 30)
ss <- P("2025-12-20 06:00:00")
check("at the summer solstice a dawn detection moves later",
      solar_rad(ss) > clock_rad(ss))
check("the transform is not inert",
      max(abs(solar_rad(dt) - clock_rad(dt))) > 1e-3)
check("every value stays on the circle [0, 2*pi]",
      {
        v <- solar_rad(many)
        all(v >= 0 & v <= 2*pi, na.rm = TRUE)
      })


cat("solar_rad -- shape and missingness\n")
check("NA in, NA out",       is.na(solar_rad(as.POSIXct(NA))))
check("length is preserved", length(solar_rad(P(c("2025-07-01 01:00:00", NA)))) == 2L)
check("it is a rotation, not a filter -- no record is lost",
      {
        mixed <- c(many[1:100], as.POSIXct(rep(NA, 7)))
        sum(!is.na(solar_rad(mixed))) == sum(!is.na(clock_rad(mixed)))
      })
check("the whole span of the study resolves",
      !any(is.na(solar_rad(many))))
check("an unknown type is refused, not silently defaulted",
      inherits(try(solar_rad(dt, "sidereal"), silent = TRUE), "try-error"))


cat("THE OFFSET BOUND -- what a one-hour error in SOLAR_OFFSET_HOURS is worth\n")
off_delta <- local({
  st3  <- get_suntimes(as.Date(many), SITE_LAT, SITE_LON, SOLAR_OFFSET_HOURS + 1)
  anc3 <- as.matrix(st3[, c("sunrise", "sunset")]) * pi / 12
  mn3  <- local({
    d  <- seq(as.Date("2025-01-01"), as.Date("2025-12-31"), by = "day")
    s  <- get_suntimes(d, SITE_LAT, SITE_LON, SOLAR_OFFSET_HOURS + 1)
    c(mean(s$sunrise), mean(s$sunset)) * pi / 12
  })
  at3 <- transtime(clock_rad(many), anc3, mnanchor = mn3, type = "average")
  max(abs(at3 - solar_rad(many))) * 86400 / (2*pi) / 60
})
cat(sprintf("  ... a -3 offset instead of -4 moves a detection by at most %.1f min\n",
            off_delta))
check("the offset error is bounded by one hour, not amplified",
      off_delta <= 60 + 1e-6)


cat("daylength_hours and time_frame_label\n")
check("the shortest day of 2025 is ~9.4 h",
      abs(daylength_hours(as.Date("2025-06-20")) - 9.4) < 0.2)
check("the longest day of 2025 is ~15.0 h",
      abs(daylength_hours(as.Date("2025-12-20")) - 15.0) < 0.2)
check("day length is symmetric about the equinoxes to within 15 min",
      abs(daylength_hours(as.Date("2025-03-21")) -
          daylength_hours(as.Date("2025-09-22"))) < 0.25)
check("a POSIXct is accepted", is.finite(daylength_hours(P("2025-06-20 13:00:00"))))
check("the clock label names the offset it assumes",
      grepl("UTC-4", time_frame_label("clock"), fixed = TRUE))
check("the solar label describes the transformation in force",
      grepl(switch(SOLAR_ANCHOR, average = "día medio",
                   equinoctial = "equinoccial", single = "amanecer"),
            time_frame_label("solar")))
check("an unknown frame is refused",
      inherits(try(time_frame_label("lunar"), silent = TRUE), "try-error"))


cat(sprintf("\n%s\n", if (.failures == 0L) "all tests passed"
                      else sprintf("%d FAILURE(S)", .failures)))
quit(status = if (.failures == 0L) 0L else 1L)
