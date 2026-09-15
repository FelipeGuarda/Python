# tests/test_seasons.R
# ─────────────────────────────────────────────────────────────────────────────
# Tests for R/00_seasons.R. Base R only, same as tests/test_contract.R -- testthat
# is not in the pehuen-analysis environment. Run:
#
#     conda run -n pehuen-analysis Rscript tests/test_seasons.R
#
# Two things are worth the file. The boundary days are off-by-one territory and the
# year straddle is where a season rule is normally wrong. And the effort split has an
# exact invariant -- sum(effort_days) == field_days for every deployment -- which is
# the whole argument that switching the denominator from campaign to season moved no
# effort. That one is asserted against the real deployments.rds when it exists, not
# only against fixtures.
# ─────────────────────────────────────────────────────────────────────────────

library(here)
here::i_am("tests/test_seasons.R")
source(here::here("R", "00_seasons.R"))

.failures <- 0L
check <- function(label, ok) {
  if (isTRUE(ok)) {
    cat(sprintf("  ok   %s\n", label))
  } else {
    .failures <<- .failures + 1L
    cat(sprintf("  FAIL %s\n", label))
  }
}
D <- function(s) as.Date(s)


cat("season_start -- boundaries\n")
check("2025-03-01 opens otoño",      season_start(D("2025-03-01")) == D("2025-03-01"))
check("2025-02-28 is still verano",  season_start(D("2025-02-28")) == D("2024-12-01"))
check("2025-06-01 opens invierno",   season_start(D("2025-06-01")) == D("2025-06-01"))
check("2025-08-31 is still invierno",season_start(D("2025-08-31")) == D("2025-06-01"))
check("2025-09-01 opens primavera",  season_start(D("2025-09-01")) == D("2025-09-01"))
check("2025-11-30 is still primavera",season_start(D("2025-11-30")) == D("2025-09-01"))
check("2025-12-01 opens verano",     season_start(D("2025-12-01")) == D("2025-12-01"))

cat("season_start -- the year straddle\n")
check("January belongs to December's verano",
      season_start(D("2026-01-14")) == D("2025-12-01"))
check("a December date and the January after it share one period",
      season_start(D("2025-12-31")) == season_start(D("2026-01-01")))

cat("season_start -- shape and missingness\n")
check("NA in, NA out",       is.na(season_start(as.Date(NA))))
check("length is preserved", length(season_start(D(c("2025-01-01", NA, "2025-07-01")))) == 3L)
check("a POSIXct is accepted",
      season_start(as.POSIXct("2025-07-04 23:30:00", tz = "UTC")) == D("2025-06-01"))
check("returns a Date",      inherits(season_start(D("2025-07-01")), "Date"))
check("empty in, empty out", length(season_start(as.Date(character(0)))) == 0L)
check("the whole span of the study resolves",
      !any(is.na(season_start(seq(D("2024-10-09"), D("2026-05-15"), by = "day")))))

cat("season_of\n")
check("July is invierno",    as.character(season_of(D("2025-07-15"))) == "Invierno")
check("January is verano",   as.character(season_of(D("2026-01-14"))) == "Verano")
check("levels are the display order",
      identical(levels(season_of(D("2025-07-15"))), SEASON_LEVELS))
check("every month lands in exactly one season",
      !any(is.na(season_of(seq(D("2025-01-01"), D("2025-12-31"), by = "day")))))

cat("season_label\n")
lab <- as.character(season_label(D("2026-01-14")))
check("verano carries both years",   lab == "Verano 2025-26")
check("otoño carries one year",      as.character(season_label(D("2026-04-01"))) == "Otoño 2026")
check("a period's start and a date inside it label the same",
      identical(as.character(season_label(D("2025-06-01"))),
                as.character(season_label(D("2025-07-20")))))
lv <- levels(season_label(D(c("2026-04-01", "2024-12-25", "2025-07-01"))))
check("levels are chronological, not alphabetical",
      identical(lv, c("Verano 2024-25", "Invierno 2025", "Otoño 2026")))
check("season_label is an ordered factor", is.ordered(season_label(D("2025-07-01"))))


cat("season_effort -- the split\n")
dep <- data.frame(
  campaign     = c("c1", "c1", "c1"),
  station_id   = c("CT01", "CT02", "CT03"),
  field_start  = D(c("2025-06-10", "2024-10-09", "2025-07-01")),
  field_end    = D(c("2025-06-20", "2025-06-11", "2025-07-01")),
  media_status = c("in_canonical", "in_canonical", "card_failure"),
  valid_effort = c(TRUE, FALSE, NA),
  stringsAsFactors = FALSE
)
dep$field_days <- as.integer(dep$field_end - dep$field_start)
eff <- season_effort(dep)

check("a window inside one season yields one row",
      nrow(eff[eff$station_id == "CT01", ]) == 1L)
check("... with every day of it",
      eff$effort_days[eff$station_id == "CT01"] == 10L)
check("a 245-day window yields the four seasons it spans",
      nrow(eff[eff$station_id == "CT02", ]) == 4L)
check("... starting in primavera 2024",
      min(eff$season_start[eff$station_id == "CT02"]) == D("2024-09-01"))
check("... and ending in invierno 2025, on its eleventh day",
      {
        r <- eff[eff$station_id == "CT02", ]
        max(r$season_start) == D("2025-06-01") &&
          r$effort_days[which.max(r$season_start)] == 10L
      })
check("a zero-length window yields no rows",
      nrow(eff[eff$station_id == "CT03", ]) == 0L)

per_dep <- aggregate(effort_days ~ station_id, data = eff, FUN = sum)
check("sum(effort_days) == field_days, per deployment",
      all(per_dep$effort_days ==
            dep$field_days[match(per_dep$station_id, dep$station_id)]))
check("media_status is carried, never re-decided",
      all(eff$media_status[eff$station_id == "CT02"] == "in_canonical"))
check("valid_effort is carried including NA",
      identical(unique(eff$valid_effort[eff$station_id == "CT01"]), TRUE))
check("rows come out in chronological order",
      !is.unsorted(eff$season_start))

cat("season_effort -- degenerate input\n")
e0 <- season_effort(dep[0, , drop = FALSE])
check("no deployments gives an empty frame, not an error", nrow(e0) == 0L)
check("... with the documented columns",
      identical(names(e0), c("campaign", "station_id", "season_start",
                             "effort_days", "media_status", "valid_effort")))
bad <- tryCatch({ season_effort(dep[, c("campaign", "station_id")]); "no error" },
                error = function(e) conditionMessage(e))
check("a deployments table missing columns stops and names them",
      grepl("field_start", bad, fixed = TRUE))


cat("season_effort -- against the real field record\n")
dep_path <- here::here("data", "deployments.rds")
if (file.exists(dep_path)) {
  real <- readRDS(dep_path)
  reff <- season_effort(real)
  check("no station-day is lost or invented",
        sum(reff$effort_days) == sum(real$field_days, na.rm = TRUE))
  check("every deployment's days are conserved exactly",
        {
          got <- aggregate(effort_days ~ campaign + station_id, data = reff, FUN = sum)
          key <- paste(real$campaign, real$station_id)
          all(got$effort_days == real$field_days[match(paste(got$campaign, got$station_id), key)])
        })
  check("winter is sampled, contra 06's old header",
        any(as.character(season_of(reff$season_start)) == "Invierno" & reff$effort_days > 0))
  check("a season period can draw days from two campaigns",
        {
          n <- aggregate(campaign ~ season_start, data = reff,
                         FUN = function(x) length(unique(x)))
          any(n$campaign > 1L)
        })
} else {
  cat("  skip data/deployments.rds absent -- run R/01_load_data.R\n")
}

if (.failures > 0) {
  cat(sprintf("\n%d test(s) FAILED\n", .failures))
  quit(save = "no", status = 1L)
}
cat("\nall tests passed\n")
