# tests/test_detection_history.R
# ─────────────────────────────────────────────────────────────────────────────
# Tests for R/00_detection_history.R and effort_admissible() in
# R/00_admissibility.R. Base R only, same as the other suites. Run:
#
#     conda run -n pehuen-analysis Rscript tests/test_detection_history.R
#
# What is worth asserting: the grid's three cell states mean what the header says
# (an unsurveyed occasion is NA, never 0), the half-open field window and the shared
# changeover day do not invent or lose a day, and against the real data the effort
# grid conserves exactly the days season_effort() hands to a rate denominator -- two
# modules, one count. The last check is the one the module header cites: no
# time-admissible episode falls on a day the effort rule excludes.
# ─────────────────────────────────────────────────────────────────────────────

library(here)
here::i_am("tests/test_detection_history.R")
source(here::here("R", "00_detection_history.R"))

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
refusal <- function(expr) tryCatch({ expr; "no error" }, error = function(e) conditionMessage(e))


# ── Fixtures ─────────────────────────────────────────────────────────────────
# Season Invierno 2025: 2025-06-01 .. 2025-08-31, 92 days -> six 14-day occasions
# and a seventh of 8 days.
WINTER <- D("2025-06-01")

dep <- data.frame(
  campaign     = c("c1",           "c1",                 "c1",           "c1",           "c2"),
  station_id   = c("A",            "B",                  "C",            "D",            "D"),
  field_start  = D(c("2025-05-01", "2025-05-01",         "2025-05-01",   "2025-05-01",   "2025-07-01")),
  field_end    = D(c("2025-06-15", "2025-09-30",         "2025-09-30",   "2025-07-01",   "2025-10-01")),
  media_status = c("in_canonical", "video_only_offline", "in_canonical", "in_canonical", "in_canonical"),
  valid_effort = c(TRUE,           NA,                   FALSE,          TRUE,           TRUE),
  stringsAsFactors = FALSE
)
dep$field_days <- as.integer(dep$field_end - dep$field_start)

rec <- data.frame(
  campaign      = c("c1", "c1", "c1", "c2", "c2", "c1", "c2", "c1"),
  station_id    = c("A",  "A",  "A",  "D",  "D",  "D",  "D",  "D"),
  species_label = c("Zorro culpeo", "Zorro culpeo", "Zorro culpeo", "Zorro culpeo",
                    "Zorro culpeo", "Liebre", "Zorro culpeo", "Zorro culpeo"),
  datetime      = as.POSIXct(c("2025-06-02 03:00", "2025-06-03 04:00", "2025-06-14 22:00",
                               "2025-08-30 01:00", "2025-07-20 05:00", "2025-06-20 05:00",
                               "2025-09-05 05:00", "2025-06-10 05:00"), tz = "UTC"),
  time_admissible = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),
  episode_30min = c("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"),
  stringsAsFactors = FALSE
)

h <- detection_history(rec, dep, "Zorro culpeo", WINTER, quiet = TRUE)


cat("effort_admissible\n")
check("detections: stills and a valid clock only",
      identical(effort_admissible(dep, "detections"), c(TRUE, FALSE, FALSE, TRUE, TRUE)))
check("sampling: video-only counts, a failed clock still counts",
      identical(effort_admissible(dep, "sampling"), c(TRUE, TRUE, TRUE, TRUE, TRUE)))
check("card_failure samples nothing",
      !effort_admissible(data.frame(media_status = "card_failure", valid_effort = NA), "sampling"))
check("a table missing the columns stops and names them",
      grepl("valid_effort", refusal(effort_admissible(dep[, "media_status", drop = FALSE]))))


cat("detection_history -- refusals\n")
check("no season refuses, naming closure",
      grepl("closure", refusal(detection_history(rec, dep, "Zorro culpeo"))))
check("NULL season refuses",
      grepl("closure", refusal(detection_history(rec, dep, "Zorro culpeo", NULL))))
check("a date inside a season refuses and names the season start",
      grepl("2025-06-01", refusal(detection_history(rec, dep, "Zorro culpeo", D("2025-07-04")))))
check("an unknown species refuses and lists the known ones",
      grepl("Liebre", refusal(detection_history(rec, dep, "Zorro", WINTER))))
check("a zero occasion length refuses",
      grepl("occasion_days", refusal(detection_history(rec, dep, "Zorro culpeo", WINTER, 0))))


cat("detection_history -- the grid\n")
check("92-day season at 14 d gives 7 occasions", nrow(h$occasions) == 7L && ncol(h$y) == 7L)
check("occasions start on the season's first day", h$occasions$start[1] == WINTER)
check("the last occasion is partial and ends on the season's last day",
      h$occasions$end[7] == D("2025-08-31") && h$occasions$start[7] == D("2025-08-24"))
check("occasions tile the season with no gap or overlap",
      all(h$occasions$start[-1] == h$occasions$end[-7] + 1L))
check("rows are the stations surveyed in the season, and only those",
      identical(rownames(h$y), c("A", "D")))
check("y and effort share their shape and names",
      identical(dimnames(h$y), dimnames(h$effort)))


cat("detection_history -- effort\n")
check("the half-open window: A's field_end (06-15) is not a survey day",
      h$effort["A", "o1"] == 14L && is.na(h$effort["A", "o2"]))
check("the changeover day is counted once, not twice",
      sum(h$effort["D", ], na.rm = TRUE) == 92L)
check("total effort is the surveyed station-days, counted independently",
      sum(h$effort, na.rm = TRUE) == 14L + 92L)
check("a 1-day occasion holds at most one day",
      all(detection_history(rec, dep, "Zorro culpeo", WINTER, 1L, quiet = TRUE)$effort %in% c(1L, NA)))


cat("detection_history -- cells\n")
check("two episodes in one occasion are one detection",
      h$y["A", "o1"] == 1L)
check("unsurveyed is NA, not 0",
      all(is.na(h$y["A", 2:7])))
check("a surveyed occasion without the species is 0",
      h$y["D", "o1"] == 0L && h$y["D", "o2"] == 0L)
check("a detection lands in its own occasion",
      h$y["D", "o4"] == 1L && h$y["D", "o7"] == 1L)
check("another species' episode does not mark this one (D o2 is liebre)",
      h$y["D", "o2"] == 0L)
check("a detection after the season is not in it",
      sum(h$y, na.rm = TRUE) == 3L)
check("a time-inadmissible record marks nothing (D 06-10 would be o1)",
      { r2 <- rec; r2$time_admissible[8] <- TRUE
        h$y["D", "o1"] == 0L &&
          detection_history(r2, dep, "Zorro culpeo", WINTER, quiet = TRUE)$y["D", "o1"] == 1L })
check("y is 0/1 exactly where effort exists",
      identical(is.na(h$y), is.na(h$effort)) && all(h$y %in% c(0L, 1L, NA)))
check("a season with no detections is all zero, not an error",
      { h0 <- detection_history(rec, dep, "Liebre", D("2025-09-01"), quiet = TRUE)
        sum(h0$y, na.rm = TRUE) == 0L && nrow(h0$y) == 1L })
check("an episode on an unsurveyed day is left out, and said",
      { r3 <- rbind(rec, transform(rec[1, ], station_id = "B", episode_30min = "e9"))
        msg <- character()
        withCallingHandlers(detection_history(r3, dep, "Zorro culpeo", WINTER),
                            message = function(m) { msg <<- c(msg, conditionMessage(m))
                                                    invokeRestart("muffleMessage") })
        any(grepl("not surveyed are left out: B/2025-06-02", msg, fixed = TRUE)) })


cat("detection_history -- against the real field record\n")
rec_path <- here::here("data", "records_all.rds")
dep_path <- here::here("data", "deployments.rds")
if (file.exists(rec_path) && file.exists(dep_path)) {
  real_rec <- readRDS(rec_path)
  real_dep <- readRDS(dep_path)
  eff  <- season_effort(real_dep)
  seasons <- sort(unique(eff$season_start))

  check("effort conserves the rate denominator's days, every season",
        all(vapply(seasons, function(s) {
          h <- detection_history(real_rec, real_dep, "Zorro culpeo", s, quiet = TRUE)
          sum(h$effort, na.rm = TRUE) ==
            sum(eff$effort_days[eff$season_start == s & effort_admissible(eff, "detections")])
        }, logical(1))))

  ep <- episodes(real_rec, quiet = TRUE)
  surveyed <- do.call(rbind, lapply(which(effort_admissible(real_dep, "detections")), function(i) {
    data.frame(station_id = real_dep$station_id[i],
               day = seq(real_dep$field_start[i], real_dep$field_end[i] - 1L, by = "day"))
  }))
  check("no time-admissible episode falls on a day the effort rule excludes (0 lost)",
        all(paste(ep$station_id, as.Date(ep$datetime)) %in%
              paste(surveyed$station_id, surveyed$day)))

  check("every 1 is an episode, and every episode is a 1 (all species, all seasons)",
        all(vapply(sort(unique(ep$species_label)), function(sp) {
          all(vapply(seasons, function(s) {
            h <- detection_history(real_rec, real_dep, sp, s, quiet = TRUE)
            e <- ep[ep$species_label == sp & season_start(ep$datetime) == s, ]
            want <- unique(paste(e$station_id,
                                 as.integer(as.Date(e$datetime) - s) %/% OCCASION_DAYS + 1L))
            got <- which(h$y == 1L, arr.ind = TRUE)
            setequal(want, paste(rownames(h$y)[got[, 1]], got[, 2]))
          }, logical(1)))
        }, logical(1))))

  hw <- detection_history(real_rec, real_dep, "Zorro culpeo", D("2025-06-01"), quiet = TRUE)
  check("culpeo, Invierno 2025: 10 of 24 stations, as measured 2026-10-06",
        sum(rowSums(hw$y, na.rm = TRUE) > 0) == 10L && nrow(hw$y) == 24L)
} else {
  cat("  skip data/*.rds absent -- run R/01_load_data.R\n")
}

if (.failures > 0) {
  cat(sprintf("\n%d test(s) FAILED\n", .failures))
  quit(save = "no", status = 1L)
}
cat("\nall tests passed\n")
