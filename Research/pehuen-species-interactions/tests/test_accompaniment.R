# tests/test_accompaniment.R
# ─────────────────────────────────────────────────────────────────────────────
# Tests for R/00_accompaniment.R. Base R only, same harness as the other tests. Run:
#
#     conda run -n pehuen-analysis Rscript tests/test_accompaniment.R
#
# The window edges are off-by-one territory, and the span (first to last frame) is
# what makes a long visit accompanied by a person who came at its end. The tag/NA
# semantics matter because NA ("not checked") must never read as "alone".
# ─────────────────────────────────────────────────────────────────────────────

library(here)
here::i_am("tests/test_accompaniment.R")
source(here::here("R", "00_admissibility.R"))
source(here::here("R", "00_accompaniment.R"))

.failures <- 0L
check <- function(label, ok) {
  if (isTRUE(ok)) {
    cat(sprintf("  ok   %s\n", label))
  } else {
    .failures <<- .failures + 1L
    cat(sprintf("  FAIL %s\n", label))
  }
}
T0 <- as.POSIXct("2025-03-10 10:00:00", tz = "UTC")
at <- function(min) T0 + min * 60

# One dog episode at CT01 from 10:00 to 10:10, unless overridden.
dog <- function(times = c(0, 10), station = "CT01", campaign = "c1", ep = "c1|CT01|dog|1",
                with_human = NA, with_vehicle = NA, admissible = TRUE) {
  n <- length(times)
  data.frame(campaign = campaign, station_id = station, datetime = at(times),
             species_latin = "Canis lupus familiaris", episode_30min = ep,
             with_human = rep(with_human, length.out = n),
             with_vehicle = rep(with_vehicle, length.out = n),
             time_admissible = admissible, stringsAsFactors = FALSE)
}
person <- function(min, subject = "human", station = "CT01", campaign = "c1",
                   admissible = TRUE) {
  data.frame(campaign = campaign, station_id = station, datetime = at(min),
             subject = subject, episode_30min = NA_character_,
             time_admissible = admissible, stringsAsFactors = FALSE)
}
run <- function(records, people) accompanied(records, people, "Canis lupus familiaris")


cat("the window, measured from the episode's span\n")
check("a person exactly 30 min after the LAST frame counts",
      run(dog(), person(40))$accompanied)
check("...one second later does not",
      !run(dog(), person(40 + 1/60))$accompanied)
check("a person exactly 30 min before the FIRST frame counts",
      run(dog(), person(-30))$accompanied)
check("a person inside the span is gap 0",
      run(dog(), person(5))$gap_min == 0)
check("gap_min is the NEAREST frame",
      run(dog(), rbind(person(100), person(55)))$gap_min == 45)
check("window_min is honoured",
      !accompanied(dog(), person(40), "Canis lupus familiaris", window_min = 15)$accompanied)

cat("where and when the person must be\n")
check("another station does not count",
      !run(dog(), person(5, station = "CT02"))$accompanied)
check("another campaign does not count",
      !run(dog(), person(5, campaign = "c2"))$accompanied)
check("team never at that station-campaign -> gap_min NA, not accompanied",
      { a <- run(dog(), person(5, station = "CT02")); is.na(a$gap_min) && !a$accompanied })
check("a person with no trustworthy clock is not evidence",
      !run(dog(), person(5, admissible = FALSE))$accompanied)

cat("human vs vehicle\n")
a <- run(dog(), rbind(person(5), person(20, "vehicle")))
check("both kinds nearby are both reported", a$near_human && a$near_vehicle)
a <- run(dog(), person(20, "vehicle"))
check("a vehicle alone is company, and is not called a person",
      a$accompanied && a$near_vehicle && !a$near_human)

cat("tags in the frame\n")
a <- run(dog(), person(500)[0, ])
check("no tags checked -> in_frame NA, and NA is not company",
      is.na(a$in_frame_human) && !a$accompanied)
a <- run(dog(with_human = c(FALSE, TRUE)), person(500))
check("one tagged frame makes the episode accompanied, with no one nearby",
      a$in_frame_human && !a$near_human && a$accompanied)
a <- run(dog(with_human = FALSE), person(500))
check("checked and untagged is FALSE, not NA", identical(a$in_frame_human, FALSE))

cat("what is judged at all\n")
check("an episode with no trustworthy clock is excluded",
      nrow(run(dog(admissible = FALSE), person(5))) == 0)
check("no episodes -> empty frame with the full set of columns",
      { a <- run(dog()[0, ], person(5)); nrow(a) == 0 && "accompanied" %in% names(a) })
two <- rbind(dog(), dog(times = c(200, 205), ep = "c1|CT01|dog|2"))
a <- run(two, person(5))
check("each episode is judged on its own span",
      identical(a$accompanied, c(TRUE, FALSE)))
check("other species are ignored",
      { x <- dog(); x$species_latin <- "Equus caballus"; nrow(run(x, person(5))) == 0 })

if (.failures > 0) {
  cat(sprintf("\n%d test(s) FAILED\n", .failures))
  quit(save = "no", status = 1L)
}
cat("\nall tests passed\n")
