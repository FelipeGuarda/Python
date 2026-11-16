# 09_domestic_accompaniment.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   For each domestic species, how many of its visits came with people: a person or
#   vehicle in the frame (co-presence tag) or at the same station within
#   ACCOMPANIMENT_WINDOW_MIN. And for the visits WITHOUT that evidence -- the ones
#   that may be a dog or horse moving on its own -- where and at what hour.
#   The rule lives in R/00_accompaniment.R; this script only applies and reports it.
#
# READ THE RESULT AS A LOWER BOUND while `in_frame_*` is NA. The producer labels each
# image once (animal > person > vehicle), so a handler beside the dog exists only as
# a tag, and the tags arrive with the Windows retro pass (copresence.CHECKED).
#
# NOT COVERED: stills with no trustworthy clock have no episode and no time, so they
# can be judged neither by proximity nor, here, by tag. They are counted and printed
# so the gap is visible (all of otoño 2025's Caballo, for one).
#
# INPUT   data/records_identified.rds, data/human_activity.rds
# OUTPUT  data/accompaniment.csv   one row per (species, episode)
# ─────────────────────────────────────────────────────────────────────────────

library(here)

here::i_am("R/09_domestic_accompaniment.R")
source(here::here("R", "00_contract.R"))
source(here::here("R", "00_admissibility.R"))
source(here::here("R", "00_accompaniment.R"))

contract_assert_current()

# Latin names as in `species_latin`. Adding a species here is the whole change.
DOMESTIC_SPECIES <- c("Canis lupus familiaris" = "Perro", "Equus caballus" = "Caballo")

records <- readRDS(here("data", "records_identified.rds"))
people  <- readRDS(here("data", "human_activity.rds"))

result <- do.call(rbind, lapply(names(DOMESTIC_SPECIES), function(sp) {
  a <- accompanied(records, people, sp)
  if (nrow(a) == 0) return(NULL)
  cbind(species = DOMESTIC_SPECIES[[sp]], a, stringsAsFactors = FALSE)
}))

write.csv(result, here("data", "accompaniment.csv"), row.names = FALSE,
          fileEncoding = "UTF-8")

pct <- function(k, n) sprintf("%d / %d (%.0f%%)", k, n, 100 * k / n)

cat(sprintf("\nAccompaniment window: %d min. Tags checked on %d of %d episodes.\n",
            ACCOMPANIMENT_WINDOW_MIN, sum(!is.na(result$in_frame_human)), nrow(result)))

for (sp in names(DOMESTIC_SPECIES)) {
  lbl <- DOMESTIC_SPECIES[[sp]]
  all_rows <- records[records$species_latin == sp, ]
  untimed  <- all_rows[!all_rows$time_admissible %in% TRUE, ]
  r <- result[result$species == lbl, ]
  cat(sprintf("\n== %s: %d stills, %d without a trustworthy clock (not judged)\n",
              lbl, nrow(all_rows), nrow(untimed)))
  if (nrow(untimed)) print(table(untimed$campaign, untimed$station_id))
  if (nrow(r) == 0) next
  cat(sprintf("Episodes accompanied: %s  (near a person %d, near a vehicle %d)\n",
              pct(sum(r$accompanied), nrow(r)), sum(r$near_human), sum(r$near_vehicle)))
  cat("By campaign (accompanied / episodes):\n")
  print(with(r, tapply(accompanied, campaign, function(x) sprintf("%d / %d", sum(x), length(x)))))
  alone <- r[!r$accompanied, ]
  if (nrow(alone)) {
    cat(sprintf("Without evidence of company: %d episodes\n", nrow(alone)))
    cat("  by station:\n"); print(table(alone$station_id))
    cat("  by hour (clock):\n"); print(table(factor(alone$hour, levels = 0:23))[table(factor(alone$hour, levels = 0:23)) > 0])
    cat(sprintf("  team never recorded at that station-campaign: %d\n", sum(is.na(alone$gap_min))))
  }
}
cat("\nWrote data/accompaniment.csv\n")
