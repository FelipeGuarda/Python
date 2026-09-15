# 00_figures.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Owns ONE decision: that a figure in this repository has a STABLE path, and what
#   to do about the fact that camtrapR refuses to give it one.
#
# WHY THIS FILE EXISTS
#   camtrapR's `writePNG = TRUE` stamps the run date into the filename it chooses —
#   `Presence_Puma_2026-09-15.png`, `activity_density_Liebre_2026-09-15.png` — and the
#   name is not configurable. `figures/` is tracked, so every run added a fresh set
#   beside the previous one and git recorded the pair as an addition plus a stranded
#   file rather than a change to one figure. Twenty-four superseded PNGs were deleted
#   by hand on 2026-09-08 and twenty-four more on 2026-09-15, which is twice that a
#   person had to notice and do it.
#
#   Renaming after the fact is better than deleting before: with a stable name the
#   same figure keeps one path for its whole life, a diff shows "this figure changed"
#   instead of two unrelated files, and nothing accumulates to be swept up later.
#   Where THIS project chooses the filename (04's per-pair plots) there is simply no
#   date in it and this file is not involved.
#
# THE DATE IS NOT LOST
#   It was never information: the figure's date is the commit's date, and for a
#   working tree it is the file's mtime. What the stamped name actually recorded was
#   which run produced it, which is exactly the question `contract_stamp.json` and the
#   contract gate answer properly — a figure that disagrees with the data refuses to
#   regenerate rather than sitting on disk under an older date.
#
# REQUIRES    nothing beyond base R.
# SOURCED BY  03, 05. Call after any camtrapR block with writePNG = TRUE.
# ─────────────────────────────────────────────────────────────────────────────


# Files camtrapR dated: "<anything>_YYYY-MM-DD.png". Anchored to the extension so a
# species name containing digits, or a directory path containing a date, cannot match.
DATED_PNG_PATTERN <- "_[0-9]{4}-[0-9]{2}-[0-9]{2}\\.png$"


# Rename every dated PNG in `dir` to its undated name, replacing any existing file of
# that name. Returns the number renamed, invisibly.
#
# Idempotent: a directory already holding only stable names yields 0 and no writes, so
# calling it twice in one run is harmless. A directory that does not exist yields 0
# rather than an error — the caller has just created it, and a run that produced no
# figures is not a failure this function should invent.
stabilize_dated_pngs <- function(dir) {
  if (!dir.exists(dir)) return(invisible(0L))

  dated <- list.files(dir, pattern = DATED_PNG_PATTERN, full.names = TRUE)
  if (length(dated) == 0) return(invisible(0L))

  stable <- sub(DATED_PNG_PATTERN, ".png", dated)

  # file.rename() will not overwrite on Windows, so the target goes first. Removing a
  # figure we are about to replace in the same call is safe; removing one we are not
  # would not be, which is why only `stable` names derived from `dated` are touched.
  existing <- stable[file.exists(stable)]
  if (length(existing)) file.remove(existing)

  ok <- file.rename(dated, stable)
  if (!all(ok)) {
    warning(sprintf("%d of %d dated figure(s) in %s could not be renamed; the stale set is still there.",
                    sum(!ok), length(ok), dir), call. = FALSE)
  }
  invisible(sum(ok))
}
