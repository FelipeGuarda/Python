# 00_contract.R
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Owns ONE decision: what the camera-trap contract published by camera-traps looks
#   like, and whether this project's derived data is current with it. Nothing else
#   in this project reads CANONICAL_STATE.json or knows where the producer lives.
#
# THE HANDSHAKE (camera-traps/docs/MANUAL-SALUD-DATOS.md, Fase 10)
#   The producer publishes; the consumer verifies; the producer never learns that this
#   consumer exists. Three checks, in this order, before anything is read:
#     1. the contract exists and parses            -> absent means REFUSE, not proceed
#     2. its schema version is the one we can read  -> a different one means REFUSE,
#                                                      we do not guess that new columns
#                                                      do not matter to us
#     3. the campaign description matches what our  -> compared as the WHOLE declared
#        data/ was built from                         block, not just the row count
#
#   Two verbs implement that. `contract_load()` is what 01_load_data.R calls before
#   opening a parquet. `contract_assert_current()` is what every downstream script
#   calls before reading an .rds: it compares the stamp 01 wrote against the contract
#   as published NOW, so a campaign re-ingested upstream cannot keep feeding last
#   month's numbers into a figure. Row 1 of the manual's Fase 10 vigilance table.
#
# A REFUSAL IS A VERDICT, NOT A CRASH
#   `refuse()` prints what did not match and exits with status 2. An R error exits
#   with status 1 and reads as a crash; a scheduled run restarts a crash and
#   investigates a refusal (manual 10F.4). Interactive sessions get stop() instead,
#   so a refusal in RStudio does not close the session.
#
# WHAT THIS DELIBERATELY DOES NOT CHECK
#   Whether the published contract matches the parquets and deployments.csv on disk
#   (`deployments_sha256`, `n_rows` against the file). That is the producer's own
#   `camtrap.canonical_state.verify`, and re-implementing it here would be the second
#   place a repair has to reach. The one exception is the per-campaign row count in
#   01_load_data.R, kept because it is the difference between a figure and a stop.
#
# REQUIRES    jsonlite, here (here::i_am() must have run in the calling script).
# SOURCED BY  every R/0*.R script. Pure functions return verdicts; only the two
#             *_load / *_assert_current entry points refuse.
# ─────────────────────────────────────────────────────────────────────────────


# The one schema version this project knows how to read. Bumping it is a deliberate
# act that must come with a re-read of camera-traps/camtrap/observations.py
# (CANONICAL_COLUMNS) and of the `needed` list in 01_load_data.R.
CONTRACT_SCHEMA_VERSION <- 4L

# Display names for the campaign slugs the producer uses. The slug is the identity
# and travels in every table; the label exists only for figure text. An unknown slug
# renders as itself rather than failing a figure over a caption.
CAMPAIGN_LABELS <- c(
  otono_2025     = "Otoño 2025",
  primavera_2025 = "Primavera 2025",
  otono_2026     = "Otoño 2026"
)

campaign_label <- function(slug) {
  slug <- as.character(slug)
  out <- unname(CAMPAIGN_LABELS[slug])
  out[is.na(out)] <- slug[is.na(out)]
  out
}


# ── Where the producer lives ──────────────────────────────────────────────────
# Derived from THIS project's location, never from the machine's. The sibling
# projects are two levels up in the FMA monorepo; FMA_MONOREPO overrides for a
# checkout laid out differently. FMA_CANONICAL_STATE overrides the contract file
# alone, which is what the tests use to point at a fixture.

monorepo_root <- function() {
  env <- Sys.getenv("FMA_MONOREPO", unset = "")
  if (nzchar(env)) return(env)
  normalizePath(here::here("..", ".."), winslash = "/", mustWork = TRUE)
}

producer_dir <- function() file.path(monorepo_root(), "camera-traps")

contract_path <- function() {
  env <- Sys.getenv("FMA_CANONICAL_STATE", unset = "")
  if (nzchar(env)) return(env)
  file.path(producer_dir(), "data", "CANONICAL_STATE.json")
}

stamp_path <- function() here::here("data", "contract_stamp.json")


# ── The verdict ───────────────────────────────────────────────────────────────

refuse <- function(reasons, what = "camera-trap contract") {
  msg <- paste0(
    "REFUSED (", what, "):\n",
    paste0("  - ", reasons, collapse = "\n")
  )
  if (interactive()) stop(msg, call. = FALSE)
  message(msg)
  quit(save = "no", status = 2L)
}


# ── Pure checks: return reasons, never exit ───────────────────────────────────

# Reads the contract file. Returns list(state = <list or NULL>, reasons = <chr>).
contract_read <- function(path = contract_path()) {
  if (!file.exists(path)) {
    return(list(state = NULL, reasons = sprintf(
      paste0("%s not found. camera-traps publishes it with ",
             "`python -m camtrap.canonical_state --publish` after re-ingesting a ",
             "campaign. Refusing to read against an unknown contract."), path)))
  }
  state <- tryCatch(
    jsonlite::fromJSON(path, simplifyVector = TRUE),
    error = function(e) e
  )
  if (inherits(state, "error")) {
    return(list(state = NULL, reasons = sprintf("%s is not valid JSON: %s",
                                                path, conditionMessage(state))))
  }
  reasons <- character()
  version <- suppressWarnings(as.integer(state$schema_version))
  if (length(version) != 1 || is.na(version) || version != CONTRACT_SCHEMA_VERSION) {
    reasons <- c(reasons, sprintf(
      paste0("%s declares schema_version %s; this project was written against %d. ",
             "The canonical table changed shape. Read ",
             "camera-traps/camtrap/observations.py (CANONICAL_COLUMNS) and update ",
             "R/00_contract.R and R/01_load_data.R deliberately -- do not just bump ",
             "the number."),
      path, format(state$schema_version), CONTRACT_SCHEMA_VERSION))
  }
  if (is.null(state$campaigns) || length(state$campaigns) == 0) {
    reasons <- c(reasons, sprintf("%s declares no campaigns.", path))
  }
  list(state = if (length(reasons)) NULL else state, reasons = reasons)
}


# Are the campaigns this project wants described by the contract?
contract_check <- function(state, campaigns) {
  reasons <- character()
  missing <- setdiff(campaigns, names(state$campaigns))
  if (length(missing)) {
    reasons <- c(reasons, sprintf(
      "campaign(s) requested but not published: %s. Re-ingest in camera-traps, then re-publish the contract.",
      paste(missing, collapse = ", ")))
  }
  for (c in setdiff(campaigns, missing)) {
    n <- state$campaigns[[c]]$n_rows
    if (is.null(n) || is.na(n) || n <= 0) {
      reasons <- c(reasons, sprintf("%s: contract declares no rows.", c))
    }
  }
  reasons
}


# One string per field, stable across a JSON round trip, so two descriptions can be
# compared field by field and the refusal can NAME what moved.
.canon <- function(x) {
  as.character(jsonlite::toJSON(x, auto_unbox = TRUE, digits = NA, null = "null",
                                na = "null"))
}

.campaign_fields <- function(block) {
  vapply(block, .canon, character(1))
}


# What 01_load_data.R writes after a successful load: the schema version and the
# declared block of every campaign it read, verbatim. Readable on purpose.
contract_stamp <- function(state, campaigns) {
  list(
    schema_version = CONTRACT_SCHEMA_VERSION,
    written_at     = format(Sys.time(), "%Y-%m-%dT%H:%M:%S", tz = "UTC"),
    campaigns      = state$campaigns[campaigns]
  )
}

contract_stamp_write <- function(state, campaigns, path = stamp_path()) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  jsonlite::write_json(contract_stamp(state, campaigns), path,
                       auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null",
                       na = "null")
  invisible(path)
}

contract_stamp_read <- function(path = stamp_path()) {
  if (!file.exists(path)) {
    return(list(stamp = NULL, reasons = sprintf(
      "%s not found: data/ has never been built by R/01_load_data.R under a verified contract. Run it first.",
      path)))
  }
  stamp <- tryCatch(jsonlite::fromJSON(path, simplifyVector = TRUE),
                    error = function(e) e)
  if (inherits(stamp, "error")) {
    return(list(stamp = NULL, reasons = sprintf("%s is not valid JSON: %s", path,
                                                conditionMessage(stamp))))
  }
  list(stamp = stamp, reasons = character())
}


# Differences between what data/ was built from and what is published now.
# Empty = current. Every finding names the campaign and the field.
contract_compare <- function(stamp, state) {
  reasons <- character()
  sv <- suppressWarnings(as.integer(stamp$schema_version))
  if (length(sv) != 1 || is.na(sv) || sv != CONTRACT_SCHEMA_VERSION) {
    reasons <- c(reasons, sprintf(
      "data/ was built under schema_version %s; this project now reads %d. Re-run R/01_load_data.R.",
      format(stamp$schema_version), CONTRACT_SCHEMA_VERSION))
  }
  for (c in names(stamp$campaigns)) {
    now <- state$campaigns[[c]]
    if (is.null(now)) {
      reasons <- c(reasons, sprintf(
        "%s: in data/ but no longer published -- a retired campaign still being served.", c))
      next
    }
    was <- .campaign_fields(stamp$campaigns[[c]])
    cur <- .campaign_fields(now)
    fields <- union(names(was), names(cur))
    moved <- fields[is.na(was[fields]) | is.na(cur[fields]) | was[fields] != cur[fields]]
    for (f in moved) {
      reasons <- c(reasons, sprintf(
        "%s.%s: data/ built from %s, published now %s",
        c, f,
        if (is.na(was[f])) "<absent>" else was[[f]],
        if (is.na(cur[f])) "<absent>" else cur[[f]]))
    }
  }
  reasons
}


# ── Entry points: the only two places that refuse ────────────────────────────

# For 01_load_data.R. Returns the verified state or refuses.
contract_load <- function(campaigns, path = contract_path()) {
  r <- contract_read(path)
  if (length(r$reasons)) refuse(r$reasons)
  reasons <- contract_check(r$state, campaigns)
  if (length(reasons)) refuse(reasons)
  message(sprintf(
    "Contract verified: schema_version %d, %s rows, %s stations, campaigns: %s",
    CONTRACT_SCHEMA_VERSION, format(r$state$n_rows_total, big.mark = ","),
    r$state$n_stations_total, paste(campaigns, collapse = ", ")))
  invisible(r$state)
}


# For every script that reads data/*.rds. Returns the stamp (its `campaigns` are the
# slugs 01 loaded, in retrieval order) or refuses.
contract_assert_current <- function(contract = contract_path(), stamp = stamp_path()) {
  s <- contract_stamp_read(stamp)
  if (length(s$reasons)) refuse(s$reasons, what = "data/ is not current")
  r <- contract_read(contract)
  if (length(r$reasons)) refuse(r$reasons)
  reasons <- contract_compare(s$stamp, r$state)
  if (length(reasons)) {
    refuse(c(reasons, "Re-run R/01_load_data.R before this script."),
           what = "data/ is not current with the published contract")
  }
  invisible(s$stamp)
}
