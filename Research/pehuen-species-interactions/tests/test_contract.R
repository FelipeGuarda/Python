# tests/test_contract.R
# ─────────────────────────────────────────────────────────────────────────────
# The consumer gate's own tests (MANUAL-SALUD-DATOS 10F.4, item 1): a control that
# cannot be shown to refuse is a documented convention, and those degrade.
#
# Base R only -- testthat is not in the pehuen-analysis environment and one file of
# assertions does not justify adding it. Run:
#
#     conda run -n pehuen-analysis Rscript tests/test_contract.R
#
# Exit status is non-zero on any failure. Each fixture is a contract written to a
# temp file; the last test runs a real subprocess so the EXIT CODE of a refusal is
# asserted, not just the reason text.
# ─────────────────────────────────────────────────────────────────────────────

library(here)
here::i_am("tests/test_contract.R")
suppressPackageStartupMessages(library(jsonlite))
source(here::here("R", "00_contract.R"))

.failures <- 0L
check <- function(label, ok) {
  if (isTRUE(ok)) {
    cat(sprintf("  ok   %s\n", label))
  } else {
    .failures <<- .failures + 1L
    cat(sprintf("  FAIL %s\n", label))
  }
}
any_match <- function(reasons, pattern) any(grepl(pattern, reasons, fixed = TRUE))

tmp <- tempfile("contract_"); dir.create(tmp)
fixture <- function(name, obj) {
  p <- file.path(tmp, name)
  if (is.character(obj)) writeLines(obj, p)
  else write_json(obj, p, auto_unbox = TRUE, digits = NA, pretty = TRUE)
  p
}

# The registry is published next to the contract, so every fixture contract in `tmp`
# shares this one; registry_path() resolves it from the contract's own directory.
registry_fixture <- function(dir, coords = "[-71.72707, -39.45183]") {
  dir.create(file.path(dir, "campaigns"), showWarnings = FALSE, recursive = TRUE)
  p <- file.path(dir, "campaigns", "estaciones.geojson")
  writeLines(sprintf(paste0('{"type": "FeatureCollection", "features": [{"type": ',
                            '"Feature", "properties": {"id": "CT01", "altitude_m": 1263}, ',
                            '"geometry": {"type": "Point", "coordinates": %s}}]}'), coords), p)
  p
}
registry_file <- registry_fixture(tmp)
registry_hash <- unname(tools::sha256sum(registry_file))

good <- list(
  schema_version = CONTRACT_SCHEMA_VERSION,
  stations_sha256 = registry_hash,
  n_stations_registry = 1L,
  columns = c("campaign", "camera_num"),
  campaigns = list(
    otono_2025 = list(n_rows = 8997L, n_stations = 21L,
                      stations = c("CT01", "CT02"), n_animal_rows = 707L,
                      observation_types = list(animal = 707L, blank = 7661L),
                      deployments_sha256 = "abc"),
    primavera_2025 = list(n_rows = 16904L, n_stations = 26L,
                          stations = c("CT01", "CT02"), n_animal_rows = 494L,
                          observation_types = list(animal = 494L, blank = 15811L),
                          deployments_sha256 = "def")
  ),
  n_rows_total = 25901L, n_stations_total = 27L
)

cat("contract_read\n")
r <- contract_read(file.path(tmp, "absent.json"))
check("absent file refuses", is.null(r$state) && any_match(r$reasons, "not found"))

r <- contract_read(fixture("bad.json", "{ not json"))
check("unparseable file refuses", is.null(r$state) && any_match(r$reasons, "not valid JSON"))

older <- good; older$schema_version <- CONTRACT_SCHEMA_VERSION - 1L
r <- contract_read(fixture("older.json", older))
check("older schema refuses and names both versions",
      is.null(r$state) && any_match(r$reasons, sprintf("schema_version %d", older$schema_version))
      && any_match(r$reasons, sprintf("against %d", CONTRACT_SCHEMA_VERSION)))

newer <- good; newer$schema_version <- CONTRACT_SCHEMA_VERSION + 1L
r <- contract_read(fixture("newer.json", newer))
check("newer schema refuses (no guessing that new columns do not matter)", is.null(r$state))

empty <- good; empty$campaigns <- list()
r <- contract_read(fixture("empty.json", empty))
check("no campaigns refuses", is.null(r$state) && any_match(r$reasons, "no campaigns"))

r <- contract_read(fixture("good.json", good))
check("valid contract passes", !is.null(r$state) && length(r$reasons) == 0)
state <- r$state

cat("contract_check\n")
check("requested campaigns present -> no reasons",
      length(contract_check(state, c("otono_2025", "primavera_2025"))) == 0)
reasons <- contract_check(state, c("otono_2025", "otono_2026"))
check("missing campaign named", any_match(reasons, "otono_2026"))
check("present campaign not blamed", !any_match(reasons, "otono_2025:"))
zero <- good; zero$campaigns$otono_2025$n_rows <- 0L
z <- contract_read(fixture("zero.json", zero))$state
check("zero rows refuses", any_match(contract_check(z, "otono_2025"), "no rows"))

cat("contract_compare\n")
stamp_file <- file.path(tmp, "stamp.json")
contract_stamp_write(state, c("otono_2025", "primavera_2025"), stamp_file)
stamp <- contract_stamp_read(stamp_file)$stamp
check("stamp round-trips against the same contract",
      length(contract_compare(stamp, state)) == 0)
check("stamp holds only the requested campaigns, in order",
      identical(names(stamp$campaigns), c("otono_2025", "primavera_2025")))

moved <- good; moved$campaigns$otono_2025$n_animal_rows <- 712L
moved$campaigns$otono_2025$observation_types$animal <- 712L
m <- contract_read(fixture("moved.json", moved))$state
reasons <- contract_compare(stamp, m)
check("n_rows unchanged but n_animal_rows moved -> refuses (row count alone is too coarse)",
      any_match(reasons, "otono_2025.n_animal_rows") && any_match(reasons, "707")
      && any_match(reasons, "712"))
check("the untouched campaign is not blamed", !any_match(reasons, "primavera_2025."))

dep <- good; dep$campaigns$primavera_2025$deployments_sha256 <- "changed"
d <- contract_read(fixture("dep.json", dep))$state
check("a hand-edited deployments.csv moves its sha256 and is caught",
      any_match(contract_compare(stamp, d), "primavera_2025.deployments_sha256"))

retired <- good; retired$campaigns$primavera_2025 <- NULL
rt <- contract_read(fixture("retired.json", retired))$state
check("a campaign in data/ but no longer published is caught",
      any_match(contract_compare(stamp, rt), "no longer published"))

old_stamp <- stamp; old_stamp$schema_version <- CONTRACT_SCHEMA_VERSION - 1L
check("a stamp from an older schema refuses",
      any_match(contract_compare(old_stamp, state), "schema_version"))

r <- contract_stamp_read(file.path(tmp, "nostamp.json"))
check("absent stamp refuses (data/ never built under a verified contract)",
      is.null(r$stamp) && any_match(r$reasons, "never been built"))

cat("station registry (schema 5)\n")
Sys.setenv(FMA_CANONICAL_STATE = file.path(tmp, "good.json"))
# Compared on the resolved path, not the string: on Windows tempfile() returns
# backslashes while dirname() normalizes to forward slashes, so identical() on the
# raw strings fails on a correct registry_path(). mustWork = FALSE because neither
# path exists -- the assertion is about path construction, not the file.
.same_path <- function(a, b) identical(
  normalizePath(a, winslash = "/", mustWork = FALSE),
  normalizePath(b, winslash = "/", mustWork = FALSE))
check("registry_path follows the contract's directory",
      .same_path(registry_path(), file.path(tmp, "campaigns", "estaciones.geojson")))
Sys.unsetenv("FMA_CANONICAL_STATE")
check("the registry the contract hashes passes",
      length(contract_registry_check(state, registry_file)) == 0)
moved_reg <- registry_fixture(file.path(tmp, "moved_registry"), "[-71.72707, -39.45184]")
reasons <- contract_registry_check(state, moved_reg)
check("a coordinate moved by 1e-5 deg is refused, naming both regenerating commands",
      any_match(reasons, "not the registry the contract publishes")
      && any_match(reasons, "build_station_registry.py") && any_match(reasons, "--publish"))
nohash <- state; nohash$stations_sha256 <- NULL
check("a contract that publishes no registry hash refuses (no silent pass)",
      any_match(contract_registry_check(nohash, registry_file), "publishes no stations_sha256"))
check("an absent registry refuses",
      any_match(contract_registry_check(state, file.path(tmp, "nowhere.geojson")), "not found"))
check("the stamp carries the registry hash", identical(stamp$stations_sha256, registry_hash))
reg <- good; reg$stations_sha256 <- strrep("0", 64)
rg <- contract_read(fixture("reg.json", reg))$state
check("a registry republished after data/ was built is caught by the stamp",
      any_match(contract_compare(stamp, rg), "stations_sha256"))

cat("campaign_label\n")
check("known slug labelled", identical(campaign_label("otono_2025"), "Otoño 2025"))
check("unknown slug renders as itself", identical(campaign_label("verano_2027"), "verano_2027"))
check("vectorised and order-preserving",
      identical(campaign_label(c("otono_2026", "x")), c("Otoño 2026", "x")))

cat("refusal shape (subprocess)\n")
# A refusal must be a message and exit status 2 -- not an R error (status 1), which
# a scheduled run reads as a crash. Asserted on a real Rscript process.
rscript <- file.path(R.home("bin"), "Rscript")
posix <- function(p) normalizePath(p, winslash = "/", mustWork = FALSE)
# The child inherits the parent's environment; system2(env=) is unreliable on
# Windows, so the override is set here and cleared after. The expression goes through
# a script file rather than -e so no shell quoting is involved.
run <- function(expr, contract) {
  script <- tempfile(fileext = ".R"); err <- tempfile()
  writeLines(c(sprintf('source("%s")', posix(here::here("R", "00_contract.R"))),
               'here::i_am("tests/test_contract.R")', expr), script)
  Sys.setenv(FMA_CANONICAL_STATE = posix(contract))
  on.exit(Sys.unsetenv("FMA_CANONICAL_STATE"), add = TRUE)
  status <- system2(rscript, c("--vanilla", script), stdout = FALSE, stderr = err)
  list(status = status, stderr = paste(readLines(err, warn = FALSE), collapse = "\n"))
}
res <- run('contract_load("otono_2025")', file.path(tmp, "older.json"))
check("contract_load on a wrong schema exits 2", identical(res$status, 2L))
check("... and says REFUSED with the reason", grepl("REFUSED", res$stderr) && grepl("schema_version", res$stderr))

res <- run('contract_load("otono_2025")', file.path(tmp, "good.json"))
check("contract_load on a valid contract exits 0", identical(res$status, 0L))

res <- run(sprintf('contract_assert_current(stamp = "%s")', posix(file.path(tmp, "nostamp.json"))),
           file.path(tmp, "good.json"))
check("contract_assert_current with no stamp exits 2", identical(res$status, 2L))

# A contract whose registry has drifted: the good contract, beside a different file.
drift_dir <- file.path(tmp, "drift")
invisible(registry_fixture(drift_dir, "[-71.7, -39.4]"))
invisible(file.copy(file.path(tmp, "good.json"), file.path(drift_dir, "good.json")))
res <- run('contract_load("otono_2025")', file.path(drift_dir, "good.json"))
check("contract_load on a registry that is not the published one exits 2",
      identical(res$status, 2L) && grepl("estaciones.geojson", res$stderr))

unlink(tmp, recursive = TRUE)
if (.failures > 0) {
  cat(sprintf("\n%d test(s) FAILED\n", .failures))
  quit(save = "no", status = 1L)
}
cat("\nall tests passed\n")
