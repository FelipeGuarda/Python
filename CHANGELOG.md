# Changelog

All notable changes to the FMA Python ecosystem (data-pipeline, camera-traps, plataforma-territorial, literatura-agent, schedule-agent, visualizaciones-artisticas) will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely — dated sections, grouped by Added / Changed / Fixed / Deferred / Closed-rejected. Internal personal project, no public versioning.

---

## 2026-08-25 — the producer boundary closes: a reason where there was a boolean, and four corrected claims

Scope: everything from the camera-traps boundary **inwards**. The consumer side was declared out
of scope for the day and is untouched. Producer-side debt is now one item (the visit-form
loader, V2-REVIEW 1.14, deferred with its shape decided).

### Changed — `camtrap/deployments.py`: `media_status`, because a boolean was asserting something false

`deployments.csv` carried `has_media: bool` plus a hand-built note that read *"deployed per the
field record, no images in the canonical table"* for otoño 2025's five image-less deployments.

Felipe checked the NAS: **four of the five were recording the whole time.** Their media is
video, held for that campaign in a separate tree
(`.../CAMPAÑAS DE RECOLECCION DE IMAGENES/Otoño 2025/Videos`). Only **CT21** recorded nothing,
and its own field note said so a year ago — *"La cámara encendió luz led pero no prendió la
pantalla, SD vacía. Se instaló otra cámara trampa."* Two more, CT22 and CT25, were found failed
at retrieval from interior humidity.

**`has_media` is a measurement; the reason it is false is a separate fact and only the reason
decides a denominator.** Those four cameras contribute REAL effort while their detections are
unreadable from `observations.parquet` — put their camera-days in a stills-based denominator and
every otoño 2025 rate is biased downwards by a plausible-looking amount, which is
`DATA-HEALTH-MANUAL` §6.3's "a plausible number from two mistakes" and strictly worse than the
visible 26-vs-21 it replaces.

- New `media_status`: `in_canonical` · `video_only_offline` · `card_failure` · `unexplained` ·
  `no_field_dates`, each documented with the denominator it licenses.
- New declared data file **`data/campaigns/media_absence.csv`** — reason, evidence (the field
  note verbatim), media location on the NAS, who checked and when. The verdict lives with its
  evidence rather than in code.
- **Two denominators, both published, neither default:** otoño 2025 is **3,816 camera-days over
  21 stations** for anything read from the canonical table, 4,318 over 25 for anything counting
  video. `CANONICAL_STATE.json`'s `camera_days` is the first, deliberately — it is the one that
  pairs with `n_rows`.
- Fail-visible in both directions: a station with no stills and no declaration reports
  `unexplained` (an unexplained gap is a question nobody asked, and absorbing it into an effort
  figure is how a wrong denominator looks right — §4E.3 applied to media), and a misspelled
  reason **raises** rather than reading as a licence to count the days.

**Residual, stated rather than silently corrected:** CT22 and CT25 stopped sampling at an
unknown point before their recorded end dates. Felipe's ruling — the field dates stand as
registered, the video is not re-read to second-guess them. For those two, `field_days` is a
ceiling, not a measurement.

### Added — capture-order evidence for every station (V2-REVIEW 1.4 / manual B6)

`timestamps.py` gained a *"Capture-order evidence, all N station(s)"* section in
`timestamps_audit.log`. The evidence tier was already computed for every station and then
**discarded for the ones that passed**, so "this station has no manifest" and "nobody looked for
one" were indistinguishable in the record.

**The review's own figures were wrong in both directions, because two measurements were being
conflated.** Ordering evidence exists for **3 / 4 / 4** stations of 21 / 26 / 27 — while a
manifest FILE covers **21 of 21** otoño 2025 stations. The difference is §3.4's corollary: that
campaign's folders carry hand-made names (`M7`, `M5`) and only a camera-created DCF folder is
ordering evidence, so the manifest row count was never the number to quote. The gap is not a
defect (§4B.3): six station-campaigns fail to order and every one has a clean clock.

### Fixed — four claims in `V2-REVIEW.md` that were false, all measured

| claim | reality |
|---|---|
| `pv_2025_2026` "kept as provenance" (§0 cond. 3, 1.1) | **deleted** in `c295999`, 2026-08-20, disk and git. Asserted otherwise for three sessions. Deletion confirmed as the decision |
| field record "audited for coordinates only" (1.7) | **23** date-flagged rows against **2** coordinate-flagged. Real gap: six columns never *collected*, `camera_datetime_observed` at **0 / 107** |
| two stations lack folder evidence (1.4) | 3 / 4 / 4 of 21 / 26 / 27, per above |
| `provenance.py` needs re-running / is unwired (1.8) | it is the **fourth flatten precondition**, fatal, pre-move. Re-run anyway: **0** multi-story stations over 35,807 rows |

### Removed — the last of `pv_2025_2026`, and one duplicate workbook

- `data/campaigns/label_conflicts_primavera_vs_pv_2026-05-27.csv` **and its live reader** —
  `load_conflicts()`, `CONFLICTS_CSV` and two output columns in `list_ciervo_guina_images.py`.
  Deleting the file alone would have left pv logic running against an absent input.
- `Anual-reports/2025/data/manual_review_ciervo_guina.md` (stale; keyed to `TC*_M*.2` and a pv
  column). The regenerable `.csv` beside it stays.
- `Anual-reports/Registro de monitoreo CT.xlsx` — an undeclared second copy of the legacy
  workbook, not marked NO LLENAR, cited by `Anual-reports/2025/README.md` as the install-date
  source. Its `Registro de instalacion` sheet was verified **identical** to the original, so it
  was a duplicate that could only ever drift.
- Two stale precedence comments rewritten (`camtrap/observations.py`,
  `Anual-reports/2025/py/apply_verdicts.py`). Measured while rewriting: **0 duplicate
  `DEDUP_KEY` rows across all 35,807**, and the 31 recurring `(camera_num, file_name)` pairs
  share no datetime — counter recycling between years, which must NOT be deduplicated (§3.5).

### Changed — legacy consolidation, and one hand-maintained list labelled as such

- `data/campaigns/legacy/` created; the historic workbook moved into it. Code impact was one
  line (`setup/build_field_notes.py`).
- `scripts/verify_order.py`'s `UNVERIFIED` renamed **`UNVERIFIED_LEGACY_2026_08`** and
  documented as a frozen historical record, not the live state. Felipe's call, and the new audit
  section immediately proved it necessary: the two **already disagree** — order is not
  established for otoño 2025 CT04, primavera CT22 and otoño 2026 CT27, none of which is on the
  list. It is not wrong (it asks a narrower question), but anyone reading it as current would be.

### Closed-rejected — a standalone `python -m camtrap.provenance` runner

Designed and declined. The check already runs fatally at the flatten, before a single file
moves, which is the one moment it can prevent the damage; a post-hoc run can only re-examine
attributions already made, and a second entry point onto the same function is a shallow module.
The one-off verification is recorded in V2-REVIEW 1.8 instead of shipped as code.

### Deferred — the visit-form loader (V2-REVIEW 1.14), highest-priority open item

The `Visitas` sheet has **0 filled rows** and the next salida is unscheduled, so the loader
earns a proper design pass rather than a rushed one. **It expires the day terreno returns.**
Shape decided: `field_notes.csv` moves to the new 20-column form shape, the 107 legacy rows
migrate into it, `FieldRecord` is rewired off `clock_state` / `camera_replaced`, and the current
CSV is snapshotted into `legacy/` first.

### Verification

241 camera-traps tests pass (was 235). Export gate `full_category_sweep` on all three campaigns
from the repo; `python -m camtrap.canonical_state` exits 0. Re-running `timestamps.py` on all
three campaigns produced **byte-identical `observations.parquet` files** — the only change in
`CANONICAL_STATE.json` is three `deployments_sha256` values. No number moved.

⚠️ **Consequence for data-pipeline:** `deployments.csv` changed, so the warehouse is stale by
design and `run_fetch.py --ct-check` will refuse. Consumer-side work for another session.

---

## 2026-08-24 — the consumer boundary closes: one station registry, the `ct_*` rebuild, and a contract that is finally read

Eight of the fifteen open items from the 2026-08-20 re-audit are closed, and they are **every
one on the data-pipeline / platform side**. §0-bis's finding held exactly: the chain from card
to canonical table was already enforced and tested; every surviving defect sat where
responsibility changed hands. One session spent entirely at that boundary cleared it.

### Added — `camera-traps/setup/build_station_registry.py` + `tests/test_station_registry.py`

`data/campaigns/estaciones.csv` now **owns station identity**. `stations.yaml` and
`camera_trap_stations.geojson` are generated from it and must not be hand-edited.

The defect: `stations.yaml` held 26 stations against the other two registries' 27, so **CT27's
315 otoño 2026 images ingested with no coordinates** and nothing raised — the same class as the
CT26 error that reached the platform and came back as a 19 km displacement.

Measured before changing anything: **all three files agreed on every value they shared** —
coordinates, `grid_id`, elevation, across all 26 common stations, zero discrepancies. The defect
was one missing row plus nothing to stop it recurring. V2-REVIEW 1.6 had implied a three-way
disagreement; there wasn't one.

- **Felipe chose the owner on evidence, not seniority.** He asked whether `estaciones.csv` was
  the original; it is not — it is the *newest* (2026-08-17 vs the March pair), and the true
  original is `CT ID and coordinates.xlsx`, which is none of the three. It owns anyway: all 27
  stations, canonical `CT##` grammar, the columns the visit form writes, already read by
  `camtrap/stations.py`, and it lives in the producer rather than a consumer.
- **One canonical spelling everywhere** (Felipe): `CT01`..`CT27`, replacing the artifacts'
  `TC-01`. Joins are on the integer `tc`, so nothing broke — only labels moved.
- **`sd_card` dropped.** The `M##` grid-module tag from the old folder names: not an SD card,
  not unique (`M15` was both CT11 and CT18), and its last reader had stopped using it. Closes
  the **S58** question by removing its subject.
- **The test is stronger than specified.** It asserts the committed artifacts equal a fresh
  render, not "all three agree on count and coordinates to 5dp" — that check restates the
  projection in a second place and passes vacuously on any field it does not enumerate, which
  is exactly how `sd_card` lived in the artifacts and in no test for five months.
- The generator rewrites `stations.yaml` only from `camera_traps:` down and refuses if another
  top-level key follows, so `reserve:`, `weather:` and the header comments survive byte for
  byte. CT26's coordinate-error note now renders from the registry's `notes` column instead of
  being a hand-written comment that can drift.

### Added — `data-pipeline/src/recovery.py` + `data/recovery/*.parquet`

**The Windows↔Linux question is permanently answered.** `weather_station` (264,943 rows,
2018-09-21 → 2026-04-13) and `weather_forecast` (4,343) cannot be refetched — a CR800 pull is a
point-in-time read and Open-Meteo serves a horizon, not an archive — so they are committed as
Parquet. Any machine can now rebuild the warehouse from the repository alone.

- **Partitioned by year.** Parquet is opaque to git and this runs on every poll; a year that has
  ended never changes, so only the current year's blob (632 KB) is rewritten rather than 17.5 MB.
- **`export` and `restore` are separate verbs and there is no `sync`** — guessing the direction
  on irreplaceable data is how the empty copy overwrites the good one. `restore` refuses when the
  database has more rows than the archive; `export` refuses when a year-file exists that the
  table has no rows for.
- **Verified** by building an empty database from `schema.sql` and restoring: 264,943 rows, all
  41 columns in order (including the 33 dynamically-added TOA5 columns), first five rows
  identical, mean temperature equal to six decimals.

### Added — `data-pipeline/src/parsers/canonical_ct.py`, `src/canonical_gate.py`

**`ct_*` rebuilt from the canonical parquets: 35,807 rows across 74 deployments** — 21 + 26 + 27
stations, the real deployment history. Reconciles exactly against the published contract,
including `animal 2,522 / blank 31,090 / human 1,424 / unknown 521 / vehicle 250` and the
3,359 human- vs 32,448 machine-classified split. The 4,013 rows with no clock are preserved:
presence needs a station, not a clock.

Replaces the ingest deleted on 2026-08-20 for re-deriving five decisions `camtrap.observations`
already owned. The projection re-derives **nothing**.

**The contract gate (V2-REVIEW §4) is now read at both ends.** `run_fetch.py --ct-check` reports
staleness, writes nothing and exits 1. It reads `CANONICAL_STATE.json` as a *file* and does not
import `camtrap` — importing the producer would have the check running the producer's code
against the producer's data, where it could only agree with itself. It fingerprints the whole
campaign description, not just row counts: the 815-row review repair moved `observation_types`
while leaving `n_rows` untouched, which a row-count check cannot see.

### Fixed — the platform was serving orphaned data

`ct_deployments` / `ct_media` / `ct_observations` held **54 / 2,948 / 2,948** rows, every one
`source='timelapse_reviewed'` (a parser deleted 2026-08-20), under pre-flatten identity
(`oto_o_2025_CT07`, ñ mangled), keyed on Timelapse GUIDs, and with `pv_2025_2026` still present
as a campaign. The platform now serves 2,522 real detections, 30 species, 27 stations, 3
campaigns.

`occupancy_pct` divided by every station in `stations.yaml`. Wrong three ways: the grid was built
up over time so early campaigns were understated; the figure moved whenever a station was added
to the registry, including stations that did not exist during the campaign; and it made a
consumer depend on the registry's *size* for a quantity the observation data already answers. Now
counted from `ct_deployments`. `n_stations_deployed` is returned so the denominator is visible,
and `occupancy_pct` is `None` — not `0` — when nothing is deployed.

### Fixed — CT27's install date, and its clock cleared

Install **2025-12-11** (Felipe), resolving the 2025-11-12 / 2025-12-11 day-month transposition.
The GPS waypoint reads 15:52:56 against a first frame of 12:49:01 — three hours apart, so either
UTC-vs-local or a 3 h-slow camera. **Decided against the retrieval trip:** CT27's last frame
(2026-05-14 14:32:04) sits in correct sequence between CT17 (14:07:45) and CT21 (15:15:15); a
3 h-slow camera would have read ~11:32 and landed out of order between CT10 (09:58) and CT15
(11:51). The clock is sound, the waypoint is UTC, and all 315 CT27 rows are time-admissible.

### Changed — three of V2-REVIEW's own specifications were wrong

- **2.8's mandated key cannot be a primary key.** `DEDUP_KEY` includes `datetime`, which is null
  in 4,013 of 35,807 rows. `(campaign, camera_num, file_name)` is unique across all of them and
  never null.
- **2.3's column contract omits `campaign`**, which the platform queries in three places and
  which existed only because `ensure_columns()` had added it dynamically — so `schema.sql` and
  the review had both drifted from the live table.
- **2.3 says timestamps are UTC.** A camera clock is wall time of unknown accuracy; there is no
  instant to recover, `TIMESTAMPTZ` would invent one (ambiguous twice a year at the DST
  boundary), and `HOUR(eventStart)` would depend on the reader's session timezone rather than on
  what the camera saw. `ct_*` are naive local; `weather_station` stays `TIMESTAMPTZ` because a
  datalogger reading *is* a known instant.

Also: `duckdb_tables().estimated_size` **lies** — it reported 24,665 / 12,222 for
`ct_media` / `ct_observations` against an actual 2,948 / 2,948. Use `COUNT(*)`.

### Removed — `literature`

0 rows, no reader in this monorepo (literature-agent is standalone and mails its summaries). The
DDL is deleted rather than left in place, because `init_schema()` runs on every connect and a
`CREATE TABLE IF NOT EXISTS` would recreate the empty table forever. Column list preserved as a
comment.

### Deferred — `data-pipeline` tests

`src/` is 1,642 lines with **no test suite at all**, while now carrying four modules whose
guarantees rest on having been run once by hand. Designed in `data-pipeline/docs/TEST-PLAN.md`
and deferred by Felipe so documentation could catch up first. stdlib `unittest`, no new
dependencies — **pytest is installed in neither environment**, which is the same gap that let a
claimed "152 tests" go unverified on 2026-08-18.

### Added — the clock verdicts cross the boundary: four columns and one view

The question that produced this: *is the warehouse now clean for a project that was not in
these sessions — an annual report — to call, without ifs or buts?* It was not, and the reason
sat at the same boundary everything else did. The rebuild carried the **rows** and dropped the
**adjudication**. `valid_date`, `valid_time_of_day`, `valid_effort` and `repair_method` are in
the canonical parquet and were in no `ct_*` column, so the entire output of the clock-repair
work was destroyed at ingest.

- **4,094 of 35,807 rows** are inadmissible on at least one axis. A consumer could recover
  4,013 of them from `timestamp IS NULL` and had no way at all to find the other **81**:
  `repair_method = 'offset_from_last_real_proxy'` recovers a trustworthy *date* from a
  neighbouring segment but cannot recover the *time of day*, so those rows carry an
  ordinary-looking timestamp. **33 are animal rows**, and they were being bucketed into the
  platform's hourly activity histogram. `eventStart IS NOT NULL` was never a sufficient test.
- `ct_observations` gains `validDate`, `validTimeOfDay`, `validEffort`, `repairMethod` —
  **copied, never re-derived**, the same rule that governs the rest of `canonical_ct.py`.
- `ct_observations_time_admissible` (**31,713 rows**, 2,070 of them animal) is the view any
  time-of-day or seasonal analysis must read. Created by the rebuild rather than declared in
  `schema.sql`, because `init_schema()` runs on every connection and a view cannot reference
  columns the current database does not have yet.
- **It is deliberately not a general `_admissible` view**, which is what was first proposed and
  is wrong: **419 animal rows have no timestamp at all** and are perfectly valid *presence*
  records. Filtering all three flags would silently under-report every species list. The flags
  are exposed individually so each analysis picks its own predicate; the view covers only the
  case that was actually being got wrong.
- `_reconcile()` gained the invariant the view rests on: no row may claim `validTimeOfDay`
  with a NULL `eventStart`. It fails the rebuild rather than letting the view admit rows it
  cannot order.

### Fixed — `ensure_columns()` would have stored the flags as strings

`db.py::_sql_type()` had no boolean branch, so pandas' nullable `boolean` fell through to
`TEXT` and stored `'true'`/`'false'`. It *looked* correct because DuckDB implicitly casts a
VARCHAR inside `WHERE`; `typeof()` and any arithmetic aggregate told the truth. The datetime
branch was left alone on purpose — `ensure_columns` is shared with `weather_station`, whose
timestamps are genuinely tz-aware.

### Fixed — the platform's two time-of-day charts

`/diel-activity` and `/overlap`'s hourly query now read the view. The histogram moves from
2,103 to 2,070 rows; hour 09 loses 13. Species totals, `last_seen` and occupancy stay on the
full table — presence does not need a clock, and `validDate` is true for all 81 rows, so a
date-scale `MAX(eventStart)` is unaffected.

### Added — `data-pipeline/README.md` → *Reading the camera-trap tables*

Three things about `ct_*` are not code defects but will produce wrong numbers if a consumer
guesses, so they are now written down where a consumer will look — in **data-pipeline**, not in
camera-traps' manual, because camera-traps must not learn that DuckDB exists:

- **`campaign` is a retrieval batch, not a season.** The ranges overlap heavily (otoño 2025
  runs 2024-10-09 → 2025-06-10; primavera 2025 runs 2025-05-14 → 2026-01-14; otoño 2026 runs
  2025-11-21 → 2026-05-15) and **2,887 otoño-2026 rows predate primavera-2025's last frame**.
  Group by campaign to describe fieldwork; slice on `eventStart` to describe a year.
- **Effort denominators are not in the database.** Deployment windows are observed-media, not
  field-recorded, and **9 of 74 deployments have none at all**. Camera-days are biased low and
  undefined for those 9. Blocked on V2-REVIEW 1.14.
- **Rows are images, not events.** `eventID` and `count` are NULL for all 35,807 rows.
  "2,522 animal detections" counts *photographs*.

### Corrected — the cross-campaign duplication does not exist

Earlier notes carried a warning that image counts were roughly doubled by cards read across two
campaigns. Measured: **31 `(station, file_name)` collisions, 0 of which share a datetime.** They
are `MMDDnnnn.JPG` filenames recycling across years — `10280073.JPG` is Oct 28 2024 in one
campaign and Oct 28 2025 in another — and the rebuild's `(campaign, station, file_name)` key
separates them correctly. The warning is withdrawn.

### Added — `camera-traps/camtrap/deployments.py`: effort becomes a published number

`field_notes.csv` has dated both ends of nearly every deployment since the legacy migration,
and **nothing ever read them**. Every consumer inferred "how long was this camera watching"
from its first and last photograph, which is circular — a camera whose battery died after two
months looks like it was *deployed* for two months. Measured: CT12 was in the ground **219
days** and photographed across **61** of them (a 3.6× overstatement of its detection rate),
and CT08 and CT10 have no observed window at all because their clocks failed, while the field
record dates both.

`camtrap/deployments.py` pairs the visit that put a card in the ground with the visit that
pulled it out and writes `data/campaigns/<campaign>/deployments.csv`:

| campaign | deployments | with images | camera-days |
|---|---|---|---|
| otono_2025 | 26 | 21 | 3,816 |
| primavera_2025 | 26 | 26 | 5,178 |
| otono_2026 | 27 | 27 | 3,981 |

Two silent failure modes are held by fixtures rather than by care:

- **`FieldRecord.window()` must not be used here.** It returns `[opening − 3 d, closing + 3 d]`
  so a clock anchor can be validated against a window it may sit just outside of. Applied to
  effort it adds six days to every camera in the reserve, and it is the obvious method to reach
  for.
- **Camera-days are date-scale.** `FieldRecord` stamps a visit with no recorded time at
  `ASSUMED_VISIT_HOUR`, so subtracting datetimes truncates whenever the two ends disagree about
  the hour — CT01's install is timed 15:13 and its retrieval is not, which read as 168 days
  instead of 169. Caught during implementation; 33 camera-days across the three campaigns.

Stations deployed with **no** images are published with `has_media = false` rather than
dropped, so the effort stays visible while the discrepancy is resolved.

### Fixed — CT27 had no deployment window at all

CT27 appears on no install sheet (it entered the grid late) and was **omitted from *Registro de
revisión Mayo 2026***, which holds 26 rows — so the field record opened its deployment and
never closed it, and it was the one deployment out of 74 that would have fallen back to an
observed window.

- **Opening: 2025-11-12 → 2025-12-11.** Not a new judgement — the day/month transposition was
  resolved earlier the same day and recorded in `otono_2026/deployment_anchors.csv`, but the
  adjudication never propagated back to the field record, which still carried the ambiguous
  date flagged `VERIFY`. Corroborated by CT27's own first frame at 2025-12-11 12:49:01.
- **Closing: 2026-05-14**, reconstructed from retrieval-trip order — CT27's last frame
  (14:32:04) falls between CT17's (14:07:45) and CT21's (15:15:15), both retrieved that day.
  Marked `source_sheet = (reconstructed)` rather than attributed to a sheet it is not on.

**74 of 74 deployments with images now carry a field window**, asserted by a test.

### Changed — `CANONICAL_STATE.json` is `schema_version: 3`

Each campaign now also describes its published effort: `n_deployments`,
`n_deployments_with_media`, `camera_days`, and a SHA-256 of `deployments.csv`. A wrong row
count is visible because a species appears or does not; a wrong denominator silently rescales
every rate in a report and nothing looks broken, so effort belongs *inside* the thing consumers
verify rather than beside it.

Nothing in `data-pipeline/src/canonical_gate.py` had to change to *check* it — `fingerprint()`
already hashes the whole campaign description, so a hand-edited `deployments.csv` moves its
sha256, moves the fingerprint, and the gate reports the database stale. Only
`SUPPORTED_SCHEMA_VERSION` moved, 2 → 3, which is the deliberate-read the constant exists to
force: the bump made `--ct-check` refuse the live database until it was rebuilt, exactly as
designed.

235 camera-traps tests pass, up from 226.

### Known — five otoño 2025 stations are deployed and have no images

`field_notes.csv` records **CT21, CT22, CT24, CT25 and CT26** as installed 2025-02-04/05 and
collected 2025-06-05/11 — roughly **623 camera-days** — and not one appears in that campaign's
`dcim_manifest.csv`, `ImageData_total.csv` or reviewed CSV; the manifest's deployment list is
CT01–CT20 plus CT23. So "the grid grew over time" does not explain otoño 2025's 21-vs-26.
Felipe is checking the NAS (2026-08-24); the working hypothesis is that the cameras were added
late and the folders are empty. Published with `has_media = false` pending that.

### Known, and flagged rather than fixed

- **`exports/Primavera-verano 2025-2026/`** still names its station directories `TC10_M3_2`.
  `station_summary` matches images by `locationName`, now `CT10`, so that campaign's thumbnails
  will not resolve. Already broken before this (the directory is pv-named), but now explicit.
- **The CR800 has not been polled since 2026-04-13** — four months of silence.
- **pehuén's `sd_card` removal is committed but unexecuted.** Felipe does not want pehuén run on
  the Linux box; without that one-line `select()` change the script would error against the
  regenerated GeoJSON.
- Two 63 MB DuckDB backups are tracked in git while the live database is gitignored. New
  backups are now ignored; removing the existing two is a history rewrite and a separate call.

---

## 2026-08-20 (later) — camera-traps: the data-health manual, and a re-audit that says the chain is not closed

### Added — `camera-traps/docs/DATA-HEALTH-MANUAL.md` (1,953 lines)
The end-to-end protocol with its reasoning attached, written field-facing: every rule states
the invariant it protects, **what analysis becomes impossible if it is skipped**, and whether
that is recoverable afterwards. Ten parts, from the conventions that must be fixed before any
fieldwork through to what the whole apparatus makes possible. Assembled from `V2-REVIEW.md`,
`HANDOFF-clock-repair.md`, the README's campaign history and fifteen session logs
(2026-06-25 → 2026-08-20), with the code as the check on all of it.

Two tables in it are new knowledge rather than assembly:
- **The recovery matrix** (§4E.8) — nine datetime error classes x five levels of available
  evidence, and what each combination restores. It makes three things explicit that were
  scattered: **class 4 (corrupt date registers) is unrecoverable with any anchor**, because an
  anchor corrects an offset and there is nothing to offset; classes 7 and 8 (piecewise offset,
  systematic shift) are unrecoverable **from the images at all** and depend entirely on the
  field record; and a date-only visit costs the time-of-day column everywhere.
- **The admissibility matrix** (§6.10) — every analysis against the three validity axes, the
  unit of analysis and the minimum sample size. Presence requires **none** of the axes, which
  is the whole argument for keeping flags rather than deleting bad dates; and **every
  count-based analysis requires episodes** — there is no row in the table for which counting
  images is correct.

Published as a private artifact for the field team and the foundation.

### The re-audit — fifteen open items, four of them unenforced guarantees
Checked against the working tree rather than against V2-REVIEW's own checkboxes, because
writing "the pipeline guarantees X" made it necessary to confirm that it does.

**Tier A — described in the manual, not enforced in code:**
1. **The station registry still disagrees with itself** (1.6). `stations.yaml` **26** ·
   `camera_trap_stations.geojson` **27** · `estaciones.csv` **27**. The 26-station file is the
   documented "single source of truth", so **CT27's 344 images ingest with no coordinates**.
   1.6 already states that the agreement test is what makes the CT26/CT27 class impossible to
   repeat; the test does not exist. Highest priority — the occupancy fix inherits it.
2. **The canonical contract is published but unverified downstream** (§4). The producer half
   works; `grep -r CANONICAL_STATE data-pipeline/` returns nothing. A contract nobody verifies
   is a comment — the exact phrasing already in `canonical_state.py`, now true of the gate.
3. **The field form has no loader.** `build_visit_template.py` writes the workbook; nothing
   reads it back (`build_field_notes.py` is the one-time legacy migration). The entire field
   protocol currently depends on an undocumented hand-transcription step. Never had a numbered
   V2 item, which is why it stayed invisible.
4. **Effort denominators wrong in the dashboard** — `detections.py:381` divides by all 27
   stations; otoño 2025 ran 21.

**Tier B:** the `ct_*` rebuild (2.1/2.2/2.3/2.5/2.8) · pehuén's Windows paths · figures not
re-rendered, and when they are, otoño 2026 falls out of `05_spatial_distribution.R:249` and
the `02_detection_summary.R` labellers, so that fix must be re-rendered *separately* from the
data change · `field_notes.csv` audited for coordinates only (57 of 106 rows flagged) ·
`provenance.py` not re-run on the re-ingested primavera · manifest coverage not stated per
campaign (1.4) · CT27's install datable from `CT 27.kml` (2025-12-11 15:52:56) and unrecorded ·
three missing regression fixtures (1.10).

**Tier C:** two superseded data files on disk · a stale pv comment in `apply_verdicts.py:143` ·
otoño 2025's video existence never confirmed · empty `count` · the seasonal puma orphan.

### The finding worth keeping
Every remaining item sits at one of the two **boundaries** — the field record coming in, or a
consumer going out. Nothing is left in the middle of the chain. That is the same shape as
every incident this refactor was built around: the defects live where responsibility changes
hands, which is exactly where nobody owns the check.

---

## 2026-08-20 — camera-traps + data-pipeline + pehuén: the stale-code sweep

A full audit of the camera-trap chain (16,752 lines across camera-traps, data-pipeline's
CT paths, Anual-reports and pehuén) replaced the pattern of finding one pre-canonical file
per session. Report: `SecondBrain/Reviews/review-state-camera-traps.md`, 17 findings.

### The root defect: two files both claimed to be the reviewed truth
`new_labeled_data_corrected.csv` carried an **unresolved** `observationType` — every row
`animal`, including the 815 where the reviewer had written that the frame holds no animal
(111 otoño 2025 / 250 primavera / 454 otoño 2026) — with no `review_resolution` column to
make the disagreement visible. Verified not to have reached pehuén's published numbers
(pehuén also required a non-empty `scientificName` and those rows are blank there, so 0 of
815 survived), but it was luck, not a control. **The file is now gone**; pehuén reads the
canonical parquet.

### pehuén was reading a retired review pass instead of a campaign — results changed
`R/01_load_data.R` loaded `pv_2025_2026` as its spring campaign and **never read
`primavera_2025` at all**. pv is a second review pass over primavera's cards, made in
April and superseded in August. Rewritten to read `observations.parquet`. Spring detections
moved substantially:

| species | was (pv) | now (primavera) |
|---|---|---:|
| Liebre | 230 | **161** |
| Zorro culpeo | 59 | **82** |
| Perro | 46 | 35 |
| Puma | 13 | 10 |
| Guiña | 8 | 7 |

Totals 850 → 789 records. Otoño 2025 and 2026 unchanged, as expected. All pehuén figures
re-rendered. The rewrite also deletes three per-campaign station-ID parsing blocks, the
SD-card cross-validation (its subject no longer exists) and the `"No reconocible"` string
filter — all owned upstream — and gives pehuén `valid_effort` for the first time (524
records sit at stations whose effort denominator is unknowable).

### `--ct` retired rather than repaired
`timelapse_reviewed.py` (196 lines) re-derived five decisions `camtrap.observations` owns
and disagreed on 515 live rows. Its campaign list named a primavera file that does not
exist, so the loop skipped primavera and ingested pv **as a campaign**. Deleted, with
`camtrap_dp.py` (81 lines, no input folder has ever existed here — its column mapping is
preserved as the `ct_*` contract in V2-REVIEW 2.3) and `dedup_primavera_2025.py` (98 lines,
premise dissolved). `ingest_all_ct_campaigns` now raises `CameraTrapIngestNotRebuilt` with
the reason. V2-REVIEW §2 went from BLOCKED to unblocked: 2.4, 2.6 and 2.7 closed.

### Added: a contract that is actually checked
`camtrap/canonical_state.py` publishes `data/CANONICAL_STATE.json` — schema version,
columns, per-campaign row/station/animal counts — and both consumers verify it before
reading. This exists because on 2026-08-19 the canonical tables went from 3,359 to 35,807
rows and **not one consumer raised an error**. Publishing stays a separate act from
ingesting, deliberately: if `timestamps.py` re-published automatically the check would
agree with whatever was just written.

Two columns added while every parquet was being rebuilt anyway (schema_version 2):
`observation_comments` (the reviewer's verbatim text — the 815-row defect was only
findable by reading it) and `classification_probability` (CLIP confidence, 509/502/800
distinct values, unrecoverable once the Timelapse export is gone). `count` and `eventID`
were measured 100% empty in all three campaigns and deliberately left out.

### `campaign_dir` is an argument, not a config key
It pointed at `Desktop/Otono_2025/SynologyDrive` — the first campaign, three campaigns on —
and the path still existed, so a wrong run looked normal. Both consumers
(`run_classification.py`, `phase1_labeling/app.py`) now take a **required** `--campaign-dir`
with no default. Felipe's call.

### Also deleted
`megadetector_campaigns.py` (imported `wildlife_detector`, never in environment.yml, so
unrunnable) and `merge_videos_to_fotos.py` (undocumented; merges video into Fotos, contrary
to the video-exclusion policy). **36.0 MB of legacy data**: pv's directory, the legacy
`CamtrapDB_*` V1 projects, three zero-reader CSVs, three `figures_pre_*` snapshot
generations, the pv-named `exports/` dir, and `records_clean`/`events_clean` (outputs of a
superseded baseline, since regenerated).

### Corrected claims
- `README.md:432` told the operator to sweep `empty`/`person` — the exact vocabulary a
  regression test asserts the gate refuses, and the defect that left 584 rows uncounted.
- The README's claim that pv "must be merged rather than discarded" was **measured false**:
  all 186 pv-only rows exist in primavera's export, 176 carry no species, and all 5 species
  the other 10 name are already recorded in the live campaigns.
- V2-REVIEW 2.8 claimed Timelapse IDs are "UUIDs regenerated on every parse". They are not
  regenerated by the parser at all; they are per-**project** GUIDs (primavera's two `.ddb`
  files share 2,387 filenames and 0 `mediaID`s), so `ct_*` keys must derive from the image.
- `HANDOFF-clock-repair.md` said "no implementation code written yet"; `clocks.py` is 757
  lines with 605 lines of tests. It is the specification of record now.
- `data-pipeline/README` said the camera-trap parser was removed. It was not, until today.
- `species.py` said 27 CLIP species + 4 extras; it is 35 (29 CLIP + 6 reviewer-added).

### pehuén: admissibility separated from the unit of analysis

`R/01_load_data.R` ended with `filter(!is.na(datetime))`, so every downstream script
inherited the strictest rule whether or not it asked. Correct for activity and overlap,
**wrong for presence/absence**, which needs a station and not a clock. Puma is recorded at
8 stations and the spatial maps showed 6. New `R/00_admissibility.R` owns
`admissible(records, "place"|"time")`, `presence()` and `episodes()`; the load-time filter
became a `time_admissible` column. `records_all.rds` 789 → **1,112** records.

Stations recovered in the presence panel: Puma 6→8, Guiña 8→10, Jabalí 7→9, Liebre 10→11,
Perro 12→13. `camtrapR::detectionMaps()` could not be used for that panel — its
`recordTable` requires `DateTimeOriginal`, so it structurally cannot hold the records that
must be included — so presence is now built directly with sf/ggplot.

**The bubble maps had also been counting IMAGES.** A camera fires 2–3 frames per trigger,
and the distortion is not uniform: ratios run 1.7× (Guiña) to 4.9× (Jabalí), so image
counts *reorder* species rather than rescaling them. Jabalí against Guiña read 84:22 by
images; by episodes it is 17:13.

**One rule, one implementation.** The first `episodes()` measured the gap from the previous
detection; `record_table`'s existing `keep_after_min_gap()` measures from the last
*retained* one (0/20/40 min is two events, not one) and groups by campaign. Its rule is the
standard camtrapR definition and correct, so `episodes()` adopted it and
`filter_independent_events()` now delegates. Both counts agree at **327**.
`04_temporal_overlap` output is byte-identical to the pre-fix run — the control that the
change moved only what it should.

Also: the loader had been reading `pv_2025_2026` — a retired April review pass — as its
spring campaign and **never reading `primavera_2025`**. Spring detections moved: Liebre
230→161, Zorro culpeo 59→82; totals 850→789 (before the admissibility change re-added the
place-admissible rows). `nanoparquet` + `jsonlite` added to `environment.yml` and
`setup_packages.R`.

### otoño 2025: CT15 and CT16 retrieval dated from the next campaign's card

A service visit is a boundary between campaigns, so the same trip is photographed twice —
and the **fresh card is the better witness**, because the old camera may have lost its clock
in the field. Felipe confirmed a technician in primavera's CT15 `06090001.JPG`
(2025-06-09 15:46) and CT16 `00300001.JPG` (16:19 — 33 minutes later, walking distance, so
the two corroborate each other).

`last_real_proxy` anchors on CT15 segment 13 and CT16 segment 8 (`segment_index` pinned,
since several segments share a camera-time start): **33 animal records gain dates**,
`date=TRUE tod=FALSE`. otoño 2025 datable animal records 289 → **322**. Report 642 → **651**
records, 262 → **269** events; the gap between 33 and 9 is 24 birds dropped by rule 4.

**pehuén is unchanged, correctly and for two reasons:** the recovered rows are
`tod=FALSE` so they cannot enter activity or overlap analysis, and for presence they were
*already* counted, because place-admissibility never depended on the clock.

**Method correction (Felipe).** A boundary frame is a witness only if it is at a working
hour AND shows a person. primavera CT19's `06060001.JPG` is 2025-06-06 03:37 with nobody in
it — a wildlife trigger on an already-deployed camera, so an upper bound, not an anchor.

**Recorded as checked-and-empty so they are not re-checked:** otoño 2025 CT19
(`03100001-3.JPG` blank, no MegaDetector detection of any kind on the `0310` prefix, 66 of
68 segments incoherent, so a single notebook date would not repair it either) and otoño 2026
CT18 (bogus stamps, no human; the May 13–15 retrieval is established from 26 sibling
cameras, so what is unknown is the date the camera *died*, ~2026-01-31 by filename).

**CT03 must not be scrapped.** Its entire "clock incoherence" is 3 frames at 23:59:28–29
where the filename rolled past midnight before the stamp did — all within 32 seconds of
midnight, against 318 frames that agree. A genuinely failing clock disagrees at arbitrary
hours (CT19 by 14 h, CT16 by 11 h, CT18 by 3 h). False positive in `clocks.py:493`, which
has no midnight tolerance; it costs 321 images including 7 puma. **Not fixed — a data
decision.**

**DCIM manifests are far from complete:** otoño 2025 21/21, primavera **4/26**, otoño 2026
**4/27**. The 2026-08-18 recovery only reached deployments whose DCIM structure survived the
flatten. Primavera CT23 is one of the misses, which is why its 1,735 images are unorderable.

### Found, not fixed — plataforma-territorial
`backend/routers/detections.py:381` computes `occupancy_pct` as
`stations_with_detections / len(_TC_COORDS)` — the denominator is every station in
`stations.yaml`, not the stations deployed in that campaign. Otoño 2025 ran 21 cameras and
is divided by 27. It is a compensation for the missing zero-detection stations, which the
all-stills rebuild solved. Gated on the `ct_*` rebuild (V2-REVIEW §2).

**204 tests pass** (190 at the start of the day; 14 new on the contract).


## 2026-08-20 — camera-traps

### Merged and consolidated
- `v2-review/canonical-row-set` merged into `main` fast-forward (`6da81e5`) and deleted,
  local and remote. Sole-maintainer repo; no branches outstanding. 190 tests pass on the
  merged tree, all three campaigns pass the export gate, canonical 35,807 rows / 27 stations.

### Found — V2 §2 is blocked, not unstarted
- `data-pipeline/src/parsers/timelapse_reviewed.py` re-derives the review-comment
  resolution that `camtrap.observations.resolve_review()` owns, and **disagrees on 515 live
  rows**. Its `NON_ANIMAL_COMMENTS` knows 4 strings and only demotes to `blank`; it has no
  rule producing `human`, `vehicle` or `unknown`. A `ct_*` ingest today would rebuild the
  815-row defect closed on 2026-08-19. Recorded as the fifth duplicate under V2-REVIEW 2.4,
  with the section marked BLOCKED. The fix is 2.3 — read `observations.parquet` and delete
  the parser — not teaching the parser the rules.

### Corrected
- V2-REVIEW 2.8 claimed Timelapse `mediaID`/`observationID` are "UUIDs regenerated on every
  parse". They are not; the parser reads those columns and generates nothing, so re-ingest
  replaces rather than duplicates. The real constraint is narrower and worse: the GUIDs are
  per *project*, not per image — primavera's legacy vs current `.ddb` share 2,387 filenames
  and **0** `mediaID`s — so `ct_*` keys must derive from the image (`DEDUP_KEY`), never be
  inherited from Timelapse.


## [2026-08-19] — camera-traps: the reviewer's verdict now reaches the canonical table

The primavera 2025 re-review finished, which made all three campaigns comparable for the
first time and immediately exposed the largest data defect of the V2 pass.

### Fixed

- **815 rows were typed `animal` while the reviewer had written that the frame holds no
  animal.** The review pass recorded its correction in free-text Spanish
  `observationComments` while the typed `observationType` column kept the classifier's
  guess, so every consumer counted the classifier's answer. Primavera's animal count was
  overstated by 50.6% (744 against 494) and counted 10 people and 4 vehicles as animals;
  otoño 2025 and otoño 2026 carried 124 and 465 such rows. `resolve_review()` in
  `camtrap/observations.py` now owns the resolution and is fail-closed — it refused the
  otoño 2026 ingest until a `Pitio}` typo was corrected. Animal counts: otoño 2025
  830→706, otoño 2026 1,785→1,320, primavera 744→494. Zero rows are now `animal` with an
  empty species.
- **`pv_2025_2026` was silently reverting the new review.** It is not a campaign but a
  second review pass over primavera, and while it sat in `CAMPAIGN_ORDER` it outranked
  primavera — so `read_campaigns` returned **169** primavera rows instead of 744, with 606
  overlapping keys restoring April labels over the 2026-08-19 review. Removed from
  `CAMPAIGN_ORDER` and `REPORT_CAMPAIGNS`; directory kept as provenance and now raises
  `UnorderedCampaign` on read.
- **`data/campaigns/otono_2025/timelapse_recognition_file.json` held 28 pre-flatten
  filenames** (`CT14/M 11_101EK113_*` against the flattened `CT14/101EK113_*`), which
  would have failed to join for those frames. Replaced with the post-flatten re-run.
- A **0-byte `TimelapseData.ddb`** had been committed for otoño 2025 — an empty database
  that looked like a database. Replaced with the real file.

### Changed

- **Video is excluded from every campaign's export, by policy.** Otoño 2026 had carried
  2,162 video rows swept as `blank` (2,158) / `human` (2) / `vehicle` (2) and **zero**
  `animal`, inflating its blank count and deflating every rate built on it, while
  primavera excluded its 2,618 videos at source. Denominators of a different *kind* are
  not comparable across campaigns. Otoño 2026's export is now 9,906 stills.
  `exports.require_stills_only()` refuses any future export carrying video, and is not
  overridable. Proven a no-op for the rest of the chain: `clocks.diagnose` already
  discarded video before ordering, so the clock diagnosis, segment table and all 26
  clean-clock verdicts came out byte-identical.
- `STILL_EXTENSIONS`/`is_still` moved from `camtrap/clocks.py` to `camtrap/exports.py`,
  which owns the Timelapse2 record shape per DESIGN_NOTES, rather than duplicating the
  extension set in a second module.
- New canonical column **`review_resolution`** records which rule resolved each row,
  including two values that mark decisions still open.
- Canonical tables rebuilt for all three campaigns: 830 / 744 / 1,785 rows, 3,359 total.

### Added

- **otoño 2025 and primavera now hold `ImageData_total.csv` in the repo**, so the export
  gate passes for all three campaigns from the repo for the first time — V2-REVIEW 1.3.
- 27 tests (179 total, up from 152). Note **pytest is not installed** in the
  `camera-traps` env; run `python -m unittest discover -s tests`.

### Changed — second pass, same day

- **The canonical table now describes every still, not only reviewed rows.** Seven
  station-campaigns were absent because they recorded no animal: CT23 (otoño 2025),
  CT01/CT06/CT17/CT22 (primavera — 6, 21, 7 and 18 frames each), CT02/CT12 (otoño 2026).
  A station missing from the table is indistinguishable from one never deployed: fine as
  a detection numerator, wrong as a trap-effort denominator, and the module docstring
  already claimed otherwise. `compose_ingest_frame()` pins the row set to the gated
  export; `resolve_observation()` takes the review's verdict where there is one and the
  sweep's where there is not, tagged `sweep_only`. Row counts 3,359 → **35,807**
  (8,997 / 16,904 / 9,906), station gap **0 in all three**.
  `new_labeled_data_corrected.csv` stays reviewed-only — pehuen reads it.
- **`STILL_EXTENSIONS`, `_datetime_raw` coalescing**: two decisions that were about to be
  duplicated got factored instead. The `timestamp`-vs-`DateTime` per-campaign quirk is
  now `attach_datetime_columns()`, used by both the reviewed load and the all-stills
  frame.

### Fixed — second pass

- **`timestamps.py` aborted before writing anything on a cp1252 console.**
  `print(audit_text)` precedes every write and the audit is drawn with box characters, so
  an ingest died having written nothing. Added the `sys.stdout.reconfigure` guard already
  present in `01_data_prep.py`, and the same guard to `anchor_candidates.py` and
  `propose_anchors.py`, which had the same gap.
- **README Step 2a now records HOW to exclude video** — Custom Selection, filter on
  `fileMediatype`, then export. The rule previously depended on remembering it.

### Decided — the two deferred label questions (Felipe)

- **A comment that cannot name a species resolves to `unknown`.** `ave` (9), `roedor` (9)
  and `churrete` (1) stay `unknown`: a class, an order, and a genus of several local
  species. The 3 review-note rows likewise.
- **Two exceptions adjudicated as identifiable animals**, added to `species.yaml`:
  `conejo` → *Oryctolagus cuniculus* (invasive) and `pitío` → *Colaptes pitius*. Two data
  cells were corrected rather than teaching the code a typo — `conejo?` → `Conejo` and
  `Pitio}` → `Pitío`. The `?` was rabbit-vs-hare and *Lepus europaeus* is the
  most-recorded species in these campaigns, so that row carries real doubt.
- Tags renamed `unknown_pending_*` → **`unknown_coarse_comment`** (19) and
  **`unknown_review_note`** (3), nothing being pending any longer.

### Verified

- **The annual report moved by exactly one record, with a named cause.**
  `01_data_prep.py` diffed at row level: **1 added, 0 removed** — CT04 `01130013.JPG`
  *Oryctolagus cuniculus*, the conejo adjudication. Records 641 → 642, events 261 → 262,
  species kept 11 → 12. **The all-stills rebuild moved nothing**, because no `sweep_only`
  row is ever typed `animal` — asserted in a test, not assumed.
- *Colaptes pitius* is in the catalogue but absent from the report: `taxonomic_group: ave`
  and rule 4 drops every bird. Expected, not a gap.
- **190 tests pass** (152 at the start of the day).

### Decided — the canonical file set (Felipe)

- **`TimelapseData.ddb` and `TimelapseTemplate.tdb` are part of the required set.** The
  `.ddb` is the only thing that can regenerate an export, and after the CSV-side video
  filter otoño 2026's `.ddb` is knowingly divergent from its CSV — committing it makes
  that visible instead of a surprise at the next export. All three campaigns now hold the
  full set.
- **Verified: all three `TimelapseTemplate.tdb` are functionally identical** — same
  `TemplateTable`, same `FolderDataTemplateTable`, same `VersionCompatabily 2.5.0.7 /
  CamtrapDP`, and the same `observationType` vocabulary
  `[animal, human, vehicle, blank, unknown, unclassified]` defaulting to `unclassified`.
  Otoño 2026's file differs by md5 only — SQLite page noise. This matters because the
  export gate's whole premise is what the template emits; the `empty`/`person` vs
  `blank`/`human` mismatch on 2026-08-11 cost 584 uncounted `human` rows, and that premise
  now rests on a checked fact.
- **Station counts differ between campaigns on purpose.** Otoño 2025 covers 21 stations,
  primavera 26, otoño 2026 27, because the grid was built up over time — cameras were
  installed as the programme went, so each campaign covers as many as existed at its
  retrieval. Recorded in the README above the campaign table so a future session does not
  read it as a defect and try to reconcile it.

### Still deferred

- **`otono_2026` remains out of `REPORT_CAMPAIGNS`** — reconfirmed on scope: the report
  covers oct 2024 – mar 2026 and that campaign runs to may 2026.
- **The three `addaxai-*` files** (primavera only, 6.9 MB) — new with the AddaxAI update;
  no module reads them and their role is undecided. Not required, and not to be quietly
  deleted meanwhile.
- **The legacy `CamtrapDB_*` project DBs** — `CamtrapDB_Otono_2025.ddb` (3.9 MB),
  `CamtrapDB_Primavera2025.ddb` (1.9 MB) + `.tdb`. All differ from the current
  `TimelapseData.ddb`, so they are superseded V1 project state, and otoño 2026 has no
  equivalent. They are the last thing keeping the file set from being identical across
  campaigns, but deleting them is a data decision — they may be the only record of the V1
  review — so it stays Felipe's call.
- **Figures not re-rendered.** Two causes left to attribute: video leaving the
  denominators, and the 815-row review repair.

### Findings not acted on

- **otoño 2025 did not need re-reviewing.** Its March review covers all 818 animal rows
  of the new export, verified by an identical `DateTime` on all 830 joined rows. The
  Desktop `ImageData_animals_classified.csv` is a fresh CLIP pass with no `reviewOutcome`
  that disagrees with the human review on 524/818 rows.
- **Whether otoño 2026's 2,158 `blank` videos were ever watched is not established.**
  Only 248 of 2,162 have a still within ±60 s, and 39 of those sit in a burst whose
  stills contain an animal.

---

## [2026-08-18] — Toolbox: the contact master can now be corrected, not only appended to

The boss edited the master while the merged copy sat unpromoted, so the two had diverged: the 20 contacts from the 3° Encuentro existed only in `_actualizado.xlsx`, and his file was still the pre-merge 4-column layout. Re-running the merge on it would have corrupted `N`.

### Added
- **`scripts/curate_master.py`** — two-pass curation of rows already on the master (`--review` → `curacion.xlsx` → `--apply`), the counterpart to `merge_contacts.py`'s append-only pass. Proposes rather than decides, for the same reason the merge does.
- **`namesplit.split_cargo`** — pulls a job title back out of `Organización` ("Directora Educación MIM", "SBAP- Jefa División Biodiversidad"). Lives beside `split_contact` because both turn on which words name an organisation and which name a person's place in one.
- **`MasterList.curate`** — takes an entire `Curation` plan rather than one edit at a time, so no caller has to know that deleting a row shifts every row beneath it and invalidates the plan's own indices. Every read and write completes before any deletion.
- **`MasterList.fill_missing_numbers`**, **`Person.numero`** — the owner adds rows without numbering them, and `N` is not the row number once the sheet is sorted by name.

### Fixed
- **`_next_number` re-used a dozen numbers.** It read up from the bottom row, which on a name-sorted sheet held `N=129` against a maximum of 141 — the 20 appended rows would have been numbered 130–149. Now one past the highest number in the column, which does not depend on row order.
- **The append wrote the literal string `nan` into `Notas`** for every row whose review-sheet note was empty (a pandas NaN stringified on the way through `revision.xlsx`).
- **`split_cargo` contradicted its own docstring**, returning `alta` on the ambiguous `Directora | Educación MIM` shape because an acronym survived *somewhere* in the remainder. Only a remainder that *starts* like an organisation now earns `alta`.
- **README and PROJECT_STATUS** both recorded the 13 Aug merge as 141 → 166 rows. It was 161.

### Verified
- **Diagnosis before any write:** 141 of 141 of the boss's rows matched on address or name — **zero deletions, no column misalignment from his A→Z sort.** His changes were the sort, one new row, two filled-in emails, and six `Nombre, Cargo` cells split with the cargo landing in `Organización`.
- **`split_cargo` against all 110 organisation values in the real list:** 17 detections, **zero false positives.** `SBAP- Depto Fondo e IECB` (a unit) and `librería naturaleza, editores` (a business) untouched by design.
- **The keep/drop rule chose correctly on all three duplicate pairs** with no hand-editing — including preferring the *later* row for Paulina Stowhas, whose newer entry carries the current employer.
- **142 → 139 → 159 rows.** `N` 1–162, no duplicates, gaps only at 50/54/134 (the merged-away rows). **The owner's yellow highlight and its note survived three row deletions** (`D51`→`D49`), confirmed through a save/reload cycle before anything shipped. **0 unexpected changes to any other row**, 0 duplicate addresses or names remaining, share copy byte-consistent with the canonical file, boss's original untouched at 142 × 4.

### Findings
- **Three duplicate pairs were pre-existing** — the sort merely made them adjacent. One (Enrique Rivera, `N` 40 and 134) held the *identical* address on both rows, so address matching had already seen it and nothing had acted.
- **The cargo problem is growing, not shrinking.** The boss added six more while splitting cells by hand. The form template exists to stop this at the source; it has not been built into a Google Form yet.
- ⚠️ **Still open:** `UC - Glaciares` is not an organisation's name, `Katherine` has no surname, the six `?` acronym expansions from 13 Aug are unconfirmed, and **the three leaked credentials still need rotating.**

---

## [2026-08-18] — Camera-traps: otoño 2026's capture order recovered; 103 GB of Synology re-downloads removed

Renaming the download folders locally made all three Synology sync tasks treat the originals as missing and restore them beside the flattened trees. Otoño 2026 had been carrying a restored copy since June without anyone noticing.

### Added
- **`data/campaigns/otono_2026/dcim_manifest.csv`** (5,748 rows — 5,724 `moved`, 24 `renamed`) — the capture-order evidence this campaign was believed to have lost permanently. Rebuilt from `flatten_log_20260616_100329.csv`, which survived the June flatten: **flattening consumes the tree, but not the record of the tree.** Covers CT14 (3 DCIM folders), CT20 (2), CT23 (2), CT24 (1).
- **`data/campaigns/otono_2026/flatten_log_20260616_100329.csv`** — the source, committed because it existed only inside a sync folder that is never uploaded (one-way *download*), so the working copy was its only copy.

### Fixed
- **`timestamps.load_manifest` docstring** claimed otoño 2026 "was flattened before the manifest existed and can never have one". Both halves of that were wrong, and the docstring now records why, along with the limit that remains.

### Verified
- **Ordering recovered:** CT14 **1,633 colliding counters → fully ordered**, CT20 **837 → ordered**, CT23 **89 → ordered**; CT24 upgraded from `counter` to `dcim_manifest+counter`. 3,561 frames total. Confirmed by running `clocks.establish_order` with and without the manifest.
- **The manifest three ways:** per-`(deployment, DCIM folder)` counts against the restored on-disk tree; against the NAS listing in the sync client's `event-db.sqlite`; against `ImageData_total.csv` — 0 manifest rows unjoined, and coverage **total** within every described deployment, which is what stops `establish_order` refusing them as partially described.
- **Conservation, finally provable.** The NAS holds exactly 2,632 / 1,836 / 1,088 / 192 files for CT14/20/23/24; the flattened trees hold the same. The old duplicate-skip discarded **nothing** — closing the CT_14 same-size-collision worry open since 2026-07-31 by arithmetic instead of assumption.
- **Deletion safety, established before touching anything.** All three tasks are `sync_direction: 2`, and the daemon log says so in words: `is one-way downloading, ignore event`. Nothing local has ever reached the NAS; nothing deleted locally can propagate to it.
- **Deletion gated per station.** Every restored file had to have a flattened counterpart matched by **size** — names differ wherever the flatten renamed on collision, so a name match would have wrongly condemned exactly the interesting files. 10,808 files, **0 unaccounted**; re-checked immediately before each removal, skipping rather than deleting on any mismatch. Freed **103.01 GB** (free space 188 → 291 GB).
- **152 tests still pass.**

### Findings
- ⚠️ **The NAS is not a complete backup.** Five full-size (~6 MB) CT04 frames in otoño 2025 exist only on the Windows box. They are *not* the known 0-byte corrupt files, and one-way-download sync will never upload them.
- **pv/primavera needed no recovery.** Its 13,814 remote DCIM files match `primavera_2025/dcim_manifest.csv` exactly; the 2,460 non-DCIM depth-3 files are the `TC23_M20.2` station un-nested on 2026-08-13. The merged primavera/pv folder produced no discrepancy.
- **What stays lost:** CT15 (1,331 frames) and CT08 (1,129) were flattened *before* upload. No folder evidence exists anywhere, so their counters still wrap undetectably.

### Deferred
- **The structural fix.** A one-way-download sync pointed at a folder we restructure will re-download forever. The tasks are **paused, not fixed** — either remove them and keep the trees as working copies, or repoint sync at a pristine mirror and flatten into a separate directory.

---

## [2026-08-14] — Camera-traps: attribution becomes a flatten precondition; two long-open chores closed

Worked entirely from the Linux laptop, which turned out to be more capable than the docs assumed.

### Added
- **`camtrap/stations.names_a_station(folder_name) -> bool`** — what a station folder looks like, in every spelling the project has used (`CT23`, `CT_23`, `TC23_M20.2`). Matched by **shape**, not by membership in `station_aliases.csv`, for one concrete reason: `100EK113` **is** an alias row (an unrenamed SD-card folder that became primavera_2025's camera 5), so a membership test would call every DCIM folder a station and refuse every deployment that contains one. Shape excludes it without needing to know what a DCF folder is — that stays owned by `clocks.dcim_folder_key`. Returns `bool` precisely so it cannot become a second route from a name to a camera number; `resolve()` and the alias table remain the only one.
- **`flatten_for_camtrapdp.find_nested_stations(files)`** — `{rel path: files beneath it}` for station folders nested inside a deployment. Reports only the **shallowest** station-shaped component (`TC23_M20.2/100EK113` is one offence, not two — the operator moves one folder, and naming its children would bury the instruction), and counts files from the already-collected walk, so a station folder holding no media misattributes nothing and is not reported.
- **8 fixtures** in `tests/test_flatten.py`. The one that matters reads `station_aliases.csv` and asserts every spelling is recognised **except `100EK113`** — the 2026-08-13 hand-check ("34 TC-style rows, 0 disagreements") turned into a test that re-runs whenever a row is added. **104 total, all passing** (was 96).

### Changed
- **Attribution is now the third flatten precondition**, alongside conservation and ordering. `flatten_for_camtrapdp.py` refuses a deployment containing a station-shaped subfolder, checked after discovery and **before a single file moves** — under `--dry-run` too, since a dry run exists to be trusted. **Deliberately fatal-always, with no override flag** (Felipe's call, asked explicitly): no arrangement puts one station folder legitimately inside another, so there is nothing to override and the fix is always to move the folder up and rename it. This closes the gap `TC23_M20.2`-inside-`TC22_M19.2` exposed on 2026-08-13 — 2,460 files that would have been attributed to camera 22 at camera 22's coordinates, with `moved=2460 renamed=0 lost=0` and every existing check passing.

### Verified
- End-to-end on scratch trees: the TC23 arrangement is refused (`CT22/TC23_M20.2 (2 file(s) would be attributed to CT22)`, exit 1); a clean tree with grid folders, DCIM folders and a real filename collision flattens exactly as before.
- **Informe Anual 2025 v2 DOCX rendered** — `bash Anual-reports/2025/render.sh`, 1.4 MB with figures embedded. Open since 2026-05-20; it only ever needed pandoc on Linux (3.1.3 present).
- **`Anual-reports/2025/figures/` mirrored to `figures_pre_reingest/`** — the ⚠️ precondition the README, CHANGELOG and PROJECT_STATUS all repeat before any re-ingest, now actually satisfied.

### Machine audit — what the Linux box can and cannot do
- **MegaDetector can run here.** RTX 4070 Laptop (8 GB), AddaxAI installed at `~/.AddaxAI_files/` with `models/det/MegaDetector 5a/md_v5a.0.0.pt`. Only the **Timelapse2 sweep** still requires Windows — it is a .NET app, and the `Timelapse/` folder on Synology holds `.dll`/`.exe` only.
- **The ingest chain never opens an image.** `anchor_candidates.py`, `propose_anchors.py` and `timestamps.py` read the export CSV, the DCIM manifest and the MegaDetector JSON. So **otoño 2025's ingest is blocked on Linux by exactly one missing file** — `ImageData_total.csv` (8,997 rows), which passed the gate on 2026-08-13 but exists only on the Windows box. Copying that one CSV unblocks the whole chain here. Caveat: confirming a `NEEDS_REVIEW` anchor still means looking at the photograph.
- **The campaign images are not on this machine.** `SynologyDrive/Datos/2. Camaras trampa/…/CAMPAÑAS DE RECOLECCION DE IMAGENES/{Otoño 2025, Primavera 2025}` exist but are empty (selective sync); 309 JPGs total under that tree, all legacy pre-Sept-2024 material.

### Added — later the same day, after Felipe reviewed the gates
Felipe's objection: every gate quotes the incident that produced it, which makes them read as point-fixes rather than rules. **Audited all six.** Five are derived from a stated premise and name the incident only as a regression witness (`clocks` P1/P2 + the repairability rule, `dcim_folder_key`, the export gate's unknown-value rejection, `establish_order`'s partial-manifest refusal, `stations.resolve()`). **One was not**: `names_a_station` enumerates the three spellings this project has used, so `Camara 23` or `Cam23` walks past it.

The sharper finding: **the pipeline already saw the alien frames.** Pooled into CT22, `establish_order` reports `2460 filename(s) do not match the MMDD+counter grammar` and returns `ordered=False` — verified by running it. But that routes the evidence to the **ordering** question, and per the documented P1 asymmetry a camera that cannot be ordered is not thereby condemned, so 2,460 frames kept camera 22's identity and coordinates. The missing gate was not a new observation; it was the right question asked of an observation we already had.

- **`camtrap/provenance.py`** — owns *how many capture stories does this folder tell?* The rule: **two or more filename shapes each forming a counter run** means more than one camera. A shape is the stem with digit runs collapsed (`IMAG0001` → `IMAG#`), **extension excluded** because these cameras fire three stills and a video and `01120001.JPG`/`01120004.AVI` are one story. A *run* rather than a group, because otoño 2026 CT_27's hand-renamed `01060117_fiscalizador.JPG` is one frame and one frame is not a sequence. It **enumerates nothing**, so a naming convention nobody has seen forms its own group automatically — the property `names_a_station` lacks.
- **Validated before being wired in, not after:** all four campaigns, **28,178 files/rows, 0 false positives.** The one false positive the first version produced was real and instructive — pv 2025-2026 CT14's 13 `101EK113_`-prefixed names, written by our own `resolve_dest`, formed their own run. **Folded into the rule rather than tuned away**: a shape that is a strict suffix of another at a separator boundary is that shape wearing a prefix, resolved transitively (a single pass leaves `#`, `X_#`, `Y_X_#` in two groups).
- **16 fixtures** in `tests/test_provenance.py`, including the cases that must *not* fire: stills-plus-video, the hand-renamed one-off, our rename prefixes, repeated names, and CT16's impossible months — a clock failure that belongs to `clocks.py`, and this module must stay out of it. **120 total, all passing** (was 104).
- It imports **nothing** from `clocks` and is stdlib-only. The design gate predicted it would consume `parse_filename`; it does not need to, because shapes are grammar-agnostic — a stronger position than the one designed, and the docstring was corrected to say so rather than left claiming the coupling.

### Changed — later the same day
- **The top-level station-convention check is fatal by default.** It warned unless `--check-stations` was passed, which left the *weaker* guard on the older failure — `100EK113` reaching Timelapse2 as a deployment cost 252 rows of camera 5 from the 2025 report for a year — while its sibling, a station folder nested inside another, refuses outright. Both answer "is this folder the camera we think it is?", and two severities for one question is how the cheaper one gets skipped. `--check-stations` is now accepted and ignored, so command lines and notes written before today keep working.
- **`names_a_station` deliberately stays narrow**, with a docstring recording why: it can name *which folder to move*, which the general check cannot; the general check catches names nobody has thought of, which it cannot. Do not widen the regex to chase a name the other already covers.

### Decided, same day — the horario-de-invierno shift: **no correction, ever** (implementation pending)
Deferred three times, now settled — and the first answer was reversed. I recommended *correct otoño 2026, flag the rest*, since the field record documents the offset (26 stations, `clock_action=shifted`, `clock_offset_hours=-1.0`) and so it is not a guess. **Felipe overturned it**: that optimises for agreement with **Chilean civil time**, which is not a target worth hitting, because the animals do not use it.

- **An unadjusted camera clock has a CONSTANT offset from UTC** — one number, exactly removable. An *adjusted* clock has a **piecewise** offset stepping at whatever dates the technicians happened to visit. Adjusting for DST destroys a recoverable constant and replaces it with a sequence that must then be documented forever. Otoño 2026's timestamps are internally consistent on UTC−3 for the whole deployment; applying −1 h would convert that clean constant into a piecewise civil-time record, moving *away* from the frame the analysis needs.
- **The clock has only ever been adjusted once**, at the May 2026 retrieval — Felipe's fact, and the one that resolves the older campaigns: they are not ambiguous, they are constant-offset since each clock was last set.
- 🛑 **`camera_datetime_observed` is empty on all 26 rows.** The schema already has the column that would have settled this. What was recorded is the *conclusion* (`shifted, -1.0`), not the *observation*. **Witness vs navigational evidence, for the third time in this project** — and the technician genuinely cannot tell the cases apart, because they compare the camera against a phone that auto-adjusts, so "fixed offset, civil time moved" and "drifted or reset" look identical.
- **Measured, and it reorders the priorities:** day length at −39.4417 across the otoño 2026 deployment runs 14.35 h → 9.86 h, a **4.92 h swing**; sunrise alone moves ~2.5 h. A crepuscular species tracking sunrise appears to shift two and a half hours in clock time while doing nothing different — **~2.5× the DST hour**, on every campaign and every station, with no field note able to rescue it.
- Exposure, for the record: obs after each transition — otoño 2025 **52** (+1 h), primavera 2025 **837** (**−1 h**, opposite sign), pv 2025-2026 **320** (−1 h), otoño 2026 **779** (+1 h).

**Pending, none started:** (1) **field protocol** — record both clocks as raw readings, never a correction; stop adjusting camera clocks, set once to a fixed offset and record which. No code, and the only item that expires. (2) **Schema** — store the instant plus each deployment's fixed offset, derive civil time for display and sun-anchored time for analysis; design gate. (3) **Sun-anchored sensitivity run in pehuen** — read-only, clock time vs double anchoring, to see what moves before rebuilding anything; design gate opened, then stopped. ⚠️ Blocker to confirm first: **R may not be installed on the Linux box**. ⚠️ Sun-anchoring **will move Dhat4**, and five of ten published pairs straddle the 0.75 Monterroso threshold.

### Still open
- **`primavera_2025` × `pv_2025_2026`** as two readings of one campaign — both parquets are local, so this is analysable on Linux and is a prerequisite for the primavera re-ingest.
- **pehuen's hardcoded `C:/Users/USUARIO/...` paths** (`R/01_load_data.R:50–58`), plus three more in `Anual-reports/2025/py/`.

Session log: `SecondBrain/Sessions/2026-08-14-camera-traps-nested-station-precondition-and-linux-audit.md`.

---

## [2026-08-13] — Camera-traps: the last campaign flattened, and `pv_2025_2026` turns out not to be a campaign

The primavera-verano download reviewed and flattened (19,522 files, 26 stations, **0 lost**). Two findings outrank the flatten itself.

### Fixed (on disk, before flattening)
- **🛑 A whole station was nested inside another.** `TC23_M20.2` — 2,460 files — sat inside `TC22_M19.2`. Flattening would have attributed all of them to camera 22, at camera 22's coordinates. **Nothing in the pipeline would have caught it:** the two cameras use different filename schemes (`IMAG####` vs `MMDDnnnn`), so there were zero collisions — the run prints `moved=2460 renamed=0 lost=0` and the conservation check passes. The already-ingested `pv_2025_2026` parquet proves the nesting is new: `TC23_M20.2` appears there as a *top-level* `rel_path` prefix, and camera 22 has 0 rows (never reviewed). → **Conservation and ordering are checked; station attribution is not.** A precondition for this goes through the design gate next session.
- 26 folders renamed `TC<n>_M<grid>` → canonical `CT01`–`CT26`. The mapping rule was **checked against all 34 TC-style rows** in `station_aliases.csv` (0 disagreements) rather than assumed.

### Changed
- **`pv_2025_2026` is not a campaign.** The field record has exactly three transitions — `otono_2025` → `primavera_2025` → `otono_2026` — and campaigns are named for the season they are **retrieved** in. The campaign opened 2025-05-14/06-11 and closed 2025-11-12/2026-01-14 is `primavera_2025`; the download's span matches it exactly. `pv_2025_2026` is a **second Timelapse2 review pass** over the same cards (396 shared camera+filename keys; `label_conflicts_primavera_vs_pv_2026-05-27.csv`). This sharpens the 2026-07-30 note: they are not consecutive campaigns to dedup by precedence, they are two readings of one campaign, and `CAMPAIGN_ORDER` currently encodes them as sequential.

### Added
- **`data/campaigns/primavera_2025/dcim_manifest.csv`** — 13,814 rows; 19,522 files in → 19,522 out, 1,935 renamed (all CT14), **0 lost**. CT02, CT08, CT11 and CT14 all earn `ORDER_MANIFEST`.

### Verified
- **The deployment window holds on data it was not written from.** Every station with a working clock has its frame span *inside* its field-record window, often to the day: CT04 opened 05-14 / closed 11-21 with frames 0514..1121; CT06 05-14/11-13 with 0514..1113; CT20 06-09/12-03 with 0609..1203; CT15's first frame *is* its install date. First real test of yesterday's `camtrap/anchors.py` against unseen data.
- **Otoño 2025's export passes the gate** — `full_category_sweep`, 8,997 rows (animal 818, human 478, vehicle 99, blank 7,602), matching the flatten exactly. Ingest is unblocked.
- **Unicode: nothing to do** — 39,173 paths, 0 non-ASCII, 0 NFD, 0 control characters.
- No code changed; **96 tests still pass.**

### Notes for the sweeps
- **~9 primavera_2025 stations show clock resets detectable from the field window alone** (CT03, CT05, CT08, CT14, CT17, CT23, CT24, CT26 carry January frames in a deployment opened in May/June). Filename-MMDD evidence — a preliminary signal, not a pipeline verdict.
- **CT16's clock is impossible, not merely wrong** — filenames `00300001.JPG` (month 00) and `16300071.JPG` (month 16). Corrupt RTC; no anchor repairs a clock emitting invalid dates.
- **Eight otoño 2025 images cannot be decoded** — six 0-byte in CT04, two ~4.6 MB in CT13 whose bytes are all-zero. MegaDetector skipped all eight; Timelapse2 labels them `blank`. **Felipe's call: leave them `blank`** (I recommended `unknown`); 8 rows of 8,997, no figure moves — recorded as an accepted limitation. A scanner reproducing MegaDetector's error log found the same 8, and **0 in primavera_2025**.
- ⚠️ **Both `primavera_2025` and `pv_2025_2026` are in `REPORT_CAMPAIGNS`.** Re-ingesting primavera_2025 at full size — 26 stations vs the current parquet's 14, 19,522 files vs 1,960 observations — will move the 2025 report substantially more than otoño 2025 will. Mirror `Anual-reports/2025/figures/` before either.
- **The horario-de-invierno shift is deferred a second time** — it keeps losing to campaign work.

---

## [2026-08-12b] — Camera-traps: otoño 2025 re-downloaded and flattened; ordering evidence tightened

The campaign was re-pulled from the Synology originals **with its SD-card subfolders intact** — the capture-order evidence otoño 2026 lost permanently. Flattening it exposed a defect in what the DCIM manifest is allowed to claim.

### Fixed
- **Only a camera-created folder counts as ordering evidence.** The manifest's claim is *frames in folder A precede frames in folder B, because the camera fills folders in name order*. That has two preconditions; only one was enforced (`establish_order` refuses a partially-described deployment). The other — that every group actually **is** a camera folder — was missing, so `flatten` recorded the whole intermediate path and any hand-made directory became evidence. Otoño 2025 CT04 held 723 loose frames under `M5` beside `M5/100EK113` and `M5/101EK113`; `M5` sorted **first** and asserted its January frames preceded the October ones. Sorted on, that is a backwards step in capture order — read by the diagnosis as a **clock reset on 2,097 frames**.
- `clocks.dcim_folder_key()` keeps only a DCF-shaped last component and runs at **both** ends — `flatten` on write, `timestamps.load_manifest` on read — so a manifest already written is corrected **by being read**. Necessary here: flattening consumes the tree it describes, so otoño 2025's manifest cannot be regenerated. The rule holds whether or not a camera can leave files beside its own DCIM folders, so it rests on no assumption about firmware.
- **Rename prefixes use the DCIM folder alone**, not the whole path. Otoño 2025 would otherwise have produced `M 11_101EK113_01160002.JPG` — a space imported into 28 filenames to disambiguate nothing. Unsafe characters are stripped (`M17 (TC20)`, `M18 (vacía, TC mala)` were real folder names).
- `--check-export` help still said `person`/`vehicle`; it is `human` now.

### Changed
- Eighteen otoño 2025 stations with a single constant folder (`M7`) previously reported `ORDER_MANIFEST` while the manifest contributed nothing — sorting by a constant is a counter sort. They now correctly report `ORDER_COUNTER`. CT14 and CT20 keep full `ORDER_MANIFEST`; CT04 is refused, fail-closed. Otoño 2026 unchanged.

### Added
- **`data/campaigns/otono_2025/dcim_manifest.csv`** — 8,997 files flattened, 8,969 moved, 28 renamed, **0 lost**.
- `tests/test_flatten.py` (9) and `TestDcimFolderKey` in `test_clocks.py` (6). **96 total, all passing.**

### Notes for the otoño 2025 sweep
- The download holds **no video at all** (8,997 files, every one `.JPG`); counter gaps run 19–25% at most stations and 0% at three. That is the 3-stills-plus-1-video trigger pattern with videos absent from this download, **not** lost stills. Confirm the videos exist on the NAS.
- CT04's 723 loose frames carry their own counter run restarting at 1, an MMDD range disjoint from and later than both DCIM folders, and no filename in common with either — a third DCIM folder flattened by hand at some earlier point.
- The existing export is animal-only (830 rows), so the campaign stays rejected by the gate until swept in full.
- **otoño 2025 is in `REPORT_CAMPAIGNS`** — re-ingest will move the 2025 annual report's numbers. Mirror `Anual-reports/2025/figures/` first.

---

## [2026-08-12] — Camera-traps: the field record becomes a pipeline input; otoño 2026 ingested

`data/campaigns/field_notes.csv` stops being a reference document and starts driving the clock diagnosis. The change that matters is the **deployment window**: it used to be derived from the anchors, and anchors exist only where a clock already broke — so 26 of otoño 2026's 27 stations had **no window at all**, and a forward clock jump (a clock set *ahead*, which keeps every capture delta positive and is invisible to backwards-step detection) could not have been detected for any of them.

Also lands two days of uncommitted work: the Camtrap DP vocabulary correction to the export gate, and the one-time migration of the monitoring workbook.

### Added
- **`camtrap/anchors.py`** — owns what the field record asserts about a camera's clock, deliberately split into two assertions with different preconditions: the **deployment window** (every station, needs no photograph) and an **anchor** (only where the clock failed, needs a datable frame). Absorbs the anchor-CSV schema and `load_anchors` from `timestamps.py`, so the file format is stated in one place.
- **`propose_anchors.py`** — joins the visit record to `anchor_candidates.csv` and writes `anchor_proposals.csv` with a `READY` / `NEEDS_REVIEW` / `NOT_NEEDED` status per segment. Promotes nothing automatically.
- **`setup/build_field_notes.py` + `data/campaigns/field_notes.csv`** — 106 visits, 27 stations, one row per physical visit (a revision closes one campaign and opens the next). The workbook held three date conventions at once — Chilean `d/m/y` as text, `m/d/y` off camera screens, and cells Excel had already parsed with the machine locale; the last are the dangerous ones because a wrong reading looks clean. Swaps are detected against each sheet's plausible window and recorded in `data_flags`; a value plausible both ways is flagged, never picked. 57 of 106 rows carry a flag. `clock_state` defaults to `unknown`, never `ok`.
- **`visit_date_only`** anchor type (APPROXIMATE). All 27 otoño 2026 opening visits record a date and no time, so an exact anchor would assert an hour nobody wrote down — which is how CT18's install anchor came to claim `14:00:00` against a notebook that says only `2025-11-14`.
- **`tests/test_anchors.py`** — 19 fixtures. **81 total, all passing** (was 59).

### Changed
- **`anchor_candidates.py`** now ranks the swept export **above** MegaDetector: `human_labelled` / `vehicle_labelled` (someone looked and said so) outrank `person_detection` / `vehicle_detection` (an unconfirmed guess). On otoño 2026 that is 584 + 25 confirmed frames and **zero** unconfirmed — the sweep corroborated every MegaDetector person hit.
- **A station with no deployment window is now reported as `unverified` clean, not clean.** CT27 is the live case: no install record, so the in-window test never ran. Its passing verdict is an absence of evidence, and both `timestamps.py` and `clocks.diagnose` now say so instead of claiming "every frame is in-window".
- **`camtrap/exports.py`** — Camtrap DP's vocabulary named constant-by-constant (`TYPE_HUMAN`, `TYPE_BLANK`, …) so consumers stop restating it; likewise MegaDetector's in `camtrap/detections.py` (`CATEGORY_PERSON`, …). The two vocabularies are deliberately spelled differently because they are different vocabularies.

### Fixed
- **The export gate used our own invented vocabulary** (`person`/`empty`) instead of Camtrap DP's (`human`/`blank`), which the Timelapse2 template emits verbatim. Otoño 2026's first properly swept export passed **only because `vehicle` is spelled the same in both** — its 584 `human` rows counted as neither assigned nor proof, so the same campaign with no vehicle frames would have been rejected as unswept *after* the sweep was done. An unrecognised `observationType` is now a hard rejection (`unrecognised_category_values`) rather than a note: a value the gate cannot interpret vanishes from the tally, which is exactly the failure above.
- **A visit is not an anchor.** CT01's notebook says the deployment ran 2025-11-24 → 2026-05-13 while its frames run 2025-11-26 → 2026-05-14 across one coherent segment with no reset. Turning that visit into an anchor would apply a two-day offset to a clock that was never wrong. Anchors are now proposed only where the segment would otherwise be refused.
- **Witness vs navigational evidence.** The first version of the proposer paired CT18 segment 0 with `11190001.JPG` — a counter-`0001` frame — for a **−5 day offset applied to ten frames whose clock was correct**. A counter-`0001` frame is the first file on a card, not a photograph of the technician; the camera simply did not trigger for five days after install. Only a frame that *witnesses* a visit can date one.

### Verified
- The 3-day visit-window tolerance is **measured, not guessed**: across the 20 otoño 2026 stations provably coherent from capture order alone (so any gap to the notebook is the notebook's imprecision), the largest excursion past a recorded visit date is **+1.67 d**. The constraint is one-sided at each edge — a frame before the install or after the retrieval is impossible, while a quiet stretch inside the window is evidence of nothing (CT06 and CT11 went 35 and 41 days from install to first trigger; CT19 stopped firing 91 days before retrieval).
- Applying the window changes **zero** verdicts while giving 26 stations a check they never had.
- **otoño 2026 ingested end-to-end**: 1,785 rows, 26 clean stations, CT18 refused on all five segments (10/32/40/3/227, `valid_effort=FALSE`) — reproducing the 2026-08-03 hand analysis mechanically.

### Open
- **The Mayo 2026 horario-de-invierno shift.** Every camera was set back 1 h at that visit, and Chile left summer time 2026-04-04, so otoño 2026 frames between those dates read **1 h ahead of local time**. A ~40-day systematic time-of-day error that no reset detector can see — an hour never breaks segment coherence. Awaiting Felipe's decision on whether to correct it.
- The other three campaigns still need Timelapse2 sweeps; nothing else can move them.

Session log: `SecondBrain/Sessions/2026-08-12-camera-traps-field-record-as-pipeline-input.md`.

---

## [2026-07-09] — Pehuen research: Monterroso overlap categorisation on Dhat4 CI

Reframed `Research/pehuen-species-interactions/R/04_temporal_overlap.R` around the Monterroso et al. (2014) classification — the ecological standard for interpreting Dhat4. Explored Watson's U² and a Dhat4 randomisation test first; both were dropped after user pushback identified a conceptual mismatch: those tests answer "**are the two activity curves different?**", not the actual biological question **"is the observed 0.79 overlap meaningful?"**. Dhat4 has no natural null value (two nocturnal species can share Dhat4 ≈ 0.8 just from both being nocturnal), so a "significance of Dhat4" test isn't well-posed. The Monterroso threshold interpretation — applied to the *whole CI*, not the point estimate — sidesteps that trap by classifying the pair rather than testing it.

### Changed
- **`R/04_temporal_overlap.R`** — single-loop stats computation (Dhat4 + 1000-boot 95% CI + Monterroso category). Category assigned from the CI: clean `Low/Moderate/High` when the whole CI sits in one band; compound labels (`Low–Moderate`, `Moderate–High`, `Low–High`) when the CI straddles a threshold, to avoid overstating confidence. Per-pair PNGs now carry an annotation strip `Overlap: <category>   (Dhat4=x, 95% CI [l, h])`. Summary figure has two vertical cutoffs (0.50, 0.75), subtle background band shading, and category appended to each pair label. Override camtrapR's default title so it uses species names, not argument identifiers (`sp1`/`sp2`).
- **`data/overlap_stats.csv`** — new output. `sp1, sp2, guild_type, n1, n2, dhat4, ci_low, ci_high, category, small_sample`. Suitable for direct inclusion in the annual report.
- **`README.md`** (pehuen) — Monterroso classification documented in the "Running the analysis" section.

### Removed
- **10 stale `2026-06-25` PNGs** in `figures/overlap_pairs/` — regenerated with today's annotations, so the old copies were duplicates without categories.

### Results at a glance (10 pairs)
- **1 pair clean High**: Guiña × Zorro culpeo — CI [0.77, 0.85] entirely above 0.75. Statistically defensible statement that these two native carnivores share a diel niche.
- **5 pairs Moderate–High** (CI straddles 0.75): Guina × Liebre, Zorro × Liebre, Puma × Guiña, Puma × Liebre, Puma × Jabalí.
- **4 pairs clean Moderate**: Guina × Perro, Zorro × Perro, Zorro × Jabalí, Puma × Zorro culpeo.
- **No pair in the Low band** — nothing at Bosque Pehuén shows unambiguous temporal avoidance under the Monterroso cutoffs.

Session log: `SecondBrain/Sessions/2026-07-09-pehuen-overlap-monterroso-categorisation.md`.

---

## [2026-06-17] — Camera-traps: Otoño 2026 campaign integrated, +Quique

May 2026 SD pull (campaign name **Otoño 2026**, slug `otono_2026`) reviewed end-to-end and staged for ingestion. CSV registered in `data-pipeline/config.yaml`. New species — Quique (*Galictis cuja*) — added to the canonical catalog with a CLIP English prompt; first project record (5 obs in this campaign). Yesterday's Vaca addition validated: 579 rows tagged Vaca in this campaign, all of which would have been mislabeled Caballo. Ingestion itself is held until CT_18's clock-reset timestamps are corrected.

### Added
- **`data-pipeline/species.yaml`** — Quique (*Galictis cuja*). CLIP prompt: `"lesser grison small mustelid weasel"`. Native, no `is_invasive` / `is_priority` flag. 29 CLIP species + 4 reviewer-discovered non-CLIP entries = 33 total (was 28+4=32).
- **`camera-traps/data/campaigns/otono_2026/new_labeled_data_reviewed.csv`** — 1785 rows, 25 deployments (CT_02 and CT_12 produced no animal triggers; timelapse parser is observation-centric so they're correctly absent from `ct_deployments`). Date range covers 2025-? through 2026-05-15, except CT_18 which has 135 rows stuck at 2017-01-01 (see below).
- **`data-pipeline/config.yaml`** — 4th `camera_traps.campaigns` entry: `name: "Otoño 2026"`. Comment block immediately above the entry flags the CT_18 timestamp issue and instructs not to run `--ct` until corrected.

### Changed
- **`camera-traps/README.md`** — Status header rewritten for 2026-06-17; species table +Quique row + filled-in Invasive/Priority cells; CLIP species count 26 → 29 in Step 3; Campaign History table now includes Otoño 2026 and corrected paths for the prior three campaigns.
- **`PROJECT_STATUS.md`** — top "Last updated" line rewritten; section 1 species.yaml count 31 → 33; section 3 Last Updated/What Changed/Integration/Blockers refreshed; component table +Otoño 2026 row; Open Items: new CT_18 timestamp-fix entry.

### Notes
- **CT_18 clock reset**: 135 rows on CT_18 carry `DateTime` 2017-01-01 (camera clock reverted to factory default at some point during the deployment). Felipe has the real deployment-start date in his field notebook; until it's transcribed, `python run_fetch.py --ct` is held on the Linux box. Once the anchor is in hand, one re-stamp + re-ingest finishes the integration. Until then the `otono_2026` entry sits dormant in config.yaml behind a comment.
- **Zero cross-campaign overlap** verified via stdlib CSV-vs-CSV check against otono_2025 / primavera_2025 / pv_2025_2026. No dedup script needed (unlike the primavera_2025 case, this is a fresh pull, not a partial re-pull).
- **CLIP horse/cow confusion** that motivated yesterday's Vaca prompt is now quantifiable: 579 Vaca rows (#1 species in this campaign) vs 70 Caballo rows — strong evidence the prompt distinguishes correctly. Revisit `clip_confidence_threshold` (0.28) only if the false-positive rate on tightly-similar pairs (Vaca↔Caballo, side/rear shots) looks bad after ingestion.

Session log: `SecondBrain/Sessions/2026-06-17-camera-traps-otono-2026-ingest-prep-and-quique.md`.

---

## [2026-06-16] — Camera-traps: review UI burst context, full-frame display, resume loader, +Vaca

Reworked Phase-1 review UI to better support species disambiguation. Reviewers now see burst context (prev/current/next thumbnails sourced from the MD JSON, including empty triggers) and full frames instead of bbox crops. CLIP classifier untouched — it still receives the bbox crop, keeping its subject-isolation accuracy. Added a startup loader that rehydrates review progress from the previously-exported CSV, eliminating the in-memory-only footgun that bit during this session.

### Added
- **`phase1_labeling/app.py`** — `load_station_index()` builds `{station: [files sorted alphabetically]}` from the MD JSON (includes empty triggers, so reviewer sees what happened before/after the animal trigger). `neighbors(fp)` returns prev/next within station, `None` at deployment boundaries. **Resume loader**: on a fresh session with empty `confirmed`, reads `new_labeled_data_reviewed.csv` and rehydrates `st.session_state.confirmed` / `outcomes`, auto-jumping to the first species batch with unconfirmed images.
- **`data-pipeline/species.yaml`** — Vaca (*Bos taurus*). Aliases: `vaca, vacuno, ganado vacuno, bovino`. CLIP prompt: `"domestic cow cattle bovine"`. `is_invasive: true`. 28 CLIP species now (was 27). Cows were being misclassified as Caballo on BP Mayo 2026.

### Changed
- **`classify_campaign/crop_utils.py` → `cropping.py`** — the file is cohesive (only cropping code, no junk drawer), so the `_utils` suffix was lazy. Imports updated in `run_classification.py`; the now-unused `crop_to_bbox` import was dropped from `phase1_labeling/app.py`. General principle to apply elsewhere when `_utils` suffixes are spotted on cohesive modules.
- **Review UI grid** — was 5-col bbox-cropped thumbnails, now 3-col triptych grid `[anterior | actual | siguiente]` with full-frame thumbnails. Burst context (typically 2-3 frames per trigger) is now the primary species-disambiguation cue alongside the proposed species label.
- **`THUMB_SIZE`** 280 → 1280, **`JPEG_QUALITY`** 75 → 85 in `phase1_labeling/app.py`. Streamlit's expand-icon lightbox shows the cached JPEG as-is, so the cached resolution determines expand quality. In-grid display is unchanged (`use_container_width=True` clamps the visual width). Memory tradeoff acceptable for local desktop review (~10-18MB cache per page).
- **`camera-traps/README.md`** — project tree (`cropping.py`), Step 4 review-UI section rewritten to describe the burst triptych + full-frame display, species table now includes Vaca and the missing Invasive cells for Caballo / Gato doméstico.

### Fixed
- **Sibling import error when launching Streamlit** — `streamlit run phase1_labeling/app.py` was failing with `ModuleNotFoundError: No module named 'classify_campaign'` because Streamlit puts the script's directory on `sys.path`, not the project root. Added a 2-line `sys.path.insert(...)` at the top of `app.py`. This was a latent issue that newer Streamlit versions surfaced (older versions added CWD to sys.path more aggressively).

### Notes
- **Habit to keep**: hit "Exportar CSV revisado" periodically during long sessions — it's now the durable checkpoint and the resume loader will pick it up on next launch.
- **CLIP horse/cow confusion** may persist on side/rear shots even with the new Vaca prompt. If false-positive rate is high after re-classification, consider tightening `clip_confidence_threshold` (currently 0.28) — but only after seeing the data.

Session log: `SecondBrain/Sessions/2026-06-16-camera-traps-review-ui-context-strip-and-resume.md`.

---

## [2026-06-02] — Observatorio: piso vegetacional layer (plataforma-territorial)

New toggleable Leaflet layer on the Observatorio page: photointerpretation of Bosque Pehuén's vegetational floor (48 polygons). Off by default; click a polygon for `BIOTOPO / Distrito / Superficie (ha)`.

### Added
- **`plataforma-demo/src/components/PisoVegetacionalLayer.jsx`** — self-contained React layer: owns its own fetch, color map, and popup. `Observatorio.jsx` only adds 1 import, 1 `useState(false)`, 1 conditional render, and 1 checkbox. Future GIS layers should follow this shape.
- **`scripts/convert_piso_vegetacional.py`** — pure-Python shapefile → GeoJSON converter (`pyshp` + `pyproj`, no GDAL). Handles ESRI ring orientation (CW = outer), reprojection EPSG:32718 → EPSG:4326, and the actual UTF-8 dbf encoding. Writes the same payload to `data/`, `public/data/`, and `dist/data/`.
- **`data/piso_vegetacional.geojson`** (+ public/dist copies) — 48 features. Source shapefile preserved under `data/piso_vegetacional_source/veg_foto_BP.*` (outside the served frontend) so the script remains reproducible.

### Changed
- **Palette designed in two iterations.** First attempt (greens only, dark → light by density) collapsed visually to ~6 distinguishable swatches across 10 classes. Final palette groups by ecological type: greens for Bosque (4 classes), ochres for Renoval (3), blues/violets for Matorral / Pradera / Estepa.
- **No polygon borders** — `stroke: false`. White borders fought the Esri satellite imagery underneath.

### Fixed
- **Mojibake (`MesÃ³fito`) in popups.** First conversion used `encoding="latin-1"` based on a misleading test; the .dbf is genuinely UTF-8 (matching the .cpg). Codepoint inspection on Windows is authoritative — the cmd console can't render `ó` and shows `�`, which is not the same as a decode failure. Switched to `encoding="utf-8"`; also resolved a silent color-fallthrough where the mismatched key was hitting `FALLBACK_COLOR` for 4 polygons.

### Documentation
- README top block, "Four Pages" table, and "Real Data" list updated.
- `PROJECT_STATUS.md` Plataforma section: Observatorio row + Priority 3 list.

### Notes
- **`DISTRITO` field documented**: physiographic / terrain-relief classification (`Plano / Ondulado / Cerrano / Montano`) — geomorphological, independent of vegetation. Currently a popup property; could become a separate layer.
- **`ESPECIES_D` field omitted** from popups for now — Felipe doesn't have a key for the species codes (NF / ND / NA / NP / AA / SC). Re-enable when a legend surfaces.

Session log: `~/Documents/Obsidian FG/SecondBrain/Sessions/2026-06-02-plataforma-piso-vegetacional-layer.md`.

---

## [2026-05-11] — 🏁 First full code review complete

The independent code review of the FMA ecosystem (started 2026-04-21) is finished. **Every finding across data-pipeline, camera-traps, and plataforma-territorial is now closed or explicitly deferred with a re-open trigger.** No silently-open work remains.

**Headline numbers:**
- 1 Critical (C1 — CR800 state-before-commit) — resolved
- ~25 Warnings (W8…W52) — all resolved
- ~50 Suggestions (S8…S78) — resolved, closed-rejected, or deferred with re-open conditions

### Fixed (today's session — Tier 1 through Tier 4)
- **Tier 1 reliability** — S50 (FastAPI health 503), S37 (mtime-keyed Streamlit cache), S10 (`managed_conn` context manager), S11 (per-event DB connections in watcher).
- **Tier 2 high-value** — S44 (`backend/paths.py`), S48 (stations.yaml coords), S55 (DuckDB CTE for `days_without_rain`), S64 (`demo_report.js`), S66 (RiskGauge pure component), S39 (`classify_all` / `apply_classifications` split).
- **Bundle A — schema authority pass** — S49 (startup drift-check; surfaced 27 real extras in the DB including `battery_voltage` accidentally absent from `ALLOWED_COLS`).
- **Bundle B — API/hook ergonomics** — S53 (docstrings + `common_name` symmetry), S59 (`useAPI` `refetch`).
- **Bundle C — camera-traps `setup/` cleanup** — S29 (`crop_to_bbox` reuse), S30 + S32 (argparse), S36 (env + CLI for `CAMPAIGNS_BASE`), S43 (extracted `setup/_fileops.py`). Net **−80 lines** despite one new file.
- **Tier 4 finalization** — S35 (`AnimalRow` dataclass), S38 (UI strings standardized to Spanish).

### Closed-rejected (with documented rationale)
- **S57** — DuckDB pushdown for `strftime` (resample is pandas-side; pushdown would add round-trips).
- **S47** — bootstrap_windows_db.py inline SCHEMA (intentional for cross-machine portability; revisit at Windows→Linux migration).
- **S14** — DuckDB secondary indices (columnar engine + composite PKs already cover hot reads at current data scale).
- **S72** — react-router-dom (overkill for 4-page internal tool; migration path documented).

### Deferred (with re-open triggers)
- **S58** — `stations.yaml` TC-11 / TC-18 both list `sd_card: M15`. Re-open when field records produce an authoritative answer.
- **S76** — Vitest tests in `plataforma-demo/`. Re-open when CI exists.

### Process
Six batches in a single day. Full narrative in `~/Documents/Obsidian FG/SecondBrain/Sessions/2026-05-11-fma-ecosystem-code-review-tier-1-and-tier-2.md`. Review state snapshots in `~/Documents/Obsidian FG/SecondBrain/Reviews/review-state-{data-pipeline,camera-traps,plataforma-territorial}.md`.

---

## [Prior to changelog adoption]

Pre-2026-05-11 history is preserved in:
- `~/Documents/Obsidian FG/SecondBrain/Sessions/` — per-session narrative logs
- `~/Documents/Obsidian FG/SecondBrain/Reviews/review-plan-fma-ecosystem.md` — master review plan with Track A–K log
- Per-project README "Last Updated" sections and `PROJECT_STATUS.md`

This `CHANGELOG.md` is the new top-level history starting 2026-05-11. Going forward, every significant change lands a one-line entry.
