# V2 REVIEW — every campaign clean, every consumer clean, before anything new starts

**Written:** 2026-08-18 · **Status:** in progress — entry condition met 2026-08-19
**Audience:** a fresh session with no memory of the 2026-08-18 audit.

---

## 0. Why this exists, and the rule it encodes

Everything in the camera-trap chain must be clean — canonical files, scripts, outputs,
**every campaign included** — before work begins on anything new. The reason is not
tidiness. Problems in this chain do not stay in it: they bleed into the annual report,
into pehuen, into the platform, and they arrive there disguised as findings. This
project has already spent four months on a coordinate error that reached the platform
and came back as a 19 km displacement, and a full session on a phantom clock reset that
was really a hand-made folder sorting first.

**Entry condition — do not start any step below until all three hold:**

1. ~~`primavera_2025`'s re-review is finished in Timelapse2~~ **MET 2026-08-19.**
2. ~~Its `ImageData_total.csv` is exported and passes `exports.read_total_export`~~
   **MET 2026-08-19** — `full_category_sweep`, `n_rows=16904`.
3. `pv_2025_2026` is confirmed retired-to-provenance, not deleted.

> **`n_rows` was wrong in this document until 2026-08-19: it said 19522.** That number
> is the *flatten's* file count, and 19,522 = 16,904 stills + 1,663 mp4 + 955 mov.
> Timelapse2 exports stills only and the export matches the still count exactly, so
> nothing was ever missing. Video is now excluded from every campaign's export by
> policy — see README Step 2a. Do not re-raise 19,522 as an export target.

**Out of scope, deliberately.** Both are real and both are larger than this review;
folding them in would stall it. Neither blocks it.
- The DST decision's piece 2 — storing the instant plus a fixed per-deployment offset
  in `observations.parquet` (`datetime` is naive local today).
- The sun-anchored sensitivity run in pehuen (piece 3).

---

## 0-bis. Re-audit, 2026-08-20 — read this before trusting any checkbox below

Checked against the working tree, not against this document's own `[ ]` marks, while writing
`DATA-HEALTH-MANUAL.md`. **Fifteen items open. Four are guarantees the manual states and the
code does not enforce**, and one of those four had no numbered item here at all.

| | item | this doc | status |
|---|---|---|---|
| **A1** | station registry disagrees: `stations.yaml` **26** · geojson **27** · `estaciones.csv` **27**; CT27's files ingest coordinateless | 1.6 | ~~open~~ **CLOSED 2026-08-24** |
| **A2** | contract published, consumer-side freshness check absent (`grep -r CANONICAL_STATE data-pipeline/` → nothing) | §4 | ~~open~~ **CLOSED 2026-08-24** |
| **A3** | **the field form has no loader** — `build_visit_template.py` writes the workbook, nothing reads it back | **1.14, new** | open |
| **A4** | `occupancy_pct` divides by all 27 stations | 1.6 / §2.5 | ~~open~~ **CLOSED 2026-08-24** (per-campaign filtering still open) |
| **B1** | the `ct_*` rebuild | 2.1–2.3, 2.5, 2.8 | ~~open~~ **CLOSED 2026-08-24** |
| **B2** | pehuén's absolute Windows paths | 1.11 | ~~open~~ **CLOSED 2026-08-24** — and it was more than two lines, see below |
| **B3** | figures not re-rendered; otoño 2026 falls out of `05_spatial_distribution.R:249` and the `02_detection_summary.R` labellers | 1.11 + §3 | open — **narrowed**, see below |
| **B4** | `field_notes.csv` audited for coordinates only, 57/106 rows flagged | 1.7 | open |
| **B5** | `provenance.py` not re-run on the re-ingested primavera | 1.8 | open |
| **B6** | manifest coverage not stated per campaign | 1.4 | open |
| **B7** | CT27 install datable from `CT 27.kml` (2025-12-11 15:52:56), unrecorded | 1.5 | ~~open~~ **CLOSED 2026-08-24** |
| **B8** | three regression fixtures | 1.10 | **1 of 3 done** — the registry-agreement check exists; manifest rebuild and deletion accounting still missing |
| **C1–C5** | two superseded data files on disk · stale pv comment `apply_verdicts.py:143` · otoño 2025 video existence unconfirmed · `count` empty · seasonal puma orphan | §3 | open |

### 0-ter. Second re-audit stamp, 2026-08-24 — the consumer boundary is closed

Eight of the fifteen items above are closed, and they are **all of the ones on the
data-pipeline / platform side**. The pattern §0-bis identified held exactly: every defect
sat at a boundary, and one session spent entirely at that boundary cleared it. What
remains is the FIELD-RECORD boundary (A3, B4, B6), pehuén's Windows-side work (B2, B3),
and cleanup (C).

**Three of this document's own specifications were wrong and are corrected below**, each
measured rather than argued:

| where | this doc said | measured 2026-08-24 |
|---|---|---|
| **2.8** | key on `DEDUP_KEY = [camera_num, file_name, datetime]` | `datetime` is **null in 4,013 of 35,807 rows**, so it cannot be a primary key. `(campaign, camera_num, file_name)` is unique across all 35,807 and never null |
| **2.3** | the `ct_deployments` column list | omits **`campaign`**, which the platform queries in three places; it existed only because `ensure_columns()` had added it dynamically |
| **2.3** | "Timestamps are UTC" | they are **naive local**, deliberately — see 2.3 below |

**A fourth correction, to §0-bis itself:** it recorded "CT27's **344** otoño 2026 images".
The canonical table holds **315**.

**What no longer needs doing:** the DCIM-manifest and station-registry work that A1 was
feared to cascade into. All three registries already agreed on every value they *shared*;
the defect was one missing row and the absence of any check to keep it from recurring.

**The pattern, and it is the useful part:** every open item sits at one of the two
**boundaries** — the field record coming in, or a consumer going out. The chain from card to
canonical table is enforced and tested. Defects survive where responsibility changes hands,
because that is where nobody owns the check.

- [x] **1.14a The deployment windows are published.** — **DONE 2026-08-24.** The half of
      1.14 that blocks every effort calculation is closed without waiting for the Excel
      round-trip: `field_notes.csv` already dated both ends of almost every deployment, and
      nothing read them. New `camtrap/deployments.py` pairs the visit that put a card in the
      ground with the visit that pulled it out and publishes
      `data/campaigns/<campaign>/deployments.csv` — **26 / 26 / 27 deployments, 12,975
      camera-days across the stations that have images.** Two traps are guarded by fixtures:
      `FieldRecord.window()` pads by ±3 d for anchor validation and would add six days to
      every camera, and subtracting visit *datetimes* truncates whenever one end carries a
      recorded time and the other is stamped at `ASSUMED_VISIT_HOUR` (CT01 read 168 days
      instead of 169). **The contract now covers effort:** `CANONICAL_STATE.json` is
      `schema_version: 3`, carrying `n_deployments`, `camera_days` and a SHA-256 of each
      `deployments.csv`. A wrong row count is visible because a species appears or does not;
      a wrong denominator silently rescales every rate in a report, so it belongs inside the
      thing consumers verify. **Prerequisite fixed in the same pass:** CT27 had no window at
      all — it never appeared on an install sheet and was omitted from *Registro de revisión
      Mayo 2026* (26 rows). Its opening was corrected from 2025-11-12 to **2025-12-11** (the
      day/month transposition you resolved on 2026-08-24, corroborated by its own first frame
      at 12:49:01) and its closing reconstructed as **2026-05-14** from retrieval-trip order,
      flagged `(reconstructed)` rather than attributed to a sheet it is not on. Result: **74
      of 74 deployments with images now have a field window**, asserted by a test. 235 tests
      pass, up from 226. Still open below: the round-trip that keeps this true for *future*
      fieldwork.

- [ ] **1.14 The field workbook has a loader.** `setup/build_visit_template.py` renders
      `Registro de visitas CT.xlsx` and **nothing reads it back**;
      `setup/build_field_notes.py` is the one-time legacy migration from the old workbook, not
      this. So a filled form is a spreadsheet somebody must transcribe by hand, and every
      guarantee in Part 2 of the manual rests on that undocumented step.
      `camtrap.visit_schema.by_label()` is the entry point, and the 2026-08-17 session already
      named this as "not yet built" — it never became an item, which is why it went unnoticed
      for three sessions. Pass: a filled template round-trips to `field_notes.csv` rows,
      with a fixture.

---

## 1. Camera-traps — the review

Each item is a check with a stated pass condition, not a task to eyeball.

- [x] **1.1 The campaign set is exactly three** — **DONE 2026-08-19.** `pv_2025_2026`
      removed from `CAMPAIGN_ORDER` (`camtrap/observations.py`) and `REPORT_CAMPAIGNS`
      (`Anual-reports/2025/py/01_data_prep.py`); directory and parquet kept as
      provenance, and `read_campaigns('pv_2025_2026')` now raises `UnorderedCampaign`.
      The stated order was followed: primavera's new parquet was written first.
      **This was more urgent than the note implied.** While pv sat in `CAMPAIGN_ORDER` it
      OUTRANKED primavera, so the moment primavera was re-ingested its fresh review was
      being silently reverted: `read_campaigns` returned **169** primavera rows instead
      of 744, and 606 keys overlapped, restoring April labels over the 2026-08-19 review
      (CT20 09240308 went from `Pteroptochos tectus` back to `Lepus europaeus`).
      Anyone re-ingesting primavera without doing this in the same session gets a
      quietly wrong table.
      ⚠️ `REPORT_CAMPAIGNS` is now `("otono_2025", "primavera_2025")` — **`otono_2026` is
      still absent from the 2025 report.** That is a scope decision, left open
      deliberately rather than patched in.

- [x] **1.2 The canonical file set is decided** — **2026-08-19** (Felipe).
      **Required per campaign**, and all three now hold every one:
      `ImageData_total.csv`, `ImageData_animals.csv`, `ImageData_animals_classified.csv`,
      `timelapse_recognition_file.json`, `new_labeled_data_reviewed.csv`,
      `new_labeled_data_corrected.csv`, `dcim_manifest.csv`, `deployment_anchors.csv`,
      `observations.parquet`, `timestamps_audit.log`, **`TimelapseData.ddb`** and
      **`TimelapseTemplate.tdb`**.
      The two Timelapse DBs are in by decision, not by accident: the `.ddb` is the only
      thing that can regenerate an export, and after the 2026-08-19 CSV-side video filter
      otoño 2026's `.ddb` is knowingly divergent from its CSV — committing it is what
      makes that divergence visible rather than a surprise at the next export.
      **Nothing in `camtrap/` reads either**, nor `ImageData_animals*.csv`: the classifier
      writes those and the ingest takes `new_labeled_data_reviewed.csv`. They are
      provenance, and that is the point.
      **Verified 2026-08-19: all three `TimelapseTemplate.tdb` are functionally
      identical** — same `TemplateTable`, same `FolderDataTemplateTable`, same
      `VersionCompatabily 2.5.0.7 / CamtrapDP`, and the same `observationType` vocabulary
      `[animal, human, vehicle, blank, unknown, unclassified]` defaulting to
      `unclassified`. Otoño 2026's file differs by md5 only, which is SQLite page-level
      noise. **This matters because the export gate's entire premise is what the template
      emits** — the `empty`/`person` vs `blank`/`human` mismatch on 2026-08-11 cost 584
      uncounted `human` rows — and it now rests on a checked fact rather than an
      assumption.
      **Two items still open, deliberately:**
      - **The `addaxai-*` files** (`addaxai-detections.csv` 2.7 MB, `addaxai-files.csv`
        4.2 MB, `addaxai-run-info.txt`), primavera only. New with the AddaxAI update;
        Felipe has not decided what role they play. No module reads them. Not required
        until that is settled, and **not** to be quietly deleted meanwhile.
      - **The legacy `CamtrapDB_*` project DBs** — `CamtrapDB_Otono_2025.ddb` (3.9 MB),
        `CamtrapDB_Primavera2025.ddb` (1.9 MB) + `.tdb`. All three DIFFER from the current
        `TimelapseData.ddb`/`TimelapseTemplate.tdb`, so they are the superseded V1 project
        state, and otoño 2026 has no equivalent. They are the last thing keeping the file
        set from being identical across campaigns. Deleting them is a data decision, not a
        cleanup — they may be the only record of the V1 review — so it is Felipe's call.

- [x] **1.3 The export gate passes for all three** — **DONE 2026-08-19**, run from the
      repo, verdict `full_category_sweep` for each:
      | campaign | rows | blank | animal | human | vehicle |
      |---|---|---|---|---|---|
      | `otono_2025` | 8,997 | 7,602 | 818 | 478 | 99 |
      | `primavera_2025` | 16,904 | 15,634 | 744 | 399 | 127 |
      | `otono_2026` | 9,906 | 7,552 | 1,749 | 582 | 23 |
      Otoño 2026's row count is post-video-filter (was 12,068 incl. 2,162 video).

- [x] **1.3b The reviewer's verdict now reaches `observation_type`** — **DONE
      2026-08-19**, and this was the largest data defect found in the V2 pass. Across
      the three campaigns **815 rows carried `observationType=animal` while the reviewer
      had written in `observationComments` that the frame holds no animal**, because the
      review pass wrote its correction into free text while the typed column kept the
      classifier's guess. Primavera's animal count was overstated by 50.6% (744 against
      494) and included 10 people and 4 vehicles.
      `camtrap/observations.py:resolve_review()` now owns the resolution, fail-closed on
      any comment it has no rule for (it refused the ingest on a `Pitio}` typo until the
      cell was fixed). Precedence, agreed with Felipe: an identified animal beats vehicle
      beats human when the review NAMES a species (37 rows: 13 Perro, 23 Caballo, 1 Vaca);
      the review wins outright when it NEGATES the animal (815 rows). The sweep's
      `observationType` is deliberately not an input — the review is the later and closer
      look — and the sweep's own `human` labels stay untouched in `ImageData_total.csv`,
      where `anchor_candidates.py` reads them.
      Resulting animal counts: otoño 2025 830→706, otoño 2026 1,785→1,320,
      primavera 744→494. New canonical column `review_resolution` carries which rule
      fired, including `unknown_pending_taxon` (21 rows) and `unknown_pending_review`
      (3 rows), which mark decisions still open — see 1.12.

- [x] **1.12 The two deferred label decisions** — **CLOSED 2026-08-19** (Felipe).
      **Ruling: a comment that cannot name a species resolves to `unknown`.** So `ave`
      (9 rows, otoño 2025), `roedor` (9, otoño 2026) and `churrete` (1) stay `unknown` —
      `ave` is a class and `roedor` an order, and `Cinclodes` is a genus of several
      species here, so recording any of them as a scientificName would assert more than
      the reviewer saw. The 3 review-note rows (`identificar`, `no reconocible pero
      identificar`, `error de imagen`) are `unknown` too.
      **Two exceptions, adjudicated as identifiable animals** and added to
      `data-pipeline/species.yaml`, which is where species decisions live:
      `conejo` -> *Oryctolagus cuniculus* (`is_invasive`) and `pitío` -> *Colaptes
      pitius*. Both now resolve through the ordinary Spanish-common-name path.
      Two data cells were corrected rather than teaching the code a typo: `conejo?` ->
      `Conejo` (the `?` was rabbit-vs-hare, and *Lepus europaeus* is the most-recorded
      species in these campaigns — read that row with the doubt in mind) and
      `Pitio}` -> `Pitío`.
      Tags renamed accordingly: `unknown_pending_taxon`/`unknown_pending_review` became
      **`unknown_coarse_comment`** (19 rows) and **`unknown_review_note`** (3), since
      nothing is pending any more.
      *Colaptes pitius* is in the catalogue but will not appear in the annual report —
      `taxonomic_group: ave` and rule 4 drops every bird.

- [ ] **1.4 DCIM manifest coverage is stated per campaign**, including the stations
      that legitimately have none. Coverage must be **total within a described
      deployment** or `establish_order` refuses it — partial coverage is worse than
      none. Known: CT15 (1,331 frames) and CT08 (1,129) in otoño 2026 have no folder
      evidence anywhere and never will; that is a limit, not a gap to fill.

- [x] **1.5 Anchors are complete or explicitly refused** — **CT27 DONE 2026-08-24.**
      Install `2025-12-11`, confirmed by Felipe, recorded in
      `otono_2026/deployment_anchors.csv` as an `install` anchor with real == camera
      datetime (offset zero by construction, the primavera CT15 idiom: provenance, not
      repair).
      **The KML timestamp is UTC, and the camera's clock is sound.** The waypoint reads
      15:52:56 and CT27's first frame reads 12:49:01 — three hours apart, which is either
      a UTC/local difference or a 3 h-slow camera. Decided against the retrieval trip:
      CT27's last frame (2026-05-14 14:32:04) sits in correct sequence between CT17
      (14:07:45) and CT21 (15:15:15); a 3 h-slow camera would have read ~11:32 and landed
      out of order between CT10 (09:58) and CT15 (11:51). So 15:52:56 UTC = 12:52:56 local,
      the camera fired 3m55s BEFORE the waypoint was marked — technician mounts it, then
      takes the fix — and all 315 CT27 rows are time-admissible.
      `real_datetime` is the camera frame rather than the waypoint, so no 4-minute repair
      is invented out of technician latency.
      **The anchor belongs to otoño 2026, not primavera.** The visit fell inside the spring
      field window, but CT27 has 0 rows in `primavera_2025` and 315 in `otono_2026` — the
      deployment it opens is otoño 2026's.
      `CT 27.kml` is **not in this repo** (GIS/ holds `areaBP.kml`, `Puntos BP TC.kml`,
      `Puntos de monitoreo TC 2024.kml`). It is not needed: the coordinates it carried are
      registered in `estaciones.csv`, and elevation 1408.06 m is now recorded there too.
      CT16 stays `unrepairable_pending` — its clock emits month `00` and month `16`, so no
      anchor can repair it. CT18 per `HANDOFF-clock-repair.md` §8.1.

- [x] **1.6 One station registry owns station identity** — **DONE 2026-08-24.**
      `estaciones.csv` owns it. `stations.yaml` and `camera_trap_stations.geojson` are
      GENERATED by `setup/build_station_registry.py` and must not be hand-edited.
      **Felipe chose the owner on the evidence, not on seniority.** He asked whether
      `estaciones.csv` was the original registry; it is not — it is the newest of the
      three (2026-08-17, vs the March pair), and the true original is
      `CT ID and coordinates.xlsx`, which is none of them. It owns anyway because it holds
      all 27, uses the canonical `CT##` grammar, carries the columns the visit form writes,
      is already read by `camtrap/stations.py`, and lives in the producer rather than a
      consumer — which is §4's direction rule.
      **The disagreement was smaller than this section claimed.** Measured before any
      change: all three files agreed on every value they *shared* — coordinates, `grid_id`
      and elevation, across all 26 common stations, zero discrepancies. The defect was one
      missing row plus nothing to keep it from recurring.
      **One canonical spelling everywhere (Felipe):** `CT01`..`CT27`. The artifacts had
      said `TC-01`, so the project had two names for one thing. Joins are on the integer
      `tc`, so nothing broke; only labels moved — including pehuén's figure labels and
      camtrapR's `Station` column, which both derive from `id`.
      **`sd_card` dropped outright.** It was the `M##` grid-module tag from the old folder
      names — not an SD card, not unique (`M15` was both CT11 and CT18), and its last
      reader (`01_load_data.R:176`) had stopped using it when the SD cross-validation was
      deleted. This closes the S58 changelog question by removing its subject.
      **The test is stronger than this item specified.** It asserts the committed artifacts
      equal a fresh render, not that "all three agree on count and coordinates to 5 decimal
      places" — that check restates the projection in a second place and passes vacuously
      on any field it does not enumerate, which is precisely how `sd_card` lived in the
      artifacts and in no test for five months.
      Still unknown: CT27's `grid_id`, for the field.

      <details><summary>The state before the fix, kept for the record</summary>

      Three disagreed:
      | file | stations | CT27 |
      |---|---|---|
      | `plataforma-territorial/data/stations.yaml` | **26** | **absent** |
      | `plataforma-territorial/data/camera_trap_stations.geojson` | 27 | present, `altitude_m: null` |
      | `camera-traps/data/campaigns/estaciones.csv` | 27 | present, `elevation_m` empty |
      `stations.yaml` is the one `data-pipeline/src/stations.py` documents as "the
      single source of truth", and it is the one missing CT27 — so CT27's 344 otoño
      2026 files ingest with no coordinates.
      **Recommendation: `estaciones.csv` becomes the owner** — it holds all 27 and
      carries the field columns the visit form writes (`grid_id`, `height_m`,
      `bearing_deg`, `detection_distance_m`), and `camtrap/stations.py` already reads
      it. The other two become generated artifacts.
      Pass: a test asserts all three agree on station count and on coordinates to 5
      decimal places. **This check is what makes CT26 and CT27 impossible to repeat.**
      Still unknown after the KML: CT27's `grid_id`. Elevation is now known (1408.06 m).

      (`344` above is wrong — the canonical table holds **315** CT27 rows.)
      </details>

- [ ] **1.7 `field_notes.csv` audited beyond coordinates.** The 2026-08-17 pass
      repaired the coordinate column and nothing else; no other column has been checked
      for the same class of error. 57 of 106 rows carry a `data_flags` entry.

- [ ] **1.8 `provenance.py` re-run across all campaigns** — one deployment, one capture
      story. Last validated 2026-08-14 on 28,178 files with 0 false positives; the
      re-ingested primavera is data it has not seen.

- [ ] **1.9 Dead and stale code removed.** Full list in §3.

- [ ] **1.10 Test suite extended.** **226 pass as of 2026-08-24** in camera-traps
      (190 on 2026-08-19, +19 through 2026-08-20, +17 for the station registry).
      Run them with `python -m unittest discover -s tests` — **pytest is not installed in
      the `camera-traps` env**, which is why the 152 figure went unverified on 2026-08-18.
      **226 pass as of 2026-08-24** (+17: `tests/test_station_registry.py`).
      Of the three regression fixtures, **one is done**: the registry-agreement check from
      1.6, written as "the committed artifacts equal a fresh render" rather than the
      three-way comparison this document specified. Still missing: the manifest rebuild
      from a flatten log, and the size-matched deletion accounting.
      **`data-pipeline` has no test suite at all** — 1,642 lines, zero tests, while it now
      carries four modules whose guarantees rest on having been run once by hand. Designed
      and deferred 2026-08-24: `data-pipeline/docs/TEST-PLAN.md`.

- [ ] **1.11 Outputs regenerated and every moved number attributed.**
      **Canonical tables rebuilt 2026-08-19, now ONE ROW PER STILL** (see 1.13):
      `otono_2025` 8,997, `primavera_2025` 16,904, `otono_2026` 9,906 — **35,807 total**
      via `read_campaigns`, of which 3,359 reviewed and 32,448 `sweep_only`.
      Post-rebuild: animal 2,522 / blank 31,090 / human 1,424 / unknown 521 /
      vehicle 250, and **zero rows are `animal` with an empty species**.
      **The annual report moved by exactly one record, and the cause is named.**
      `01_data_prep.py` output diffed before and after at row level: **1 row added, 0
      removed** — CT04 `01130013.JPG` *Oryctolagus cuniculus*, i.e. the `conejo?`
      adjudication from 1.12. Final records 641 -> 642, events 261 -> 262, species kept
      11 -> 12. **The all-stills rebuild itself moved nothing**, because no `sweep_only`
      row is ever typed `animal` (asserted in `tests/test_ingest_frame.py`) and the
      report filters on `animal` + non-empty species.
      Figures are **not** re-rendered yet. Remaining causes to attribute when they are:
      video leaving the denominators, and the 815-row review repair.

- [x] **1.13 The canonical table describes every still, not only reviewed rows** —
      **DONE 2026-08-19.** Seven station-campaigns were absent from the tables because
      they recorded no animal: **CT23** (otoño 2025), **CT01/CT06/CT17/CT22** (primavera —
      6, 21, 7 and 18 frames each), **CT02/CT12** (otoño 2026). A station missing from the
      table is indistinguishable from one never deployed, which is fine for a detection
      numerator and wrong for a trap-effort denominator — and the module docstring already
      promised otherwise.
      `observations.compose_ingest_frame()` now pins the row set to the gated export and
      attaches the review where one exists; `resolve_observation()` decides where each
      row's verdict comes from — the review for reviewed rows, the sweep for the rest,
      tagged `sweep_only`. Station gap is now **0 in all three campaigns**.
      `new_labeled_data_corrected.csv` deliberately stays reviewed-only: pehuen reads it
      and has no use for 32,000 swept rows it would filter straight back out.
      **The remaining station-count difference is RESOLVED, 2026-08-19 (Felipe): the
      grid was built up over time.** Cameras were installed as the programme went, so
      otoño 2025 covers **21** stations, primavera **26** and otoño 2026 **27** because
      that is how many existed at each retrieval. It is the real deployment history, not a
      pipeline gap, and it must NOT be "fixed" — a later session finding 21 against 27
      should read this line and stop. What has to stay equal across campaigns is the
      *file set* (1.2) and the *row-set rule* (every still in the export), never the
      station count.

---

## 2. Data-pipeline — the DuckDB rebuild

> ## ✅ CLOSED 2026-08-24. 2.1, 2.2, 2.3, 2.5 and 2.8 are all done; §2 is complete.
>
> The database now holds **35,807 rows across 74 deployments** (21 + 26 + 27 stations —
> the real deployment history), rebuilt from `observations.parquet` and reconciling
> exactly against the published contract. `weather_station` (264,943 rows) and
> `weather_forecast` (4,343) are committed as per-year Parquet, so **the Windows↔Linux
> question is permanently answered**: any machine can rebuild the warehouse from the
> repository alone. `literature` was dropped — 0 rows, no reader.
>
> **Three specifications in this section were wrong.** They are corrected in place below,
> each measured rather than argued: the mandated key cannot be a primary key (2.8), the
> column contract omits `campaign` (2.3), and the timestamps are not UTC (2.3).

> **UNBLOCKED 2026-08-20, by deletion rather than by repair.** This section was blocked
> the same day because `timelapse_reviewed.py` re-derived the review-comment resolution and
> disagreed with `camtrap.observations` on 515 live rows — ingesting would have rebuilt the
> defect 1.3 closed. **2.4 and 2.6 are now done and 2.7 is moot:** both parsers, the
> dedup script, the `--ct` campaign list and the ingest functions are gone, and
> `ingest_all_ct_campaigns` raises `CameraTrapIngestNotRebuilt` with the reason. Nothing can
> now silently ingest the wrong thing. **What remains is 2.1, 2.2, 2.3, 2.5 and 2.8** —
> building the replacement from `observations.parquet` and recovering the irreplaceable
> weather tables from the Linux box.

**State as of 2026-08-18.** The Windows `fma_data.duckdb` (1.5 MB, 2026-03-31) has
**zero** camera-trap rows — `ct_deployments`, `ct_media`, `ct_observations` all empty,
`literature` empty; only weather has data. The populated database is on the **Linux**
machine. `fma_data.duckdb.bak-2026-05-27` (60 MB) holds 41 deployments / 1,622 media /
1,622 observations under **pre-flatten, pre-rename** identity
(`primavera_verano_2025_2026_TC20_M17.2`, and `oto_o_2025_CT07` with the ñ mangled),
covering only two campaigns.

- [x] **2.1 Split the decision by regenerability — do not move the file as a whole.**
      **DONE 2026-08-24.** Measured on the Linux box before deciding: `weather_station`
      **264,943** rows spanning 2018-09-21 → 2026-04-13, `weather_forecast` **4,343**,
      `literature` **0**, and the `ct_*` tables **54 / 2,948 / 2,948** — every one of them
      `source='timelapse_reviewed'`, a parser deleted on 2026-08-20, under pre-flatten
      identity (`oto_o_2025_CT07`, ñ mangled) and keyed on Timelapse GUIDs. Orphaned, not
      stale. Dropped and rebuilt.
      ⚠️ `duckdb_tables().estimated_size` **lies** — it reported 24,665 / 12,222 for
      `ct_media` / `ct_observations` against actual 2,948 / 2,948. Use `COUNT(*)`.
      `literature` was dropped with them (Felipe): 0 rows and no reader in this monorepo.
      Its DDL is removed rather than left in place, because `init_schema()` runs on every
      connect and a `CREATE TABLE IF NOT EXISTS` would recreate the empty table forever.

      <details><summary>Original text</summary>
      | tables | regenerable? | action |
      |---|---|---|
      | `ct_deployments`, `ct_media`, `ct_observations` | yes, from `observations.parquet` (in git, both machines) | rebuild, never migrate |
      | `weather_station`, `weather_forecast`, `literature` | **no** — CR800 pulls and open-meteo history cannot be refetched retroactively | recover from Linux |
      First command, before deciding anything:
      `duckdb fma_data.duckdb -c "select table_name, estimated_size from duckdb_tables() order by table_name"`
      If Linux's weather tables exceed the backup's 264,943 `weather_station` rows,
      that copy is the one to preserve.
      </details>

- [x] **2.2 Export the irreplaceable tables to Parquet and commit them** —
      **DONE 2026-08-24**, `data-pipeline/src/recovery.py`, committed under
      `data-pipeline/data/recovery/`.
      **Partitioned by year, which this item did not anticipate.** Parquet is opaque to
      git and the export runs on every poll, so a single 17.5 MB blob would be stored
      again in full each time. A year that has ended never changes, so per-year files mean
      only the current year's blob (632 KB) is ever rewritten.
      **`export` and `restore` are separate verbs and there is no `sync`.** Guessing the
      direction on irreplaceable data is how the empty copy overwrites the good one — the
      hazard 2.1 names. `restore` refuses when the database has MORE rows than the archive;
      `export` refuses when a year-file exists that the table has no rows for.
      **Verified the way that matters:** an empty database built from `schema.sql` alone,
      then restored — 264,943 rows, all 41 columns in the same order (including the 33
      dynamically-added TOA5 columns), first five rows identical, mean temperature equal to
      six decimals. `python -m src.recovery verify` is the standing check.
      Note: `literature` was on this list and dropped from it — it holds 0 rows, so there
      is nothing irreplaceable to preserve.

- [x] **2.3 Rebuild `ct_*` from `observations.parquet`**, not from the reviewed CSVs —
      **DONE 2026-08-24**, `data-pipeline/src/parsers/canonical_ct.py`. 35,807 rows,
      74 deployments, reconciling exactly. `run_fetch.py --ct`.

      **Two corrections to the contract below, both measured:**

      **(a) `campaign` is missing from the `ct_deployments` column list.** The platform
      queries it in three places (`/summary-stats`, and the campaign filters). It existed
      in the live table only because `ensure_columns()` had added it dynamically at some
      past ingest, so `schema.sql` and this document had both drifted from reality.
      Rebuilding to the list as written would have broken the platform silently. Now
      declared. Same for `observationComments`, `reviewOutcome` and `reviewResolution`.

      **(b) "Timestamps are UTC" is wrong, and storing them as UTC would be worse.**
      A camera clock reading is wall time of unknown accuracy — there is no instant to
      recover, and 11% of rows have no datetime at all. `TIMESTAMPTZ` would force this
      table to invent a UTC offset per row, ambiguous twice a year at the DST boundary,
      and would make `HOUR(eventStart)` depend on the reader's session timezone rather than
      on what the camera saw. The diel-activity figure needs the camera's LOCAL hour.
      So `ct_*` timestamps are **naive local (America/Santiago wall time)**, while
      `weather_station` stays `TIMESTAMPTZ` because a datalogger reading IS a known
      instant. That asymmetry is deliberate and documented in `schema.sql`. Consistent
      with the horario-de-invierno decision (no DST correction, ever) and with this
      document's own note that `datetime` is naive local today.

      **Also decided, and flagged at the time as judgement calls:**
      `deploymentStart`/`deploymentEnd` are the **observed** window (min/max media
      timestamp), carried with a new `deploymentWindowSource` column so the provenance is
      explicit rather than implicit in a docstring — when the field form's loader lands
      (1.14) real windows arrive as `field_record` and consumers can tell them apart.
      `locationID` stays the camera number as text, because the platform does
      `int(locationID)` and logs-and-skips anything else. `count` and `eventID` stay NULL.

      <details><summary>The original column contract, preserved from camtrap_dp.py</summary>

      **The column contract**, preserved here 2026-08-20 from `camtrap_dp.py` before that
      parser was deleted — it was the only written statement of the `ct_*` shape, and it
      had never had an input folder to parse:

      | table | columns |
      |---|---|
      | `ct_deployments` | `deploymentID`, `locationID`, `locationName`, `latitude`, `longitude`, `deploymentStart`, `deploymentEnd`, `cameraID`, `cameraModel`, `habitat`, `source` |
      | `ct_media` | `mediaID`, `deploymentID`, `timestamp`, `fileName`, `filePath`, `fileMediatype`, `source` |
      | `ct_observations` | `observationID`, `deploymentID`, `mediaID`, `eventID`, `eventStart`, `eventEnd`, `observationType`, `scientificName`, `count`, `classificationMethod`, `classificationProbability`, `source` |

      Timestamps are UTC; `count` is nullable `Int64`; `source` names the producer.
      **Keys must derive from the image, never be inherited from Timelapse** — see 2.8.
      `count` and `eventID` are empty in all three campaigns (measured 2026-08-20), so they
      stay nullable and unpopulated rather than being invented.
      </details>

- [x] **2.4 Retire `timelapse_reviewed.py`'s duplicate derivations** — DONE 2026-08-20.
      It re-derived five decisions `camtrap/observations.py` owns: station→camera number,
      coordinates, Spanish→Latin species, Santiago→UTC, **and the review-comment
      resolution**. The module docstring's claim that "every consumer reads this shape and
      nothing else" was false while that parser existed; deleting it makes the claim true.

      **The fifth duplicate is why this was blocking rather than cosmetic.**
      `NON_ANIMAL_COMMENTS` knew four comment strings and only ever demoted to `blank`. It
      had **no rule producing `human`, `vehicle` or `unknown` from a comment** — the rules
      `resolve_review()` was written to own. Had it run, the `ct_*` rebuild would have
      reproduced the exact 815-row defect closed in 1.3:

      | comment | live rows | `camtrap.observations` | `timelapse_reviewed.py` |
      |---|---|---|---|
      | `No reconocible` | 500 | `unknown` | unmapped -> stays `animal` |
      | `humano` | 10 (pv) | `human` | unmapped -> stays `animal` |
      | `vehiculo` | 4 (pv) | `vehicle` | unmapped -> stays `animal` |
      | `error de imagen` | 1 (ot25) | `unknown` | `blank` |

      (`no aparece imagewn`, the fourth string it knows, exists **only** in
      `pv_2025_2026` — which is why the fail-closed resolver in camera-traps never met
      it, and why the parser looks harmless on inspection.)

      It was **not** fixed by teaching the parser the new rules — that would have been a
      second place a repair must reach, and per §0 it would not have reached it. The parser
      was deleted. 2.3 is the replacement: the consumer reads `observations.parquet`, where
      the resolution has already happened.

      Also deleted with it: `camtrap_dp.py` (81 lines, parsed a Camtrap DP folder that has
      never existed in this monorepo — its column mapping is preserved under 2.3), the
      `ingest_camtrap_dp` / `ingest_timelapse_reviewed` functions, and the watcher's folder
      and CSV branches that called them.

- [x] **2.5 Registry dependency fixed** — **DONE 2026-08-24**, see 1.6. Verified in the
      rebuilt table: **0 deployments are missing coordinates**, and CT27 carries
      `-39.45689, -71.72243`.

- [x] **2.6 Delete `scripts/dedup_primavera_2025.py`** — DONE 2026-08-20. Its whole
      premise had dissolved: pv is no longer a separate campaign, and the "unmappable
      `100EK113` folder" it excluded was resolved into CT05 by the 2026-08-13 flatten.
      **This was a data change, not just a deletion** — those records stop being excluded.
      Its output file (`new_labeled_data_reviewed.dedup.csv`) did not exist on disk, which
      is why the campaign loop skipped primavera and ingested pv instead.

- [x] **2.7 `config.yaml`'s `camera_traps.campaigns`** — DONE 2026-08-20, by removal
      rather than rewrite. There is no per-campaign CSV list any more: the replacement reads
      `observations.parquet` per campaign, so the setting has nothing to configure. The
      block is replaced by a comment recording what it used to do and why it was wrong
      (it named a nonexistent primavera file and so ingested pv as a campaign).

- [x] **2.8 Reconcile** — **DONE 2026-08-24.** `_reconcile()` runs at the end of every
      rebuild and RAISES on any mismatch, reading counts back from the DATABASE rather
      than from the frames just built (a reconciliation that trusts the frames checks
      nothing). It also asserts `ct_media` and `ct_observations` are 1:1.

      Reconciled: `otono_2025` 8,997 / 21 · `primavera_2025` 16,904 / 26 ·
      `otono_2026` 9,906 / 27 — **35,807 rows, 74 deployments**, matching the published
      contract exactly, including `animal 2,522 / blank 31,090 / human 1,424 /
      unknown 521 / vehicle 250` and the 3,359 human- vs 32,448 machine-classified split.

      **THE KEY THIS ITEM MANDATES CANNOT BE A PRIMARY KEY.** `DEDUP_KEY = [camera_num,
      file_name, datetime]` — but `datetime` is **NULL in 4,013 of 35,807 rows** (11%),
      the clock-failure stills that must stay in the table because presence needs a
      station and not a clock. Measured across all three campaigns,
      **`(campaign, camera_num, file_name)` is unique for all 35,807 rows and never null**,
      so that is what identity derives from. The guarantee this item actually cares about
      is unchanged and is preserved: keys are derived from the image, never inherited from
      Timelapse.

      **The rebuild is whole, never incremental.** The table is small and the parquets are
      the entire truth, so a full replace costs nothing and cannot strand a row from a
      campaign that shrank or was retired — which is exactly how `pv_2025_2026` lived on
      in this database as a phantom campaign after being dropped upstream. Verified gone.

      <details><summary>The 2026-08-20 correction about Timelapse GUIDs, still true</summary>

      **Correction, 2026-08-20 — the earlier note here was wrong in a way that pointed
      at the wrong risk.** `upsert_df` is indeed `INSERT OR REPLACE` on
      `mediaID`/`observationID`, but those are **not** "regenerated on every parse":
      `timelapse_reviewed.py` reads them straight out of the export's `mediaID` /
      `observationID` columns and generates nothing. They are GUIDs minted by Timelapse2
      and stored in the project's `.ddb` `DataTable`, so they are **stable across every
      export of the same project** — re-running ingest replaces, it does not duplicate.

      What *is* true is narrower and worse: the GUIDs are **per project, not per image**.
      Joining primavera's legacy `CamtrapDB_Primavera2025.ddb` against its current
      `TimelapseData.ddb` on `File` gives 2,387 shared filenames and **0 shared
      `mediaID`s**. So a rebuilt or re-created Timelapse project mints an entirely new
      identity for the same photograph, and any `ct_*` row keyed on the old one is
      orphaned rather than updated.

      The consequence is a design requirement, not a procedure: **`ct_*` keys must be
      derived from the image, not inherited from Timelapse.** `observations.parquet`
      already carries a stable natural key — `DEDUP_KEY = [camera_num, file_name,
      datetime]` — and 2.3 says to build from the parquet. Do that and this whole class
      of breakage disappears; inherit Timelapse GUIDs and the next project rebuild
      silently forks the table.
      </details>

---

## 3. Stale code inventory (audited 2026-08-18)

### Breaks — silently, not loudly

> **Status 2026-08-24.** The three "breaks silently" entries are resolved: the loader was
> rewritten onto the canonical parquet (2026-08-20), and `config.yaml`'s `camera_traps`
> block was removed (2.7). What remains in this section is the C-tier cleanup and the
> otoño-2026 exclusion in B3.

- `Research/pehuen-species-interactions/R/01_load_data.R:52` — `PATH_PV` points at
  `pv_2025_2026/new_labeled_data_corrected.csv`, which will no longer be regenerated.
- Same file, `:196–205` — parses station labels as `^TC(\d+)_` (`TC10_M3.2`). The
  re-ingested primavera carries canonical `CT##`, so `tc_num` becomes NA on every row
  and the geojson join drops the whole campaign. **Fails silently: a smaller dataset,
  no error.**
- `data-pipeline/config.yaml:32–37` — points at
  `primavera_2025/new_labeled_data_reviewed.dedup.csv` (generator about to be deleted)
  and the separate pv entry.

### Dead or wrong after the re-ingest

| Location | What rots |
|---|---|
| ~~`data-pipeline/scripts/dedup_primavera_2025.py`~~ | **deleted 2026-08-20** (2.6) |
| `camtrap/observations.py:70–79, 185` | `CAMPAIGN_ORDER` entry + both precedence comments (396 overlap / 31 conflicts) |
| `camtrap/observations.py:7–9` | per-campaign export quirks — `filePath` populated in primavera_2025, `timestamp` only there. A fresh export has different quirks |
| `Anual-reports/2025/py/01_data_prep.py:6, 71` | `REPORT_CAMPAIGNS` |
| ~~`Anual-reports/2025/py/list_ciervo_guina_images.py:34–44, 122`~~ | **fixed 2026-08-20** — reads otoño 2025 / primavera 2025 / otoño 2026; pv dropped |
| `Anual-reports/2025/py/apply_verdicts.py:143` | comment on primavera→pv survival |
| `data/campaigns/label_conflicts_primavera_vs_pv_2026-05-27.csv` | a conflict that no longer has two sides |
| `Anual-reports/2025/data/manual_review_ciervo_guina.md` | every row keyed to `TC*_M*.2` paths and a `pv_2025_2026` column |
| `exports/Primavera-verano 2025-2026/` | thumbnail tree named for the old station convention |

### Pre-existing defects the re-ingest will expose

**Read statically — R is not installed on this Windows box, so this was not executed.**

`01_load_data.R` emits campaign labels `Otono_2025`, `PrimaveraVerano_2025_2026`,
`Otono_2026`, and never recodes them. But:

- `05_spatial_distribution.R:184` builds its grid with
  `campaign = c("Otono_2025", "Primavera_2025")`. `Primavera_2025` is a label the
  loader has never produced, so the `left_join` matches nothing and the NA→0 replace
  turns the whole panel into zeros. Otoño 2026 is excluded outright.
- `02_detection_summary.R:79, 106, 146` — same non-existent key in `labeller()`, so
  those facets fall through to the raw label.

⚠️ **If the loader is updated to emit `Primavera_2025`, script 05 starts matching for
the first time and that figure changes from zeros to real data** — which will look like
the re-ingest moved it when it actually un-broke a join. Fix and re-render this
*separately* from the re-ingest so the two effects are not attributed to each other.

> **Re-measured 2026-08-24 — B3 is narrower than written above.** The loader now emits
> `Primavera_2025`, so script 05's join is already un-broken and that half is done. What
> is still wrong is only **otoño 2026 falling out**: `05_spatial_distribution.R:249` builds
> its grid on `c("Otono_2025", "Primavera_2025")` and
> `02_detection_summary.R:83,110,150` label only those two.
>
> **A third effect now lands in the same figures:** `id` renders `CT01` instead of `TC-01`
> (1.6), so pehuén's station labels and camtrapR's `Station` column both change value. No
> join breaks — the join is on the integer `tc_num`, and `record_table$Station` and
> `CTtable$Station` both derive from `id`, so they move together.
>
> That makes **three** distinct causes queued for one set of figures. Render them as
> separate steps or the attribution is lost.
>
> Felipe does not want pehuén run on the Linux box, so this is Windows-side work. The
> `sd_card` removal at `01_load_data.R:176` is therefore **committed but unexecuted** —
> it is a one-line `select()` change, and without it the script would error against the
> regenerated GeoJSON, but nobody has run it.

---

## 4. The gate — the DuckDB step cannot be silently skipped

**Direction is the design, and it is never reversed: camera-traps publishes,
data-pipeline verifies. camera-traps must not learn that DuckDB exists.**

- **camera-traps side.** The ingest writes `data/campaigns/CANONICAL_STATE.json` — per
  campaign: campaign name, row count, a hash of `observations.parquet`, and the write
  timestamp. Regenerated on every ingest, committed.
- **data-pipeline side.** A freshness check reads that file, compares it against a
  small `ct_ingest_state` table it maintains, and **refuses to report success while
  they diverge.** A `--check` mode reports staleness without writing.
- **Fail-closed.** A missing or unreadable state file means refuse, not proceed — the
  same posture as the flatten preconditions, and for the same reason: the failure this
  prevents is silent.

This is the enforceable half, because it lives in the tool that builds the DB rather
than in anyone's memory. A memory entry exists as well; it is the weak half and must
not be relied on.

## ✅ BOTH PIECES ARE WRITTEN — 2026-08-24

- **camera-traps side:** `camtrap/canonical_state.py` + `data/CANONICAL_STATE.json`,
  shipped 2026-08-20.
- **data-pipeline side:** `src/canonical_gate.py` + the `ct_ingest_state` table.
  `python run_fetch.py --ct-check` reports staleness, writes nothing, and **exits 1** —
  reporting drift without a non-zero exit is how a stale database goes on being served
  while a log line nobody reads says otherwise.

**It reads the JSON as a FILE and does not import `camtrap`.** That is the direction rule
made structural: importing the producer would have the check running the producer's own
code against the producer's own data, where it could only ever agree with itself.

**The gate runs FIRST and the state is stamped LAST**, so an interrupted rebuild reports
stale rather than finished.

**One addition beyond this specification.** The gate fingerprints the *whole* campaign
description (SHA-256 over the sorted JSON), not just row counts. The 815-row review repair
of 1.3b moved `observation_types` and `n_animal_rows` while leaving `n_rows` untouched —
invisible to a row-count comparison, and precisely the change a consumer most needs to
notice.

**Tested that it actually refuses:** missing state file → refuse with instructions;
malformed JSON → refuse; unknown `schema_version` → refuse; tampered row count → caught;
`ct_ingest_state` wiped → all three campaigns reported never-ingested; a campaign in the
database but not published → flagged as the `pv_2025_2026` shape.

⚠️ Those checks were run by hand once. **They are not in a test suite** — `data-pipeline`
has none. See `data-pipeline/docs/TEST-PLAN.md`.

---

## 5. Project boundaries — the standard to hold

Audited 2026-08-18. Direction is mostly sound; the defects are about *how* the reach
happens, not which way.

- **camera-traps → data-pipeline:** only `species.yaml`, a data file, with an
  `FMA_SPECIES_YAML` override. `classify_campaign/species.py` deliberately duplicates
  `data-pipeline/src/species.py` and says why — camera-traps runs on Windows in its own
  env and cannot assume data-pipeline is present. **This is correct; leave it.**
  There is **no import cycle.**
- **data-pipeline → camera-traps:** reads campaign CSVs by relative path. Fine in
  direction; §2.3 narrows it to the canonical parquet alone.
- **Research → camera-traps, plataforma-territorial:** one-directional. ~~Reaches in by
  hardcoded absolute Windows paths (`C:/Users/USUARIO/...`), which is why pehuen cannot run
  on Linux.~~ **FIXED 2026-08-24**, and it was not the two-line change it looked like.

  The paths now derive from the project's own location — but that only works once `here()`
  points at the right place, and **it did not.** There was no `.Rproj` or `.here` anywhere
  in the repo, so `rprojroot` walked up to the monorepo's `.git` and `here::here()` returned
  `<root>` instead of `<root>/Research/pehuen-species-interactions`. **All 48 `here()` calls
  across the six scripts were therefore resolving to a top-level `data/` that does not
  exist** — reads would error, writes would create a stray directory. The absolute paths had
  been masking it, because they were the only paths in the project that did not go through
  `here()`.

  Fix: a committed `.here` anchor file (`has_file(".here")` is rprojroot's highest-priority
  criterion), then `MONOREPO <- normalizePath(here::here("..", ".."))` with an
  `FMA_MONOREPO` override and a fail-fast check that both sibling projects exist. No
  absolute path remains anywhere in the project.

  **Verified on Linux:** all 7 scripts parse, `sf` loads (the Mingw-w64 pseudo-relocation
  error was a Windows/Git-Bash problem and does not occur here), the regenerated GeoJSON
  reads as 27 features with `CT01`-form ids and no `sd_card`, and the corrected `select()`
  in `01_load_data.R` succeeds.

  **Still not runnable here, but not for a code reason:** `camtrapR`, `nanoparquet` and
  `arrow` are absent from the `pehuen-analysis` env. All three are CRAN-only by design and
  `environment.yml` says so; `Rscript setup_packages.R` has simply never been run on this
  box. One command, not a defect.
- **The unresolved one:** `species.yaml` is "canonical source for data-pipeline,
  camera-traps, and plataforma-territorial" while living inside one of the three. Its
  home is arbitrary. Mitigated by the env var, so this is an observation, not a defect
  to fix under this review.

**Rule to hold throughout:** a consumer reads the canonical table and nothing else.
Every time a consumer re-derives something the producer already owns, that derivation
becomes a second place a repair has to reach — and the repair will not reach it. That
is the CT26 failure in general form, and it is why §2.4 exists.
