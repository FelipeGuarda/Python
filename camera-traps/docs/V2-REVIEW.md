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
| **A1** | station registry disagrees: `stations.yaml` **26** · geojson **27** · `estaciones.csv` **27**; CT27's 344 files ingest coordinateless | 1.6 | open — **do first**, A4 inherits it |
| **A2** | contract published, consumer-side freshness check absent (`grep -r CANONICAL_STATE data-pipeline/` → nothing) | §4 | open |
| **A3** | **the field form has no loader** — `build_visit_template.py` writes the workbook, nothing reads it back | **1.14, new** | open |
| **A4** | `occupancy_pct` divides by all 27 stations | 1.6 / §2.5 | open, gated on §2 |
| **B1** | the `ct_*` rebuild | 2.1–2.3, 2.5, 2.8 | open — Linux |
| **B2** | pehuén's absolute Windows paths | 1.11 | open |
| **B3** | figures not re-rendered; otoño 2026 falls out of `05_spatial_distribution.R:249` and the `02_detection_summary.R` labellers | 1.11 + §3 | open — **re-render that separately** |
| **B4** | `field_notes.csv` audited for coordinates only, 57/106 rows flagged | 1.7 | open |
| **B5** | `provenance.py` not re-run on the re-ingested primavera | 1.8 | open |
| **B6** | manifest coverage not stated per campaign | 1.4 | open |
| **B7** | CT27 install datable from `CT 27.kml` (2025-12-11 15:52:56), unrecorded | 1.5 | open |
| **B8** | three regression fixtures | 1.10 | open |
| **C1–C5** | two superseded data files on disk · stale pv comment `apply_verdicts.py:143` · otoño 2025 video existence unconfirmed · `count` empty · seasonal puma orphan | §3 | open |

**The pattern, and it is the useful part:** every open item sits at one of the two
**boundaries** — the field record coming in, or a consumer going out. The chain from card to
canonical table is enforced and tested. Defects survive where responsibility changes hands,
because that is where nobody owns the check.

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

- [ ] **1.5 Anchors are complete or explicitly refused.** CT27's install is now
      datable — the GPS waypoint in `CT 27.kml` reads 2025-12-11 15:52:56, which
      resolves the 2025-11-12 / 2025-12-11 ambiguity (the two candidates are a
      day/month transposition of each other). Record it as evidence reconciled against
      the field record, **not** as an install date written in silently. CT16 stays
      `unrepairable_pending` — its clock emits month `00` and month `16`, so no anchor
      can repair it. CT18 per `HANDOFF-clock-repair.md` §8.1.

- [ ] **1.6 One station registry owns station identity.** Three disagree today:
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

- [ ] **1.7 `field_notes.csv` audited beyond coordinates.** The 2026-08-17 pass
      repaired the coordinate column and nothing else; no other column has been checked
      for the same class of error. 57 of 106 rows carry a `data_flags` entry.

- [ ] **1.8 `provenance.py` re-run across all campaigns** — one deployment, one capture
      story. Last validated 2026-08-14 on 28,178 files with 0 false positives; the
      re-ingested primavera is data it has not seen.

- [ ] **1.9 Dead and stale code removed.** Full list in §3.

- [ ] **1.10 Test suite extended.** **190 pass as of 2026-08-19** (+38: 20 for the
      review-comment resolution, 7 for the stills-only export precondition, 11 for the
      ingest row set and where each row's verdict comes from).
      Run them with `python -m unittest discover -s tests` — **pytest is not installed in
      the `camera-traps` env**, which is why the 152 figure went unverified on 2026-08-18.
      Still to add regression fixtures
      for what the 2026-08-18 session established: the manifest rebuild from a flatten
      log, the size-matched deletion accounting, and the registry-agreement check from
      1.6.

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

- [ ] **2.1 Split the decision by regenerability — do not move the file as a whole.**
      | tables | regenerable? | action |
      |---|---|---|
      | `ct_deployments`, `ct_media`, `ct_observations` | yes, from `observations.parquet` (in git, both machines) | rebuild, never migrate |
      | `weather_station`, `weather_forecast`, `literature` | **no** — CR800 pulls and open-meteo history cannot be refetched retroactively | recover from Linux |
      First command, before deciding anything:
      `duckdb fma_data.duckdb -c "select table_name, estimated_size from duckdb_tables() order by table_name"`
      If Linux's weather tables exceed the backup's 264,943 `weather_station` rows,
      that copy is the one to preserve.

- [ ] **2.2 Export the irreplaceable tables to Parquet and commit them**, so the
      recovery is repeatable rather than a one-time file copy. This is what makes the
      Windows↔Linux migration stop being a blocker: the only data that must travel is
      this, once.

- [ ] **2.3 Rebuild `ct_*` from `observations.parquet`**, not from the reviewed CSVs.

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

- [ ] **2.5 Registry dependency fixed** — see 1.6. Until then CT27 ingests coordinateless.

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

- [ ] **2.8 Reconcile.** DB row counts equal parquet row counts exactly, per campaign.

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

---

## 3. Stale code inventory (audited 2026-08-18)

### Breaks — silently, not loudly

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

**Neither piece is written yet.** Each gets its own design gate, written against real
re-ingested data rather than guessed now.

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
- **Research → camera-traps, plataforma-territorial:** one-directional, but reaches in
  by **hardcoded absolute Windows paths** (`C:/Users/USUARIO/...`), which is why pehuen
  cannot run on Linux. Parameterise.
- **The unresolved one:** `species.yaml` is "canonical source for data-pipeline,
  camera-traps, and plataforma-territorial" while living inside one of the three. Its
  home is arbitrary. Mitigated by the env var, so this is an observation, not a defect
  to fix under this review.

**Rule to hold throughout:** a consumer reads the canonical table and nothing else.
Every time a consumer re-derives something the producer already owns, that derivation
becomes a second place a repair has to reach — and the repair will not reach it. That
is the CT26 failure in general form, and it is why §2.4 exists.
