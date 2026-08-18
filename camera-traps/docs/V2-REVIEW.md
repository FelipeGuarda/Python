# V2 REVIEW — every campaign clean, every consumer clean, before anything new starts

**Written:** 2026-08-18 · **Status:** pending, **not started**
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

1. `primavera_2025`'s re-review is finished in Timelapse2 (all 19,522 rows classified;
   they are `unclassified` as of 2026-08-18).
2. Its `ImageData_total.csv` is exported and **passes `exports.read_total_export`**
   with `verdict='full_category_sweep'`, `n_rows=19522`.
3. `pv_2025_2026` is confirmed retired-to-provenance, not deleted.

**Out of scope, deliberately.** Both are real and both are larger than this review;
folding them in would stall it. Neither blocks it.
- The DST decision's piece 2 — storing the instant plus a fixed per-deployment offset
  in `observations.parquet` (`datetime` is naive local today).
- The sun-anchored sensitivity run in pehuen (piece 3).

---

## 1. Camera-traps — the review

Each item is a check with a stated pass condition, not a task to eyeball.

- [ ] **1.1 The campaign set is exactly three.** `otono_2025`, `primavera_2025`,
      `otono_2026`. `pv_2025_2026` retired: directory kept as provenance, removed from
      `CAMPAIGN_ORDER` (`camtrap/observations.py`) and `REPORT_CAMPAIGNS`
      (`Anual-reports/2025/py/01_data_prep.py`).
      **Order matters:** drop it from `CAMPAIGN_ORDER` only *after* the new
      `primavera_2025` parquet exists, or reading pv's still-present parquet raises
      `UnorderedCampaign`.

- [ ] **1.2 The canonical file set is identical across campaigns.** It is not today —
      otoño 2025 carries `CamtrapDB_Otono_2025.ddb`, primavera a `.tdb`, pv neither a
      manifest nor an export. Decide the required set, write it down, assert it.
      Candidate required set per campaign: `ImageData_total.csv`,
      `timelapse_recognition_file.json`, `new_labeled_data_reviewed.csv`,
      `new_labeled_data_corrected.csv`, `dcim_manifest.csv`, `deployment_anchors.csv`,
      `observations.parquet`, `timestamps_audit.log`.
      Pass: a check lists, per campaign, present / missing / unexpected — and every
      "missing" is either fixed or justified in one line here.

- [ ] **1.3 The export gate passes for all three**, verdict recorded per campaign.
      Known as of 2026-08-18: otoño 2025 and otoño 2026 pass; primavera has no export.

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

- [ ] **1.10 Test suite extended.** 152 pass as of 2026-08-18. Add regression fixtures
      for what the 2026-08-18 session established: the manifest rebuild from a flatten
      log, the size-matched deletion accounting, and the registry-agreement check from
      1.6.

- [ ] **1.11 Outputs regenerated and every moved number attributed.** Diff against
      `Anual-reports/2025/figures_pre_reingest/`. Expect large movement: primavera goes
      from 14 stations / 1,960 observations to 26 stations / 19,522 files. A number that
      moves without a named cause is a finding, not noise.

---

## 2. Data-pipeline — the DuckDB rebuild

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

- [ ] **2.4 Retire `timelapse_reviewed.py`'s duplicate derivations** — station→camera
      number, coordinates, Spanish→Latin species, Santiago→UTC. `camtrap/observations.py`
      owns all four and documents itself as owning them ("Every consumer … reads this
      shape and nothing else"). That claim is false today; 2.3 and 2.4 make it true and
      delete a parser.

- [ ] **2.5 Registry dependency fixed** — see 1.6. Until then CT27 ingests coordinateless.

- [ ] **2.6 Delete `scripts/dedup_primavera_2025.py`.** Its whole premise dissolves:
      pv is no longer a separate campaign, and the "unmappable `100EK113` folder" it
      excludes was resolved into CT05 by the 2026-08-13 flatten — so those records stop
      being excluded, which is a data change, not just a deletion.

- [ ] **2.7 Rewrite `config.yaml`'s `camera_traps.campaigns`** to the three campaigns.

- [ ] **2.8 Reconcile.** DB row counts equal parquet row counts exactly, per campaign.
      Note `upsert_df` is `INSERT OR REPLACE` on `mediaID`/`observationID`, which are
      **UUIDs regenerated on every parse** — so re-running ingest against a populated
      table duplicates rather than replaces. Loading into the currently-empty DB is the
      only clean path.

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
| `data-pipeline/scripts/dedup_primavera_2025.py` | entire premise (see 2.6) |
| `camtrap/observations.py:70–79, 185` | `CAMPAIGN_ORDER` entry + both precedence comments (396 overlap / 31 conflicts) |
| `camtrap/observations.py:7–9` | per-campaign export quirks — `filePath` populated in primavera_2025, `timestamp` only there. A fresh export has different quirks |
| `Anual-reports/2025/py/01_data_prep.py:6, 71` | `REPORT_CAMPAIGNS` |
| `Anual-reports/2025/py/list_ciervo_guina_images.py:34–44, 122, 166` | reads pv's reviewed CSV and the `exports/Primavera-verano 2025-2026/species` thumbnails |
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
