# FMA Project Status

**Last updated:** 2026-08-24 — **the consumer boundary is closed.** `estaciones.csv` now owns
station identity and the platform's two station files are generated from it, with a test that
fails if either is hand-edited; this ends the class of defect that put CT26 19 km outside the
reserve and left CT27's 315 images without coordinates. Stations are spelled `CT01`..`CT27`
**everywhere** now — the platform's `TC-01` dialect is gone. The `ct_*` tables were dropped and
rebuilt from the canonical parquets: **35,807 rows across 74 deployments** (21 + 26 + 27
stations, the real deployment history), reconciling exactly against the published contract,
replacing 2,948 orphaned rows written by a parser deleted four days earlier and still carrying
`pv_2025_2026` as a phantom campaign. The platform now serves **2,522 real detections** where it
had been serving that wreckage. `weather_station` (264,943 rows) and `weather_forecast` are
committed as per-year Parquet, so **the Windows↔Linux question is permanently answered** — any
machine rebuilds the warehouse from the repo alone, verified by restoring into an empty database
and matching all 41 columns and the mean temperature to six decimals. The canonical contract is
finally read at both ends (`run_fetch.py --ct-check`), and it fingerprints the whole campaign
description rather than row counts, because the 815-row review repair moved verdicts without
moving `n_rows`. **Three of V2-REVIEW's own specifications were measured wrong and corrected:**
the key it mandates cannot be a primary key (`datetime` is null in 4,013 of 35,807 rows), its
column list omits `campaign`, and its "timestamps are UTC" is both false and undesirable for a
camera clock. **New debt:** `data-pipeline` has no test suite at all while now holding four
modules that guard this boundary — designed in `data-pipeline/docs/TEST-PLAN.md`, deferred.
226 camera-traps tests pass.

**Prior — 2026-08-19** (camera-traps: **the reviewer's verdict now reaches the canonical table, and `pv_2025_2026` was silently reverting the new review.** The primavera 2025 re-review finished, making all three campaigns comparable for the first time — and that comparison exposed the largest data defect of the V2 pass: **815 rows across the three campaigns were typed `animal` while the reviewer had written in `observationComments` that the frame holds no animal.** The review pass wrote its correction into free text while the typed column kept the classifier's guess, so every consumer counted the classifier. Primavera's animal count was overstated by 50.6% — 744 against 494 — and counted 10 people and 4 vehicles as animals. `resolve_review()` in `camtrap/observations.py` now owns the resolution and is fail-closed: it refused the otoño 2026 ingest outright until a `Pitio}` typo was fixed. **A second defect was worse because it was invisible:** `pv_2025_2026` is not a campaign but a second review pass over primavera, and while it sat in `CAMPAIGN_ORDER` it OUTRANKED primavera — so the moment primavera was re-ingested, `read_campaigns` returned **169** of its 744 rows, with 606 overlapping keys restoring April labels over the brand-new review. Anyone re-ingesting primavera without dropping pv in the same session would have gotten a quietly wrong table. **Also:** video is now excluded from every export by policy — otoño 2026 had been carrying 2,162 videos swept as `blank` with zero `animal`, while primavera excluded its 2,618 at source, making the two campaigns' denominators different in kind; the export gate now refuses video and cannot be overridden. Animal counts: otoño 2025 830→706, otoño 2026 1,785→1,320, primavera 744→494, and **zero rows are now `animal` with an empty species**. All three campaigns pass the export gate from the repo for the first time. 179 tests pass, up from 152. **Second pass the same day closed the row-set defect underneath all of this:** the canonical table described only reviewed rows, so a station that recorded no animal was absent from it entirely — seven station-campaigns were missing (CT23 in otoño 2025; CT01/CT06/CT17/CT22 in primavera, with 6, 21, 7 and 18 frames each; CT02/CT12 in otoño 2026). Absent is indistinguishable from never-deployed, which is fine as a detection numerator and wrong as a trap-effort denominator. The table now holds **one row per still**: 3,359 → **35,807 rows** and the station gap is **0 in all three campaigns**. **The annual report moved by exactly one record and the cause is named** — diffed at row level, 1 added and 0 removed, CT04 01130013.JPG *Oryctolagus cuniculus*, which is the `conejo?` adjudication, not the rebuild; the rebuild moved nothing because no swept-only row is ever typed `animal`, and a test asserts it. Felipe also settled the two deferred label questions (a comment that cannot name a species stays `unknown`; `conejo` and `pitío` were adjudicated as real animals and added to species.yaml), and `timestamps.py` no longer aborts before writing anything on a cp1252 console. **190 tests pass**, up from 152.)

**✅ 2026-08-20 — the stale-code sweep: the re-ingest finally reaches the documents.**
A full audit of the camera-trap chain (16,752 lines; report in
`SecondBrain/Reviews/review-state-camera-traps.md`, 17 findings) replaced the pattern of
discovering one pre-canonical file per session. **The root defect was two files both
claiming to be the reviewed truth:** `new_labeled_data_corrected.csv` typed every row
`animal`, including the 815 the reviewer had negated. It is deleted; pehuén reads the
canonical parquet. **pehuén was reading `pv_2025_2026` — a retired April review pass — as
its spring campaign and never read `primavera_2025` at all**, so its results changed:
spring Liebre 230 → 161, Zorro culpeo 59 → 82, totals 850 → 789 records. Every pehuén and
annual-report figure re-rendered from corrected input. **V2-REVIEW §2 is unblocked**
(2.4/2.6/2.7 closed): `timelapse_reviewed.py`, `camtrap_dp.py` and
`dedup_primavera_2025.py` are deleted and `--ct` raises `CameraTrapIngestNotRebuilt`
rather than silently ingesting pv. **New:** `camtrap/canonical_state.py` publishes
`data/CANONICAL_STATE.json` and both consumers verify it before reading — the gate that
was missing when the tables went 3,359 → 35,807 rows without one consumer erroring.
Schema version 2 adds `observation_comments` and `classification_probability`.
`campaign_dir` is now a required `--campaign-dir` argument instead of a config key that
had pointed at the first campaign for three campaigns. **36.0 MB of legacy data deleted.**
204 tests pass. **Later the same day:** pehuén's loader was rewritten onto
`observations.parquet` (it had been reading the retired `pv_2025_2026` as its spring
campaign and never reading `primavera_2025` — spring Liebre 230→161, Zorro culpeo 59→82),
and a second, structural defect was fixed with it: a load-time `filter(!is.na(datetime))`
imposed the clock rule on every consumer, so presence/absence — which needs a station, not
a clock — lost every camera whose clock failed. Puma showed 6 stations against 8 in the
data. `R/00_admissibility.R` now owns `admissible()`/`presence()`/`episodes()`; the spatial
maps also stopped counting IMAGES, which at ratios of 1.7×–4.9× had been reordering species
rather than rescaling them. The independence rule went from two implementations to one and
the two counts now agree at 327. **Field anchors:** Felipe confirmed a technician in
primavera's CT15 and CT16 first frames (2025-06-09, 33 min apart), which dates otoño 2025's
retrieval at those stations — 33 animal records recovered, report 642→651 records /
262→269 events. **CT03 was not faulty and is now recovered:** its whole "incoherence" was 3 frames within
32 seconds of midnight — a filename rollover, not a corrupt clock. `MIDNIGHT_TOLERANCE`
(120 s) added to the P2 test with Felipe's approval; +72 animal records, report 651→717
records / 269→323 events, all 54 new events at CT03. The tolerance only ever forgives, so
it cannot admit a corrupt clock. Also: `01_data_prep.py` adopted pehuén's last-retained
independence rule (it changed zero numbers — the divergence was latent), and
`scripts/verify_order.py` now checks the three manifest-less stations by reconstructing
DCIM folders from datetimes — all three consistent, method validated against primavera
CT14 (9 folders recovered, manifest says 9). **Those three were never "lost"**: all are
`clock_clean` at 100% `valid_date`, and the manifest gap is 3 stations, not 45. 

**📘 NEW — `camera-traps/docs/DATA-HEALTH-MANUAL.md`** (2026-08-20, 1,953 lines).
The end-to-end protocol with its justification: field procedure, storage, ingest gates, the
nine-class datetime error taxonomy, and what each analysis requires. Two tables in it are new
knowledge rather than assembly — the **recovery matrix** (§4E.8: error class x available
evidence -> what is restored; class 4, corrupt date registers, is unrecoverable with *any*
anchor) and the **admissibility matrix** (§6.10: analysis x validity axis x unit x minimum n;
every count-based row requires episodes, none is correct on images). Published as a private
artifact for the field team. Its §Part 9 is the audit reproduced below, and it is deliberately
part of the document: a manual that describes checks which are not running is worse than none.

**➡️ NEXT SESSION IS ON THE LINUX BOX (Felipe, 2026-08-20).** `main` is at `aecc64b`,
pushed; `git pull` first — four commits landed today. Plan: the `ct_*` rebuild (V2-REVIEW
§2), plus preparing a DuckDB copy to migrate to Windows next week.

**Do not migrate the whole database.** V2-REVIEW 2.1 splits it by regenerability and the
split matters here: `ct_deployments` / `ct_media` / `ct_observations` are **regenerable**
from `observations.parquet`, which is in git on both machines, and the Linux copy's `ct_*`
rows are keyed on **Timelapse GUIDs** — proven per-project, not per-image (primavera's two
`.ddb` files share 2,387 filenames and 0 `mediaID`s). Carrying them over imports rows that
the rebuild cannot match or replace, so they would be orphans. **Rebuild `ct_*` here;
migrate only what cannot be refetched:** `weather_station`, `weather_forecast`,
`literature` — CR800 pulls and open-meteo history are gone if lost. 2.2 asks for those as
committed parquet rather than a copied binary, which also makes the migration repeatable
instead of a one-time file move.

**Two things will bite on Linux.** `run_fetch.py --ct` now raises
`CameraTrapIngestNotRebuilt` by design — that is the retirement, not a break. And pehuén
still hardcodes absolute Windows paths (`R/01_load_data.R`, V2-REVIEW 1.11), so it cannot
run there until they are parameterised; it also needs `nanoparquet` (`Rscript
setup_packages.R`).

**⛔ STILL OPEN — re-audited 2026-08-20 against the working tree, not against
V2-REVIEW's own checkboxes. Fifteen items, and four are guarantees the new manual
describes that the code does not enforce.** Do not treat the chain as closed. Full text in
`camera-traps/docs/DATA-HEALTH-MANUAL.md` §Part 9.

> ## ✅ 2026-08-24 — eight of the fifteen are closed, and they are all of the consumer-side ones
>
> **Closed: A1, A2, A4, B1, B7, and one third of B8.** The prediction this audit made held
> exactly — every remaining defect sat at a boundary, and one session spent entirely at the
> consumer boundary cleared it. **What is left is the FIELD-RECORD side (A3, B4, B6),
> pehuén's Windows-side work (B2, B3), and Tier C.**
>
> - **A1** — `estaciones.csv` owns station identity; the other two are generated by
>   `camera-traps/setup/build_station_registry.py`, and a test asserts they equal a fresh
>   render. One canonical spelling everywhere now: `CT01`..`CT27`, not `TC-01`. `sd_card`
>   dropped. **The disagreement was smaller than stated below:** all three files already
>   agreed on every value they *shared*; the defect was one missing row and no check to keep
>   it from recurring. (Also: CT27 has **315** rows, not the 344 recorded below.)
> - **A2** — `data-pipeline/src/canonical_gate.py` + `ct_ingest_state`. `run_fetch.py
>   --ct-check` reports drift, writes nothing, exits 1. It reads the JSON as a *file* and does
>   not import `camtrap`, or the check would agree with itself by construction.
> - **A4** — `occupancy_pct` now counts from `ct_deployments`, not the registry. Per-campaign
>   filtering is still open.
> - **B1** — done: **35,807 rows, 74 deployments** (21+26+27), reconciling exactly against the
>   contract. Weather committed as per-year Parquet, so the Windows↔Linux question is
>   permanently answered: any machine can rebuild the warehouse from the repo alone.
> - **B7** — CT27 install `2025-12-11`, its clock cleared against the retrieval sequence.
>
> **Three of V2-REVIEW's own specifications turned out to be wrong** and are corrected in
> place there: 2.8's mandated key cannot be a primary key (`datetime` is null in 4,013 of
> 35,807 rows), 2.3's column list omits `campaign`, and 2.3's "timestamps are UTC" is both
> false and undesirable. Full detail in `CHANGELOG.md` 2026-08-24.
>
> **New debt from the same session:** `data-pipeline` has **no test suite at all** (1,642
> lines) while now carrying four modules whose guarantees rest on one manual run each.
> Designed and deferred: `data-pipeline/docs/TEST-PLAN.md`.

*Tier A — documented guarantees that are not enforced:*
- ~~**A1 · The station registry still disagrees with itself**~~ — **CLOSED 2026-08-24.**
  (V2-REVIEW 1.6). Was: `stations.yaml` **26** entries · `camera_trap_stations.geojson`
  **27** · `estaciones.csv` **27**, so **CT27's otoño 2026 images ingested with no
  coordinates**. Same class as the CT26 error that returned as a 19 km displacement.
- ~~**A2 · The canonical contract is published but nothing verifies it**~~ — **CLOSED
  2026-08-24.** (V2-REVIEW §4). Both halves now exist, and the gate fingerprints the whole
  campaign description rather than row counts alone.
- **A3 · The field form has no loader.** `setup/build_visit_template.py` writes
  `Registro de visitas CT.xlsx`; nothing reads it back. `setup/build_field_notes.py` is the
  one-time legacy migration, not this. A filled form is a spreadsheet that must be
  transcribed by hand, so the whole field protocol rests on an undocumented manual step.
  `camtrap.visit_schema.by_label()` is the entry point. **Never got a numbered V2 item —
  which is why it has been invisible.**
- ~~**A4 · Effort denominators wrong in the dashboard.**~~ — **CLOSED 2026-08-24.**
  `occupancy_pct` divided by `len(_TC_COORDS)` rather than the stations deployed. Now counted
  from `ct_deployments`, with `n_stations_deployed` returned so the denominator is visible.
  It was worse than recorded here: `/species-list` has **no campaign filter at all**, so it
  was a cross-campaign aggregate over an unrelated number. Per-campaign occupancy needs a
  filter in numerator and denominator both and is **still open**.

*Tier B — known defects, known fixes:*
- ~~**B1** the `ct_*` rebuild (V2-REVIEW 2.1/2.2/2.3/2.5/2.8)~~ — **CLOSED 2026-08-24.**
  35,807 rows, 74 deployments. Identity derives from `(campaign, camera_num, file_name)`,
  never from Timelapse GUIDs. `literature` dropped (0 rows, no reader).
- **B2** pehuén hardcodes absolute Windows paths (`R/01_load_data.R:107-108`). **Deliberately
  left** — Felipe does not want pehuén running on the Linux box, so this is Windows-side work
  rather than a blocker
- **B3** figures/tables not re-rendered. **Narrowed 2026-08-24:** the loader now emits
  `Primavera_2025`, so script 05's join is already un-broken. What remains is otoño 2026
  falling out of `05_spatial_distribution.R:249` and `02_detection_summary.R:83,110,150` —
  **plus a third cause now queued for the same figures**, station labels rendering `CT01`
  instead of `TC-01`. Render the three separately or the attribution is lost
- **B4** `field_notes.csv` audited for coordinates only; 57 of 106 rows carry `data_flags`
- **B5** `provenance.py` not re-run on the re-ingested primavera — data it has not seen
- **B6** DCIM manifest coverage not stated per campaign, including the stations that
  legitimately have none (V2-REVIEW 1.4)
- ~~**B7** CT27's install is datable from `CT 27.kml`~~ — **CLOSED 2026-08-24.** Install
  `2025-12-11`, recorded in `otono_2026/deployment_anchors.csv` as provenance, offset zero by
  construction. The waypoint's `15:52:56` is **UTC**: the camera's first frame reads 12:49:01
  and its last frame sits in correct order in the retrieval trip, so the clock is sound and
  the 3 h gap is the offset, not a fault. The KML itself is not in this repo and is not needed
- **B8** ~~three~~ **two** missing test fixtures (V2-REVIEW 1.10): manifest rebuild from a
  flatten log, size-matched deletion accounting. **Registry agreement is done.**
- **B9 · NEW — `data-pipeline` has no test suite.** 1,642 lines, zero tests, now carrying
  `canonical_ct`, `canonical_gate`, `recovery` and `_reconcile`, whose guarantees rest on one
  manual run each. Designed in `data-pipeline/docs/TEST-PLAN.md`; deferred by Felipe

*Tier C — housekeeping:* `data/campaigns/label_conflicts_primavera_vs_pv_2026-05-27.csv` and
`Anual-reports/2025/data/manual_review_ciervo_guina.md` still on disk (superseded);
stale pv comment at `Anual-reports/2025/py/apply_verdicts.py:143`; otoño 2025's video
existence on the NAS never confirmed; `count` empty in all campaigns; `06_seasonal_puma.png`
orphaned at 27 records against its >=30 threshold.

**Where the work actually is:** the chain from card to canonical table is enforced and
tested. The two ends — the field record going in (A3) and the consumers coming out (A1, A2,
A4, B1) — are where every remaining item sits. That is not a coincidence: they are the two
boundaries where the work crosses out of `camera-traps` into something else.

> **2026-08-24 — this prediction was tested and held.** The consumer boundary was taken as a
> single session's work and **every item on it closed** (A1, A2, A4, B1, B7). Nothing in the
> middle of the chain needed touching. What is left is the *other* boundary — the field
> record coming in (A3, B4, B6) — plus pehuén's Windows-side rendering and Tier C.
>
> The one thing the audit did not anticipate: closing the consumer boundary created a **new**
> instance of the same pattern one layer down. `data-pipeline` now owns four modules that
> guard the boundary, and **nothing guards them** (B9). A check nobody runs is a comment,
> which is the sentence this whole refactor was built around.

**🔄 IN PROGRESS — camera-traps `docs/V2-REVIEW.md`** (opened 2026-08-18; entry condition met and six items closed 2026-08-19). Felipe's call: **nothing new starts until the camera-trap chain is clean end to end.** Closed: **1.1** campaign set is exactly three (pv dropped from `CAMPAIGN_ORDER` and `REPORT_CAMPAIGNS`, kept as provenance), **1.2** the canonical file set is decided — `TimelapseData.ddb` and `TimelapseTemplate.tdb` are required and all three campaigns hold the full set, with all three templates verified functionally identical down to the `observationType` vocabulary the export gate depends on, **1.3** export gate passes for all three from the repo, **1.3b** the reviewer's verdict reaches `observation_type`, **1.12** both deferred label decisions, **1.13** the canonical table describes every still — and the station-count difference (21 / 26 / 27) is **resolved, not a gap**: the grid was built up over time, so each campaign covers as many stations as existed at its retrieval. **Updated 2026-08-24 — §2 is COMPLETE and §4 is shipped.** Also closed: **1.5** (CT27's anchor), **1.6** (one station registry), **2.1/2.2** (the regenerability split and the committed weather Parquet), **2.3** (the rebuild), **2.5**, **2.8** (reconciliation), and the **§4 gate** at both ends. Still open: **1.4**, **1.7**–**1.9**, **1.10** (two of three fixtures remain, and `data-pipeline` still has no suite at all), **1.11**, **1.14**. **Integration Status:** `In Progress [REMAINING: V2-REVIEW 1.4, 1.7–1.11, 1.14, §3 cleanup]`. **Blockers/Notes:** ~~the DuckDB `ct_*` rebuild is untouched~~ — done; the live database is current with the published contract, verify with `python run_fetch.py --ct-check`. **consumers now see ~16x more rows per campaign** — anything reading `observations.parquet` must filter on `observation_type`, and `01_data_prep.py` already does; pytest is absent from the `camera-traps` env, use `python -m unittest discover -s tests`; two file-set items stay deliberately open — the three `addaxai-*` files (primavera only, new with the AddaxAI update, role undecided) and the legacy `CamtrapDB_*` V1 project DBs, which are the last thing keeping the file set from being identical across campaigns but whose deletion is a data call, not cleanup.

**Prior — 2026-08-18** — toolbox (**the contact master and the merge output had diverged: the boss edited the pre-merge file while the merged copy sat unpromoted, and the pipeline could not be re-run without corrupting `N`.** Diagnosed first, in full: 141 of 141 original rows intact, **zero deletions, no column misalignment from his sort**; his changes were an A→Z sort by name, one new row (`Katherine / Educa Mac`, `N` blank), two previously-blank emails filled, and six packed `Nombre, Cargo` cells split — into which he put the **cargo**, not the organisation. **The blocking regression:** `_next_number` read the bottom row of the sheet, which after a name sort holds `N=129` on a list whose highest number is 141 — re-running the merge would have issued `N` 130–149 and **collided on twelve existing rows**. Now one past the maximum, order-independent. **New `curate_master.py` + `namesplit.split_cargo`:** the pipeline could only ever append; it can now correct what is already there. Job titles move out of `Organización` into `Notas` (not in the shared copy, and never deleted — it is the owner's text), and one person occupying two rows becomes one row holding both addresses. Same two-pass shape as the merge, and for the same reason: the rules can see a cell is wrong far more reliably than they can see what it should say. **`split_cargo` exercised against all 110 organisation values in the real list — 17 detections, zero false positives**; `SBAP- Depto Fondo e IECB` (a unit, not a role) and `librería naturaleza, editores` (a business) are left alone by design, and the genuinely ambiguous shape — `Directora | Educación MIM` versus `Coordinador | Centros UC` — comes back at `media` instead of being guessed. **Three duplicate pairs the sort made visible, all pre-existing:** Enrique Rivera (`N` 40/134, *identical* address), Cristián Becker (`@mnhn.gob.cl` / `@mnhn.cl`), Paulina Stowhas (`N` 50/140 — MinRel → Fondo Naturaleza, i.e. a job change, so the former ministry became a note reading `antes: …` rather than vanishing). **Run and verified:** 142 → 139 → **159 rows**; `N` 1–162 with no duplicates and gaps only at 50/54/134 (the merged-away rows); the owner's yellow highlight and its note survived three row deletions; **0 unexpected changes to any other row**; 0 duplicate addresses or names remaining; share copy byte-consistent. **Fixed en route:** the append wrote the literal string `nan` into `Notas` for every row whose review-sheet note was empty. **Still open and not ours to decide:** 15→0 cargos, but `UC - Glaciares` is still not an organisation's name; `Katherine` has no surname; the six `?` acronym expansions from 13 Aug are unconfirmed; **the three leaked credentials still need rotating.**)

**Prior — 2026-08-14** (camera-traps: **attribution becomes the third flatten precondition, and the Linux box turns out to be more capable than these docs assumed.** `setup/flatten_for_camtrapdp.py` now **refuses — always, with no override flag** — to flatten a deployment containing a station-shaped subfolder, checked after discovery and **before a single file moves**, under `--dry-run` too. This closes the gap `TC23_M20.2`-inside-`TC22_M19.2` exposed on 2026-08-13: 2,460 files that would have been attributed to camera 22 at camera 22's coordinates, with `moved=2460 renamed=0 lost=0` and every existing check passing. Fatal-always was Felipe's explicit call — no arrangement puts one station folder legitimately inside another, so there is nothing to override. New **`camtrap/stations.names_a_station()`** owns what a station folder looks like, matched by **shape** rather than by membership in `station_aliases.csv`, because `100EK113` **is** an alias row (an unrenamed SD-card folder = primavera_2025's camera 5) and a membership test would call every DCIM folder a station and refuse every flatten there has ever been; shape excludes it without knowing what a DCF folder is, which stays owned by `clocks.dcim_folder_key`. It returns `bool` so it cannot become a second name→camera route. A fixture reads the alias file and asserts every spelling is recognised **except** `100EK113` — the 2026-08-13 hand-check ("34 TC-style rows, 0 disagreements") now re-runs on every commit. **104 tests pass** (was 96); verified end-to-end on scratch trees, with a clean tree flattening unchanged. **Two long-open chores closed:** the **Informe Anual 2025 v2 DOCX is rendered** (1.4 MB, figures embedded — open since 2026-05-20, it only ever needed pandoc on Linux), and **`figures/` is mirrored to `figures_pre_reingest/`**, satisfying the re-ingest precondition all three docs repeat. **Machine audit — the important part for planning:** this box has an **RTX 4070 (8 GB) with AddaxAI and `md_v5a.0.0.pt` already installed, so MegaDetector can run on Linux**; only the **Timelapse2 sweep** still needs Windows (.NET). And the ingest chain — `anchor_candidates.py` → `propose_anchors.py` → `timestamps.py` — **never opens an image**, it reads CSVs, the DCIM manifest and the MegaDetector JSON. So **otoño 2025's ingest is blocked here by exactly one missing file:** `ImageData_total.csv` (8,997 rows), which passed the gate on 2026-08-13 but lives only on the Windows box. Copy that one CSV and the whole chain runs on Linux. The campaign images themselves are **not** on this machine — the Synology `CAMPAÑAS DE RECOLECCION/{Otoño 2025, Primavera 2025}` folders exist but are empty. **Later the same day, after Felipe reviewed the gates:** he objected that each gate quotes the incident that produced it, reading as a point-fix rather than a rule. Audited all six — five are derived from a stated premise and name the incident only as a regression witness; **`names_a_station` was the exception**, enumerating the three spellings we have used. The sharper finding: **the pipeline already saw TC23's 2,460 alien frames** — `establish_order` reports them as unparseable filenames, verified by running it — but filed them under *ordering*, where a failure does not condemn a camera, so they kept camera 22's identity anyway. The missing gate was the right **question** asked of evidence we already had. New **`camtrap/provenance.py`** owns the general rule, *one deployment, one capture story*: two filename shapes each forming their own counter run is a second camera. It enumerates nothing, so `Camara 23` is caught as readily as `TC23_M20.2`; **validated before being wired in — 28,178 files across all four campaigns, 0 false positives**, with the one measured false positive (our own `101EK113_` rename prefixes) folded into the rule rather than tuned away. Imports nothing from `clocks`, stdlib-only. Also **the top-level station check is now fatal by default** — it warned unless `--check-stations`, leaving the weaker guard on the failure that cost 252 rows of camera 5, while its sibling refused outright; `--check-stations` is now accepted and ignored so old command lines still run. **120 tests pass** (was 104). **Horario-de-invierno: DECIDED — no DST correction, ever** (implementation pending, none started). My first recommendation (correct otoño 2026, flag the rest) was **overturned by Felipe and withdrawn**: it optimised for agreement with Chilean civil time, which the animals do not use. An **unadjusted** camera clock has a **constant** offset from UTC — one number, exactly removable; an *adjusted* one has a **piecewise** offset stepping at whatever dates the technicians visited, so adjusting for DST destroys a recoverable constant. Otoño 2026 is internally consistent on UTC−3 throughout, and Felipe's decisive fact — **the clock has only ever been adjusted once**, at the May 2026 retrieval — makes the older campaigns constant-offset rather than ambiguous. 🛑 **`camera_datetime_observed` is empty on all 26 rows**: the schema already had the column that would have settled this, and what was recorded is the *conclusion* (`shifted, -1.0`), not the *observation* — witness vs navigational evidence for the third time, and the technician cannot tell the cases apart anyway because their phone auto-adjusts. **Measured:** day length across the otoño 2026 deployment swings **4.92 h** (14.35 → 9.86 h at −39.4417); sunrise moves ~2.5 h, i.e. **~2.5× the DST hour**, on every campaign and station. **Pending:** (1) **field protocol** — record both clocks as raw readings, never a correction; stop adjusting clocks, set once to a fixed offset (no code, and the only item that expires); (2) **schema** — store the instant + fixed offset, derive civil and sun-anchored time (design gate); (3) **sun-anchored sensitivity run in pehuen** — read-only, clock time vs double anchoring (Vazquez et al. 2019 / `activity`), design gate opened then stopped. ⚠️ Confirm first whether **R is installed on the Linux box**. ⚠️ Sun-anchoring **will move Dhat4** and five of ten published pairs straddle the 0.75 Monterroso threshold. **Also still open, all doable from Linux:** the `primavera_2025` × `pv_2025_2026` merge model (both parquets are local; a prerequisite for the primavera re-ingest), and pehuen's hardcoded `C:/Users/USUARIO/...` paths. Pending commit + push.)

**Prior — 2026-08-13** — later session (camera-traps: **the last campaign is flattened, and `pv_2025_2026` turns out not to be a campaign at all.** The primavera-verano download reviewed and flattened — 19,522 files, 26 stations, 13,814 moved, 1,935 renamed (all CT14), **0 lost**; `dcim_manifest.csv` (13,814 rows) staged in `data/campaigns/primavera_2025/`, with CT02, CT08, CT11 and CT14 all earning `ORDER_MANIFEST`. **Two findings outrank the flatten.** (1) 🛑 **A whole station was nested inside another** — `TC23_M20.2`, 2,460 files, sat inside `TC22_M19.2`, and flattening would have attributed every one of them to camera 22 at camera 22's coordinates. **Nothing in the pipeline would have caught it:** the two cameras use different filename schemes (`IMAG####` vs `MMDDnnnn`), so there were zero collisions — the run prints `moved=2460 renamed=0 lost=0` and the conservation check passes. The already-ingested `pv_2025_2026` parquet proves the nesting is *new* (`TC23_M20.2` appears there as a top-level `rel_path` prefix; camera 22 has 0 rows, never reviewed). **Conservation and ordering are checked; station attribution is not** — a precondition for that is the next design-gate item. (2) **`pv_2025_2026` is not a campaign.** The field record has exactly three transitions — `otono_2025` → `primavera_2025` → `otono_2026` — and campaigns are named for the season they are **retrieved** in, so the campaign opened 2025-05-14/06-11 and closed 2025-11-12/2026-01-14 is `primavera_2025`, whose window the download's span matches exactly. `pv_2025_2026` is a **second Timelapse2 review pass** over the same cards (396 shared camera+filename keys). This sharpens the 2026-07-30 note: they are not consecutive campaigns to dedup by precedence but **two readings of one campaign**, while `CAMPAIGN_ORDER` encodes them as sequential. **Verified:** the deployment window from 2026-08-12 holds on data it was never written from — every station with a working clock has its frame span *inside* its field-record window, often to the day (CT04 opened 05-14/closed 11-21, frames 0514..1121; CT06 05-14/11-13, 0514..1113; CT20 06-09/12-03, 0609..1203; CT15's first frame *is* its install date). **Otoño 2025's export now passes the gate** (`full_category_sweep`, 8,997 rows — animal 818, human 478, vehicle 99, blank 7,602), so its ingest is unblocked. Unicode: nothing to do (39,173 paths, 0 non-ASCII, 0 NFD). **No code changed; 96 tests still pass.** **Notes for the sweeps:** ~9 primavera_2025 stations show clock resets detectable from the field window alone (CT03, CT05, CT08, CT14, CT17, CT23, CT24, CT26 carry January frames in a May/June deployment — filename-MMDD evidence, preliminary, not a verdict); **CT16's clock is impossible, not merely wrong** (`00300001.JPG` month 00, `16300071.JPG` month 16 — corrupt RTC, no anchor repairs it); eight otoño 2025 images cannot be decoded (six 0-byte in CT04, two ~4.6 MB all-zero in CT13) and **Felipe's call is to leave them `blank`** against my recommendation of `unknown` — 8 rows of 8,997, no figure moves, recorded as an accepted limitation; a scanner reproducing MegaDetector's error log found the same 8 and **0 in primavera_2025**. **Next: (1) otoño 2025 ingest — unblocked; (2) MegaDetector + sweep on primavera_2025 (Felipe); (3) the nested-station precondition, design gate; (4) the horario-de-invierno shift, design gate — deferred a second time, it keeps losing to campaign work.** ⚠️ **Both `primavera_2025` and `pv_2025_2026` are in `REPORT_CAMPAIGNS`** — re-ingesting primavera_2025 at full size (26 stations vs the parquet's 14, 19,522 files vs 1,960 observations) will move the 2025 report substantially more than otoño 2025 will; mirror `Anual-reports/2025/figures/` before either. Pending commit + push.)

**Prior — 2026-08-13** (toolbox: **new `toolbox/` consolidates recurring operational scripts that belong to no single project**, replacing the ad-hoc `Envio correos/` and `Transforma MOV a MP4/` folders — one conda env (`toolbox`), one README indexing every script, participant data confined to a gitignored `data/`. **Security finding on migration: three plaintext credentials were committed and pushed** — `convocatorias@` and `felipe.guarda@` (fundacionmaradentro.cl) plus `xdelavega@minciencia.gob.cl`, which is not ours. Files moved out of the index, but the secrets remain in history and **still need rotating**; nothing here fixes that. **Delivered:** `lib/rosters.py` (contact identity — column/header-row detection, address normalisation, `extract_emails` for multi-address cells), `lib/namesplit.py` (splitting "Nombre - Organización" free text, and how much to trust each split), `lib/master_list.py` (the canonical contact list's structure — column meanings, `N` continuation, autofilter growth, formatting preservation), `lib/mailer.py` (bulk mail; **dry-run by default** and a send ledger, replacing scripts that mailed every row on execution), plus `merge_contacts.py`, `excel_crosscheck.py`, `send_campaign.py`, `video_to_mp4.py`. **Contact merge run on the real master:** 141 → 161 rows from two event files; 33 new addresses found, **8 flagged as people already on the list under a different address** (personal vs institutional — e.g. `ctala@mma.gob.cl` vs `chariftala@gmail.com`), which address matching alone cannot see. Master restructured 4 → 9 columns with a generated 5-column `*_COMPARTIR.xlsx` for circulation; the original is never written to. **Four bugs found and fixed during the work:** header detection accepted banner rows and silently reported *zero* changes (worst possible failure mode — the counts looked right); difflib alone missed `Natasha Pons` / `Natasha Pons Majmut` at 0.77, so token-subset matching was added for Spanish double surnames; org-only rows with a blank `Nombre` vanished from the share copy and would have been re-added as new next time; `N`'s formulas read back empty after openpyxl saves, so they are frozen to their values during restructure. **Design decisions:** two-pass merge (`--review` → human corrects → `--apply`) because the sources pack name+organisation in six different shapes and some cannot be split by any rule — proposing beats guessing; `Origen`/`Fecha` are CLI flags, never inferred, written only on appended rows; appended rows copy font/border/alignment but **not fill**, so a highlight is never invented on a row nobody marked. **Next:** review `revision.xlsx` (8 probable duplicates default to `NO`, 2 low-confidence splits), then re-run `--apply` and replace the canonical file; build the 6-question standardised form so the review pass becomes a formality; rotate the three credentials. `send_campaign.py` has **never actually delivered mail** — rendering and dry runs only. Pending commit + push.)

**Prior — 2026-08-12** — later session (camera-traps: **otoño 2025 re-downloaded WITH its SD-card subfolders and flattened; the manifest's ordering claim tightened.** 8,997 files, 8,969 moved, 28 renamed, **0 lost**; `dcim_manifest.csv` staged in the campaign folder. Flattening it exposed a defect: the manifest's claim is *frames in folder A precede those in folder B because the camera fills folders in name order*, which has **two** preconditions, and only one was enforced. `establish_order` already refused a partially-described deployment; nothing checked that each group **is** a camera-created folder. So `flatten` recorded the whole intermediate path, and CT04's **723 loose frames under `M5`** — sitting beside `M5/100EK113` and `M5/101EK113` — sorted **first**, asserting its January frames preceded the October ones. Sorted on, that is a backwards step in capture order, i.e. **a phantom clock reset on 2,097 frames**. New `clocks.dcim_folder_key()` keeps only a DCF-shaped last component and runs at **both ends** — `flatten` on write, `timestamps.load_manifest` on read — so a manifest already written is corrected **by being read**, which is necessary because flattening consumes the tree it describes and otoño 2025's manifest cannot be regenerated. **The rule holds whether or not a camera can leave files beside its own DCIM folders**, so it rests on no firmware assumption. Effect: CT14 and CT20 keep full `ORDER_MANIFEST` (100/101/102EK113, MMDD strictly sequential); CT04 is refused fail-closed; 18 stations with one constant folder (`M7`) that had been claiming `ORDER_MANIFEST` while the manifest contributed nothing now correctly report `ORDER_COUNTER`. Otoño 2026 unchanged. Also: rename prefixes use the DCIM folder alone so a grid folder's name cannot leak into a filename (otoño 2025 would have produced `M 11_101EK113_...`), unsafe characters stripped; stale `person`/`vehicle` help text fixed. **96 tests pass** (was 81). **Findings for the sweep:** the download holds **no video at all** (8,997 files, all `.JPG`) with 19–25% counter gaps at most stations and 0% at three — the 3-stills-plus-1-video pattern with videos absent from this download, *not* lost stills, so confirm they exist on the NAS; CT04's loose frames are a third DCIM folder someone flattened by hand (own counter run from 1, disjoint later MMDD range, zero shared filenames). **Next session (2026-08-13): (1) the horario-de-invierno shift — deferred deliberately, needs a design gate; (2) otoño 2025 ingest once MegaDetector + the full-category sweep land — Felipe left MegaDetector running overnight; (3) primavera/pv likely ready too, so re-verify every fix end-to-end across all campaigns and watch for anything new.** ⚠️ **otoño 2025 is in `REPORT_CAMPAIGNS` — mirror `Anual-reports/2025/figures/` before re-ingest; the 2025 report's numbers will move.** Pending commit + push.)

**Prior — 2026-08-12** (camera-traps: **the field record is now a first-class pipeline input, and otoño 2026 is fully ingested for the first time.** Two prior days' uncommitted work landed first: (a) **the export gate was using the wrong vocabulary** — `observationType` is Camtrap DP's controlled list, which the Timelapse2 template emits verbatim as `human`/`blank`, while the gate had invented `person`/`empty`. Otoño 2026's first properly swept export passed **only because `vehicle` is spelled the same in both**; its 584 `human` rows counted as neither assigned nor proof, so the same campaign without vehicle frames would have been rejected as unswept *after* the sweep was done. An unrecognised value is now a hard rejection (`unrecognised_category_values`) rather than an advisory note. (b) **`data/campaigns/field_notes.csv`** — 106 visits, 27 stations, migrated once from `Registro de monitoreo CT.xlsx` by `setup/build_field_notes.py`, with all three of the workbook's date conventions reconciled and every inference recorded in `data_flags` (57 of 106 rows flagged). **Today's work:** new **`camtrap/anchors.py`** owns what the field record asserts about a clock, split into two assertions with different preconditions — the **deployment window** (every station, needs no photo) and an **anchor** (only where the clock failed, needs a datable frame). **The window is the headline change:** it previously came from the anchors, which exist only where a clock already broke, so **26 of otoño 2026's 27 stations had no window at all** and a FORWARD jump — a clock set ahead, which keeps every capture delta positive — was undetectable for them. Verified: with the field-notes window, 26 stations gain one and **not a single verdict changes**, so those clean verdicts are now corroborated rather than merely unchallenged. A visit-derived window uses a **3-day tolerance** (vs the anchors' 1 h) because a notebook date has day precision and the visit spans days; the bound is **measured, not guessed** — across the 20 stations provably coherent from capture order alone, the largest excursion past a recorded visit date is +1.67 d. **Two design rules came out of real failures caught during the build:** (1) *a visit is not an anchor* — CT01's notebook says 2025-11-24 → 2026-05-13 while its frames run 2025-11-26 → 2026-05-14 across one coherent segment, so forcing the visit date on would apply a two-day offset to a clock that was never wrong; anchors are proposed only where the segment would otherwise be refused, and a clean camera gets `NOT_NEEDED`. (2) *witness vs navigational evidence* — the first proposer paired CT18 segment 0 with `11190001.JPG` for a **−5 day offset on ten frames whose clock was correct**, because a counter-`0001` frame is the first file on a card, not a photo of the technician; only a labelled `human`/`vehicle` frame can date a visit. New `propose_anchors.py` writes `anchor_proposals.csv` and **promotes nothing automatically**; a segment it cannot pair becomes an explicit `unrepairable_pending` row, because a station missing from the anchor file and one known to be unanchorable look identical downstream. Also: `visit_date_only` added as an APPROXIMATE anchor type (all 27 opening visits are date-only, so asserting an hour nobody recorded is exactly how CT18's install anchor came to claim `14:00:00`); `anchor_candidates.py` now ranks the **swept export above MegaDetector** (584 `human_labelled` + 25 `vehicle_labelled`, and `person_detection` drops to **zero** — the sweep confirmed every MD person hit); CT27 is reported as **unverified clean** rather than clean, since with no install record its in-window test never ran. **81 tests pass** (was 59). **otoño 2026 ingested:** 1,785 rows, 26 clean stations, CT18 refused on all five segments — reproducing the 2026-08-03 hand analysis exactly, now mechanically. **Next:** the other three campaigns still need Timelapse2 sweeps — nothing else can move them. Then pehuen (contaminated by CT18) and its hardcoded Windows paths. **Open decision for Felipe: the Mayo 2026 horario-de-invierno shift.** Every camera was set back 1 h at that visit and Chile left summer time 2026-04-04, so otoño 2026 frames between those dates read **1 h ahead of local time** — a ~40-day systematic time-of-day error that no reset detector can see, because an hour never breaks coherence. Pending commit + push.)

**Prior — 2026-08-03** (camera-traps: **the segment-aware repair is wired into ingest and the export gate is enforced — handoff steps 2–5 done.** Felipe settled the open gate rule: **gate on `person`/`vehicle`, with an override**, on the grounds that under the new field protocol a person frame should always exist, so its absence is worth flagging. **Delivered:** (1) `timestamps.py` rewired onto `clocks.repair_plan()` — per-segment offsets, `classify_epochs` deleted, hard-fail without a valid all-images export, and an audit log that prints every segment with its verdict, its anchor and `unaccounted_days`. New `clocks.segment_for_rows()` places every row (videos and unparseable stamps included) in its segment, or in none — in which case the row is refused, never guessed; a one-segment camera claims all its rows, since a camera that never reset has no split to attribute and videos are the majority of rows at some stations. (2) `camtrap/exports.py` owns the two Timelapse2 exports and the gate. The rule cannot be mere presence-of-categories because **`unclassified` doubles as `empty`** in our template, so `{animal, unclassified}` — today's otoño 2026 file — *looks* labelled while nothing was assigned; that verdict (`categories_never_assigned`) is **not overridable**, while a genuinely person-free sweep is admitted by a signed `export_gate_override.txt` (verified_by / date / reason — a file, not a flag, so the decision carries a name and travels with the data). Enforced at three points: ingest, `python -m camtrap.exports <csv>` for an immediate check while Timelapse2 is still open, and `flatten_for_camtrapdp.py --check-export`. **Confirmed rejecting the real otoño 2026 export.** (3) New `anchor_candidates.py` + `camtrap/detections.py` join the MegaDetector JSON to the total export and list every person/vehicle detection, counter-`0001` frame and segment boundary, tagged with the segment it sits in and whether that segment still needs an anchor. **Finding: MegaDetector already detected 595 person + 28 vehicle frames in otoño 2026** that the Timelapse2 sweep never recorded — 17 stations have an install-side candidate and 7 a retrieval-side one, so the sweep is confirmation work, not search. For CT18 the frame to inspect is `11190001.JPG` (camera-time 2025-11-19 06:41, counter 0001), the only candidate inside segment 0. (4) `valid_effort` added to `CANONICAL_COLUMNS` (station-level) and optional `segment_index` to the anchor CSV; the corrected CSV now carries 7 added columns. (5) **New finding: `CT_02` and `CT_12` were missing from `station_aliases.csv`** — 23 images, no animal records, so they never appeared in the animal-only export and were invisible until the all-images export was read; consistent with the 2026-06-17 note that both produced no animal triggers. **59 fixtures pass**, 34 new (`tests/test_exports.py`, `tests/test_timestamps.py`, plus row-assignment cases in `test_clocks.py`); the regression test that matters asserts two segments of one camera get two *different* offsets. **Validated end-to-end** on a scratch copy of otoño 2026 with `unclassified` relabelled to `empty`: CT18 reproduces the hand analysis exactly (5 segments 10/32/40/3/227, all refused, install anchor inside no segment, `valid_effort=FALSE`) and the canonical parquet shows CT18 as the only station out of the effort denominator. **Next:** nothing further is possible in code until the Timelapse2 sweeps exist — all four campaigns are still rejected by the gate. Then handoff steps 6–7: re-diagnose all four campaigns (mirror `figures/` first) and fix pehuen, which additionally needs its hardcoded `C:/Users/USUARIO/...` paths parameterised before it can run on Linux. Pending commit + push.)

**Prior — 2026-07-31** (camera-traps: **flatten now preserves capture order; `camtrap/clocks.py` implements the segment-aware repair rule.** Felipe resolved the two blockers from 2026-07-30: (1) **the Synology originals are untouched**, so every campaign except otoño 2026 can be re-downloaded with its DCIM subfolders intact — the ordering evidence is *not* lost; otoño 2026 was flattened before upload and has **no backup**; (2) the hard-fail export gate stays hard ("getting everything is more important than getting results now"). That put the flatten fix on the critical path, because re-downloading with the old script would destroy the evidence a second time. **Delivered:** `setup/flatten_for_camtrapdp.py` writes a `dcim_manifest.csv` sidecar (deployment, dcim_folder, original/flat name, size, mtime, action) — no renames, so existing `file_name` joins are unaffected — appended per deployment so an interrupted run still describes its moves, and including already-flat files so a *partial* manifest is visible instead of silently misleading. It **no longer treats same-name/same-size as a duplicate to skip**, which is precisely the signature of a reset-clock camera re-emitting `0101xxxx` names, and a conservation check now aborts the run if a deployment ends up short. `--dry-run` predicts renames correctly instead of reporting zero. **New `camtrap/clocks.py`** owns segments, order evidence (`dcim_manifest+counter` > `counter` > `none`), coherence, and the rule *a segment is repairable iff coherent AND containing ≥1 anchor*; adds `valid_effort` as a **station-level** flag (a camera dead at an unknown date leaves the effort *denominator*, not just the numerator); detects splits from capture order **or** the deployment window, which is what makes a forward jump visible where `year < 2024` was blind. Key design call: **failing the ordering precondition does not condemn a camera** — an in-window sequence whose filenames agree with their own stamps demonstrably never reset, which is what keeps otoño 2026's five flattened wrap cameras usable. **25 fixtures** (`tests/test_clocks.py`, stdlib unittest so it runs on both machines) cover scenarios A–G, both preconditions, forward jumps, video exclusion, ambiguous anchors and the partial manifest. **Validated on real data:** CT_18 otoño 2026 returns 5 segments (10/32/40/3/227) matching the hand analysis, with incoherence localised to segment 4 and the install anchor correctly rejected as falling inside no segment. Also removed the stale nested `camera-traps/.git` (April 2026, no remote) that made every `git` call inside that folder report the July work as untracked; its history is bundled in `~/Dev/_archive/` together with three files that existed nowhere else. **Next:** `timestamps.py` still uses `classify_epochs` and one offset per station — handoff steps 2–5. **Open:** the export gate's exact rule, since `unclassified` doubles as `empty` and would let today's category-less otoño 2026 export pass. Pending commit + push.)

**Prior — 2026-07-30** — later session (camera-traps: **CRITICAL — clock-reset diagnosis was structurally blind; CT18 contaminated the pehuen analysis.** Our protocol for deciding whether a broken camera clock can be repaired diagnosed resets from the **animal-only** CSV. Otoño 2026's all-images export (12,068 rows vs 1,785) shows **CT18 reset its clock 4 times, not once** — so the shipped `last_real_proxy` single offset (+3329 d) stamps segments 1–3 as starting 2026-02-12 when they actually occurred in Nov/Dec 2025, and 44.8 d is unaccounted. The anchor's own note ("2 real-time photos before reset") was itself written from the blind view — the camera ran correctly for 9.4 days / 10 photos. **CT18's install anchor is uncorroborated:** it asserts 2025-11-14 offset-zero, but the first image on the card is an animal at 2025-11-19 06:41, while every other camera's first frame is a midday `unclassified` install photo of the technician — invisible because nobody labels people. **Spill: annual report is CLEAN** (`REPORT_CAMPAIGNS` excludes otoño 2026), but **pehuen has 65 focal-species records with fabricated dates** (jabalí 33, culpeo 16, liebre 12, perro 2, puma 2 — exactly 1 trustworthy); `record_table`'s `valid_time_of_day` filter protected activity/overlap, but `valid_date` was never consumed, so `02_detection_summary` and `06_seasonal_detection_maps` are contaminated. **Blind spot covers all four campaigns** — only otoño 2026 has an all-images export, so every existing `unrepairable_pending` note (otoño 2025 CT15/CT16/CT19, primavera/pv CT16) is a lower bound from animal-only data. **Spec agreed** (Felipe rejected both of my first two criteria): two preconditions — ordering established, segment coherent — one rule — *a segment is repairable iff coherent AND containing ≥1 anchor* — and three independent flags `valid_date` / `valid_time_of_day` / **new `valid_effort`**. Full-category export (empty/animal/person/vehicle) becomes a hard ingest gate. **No code written**; complete handoff in `camera-traps/docs/HANDOFF-clock-repair.md`. **Blockers:** (1) DCIM-subfolder question — the filename counter is per-folder and wraps at 999, `RelativePath` keeps only the deployment name, and `flatten_for_camtrapdp.py:resolve_dest` discards the subfolder except on filename collision, so capture order may be unrecoverable for the five cameras with >999 images; (2) Felipe's field notebook for CT18's install/visit dates and whether the older campaigns have install photos at all. **Supersedes the next entry's claim** that CT15/CT16/CT19 need no field notebook — the filename-MMDD trick works only for year-only errors and must be verified per camera. Pending commit + push.)

**Prior — 2026-07-30** (camera-traps: **canonical observation table implemented; annual report rewired.** New `camtrap/` boundary package — `stations.py` owns the canonical station convention `CT01`–`CT27` (historical spellings in `data/campaigns/station_aliases.csv`, data not code) and `observations.py` owns the canonical table, emitted by `timestamps.py` as `observations.parquet` beside the existing `_corrected.csv`. `01_data_prep.py` now reads it via `read_campaigns()`; ~190 lines of duplicated clock repair, station parsing and species recovery deleted. **Report numbers changed: 419 → 369 events** — cross-campaign dedup removed 325 double-counted images (primavera_2025 is almost entirely superseded by pv_2025_2026, which re-reviewed the same SD cards), and 143 otoño-2025 CT15/CT16/CT19 records are now excluded because timestamps.py refuses the offset guess the old code made. A near-miss caught in the process: naive dedup would have kept the *earlier* campaign's labels and silently reverted 31 adjudicated species, including demoting a puma — precedence is now explicit in `CAMPAIGN_ORDER` and every label change is printed. Old figures preserved in `figures_pre_canonical/`. **Next:** add real anchors for CT15/CT16/CT19 to recover those 143 records (camera filenames encode MMDD, so no field notebook is strictly required); migrate pehuen + data-pipeline to `observations.parquet` and retire `_corrected.csv`. Pending commit + push.)

**Prior — 2026-07-29** (camera-traps: **architecture review — canonical observation schema agreed, implementation pending.** Added `## DESIGN_NOTES` to `camera-traps/README.md` recording external file-format leakage as the project's dominant decay risk (Timelapse2 CSV + MegaDetector JSON decoded independently at 6+ call sites). **Critical:** `Anual-reports/2025/py/01_data_prep.py` still reads `_reviewed.csv` and re-implements clock repair with different rules than `timestamps.py` — the `_corrected.csv` contract from 2026-06-25 is not honoured by the annual report, so report and pehuen can derive different timestamps from the same raw data. **Decision:** replace per-consumer normalisation with a canonical observation table (`camtrap/observations.py` → `observations.parquet`, keyed on `campaign, camera_num, file_name`) written once at ingest, so downstream repos need no shared code — just `read_parquet`. Canonical station ID `CT01`–`CT27`, enforced by a validation gate at the Synology folder level in `setup/flatten_for_camtrapdp.py`; the `_M##.2` grid suffix moves to a station registry (it is many-to-one — grid M15.2 holds cameras 11 and 18). Legacy station spellings go in `data/campaigns/station_aliases.csv` — data, not code. Also confirmed `100EK113` (Primavera 2025) is **camera 5**, an unrenamed SD-card folder duplicating `pv_2025_2026 / TC5_M9.2` (14 files, timestamps identical to the second); its 252 rows are dropped as unmappable today and no detections are lost, but only because the same images arrive via pv. **No code written this session** — fixing the report will change its figures, so it needs a figure diff. Pending commit + push.)

**Prior — 2026-06-17** (camera-traps: **Otoño 2026 campaign integrated** — May 2026 SD pull reviewed, 1785 observations, 25 deployments (CT_02 and CT_12 produced no animal triggers; timelapse parser is observation-centric so they're correctly absent from `ct_deployments`). Vaca payoff confirmed: 579 rows tagged Vaca (top species in campaign) that would have been mislabeled Caballo without yesterday's species addition. Added Quique (*Galictis cuja*) to `data-pipeline/species.yaml` with CLIP English prompt — 5 obs in this campaign, native mustelid, first project record. CSV staged at `camera-traps/data/campaigns/otono_2026/` and registered in `data-pipeline/config.yaml`. Zero overlap with otono_2025 / primavera_2025 / pv_2025_2026 — no dedup script needed. **Pending:** CT_18 timestamps — 135 rows currently dated 2017-01-01 (camera clock reverted to factory). Real deployment-start anchor pending from field notebook; do **not** run `--ct` on the Linux box until those are corrected (see comment in `config.yaml`). Pending commit + push.)
**Owner:** Felipe Guarda — Fundación Mar Adentro
**Field site:** Bosque Pehuén, La Araucanía, Chile — reserve center -39.4417°, -71.7420° (canonical: `plataforma-territorial/data/stations.yaml` → `reserve.center`)

---

## 🏁 Milestone — First full code review complete (2026-05-11)

The independent code review of the three actively-developed projects (`data-pipeline`, `camera-traps`, `plataforma-territorial`) — started 2026-04-21 — is finished. **Every finding is closed or explicitly deferred with a re-open trigger.** No silently-open work remains.

- **Resolved:** 1 Critical · all Warnings · most Suggestions
- **Closed-rejected (with rationale):** S57, S47, S14, S72
- **Deferred (with re-open conditions):** S58 (field verification), S76 (Vitest tests, until CI exists)

Full narrative: `CHANGELOG.md` (top entry) + `~/Documents/Obsidian FG/SecondBrain/Sessions/2026-05-11-fma-ecosystem-code-review-tier-1-and-tier-2.md`. Per-project snapshots: `~/Documents/Obsidian FG/SecondBrain/Reviews/review-state-*.md`.

---

## Two-Machine Architecture

| Machine | Role | Projects |
|---|---|---|
| **Personal laptop** (PopOS Linux) | Code, DuckDB, data pipeline, React frontend + FastAPI backend | `data-pipeline/`, `plataforma-territorial/`, `literatura-agent/`, `visualizaciones-artisticas/` |
| **Office desktop** (Windows → future Linux) | GPU-dependent raw analysis: MegaDetector, CLIP, Timelapse2 review | `camera-traps/`, `plataforma-territorial/` Phase 3 only (camera tab) |

**Rule:** Platform repo is shared, but Phase 3 camera tasks = office, everything else = home. Always commit before switching machines.

**Handoff protocol:**
- Office → Home: commit reviewed CSV → pull at home → run ingestor
- Home → Office: commit platform code → pull before starting Phase 3 work

### ⚠️ Pending: Migration to Linux at office

The personal laptop (Linux) will NOT be left on permanently — it's a personal machine. This means:
- **DuckDB (`fma_data.duckdb`), data-pipeline service, and plataforma backend must migrate to the office machine once it switches to Linux.**
- Until then, the platform can only be used from the laptop itself or when it's manually on and the service running.
- `bootstrap_windows_db.py` is a temporary workaround for Windows office — it will not be needed post-migration.

**Migration checklist (when office switches to Linux):**
- [ ] Copy `fma_data.duckdb` to office Linux machine
- [ ] Set up conda environment (`data-pipeline`, `plataforma-territorial`)
- [ ] Enable `fma-pipeline.service` and `fma-platform.service` as systemd user services
- [ ] Update Tailscale or local network access so CR800 fetcher still reaches the weather station
- [ ] Update `literatura-agent` cron to run on office machine
- [ ] Verify `schedule-agent` still works (Google API credentials)

---

## Resolved Decision: React/Vite is canonical (2026-05-12)

React/Vite + FastAPI is the definitive platform stack. Streamlit references in `plataforma-territorial/README.md` are stale and should be cleaned up next time the README is touched.

---

## Dependency Chain

```
data-pipeline (writes DuckDB)
    ↓
plataforma-territorial/backend (reads DuckDB, serves API)
    ↓
plataforma-territorial/plataforma-demo (React frontend, consumes API)

camera-traps (produces reviewed CSVs)
    ↓
data-pipeline (ingests CSVs into ct_* tables)

visualizaciones-artisticas (reads DuckDB for art generation)
```

---

## Project Status

### 1. Data Pipeline (`data-pipeline/`) — OPERATIVO

Running as systemd service (`fma-pipeline.service`). Full pipeline with real data flowing.

**Canonical catalogs (2026-04-27, updated 2026-06-17):** `species.yaml` (33 entries — 29 CLIP + 4 reviewer-discovered non-CLIP + invasive/priority flags) is the single source of truth across the ecosystem. Sibling loaders in camera-traps and plataforma-territorial/backend read this same file. Pairs with `plataforma-territorial/data/stations.yaml` (also now consumed end-to-end after Track B).

**Code review Batch A+B (2026-04-27):** 9 warnings resolved — dead deps removed (W9), `_STATE_PATH` centralised to `src/paths.py` (W15), `cr800_session()` context manager (W18), Open-Meteo/CR800 fault isolation in `run_once()` (W19), `open_meteo.py` DST-safe `tz_localize` (W16), `_process_raw` made public (W12), algorithmic DST dates in `recover_dst_gaps.py` (W13), silent `count=1` default fixed (W14), `toa5.py` column-drop now logged (W17). **2026-05-06:** S12 (`recover_dst_gaps.py` moved to `scripts/`, path resolution updated), S13 (`run_watcher.py` connection cleanup in `finally` block). Remaining: C1 + W8 (Batch E, needs Opus).

**Live data:** 264,944 rows weather_station · 168 rows weather_forecast · 7,652 rows ct_observations · 20,095 ct_media · 106 ct_deployments (Otoño 2025 + Primavera-verano 2025-2026 ingested 2026-04-15)

| Component | Status | Notes |
|---|---|---|
| DuckDB schema (6 tables) | Done | `fma_data.duckdb` ~42 MB |
| Open-Meteo fetcher | Done | Hourly, 16-day forecast (extended 2026-04-21) |
| Camera trap legacy parser | Done | Parses Timelapse2 CSV |
| Camtrap DP parser | Done | Awaiting test with real DP package |
| TOA5 parser (CR800 backfill) | Done | Column names fixed 2026-04-07 (RH_Avg, WindDir_Avg, incomingSW_Avg) |
| CR800 live fetcher | Done | Working via Tailscale VPN |
| File watcher daemon | Done | Monitors `data/incoming/` (not activated) |
| APScheduler daemon | Done | Open-Meteo hourly, CR800 weekly |
| systemd user service | **Done** | Enabled, starts on boot |

**Pending:**
- [ ] Tabla `literatura` pendiente de poblar (literatura-agent integration)
- [ ] Camtrap DP parser: test with real data
- [ ] Watcher de carpeta incoming: activate

---

### 2. Plataforma Territorial (`plataforma-territorial/`) — EN PROGRESO

React/Vite frontend with 4 pages. FastAPI backend operational with real endpoints.
**Access Linux:** `plataforma` alias → `http://localhost:8000` (systemd service).
**Access Windows:** `conda run -n plataforma-territorial uvicorn backend.main:app --port 8000`

**Two-machine data note:** DuckDB lives on Linux (written by data-pipeline service). On Windows, run `python bootstrap_windows_db.py` from `plataforma-territorial/` to seed a local DB with Open-Meteo data (90-day archive + 7-day forecast). Enough for Meteo and Riesgo tabs. No Tailscale needed.

**dist/ sync fix (2026-03-31):** Removed `dist` from `.gitignore`. Built frontend is now committed to git. Both machines get the same compiled UI via `git pull` — no per-machine rebuild needed.

**Code review (2026-04-21 → 2026-05-07):** Full review of Blocks 3-5 complete; artifacts in `~/Documents/Obsidian FG/SecondBrain/Reviews/`. Track B closed the station-registry + species-catalog cross-project chains (W11/W23/W32/W33/W47/W51 map-center half). **Track C closed (2026-04-29):** W41 App.jsx 1805→37 line decomposition into 24 modules. New endpoints: `/api/config/geography`, `/api/config/species`. Track A (CR800 backfill safety) still queued. **2026-05-06:** S52, S54, S60, S61, S74. **2026-05-07 (early):** S67 + W44 chart slice via `src/styles/chart.js`. **2026-05-07 (Track K — same day):** W44 fully closed via CSS Modules pass — 246 → 25 inline `style={{}}` sites (90% reduction); 19 new `.module.css` files; `src/styles/vars.css` exposes 13 `--color-*` palette properties. Same change closed S65 (Card + SectionLabel accept className), S77 (Asistente :hover), S78 (Reportes @media print). Build 1.91s, CSS 36.5 kB / 10.8 kB gzipped.

| Component | Status | Notes |
|---|---|---|
| Observatorio (map) | **Real data** | 23 canonical stations from DuckDB, species counts + thumbnails in popups; piso vegetacional overlay (2026-06-02) — 48 polygons by BIOTOPO, off by default |
| Dashboard — Meteo tab | **Real data** | Year of history, variable selector, wind rose, comparison mode |
| Dashboard — Fire risk tab | **Real data** | All visuals use Open-Meteo exclusively. Polar chart color matches gauge. Fixed 3-week bar chart (no navigation). Wind compass from forecast. Freshness timestamps on all widgets. |
| Dashboard — Cameras tab | **Real data** | Diel activity chart, summary stats, station grid — all from DuckDB |
| Dashboard — Fauna tab | **Real data** | Species bar chart + stats + priority/invasive alerts — all real |
| Asistente (AI chat) | Mock data | Placeholder responses |
| Reportes (newsletter) | Mock data | Draft generator with typing animation |
| FastAPI backend | **Working** | Serves API + built frontend from single port 8000 |
| Deployment | **Done** | Linux: `fma-platform` systemd + `plataforma`/`plataforma-stop` aliases |
| Station coordinates | Done | `data/stations.yaml` + GeoJSON files |
| BP boundary polygon | Done | **Under review — confirm delimitation** |

**FastAPI endpoints live:**
- `GET /api/weather/current`, `/history`, `/forecast`
- `GET /api/fire-risk/current`, `/forecast`, `/history?days=`
- `GET /api/detections/recent`, `/species-summary`, `/stations`, `/station-summary`, `/station-images/{id}`
- `GET /ct-images/<campaign>/stations/<id>/<file>` (static mount)
- `GET /api/health`

**Priority 1 — Connect frontend to real endpoints:**
- [x] Replace mock data in fire risk tab with real API calls ← done 2026-03-30
- [x] Observatorio map stations from real DuckDB data ← done 2026-04-15
- [x] Cámaras trampa dashboard tab — diel activity, summary stats, station grid ← done 2026-04-17
- [x] Fauna tab: real stats + priority/invasive species alerts ← done 2026-04-17
- [x] Resize thumbnails (Pillow 1000px in export_best_images.py) + lightbox in Observatorio popups ← done 2026-04-16 (commit 22f6a08)
- [ ] Cámaras tab Phase 3.4 extensions: species×station heatmap, image gallery
- [ ] Retrain `fire_model.pkl` with current scikit-learn (pickle incompatible → ml_probability returns null)
- [ ] Include ML index alongside rule-based index in fire risk view
- [x] Extended Open-Meteo forecast from 7 to 16 days — bar chart next week now populated ← done 2026-04-21

**Priority 2 — Asistente with real Claude API:**
- [ ] Connect Asistente tab to Claude API (Sonnet + tool use)
- [ ] Implement DuckDB query tools: current risk, recent detections, trends
- [ ] Each response with calculated values must cite its formula and input data (methodological transparency)

**Priority 3 — Observatorio: real map layers:**
- [x] Verify real coordinates for all camera stations and weather station ← confirmed 2026-05-12
- [x] Piso vegetacional (fotointerpretación) — 48 BIOTOPO polygons, toggle layer ← done 2026-06-02
- [ ] BP boundary delimitation — final polygon version pending (carried from Open Items)
- [ ] DISTRITO (geomorphological) layer — same source GeoJSON, 4-color physiographic classification; queued
- ~~Optional layers: fire risk zones, historical fire perimeters~~ — dropped 2026-05-12 (out of scope)

---

### 3. Camera Traps (`camera-traps/`) — FASE 1 OPERATIVA · Informe Anual 2025 publicado y corregido

CLIP classification pipeline and Streamlit review UI are production-quality. Four campaigns reviewed: Otoño 2025, Primavera 2025, Primavera-verano 2025-2026, and Otoño 2026 (latest). Species list sourced from canonical `data-pipeline/species.yaml` via sibling loader (Track B, 2026-04-27) — 29 CLIP species + 4 non-CLIP entries (33 total).

**Last Updated:** 2026-08-14
**What Changed:** Attribution joined conservation and ordering as a flatten precondition, enforced twice over. `stations.names_a_station()` recognises a station *folder* by name — precise, and it can say which folder to move. New `camtrap/provenance.py` recognises a second camera by its *frames* and enumerates nothing, so an intruder folder called anything at all is caught; validated at 0 false positives over 28,178 real files before being wired in. The top-level station check also became fatal by default. 120 tests pass. Also rendered the Informe Anual 2025 v2 DOCX and mirrored `figures/` → `figures_pre_reingest/`. Prior session: primavera_2025 flattened, `pv_2025_2026` identified as a review pass (2026-08-13).
**Integration Status:** Pending [Timelapse2 sweeps — Windows only] · otoño 2025 needs one CSV copied. The Timelapse2 sweep is the sole Windows-bound step: **MegaDetector runs on this Linux box** (RTX 4070 + AddaxAI + `md_v5a.0.0.pt`), and the ingest chain reads CSVs without ever opening an image. Otoño 2025 passed the gate on 2026-08-13, so copying its `ImageData_total.csv` (8,997 rows) off the Windows machine unblocks `anchor_candidates.py` → `propose_anchors.py` → `timestamps.py` entirely on Linux.
**Prior What Changed (2026-08-03):** Clock repair is now segment-aware end to end. `timestamps.py` consumes `camtrap/clocks.py` and applies one offset **per segment** instead of one per station; the full-category export gate (`camtrap/exports.py`) is enforced at ingest, at export time (`python -m camtrap.exports`) and at flatten; `anchor_candidates.py` produces the short list of frames that could become clock anchors. `valid_effort` joined the canonical schema as a station-level flag. 59 fixtures pass. Prior sessions: `camtrap/` boundary package + canonical `observations.parquet` (2026-07-30, report 419 → 369 events), DCIM manifest at flatten time + `clocks.py` (2026-07-31).
**Prior Integration Status (2026-08-03):** Pending [Timelapse2 full-category sweeps]. The pipeline is complete and validated on a scratch copy, but **all four campaigns are rejected by the export gate** — none has an all-images export with `person`/`vehicle` assigned, so no campaign can be re-ingested and DuckDB ingestion (`python run_fetch.py --ct`) stays held. For otoño 2026 the evidence already exists: MegaDetector found 595 person frames the sweep never recorded.
**Blockers/Notes (2026-08-14):** (0a) **Horario-de-invierno is decided: no DST correction, ever** — the pipeline should store the *instant* plus each deployment's fixed UTC offset, not naive civil time, and the analysis should run in sun-anchored time. Three implementation pieces pending, none started; the **field-protocol change is the only one that expires** (record both clocks as raw readings; stop adjusting camera clocks). See the header entry. (0) **The one cheap unblock:** copy otoño 2025's `ImageData_total.csv` (8,997 rows, gate-passing) off the Windows box into `data/campaigns/otono_2025/` — the ingest chain never opens an image, so that single file is all that stands between Linux and a full otoño 2025 ingest. The campaign images are not synced here (`CAMPAÑAS DE RECOLECCION/{Otoño 2025, Primavera 2025}` are empty). (0b) **MegaDetector is no longer Windows-bound** — RTX 4070 + AddaxAI + `md_v5a.0.0.pt` are installed on the Linux box; only Timelapse2 (.NET) still requires Windows. Older, from 2026-08-03: (1) The Timelapse2 sweep is the only thing on the critical path. (2) CT18's install date + any maintenance visit still needed from the field notebook — now the difference between recovering segment 0's 10 frames and losing them; inspect `11190001.JPG`. (3) Do the older campaigns have install photos? Decides whether otoño 2025's 143 dropped records are recoverable. (4) pehuen's R scripts hardcode `C:/Users/USUARIO/...` (`R/01_load_data.R:50–58`) so step 7 cannot run on Linux until parameterised. (5) Re-flatten every campaign with the new script before re-ingesting — the DCIM manifest only exists for runs made after 2026-07-31. Older, still open: re-map `100EK113` to CT5 (confirmed by photo, not re-ingested); run `render.sh` on Linux for the Informe Anual 2025 `.docx`; decide whether `source_code_CT_2025/` goes to git or only Drive.

| Component | Status | Notes |
|---|---|---|
| MegaDetector integration | Done | Via AddaxAI on Windows desktop |
| CLIP classification | Done | `run_classification.py` — CSV-only workflow, no DB dependency |
| Streamlit review UI | Done | `phase1_labeling/app.py` — handles empty filePath column |
| GIS data (KML → GeoJSON) | Done | Boundary + 26 station coordinates (TC-26 fixed 2026-03-30) |
| Otoño 2025 classification | Done | 697 animal obs reviewed |
| Primavera-verano 2025-2026 | Done | 500 animal obs reviewed |
| Otoño 2026 (May 2026 pull) | Reviewed; ingest blocked by export gate | 1785 obs / 27 deployments (CT_02 + CT_12 have no animal records — alias rows added 2026-08-03). Needs a full-category Timelapse2 sweep before re-ingest; CT18's 5 clock segments are all refused pending an anchor. |
| Species image export | Done | `export_best_images.py`: auto-discovers campaigns; 155 species images + 103 station images in `exports/` (gitignored); filenames traceable to source |
| **Informe Anual 2025** | **Done (v2)** | `Anual-reports/2025/` — markdown source, 6 figuras, pipeline reproducible, revisión visual aplicada (2026-06-02). **369 eventos** tras la tabla canónica (2026-07-30; antes 419), 11 especies, 22/26 CTs con detecciones. Las cifras pueden moverse otra vez cuando se re-ingesten las campañas — figuras previas en `figures_pre_canonical/`. |
| EfficientNetV2 fine-tuning | Planned | Needs ≥50 reviewed images/species — now viable for common species |
| Otoño 2025 videos | Deferred | 2,593 videos, MegaDetector not run — process post-migration on Linux |

**Pre-migration GPU work: COMPLETE.** Both campaigns reviewed and exported. Ready for OS migration.

**Post-migration next steps (Linux):** Ingest reviewed CSVs into DuckDB (Phase 3.1 in plataforma-territorial); station thumbnails in `exports/*/stations/` are ready for `plataforma-territorial/data/thumbnails/`.

Note: `config.yaml` and `NEXT_SESSION.md` have Windows paths — intentional (raw analysis runs on Windows desktop).

---

### 4. Literatura Agent (`literatura-agent/`) — DEPLOYED

Weekly cron script. Fetches from arXiv, OpenAlex, SciELO, Semantic Scholar (PubMed and CORE removed 2026-05-05). Claude Haiku scores each paper 1–5 for FMA relevance and drops scores < 3 before summarizing. Sends HTML email in Spanish.

**Last updated:** 2026-05-06 — `.gitignore` added; `papers_dump.csv` untracked; Semantic Scholar API key confirmed in `.env`.

**Pending:** Gmail app password in `.env` · end-to-end dry-run (`python run.py`).

---

### 5. Schedule Agent (`schedule-agent/`) — DEPLOYED

Monday scheduling: reads Google Tasks → Claude generates weekly plan → creates Google Calendar events → Flask approval UI.

**Status:** No action needed.

---

### 6. Visualizaciones Artísticas (`visualizaciones-artisticas/`) — EN ESPERA

Generative art from field data. Requires real DuckDB data. Volumetric bird songs visualization is complete.

- [ ] **Retrato Diario:** generative daily portrait of territory (risk + weather + detected species)
- [ ] **Constelación de Especies:** circular star map by activity time, distance by rarity
- [ ] **Río de Sonidos:** bird song visualization (requires audio files — see Acoustic Devices below)
- [ ] **Año Térmico:** circular calendar of annual temperature and risk

---

### 7. Dispositivos Acústicos — SIN CÓDIGO (datos aún no recuperados)

FMA has acoustic monitoring devices deployed in the field. Audio files not yet downloaded.

**Fase 1 — Recuperación e ingesta:**
- [ ] Download recordings from physical devices
- [ ] Define folder structure and naming convention (device, date, time)
- [ ] Add audio ingestor to data-pipeline: folder watcher → `acoustic` table in DuckDB (metadata only: device, timestamp, duration, file path)

**Fase 2 — Análisis de audio:**
- [ ] Species identification by vocalization — primary option: **BirdNET** (Cornell Lab, open source)
- [ ] Output: acoustic detections with species, confidence, timestamp → `acoustic_detections` table
- [ ] Integrate acoustic detections into pipeline alongside camera trap detections

**Fase 3 — Platform integration:**
- [ ] New "Acústica" tab in Dashboard or expand Fauna tab
- [ ] Acoustic device markers on Observatorio map
- [ ] Camera trap vs acoustic comparison for same species

Note: `visualizaciones-artisticas/` has the "Río de Sonidos" concept already designed, plus a reference project in `Volumetric bird songs/`. Audio files from this project feed those visualizations directly.

---

## Open Items

- [x] **TC-26 coordinates** — grid 22, SD M23. Spreadsheet has wrong coords (30 km off). Get correct GPS from field team.
- [ ] **BP boundary delimitation** — polygon under review. Confirm which version to use.
- [x] **Otoño 2025 camera trap processing** — done. Both campaigns reviewed, backed up, species images exported.
- [x] **Informe Anual 2025 — revisión visual ciervo/güiña aplicada (2026-06-02).** Ciervo rojo 7 → 1 cámara, Güiña 7 → 6 cámaras. Pipeline ampliado (`apply_verdicts.py`), informe re-escrito con sec. 1.6 documentando el protocolo. Ver `camera-traps/Anual-reports/2025/data/correction_log.txt`.
- [ ] **100EK113 → CT5 re-mapping.** Sospecha confirmada visualmente (sesión 2026-05-27 + verdicts 2026-06-02) pero aún no re-ingestado en DuckDB ni en los CSVs de campaña.
- [ ] **Otoño 2026 — CT_18 timestamp fix (2026-06-17).** 135 rows con DateTime 2017-01-01 (reloj de la cámara reseteado a fábrica). Cuaderno de campo tiene la fecha real de despliegue; cuando esté a mano, re-timestampear y correr `python run_fetch.py --ct` en Linux. Mientras tanto, **no ingestar** Otoño 2026 (config.yaml ya lo deja anotado).
- [x] **Render del DOCX del Informe Anual 2025 v2 (2026-08-14).** `bash camera-traps/Anual-reports/2025/render.sh` con pandoc 3.1.3 → `informe_anual_2025.docx`, 1.4 MB, figuras embebidas. Ojo: las cifras del informe se moverán con el re-ingest, así que este DOCX corresponde a la tabla canónica de 2026-07-30 (369 eventos). `figures/` respaldado en `figures_pre_reingest/`.
- [ ] **Subir bundle `source_code_CT_2025/` a Drive de FMA** (2026-06-03). Zip de 2.7 MB autocontenido, listo para entregar al colega. Decidir antes si se commitea o sólo Drive + borrar local.
- [ ] **Flora plot coordinates** — not yet available.
- [ ] **Aves en BP/** — contains bird list comparison notebooks and Excel files. No README. Appears to be taxonomic reference data for camera trap species list. Document before using in platform.
- [x] **Meteo tab** — label fixed to "Última medición", wind rose moved below charts (larger).
- [x] **Comparison mode** — implemented. Two-period comparison with stacked charts, side-by-side wind roses, dual stats table.
- [x] **Fire risk backend** — `fire_risk.py` ported to FastAPI with real DuckDB data.
- [x] **Fire risk frontend** — tab connected to real API; polar plot, gauge, compass, 3-week bar chart with today indicator.

---

## File Map

```
/home/fguarda/Dev/Python/                ← git repo root
├── PROJECT_STATUS.md                    ← THIS FILE
├── GIT_WORKFLOW_GUIDE.md                ← git workflow reference
├── fma_data.duckdb                      ← central database (~42 MB)
│
├── camera-traps/                        ← image analysis pipeline (Windows + Linux)
│   ├── README.md
│   └── GIS/                             ← source KML/Excel files
│
├── data-pipeline/                       ← DuckDB ingestion service
│   ├── README.md
│   ├── BUILD_CONTEXT.md                 ← original build spec (reference)
│   ├── config.yaml
│   ├── schema.sql
│   ├── run_fetch.py                     ← scheduler/CLI entry point
│   ├── run_watcher.py                   ← file watcher entry point
│   └── src/
│
├── plataforma-territorial/              ← React platform + FastAPI backend
│   ├── README.md
│   ├── data/                            ← GeoJSON + stations.yaml
│   ├── backend/                         ← FastAPI (weather, fire risk, detections)
│   └── plataforma-demo/                 ← React/Vite app
│
├── literatura-agent/                    ← weekly paper summarizer (deployed)
├── schedule-agent/                      ← Monday scheduler (deployed)
└── visualizaciones-artisticas/          ← generative art pieces
```
