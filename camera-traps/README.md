# Camera Traps — Species Recognition Pipeline

Automated species identification pipeline for camera-trap deployments at Fundación Maradentro (Reserva Costera Valdiviana and associated sites). Combines MegaDetector animal detection with CLIP zero-shot classification and a Streamlit human-review interface.

---

## ⚠️ Read this before re-ingesting a campaign

Three standing hazards. Each one is silent — nothing in the pipeline will stop you.

### 1. Re-ingesting will move the 2025 annual report's numbers — mirror `figures/` first

`REPORT_CAMPAIGNS` in `Anual-reports/2025/py/01_data_prep.py:71` is
`("otono_2025", "primavera_2025", "pv_2025_2026")` — **three of the four campaigns**.
Any re-ingest of any of them changes the published report.

```bash
cp -r Anual-reports/2025/figures Anual-reports/2025/figures_pre_<date>
```

The 2026-07-30 re-ingest already moved it **419 → 369 events**, and
`figures_pre_canonical/` exists because of it. The next one is bigger:

| Campaign | Current `observations.parquet` | After re-ingest | Why |
|---|---|---|---|
| `primavera_2025` | 1,960 obs, **14 stations** | 19,522 files, **26 stations** | The 2026-08-13 download is the full campaign; the existing parquet is a partial ingest |
| `otono_2025` | — | 8,997 files | Re-downloaded 2026-08-12 with DCIM subfolders |
| `pv_2025_2026` | 792 obs, 21 stations | see §2 | Not a campaign |

Do the diff deliberately. A number that moves for a known reason is a correction;
the same number moving unnoticed is a defect.

### 2. `pv_2025_2026` is not a campaign — it is a second review pass

The field record (`data/campaigns/field_notes.csv`) has exactly three transitions:
`otono_2025` → `primavera_2025` → `otono_2026`. **Campaigns are named for the season
they are *retrieved* in**, so the deployment that ran May 2025 → Jan 2026 is
`primavera_2025`. `pv_2025_2026` appears nowhere in it — it is a second Timelapse2 pass
over the same SD cards (396 shared `(camera, file_name)` keys; see
`label_conflicts_primavera_vs_pv_2026-05-27.csv`).

`CAMPAIGN_ORDER` in `camtrap/observations.py:75` currently treats them as **consecutive
campaigns**, deduplicating by precedence. That is the wrong shape for two readings of
one deployment — but pv holds *adjudicated* labels (a naive dedup once nearly demoted a
puma), so it cannot simply be dropped. **Unresolved.** Decide it before re-ingesting
either one.

### 3. CT16's clock is corrupt, not offset — no anchor can repair it

Primavera 2025 CT16 emits filenames `00300001.JPG` (**month 00**) and `16300071.JPG`
(**month 16**). Those are not wrong dates, they are *impossible* ones: the camera's RTC
is producing invalid values, not a consistent shift.

This is almost certainly the explanation for the **chronic TC-16 problem recorded
across campaigns** in the notes below. An anchor repairs a clock that is wrong by a
fixed amount; it cannot repair one that is not a clock. Expect CT16 to be refused, and
treat any past repair of it as suspect.

Related, from the same download — **~9 stations carry January frames in a deployment
opened in May/June** (CT03, CT05, CT08, CT14, CT17, CT23, CT24, CT26), i.e. clock
resets detectable from the deployment window alone. Filename-MMDD evidence, preliminary
until the sweep lands.

### And one gap the pipeline used to leave — closed 2026-08-14

Flatten verified that files were **conserved** and that they were **ordered**. Nothing
verified they were **attributed** to the right camera. Primavera 2025 arrived with one
station's 2,460 files inside another station's folder, and every existing check passed.

**Attribution is now the third precondition**, alongside conservation and ordering, and
it is enforced twice over — see [Step 1b](#step-1b--flatten-folder-structure):

| | Recognises | Catches | Misses |
|---|---|---|---|
| `stations.names_a_station()` | a station **folder**, by name | `TC23_M20.2` — and says *which folder to move* | a folder called `Camara 23` |
| `provenance.multiple_capture_stories()` | a second camera, by its **frames** | any intruder, whatever the folder is called | nothing name-shaped; it enumerates nothing |

Both are fatal, neither is overridable. The general one was validated across all four
campaigns before it was wired in: **28,178 files, 0 false positives.**

**The lesson worth keeping.** The pipeline *already saw* TC23's 2,460 alien frames —
`establish_order` reported them as unparseable filenames — but filed them under
*ordering*, where a failure does not condemn a camera. The gate that was missing was
not a new observation; it was the right **question** asked of an observation we already
had. When a check fires for a reason that surprises you, ask whether it is answering
the question you think it is.

---

## Status

**Last Updated:** 2026-08-14 — **attribution becomes the third flatten precondition, and the Linux box is confirmed as more capable than the docs assumed.** `flatten_for_camtrapdp.py` now refuses — always, no override flag — to flatten a deployment containing a station-shaped subfolder, closing the gap `TC23_M20.2`-inside-`TC22_M19.2` exposed: conservation and ordering were checked, attribution never was. `camtrap/stations.names_a_station()` owns what a station folder looks like, by **shape** rather than by membership in `station_aliases.csv` — the alias table contains `100EK113`, so a membership test would call every DCIM folder a station and refuse every flatten there has ever been. A fixture asserts the shape rule accepts every alias spelling *except* that one, so the 2026-08-13 hand-check ("34 TC-style rows, 0 disagreements") now re-runs on every commit. **104 tests pass** (was 96). Verified end-to-end on scratch trees: the TC23 arrangement is refused with a `2 file(s) would be attributed to CT22` message; a clean tree flattens unchanged, collision renames included. Also done: **the Informe Anual 2025 v2 DOCX is finally rendered** (`render.sh`, pandoc 3.1.3 — open since May, it only ever needed pandoc), and `figures/` is mirrored to `figures_pre_reingest/` ahead of the re-ingest. **Machine audit:** this box has an RTX 4070 (8 GB) with AddaxAI and `md_v5a.0.0.pt` already installed, so **MegaDetector can run on Linux** — only the Timelapse2 sweep still needs Windows. The whole ingest chain (`anchor_candidates.py` → `propose_anchors.py` → `timestamps.py`) reads CSVs and never opens an image, so **otoño 2025's ingest is blocked here by exactly one missing file**: `ImageData_total.csv` (8,997 rows), which exists only on the Windows box. The campaign images are not on this machine — the Synology folders `CAMPAÑAS DE RECOLECCION/{Otoño 2025, Primavera 2025}` are present but empty.
**Same day, after review:** Felipe pushed back on the shape of these gates — each one quotes the incident that produced it, which makes them read as point-fixes rather than rules. Audited all six: five are derived from a stated premise (`clocks` P1/P2, `dcim_folder_key`, the export gate's unknown-value rejection, `establish_order`'s partial-manifest refusal, `resolve()`), and **one was not** — `names_a_station` enumerates the three spellings we have used. Worse: the pipeline **already saw** TC23's 2,460 alien frames and filed them under *ordering*, where a failure does not condemn a camera. The evidence was never missing, only misfiled. New **`camtrap/provenance.py`** owns the general rule — *one deployment, one capture story*: two filename shapes each forming their own counter run is what a second camera looks like, and it enumerates nothing, so a folder called `Camara 23` is caught as readily as `TC23_M20.2`. Validated **before** being wired in: **28,178 files across all four campaigns, 0 false positives**, with the one measured false positive (our own `101EK113_` rename prefixes) folded into the rule rather than tuned away. It imports nothing from `clocks` and is stdlib-only — shapes are grammar-agnostic, which the design did not anticipate and which is the stronger position. Also: **the top-level station check is now fatal by default** (it was a warning unless `--check-stations`, leaving the weaker guard on the failure that cost 252 rows of camera 5; `--check-stations` is now accepted and ignored so old command lines still run). **120 tests pass** (was 104).
**Prior (2026-08-13):** **all four downloads are now flattened and two campaigns are gate-ready.** Primavera 2025 re-downloaded and flattened (19,522 files, 26 stations, 13,814 moved, 1,935 renamed, **0 lost**); `dcim_manifest.csv` staged, with CT02/CT08/CT11/CT14 earning `ORDER_MANIFEST`. Otoño 2025's export **passes the gate** (`full_category_sweep`, 8,997 rows — animal 818, human 478, vehicle 99, blank 7,602), so its ingest is unblocked for the first time. Two findings outranked the flatten and are recorded in the ⚠️ block above: **a whole station (`TC23_M20.2`, 2,460 files) was nested inside another** and would have been attributed to camera 22 with every existing check passing — the pipeline verifies conservation and ordering but never *attribution*; and **`pv_2025_2026` is not a campaign** but a second review pass over Primavera 2025, which the field record settles outright. Also verified: the deployment window built on 2026-08-12 **holds on 26 stations it was never written against** — every working-clock station's frames fall inside its field-record window, often to the day. No code changed; **96 tests pass**.
**Prior (2026-08-03):** the segment-aware repair is now wired into ingest: `timestamps.py` consumes `camtrap/clocks.py`, the full-category export gate is enforced, and `anchor_candidates.py` finds the anchors
**What Changed:** Handoff steps 2–5 are done, so the 2026-07-31 verdicts now reach the data instead of only the analysis. (1) **`timestamps.py` rewired** — it diagnoses every clock from `ImageData_total.csv`, applies a **separate offset per segment** via `clocks.repair_plan()`, and `classify_epochs` (the `year < 2024` test that applied one offset per station) is deleted. New `clocks.segment_for_rows()` maps every row — videos and unparseable stamps included — to its segment, or to none, in which case the row is refused rather than guessed. (2) **The export gate is enforced** (`camtrap/exports.py`): ingest refuses any export where neither `person` nor `vehicle` appears, because `unclassified` doubles as `empty` in our template and a `{animal, unclassified}` file therefore *looks* labelled while nothing was assigned. That verdict cannot be overridden; a genuinely person-free campaign is admitted by a signed `export_gate_override.txt`. Three enforcement points: ingest, `python -m camtrap.exports <csv>` for an immediate check at export time, and `flatten_for_camtrapdp.py --check-export`. **Today's otoño 2026 export is rejected** — verified. (3) **`anchor_candidates.py`** (new) joins the MegaDetector JSON to the total export and lists every person/vehicle detection, counter-`0001` frame and segment boundary with the segment it sits in. On otoño 2026 it finds **595 person + 28 vehicle frames** that MegaDetector already detected and the Timelapse2 sweep never recorded — 17 stations have an install-side candidate, 7 a retrieval-side one. (4) **Schema** — `valid_effort` added to `CANONICAL_COLUMNS` (station-level: FALSE leaves the effort denominator, not just the numerator) and optional `segment_index` to the anchor CSV. The corrected CSV now carries 7 new columns, adding `valid_effort` and `clock_segment`. (5) **`station_aliases.csv` gained `CT_02` and `CT_12`** — 23 images across two deployments that have no animal records, so they never appeared in the animal-only export and were invisible until the all-images export was read. 59 fixtures pass (`python3 -m unittest discover -s tests`), 34 of them new.
**Prior (2026-07-31):** DCIM manifest + `camtrap/clocks.py` — capture order preserved at flatten time, clock repair made segment-aware.
**What Changed (2026-07-31):** Two changes, both prerequisites for re-ingesting the campaigns from Synology. (1) `setup/flatten_for_camtrapdp.py` now writes a `dcim_manifest.csv` sidecar recording which SD-card DCIM folder every frame came from. Flattening pools `xxxx0001.JPG` from every folder into one directory and Timelapse2's `RelativePath` keeps only the deployment name, so capture order — the only way to detect a clock reset — used to be destroyed by this step. Nothing is renamed that was not renamed before, so existing joins on `file_name` are unaffected. The same script **no longer skips same-name/same-size files as duplicates**: that is exactly what a reset-clock camera emits, and a conservation check now aborts the run if any deployment ends up with fewer files than it should. (2) New `camtrap/clocks.py` owns clock-failure diagnosis — segments, capture-order evidence, coherence, and the repairability rule *a segment is repairable iff it is coherent AND contains ≥1 anchor*. It replaces the old binary `year < 2024` test, which could not see a forward jump, and it emits the third validity axis `valid_effort`. 25 fixtures in `tests/test_clocks.py` cover Felipe's scenarios A–G plus both precondition failures; run with `python3 -m unittest discover -s tests`. Verified against the real otoño 2026 export: CT_18 comes back as **5 segments** (10 / 32 / 40 / 3 / 227 frames), reproducing the 2026-07-30 hand analysis, and every segment is refused — including via its uncorroborated install anchor, which falls inside no segment.
**Prior (2026-07-30):** New `camtrap/` boundary package. `camtrap/stations.py` owns the canonical station convention (`CT01`–`CT27`) with historical spellings resolved through `data/campaigns/station_aliases.csv` (data, not code); `camtrap/observations.py` owns the canonical observation table, written by `timestamps.py` as `observations.parquet` alongside the existing `_corrected.csv`. `Anual-reports/2025/py/01_data_prep.py` now reads it via `read_campaigns()` — ~190 lines of duplicated clock repair, station parsing and species recovery deleted. **The report's numbers changed: 419 → 369 events.** Two causes, both corrections: cross-campaign dedup removed 325 double-counted images (primavera_2025 is almost entirely superseded by pv_2025_2026), and 143 records from otoño 2025 CT15/CT16/CT19 are now excluded because `timestamps.py` refuses to guess an offset the old code guessed. Previous figures preserved in `figures_pre_canonical/`.
**Prior (2026-06-25):** New module `timestamps.py` detects camera-clock-reset issues (EXIF reverts to 2017 epoch) and repairs them at the source using field-provided anchors. Each campaign now carries a `deployment_anchors.csv` and produces a `new_labeled_data_corrected.csv` that downstream projects consume in place of the raw reviewed CSV. CT_18 Otoño 2026 (135 bogus rows) repaired via `last_real_proxy` anchor — dates approximate, time-of-day flagged unreliable. CT-15/CT-16/CT-19 Otoño 2025 and TC-16 Primavera/PV (159 + 68 + 3 rows) marked `unrepairable_pending` until field anchors are recovered. See [Step 4b — Timestamp quality](#step-4b--timestamp-quality-check--repair).
**Integration Status:** Pending [full-category exports]. The code path is complete and validated end-to-end, but **no campaign can be re-ingested yet**: all four exports are animal-only or unswept, so the gate rejects every one of them. Validation was done on a scratch copy of otoño 2026 with `unclassified` relabelled to `empty` — CT18 reproduces the 2026-07-30 hand analysis exactly (5 segments of 10/32/40/3/227, every one refused, `valid_effort=FALSE`, install anchor falling inside no segment) and writes a 1,785-row `observations.parquet` in which CT18 is the only station out of the effort denominator. REMAINING once the exports land: handoff steps 6–7 — re-diagnose all four campaigns and regenerate `observations.parquet` (mirror `figures/` first, since otoño 2025 is in `REPORT_CAMPAIGNS` and its numbers may move), then fix pehuen. Note that existing `observations.parquet` files predate `valid_effort`, so `read_campaigns()` across old and new files will show it as null until every campaign is re-ingested.
**Blockers/Notes (2026-08-03):** **The one thing that unblocks everything is the Timelapse2 sweep.** For otoño 2026 the anchor evidence already exists — MegaDetector found 595 person frames — so the sweep is confirmation work, not search. Two field questions still gate specific data: CT18's install date and any maintenance visit (`docs/HANDOFF-clock-repair.md` §8.1), now the difference between recovering segment 0's 10 frames and losing them, with `11190001.JPG` (camera-time 2025-11-19 06:41, counter 0001) the frame to look at; and whether the older campaigns have install photos (§8.2), which decides whether otoño 2025's 143 dropped records are recoverable. Also: **pehuen's R scripts hardcode Windows paths** (`C:/Users/USUARIO/...` in `R/01_load_data.R:50–58`), so handoff step 7 cannot run from the Linux laptop until they are parameterised.
**Blockers/Notes (2026-07-31):** **Re-flatten every campaign with the new script before re-ingesting** — the manifest only exists for runs made after this change. Otoño 2026 was flattened before uploading to Synology and has **no pre-flatten backup**, so its five cameras with >999 images (CT_14 2632, CT_20 1836, CT_15 1331, CT_08 1129, CT_23 1088 frames) can never satisfy the ordering precondition; `clocks.py` deliberately still passes them because their clocks are clean and a camera that never reset needs no ordering. What cannot be recovered for otoño 2026 is whether the old duplicate-skip discarded any frame: CT_14 has 24 collision-renamed files (`102EK113_0119xxxx.JPG`) that survived only because their sizes differed, and a same-size sibling would have vanished silently. Its counters wrap, so gaps are undetectable. Bounding that needs per-camera file counts, which are gone with the pre-flatten tree.
**Prior Integration Status (2026-07-30):** CT_18 Otoño 2026 fix is now data-side, not config-side. `data-pipeline/config.yaml` should be pointed at `_corrected.csv` paths in the next Linux session before running `python run_fetch.py --ct`.
**Blockers/Notes (2026-07-30):** **143 records recoverable with field anchors.** Otoño 2025 CT15/CT16/CT19 are `unrepairable_pending`, so 143 animal records — including 6 puma, 3 guiña, 2 pudú — are excluded from the report. The old code recovered them by guessing `install_year - 2017`; `timestamps.py` will not guess. Two routes to recover them: Felipe's field notebook, **or** the camera filenames, which encode `MMDD` (`01230193.JPG` = Jan 23) and so pin the true date against the bogus 2017 EXIF stamp without any field data. The verdict notes in `manual_review_verdicts_2026-06-02.csv` already record the `+8yr` offset. Adding proper anchor rows is the single highest-value data task outstanding. Also: **`pehuen` and `data-pipeline` still read `_corrected.csv`** — both are still emitted, so nothing is broken, but migrating them to `observations.parquet` and retiring the CSV is the remaining half of this refactor. `export_best_images.py`, `run_classification.py` and the review UI still decode the Timelapse2 CSV / MegaDetector JSON directly (findings F002–F009 in the 2026-07-29 review).
**Blockers/Notes:** Outstanding anchor data needed from Felipe's field notebook for CT-15 / CT-16 / CT-19 (Otoño 2025) and chronic TC-16 issue across campaigns — until then those rows pass through with `valid_date=FALSE, valid_time_of_day=FALSE` (counted as station presence but excluded from any time analysis). CLIP horse/cow confusion may still appear on side/rear shots; revisit `clip_confidence_threshold` (0.28) only after the new run lands. Pandoc still required for `Anual-reports/2025/render.sh`. Annual report uses the canonical `plataforma-territorial/data/{boundary,camera_trap_stations}.geojson` files directly; legacy GIS files in `camera-traps/GIS/` are deprecated.

---

## Project Structure

```
camera-traps/
├── README.md                    ← this file
├── config.yaml                  ← per-campaign configuration (edit before each run)
├── environment.yml              ← conda environment definition
├── run_classification.py        ← Step 2: CLIP classification entry point
├── timestamps.py                ← Step 4b: segment-aware clock repair (ingest gate)
├── anchor_candidates.py         ← Step 4a: the short list of possible clock anchors
├── propose_anchors.py           ← Step 4a-bis: field visits → reviewable anchor rows
│
├── camtrap/                     ← boundary layer (one module per external format)
│   ├── stations.py              ← canonical station convention CT01..CT27 + aliases
│   ├── observations.py          ← canonical observation table (the data contract)
│   ├── clocks.py                ← clock-failure diagnosis + the repairability rule
│   ├── anchors.py               ← what the FIELD RECORD asserts: deployment windows,
│   │                              the anchor CSV, and visit→anchor pairing
│   ├── exports.py               ← the two Timelapse2 exports + full-category gate
│   └── detections.py            ← the MegaDetector JSON
│
├── tests/                       ← stdlib unittest; python3 -m unittest discover -s tests
│   ├── test_clocks.py           ← the repair RULE (Felipe's scenarios A–G)
│   ├── test_anchors.py          ← a visit is not an anchor; witness vs navigational
│   ├── test_exports.py          ← the export gate + its override
│   └── test_timestamps.py       ← the PLUMBING (per-segment offsets reach the rows)
│
├── classify_campaign/           ← CLIP classification package
│   ├── clip_classifier.py       ← zero-shot CLIP classifier (cosine similarity)
│   ├── cropping.py              ← MegaDetector bbox crop + resize
│   └── data_loader.py           ← loads animals from ImageData_animals.csv + MD JSON
│
├── phase1_labeling/             ← human review Streamlit app
│   └── app.py                   ← review UI (batch by species, export reviewed CSV)
│
├── setup/                       ← pre-processing utilities (run once per campaign)
│   ├── build_field_notes.py     ← ONE-TIME: monitoring workbook → field_notes.csv
│   ├── flatten_for_camtrapdp.py ← flatten per-camera subfolders to deployment level
│   ├── fix_unicode_filenames.py ← NFD → NFC filename normalization (Synology sync fix)
│   ├── create_junction.py       ← Windows junction for accented-path workaround
│   └── megadetector_campaigns.py← MegaDetector v6 CLI wrapper (alternative to AddaxAI)
│
└── Anual-reports/               ← deliverable reports (separate from the pipeline above)
    ├── 2022_2024_legacy methodology.pdf
    ├── REVISIÓN DISEÑO METODOLÓGICO DE CONAF.pdf
    ├── Resultados de evaluación Megadetector.docx.pdf
    ├── Registro de monitoreo CT.xlsx
    └── 2025/                    ← Informe anual 2025 (oct 2024 – mar 2026), self-contained
        ├── informe_anual_2025.md  ← Spanish narrative source
        ├── render.sh             ← pandoc helper → DOCX (for Word review)
        ├── README.md
        ├── py/                   ← 01_data_prep.py + 02_figures_tables.py
        ├── data/                 ← records_clean.parquet, events_clean.parquet, prep_log.txt
        └── figures/              ← 6 PNGs embedded by the .md
```

---

## Full Campaign Workflow

Each new deployment goes through four steps. Steps 1a and 1b are run once on arrival of new data; Steps 2 and 3 are run repeatedly as more campaigns accumulate.

### Step 0 — Data arrives from Synology

Images are stored at:
```
C:\Users\USUARIO\SynologyDrive\2. Camaras trampa (SC)\SynologyDrive\
  DATOS_GRILLA CÁMARAS TRAMPA\2. CAMPAÑAS DE RECOLECCION DE IMAGENES\
  <Season YYYY>\Fotos\
    <deployment-id>\   ← one subfolder per camera station
      *.JPG
```

### Step 0-bis — The field visit record

`data/campaigns/field_notes.csv` is the canonical record of who went where and when:
one row per **visit**, 106 visits across 27 stations. A visit is a physical event, not
a property of a campaign — at Bosque Pehuén every revision swaps the card, so one
visit **closes** one campaign and **opens** the next (`campaign_closed` /
`campaign_opened`).

It was migrated once from `data/campaigns/Registro de monitoreo CT.xlsx` by
`setup/build_field_notes.py`. **The CSV is canonical; the workbook is legacy** and is
kept only for provenance. Every inferred or corrected value is recorded in the row's
`data_flags` column, so a reader can see what was deduced without going back to the
script — 57 of 106 rows carry a flag.

Two consumers, and they need different things from it:

- **Deployment windows** (`camtrap/anchors.py`) — every station, every campaign. This
  is what makes a forward clock jump detectable.
- **Anchor proposals** (Step 4a-bis) — only for stations whose clock actually failed.

Dates in the workbook were a genuine hazard: it held three conventions at once —
Chilean `d/m/y` typed as text, `m/d/y` read off camera screens, and cells Excel had
already parsed using the machine locale. The last are the dangerous ones, because a
wrong reading looks clean. `clock_state` defaults to `unknown`, never `ok`: a visit
with no remark is not evidence the clock was fine.

If a new campaign's visits are not in this file, its stations get no deployment window
and their clean verdicts are reported as **unverified**.

### Step 1a — Fix Unicode filenames (if needed)

Synology/Linux sometimes syncs filenames in NFD Unicode form (decomposed accents), which breaks some Windows tools. Check and fix:

```bash
# Check — safe, no changes
python setup/fix_unicode_filenames.py

# Fix — renames in-place
python setup/fix_unicode_filenames.py --apply
```

> **Note:** Open the script and set `ROOT_DIR` to the campaign's `Fotos` folder first.

### Step 1b — Flatten folder structure

CamtrapDP and Timelapse2 expect images directly in the deployment folder, not in sub-subfolders. Flatten if the camera wrote into date-named subdirectories:

```bash
# Preview
python setup/flatten_for_camtrapdp.py "C:\path\to\Season YYYY\Fotos" --dry-run

# Apply
python setup/flatten_for_camtrapdp.py "C:\path\to\Season YYYY\Fotos"
```

**Copy the resulting `dcim_manifest.csv` into `data/campaigns/<campaign>/`.** Once the
tree is flat it is the only surviving record of which SD-card folder each frame came
from, and flattening consumes the tree that produced it — it cannot be regenerated.

> **What the manifest lets us claim, and what it does not.** The claim is *every frame
> in folder A was captured before every frame in folder B, because the camera fills its
> folders in name order.* That premise holds only when **both** conditions do:
>
> 1. **every group is a folder the camera created** — `100EK113`, `101EK113`. A folder
>    a person made says nothing about capture order (`clocks.dcim_folder_key`).
> 2. **every frame belongs to such a group** — otherwise a group is left unplaced
>    (`clocks.establish_order`).
>
> Otoño 2025 CT04 is why condition 1 exists: 723 loose frames sat under `M5` beside
> `M5/100EK113` and `M5/101EK113`. Recording the whole path made `M5` sort *first*,
> asserting its January frames preceded the October ones — a backwards step in capture
> order, which the diagnosis reads as a clock reset. On 2,097 frames.
>
> A deployment failing either condition is **refused, not guessed**: it drops to
> counter-only ordering, and per the P1 asymmetry that does not condemn it — a camera
> whose frames sit in-window and agree with their own filenames demonstrably never
> reset, ordered or not.

**Keep the deployment folder shallow: `CT04/*.JPG`, with the camera's own
`100EK113/` subfolders and nothing else.** An intermediate grid folder (`M5`, `M 11`)
is harmless to ordering but its name can leak into filenames when two frames collide,
so prefer not to create one.

> ⚠️ **One station's folder must never sit inside another's — the script now refuses.**
> Primavera 2025 arrived with `TC23_M20.2/` (2,460 files) nested inside `TC22_M19.2/`.
> Flattening would have moved all 2,460 into camera 22's deployment, at camera 22's
> coordinates. **The run would have looked perfect:** the two cameras use different
> filename schemes (`IMAG####` vs `MMDDnnnn`), so there were no collisions —
> `moved=2460 renamed=0 lost=0`, conservation check passed.
>
> Since 2026-08-14 that is a hard precondition, checked after discovery and **before a
> single file moves** — under `--dry-run` too, since a dry run exists to be trusted:
>
> ```
> ERROR: 1 station folder(s) are nested inside a deployment:
>     CT22/TC23_M20.2  (2460 file(s) would be attributed to CT22)
> ```
>
> **There is no flag to override it.** No arrangement puts one station folder
> legitimately inside another, so the fix is always the same — move the folder up to the
> DataPackage root and rename it canonically. `camtrap/stations.names_a_station()`
> decides what counts as station-shaped (`CT23`, `CT_23`, `TC23_M20.2`); a DCIM folder
> never does, `100EK113` included, even though that one is a real alias row.
>
> ⚠️ **And a second, general check behind it: one deployment, one capture story.**
> The check above recognises a station *folder* by name, so it knows the three
> spellings we have used and no others — a folder called `Camara 23` walks past it.
> `camtrap/provenance.multiple_capture_stories()` recognises a second camera by its
> **frames** instead, and enumerates nothing:
>
> ```
> ERROR: 1 deployment(s) contain frames from more than one camera:
>     CT22
>         IMAG#      2460 frame(s)   e.g. IMAG0001.JPG, IMAG0002.JPG
>         #            50 frame(s)   e.g. 05120001.JPG, 05120002.JPG
> ```
>
> Two filename shapes, each its own counter run, is what a separate camera looks like.
> Narrow-and-precise in front (it can say *which folder to move*), general-and-vague
> behind (it catches names nobody has thought of). **Measured across all four
> campaigns — 28,178 files, 0 false positives**, including the cases that must not
> fire: stills-plus-video, hand-renamed one-offs, our own `101EK113_` rename prefixes,
> and CT16's impossible months (a clock failure, not a provenance one).
>
> The evidence was never missing, only misfiled. Pooled into CT22, `establish_order`
> already reported `2460 filename(s) do not match the MMDD+counter grammar` — but it
> read that as an **ordering** problem, and failing to order does not condemn a camera,
> so the frames kept camera 22's identity anyway.

### Step 1c — Run MegaDetector via AddaxAI

1. Create a Windows junction so AddaxAI can reach the path without crashing on accented characters:

   ```bash
   # Edit create_junction.py: set `target` and `link` for the new campaign, then:
   python setup/create_junction.py
   # Creates e.g. C:\ADDAX\Otono_2025 → <accented Synology path>\Fotos
   ```

2. Open **AddaxAI**, point it at the junction path (e.g. `C:\ADDAX\Otono_2025`), run MegaDetector v5b.
3. Copy the resulting `timelapse_recognition_file.json` into the campaign `Fotos` folder.

Alternatively, run MegaDetector v6 directly (requires `wildlife_detector` package):

```bash
python setup/megadetector_campaigns.py \
  --input_dir "C:\ADDAX\Otono_2025" \
  --output_json "C:\path\to\Fotos\timelapse_recognition_file.json"
```

### Step 2 — Export TWO CSVs from Timelapse2

Every campaign needs **two** exports, because they answer different questions and
neither substitutes for the other.

**2a. The full sweep, then the all-images export — `ImageData_total.csv`**

1. Open the campaign's Timelapse2 project (`.tdb` template + image folder).
2. **Sweep every image**, assigning one category to each:
   `empty` / `animal` / `person` / `vehicle`.
3. Clear all filters. **File → Export data as CSV** → save as `ImageData_total.csv`
   in the campaign folder.
4. Check it before you close Timelapse2:

   ```bash
   python -m camtrap.exports "data/campaigns/<name>/ImageData_total.csv"
   ```

Why the sweep is mandatory: a clock reset that happens **between two animal photos
is invisible in an animal-only export.** That is not hypothetical — it is how otoño
2026 CT18 was recorded as one reset when it had **four**, and how a single offset came
to be applied across all of them, putting fabricated dates into the pehuén analysis.

Why `human` matters: install and retrieval photos of the technician **are** the clock
anchors. Each one buys back a whole segment of a broken clock, and without them there
is nothing to anchor to.

`observationType` uses **Camtrap DP's controlled vocabulary**, which the Timelapse2
template emits verbatim — `animal`, `human`, `vehicle`, `blank`, `unknown`,
`unclassified`. Note `blank` (not `empty`) and `human` (not `person`). MegaDetector's
own categories are a *different* vocabulary that does say `person`; `camtrap/exports.py`
owns the Camtrap DP one and `camtrap/detections.py` owns MegaDetector's.

> **The gate.** `timestamps.py` refuses to run unless `human` or `vehicle` appears in
> `observationType`. Presence of categories cannot be the test on its own, because in
> our Timelapse2 template `unclassified` doubles as `blank` — so a `{animal,
> unclassified}` export *looks* labelled while nothing was ever assigned. That verdict
> (`categories_never_assigned`) cannot be overridden.
>
> A value outside the vocabulary is also refused outright (`unrecognised_category_values`)
> rather than merely noted. An uninterpretable category counts as neither assigned nor
> proof, so a note would let a whole category vanish from the tally — which is exactly
> what happened on 2026-08-11, when 584 `human` rows went uncounted and otoño 2026's
> first real sweep passed only because `vehicle` is spelled the same in both vocabularies.
>
> For a campaign genuinely swept in full that really contains no human or vehicle,
> record the exception in `data/campaigns/<name>/export_gate_override.txt`:
>
> ```
> verified_by: Felipe Guarda
> date: 2026-08-03
> reason: swept all 12068 images; the technician serviced this camera without
>         triggering it, so no human frame exists on the card
> ```
>
> A file rather than a flag, so the decision carries a name and a date and travels
> with the data.

**2b. The animal-only export for the classifier — `ImageData_animals.csv`**

1. Filter to `observationType = animal`.
2. **File → Export data as CSV** → save as `ImageData_animals.csv`.

This one feeds CLIP (Step 3) and is used for nothing else. It needs at minimum
`RelativePath`, `File`, `observationType` and `fileMediatype`.

> **Note:** The `filePath` column may be empty depending on how the Timelapse2 project
> was set up — the pipeline handles both cases using `RelativePath + File` as a fallback.

**Field protocol — do this at every visit, it is what makes repair possible**

At install, at every mid-deployment visit, and at retrieval:

1. Note the **wall-clock time** on your phone, to the minute.
2. Note what the **camera's own screen** says at install. This is the only thing that
   distinguishes "the clock was right and later reset" from "the clock was already
   wrong when installed", and it costs five seconds.
3. **Trigger the camera deliberately** (wave a hand at the PIR, or open the case to
   fire the wakeup photo) so a person frame exists in the sequence.
4. Add a row to `deployment_anchors.csv` (schema in [Step 4b](#step-4b--timestamp-quality-check--repair)).

A **mid-visit** anchor is the only thing that can rescue an interior segment: on a
camera that reset four times, install and retrieval recover two of five segments and
nothing else does.

### Step 3 — CLIP zero-shot classification

Edit `config.yaml` to point at the new campaign (see [Configuration](#configuration)), then:

```bash
conda activate species-classifier
cd C:\Users\USUARIO\Dev\Python\camera-traps

python run_classification.py
# or: python run_classification.py --config config.yaml
```

**What it does:**
- Reads `ImageData_animals.csv` — the pre-filtered list of animal images from Timelapse2 review
- For each image, looks up its bounding box in `timelapse_recognition_file.json` (MegaDetector output)
- Images not found in the JSON (confirmed by reviewer but missed by MegaDetector) are classified on the full frame
- Crops each image to the MegaDetector bounding box (5% padding)
- Classifies with CLIP (`openai/clip-vit-base-patch32`) against the 29-species English prompts
- Scores < `clip_confidence_threshold` (0.28) → marked `"No reconocible"` instead of a forced guess
- Writes `ImageData_animals_classified.csv` to the campaign folder

### Step 4 — Human review (Streamlit)

```bash
streamlit run phase1_labeling/app.py
# Opens at http://localhost:8501
```

The UI:
- Groups images by CLIP-proposed species, one batch page per species sorted by detection count
- **Burst context**: each image is shown as a triptych `[anterior | actual | siguiente]` — the previous and next frames from the same station (full-frame, sourced from the MD JSON so empty triggers are included). Camera traps fire in bursts of 2–3 frames, so the neighbours often show the same animal at a different angle — strong cue for species ID.
- Full-frame thumbnails (no bbox crop) so habitat / scale context is visible. The CLIP classifier itself still sees the cropped subject — only the human-review UI displays the full frame.
- **Two confirm buttons** per batch:
  - *Confirmar todo como X* — bulk confirm all images as the proposed species
  - *Confirmar con cambios* — apply any per-image dropdown edits before confirming
- Exports `new_labeled_data_reviewed.csv` — CamtrapDP format + `reviewOutcome` column (`"confirmed"` / `"corrected"`)

### Step 4a — Find the clock anchors

Before repairing anything, get the short list of frames that could become anchors:

```bash
python anchor_candidates.py --campaign <name>
# only the frames inside a segment that is still unrepairable:
python anchor_candidates.py --campaign <name> --unanchored-only
```

It lists, per station, every frame worth opening — with the segment each one sits in
and whether that segment still needs an anchor. Writes `anchor_candidates.csv` and
prints a per-station summary naming what is still unrepairable and what could
rescue it.

Candidates come in two kinds, and the difference decides what they can be used for:

| | kind | what it proves |
|---|---|---|
| **witness** | `human_labelled`, `vehicle_labelled` | from the swept export — someone looked at the frame and said a person was there, so it can **date a visit** |
| | `person_detection`, `vehicle_detection` | from MegaDetector — the same claim, unconfirmed |
| **navigational** | `counter_0001`, `segment_edge` | where a card was swapped or a segment begins — says **where to look**, never when a frame was taken |

Only a witness frame can become an anchor. A counter-`0001` frame sits at the start of
a card, not at the moment of a visit: CT18's segment 0 opens with `11190001.JPG` five
days *after* the install, because nothing triggered the camera until then. Pairing
those two would apply a −5 day offset to ten frames whose clock was correct.

Note the two vocabularies. `human`/`vehicle` are **Camtrap DP** values from the swept
export; `person`/`vehicle` are **MegaDetector's** own category names. They mean the
same thing and are deliberately spelled differently, because the modules that own them
are different (`camtrap/exports.py` and `camtrap/detections.py`).

This report is **not** gated on the full-category export: a campaign that fails the
gate is exactly the one that needs the list. Run it on whatever export exists.

### Step 4a-bis — Propose anchors from the field record

```bash
python propose_anchors.py --campaign <name>
python propose_anchors.py --campaign <name> --write   # appends READY rows only
```

Joins `data/campaigns/field_notes.csv` (the wall clock of every visit) to
`anchor_candidates.csv` (what the camera's clock said) and writes
`anchor_proposals.csv`, one row per segment, each `READY`, `NEEDS_REVIEW` or
`NOT_NEEDED`.

> **A visit is not an anchor.** The notebook records when someone *visited*, not what
> the clock *read*. For a camera whose clock is fine, forcing the visit date on as an
> anchor applies the notebook's own imprecision to correct data — CT01's notebook says
> 2025-11-24 → 2026-05-13 while its frames run 2025-11-26 → 2026-05-14 across one
> coherent segment. So an anchor is proposed **only** where the segment would
> otherwise be refused. A clean camera gets `NOT_NEEDED` and no row.

Nothing is promoted automatically. Review the file, open the frames it names, then
move accepted rows into `deployment_anchors.csv` by hand — that file is the one place
a human signature still means something. A segment that cannot be paired becomes an
`unrepairable_pending` row rather than no row at all, so the refusal is **written
down**: a station missing from the anchor file and a station known to be unanchorable
look identical downstream, and only one of them is a decision anybody made.

An opening visit recorded with a date but no time (all 27 of otoño 2026's) yields a
`visit_date_only` anchor, which is APPROXIMATE — the date is recovered and
`valid_time_of_day` stays FALSE, so activity analysis never sees it. Asserting an hour
nobody wrote down is how CT18's install anchor came to claim `14:00:00` against a
notebook that says only `2025-11-14`.

### Step 4b — Timestamp quality check & repair

After review, every campaign **must** be processed through `timestamps.py`
before downstream consumers (data-pipeline, pehuen-species-interactions,
annual report) read the data. It diagnoses each camera's clock from the
all-images export, splits it into **segments** at each reset, and applies a
**separate offset per segment** from the field anchors.

```bash
conda activate species-classifier
cd C:\Users\USUARIO\Dev\Python\camera-traps

# Audit only (no files written)
python timestamps.py --campaign <name> --dry-run

# Apply repair + write new_labeled_data_corrected.csv
python timestamps.py --campaign <name>
```

**The rule** (`camtrap/clocks.py`, spec in `docs/HANDOFF-clock-repair.md` §5):

> A segment is repairable **iff** it is coherent **and** contains at least one anchor.
> The number of repairable segments equals the number of segments an anchor falls
> inside.

So anchors are cheap and each one buys back a whole segment. Two preconditions fail
closed: capture order must be establishable (from `dcim_manifest.csv` + the filename
counter), and within a segment the filename's `MMDD` must agree with its own
`DateTime`. Failing the ordering precondition does **not** by itself condemn a camera
— a camera whose frames all sit inside the deployment window and agree with their own
filenames demonstrably never reset, whether or not we can order it.

It **hard-fails** when there is no valid full-category `ImageData_total.csv`; see
[Step 2](#step-2--export-two-csvs-from-timelapse2). There is deliberately no fallback
to the animal-only export.

**Anchor CSV schema** (one row per anchor event):

| column | meaning |
|---|---|
| `station_id` | canonical station ID (`CT01`..`CT27`); resolved via `camtrap.stations`, so it need not match the campaign's raw `Deployments` spelling |
| `anchor_type` | `install` / `mid_visit` / `retrieval` / `last_real_proxy` / `visit_date_only` / `unrepairable_pending` |
| `real_datetime` | wall-clock time at the anchor moment (YYYY-MM-DD HH:MM:SS) |
| `camera_datetime` | what the camera's clock said at that moment (= trigger photo's EXIF stamp) |
| `source` | provenance: `field_notebook`, `trigger_photo`, etc. |
| `notes` | free text |
| `segment_index` | **optional.** Normally an anchor finds its own segment by strict containment of `camera_datetime`. Set this only when containment cannot place it |

When `anchor_type ∈ {install, mid_visit, retrieval}` AND a trigger photo was
captured at the visit, `camera_datetime` IS that photo's EXIF stamp — the
offset `real_datetime − camera_datetime` is exact, and repaired rows get
both `valid_date=TRUE` and `valid_time_of_day=TRUE`.

When the camera was not firing at the visit, use `anchor_type=last_real_proxy`
with `camera_datetime` = last bogus photo's stamp — repaired rows get
`valid_date=TRUE` but `valid_time_of_day=FALSE` (rotation uncertainty
unbounded). For pehuén this excludes them from activity/overlap analyses but
keeps them for occupancy/spatial.

When the visit is recorded with a date but **no time**, use
`anchor_type=visit_date_only`. It is approximate for the same reason and carries the
same consequence (`valid_date=TRUE`, `valid_time_of_day=FALSE`) — the hour is assumed
to be noon, which bounds the date error at ±12 h and never reaches an output.

**The deployment window now comes from `data/campaigns/field_notes.csv`**, falling
back to the anchors only when the field record cannot supply both ends. This matters
because the window is the *only* way a FORWARD jump is visible — a clock set ahead
keeps every capture delta positive — and anchors exist only where a clock already
broke, so before field notes 26 of otoño 2026's 27 stations had no window at all.
A visit-derived window uses a 3-day tolerance rather than the anchors' 1 h: a notebook
date has day precision and the visit itself spans several days. The bound is measured,
not guessed — across the 20 stations provably coherent from capture order alone, the
largest excursion past a recorded visit date is +1.67 d.

A station for which no window can be built (CT27 — no install record) is reported as
**unverified clean** rather than clean: the in-window test never ran, so its passing
verdict is an absence of evidence.

When no field anchor exists yet, use `anchor_type=unrepairable_pending` with
empty `real_datetime` and `camera_datetime` — the row documents that the
clock issue is known and awaiting field info; **every** row of that station is
refused until it arrives.

**When to set `segment_index`.** Two real cases, both from otoño 2026 CT18: an anchor
that falls inside **no** segment (the install anchor asserts camera-time 2025-11-14
14:00, but the first frame on the card is 11-19 06:41), and an anchor that falls
inside **several** (segments 1, 2 and 3 all begin 2017-01-01). Both repair nothing on
their own — the anchor is uncorroborated or ambiguous, and that is the honest answer.
`segment_index` is an assertion that someone checked by eye which segment the anchor
belongs to; say who in `notes`. Run `anchor_candidates.py` first.

**Output:** `data/campaigns/<name>/new_labeled_data_corrected.csv` (reviewed
CSV plus seven columns: `datetime_corrected`, `valid_date`, `valid_time_of_day`,
`valid_effort`, `clock_segment`, `repair_method`, `repair_anchor_source`),
`observations.parquet` (the canonical table) and `timestamps_audit.log`.

**Three independent validity flags.** They must stay independent — a pure year error
preserves time-of-day exactly, so those rows are still valid for activity and overlap
analysis:

| flag | question | scope |
|---|---|---|
| `valid_date` | is the date trustworthy? | per row |
| `valid_time_of_day` | is the time-of-day trustworthy? | per row |
| `valid_effort` | are this camera's trap-nights knowable? | **per station** |

`valid_effort=FALSE` means the camera's operating period is unknown, so it must leave
the effort **denominator** as well as the numerator — for every row, including rows
whose own date is fine. Consumers computing any rate (detections per 100 trap-nights)
must drop those stations entirely rather than only excluding their records.

**Important:** Downstream projects read `_corrected.csv` / `observations.parquet`,
**not** `_reviewed.csv`. The reviewed CSV is the immutable reviewer output; the
corrected files fix the clock issues at the source.

See the `timestamps.py` and `camtrap/clocks.py` module docstrings for the full
algorithm, and run the fixtures with `python3 -m unittest discover -s tests`.

### Step 5 — Export best images

After review, export the best images per species and per station for sharing / platform display:

```bash
conda activate species-classifier
cd C:\Users\USUARIO\Dev\Python\camera-traps

python export_best_images.py
```

**What it does:**
- Auto-discovers all campaigns under the Synology base path that have both `new_labeled_data_reviewed.csv` and `timelapse_recognition_file.json` (handles both root and `Fotos/` layouts)
- Resolves species for rows reviewed via "Otro (especificar)" using a case-insensitive Spanish name lookup
- Produces two outputs per campaign:
  - `exports/<campaign>/species/<common_latin>/` — top 5 images per species globally, ranked by MegaDetector confidence (for sharing / reports)
  - `exports/<campaign>/stations/<station>/` — top 3 images per station (any species, ranked by confidence) — ready for platform map popups
- Filenames: `{station}_{original_filename}.jpg` — fully traceable back to the source image
- Species not in the known map get a `_UNKNOWN_` prefix so they are easy to spot

> Edit `TOP_N_SPECIES` and `TOP_N_STATION` at the top of `export_best_images.py` to change image counts. New campaigns are picked up automatically — no config needed.

---

## Configuration (`config.yaml`)

```yaml
# ── Paths ────────────────────────────────────────────────────────────────────
campaign_dir:      "C:/path/to/Season YYYY"   # ← update for each campaign
megadetector_json: "timelapse_recognition_file.json"
input_csv:         "ImageData_animals.csv"     # Timelapse2 export filtered to observationType=animal
output_csv:        "ImageData_animals_classified.csv"

# ── Detection filtering ───────────────────────────────────────────────────────
animal_confidence_threshold: 0.38   # MegaDetector detection threshold
animal_category: "1"                # '1' = animal in timelapse JSON

# ── CLIP model ────────────────────────────────────────────────────────────────
clip_model: "openai/clip-vit-base-patch32"
clip_confidence_threshold: 0.28   # scores below this → "No reconocible"

# ── Output metadata ───────────────────────────────────────────────────────────
classified_by:         "CLIP zero-shot"
classification_method: "machine"
```

Only `campaign_dir`, `input_csv`, and `output_csv` change between campaigns.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate species-classifier
```

### GPU Requirements

The RTX 5060 Ti is Blackwell architecture (sm_120). Standard PyTorch does **not** support it — you need the cu128 build:

```bash
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128 \
  --force-reinstall --no-deps
```

This is already done in the working environment. Only redo if rebuilding the conda env from scratch.

---

## Species List

Canonical species catalog lives at `data-pipeline/species.yaml` (shared by `data-pipeline`, `camera-traps`, and `plataforma-territorial`). 29 species carry an `english:` CLIP prompt; a few reviewer-discovered species (Chingue, Cachaña, Fío-fío, Libélula) are catalogued but not passed to the classifier.

| Spanish | Latin | Notes |
|---|---|---|
| Zorro culpeo | *Lycalopex culpaeus* | |
| Puma | *Puma concolor* | Priority |
| Guiña | *Leopardus guigna* | Priority |
| Jabalí | *Sus scrofa* | Invasive |
| Liebre | *Lepus europaeus* | Invasive |
| Visón | *Neogale vison* | Invasive |
| Perro | *Canis lupus familiaris* | Invasive |
| Caballo | *Equus caballus* | Invasive |
| Vaca | *Bos taurus* | Invasive |
| Quique | *Galictis cuja* | Native mustelid (added 2026-06-17) |
| Gato doméstico | *Felis catus* | Invasive |
| Monito del monte | *Dromiciops gliroides* | |
| Ratón cola larga | *Abrothrix longipilis* | |
| Chucao | *Scelorchilus rubecula* | |
| Hued hued | *Pteroptochos tectus* | |
| Rayadito | *Aphrastura spinicauda* | |
| Concón | *Strix rufipes* | |
| Carpintero | *Campephilus magellanicus* | |
| Bandurria | *Theristicus melanopis* | |
| Queltehue | *Vanellus chilensis* | |
| Tiuque | *Milvago chimango* | |
| Peuquito | *Accipiter chilensis* | |
| Zorzal | *Turdus falcklandii* | |
| Cometocino | *Phrygilus gayi* | |
| Picaflor | *Sephanoides sephaniodes* | |
| Diucón | *Xolmis pyrope* | |
| Traro | *Caracara plancus* | |
| Ciervo rojo | *Cervus elaphus* | Invasive |
| Pudú | *Pudu puda* | Priority |

---

## Known Limitations (Phase 1 / CLIP)

- **Forced-choice**: CLIP always picks the best match from the species list, even for empty or ambiguous images. The 0.28 threshold filters the worst ~12% but doesn't eliminate all errors.
- **Diucón over-classification**: CLIP's "fire-eyed diucon" embedding attracts dark, ambiguous images. High false-positive rate for this species.
- **Solution**: Phase 2 — custom EfficientNetV2 classifier trained on human-reviewed data.

---

## Campaign History

Per the field record there are **three** deployments, not four. Campaigns are named for
the season they are **retrieved** in.

| Campaign | Ran | Retrieved | Status |
|---|---|---|---|
| Otoño 2025 | 2024-10-09 → 2025-05-14/06-11 | autumn 2025 | Re-downloaded and flattened 2026-08-12 (8,997 files). Export passes the gate. **CT15/CT16/CT19 `unrepairable_pending`**; 8 images undecodable (six 0-byte CT04, two all-zero CT13) and labelled `blank` — a known, accepted limitation |
| Primavera 2025 | 2025-05-14/06-11 → 2025-11-12/2026-01-14 | spring 2025 | Re-downloaded and flattened 2026-08-13 (19,522 files, **26 stations**). Existing parquet is a **partial** ingest of 14 stations. **CT16 clock corrupt — see §3 above** |
| Otoño 2026 | 2025-11-12/2026-01-14 → 2026-05-13/15 | autumn 2026 | Ingested 2026-08-12 — 1,785 rows, 26 clean stations. **CT_18 refused on all five segments** (4 resets, not 1; `docs/HANDOFF-clock-repair.md`). **Carries the unfixed horario-de-invierno shift** |

**`pv_2025_2026` is not in this table because it is not a campaign** — it is a second
Timelapse2 review pass over Primavera 2025's cards. See §2 above. Its reviewed CSV
(`data/campaigns/pv_2025_2026/new_labeled_data_reviewed.csv`) still holds adjudicated
labels that Primavera 2025's does not, so it must be merged rather than discarded.

Reviewed CSVs live at `data/campaigns/<campaign>/new_labeled_data_reviewed.csv`.

---

## DESIGN_NOTES

**Dominant coupling risk — external file-format shape leakage.** The Timelapse2 CamtrapDP CSV schema (`filePath` vs `RelativePath`+`File`, `observationType`, `scientificName`, `Deployments`) and the MegaDetector JSON schema (`images[].detections[].category/conf/bbox`) are each re-derived independently across `classify_campaign/data_loader.py`, `run_classification.py`, `phase1_labeling/app.py`, `export_best_images.py`, `timestamps.py`, and `Anual-reports/2025/py/*`. One vendor export change — or one more path-fallback rule — forces the same edit in six files; this decay has already produced duplicate `extract_camera_num` and `EPISODE_GAP`/`build_events` definitions in the report scripts.

**Boundary that must hold.** One module owns each external format: a single record reader owns the Timelapse2 row (path resolution, animal/video filter, column names), and `classify_campaign/data_loader.py` owns the MegaDetector JSON. Everything downstream — Streamlit review UI, image export, timestamp repair, annual-report scripts — consumes resolved records and must stay unaware of column names, JSON keys, and on-disk campaign layout.
