# Camera Traps — Species Recognition Pipeline

Automated species identification pipeline for camera-trap deployments at Fundación Maradentro (Reserva Costera Valdiviana and associated sites). Combines MegaDetector animal detection with CLIP zero-shot classification and a Streamlit human-review interface.

---

## Status

**Last Updated:** 2026-08-03 — the segment-aware repair is now wired into ingest: `timestamps.py` consumes `camtrap/clocks.py`, the full-category export gate is enforced, and `anchor_candidates.py` finds the anchors
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
│
├── camtrap/                     ← boundary layer (one module per external format)
│   ├── stations.py              ← canonical station convention CT01..CT27 + aliases
│   ├── observations.py          ← canonical observation table (the data contract)
│   ├── clocks.py                ← clock-failure diagnosis + the repairability rule
│   ├── exports.py               ← the two Timelapse2 exports + full-category gate
│   └── detections.py            ← the MegaDetector JSON
│
├── tests/                       ← stdlib unittest; python3 -m unittest discover -s tests
│   ├── test_clocks.py           ← the repair RULE (Felipe's scenarios A–G)
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

It joins the MegaDetector JSON to `ImageData_total.csv` and lists, per station, every
person/vehicle detection, every counter-`0001` frame (an SD-card folder start, i.e.
where a card was swapped or the camera rebooted), and every segment boundary — with
the segment each one sits in and whether that segment still needs an anchor. Writes
`anchor_candidates.csv` and prints a per-station summary naming what is still
unrepairable and what could rescue it.

This report is **not** gated on the full-category export: a campaign that fails the
gate is exactly the one that needs the list. Run it on whatever export exists.

Open the candidate images for a segment that needs an anchor. If a frame shows a
person at a moment you can date — a visit in the field notebook, a phone photo with
its own timestamp — add an anchor row.

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
| `anchor_type` | `install` / `mid_visit` / `retrieval` / `last_real_proxy` / `unrepairable_pending` |
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

| Campaign | Status | Reviewed CSV |
|---|---|---|
| Primavera 2025 | Complete — largely superseded by PV 2025-2026 (see note) | `data/campaigns/primavera_2025/new_labeled_data_reviewed.csv` |
| Otoño 2025 | Complete | `data/campaigns/otono_2025/new_labeled_data_reviewed.csv` |
| Primavera-verano 2025-2026 | Complete | `data/campaigns/pv_2025_2026/new_labeled_data_reviewed.csv` |
| Otoño 2026 | Reviewed; **CT_18 clock unrepairable — 4 resets, not 1. Dates fabricated by the current repair; see `docs/HANDOFF-clock-repair.md`** | `data/campaigns/otono_2026/new_labeled_data_reviewed.csv` |

---

## DESIGN_NOTES

**Dominant coupling risk — external file-format shape leakage.** The Timelapse2 CamtrapDP CSV schema (`filePath` vs `RelativePath`+`File`, `observationType`, `scientificName`, `Deployments`) and the MegaDetector JSON schema (`images[].detections[].category/conf/bbox`) are each re-derived independently across `classify_campaign/data_loader.py`, `run_classification.py`, `phase1_labeling/app.py`, `export_best_images.py`, `timestamps.py`, and `Anual-reports/2025/py/*`. One vendor export change — or one more path-fallback rule — forces the same edit in six files; this decay has already produced duplicate `extract_camera_num` and `EPISODE_GAP`/`build_events` definitions in the report scripts.

**Boundary that must hold.** One module owns each external format: a single record reader owns the Timelapse2 row (path resolution, animal/video filter, column names), and `classify_campaign/data_loader.py` owns the MegaDetector JSON. Everything downstream — Streamlit review UI, image export, timestamp repair, annual-report scripts — consumes resolved records and must stay unaware of column names, JSON keys, and on-disk campaign layout.
