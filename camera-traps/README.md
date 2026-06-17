# Camera Traps — Species Recognition Pipeline

Automated species identification pipeline for camera-trap deployments at Fundación Maradentro (Reserva Costera Valdiviana and associated sites). Combines MegaDetector animal detection with CLIP zero-shot classification and a Streamlit human-review interface.

---

## Status

**Last Updated:** 2026-06-17 — Otoño 2026 campaign integrated + Quique added
**What Changed:** Reviewed CSV for the May 2026 SD pull (campaign name **Otoño 2026**) staged at `data/campaigns/otono_2026/` and registered in `data-pipeline/config.yaml`. 1785 observations across 25 deployments (CT_02 and CT_12 produced no animal triggers; the timelapse parser is observation-centric so they're correctly absent from `ct_deployments`). Vaca payoff confirmed: 579 rows tagged Vaca (top species in this campaign) that would have been Caballo without the species addition. Added Quique (*Galictis cuja*) to the shared species catalog with a CLIP English prompt — 5 observations in this campaign, native mustelid, first record in the project. CSV-vs-CSV overlap check against otono_2025 / primavera_2025 / pv_2025_2026 returned zero hits, so no dedup script needed.
**Integration Status:** Pending CT_18 timestamp fix. 135 rows on CT_18 carry a 2017-01-01 DateTime (camera clock reverted to factory default); real deployment-start anchor pending from field notebook. Linux ingestion (`python run_fetch.py --ct`) is held until those timestamps are corrected.
**Blockers/Notes:** Do **not** run `--ct` on the Linux box until CT_18 is fixed — see the comment block on the new Otoño 2026 entry in `data-pipeline/config.yaml`. Once fixed, ingestion is one command. CLIP horse/cow confusion may still appear on side/rear shots; revisit `clip_confidence_threshold` (0.28) only after the new run lands. Pandoc still required for `Anual-reports/2025/render.sh`. Annual report uses the canonical `plataforma-territorial/data/{boundary,camera_trap_stations}.geojson` files directly; legacy GIS files in `camera-traps/GIS/` are deprecated.

---

## Project Structure

```
camera-traps/
├── README.md                    ← this file
├── config.yaml                  ← per-campaign configuration (edit before each run)
├── environment.yml              ← conda environment definition
├── run_classification.py        ← Step 2: CLIP classification entry point
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

### Step 2 — Export animal image list from Timelapse2

Before running CLIP classification, export the animal observations from Timelapse2:

1. Open the campaign's Timelapse2 project (`.tdb` template + image folder)
2. In Timelapse2, filter to `observationType = animal`
3. **File → Export data as CSV** → save as `ImageData_animals.csv` in the campaign folder

This CSV is the input to the classifier. It should contain one row per animal image with at minimum `RelativePath`, `File`, `observationType`, and `fileMediatype` columns (standard Timelapse2 CamtrapDP export).

> **Note:** The `filePath` column may be empty depending on how the Timelapse2 project was set up — the pipeline handles both cases automatically using `RelativePath + File` as a fallback.

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
| Primavera 2025 | Complete | `data/campaigns/primavera_2025/new_labeled_data_reviewed.dedup.csv` |
| Otoño 2025 | Complete | `data/campaigns/otono_2025/new_labeled_data_reviewed.csv` |
| Primavera-verano 2025-2026 | Complete | `data/campaigns/pv_2025_2026/new_labeled_data_reviewed.csv` |
| Otoño 2026 | Reviewed; ingestion pending CT_18 timestamp fix | `data/campaigns/otono_2026/new_labeled_data_reviewed.csv` |
