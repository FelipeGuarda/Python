# Camera Traps — Species Recognition Pipeline

Automated species identification pipeline for camera-trap deployments at Fundación Maradentro (Reserva Costera Valdiviana and associated sites). Combines MegaDetector animal detection with CLIP zero-shot classification and a Streamlit human-review interface.

---

## Status

**Last Updated:** 2026-05-11 — code review complete
**What Changed:** Today closed the camera-traps half of the FMA-ecosystem review. Tier 1: S37 mtime-keyed Streamlit cache. Tier 2: S39 `run_classification.py main()` split into `classify_all()` + `apply_classifications()`. Bundle C: S29 (`load_thumbnail` now reuses `crop_to_bbox`), S30 + S32 (argparse for `setup/fix_unicode_filenames.py` + `setup/create_junction.py`), S36 (env+CLI override for `CAMPAIGNS_BASE`), S43 (new `setup/_fileops.py` with `is_target` / `move_file` / `cleanup_empty_dirs`). Tier 4: S35 `AnimalRow` dataclass replaces in-place CSV-row mutation in `export_best_images.py`; S38 UI strings standardized to Spanish in `phase1_labeling/app.py`. **First full code review now complete:** every finding in this project is closed. See repo-root `CHANGELOG.md`.
**Integration Status:** Ready
**Blockers/Notes:** Sibling-loader pattern chosen over shared import so Windows runs work without data-pipeline on PYTHONPATH. 8 Spanish display names changed to canonical form via species.yaml — still flagged for biological review with Felipe; species.yaml is the single edit point if any are wrong.

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
│   ├── crop_utils.py            ← MegaDetector bbox crop + resize
│   └── data_loader.py           ← loads animals from ImageData_animals.csv + MD JSON
│
├── phase1_labeling/             ← human review Streamlit app
│   └── app.py                   ← review UI (batch by species, export reviewed CSV)
│
└── setup/                       ← pre-processing utilities (run once per campaign)
    ├── flatten_for_camtrapdp.py ← flatten per-camera subfolders to deployment level
    ├── fix_unicode_filenames.py ← NFD → NFC filename normalization (Synology sync fix)
    ├── create_junction.py       ← Windows junction for accented-path workaround
    └── megadetector_campaigns.py← MegaDetector v6 CLI wrapper (alternative to AddaxAI)
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
- Classifies with CLIP (`openai/clip-vit-base-patch32`) against the 26-species English prompts
- Scores < `clip_confidence_threshold` (0.28) → marked `"No reconocible"` instead of a forced guess
- Writes `ImageData_animals_classified.csv` to the campaign folder

### Step 4 — Human review (Streamlit)

```bash
streamlit run phase1_labeling/app.py
# Opens at http://localhost:8501
```

The UI:
- Groups images by CLIP-proposed species, one batch page per species sorted by detection count
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

27 species configured in `config.yaml`. Each entry has a Spanish name (written to `observationComments`), Latin binomial (`scientificName`), and English CLIP prompt.

| Spanish | Latin | Notes |
|---|---|---|
| Zorro culpeo | *Lycalopex culpaeus* | |
| Puma | *Puma concolor* | |
| Guiña | *Leopardus guigna* | |
| Jabali | *Sus scrofa* | Invasive |
| Liebre | *Lepus europaeus* | Invasive |
| Visón | *Neogale vison* | Invasive |
| Perro | *Canis lupus familiaris* | |
| Caballo | *Equus caballus* | |
| Gato doméstico | *Felis catus* | |
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
| Pudú | *Pudu puda* | |

---

## Known Limitations (Phase 1 / CLIP)

- **Forced-choice**: CLIP always picks the best match from the species list, even for empty or ambiguous images. The 0.28 threshold filters the worst ~12% but doesn't eliminate all errors.
- **Diucón over-classification**: CLIP's "fire-eyed diucon" embedding attracts dark, ambiguous images. High false-positive rate for this species.
- **Solution**: Phase 2 — custom EfficientNetV2 classifier trained on human-reviewed data.

---

## Campaign History

| Campaign | Status | Reviewed CSV |
|---|---|---|
| Primavera 2025 | Complete | `data/campaigns/primavera_2025/new_labeled_data_reviewed.csv` |
| Otoño 2025 | Complete (694 obs) | `<Synology>/Otoño 2025/Fotos/new_labeled_data_reviewed.csv` |
| Primavera-verano 2025-2026 | In progress | `data/campaigns/primavera_verano_2025_2026/` |
