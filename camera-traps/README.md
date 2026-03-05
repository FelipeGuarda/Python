# Camera Traps — Species Recognition Pipeline

Automated species identification pipeline for camera-trap deployments at Fundación Maradentro (Reserva Costera Valdiviana and associated sites). Combines MegaDetector animal detection with CLIP zero-shot classification and a Streamlit human-review interface.

---

## Project Structure

```
camera-traps/
├── README.md                    ← this file
├── NEXT_SESSION.md              ← technical notes and next steps
├── config.yaml                  ← per-campaign configuration (edit before each run)
├── environment.yml              ← conda environment definition
├── run_classification.py        ← Step 2: CLIP classification entry point
├── old animal data DB.csv       ← historical species frequency table
│
├── classify_campaign/           ← CLIP classification package
│   ├── clip_classifier.py       ← zero-shot CLIP classifier (cosine similarity)
│   ├── crop_utils.py            ← MegaDetector bbox crop + resize
│   └── data_loader.py           ← loads animals from MD JSON + Timelapse2 .ddb
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

### Step 2 — CLIP zero-shot classification

Edit `config.yaml` to point at the new campaign (see [Configuration](#configuration)), then:

```bash
conda activate species-classifier
cd C:\Users\USUARIO\Dev\Python\camera-traps

python run_classification.py
# or: python run_classification.py --config config.yaml
```

**What it does:**
- Reads MegaDetector detections ≥ `animal_confidence_threshold` (0.38) from `timelapse_recognition_file.json`
- Also reads `DeleteFlag=true` images from the Timelapse2 `.ddb` (catches animals MegaDetector missed)
- Crops each image to the MegaDetector bounding box (5% padding)
- Classifies with CLIP (`openai/clip-vit-base-patch32`) against the 26-species English prompts
- Scores < `clip_confidence_threshold` (0.28) → marked `"No reconocible"` instead of a forced guess
- Writes `new_labeled_data_classified.csv` to the campaign folder

### Step 3 — Human review

```bash
streamlit run phase1_labeling/app.py
# Opens at http://localhost:8501
```

The UI:
- Groups images by CLIP-proposed species
- Species with ≥ `min_historical_count` (31) records in `old animal data DB.csv` get dedicated batch pages
- Rare species + low-confidence images → "Otras especies" batch at the end
- **Two confirm buttons** per batch:
  - *Confirmar todo como X* — bulk confirm all images as the proposed species
  - *Confirmar con cambios* — apply any per-image dropdown edits before confirming
- Exports `new_labeled_data_reviewed.csv` — CamtrapDP format + `reviewOutcome` column (`"confirmed"` / `"corrected"`)

---

## Configuration (`config.yaml`)

```yaml
# ── Paths ────────────────────────────────────────────────────────────────────
campaign_dir:     "C:/path/to/Season YYYY"   # ← update for each campaign
megadetector_json: "timelapse_recognition_file.json"
database:          "CamtrapDB_<season>.ddb"
input_csv:         "new_labeled_data_CamptrapDP.csv"
output_csv:        "new_labeled_data_classified.csv"

# ── Historical data ───────────────────────────────────────────────────────────
old_data_csv:           "old animal data DB.csv"   # relative to config file
min_historical_count:   31   # species below this get no dedicated batch page

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

Only `campaign_dir`, `database`, `input_csv`, and `output_csv` change between campaigns.

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

26 species configured in `config.yaml`. Each entry has a Spanish name (written to `observationComments`), Latin binomial (`scientificName`), and English CLIP prompt.

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

---

## Known Limitations (Phase 1 / CLIP)

- **Forced-choice**: CLIP always picks the best match from the species list, even for empty or ambiguous images. The 0.28 threshold filters the worst ~12% but doesn't eliminate all errors.
- **Diucón over-classification**: CLIP's "fire-eyed diucon" embedding attracts dark, ambiguous images. High false-positive rate for this species.
- **Solution**: Phase 2 — custom EfficientNetV2 classifier trained on human-reviewed data (see `NEXT_SESSION.md`).

---

## Campaign History

| Campaign | Status | Reviewed CSV |
|---|---|---|
| Primavera 2025 | Complete | `new_labeled_data_reviewed.csv` in campaign folder |
| Otoño 2025 | MegaDetector pending | Junction: `C:\ADDAX\Otono_2025` |
