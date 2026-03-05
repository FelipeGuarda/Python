# Camera Traps — Session Handoff
> Updated 2026-03-05. Read this first in any new session.

## Project Location
`C:\Users\USUARIO\Dev\Python\camera-traps`

## Status
**Phase 1 is complete and working.**
- Zero-shot CLIP classification pipeline: `run_classification.py` ✓
- Human review Streamlit UI: `phase1_labeling/app.py` ✓
- First campaign classified and reviewed: Primavera 2025 ✓
- Output: `new_labeled_data_reviewed.csv` in campaign_dir ✓

**Phase 2 (custom EfficientNetV2) not started** — pending more reviewed data from future campaigns.

---

## How To Run

```bash
conda activate species-classifier
cd C:\Users\USUARIO\Dev\Python\camera-traps

# Step 1 — classify a new campaign
python run_classification.py

# Step 2 — human review
streamlit run phase1_labeling/app.py
# → http://localhost:8501
```

---

## What Was Built This Session

### Pipeline (`run_classification.py` + `classify_campaign/`)
- Reads MegaDetector v5b JSON (`timelapse_recognition_file.json`) + Timelapse2 `.ddb`
- Sources: MD detections ≥ 0.38 confidence + `DeleteFlag=true` images (MD misses)
- Crops each image to MegaDetector bbox (5% padding)
- CLIP zero-shot classification against 26 species (English prompts)
- Confidence threshold: scores < 0.28 → `"No reconocible"` instead of forced guess
- Output: CamtrapDP CSV with `observationType`, `scientificName`, `observationComments`, `classificationProbability`

### Review UI (`phase1_labeling/app.py`)
- Groups images by CLIP-proposed species
- Common species (≥31 records in `old animal data DB.csv`) get dedicated batch pages
- Rare species + low-confidence images → single "Otras especies" batch at the end
- Two confirm buttons: bulk confirm (all as proposed) or confirm with individual edits
- Tracks `reviewOutcome`: `"confirmed"` vs `"corrected"` per image
- Special options: "No es un animal", "No reconocible", "Otro (especificar)" (free text)
- Exports `new_labeled_data_reviewed.csv` — CamtrapDP format + `reviewOutcome` column

### Key Decisions Made
- **0.38 MegaDetector threshold**: 80% precision / 97% recall (user's research)
- **0.28 CLIP threshold**: bottom ~12% of score distribution; based on empirical analysis of Primavera 2025 scores
- **Guiña cutoff (31 records)**: species with fewer historical records don't get own batch page
- **GPU**: RTX 5060 Ti is Blackwell (sm_120) — requires PyTorch 2.10+cu128, NOT cu124
- **Wrong JSON warning**: `image_recognition_file.json` was from a tropical AddaxAI model — irrelevant, deleted

---

## Primavera 2025 Campaign — Current File State

**Location:** `C:\Users\USUARIO\SynologyDrive\2. Camaras trampa (SC)\SynologyDrive\DATOS_GRILLA CÁMARAS TRAMPA\2. CAMPAÑAS DE RECOLECCION DE IMAGENES\Primavera 2025\`

| File | What it is |
|---|---|
| `CamtrapDB_Primavera2025.ddb` | Timelapse2 SQLite database (source of truth) |
| `CamtrapDB_Primavera2025.tdb` | Timelapse2 template database |
| `timelapse_recognition_file.json` | MegaDetector v5b output — USE THIS |
| `new_labeled_data_CamptrapDP.csv` | Original 1,960-row CamtrapDP CSV (input) |
| `new_labeled_data_classified.csv` | CLIP classification output (400 classified) |
| `new_labeled_data_reviewed.csv` | Human-reviewed final output ← **main product** |

---

## Known CLIP Limitations

- **Forced-choice problem**: CLIP always picks the best match from the species list — even for garbage images. The 0.28 threshold catches the worst ~12% but doesn't solve the problem fully.
- **Diucón / generic bird issue**: Small dark birds in CLIP's embedding space attract ambiguous/dark images. A species-specific trained model (Phase 2) will fix this.
- **Solution**: Phase 2 custom EfficientNetV2 model trained on reviewed data.

---

## Next Steps

### Short term — more campaigns
1. Run `run_classification.py` + review UI for each new campaign
2. Accumulate reviewed CSVs across campaigns → build training dataset for Phase 2

### Medium term — Phase 2 classifier
Create `phase2_classifier/` directory with:
- `dataset.py` — PyTorch Dataset from reviewed CSV(s); crop + resize 224×224; augmentation for rare species
- `train.py` — EfficientNetV2-S via `timm`; two-stage (freeze backbone then unfreeze); class weights for imbalance
- `evaluate.py` — confusion matrix, per-species F1
- `classify.py` — inference CLI replacing CLIP in `run_classification.py`

Training data requirements: aim for ≥50 images per species before training. Currently have Primavera 2025 (400 images reviewed). Need more campaigns before Phase 2 is viable for rare species (Puma, Guiña, etc.).

### Possible short-term CLIP improvements (optional)
- **Better prompts**: more descriptive English text per species (e.g. size, colour, habitat context)
- **Prompt ensembling**: average embeddings from 3–5 different prompt templates per species
- Neither replaces Phase 2 but could improve intermediate results
