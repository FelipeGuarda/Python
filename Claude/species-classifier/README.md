# Clasificador de Especies — FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Status:** Phase 1 complete. Phase 2 (custom model) pending.
**Conda env:** `species-classifier` (Python 3.11, PyTorch 2.10+cu128, transformers, streamlit)

---

## What This Project Does

A two-phase pipeline to classify animals in FMA's camera trap images.

**Phase 1 — Zero-shot CLIP classification + human review UI:** Automatically classifies images using CLIP (`openai/clip-vit-base-patch32`) with zero-shot text prompts, then presents results in a Streamlit app for rapid human review and correction. Outputs a CamtrapDP-format CSV ready for analysis.

**Phase 2 — Custom EfficientNetV2 classifier:** Train a fine-tuned model on the reviewed labeled data from Phase 1. Will replace CLIP for future campaigns, providing faster and more accurate species-specific classification without human review.

---

## Current File Structure

```
species-classifier/
├── config.yaml                    ← all paths, thresholds, species list
├── run_classification.py          ← Phase 1 entry point: classify a campaign
├── environment.yml                ← conda environment definition
├── old animal data DB.csv         ← 18,473 rows of historical labeled data (reference)
│
├── classify_campaign/             ← classification pipeline modules
│   ├── data_loader.py             ← loads MegaDetector JSON + .ddb → image list
│   ├── clip_classifier.py         ← zero-shot CLIP classifier
│   └── crop_utils.py              ← crops PIL images to MegaDetector bboxes
│
└── phase1_labeling/
    └── app.py                     ← Streamlit human review UI
```

---

## How To Run — Phase 1

### 1. Classify a campaign

```bash
conda activate species-classifier
cd C:\Users\USUARIO\Dev\Python\Claude\species-classifier
python run_classification.py
# or with explicit config:
python run_classification.py --config config.yaml
```

Reads from `config.yaml`:
- `campaign_dir` — path to campaign folder
- `megadetector_json` — MegaDetector v5b output JSON (must be `timelapse_recognition_file.json`, NOT `image_recognition_file.json`)
- `database` — Timelapse2 `.ddb` SQLite database
- `input_csv` — CamtrapDP CSV (`new_labeled_data_CamptrapDP.csv`)
- `animal_confidence_threshold` — MegaDetector confidence cutoff (0.38 = 80% precision, 97% recall)
- `clip_confidence_threshold` — CLIP cosine similarity cutoff (0.28); images below this → `No reconocible`

Outputs `new_labeled_data_classified.csv` in `campaign_dir`.

### 2. Human review

```bash
streamlit run phase1_labeling/app.py
# open http://localhost:8501
```

- Groups classified images into batches by species
- Common species (≥31 records in historical data) get dedicated batch pages
- Rare/low-confidence images grouped into "Otras especies" batch
- Per-image dropdowns for corrections; tracks `reviewOutcome` (confirmed / corrected)
- Exports `new_labeled_data_reviewed.csv` — same CamtrapDP format + `reviewOutcome` column

---

## Classification Workflow Detail

```
campaign_dir/
    timelapse_recognition_file.json   ← MegaDetector v5b detections
    CamtrapDB_Primavera2025.ddb       ← Timelapse2 SQLite DB (has DeleteFlag)
    new_labeled_data_CamptrapDP.csv   ← 1,960-row input (all unclassified)
         ↓
data_loader.py:
    → MegaDetector detections with category='1' AND conf ≥ 0.38  (396 images)
    → DataTable rows where DeleteFlag='true' (animals MD missed)  (  4 images)
    → skip videos (fileMediatype contains 'video')
    = 400 images to classify
         ↓
clip_classifier.py:
    → crop each image to MegaDetector bbox (5% padding)
    → encode with CLIP ViT-B/32 on CUDA (RTX 5060 Ti, sm_120, PyTorch cu128)
    → cosine similarity vs. 26 species text embeddings
    → if best score < 0.28 → "No reconocible"
    → else → best-matching species (Spanish name + Latin binomial)
         ↓
run_classification.py:
    → writes results back into new_labeled_data_CamptrapDP.csv rows via filePath
    → output: new_labeled_data_classified.csv (same 1,960 rows; 400 classified)
```

---

## config.yaml — Key Parameters

| Key | Value | Notes |
|---|---|---|
| `animal_confidence_threshold` | 0.38 | MegaDetector cutoff (80% precision, 97% recall) |
| `clip_confidence_threshold` | 0.28 | CLIP cutoff; below → "No reconocible" (bottom ~12% of scores) |
| `min_historical_count` | 31 | Species with fewer records in old DB get no dedicated batch page |
| `clip_model` | `openai/clip-vit-base-patch32` | Cached locally after first download |

---

## Important Notes

- **MegaDetector JSON:** Always use `timelapse_recognition_file.json`. The file `image_recognition_file.json` (now deleted) was output from a wrong tropical-species AddaxAI model — it is irrelevant to Chilean fauna.
- **GPU:** RTX 5060 Ti (Blackwell sm_120) requires PyTorch ≥ 2.10+cu128. Do NOT install cu124 builds.
- **CLIP limitation:** Zero-shot classification always forces a choice from the species list. Low-confidence threshold (0.28) catches the worst cases but CLIP accuracy is inherently limited. The custom Phase 2 model will address this.
- **CamtrapDP format:** Output CSVs follow CamtrapDP standard. `observationComments` = Spanish name, `scientificName` = Latin binomial.

---

## Phase 2 — Custom EfficientNetV2 Classifier (Planned)

Once enough reviewed data is accumulated across campaigns:

```
new_labeled_data_reviewed.csv  (from Phase 1 review sessions)
    ↓
phase2_classifier/dataset.py   ← PyTorch Dataset: crop + resize 224×224 + augmentation
phase2_classifier/train.py     ← fine-tune EfficientNetV2-S (timm), class weights for imbalance
phase2_classifier/evaluate.py  ← confusion matrix, per-species F1, error analysis
phase2_classifier/classify.py  ← inference CLI: folder → CSV
```

Key decisions:
- Always crop to MegaDetector bbox before training/inference
- Class weights mandatory (Puma: 54 records vs Perro: 557)
- Two-stage training: freeze backbone → train head, then unfreeze all
- Target: replace CLIP in `run_classification.py` with the trained model

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| GPU | RTX 5060 Ti (Blackwell sm_120) — requires PyTorch cu128 |
| Animal detection | MegaDetector v5b (pre-run, output in `timelapse_recognition_file.json`) |
| Phase 1 classifier | CLIP `openai/clip-vit-base-patch32` via HuggingFace `transformers` |
| Phase 1 review UI | Streamlit |
| Phase 2 model | EfficientNetV2-S via `timm` (planned) |
| Training framework | PyTorch (planned) |
| Data format | CamtrapDP (CSV) + Timelapse2 `.ddb` SQLite |
| Environment | Conda (`species-classifier`) |
