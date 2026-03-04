# Species-Classifier — Discoveries & Next Steps
> Generated 2026-03-03. Use this file to resume work in a new YOLO-mode chat.

## Project Location
`C:\Users\USUARIO\Dev\Python\Claude\species-classifier`

## Status
**Nothing is built yet.** Only the plan (README.md), conda environment definition, and one labeled CSV exist.
This is a greenfield project — start from scratch following the plan.

---

## What Exists Right Now

| File | What it is |
|------|-----------|
| `README.md` | Full design document — read it. Has data flow, UI mockup, file structure plan, design decisions |
| `environment.yml` | Conda env `species-classifier` — **not yet created on this machine** |
| `old animal data DB.csv` | 18,473 rows of labeled camera trap data (see below) |

---

## Existing Labeled Data — Key Facts

**File:** `old animal data DB.csv`
**Rows:** 18,473 total
**Columns:** `RootFolder, File, RelativePath, DateTime, Animal, Person, Especie, Counter0, Note0`

**Species distribution** (only rows with `Especie` filled in — animal detections):

| Count | Species | Notes |
|-------|---------|-------|
| 557 | Perro | Non-target — flag, don't misclassify |
| 557 | Jabali | Invasive priority |
| 538 | Caballo | Non-target |
| 316 | Zorro culpeo | Native predator priority |
| 169 | Liebre | Invasive |
| 54 | Puma | Native predator priority — rare |
| 51 | Chucao | Native bird |
| 31 | Guiña | Native predator priority — rare |
| 20 | Zorzal | |
| 20 | Hued hued | |
| 18 | Bandurria | |
| 14 | Gato | |
| 12 | Ratón cola larga | |
| 11 | Queltehue | |
| 7 | Visón | Invasive priority |
| 6 | Monito del monte | |
| 4 | Carpintero | |
| 4 | Tiuque | |
| 3 | Peuquito | |
| 3 | Rayadito | |
| 3 | Concón | |
| 3 | Cometocino | |
| 2 | Picaflor | |
| 2 | Diucón | |
| 1 | Traro | |

**Critical notes:**
- The CSV has **encoding issues** — some Spanish characters (ñ, ó, í) are corrupted when read with wrong encoding. Use `encoding='utf-8-sig', errors='replace'` or detect actual encoding first.
- Rows where `Especie` is blank are non-animal triggers (person, empty frame) — filter them out.
- `Animal=No` rows = no animal detected — also filter out.
- **Extreme class imbalance**: top 3 classes (~550 each), Traro has 1. Must use class weights or oversampling.
- Image paths in the CSV are relative to some root in `Camaras trampa/` — need to resolve actual image locations before building anything.

---

## Actual Image Data Location

The CSV has `RootFolder=Fotos` and paths like `2022\Araucarias\CT10_06_12_22`. Actual images are in a `Camaras trampa/` directory (separate project). Check:
- `C:\Users\USUARIO\Dev\Python\Claude\camera-trap-analyzer\` — may have Megadetector bounding box JSONs
- Search for the `Camaras trampa\` root folder on the filesystem

**Before building anything:** verify you can resolve a full image path from a CSV row. Nothing works without images.

---

## Planned File Structure (from README — nothing created yet)

```
species-classifier/
├── config.yaml                    <- create first
├── phase1_labeling/
│   ├── embed.py                   <- CLIP embedding of cropped chips
│   ├── cluster.py                 <- UMAP + HDBSCAN clustering
│   └── app.py                     <- Streamlit labeling UI
├── phase2_classifier/
│   ├── dataset.py                 <- PyTorch Dataset from labeled CSV
│   ├── train.py                   <- EfficientNetV2 fine-tuning
│   ├── evaluate.py                <- confusion matrix, per-species metrics
│   └── classify.py                <- inference CLI
├── data/
│   ├── labeled_dataset.csv        <- output of Phase 1
│   └── model/                     <- saved weights
└── notebooks/
    ├── explore_embeddings.ipynb
    └── evaluate_model.ipynb
```

---

## Tech Stack

| What | Tool |
|------|------|
| Embeddings (Phase 1) | CLIP `openai/clip-vit-base-patch32` via `transformers` or `clip` package |
| Dimensionality reduction | UMAP (`umap-learn`) |
| Clustering | HDBSCAN or K-means (`hdbscan`, `scikit-learn`) |
| Labeling UI | Streamlit |
| Classifier (Phase 2) | EfficientNetV2-S via `timm` |
| Training framework | PyTorch |
| Detection pre-step | Megadetector v6 (`wildlife_detector`) — already working in another project |
| Environment | Conda (`species-classifier` env — not created yet) |

---

## Key Design Decisions (don't deviate)

1. **Always crop to Megadetector bounding box first** — never embed or train on the full image.
2. **CLIP for Phase 1 only** — zero-shot clustering, no training data needed.
3. **Class weights mandatory** — dataset is extremely imbalanced (Puma 54, Traro 1 vs Perro 557).
4. **Two phases are independent** — Phase 1 generates `labeled_dataset.csv`, Phase 2 trains on it.
5. **Species names are in Spanish** — keep them as-is in all CSVs and model outputs.

---

## Execution Order for Next Session

### Step 0 — Verify prerequisites (5 min)
- [ ] Create the conda env: `conda env create -f environment.yml`
- [ ] Find where the actual camera trap images live (resolve a path from `old animal data DB.csv`)
- [ ] Check `camera-trap-analyzer/` for Megadetector bounding box JSONs for the same dataset

### Step 1 — Fix the existing data CSV (15 min)
- [ ] Write a script/notebook to audit `old animal data DB.csv`:
  - Detect and fix encoding issues (Guiña, Ratón, Concón, etc.)
  - Resolve full image paths
  - Filter to rows with valid `Especie` and `Animal=Yes`
  - Export clean `data/labeled_dataset_raw.csv`

### Step 2 — Build `phase1_labeling/embed.py` (30 min)
- [ ] Input: image directory + optional Megadetector bbox JSON
- [ ] Crop each image to bbox (full image fallback if no bbox)
- [ ] Run CLIP on each crop
- [ ] Save embeddings to `data/embeddings.npz` (image paths as keys)

### Step 3 — Build `phase1_labeling/cluster.py` (20 min)
- [ ] Load `embeddings.npz`
- [ ] UMAP to 2D
- [ ] HDBSCAN clustering
- [ ] Save cluster assignments to `data/clusters.csv`

### Step 4 — Build `phase1_labeling/app.py` (45 min)
- [ ] Streamlit: shows one cluster at a time as thumbnail grid
- [ ] Species dropdown (from existing species list)
- [ ] "Confirm all" -> writes labels to `data/labeled_dataset.csv`
- [ ] Per-image override before confirming
- [ ] Skip cluster option
- [ ] Cluster N of M counter in header

### Step 5 — Phase 2 (after enough labeled data)
- `dataset.py` -> PyTorch Dataset, crop + resize 224x224, augmentation for rare species
- `train.py` -> two-stage: freeze backbone + train head, then unfreeze all
- `evaluate.py` -> confusion matrix, per-species F1
- `classify.py` -> CLI: `python classify.py --input_dir /photos/ --output_csv results.csv`

---

## Future Features (from README — implement later)
- Active learning: use model confidence to prioritize hard cases
- Integration with data-pipeline -> write labels to DuckDB
- Confidence threshold UI in labeling app (skip high-confidence predictions)
- Multi-species per frame (low priority, requires different output head)
