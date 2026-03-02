# Clasificador de Especies — FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Status:** Planned / Not yet built.
**Depends on:** Existing labeled camera trap data from `Camaras trampa/` project.

---

## What This Project Does

A two-phase project to build a custom species classifier for FMA's camera trap images.

**Phase 1 — Labeling speed-up app:** A tool that clusters unlabeled camera trap images using CLIP embeddings and presents them in visually similar batches for rapid human labeling. Goal: label 10x faster than reviewing one image at a time.

**Phase 2 — Custom classifier:** Train a fine-tuned EfficientNetV2 model on the labeled dataset. Deploy it so future camera trap campaigns can be auto-classified without Timelapse2 manual review.

---

## Why Two Phases

The bottleneck today is labeling. Megadetector v6 detects *that there's an animal* but doesn't identify the species. A human must then look at each photo in Timelapse2 and assign a species label. For large campaigns (thousands of photos), this takes many hours.

Phase 1 makes labeling faster by showing visually similar images together — you label one and confirm the rest in the cluster. Phase 2 eliminates the labeling step for future campaigns by using the accumulated labeled data to train a species-specific model.

---

## Current State

- **Not yet built.**
- Existing labeled data: CSVs in `Camaras trampa/` with species labels for Bosque Pehuén campaigns
- Species in existing dataset: ~26 species (see camera-trap-analyzer/README.md for full list)
- Megadetector v6 is already integrated (`Megadetector_para_campañas.py`) — this project builds on top of it

---

## Phase 1 — Labeling Speed-Up App

### How It Works

```
Unlabeled JPG images (from field campaign)
    ↓
Megadetector v6 → crop bounding boxes → individual animal chips
    ↓
CLIP (ViT-B/32) → encode each chip → 512-dim embedding
    ↓
UMAP → reduce to 2D for visualization
K-means or HDBSCAN → cluster by visual similarity
    ↓
Streamlit app:
    - Show one cluster at a time (grid of thumbnails)
    - User types species name once → applies to all images in cluster
    - User can override individual images before confirming
    - Confirmed labels saved to CSV
    ↓
Output: labeled_dataset.csv with {image_path, species, confidence: "human"}
```

### UI Design (Streamlit)

```
┌─────────────────────────────────────────────┐
│  Cluster 12 of 47  │  Suggested: Zorro culpeo│
│                                              │
│  [img] [img] [img] [img] [img]              │
│  [img] [img] [img] [img] [img]              │
│                                              │
│  Species: [Zorro culpeo        ▼]           │
│  [✓ Confirm all]  [Skip cluster]            │
│  [Override individual images...]            │
└─────────────────────────────────────────────┘
```

---

## Phase 2 — Custom EfficientNetV2 Classifier

### How It Works

```
labeled_dataset.csv + corresponding JPG files
    ↓
Preprocessing:
    - Crop to Megadetector bounding box
    - Resize to 224×224
    - Augmentation (flip, brightness, blur) for rare species
    ↓
Fine-tune EfficientNetV2-S (pretrained on ImageNet):
    - Freeze backbone → train head (10 epochs)
    - Unfreeze → fine-tune full model (20 epochs)
    - Class weights for imbalanced species
    ↓
Evaluate: per-species accuracy, confusion matrix, F1
    ↓
Export: model.pt (PyTorch) or model.h5 (Keras/TF)
    ↓
Deploy:
    - CLI script: python classify.py --input_dir /photos/ --output_csv results.csv
    - Or: integrate into data-pipeline as an auto-classification step
```

### Target Species (priority)

Focus classification on ecologically important species. Rare/ambiguous species may be left as "unclassified" for human review:

| Category | Species |
|---|---|
| Native predators | Puma, Guiña, Zorro culpeo |
| Invasive mammals | Jabali, Visón americano, Liebre europea, Rata negra |
| Native birds | Chucao, Concón, Carpintero, Picaflor, Rayadito |
| Other wildlife | Monito del monte, Chingue, Hued hued, Ciervo rojo |
| Non-target | Caballo, Perro (domestic animals — flag but don't misclassify) |

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Language | Python 3.11 |  |
| Phase 1 — Embedding | CLIP (`openai/clip-vit-base-patch32`) | Via `transformers` or `clip` package |
| Phase 1 — Clustering | UMAP + HDBSCAN (or K-means) | `umap-learn`, `hdbscan` |
| Phase 1 — UI | Streamlit | Labeling app |
| Phase 2 — Model | EfficientNetV2-S | `timm` or `tensorflow.keras.applications` |
| Phase 2 — Training | PyTorch + `timm`, or TF/Keras |  |
| Phase 2 — Experiment tracking | MLflow (optional) | Track accuracy across runs |
| Detection (pre-step) | Megadetector v6 (`wildlife_detector`) | Already working |
| Environment | Conda |  |

---

## File Structure (planned)

```
species-classifier/
├── config.yaml                    ← species list, model paths, training params
├── requirements.txt
│
├── phase1_labeling/
│   ├── embed.py                   ← CLIP embedding of all cropped chips
│   ├── cluster.py                 ← UMAP + clustering
│   └── app.py                     ← Streamlit labeling UI
│
├── phase2_classifier/
│   ├── dataset.py                 ← PyTorch Dataset class from labeled CSV
│   ├── train.py                   ← fine-tuning script
│   ├── evaluate.py                ← confusion matrix, per-species metrics
│   └── classify.py                ← inference CLI: folder of images → CSV
│
├── data/
│   ├── labeled_dataset.csv        ← output of Phase 1 labeling
│   └── model/                     ← saved model weights
│
└── notebooks/
    ├── explore_embeddings.ipynb   ← UMAP cluster visualization
    └── evaluate_model.ipynb       ← confusion matrix, error analysis
```

---

## Ideas & Future Features

- **Active learning**: After Phase 2 model is trained, use it to identify the images it's *least confident* about and prioritize those for human review — iteratively improve accuracy with minimal labeling effort
- **Integration with data-pipeline**: Auto-classify new camera trap batches as they arrive, write species labels directly to DuckDB
- **Confidence threshold UI**: In the Streamlit labeling app, show the model's prediction + confidence — user only reviews low-confidence cases
- **Multi-species detection**: Handle images with multiple animals in frame (requires more complex output head)

---

## Key Design Decisions

1. **CLIP for Phase 1, not the final model**: CLIP's zero-shot visual embeddings are excellent for clustering without any labeled data. The clusters are often semantically meaningful (similar-looking animals group together) even without training.
2. **EfficientNetV2 for Phase 2**: Strong accuracy/compute tradeoff, well-supported by `timm`, good results on fine-grained classification tasks with limited data.
3. **Crop to bounding box before embedding/training**: Training on full images would force the model to also learn to locate the animal. Cropping (using Megadetector boxes) lets the classifier focus purely on species identity.
4. **Class weights for imbalance**: Puma and Guiña sightings are rare vs. common species like Ratón cola larga. Class weighting is essential to prevent the model from ignoring rare priority species.

---

## Context for AI Sessions

When starting a new Claude session on this project:

1. This is a **two-phase ML project**: Phase 1 = labeling tool (CLIP + Streamlit), Phase 2 = classifier (EfficientNetV2 fine-tuning)
2. Input data: JPG images from camera traps + Megadetector bounding boxes JSON
3. Existing labeled data is in `Camaras trampa/` CSVs — species names are in Spanish
4. The key pre-step is always: run Megadetector → crop bounding boxes → then embed or classify the *crop*, not the full image
5. CLIP model: `openai/clip-vit-base-patch32` via HuggingFace `transformers` or the original OpenAI `clip` package
6. Target: ~26 species, highly imbalanced dataset (some species have 10 records, others have 500+)
7. See `camera-trap-analyzer/README.md` for the full species list and data schema
