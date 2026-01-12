# Fire Risk Dashboard — Directory Structure

## Overview

Clean, organized structure separating core dashboard code from ML components.

## Root Directory

```
Fire risk dashboard/
│
├── Core Dashboard Files
│   ├── app.py                          # Main Streamlit dashboard
│   ├── config.py                       # Configuration & scoring bins
│   ├── data_fetcher.py                 # Open-Meteo API integration
│   ├── risk_calculator.py              # Fire risk computation
│   ├── visualizations.py               # Charts & plots
│   └── map_utils.py                    # Regional wind map
│
├── Configuration
│   ├── requirements.txt                # Python dependencies
│   ├── environment.yml                 # Conda environment
│   └── .gitignore                      # Git exclusions
│
├── Documentation
│   ├── README.md                       # Main project documentation
│   ├── DIRECTORY_STRUCTURE.md          # This file
│   └── Final_Project.pdf               # Academic paper
│
├── Data Tools
│   └── download_dataverse.py           # Download Chilean fire data
│
├── Documents/                          # Academic papers & presentations
│   ├── final_paper.md
│   ├── Guarda_final_paper.docx
│   └── ... (presentations, bibliography, etc.)
│
└── ml_model/                           # Machine Learning components
    ├── README.md                       # ML documentation
    ├── QUICK_START_ML.md               # Quick guide
    ├── ML_IMPLEMENTATION_SUMMARY.md    # Complete ML docs
    │
    ├── Scripts
    │   ├── prepare_training_data.py    # Build training dataset
    │   └── train_fire_model.py         # Train & evaluate model
    │
    ├── Model & Data
    │   ├── fire_model.pkl              # Trained Random Forest
    │   └── training_data.csv           # Training dataset
    │
    ├── data/                           # Raw fire data
    │   └── cicatrices_incendios_resumen.geojson
    │
    └── plots/                          # Evaluation plots
        ├── feature_importance.png
        ├── confusion_matrix.png
        └── roc_curve.png
```

## File Counts

- **Core dashboard**: 6 files (app.py + 5 modules)
- **ML components**: All in `ml_model/` subdirectory
- **Documentation**: Clear separation between main docs and ML docs
- **Data**: Organized in `ml_model/data/` and `ml_model/plots/`

## Running the Dashboard

From root directory:

```bash
conda activate fire_risk_dashboard
streamlit run app.py
```

The dashboard automatically loads `ml_model/fire_model.pkl` for ML predictions.

## Working with ML Components

From root directory:

```bash
cd ml_model

# Retrain model
python prepare_training_data.py  # Step 1: Get data
python train_fire_model.py       # Step 2: Train

# Return to root
cd ..
streamlit run app.py
```

## Benefits of This Structure

✅ **Clean root directory** — Only core dashboard files visible
✅ **Organized ML components** — All in one subdirectory
✅ **Easy navigation** — Clear separation of concerns
✅ **Git-friendly** — Large data files in `ml_model/data/` (gitignored)
✅ **Scalable** — Easy to add more ML experiments in `ml_model/`

## Path Updates Made

All import paths were updated to reference the new structure:

- `app.py` → loads model from `ml_model/fire_model.pkl`
- `prepare_training_data.py` → reads/writes to `ml_model/data/` and `ml_model/`
- `train_fire_model.py` → saves plots to `ml_model/plots/`

Everything works transparently — no manual path configuration needed!

