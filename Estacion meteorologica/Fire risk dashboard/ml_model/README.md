# ML Model Directory

This directory contains all machine learning components for the fire risk dashboard.

## Directory Structure

```
ml_model/
├── README.md                           # This file
├── QUICK_START_ML.md                   # Quick start guide
├── ML_IMPLEMENTATION_SUMMARY.md        # Complete documentation
│
├── prepare_training_data.py            # Step 1: Build training dataset
├── train_fire_model.py                 # Step 2: Train model
│
├── fire_model.pkl                      # Trained Random Forest model
├── training_data.csv                   # Training dataset (fires + non-fires)
│
├── data/                               # Raw data
│   └── cicatrices_incendios_resumen.geojson  # Historical fires from Dataverse
│
└── plots/                              # Evaluation plots
    ├── feature_importance.png          # Shows what ML learned
    ├── confusion_matrix.png            # Prediction accuracy breakdown
    └── roc_curve.png                   # Discrimination ability
```

## Quick Start

### 1. Run Dashboard (uses existing model)

From dashboard root:
```bash
conda activate fire_risk_dashboard
streamlit run app.py
```

The dashboard automatically loads `ml_model/fire_model.pkl` and displays ML predictions.

### 2. Retrain Model (from this directory)

```bash
cd ml_model

# Step 1: Prepare training data (~1-2 hours for full dataset)
python prepare_training_data.py

# Step 2: Train model (~5 minutes)
python train_fire_model.py

# Step 3: Return to root and restart dashboard
cd ..
streamlit run app.py
```

## Model Details

- **Algorithm**: Random Forest (100 trees)
- **Features**: temp_c, rh_pct, wind_kmh, days_no_rain
- **Training data**: 616 fires + 616 non-fires from Araucania (2015-2024)
- **Current performance**: 80% accuracy, 0.85 ROC AUC

## Configuration

Edit `prepare_training_data.py` to adjust:
- `MAX_SAMPLES`: Set to `None` for full dataset, or a number (e.g., 50) for quick testing
- `START_YEAR`, `END_YEAR`: Adjust time range
- `MIN_FIRE_SIZE_HA`: Filter by fire size

## Files Explanation

- **prepare_training_data.py**: Downloads historical fires, fetches weather data, creates balanced training set
- **train_fire_model.py**: Trains Random Forest, evaluates performance, saves model and plots
- **fire_model.pkl**: Trained model file loaded by main dashboard (`app.py`)
- **training_data.csv**: Processed training dataset (temp, humidity, wind, dry days + fire/no-fire label)
- **data/**: Raw fire dataset from Datos para Resiliencia
- **plots/**: Evaluation visualizations for presentations

## Integration with Dashboard

The main dashboard (`../app.py`) loads the model:

```python
model = joblib.load("ml_model/fire_model.pkl")
```

For each day, it predicts fire probability:

```python
probability = model.predict_proba([[temp, humidity, wind, dry_days]])[0][1]
```

Then displays alongside the rule-based risk index.

## See Also

- `QUICK_START_ML.md` — 5-minute test guide
- `ML_IMPLEMENTATION_SUMMARY.md` — Complete documentation with presentation tips
- `../README.md` — Main dashboard documentation

