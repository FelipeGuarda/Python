# Fire Risk Dashboard — Project Structure

## Overview

Fire risk visualization dashboard for Bosque Pehuén, integrating rule-based fire danger indices with machine learning predictions trained on historical Chilean fire data.


## Core Application Files

### Main Dashboard
- **`app.py`** (489 lines) — Streamlit application entry point

### Configuration & Data
- **`config.py`** (81 lines) — Configuration constants

- **`data_fetcher.py`** (74 lines) — API data retrieval

### Risk Calculation
- **`risk_calculator.py`** (111 lines) — Fire risk computation

### Visualization
- **`visualizations.py`** (391 lines) — Chart generation

- **`map_utils.py`** (191 lines) — Regional map utilities


## Machine Learning Components

### Directory: `ml_model/`

#### Training Scripts
- **`prepare_training_data.py`** (352 lines)
  - Outputs: `training_data.csv`

- **`train_fire_model.py`** (280 lines)
  - Outputs: `fire_model.pkl`, evaluation plots

- **`validate_model_agreement.py`** (452 lines)
  - Outputs: `validation_results.json`, Bland-Altman plot

#### Model Artifacts
- **`fire_model.pkl`** — Trained Random Forest model

- **`validation_results.json`** — Statistical validation results

#### Evaluation Plots (`plots/`)
- **`feature_importance.png`** — Shows ML learned priorities
- **`confusion_matrix.png`** — Prediction accuracy breakdown
- **`roc_curve.png`** — Model discrimination ability
- **`bland_altman.png`** — Agreement analysis plot

#### Data (gitignored)
- **`data/cicatrices_incendios_resumen.geojson`** — Historical fires
- **`training_data.csv`** — Processed training dataset (1,230 samples)


## Utilities

- **`download_dataverse.py`** (241 lines)
  - Helper script for downloading datasets from Datos para Resiliencia

## Documentation

- **`README.md`** — Main project documentation

- **`ml_model/README.md`** — ML-specific documentation

## Configuration Files

- **`requirements.txt`** — Python dependencies

- **`environment.yml`** — Conda environment specification

- **`.gitignore`** — Version control exclusions

## Academic Deliverables (`Documents/`)

- `Guarda_final_paper.pdf`
- `Guarda_proposal.pdf`
- `Guarda_proposal_revised.pdf`
- `Guarda_bibliography.pdf`
- `Guarda_video_presentation.mp4`
- `PPT video presentation.pptx`

## Running the Dashboard

```bash
# Install dependencies
conda env create -f environment.yml
conda activate fire_risk_dashboard

# Run dashboard
streamlit run app.py
```

The dashboard will:
1. Load the trained ML model
2. Fetch current weather data from Open-Meteo
3. Compute rule-based risk scores
4. Generate ML fire probability predictions
5. Display both methods with agreement indicator



## License

CC BY-NC 4.0

## Author

Felipe Guarda  
Fundación Mar Adentro  
2026
