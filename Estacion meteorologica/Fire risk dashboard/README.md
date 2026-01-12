Fire Risk Dashboard — Bosque Pehuén

Author: Felipe Guarda
Institution: Fundación Mar Adentro
Date: 2025
License: CC BY-NC 4.0

1. Overview

This project develops an interactive data visualization dashboard to estimate and communicate wildfire risk in Bosque Pehuén—Fundación Mar Adentro’s privately protected area located in the Andean foothills of southern Chile—and its surrounding landscape.
The dashboard integrates meteorological data obtained from the Open-Meteo API, an open-source global weather-model platform that provides hourly forecasts and recent historical data at user-defined coordinates under a CC BY 4.0 licence.
All data originate from large-scale numerical weather prediction models (e.g., ECMWF IFS, GFS, ICON) and represent modelled atmospheric conditions rather than direct in-situ sensor readings.

The system computes a composite wildfire-risk index using four environmental variables:

- air temperature (°C),
- relative humidity (%),
- wind speed (km/h), and
- consecutive days without precipitation > 2 mm.

The index combines standardized Chilean fire-danger weighting schemes for temperature, humidity, and wind speed, while the days-without-rain variable follows the approach proposed in Argentina’s Rodríguez-Moretti Index (IRM) (Dentoni & Muñoz, 2012).All calculations are implemented in Python and visualized interactively through the Streamlit framework.


2. Methodological basis
2.1 Data source

Meteorological inputs are retrieved via the Open-Meteo API, which provides time-series data for temperature, humidity, wind, and precipitation at the selected geographic coordinate.
The Bosque Pehuén study area is represented by its UTM 19S (WGS 84) location:
Easting = 263221 m, Northing = 5630634 m (≈ lat -39.61°, lon -71.71°).

2.2 Variables and scoring

Each variable is assigned a partial score according to predefined ranges, producing a total risk index from 0 to 100 points.
Humidity contributes inversely (high humidity → lower risk), while the remaining variables contribute directly.
The variables are weighted differently: temperature and humidity each contribute up to 25 points, wind speed up to 15 points, and days without rain up to 35 points.

Risk scores are computed for the afternoon hours (14:00–16:00 local time, 2–4 PM) when fire danger typically peaks, and these values are averaged to produce a single daily risk score per day.

| Variable          | Unit | Relationship with risk | Scoring range |
| ----------------- | ---- | ---------------------- | ------------- |
| Air temperature   | °C   | Positive               | 0–25          |
| Relative humidity | %    | Inverse                | 0–25          |
| Wind speed        | km/h | Positive               | 0–15          |
| Days without rain | days | Positive               | 0–35          |


2.3 Output interpretation
| Total score | Category  | Description                                       |
| ----------- | --------- | ------------------------------------------------- |
| 0–20        | Low       | Favorable conditions; minimal ignition potential. |
| 21–40       | Moderate  | Limited fire spread potential.                    |
| 41–60       | High      | Increased flammability; vigilance required.       |
| 61–80       | Very High | Rapid ignition and propagation possible.          |
| 81–100      | Extreme   | Critical conditions; suppression difficult.       |


3. Technical implementation
The dashboard is developed in Python 3.11 using open-source libraries:

| Library            | Function                                    |
| ------------------ | ------------------------------------------- |
| **Streamlit**      | User interface and app framework            |
| **Plotly**         | Interactive polar and time-series charts    |
| **Pydeck**         | Spatial visualization of regional wind patterns (uses Carto basemap) |
| **Pandas / NumPy** | Data management and computation             |
| **Requests**       | API calls to Open-Meteo                     |


The app executes locally or through Streamlit Cloud.
It automatically retrieves data from Open-Meteo, processes them, computes risk scores, and renders:

- a daily polar plot showing variable contributions,
- a tabular summary of scores,
- a multi-day risk forecast,
- a wind compass with wind rose-style wedge visualization (color-coded by risk level), and
- a regional map displaying wind flow patterns with geographic context.

3.1 Project structure and modular architecture

The codebase is organized into a modular architecture that separates concerns for maintainability, testability, and clarity. The application was refactored from a monolithic `app.py` into specialized modules, each handling a distinct responsibility.

**Module overview:**

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `app.py` | Main Streamlit application entry point | Orchestrates UI, data flow, and user interactions |
| `config.py` | Configuration constants and parameters | Scoring bins, risk color schemes, geographic bounds, timezone settings |
| `data_fetcher.py` | External API data retrieval | `fetch_open_meteo()` — fetches hourly/daily weather data with caching |
| `risk_calculator.py` | Fire risk computation logic | `risk_components()`, `compute_days_without_rain()`, `best_hour_by_day()`, `color_for_risk()` |
| `visualizations.py` | Chart and plot generation | `create_polar_plot()`, `create_wind_compass()`, `create_forecast_charts()` |
| `map_utils.py` | Regional map visualization utilities | `create_wind_flow_field()`, `create_map_layers()`, `create_map_view_state()` |

**Data flow:**

1. **Data fetching** (`data_fetcher.py`): Retrieves raw meteorological data from Open-Meteo API (hourly and daily forecasts)
2. **Risk calculation** (`risk_calculator.py`): Processes raw data to compute risk scores using configurable scoring bins. Hourly data from 14:00–16:00 (2–4 PM) is averaged per day to compute daily risk scores.
3. **Visualization** (`visualizations.py`): Generates interactive Plotly charts from computed risk data
4. **UI orchestration** (`app.py`): Coordinates all modules, manages Streamlit session state, and renders the dashboard

**Module dependencies:**

```
app.py
├── config.py (constants, scoring bins, geographic bounds)
├── data_fetcher.py (API data retrieval)
│   └── config.py (timezone)
├── risk_calculator.py (risk computation)
│   └── config.py (scoring bins, risk colors)
├── visualizations.py (chart generation)
│   └── risk_calculator.py (color_for_risk)
└── map_utils.py (map utilities)
    └── config.py (geographic bounds)
```

4. Environment and reproducibility

The project is distributed with a Conda environment file (environment.yml) ensuring reproducible execution across platforms.

To recreate the environment:

conda env create -f environment.yml
conda activate fire_risk_dashboard
streamlit run app.py

The file specifies Python 3.11 and the following key dependencies:
streamlit, plotly, pydeck, pandas, numpy, and requests.


5. References

- Dentoni, M. C. & Muñoz, M. M. (2012). Sistemas de evaluación de peligro de incendios. Informe Técnico Nº 1. Plan Nacional de Manejo del Fuego – Programa Nacional de Evaluación de Peligro de Incendios y Alerta Temprana. Esquel, Chubut, Argentina. ISSN 2313-9420.
- Open-Meteo (2022–2025). Weather Forecast API. Retrieved from https://open-meteo.com/en/docs
- Gil-Romera, G., et al. (2023). “Environmental Forest Fire Danger Rating Systems and Indices Around the Globe: A Review.” Land, 12(1), 194. https://doi.org/10.3390/land12010194

6. Recent updates (2025)

**Regional wind map:**
- Fixed base map display issue by switching from Mapbox (requires API key) to Carto basemap (`carto-positron` style)
- The map now displays the geographic context of the Araucania region without requiring external API key configuration
- Displays wind flow field visualization with directional fading to show wind patterns
- Overlay layers (wind streamlines, Bosque Pehuén marker) provide geographic context

**Wind compass visualization enhancements:**
- Replaced arrow-based wind direction indicator with wind rose-style wedge visualization
- The wedge extends from the center of the compass in the direction the wind is coming from, following standard meteorological convention
- Wind compass wedge color now dynamically matches the fire risk index color for the selected day, providing visual consistency across the dashboard
- The wedge uses a 30-degree angular width and extends up to 90% of the compass radius (scaled by wind speed) for clear visibility

**Technical details:**
- Map basemap: Uses `pdk.map_styles.LIGHT` (Carto basemap) in `app.py`
- Wind compass: `create_wind_compass()` function in `visualizations.py` accepts `risk_color` parameter and renders wedge visualization
- Color integration: Wind compass receives the risk color calculated from `color_for_risk()` function, ensuring visual consistency with the risk index display
- Map visualization: `create_wind_flow_field()` in `map_utils.py` generates directional wind streamlines with opacity gradients


7. Machine Learning Fire Prediction

This dashboard includes a **Random Forest classifier** trained on 20 years of Chilean fire history to provide data-driven fire probability predictions.

### How it works

The ML model uses the **same 4 weather variables** as the rule-based risk index:
- Temperature (°C)
- Relative humidity (%)
- Wind speed (km/h)
- Consecutive days without rain

### Training data

- **Source**: Historical fire scar dataset from Datos para Resiliencia (`doi:10.71578/XAZAKP`)
- **Region**: La Araucanía (same as dashboard focus area)
- **Time period**: 2015-2024
- **Samples**: 616 fires + 616 non-fire days (balanced dataset)
- **Features**: Afternoon weather conditions (14:00-16:00) matched to fire dates

### Model performance

- **Accuracy**: 80% on test set
- **Cross-validation**: 81% (±8%)
- **ROC AUC**: 0.85 (good discrimination)

### Feature importance

The model learned that these variables matter most for fire prediction:
1. **Days without rain**: 38% importance (most critical)
2. **Temperature**: 37% importance
3. **Wind speed**: 17% importance
4. **Relative humidity**: 8% importance

This validates the Chilean fire danger standards used in the rule-based scoring system!

### Dashboard integration

The ML prediction appears in the dashboard alongside the rule-based risk index:
- **Rule-Based Risk Index**: Uses Chilean fire danger standards (fixed scoring bins)
- **ML Fire Probability**: Learned from actual fire patterns (data-driven)
- **Agreement indicator**: Shows when both methods agree (within 15 points)

### Scripts for ML workflow

- `prepare_training_data.py` - Downloads fire data and fetches historical weather to build training dataset
- `train_fire_model.py` - Trains Random Forest model and generates evaluation plots
- `fire_model.pkl` - Trained model (loaded automatically by dashboard)

### Retraining the model

To retrain with more data or different parameters:

```bash
# 1. Prepare training data (adjust MAX_SAMPLES in script if needed)
python prepare_training_data.py

# 2. Train model
python train_fire_model.py

# 3. Review evaluation plots:
#    - feature_importance.png
#    - confusion_matrix.png
#    - roc_curve.png

# 4. Restart dashboard to load new model
streamlit run app.py
```

8. Data download utility (pyDataverse)

This project includes a helper script (`download_dataverse.py`) for downloading datasets from **Datos para Resiliencia** (https://datospararesiliencia.cl) using the pyDataverse library.

### Installation

```bash
pip install pyDataverse==0.3.3
```

This package is already included in the `fire_risk_dashboard` conda environment.

### Obtaining an API key

1. Visit https://datospararesiliencia.cl
2. Log in or create an account
3. Navigate to your user profile → Developer Tools
4. Generate an API key

### Usage examples

**List available files in a dataset:**

```bash
python download_dataverse.py --api-key YOUR_API_KEY --dataset-id "doi:10.71578/XAZAKP" --list-files
```

**Download all files as a single zip:**

```bash
python download_dataverse.py --api-key YOUR_API_KEY --dataset-id "doi:10.71578/XAZAKP" --download-all
```

**Download all files individually:**

```bash
python download_dataverse.py --api-key YOUR_API_KEY --dataset-id "doi:10.71578/XAZAKP" --download-all --individual
```

**Download specific files:**

```bash
python download_dataverse.py --api-key YOUR_API_KEY --dataset-id "doi:10.71578/XAZAKP" --files "filename1.csv" "filename2.geojson"
```

**Using environment variables (recommended for security):**

In bash (Git Bash / MSYS):
```bash
export DATAVERSE_API_KEY="your-api-key-here"
export DATAVERSE_DATASET_ID="doi:10.71578/XAZAKP"
python download_dataverse.py --download-all
```

In PowerShell:
```powershell
$env:DATAVERSE_API_KEY="your-api-key-here"
$env:DATAVERSE_DATASET_ID="doi:10.71578/XAZAKP"
python download_dataverse.py --download-all
```

### Default dataset

The script defaults to dataset `doi:10.71578/XAZAKP` (Chilean fire scar database) from Datos para Resiliencia. You can override this with `--dataset-id` or the `DATAVERSE_DATASET_ID` environment variable.

For more information, see the script's built-in help:

```bash
python download_dataverse.py --help
```


8. Future development

- Integration of local station data to compare observed vs. modelled conditions.
- Automated data archiving and alert system for high-risk thresholds.
- Multi-year analysis of historical trends using ERA5 reanalysis data.
- Enhanced regional visualization capabilities as needed.