Fire Risk Dashboard — Bosque Pehuén

Author: Felipe Guarda
Institution: Fundación Mar Adentro
Date: 2026
License: CC BY-NC 4.0

1. Overview

This project develops an interactive data visualization dashboard to estimate and communicate wildfire risk in Bosque Pehuén —Fundación Mar Adentro's privately protected area located in the Andean foothills of southern Chile— and its surrounding landscape.

The dashboard integrates meteorological data obtained from the Open-Meteo API, an open-source global weather-model platform that provides hourly forecasts and recent historical data at user-defined coordinates under a CC BY 4.0 licence. All data originate from large-scale numerical weather prediction models (e.g., ECMWF IFS, GFS, ICON) and represent modelled atmospheric conditions rather than direct in-situ sensor readings.

The system employs two complementary approaches for risk assessment: a rule-based index following Chilean fire danger standards, and a machine learning model trained on historical fire data from the Araucanía region. Both methods use the same four environmental variables:

- air temperature (°C),
- relative humidity (%),
- wind speed (km/h), and
- consecutive days without precipitation > 2 mm.

The rule-based index combines standardized fire-danger weighting schemes for temperature, humidity, and wind speed, while the days-without-rain variable follows the approach proposed in Argentina's Rodríguez-Moretti Index (IRM) (Dentoni & Muñoz, 2012). The machine learning component provides data-driven validation using a Random Forest classifier trained on 616 historical fires from La Araucanía (1984-2018). All calculations are implemented in Python and visualized interactively through the Streamlit framework.


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
| Total score | Category  |
| ----------- | --------- |
| 0–20        | Low       |
| 21–40       | Moderate  |
| 41–60       | High      |
| 61–80       | Very High |
| 81–100      | Extreme   |


3. Technical implementation
The dashboard is developed in Python 3.11 using open-source libraries:

| Library            | Function                                    |
| ------------------ | ------------------------------------------- |
| **Streamlit**      | User interface and app framework            |
| **Plotly**         | Interactive polar and time-series charts    |
| **Pydeck**         | Spatial visualization of regional wind patterns (uses Carto basemap) |
| **Pandas / NumPy** | Data management and computation             |
| **Requests**       | API calls to Open-Meteo                     |
| **Scikit-learn**   | Machine learning (Random Forest classifier) |
| **Joblib**         | Model serialization and loading             |
| **Matplotlib**     | Evaluation plot generation                  |


The app executes locally or through Streamlit Cloud.
It automatically retrieves data from Open-Meteo, processes them, computes both rule-based and ML-based risk assessments, and renders:

- dual semi-circular gauges showing rule-based risk index and ML fire probability,
- agreement indicator between both methods,
- a daily polar plot showing variable contributions,
- a tabular summary of scores,
- a multi-day risk forecast with both predictions,
- a wind compass with wind rose-style wedge visualization (color-coded by risk level), and
- a regional map displaying wind flow patterns with geographic context.

3.1 Project structure and modular architecture

The codebase is organized into a modular architecture that separates concerns for maintainability, testability, and clarity.

**Core dashboard modules:**

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `app.py` | Main Streamlit application entry point | Orchestrates UI, data flow, integrates both risk methods |
| `config.py` | Configuration constants and parameters | Scoring bins, risk color schemes, geographic bounds, timezone settings |
| `data_fetcher.py` | External API data retrieval | `fetch_open_meteo()` — fetches hourly/daily weather data with caching |
| `risk_calculator.py` | Rule-based fire risk computation | `risk_components()`, `compute_days_without_rain()`, `best_hour_by_day()`, `color_for_risk()` |
| `visualizations.py` | Chart and plot generation | `create_polar_plot()`, `create_wind_compass()`, `create_forecast_charts()`, `create_dual_gauge()` |
| `map_utils.py` | Regional map visualization utilities | `create_wind_flow_field()`, `create_map_layers()`, `create_map_view_state()` |

**Machine learning modules (`ml_model/` directory):**

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `prepare_training_data.py` | Training dataset generation | Downloads fire data, fetches historical weather, creates balanced dataset |
| `train_fire_model.py` | Model training and evaluation | Trains Random Forest, generates evaluation plots, saves model |
| `validate_model_agreement.py` | Statistical validation | McNemar's test, Concordance Correlation, Bland-Altman analysis |
| `fire_model.pkl` | Trained model artifact | Loaded by dashboard for ML predictions |

**Data flow:**

1. **Data fetching** (`data_fetcher.py`): Retrieves raw meteorological data from Open-Meteo API (hourly and daily forecasts)
2. **Risk calculation**: 
   - Rule-based (`risk_calculator.py`): Computes scores using configurable scoring bins
   - ML-based (`fire_model.pkl`): Predicts fire probability using trained Random Forest
3. **Visualization** (`visualizations.py`): Generates interactive Plotly charts from both risk assessments
4. **UI orchestration** (`app.py`): Coordinates all modules, displays dual predictions, shows agreement indicator

**Module dependencies:**

```
app.py
├── config.py (constants, scoring bins, geographic bounds)
├── data_fetcher.py (API data retrieval)
│   └── config.py (timezone)
├── risk_calculator.py (rule-based risk computation)
│   └── config.py (scoring bins, risk colors)
├── visualizations.py (chart generation)
│   └── risk_calculator.py (color_for_risk)
├── map_utils.py (map utilities)
│   └── config.py (geographic bounds)
└── ml_model/
    ├── fire_model.pkl (trained Random Forest, loaded via joblib)
    └── validation_results.json (statistical agreement metrics)
```

4. Environment and reproducibility

The project is distributed with Conda environment files to make installs reproducible.

**4.1 Prerequisites**

**Miniforge/Conda** (recommended method):
- All OS: Download from https://github.com/conda-forge/miniforge
- Windows: Use the installer; check "Add to PATH" option
- macOS/Linux: Run the `.sh` installer in terminal

**First-time conda setup** (required once after installing Miniforge):
```
conda init bash
```
Then **close and reopen your terminal**. This enables the `conda activate` command. On Windows with PowerShell, use `conda init powershell` instead.

**Python 3.11** (only needed for pip method):
- Windows: Download from https://python.org (check "Add to PATH")
- macOS: `brew install python@3.11` or download from python.org
- Linux: `sudo apt install python3.11 python3.11-venv` or equivalent

**4.2 Installation**

**Recommended (conda — works on Windows / macOS / Linux):**

```
conda env create -f environment.yml
conda activate fire_risk_dashboard
streamlit run app.py
```

**Alternative (pip — works on any OS with Python 3.11):**

```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
streamlit run app.py
```

Both `environment.yml` and `requirements.txt` are **fully version-pinned** for reproducibility.

**4.3 Troubleshooting**

| Error | Cause | Solution |
|-------|-------|----------|
| `conda: command not found` | Miniforge not installed or not in PATH | Install Miniforge and restart terminal |
| `CondaError: Run 'conda init'...` | Shell not initialized for conda | Run `conda init bash` (or `powershell`), then **close and reopen terminal** |
| `CondaValueError: prefix already exists` | Environment already exists | Run `conda env remove -n fire_risk_dashboard` first |
| Solver hangs or takes forever | Package conflicts or slow network | Try `conda clean --all` then retry; or use `mamba` instead of `conda` |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Corporate firewall/proxy | Configure proxy or run `conda config --set ssl_verify false` (temporary) |
| `PermissionError` (macOS/Linux) | Using sudo with conda | Never use `sudo` with conda; reinstall Miniforge in user directory |
| Long path errors (Windows) | Windows 260-char path limit | Extract the zip to a shorter path (e.g., `C:\projects\`) |
| `Python 3.11 not found` (pip method) | Wrong Python version | Install Python 3.11; use `python3.11` instead of `python` |
| Apple Silicon (M1/M2/M3) issues | Package not built for arm64 | Try `CONDA_SUBDIR=osx-64 conda env create -f environment.yml` |

5. Dual Risk Assessment Methodology

The dashboard provides two complementary fire risk assessments displayed side-by-side:

**Rule-Based Risk Index (0-100 points)**
- Transparent scoring based on Chilean fire danger standards
- Uses predefined scoring bins for each variable
- Interpretable and aligned with operational fire management protocols

**Machine Learning Fire Probability (0-100%)**
- Random Forest classifier trained on 616 historical fires from La Araucanía (1984-2018)
- Data-driven predictions learned from actual fire occurrence patterns
- Validates the importance of variables used in the rule-based system (days without rain and temperature emerge as most critical)

**Agreement Indicator**
- Shows when both methods align, providing confidence in the risk assessment
- Disagreement highlights cases where empirical fire behavior may deviate from standardized risk categories

The trained model (`fire_model.pkl`) and training scripts are included in the `ml_model/` directory. See `ml_model/README.md` for technical details.

