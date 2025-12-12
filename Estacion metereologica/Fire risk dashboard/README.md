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

Each variable is assigned a partial score from 0 to 25 points according to predefined ranges, producing a total risk index from 0 to 100 points.
Humidity contributes inversely (high humidity → lower risk), while the remaining variables contribute directly.

| Variable          | Unit | Relationship with risk | Scoring range |
| ----------------- | ---- | ---------------------- | ------------- |
| Air temperature   | °C   | Positive               | 0–25          |
| Relative humidity | %    | Inverse                | 0–25          |
| Wind speed        | km/h | Positive               | 0–25          |
| Days without rain | days | Positive               | 0–25          |


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
| **Pydeck**         | Spatial visualization of regional risk maps (uses Carto basemap) |
| **Pandas / NumPy** | Data management and computation             |
| **PyProj**         | Coordinate conversion (UTM → WGS 84)        |
| **Requests**       | API calls to Open-Meteo                     |


The app executes locally or through Streamlit Cloud.
It automatically retrieves data from Open-Meteo, processes them, computes risk scores, and renders:

- a daily polar plot showing variable contributions,
- a tabular summary of scores,
- a multi-day risk forecast,
- a wind compass with wind rose-style wedge visualization (color-coded by risk level), and
- a regional map displaying modelled risk gradients with geographic context.

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
| `map_utils.py` | Regional map visualization utilities | `create_araucania_grid()`, `grid_forecast()`, `create_map_layers()`, `create_map_view_state()` |

**Data flow:**

1. **Data fetching** (`data_fetcher.py`): Retrieves raw meteorological data from Open-Meteo API
2. **Risk calculation** (`risk_calculator.py`): Processes raw data to compute risk scores using configurable scoring bins
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
    ├── data_fetcher.py (grid data fetching)
    ├── risk_calculator.py (risk computation)
    └── config.py (geographic bounds, grid step)
```

**Benefits of modular structure:**

- **Separation of concerns**: Each module has a single, well-defined responsibility
- **Reusability**: Functions can be imported and used independently
- **Testability**: Individual modules can be tested in isolation
- **Maintainability**: Changes to scoring logic, visualization, or data sources are localized
- **Clarity**: Code organization makes the codebase easier to understand and navigate

**For AI replication:**

When replicating or extending this codebase, maintain the modular structure:
- Keep configuration constants in `config.py`
- Isolate API calls in `data_fetcher.py`
- Centralize risk calculation logic in `risk_calculator.py`
- Separate visualization code in `visualizations.py`
- Keep map-specific utilities in `map_utils.py`
- Use `app.py` only for Streamlit UI orchestration and data flow coordination


4. Environment and reproducibility

The project is distributed with a Conda environment file (environment.yml) ensuring reproducible execution across platforms.

To recreate the environment:

conda env create -f environment.yml
conda activate fire_risk_dashboard
streamlit run app.py

The file specifies Python 3.11 and the following key dependencies:
streamlit, plotly, pydeck, pandas, numpy, requests, and pyproj.


5. References

- Dentoni, M. C. & Muñoz, M. M. (2012). Sistemas de evaluación de peligro de incendios. Informe Técnico Nº 1. Plan Nacional de Manejo del Fuego – Programa Nacional de Evaluación de Peligro de Incendios y Alerta Temprana. Esquel, Chubut, Argentina. ISSN 2313-9420.
- Open-Meteo (2022–2025). Weather Forecast API. Retrieved from https://open-meteo.com/en/docs
- Gil-Romera, G., et al. (2023). “Environmental Forest Fire Danger Rating Systems and Indices Around the Globe: A Review.” Land, 12(1), 194. https://doi.org/10.3390/land12010194

6. Recent updates (2025)

**Regional risk map improvements:**
- Fixed base map display issue by switching from Mapbox (requires API key) to Carto basemap (`carto-positron` style)
- The map now displays the geographic context of the Araucania region without requiring external API key configuration
- All overlay layers (risk grid hexagons, wind vectors, Bosque Pehuén marker) continue to function as before

**Wind compass visualization enhancements:**
- Replaced arrow-based wind direction indicator with wind rose-style wedge visualization
- The wedge extends from the center of the compass in the direction the wind is coming from, following standard meteorological convention
- Wind compass wedge color now dynamically matches the fire risk index color for the selected day, providing visual consistency across the dashboard
- The wedge uses a 30-degree angular width and extends to 75% of the compass radius for clear visibility

**Technical details:**
- Map basemap: Changed from `mapbox://styles/mapbox/light-v9` to `carto-positron` in `app.py`
- Wind compass: Updated `create_wind_compass()` function in `visualizations.py` to accept `risk_color` parameter and render wedge instead of arrow
- Color integration: Wind compass now receives the risk color calculated from `color_for_risk()` function, ensuring visual consistency with the risk index display


7. Future development

- Integration of local station data to compare observed vs. modelled conditions.
- Addition of legend and spatial interpolation for the regional risk map.
- Automated data archiving and alert system for high-risk thresholds.
- Multi-year analysis of historical trends using ERA5 reanalysis data.