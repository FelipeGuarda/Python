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
| **Pydeck**         | Spatial visualization of regional risk maps |
| **Pandas / NumPy** | Data management and computation             |
| **PyProj**         | Coordinate conversion (UTM → WGS 84)        |
| **Requests**       | API calls to Open-Meteo                     |


The app executes locally or through Streamlit Cloud.
It automatically retrieves data from Open-Meteo, processes them, computes risk scores, and renders:

- a daily polar plot showing variable contributions,
- a tabular summary of scores,
- a multi-day risk forecast, and
- a regional map displaying modelled risk gradients.


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

6. Future development

- Integration of local station data to compare observed vs. modelled conditions.
- Addition of legend and spatial interpolation for the regional risk map.
- Automated data archiving and alert system for high-risk thresholds.
- Multi-year analysis of historical trends using ERA5 reanalysis data.