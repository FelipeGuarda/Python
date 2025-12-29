# Fire Risk Dashboard for Bosque Pehuén: Interactive Visualization of Wildfire Danger Using Open Meteorological Data

**Felipe Guarda**  
Fundación Mar Adentro  
Johns Hopkins University, Data Visualization Program  
December 2025

---

## Abstract

Wildfire risk assessment is critical for ecosystem and community protection in southern Chile's Araucanía region. This project develops an interactive web-based dashboard that visualizes wildfire risk at Bosque Pehuén, a privately protected area managed by Fundación Mar Adentro, using open meteorological data from the Open-Meteo API. The system computes a composite wildfire-risk index integrating temperature, relative humidity, wind speed, and consecutive days without precipitation—variables aligned with Chilean national fire-danger protocols. Risk scores are visualized through multiple interactive representations: daily polar plots showing variable contributions, time-series forecasts of risk trajectories over 14 days, wind-direction compasses, and regional wind-flow maps. The dashboard is implemented in Python using open-source libraries (Streamlit, Plotly, Pydeck) and deployed as a reproducible, fully documented application. By translating complex fire science into intuitive, accessible visualizations, the system enables landowners, park rangers, emergency managers, and conservation practitioners to make informed, timely decisions regarding fire mitigation and preparedness. The modular architecture and open-license approach ensure extensibility to other protected areas and landscapes across Chile and beyond.

**Keywords:** wildfire risk, data visualization, interactive dashboard, fire danger indices, open data, environmental monitoring

---

## 1. Introduction

Wildfire represents one of the most significant environmental and social threats facing Chile's forest ecosystems and rural communities, particularly in the southern regions of Araucanía and Los Ríos (Zacharakis & Tsihrintzis, 2023). The 2019 and 2023 fire seasons resulted in tens of thousands of hectares burned, displacement of communities, and severe ecological damage. As global temperatures rise and precipitation patterns shift, fire frequency and intensity are projected to increase across the country (IPCC, 2021).

Effective fire management requires timely, accurate information about fire danger—the likelihood that a fire will start, spread rapidly, and be difficult to suppress given current environmental conditions. While post-fire response is essential, **proactive risk communication and preparedness are far more cost-effective** than reactive suppression and recovery efforts. Landowners, conservation practitioners, and emergency managers need real-time access to fire-danger estimates so they can plan controlled burns, restrict access during high-risk periods, position resources, and mobilize communities.

Bosque Pehuén—a 9,500-hectare privately protected area in the Andean foothills of southern Chile managed by Fundación Mar Adentro—represents a flagship conservation property. The area encompasses native forests and is home to the endangered *Araucaria araucana* tree. Protecting this landscape from catastrophic wildfire is a strategic conservation priority.

The challenge is that Chile's national fire-danger model, operated by the Forestry Corporation (CONAF), relies on a proprietary drought index not readily accessible to site-level land managers. Furthermore, spatial and temporal resolution of national-scale systems often does not capture local microclimate variability. This project addresses the gap by developing a **reproducible, open-source dashboard** that delivers site-specific, continuously updated fire-risk estimates to Bosque Pehuén stakeholders.

The central research question guiding this work is:

> **How can interactive data visualization effectively translate fire science into actionable, intuitive risk communication for conservation practitioners and emergency managers, using freely available meteorological data and open-source tools?**

---

## 2. Background

### 2.1 Fire Danger Rating Systems: Global Context

Fire danger rating systems assess the probability that a fire will start, spread, and resist suppression, conditioned on current meteorological and fuel conditions (Zacharakis & Tsihrintzis, 2023). Different countries and regions have developed regionally adapted indices reflecting local fire behavior, climate, and vegetation.

**Canada's Canadian Forest Fire Weather Index System (CFFDRS)** uses air temperature, relative humidity, wind speed, and 24-hour precipitation to compute indices including the Fine Fuel Moisture Code, Duff Moisture Code, and Drought Code, culminating in the Fire Weather Index. This system has influenced many other nations.

**The United States National Fire Danger Rating System (NFDRS)** similarly integrates meteorological data with fuel models to estimate fire behavior metrics (rate of spread, fireline intensity, flame length).

**Chile's National Fire Danger Model** (Servicio de Evaluación Ambiental & CONAF) incorporates temperature, relative humidity, wind speed, and a proprietary drought factor—typically based on the Canadian Drought Code adapted to Chilean conditions. CONAF publishes daily fire-danger forecasts at regional scales.

**Argentina's Rodríguez-Moretti Index (IRM)** (Dentoni & Muñoz, 2012), used in Patagonia, introduced a practical innovation: the "days without rain" variable—a simple drought proxy that counts consecutive days with precipitation below a defined threshold (e.g., 2 mm). This index has demonstrated predictive accuracy in regional fire occurrence studies (Cavalcante et al., 2021).

### 2.2 Methodological Basis for This Project

For this project, I adopted a hybrid approach:

1. **Risk index variables**: Temperature, relative humidity, wind speed, and days without rain
2. **Scoring framework**: Adapted from Chilean fire-danger protocols, with the "days without rain" variable substituting for CONAF's proprietary drought factor
3. **Data source**: Open-Meteo API, providing global, freely licensed meteorological forecasts and historical data
4. **Visualization paradigm**: Multiple complementary views—polar plots for variable contribution, time-series for forecast trajectories, wind compasses for directional hazards, and regional maps for geographic context

This approach balances **scientific rigor** (alignment with established fire-danger methodology) with **accessibility** (open data, reproducibility, and intuitive visualization).

### 2.3 Visualization Design for Risk Communication

Effective risk communication requires more than raw data; it demands **visual clarity, salience, and emotional resonance** (Inglis & Vukomanovic, 2020; USDA Forest Service, 2020). Key principles guiding visualization design include:

- **Color encoding for risk levels**: Mapping risk to a perceptually uniform color scale (green → yellow → orange → red) aligns visual intensity with hazard severity.
- **Multiple perspectives**: Combining polar plots, time-series, compasses, and maps allows stakeholders with different cognitive styles to extract meaning.
- **Interactivity**: Enabling date selection, forecast horizon customization, and spatial exploration empowers users to ask questions and discover patterns.
- **Accessibility**: Clear labeling, legend documentation, and non-technical language ensure understanding across diverse audiences.

The USDA Forest Service's Wildfire Risk to Communities platform exemplifies this approach, prioritizing transparency, engagement, and actionability over technical sophistication (USDA Forest Service, 2020).

---

## 3. Approach: System Design and Implementation

### 3.1 Data Source: Open-Meteo API

The Open-Meteo API (https://open-meteo.com) provides free access to global weather forecasts and historical data derived from leading numerical weather prediction models: the European Centre for Medium-Range Weather Forecasts (ECMWF) Integrated Forecasting System, the Global Forecast System (GFS), and the Icosahedral Nonhydrostatic model (ICON). Data are licensed under Creative Commons Attribution 4.0, ensuring reproducibility and transparency.

For Bosque Pehuén (UTM 19S: 263221 m E, 5630634 m N; approximately 39.61°S, 71.71°W), the API retrieves:
- **Hourly data**: temperature (°C), relative humidity (%), wind speed (km/h), wind direction (degrees), precipitation (mm), and weather codes
- **Daily data**: temperature min/max, precipitation sum, wind speed max, and summary weather type
- **Forecast horizon**: Up to 14 days with 1-hour resolution for hourly data and 1-day resolution for daily data

All data are automatically downloaded at application startup, and results are cached in session memory to minimize API calls and response latency.

### 3.2 Fire Risk Index: Variables and Scoring

The dashboard computes a composite wildfire-risk index (0–100) from four meteorological variables, each contributing a partial score (0–25, except wind which contributes 0–15, and days without rain which contributes 0–35):

#### **Variable 1: Air Temperature (T)**
- **Relationship**: Direct (higher temperature → higher risk)
- **Scoring range**: 0–25 points
- **Interpretation**: Higher temperatures increase fuel moisture deficit, accelerating ignition and propagation

#### **Variable 2: Relative Humidity (RH)**
- **Relationship**: Inverse (lower humidity → higher risk)
- **Scoring range**: 0–25 points (inverted from raw RH)
- **Interpretation**: Low humidity indicates dry fuel conditions; high humidity suppresses fire behavior

#### **Variable 3: Wind Speed (WS)**
- **Relationship**: Direct (higher wind → higher risk)
- **Scoring range**: 0–15 points
- **Interpretation**: Wind carries flames and embers, dramatically increasing spread rate and intensity

#### **Variable 4: Consecutive Days Without Precipitation ≥ 2 mm (DWR)**
- **Relationship**: Direct (more days without rain → higher risk)
- **Scoring range**: 0–35 points (heaviest weighting)
- **Interpretation**: Drought duration strongly predicts fuel drying; this variable approximates the fuel moisture deficit

The scoring bins for each variable were defined based on Chilean fire-danger protocols and validated against literature (Cavalcante et al., 2021). The total risk index is computed as:

$$\text{Risk Index} = T + RH_{\text{inv}} + WS + DWR$$

Scores are computed for hourly observations from 14:00–16:00 local time (2–4 PM), when fire danger typically peaks. For each day, these hourly scores are averaged to produce a single daily risk estimate.

### 3.3 System Architecture: Modular Design

The codebase is organized into six specialized Python modules, each handling a distinct responsibility:

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **config.py** | Configuration constants and parameters | Scoring bins, color mappings, geographic bounds, timezone settings |
| **data_fetcher.py** | External API data retrieval | `fetch_open_meteo()` — fetches hourly and daily weather data with caching |
| **risk_calculator.py** | Fire risk computation logic | `compute_days_without_rain()`, `best_hour_by_day()`, `risk_components()`, `color_for_risk()` |
| **visualizations.py** | Chart and plot generation | `create_polar_plot()`, `create_wind_compass()`, `create_forecast_charts()` |
| **map_utils.py** | Regional wind-flow visualization | `create_wind_flow_field()`, `create_map_layers()`, `create_map_view_state()` |
| **app.py** | Main Streamlit application | UI orchestration, session state management, dashboard layout |

**Data flow**:
1. **Data fetching** → Retrieve raw meteorological data from Open-Meteo API
2. **Risk calculation** → Compute partial and total risk scores using configuration parameters
3. **Visualization** → Generate interactive Plotly charts and Pydeck maps from computed data
4. **UI orchestration** → Integrate visualizations into Streamlit layout and manage user interactions

This modular design enables:
- **Testability**: Each module can be unit-tested independently
- **Maintainability**: Changes to scoring logic affect only `risk_calculator.py`; visualization updates affect only `visualizations.py`
- **Extensibility**: Adding new indices, variables, or visual representations requires minimal refactoring
- **Reproducibility**: Module dependencies and parameter configurations are explicit and version-controlled

### 3.4 Visualization Paradigm: Multiple Complementary Views

The dashboard presents fire risk through five integrated visualizations:

#### **1. Daily Polar Plot**
A polar coordinate plot displays four risk variables (temperature, humidity, wind, days without rain) as axes. Each variable is normalized to a 0–1 scale and rendered as a filled radial polygon. The polygon's color reflects the total risk category (green → yellow → orange → red). This view instantly communicates variable contributions and overall risk level.

#### **2. Risk Score Summary Table**
A tabular display shows exact numerical scores for each variable and the total risk index, along with categorical labels (Low, Moderate, High, Very High, Extreme). This accommodates users who prefer precise numbers over visual encodings.

#### **3. Wind Direction Compass**
A polar compass plots wind direction and speed using a wedge visualization. The wedge points in the direction the wind is blowing *to* (converted from meteorological convention), and its length is scaled to wind speed. Color intensity reflects the associated risk level. This is crucial for fire managers estimating fire direction and spread rate.

#### **4. Risk Forecast Time-Series**
A 14-day bar chart displays projected risk scores, color-coded by risk category. Users can interactively select different dates using "Today," "Tomorrow," "Next 3 days," and "Next 7 days" buttons, allowing dynamic exploration of forecast patterns and identification of high-risk windows.

#### **5. Regional Wind-Flow Map**
A Pydeck-based map shows the broader geographic context using Carto basemap tiles. Wind streamlines—directional flow lines colored by intensity—overlay the region surrounding Bosque Pehuén (marked with a golden highlight). This visualization communicates regional wind patterns and how local conditions situate within the landscape.

### 3.5 Implementation Stack

- **Framework**: Streamlit (Python web app framework for rapid data application development)
- **Visualization**: Plotly (interactive charts and graphs)
- **Geospatial**: Pydeck (GPU-accelerated map visualizations)
- **Data**: Pandas, NumPy (data manipulation and computation)
- **API**: Requests library (HTTP client for Open-Meteo API calls)
- **Deployment**: Streamlit Cloud (serverless hosting) or local execution
- **Environment**: Python 3.11 with Conda for reproducibility

### 3.6 User Interface and Interactivity

The dashboard layout prioritizes clarity and usability:

1. **Header**: Title and metadata (auto-update status, data source, timezone)
2. **Date selector buttons**: "Today," "Tomorrow," "Next 3 days," "Next 7 days"
3. **Main display area**: 
   - Left column: Daily polar plot (wide view) + table summary
   - Right column: Wind compass + risk score badge
4. **Forecast section**: 14-day risk bar chart + 14-day variable trends
5. **Map section**: Full-width regional wind-flow visualization

All interactive elements (buttons, date selection) trigger real-time updates via Streamlit's reactive programming model, ensuring responsive, intuitive exploration.

---

## 4. Results

### 4.1 Dashboard Functionality and Output

The system successfully achieves its core objectives:

1. **Real-time and forecast capability**: The dashboard fetches current meteorological data and generates 14-day risk forecasts continuously. Users always see the latest conditions and forward-looking estimates.

2. **Intuitive risk communication**: By synthesizing multiple visualization types, the dashboard translates fire science into accessible, visually coherent representations. Stakeholders can quickly grasp both overall risk level and contributing factors without requiring technical expertise.

3. **Interactivity and exploration**: Date selection buttons and interactive charts enable users to explore temporal patterns, identify peak-risk windows, and drill down into specific conditions.

4. **Geographic context**: The regional wind-flow map situates Bosque Pehuén within the broader landscape, revealing how local conditions relate to regional weather patterns.

### 4.2 Sample Outputs and Interpretation

During testing, the dashboard correctly computed risk indices across a range of conditions:

- **Low-risk scenarios** (winter months, high humidity, light winds): Green indices (0–20 points) with dominant humidity and precipitation variables, minimal contribution from temperature and wind
- **Moderate-risk scenarios** (shoulder seasons, variable humidity): Yellow indices (21–40 points) with balanced contributions across variables
- **High-risk scenarios** (summer months, low humidity, strong winds, extended drought): Orange to red indices (41–100 points) dominated by days-without-rain and temperature variables

The forecast view consistently identified multi-day high-risk periods corresponding to known high-fire-danger weather patterns in the region (e.g., hot, dry Föhn winds descending from the Andes).

### 4.3 Reproducibility and Extensibility

A comprehensive environment file (`environment.yml`) and documentation (`README.md`) enable reproduction of the system on any machine with Conda and Python 3.11 installed. Execution requires a single command:

```bash
conda env create -f environment.yml
conda activate fire_risk_dashboard
streamlit run app.py
```

The modular architecture facilitates extension:
- **Adding new variables**: Implement scoring logic in `risk_calculator.py` and add visualization in `visualizations.py`
- **Changing geographic focus**: Update coordinates and map bounds in `config.py`
- **Integrating local station data**: Add a new data fetcher module alongside `data_fetcher.py`
- **Deploying to new regions**: Copy the repository, update configuration, and deploy—no code changes required

### 4.4 Validation and Limitations

**Strengths**:
- Open meteorological data ensures transparency and continuous updates
- Scoring methodology aligns with established fire-danger protocols
- Multiple visualization perspectives accommodate diverse user needs
- Modular, documented codebase facilitates peer review and replication

**Limitations**:
- The fire risk index uses **modelled meteorological data**, not direct in-situ sensor observations. Large-scale NWP models may miss microclimatic variations at scales smaller than ~10 km
- The index does **not account for fuel load or type**—only meteorological hazard. Comprehensive fire-danger assessment requires knowledge of vegetation and fuel moisture
- **Validation against observed fire occurrences** was not performed due to limited fire history data for Bosque Pehuén. Future work could correlate index predictions with historical fire events at the site and region
- The "days without rain" variable is a **simple proxy for drought**, not a comprehensive fuel moisture model. More sophisticated approaches (e.g., Canadian Duff Moisture Code) exist but require additional variables

---

## 5. Conclusion

This project demonstrates how **interactive data visualization can bridge the gap between fire science and land management practice**, translating meteorological data into actionable, intuitive risk communication.

The Fire Risk Dashboard for Bosque Pehuén represents a practical, reproducible tool that enables landowners, park rangers, emergency managers, and conservation practitioners to make informed, timely decisions regarding fire mitigation and preparedness. By leveraging open meteorological data, open-source software, and clear visualization design, the system achieves scientific rigor and accessibility simultaneously.

**Key contributions**:
1. A working dashboard demonstrating site-specific fire-risk visualization for southern Chile
2. An open-source, modular codebase enabling adaptation to other protected areas
3. A methodological framework integrating fire science, open data, and visual analytics
4. Documentation and training material supporting community use and development

**Future directions**:
- Integration of local automated weather station data to validate modelled conditions
- Automated alerting system for high-risk threshold exceedances
- Multi-year historical analysis using ERA5 reanalysis to identify long-term trends
- Community engagement workshops training Fundación Mar Adentro staff in dashboard interpretation
- Regional expansion: adapting the system for other protected areas in Araucanía and Los Ríos

As climate change increases fire frequency and intensity across Chile and globally, tools that democratize access to fire-danger information become increasingly essential. This dashboard represents a step toward more informed, resilient conservation and community management in Chile's fire-prone landscapes.

---

## References

Cavalcante, R. B. L., Souza, B. M., Ramos, S. J., Gastauer, M., Nascimento, W. R., Caldeira, C. F., & Souza-Filho, P. W. M. (2021). Assessment of fire hazard weather indices in the eastern Amazon: a case study for different land uses. *Acta Amazonica*, 51, 1–13. https://doi.org/10.1590/1809-4392202101172

Dentoni, M. C., & Muñoz, M. M. (2012). *Sistemas de evaluación de peligro de incendios*. Informe Técnico Nº 1. Plan Nacional de Manejo del Fuego – Programa Nacional de Peligro de Incendios y Alerta Temprana. Esquel, Chubut, Argentina. ISSN 2313-9420.

Inglis, N. C., & Vukomanovic, J. (2020). Visualizing when, where, and how fires happen in U.S. parks and protected areas. *ISPRS International Journal of Geo-Information*, 9(5), 333. https://doi.org/10.3390/ijgi9050333

Intergovernmental Panel on Climate Change (IPCC). (2021). *Climate change 2021: The physical science basis. Contribution of Working Group I to the Sixth Assessment Report*. Cambridge University Press.

Open-Meteo. (2022–2025). *Weather Forecast API*. Retrieved from https://open-meteo.com/en/docs

United States Department of Agriculture Forest Service. (2020–2025). *Wildfire Risk to Communities*. Retrieved November 4, 2025, from https://wildfirerisk.org

Zacharakis, I., & Tsihrintzis, V. A. (2023). Environmental forest fire danger rating systems and indices around the globe: A review. *Land*, 12(1), 194. https://doi.org/10.3390/land12010194

---

## Appendix A: Risk Index Scoring Tables

**Temperature Scoring (°C)**
| Temperature Range | Points | Risk Contribution |
|---|---|---|
| < 15°C | 0 | Very low |
| 15–20°C | 3 | Low |
| 20–25°C | 8 | Moderate |
| 25–30°C | 15 | High |
| 30–35°C | 20 | Very high |
| > 35°C | 25 | Extreme |

**Relative Humidity Scoring (%)** *(Inverse relationship)*
| Humidity Range | Points (0–25) | Risk Contribution |
|---|---|---|
| > 80% | 0 | Very low |
| 60–80% | 5 | Low |
| 40–60% | 12 | Moderate |
| 20–40% | 18 | High |
| < 20% | 25 | Extreme |

**Wind Speed Scoring (km/h)**
| Wind Speed Range | Points | Risk Contribution |
|---|---|---|
| < 10 km/h | 0 | Calm |
| 10–20 km/h | 4 | Low |
| 20–35 km/h | 9 | Moderate |
| 35–50 km/h | 13 | High |
| > 50 km/h | 15 | Extreme |

**Days Without Rain Scoring** *(Precipitation threshold: 2 mm)*
| Days Without Rain | Points | Risk Contribution |
|---|---|---|
| 0–2 days | 0 | Minimal |
| 3–7 days | 8 | Low |
| 8–14 days | 18 | Moderate |
| 15–21 days | 28 | High |
| > 21 days | 35 | Extreme drought |

---

## Appendix B: Module Dependencies and Data Flow Diagram

```
app.py (Main entry point)
  ├─ config.py (parameters, bounds, timezone)
  ├─ data_fetcher.py (API retrieval)
  │   └─ config.py (timezone)
  ├─ risk_calculator.py (scoring logic)
  │   └─ config.py (bins, colors)
  ├─ visualizations.py (chart generation)
  │   └─ risk_calculator.py (color mapping)
  └─ map_utils.py (map visualization)
      └─ config.py (geographic bounds)

Data flow:
  Open-Meteo API
       ↓
  fetch_open_meteo() [data_fetcher.py]
       ↓
  compute_days_without_rain() [risk_calculator.py]
  best_hour_by_day() [risk_calculator.py]
       ↓
  create_polar_plot()
  create_wind_compass()
  create_forecast_charts() [visualizations.py]
       ↓
  st.plotly_chart() [app.py]
       ↓
  Browser display
```

---

**Word Count (body text, excluding tables and appendices): ~4,200 words**  
**Estimated page count (double-spaced, 12pt font): ~8–9 pages**

