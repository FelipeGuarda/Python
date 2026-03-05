# Plataforma Territorial FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Status:** Planned / Not yet built.
**Deployment target:** Single Streamlit Cloud URL. One app, four pages.

---

## What This Project Does

A unified web platform that brings together all of FMA's environmental monitoring data into one publicly accessible interface. It replaces the current fragmented setup (separate dashboard scripts, standalone notebooks) with a single, polished Streamlit multi-page application.

All pages share the same data source: the `fma_data.duckdb` database maintained by the **data-pipeline** project.

---

## Architecture: Four Pages

```
Plataforma Territorial FMA
├── 1. Observatorio    ← interactive map — FLAGSHIP page
├── 2. Dashboard       ← tabbed data dashboard
├── 3. Asistente       ← AI chat with methodology transparency
└── 4. Reportes        ← newsletter draft + download
```

---

## Page 1 — Observatorio (FLAGSHIP)

**What it is:** An interactive map of Bosque Pehuén and the surrounding territory.

**Data layers (planned):**
- Camera trap station locations + species richness per station
- Fire risk zones (color-coded by current risk level)
- Weather station location + current conditions
- Historical fire perimeters (CONAF data)
- Vegetation / land cover (from remote sensing)
- Protected area boundary

**Key UX decisions:**
- Map-first: the map is the primary interface, not a sidebar
- Clicking a camera station opens a popup with recent species detections
- Clicking the weather station shows current conditions and risk level
- A time slider controls the temporal view (e.g., fire risk over the last 30 days)

**Likely tech:** `pydeck` or `folium` + `streamlit-folium` for the map; layers built from DuckDB queries.

---

## Page 2 — Dashboard

**What it is:** Tabbed data dashboard aggregating all monitoring streams.

**Tabs:**
| Tab | Content |
|---|---|
| Riesgo de Incendio | Current fire risk gauge (rule-based + ML), forecast, polar contribution plot |
| Meteorología | Temperature, humidity, wind, precipitation time series from weather station |
| Cámaras Trampa | Species activity charts, recent detections, station activity map |
| Fauna & Especies | Species list, activity patterns, occupancy summaries |

**Data flow:** All tabs query `fma_data.duckdb` tables (`weather_station`, `weather_forecast`, `fire_risk`, `camera_trap`).

**Design principle:** Tabs are independent — each loads only its own data. No cross-tab state.

---

## Page 3 — Asistente

**What it is:** A conversational AI interface that answers questions about the platform's data and methodology.

**Two modes:**
1. **Data questions**: "¿Cuál es el riesgo de incendio hoy?" → queries DuckDB, formats answer
2. **Methodology questions**: "¿Cómo se calcula el índice de riesgo?" → explains the rule-based index, scoring bins, ML model validation

**Methodology transparency principle:** Every answer that involves a computed value must show its source — which formula, which data, which model produced the number. No black-box outputs.

**Implementation:** Claude API (Sonnet) with tool use:
- `query_fire_risk(date)` → returns current/historical risk values
- `query_species(station, date_range)` → returns detection records
- `query_weather(date_range)` → returns met data
- `explain_methodology(topic)` → returns structured explanation from a methodology knowledge base

**Language:** Spanish only. All prompts, system messages, and outputs in Spanish.

---

## Page 4 — Reportes

**What it is:** A report drafting assistant that generates a monthly newsletter section.

**Workflow:**
1. User selects a date range (default: last calendar month)
2. The app queries DuckDB for all monitoring data in that period
3. Claude Sonnet drafts a 3–5 paragraph Spanish summary (fire risk highlights, notable wildlife detections, weather anomalies)
4. User can edit the draft inline (Streamlit `st.text_area`)
5. Download as formatted Word `.docx` or copy as plain text

**Output style:** Written for a general conservation audience — no jargon, emphasis on narrative over numbers.

---

## Current State

- **Not yet built.** The fire-risk dashboard (existing project at `Estacion meteorologica/Fire risk dashboard/`) implements roughly what Page 2's "Riesgo de Incendio" tab will become. That code will be refactored/ported into this platform.
- DuckDB schema and data pipeline must be built first (see `data-pipeline/README.md`).

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI framework | Streamlit (multi-page app) |
| Map | pydeck or streamlit-folium |
| Charts | Plotly |
| Database | DuckDB (read-only from the platform's side) |
| AI (Asistente + Reportes) | Anthropic Claude API (Sonnet + tool use) |
| Document export | `python-docx` |
| Deployment | Streamlit Cloud |
| Language | Python 3.11 |

---

## File Structure (planned)

```
plataforma-territorial/
├── .env                          ← ANTHROPIC_API_KEY, DB path
├── config.yaml                   ← site coordinates, layer toggles, Claude model
├── requirements.txt
│
├── app.py                        ← Streamlit entry point + page routing
│
├── pages/
│   ├── 1_observatorio.py         ← interactive map page
│   ├── 2_dashboard.py            ← tabbed dashboard
│   ├── 3_asistente.py            ← AI chat page
│   └── 4_reportes.py             ← report drafting page
│
├── src/
│   ├── db.py                     ← DuckDB connection + query helpers
│   ├── map_layers.py             ← pydeck layer builders
│   ├── charts.py                 ← Plotly chart functions (shared across pages)
│   ├── risk.py                   ← fire risk calculation (ported from existing dashboard)
│   ├── assistant.py              ← Claude tool-use orchestration for Asistente
│   ├── report_builder.py         ← Claude draft generation + docx export
│   └── methodology.py            ← structured methodology knowledge base (for transparency)
│
└── data/
    └── fma_data.duckdb           ← symlink or path to the data pipeline's DB file
```

---

## Data Dependencies

This platform is **read-only** with respect to data. It depends on:

| Data | Source | Table in DuckDB |
|---|---|---|
| Weather station readings | data-pipeline (CR800) | `weather_station` |
| Weather forecast | data-pipeline (Open-Meteo) | `weather_forecast` |
| Fire risk index | data-pipeline (computed) | `fire_risk` |
| Camera trap records | data-pipeline (Timelapse2 CSV) | `camera_trap` |
| Literature summaries | literatura-agent | `literatura` |

---

## Ideas & Future Features

- **Public/private mode**: A public URL showing aggregated/anonymized data; a private view with full detail for FMA staff
- **Species alert**: Push notification (email or Slack) when a priority species (Puma, Guiña) is detected at a station
- **Comparison view**: Compare this week's fire risk against same week last year
- **Exportable map**: Download the current Observatorio view as a PNG or PDF for reports

---

## Key Design Decisions

1. **Single URL, multi-page Streamlit**: One Streamlit Cloud deployment instead of multiple separate apps. Easier to share, easier to maintain.
2. **DuckDB as the only data source**: All pages query the same DB — no page-level API calls. Data freshness is the pipeline's responsibility, not the platform's.
3. **Methodology transparency in the Asistente**: Every computed answer must cite its formula. This is non-negotiable for conservation credibility — stakeholders need to understand how risk numbers are produced.
4. **Spanish throughout**: Platform language is Spanish. All UI labels, AI outputs, and report drafts are in Spanish.
5. **Page 1 (Observatorio) is the flagship**: It should load fast, look polished, and be the first thing anyone sees. The map communicates the territory before any numbers do.
6. **Reportes as drafting aid, not automated publishing**: Claude drafts, human edits, human publishes. The AI is a writing accelerator, not a replacement.

---

## Context for AI Sessions

When starting a new Claude session on this project:

1. This is a **Streamlit multi-page app** — entry point is `app.py`, each page is a file in `pages/`
2. All data comes from `fma_data.duckdb` — the platform never writes to the database
3. The Asistente page uses Claude with **tool use** — tools are thin wrappers around DuckDB queries
4. The existing fire-risk dashboard code (`Estacion meteorologica/Fire risk dashboard/`) is the reference implementation for Page 2's fire risk tab — port it, don't rewrite from scratch
5. Bosque Pehuén coordinates: lat -39.61°, lon -71.71° (UTM 19S: E=263221, N=5630634)
6. Target audience for the platform: FMA staff + conservation partners + general public — language must be accessible
7. The data-pipeline project must exist and be running before this platform has any real data to display
