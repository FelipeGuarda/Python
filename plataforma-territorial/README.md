# Plataforma Territorial FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Status:** React/Vite prototype working. Backend and real data not yet connected.

---

## Architecture Decision: React + FastAPI (not Streamlit)

The original design called for a Streamlit app. After building a working React/Vite prototype,
the decision is to **keep React as the frontend and add a FastAPI backend** when real data is ready.

**Why React over Streamlit:**
- The Observatorio map (the flagship page) needs full UI control — interactive layers, custom
  popups, animated overlays. Streamlit's map components can't deliver this.
- The Asistente chat needs streaming Claude responses and proper conversation UX. Streamlit's
  `st.chat_message` re-runs the whole script on every message.
- The prototype already exists, works, and looks right. Rebuilding in Streamlit would produce
  a worse version of what already exists.

**The trade-off:** React requires a backend API to reach Python/DuckDB. That backend is a
small FastAPI app — roughly 6 endpoints — and lives at `plataforma-territorial/backend/`.

**Full stack (when real data is connected):**
```
plataforma-demo/     ← React/Vite frontend (runs on port 5173)
backend/             ← FastAPI backend (runs on port 8000)
../data-pipeline/    ← writes fma_data.duckdb
```

---

## How to Run

### Demo (prototype with mock data) — works today

```bash
cd plataforma-territorial/plataforma-demo
npm install          # first time only
npm run dev          # opens at http://localhost:5173
```

### Real platform (when backend is ready) — same frontend command

```bash
# Terminal 1 — backend
cd plataforma-territorial/backend
pip install -r requirements.txt   # first time only
uvicorn main:app --reload         # runs at http://localhost:8000

# Terminal 2 — frontend (same as demo)
cd plataforma-territorial/plataforma-demo
npm run dev                       # opens at http://localhost:5173
```

The frontend automatically switches from mock data to real API calls when the backend is
reachable. No other changes needed — same URL, same page.

---

## Four Pages

```
Plataforma Territorial FMA
├── 1. Observatorio    ← interactive map — FLAGSHIP page
├── 2. Dashboard       ← tabbed data dashboard
├── 3. Asistente       ← AI chat with methodology transparency
└── 4. Reportes        ← newsletter draft + download
```

---

## GPS Coordinates and Map Stations

### Known coordinates

| Location | Lat | Lon | Source | Notes |
|---|---|---|---|---|
| Bosque Pehuén (general) | -39.61 | -71.71 | `data-pipeline/config.yaml` | UTM 19S: E=263221 N=5630634 |
| Estación meteorológica CR800 | -39.61 | -71.71 | `Estacion meteorologica/config.py` | Same point — confirm exact position |
| Araucanía region bounds | -40.0 / -38.0 | -73.0 / -71.0 | `Estacion meteorologica/config.py` | Used as map extent |

### Boundary and station coordinates — available, not yet added to repo

The following real field data exists and will be added to `data/` when available:
- **Bosque Pehuén boundary polygon** — full perimeter points (replaces the SVG placeholder)
- **Camera trap station coordinates** — all deployed stations with GPS positions
- **Flora plot coordinates** — all vegetation monitoring plot positions
- **Meteorological station** — exact CR800 position (currently approximated as -39.61, -71.71)

**When adding coordinates, store them here:**
- `plataforma-territorial/data/boundary.geojson` — reserve boundary polygon
- `plataforma-territorial/data/stations.yaml` — all monitoring stations + flora plots

**Suggested stations.yaml format:**
```yaml
stations:
  - id: cam_01
    name: Cámara Araucaria Norte
    type: camera
    lat: -39.xxx
    lon: -71.xxx
  - id: met_01
    name: Estación Meteorológica CR800
    type: weather
    lat: -39.xxx
    lon: -71.xxx
  - id: flora_01
    name: Parcela Flora 01
    type: flora_plot
    lat: -39.xxx
    lon: -71.xxx
```

This file becomes the authoritative source for all station positions —
both the platform and the data pipeline should read from it.

---

## Backend API (to build — 6 endpoints cover everything)

```
GET  /api/stations                  → station list with coordinates + current status
GET  /api/weather/current           → latest readings from DuckDB (weather_station table)
GET  /api/weather/forecast          → 7-day Open-Meteo data (weather_forecast table)
GET  /api/fire-risk/current         → today's FRI rule-based + ML score (fire_risk table)
GET  /api/detections/recent         → last N camera trap records (camera_trap table)
POST /api/asistente/chat            → streams Claude API response with tool use
```

All endpoints read from `fma_data.duckdb`. The platform never writes to the database.

**Reference for fire risk endpoint:** `Estacion meteorologica/Fire risk dashboard/` —
port `risk_calculator.py` and `fire_model.pkl` directly. Do not rewrite.

---

## File Structure

```
plataforma-territorial/
│
├── plataforma-demo/              ← React/Vite frontend (working prototype)
│   ├── src/
│   │   ├── App.jsx               ← single-file app (all pages + components)
│   │   ├── main.jsx
│   │   ├── index.css             ← minimal reset only
│   │   └── App.css               ← #root sizing only
│   ├── package.json
│   └── vite.config.js
│
├── backend/                      ← FastAPI backend (to build)
│   ├── main.py                   ← FastAPI entry point + route registration
│   ├── db.py                     ← DuckDB connection + query helpers
│   ├── routers/
│   │   ├── weather.py
│   │   ├── fire_risk.py
│   │   ├── detections.py
│   │   └── asistente.py          ← Claude tool-use orchestration
│   ├── risk/                     ← ported from Estacion meteorologica/
│   │   ├── calculator.py
│   │   └── fire_model.pkl
│   └── requirements.txt
│
└── data/
    ├── stations.yaml             ← authoritative station coordinates (to create)
    └── fma_data.duckdb           ← symlink to data-pipeline output
```

---

## Data Dependencies

The platform is **read-only** with respect to data. All writes come from `data-pipeline/`.

| Data | Source project | DuckDB table |
|---|---|---|
| Weather station readings | data-pipeline (CR800 fetcher) | `weather_station` |
| Weather forecast | data-pipeline (Open-Meteo fetcher) | `weather_forecast` |
| Fire risk index | data-pipeline (computed daily) | `fire_risk` |
| Camera trap records | data-pipeline (Timelapse2 CSV watcher) | `camera_trap` |
| Acoustic detections | data-pipeline (audio analysis pipeline — not yet built) | `acoustic_detections` |
| Literature summaries | literatura-agent | `literatura` |

**Build order:** data-pipeline must be running before the backend has real data to serve.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | React 19 + Vite 7 |
| Charts | Recharts |
| Map (real) | Mapbox GL JS or Leaflet (to decide when building Observatorio) |
| Backend | FastAPI + Uvicorn |
| Database | DuckDB (read-only from backend) |
| AI (Asistente + Reportes) | Anthropic Claude API — Sonnet + tool use |
| Deployment | TBD (Vercel for frontend + Render/Railway for backend, or self-hosted) |
| Language | JSX (frontend) + Python 3.11 (backend) |

---

## Key Design Decisions

1. **React + FastAPI, not Streamlit** — see Architecture Decision section above.
2. **DuckDB as the only data source** — all pages query the same DB via the backend. Data
   freshness is the pipeline's responsibility, not the platform's.
3. **Methodology transparency in Asistente** — every computed answer must cite its formula
   and data source. Non-negotiable for conservation credibility.
4. **Spanish throughout** — all UI labels, AI outputs, and report drafts in Spanish.
5. **Observatorio is the flagship** — loads fast, map-first, polished. It's the first thing
   anyone sees.
6. **Reportes as drafting aid, not automated publishing** — Claude drafts, human edits,
   human publishes.
7. **Station coordinates in a flat file** — `data/stations.yaml` is the single source of
   truth for all station positions. Both frontend and backend read from it (via the API).
   Timelapse2 is the origin, but it shouldn't be the runtime dependency.

---

## Context for AI Sessions

1. Frontend is React 19 + Vite — entry point `plataforma-demo/src/App.jsx` (single file, all pages)
2. All pages currently use mock data — replacing mock data = connecting to FastAPI endpoints above
3. Backend doesn't exist yet — build it in `plataforma-territorial/backend/`
4. Fire risk logic already exists — port from `Estacion meteorologica/Fire risk dashboard/`, don't rewrite
5. Asistente uses Claude with **tool use** — tools are thin wrappers around DuckDB queries
6. Bosque Pehuén coordinates: lat -39.61, lon -71.71
7. Camera station coordinates are not yet in any config — must be exported from Timelapse2 first
8. The data-pipeline project must be running before the backend has any real data to serve
9. Target audience: FMA staff + conservation partners + general public — accessible Spanish language
