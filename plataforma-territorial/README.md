# Plataforma Territorial FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Status:** React/Vite frontend working with real map (Leaflet + satellite imagery). Backend not yet built.

---

## Architecture: React + FastAPI

The frontend is a React/Vite app. When real data is connected, a small FastAPI backend will serve DuckDB queries to the frontend.

```
plataforma-demo/     ← React/Vite frontend (port 5173)
backend/             ← FastAPI backend (port 8000) — to build
../data-pipeline/    ← writes fma_data.duckdb
```

---

## How to Run

```bash
cd plataforma-territorial/plataforma-demo
npm install          # first time only
npm run dev          # opens at http://localhost:5173
```

Requires Node.js 20.19+ (use nvm: `nvm use 22`).

---

## Four Pages

| Page | Status | Description |
|---|---|---|
| Observatorio | Real map (Leaflet + Esri satellite) | Interactive map with BP boundary polygon + 25 camera trap markers + weather station |
| Dashboard | Mock data | Tabbed dashboard: fire risk, meteorology, cameras, fauna |
| Asistente | Mock data | AI chat with methodology transparency |
| Reportes | Mock data | Newsletter draft generator |

---

## Real Data Already Added

- `data/boundary.geojson` — Bosque Pehuén reserve boundary (86 vertices, from areaBP.kml)
- `data/camera_trap_stations.geojson` — 25 installed cameras with real coordinates (from field Excel)
- `data/stations.yaml` — all monitoring stations: 1 weather + 25 cameras (TC-01 to TC-25)
- TC-26 excluded due to erroneous coordinates in source spreadsheet (flagged as TODO)
- **NOTE:** BP boundary delimitation is under review — confirm which polygon to use

---

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | React 19 + Vite 7 |
| Charts | Recharts |
| Map | Leaflet (react-leaflet) + Esri satellite tiles |
| Backend (planned) | FastAPI + Uvicorn |
| Database | DuckDB (read-only from backend) |
| AI (Asistente + Reportes) | Anthropic Claude API |

---

## Backend API (to build — 6 endpoints)

```
GET  /api/stations              → station list from stations.yaml
GET  /api/weather/current       → latest CR800 readings
GET  /api/weather/forecast      → 7-day Open-Meteo data
GET  /api/fire-risk/current     → FRI rule-based + ML score
GET  /api/detections/recent     → last N camera trap observations
POST /api/asistente/chat        → streams Claude API response
```

Fire risk logic: port from `Estacion meteorologica/Fire risk dashboard/`, don't rewrite.

---

## Key Design Decisions

1. **React + FastAPI, not Streamlit** — map needs full UI control, chat needs streaming
2. **DuckDB as the only data source** — all pages query the same DB via the backend
3. **Methodology transparency** — every computed answer cites its formula and data source
4. **Spanish throughout** — all UI, AI outputs, and reports in Spanish
5. **Station coordinates in flat files** — `data/stations.yaml` is the single source of truth
