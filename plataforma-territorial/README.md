# Plataforma Territorial FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Last Updated:** 2026-03-27
**What Changed:** FastAPI now serves the built frontend directly; systemd service + shell aliases for on-demand startup.
**Integration Status:** Ready (platform) | Pending [fire risk ML retrain, cameras/fauna tabs]

---

## Architecture: React + FastAPI

```
plataforma-demo/     ← React/Vite frontend (built to dist/)
backend/             ← FastAPI backend (port 8000) — serves API + built frontend
../data-pipeline/    ← writes fma_data.duckdb on schedule (systemd, always on)
```

---

## How to Run

### Normal use (single command)

```bash
plataforma          # starts the backend; open http://localhost:8000
plataforma-stop     # stop when done
```

These are shell aliases defined in `~/.bashrc`. They control the `fma-platform` systemd user service.

### After changing frontend code

```bash
cd plataforma-demo
npm run build       # rebuilds dist/ — backend picks it up immediately
```

### Frontend development (hot reload)

```bash
# Terminal 1
uvicorn backend.main:app --port 8000

# Terminal 2
cd plataforma-demo
npm run dev         # opens at http://localhost:5173
```

---

## Four Pages

| Page | Status | Description |
|---|---|---|
| Observatorio | Real data | Leaflet + Esri satellite, BP boundary polygon, 25 camera markers |
| Dashboard — Meteo | **Real data** | Year of weather history, variable selector, wind rose, comparison mode |
| Dashboard — Fire risk | Mock data | Backend ready, frontend not connected |
| Dashboard — Cameras/Fauna | Mock data | Pending |
| Asistente | Mock data | AI chat placeholder |
| Reportes | Mock data | Newsletter draft generator |

---

## Services

| Service | Auto-start | Command |
|---|---|---|
| `fma-pipeline` | On boot | `systemctl --user start fma-pipeline` |
| `fma-platform` | On demand | `plataforma` / `plataforma-stop` |

Logs: `journalctl --user -u fma-platform -f`

---

## Real Data

- `data/boundary.geojson` — Bosque Pehuén reserve boundary (86 vertices)
- `data/camera_trap_stations.geojson` — 25 cameras with real coordinates
- `data/stations.yaml` — single source of truth for all monitoring stations
- TC-26 excluded (erroneous coordinates in source spreadsheet)
- **NOTE:** BP boundary delimitation under review — confirm which polygon to use

---

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | React 19 + Vite 7 |
| Charts | Recharts |
| Map | Leaflet (react-leaflet) + Esri satellite tiles |
| Backend | FastAPI + Uvicorn |
| Database | DuckDB (read-only from `../fma_data.duckdb`) |
| AI (planned) | Anthropic Claude API (Asistente + Reportes) |

---

## API Endpoints (live)

```
GET  /api/health
GET  /api/weather/current
GET  /api/weather/history?start=&end=&resolution=&variables=
GET  /api/weather/forecast
GET  /api/fire-risk/current
GET  /api/fire-risk/forecast
GET  /api/detections/recent
GET  /api/detections/species-summary
GET  /api/detections/stations
```

---

## Key Design Decisions

1. **React + FastAPI, not Streamlit** — map needs full UI control, chat needs streaming
2. **DuckDB as the only data source** — all pages query the same DB via the backend
3. **FastAPI serves the built frontend** — single process, single port, no proxy needed
4. **Methodology transparency** — every computed answer cites its formula and data source
5. **Spanish throughout** — all UI, AI outputs, and reports in Spanish
6. **Station coordinates in flat files** — `data/stations.yaml` is the single source of truth
