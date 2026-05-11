# Plataforma Territorial FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Last Updated:** 2026-05-11 — code review complete
**What Changed:** Closed the plataforma half of the FMA-ecosystem review in five batches today. Tier 1: S50 (`/api/health` raises 503 on DB failure). Tier 2: S44 (`backend/paths.py`), S48 (stations.yaml coords — fixed ~17 km drift in bootstrap), S55 (DuckDB CTE for `days_without_rain`), S64 (`demo_report.js`), S66 (RiskGauge pure presentational component). Bundle A — schema authority: S49 (startup drift-check; surfaced 27 real DB columns missing from `ALLOWED_COLS` including `battery_voltage`). Bundle B: S53 (species-endpoint docstring tightening + `common_name` symmetry on `/species-summary`), S59 (`useAPI` now exposes `refetch`). Tier 4: S72 closed-rejected (SPA-by-design comment added to `App.jsx`); S76 deferred until CI exists. **First full code review now complete:** every finding in this project is closed or has an explicit re-open trigger. See repo-root `CHANGELOG.md`.
**Integration Status:** Ready [Observatorio map; 26 TCs from canonical stations.yaml; species classification via API; modular frontend per W41; CSS-Modules styling per W44; schema-drift visible at boot per S49] | Pending [`battery_voltage` curation in `ALLOWED_COLS` — surfaced by S49 drift check; cámaras trampa dashboard tab polish; fauna tab real data; Asistente real Claude API]
**Blockers/Notes:** Code review complete. Deferred items with explicit triggers: S58 needs Felipe's field records (TC-11 vs TC-18 SD-card duplication in stations.yaml); S76 re-opens when CI lands. 8 Spanish display names changed to canonical form via species.yaml — still flagged for biological review before user-facing release.

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
| Observatorio | Real data | Leaflet + Esri satellite, BP boundary polygon, 26 camera markers |
| Dashboard — Meteo | **Real data** | Year of weather history, variable selector, wind rose, comparison mode |
| Dashboard — Fire risk | **Real data** | Polar contribution chart, FRI gauge, wind compass, 3-week bar chart with history+forecast |
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
- `data/camera_trap_stations.geojson` — 26 cameras with real coordinates
- `data/stations.yaml` — single source of truth for all monitoring stations
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
GET  /api/fire-risk/history?days=
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
