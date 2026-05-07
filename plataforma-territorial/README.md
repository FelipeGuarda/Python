# Plataforma Territorial FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Last Updated:** 2026-05-07
**What Changed:** Frontend styling architecture (W44) closed via a CSS Modules pass. 246 → 25 inline `style={{}}` sites (90% reduction). New `src/styles/vars.css` exposes the design palette as 13 `--color-*` CSS custom properties; 19 new `.module.css` files (one per component, co-located). `Card` and `SectionLabel` now accept `className` + `style` props (closes S65). Asistente suggestion-button hover hack replaced with `:hover` rule (closes S77). `Reportes.module.css` adds `@media print` block for clean PDF/print export (closes S78). Earlier same day: `src/styles/chart.js` extracted Recharts prop-style repetition (closes S67 + W44 chart slice). Build clean: `npm run build`, 1.91s, CSS 20.7 → 36.5 kB / 10.8 kB gzipped.
**Integration Status:** Ready [Observatorio map; 26 TCs from canonical stations.yaml; species classification via API; modular frontend per W41; CSS-Modules styling per W44] | Pending [cámaras trampa dashboard tab polish, fauna tab real data, weather-station coords small W51 follow-up, sampleDraft function-form (S64), Recharts thresholds shared with backend (S66)]
**Blockers/Notes:** Frontend technical debt is largely resolved. Cross-project chains closed: station-registry, species-catalog, frontend-decomposition, frontend-styling. Remaining open: station-count drift (W39/W46 already closed; this is residual), DST/timezone (Block 2 W8), risk thresholds (S66). Full table in `review-plan-fma-ecosystem.md`. 8 Spanish display names changed to canonical form via species.yaml — flag for biological review with Felipe before user-facing release.

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
