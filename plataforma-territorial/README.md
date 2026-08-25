# Plataforma Territorial FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Last Updated:** 2026-08-24 — the dashboard's camera-trap numbers are real, and its two time-of-day charts stopped inventing activity
**What Changed:** Three fixes at the DuckDB boundary. (1) **Occupancy divides by the deployed
grid, not the station registry.** A species seen at 12 of the 21 stations actually deployed
in a campaign was being reported as 12/27. The denominator is now one function,
`_n_stations_deployed()`, and **every endpoint that returns an occupancy count also returns
its own denominator**, so the frontend never picks one — `CamarasTab.jsx` had been dividing
by `geo.camera_trap_count` and disagreeing with its own tooltip. The dead `totalStations`
prop is gone. (2) **`/diel-activity` and `/overlap` read `ct_observations_time_admissible`**,
not `ct_observations`: 33 animal rows carry a trustworthy date and an untrustworthy time of
day, and bucketing them by `HOUR()` placed activity at hours nothing was photographed. Species
totals, `last_seen` and occupancy stay on the full table — presence does not need a clock.
(3) **`bootstrap_windows_db.py` deleted** — it carried an inline copy of the schema that was
wrong in both columns and types, and `CREATE TABLE IF NOT EXISTS` meant running it first on a
fresh machine would have silently poisoned the real schema. Superseded by
`data-pipeline/src/recovery.py`. Stations are spelled `CT01`..`CT27` throughout; the `TC-01`
dialect is gone.
**Prior — 2026-06-02** — added piso-vegetacional GeoJSON layer to Observatorio
**What Changed:** New toggleable Leaflet layer on Observatorio: photointerpretation of Bosque Pehuén's vegetational floor (48 polygons, colored by BIOTOPO across 3 semantic groups — greens for Bosque, ochres for Renoval, blues/violets for Matorral/Pradera/Estepa). Layer is **off by default**; click a polygon for `BIOTOPO / Distrito / Superficie`. Self-contained component (`PisoVegetacionalLayer.jsx`) — adding more GIS layers later follows the same shape. Conversion is reproducible via `scripts/convert_piso_vegetacional.py` (shapefile → GeoJSON, UTM 18S → WGS84, latin-1/utf-8 quirks handled).
**Integration Status:** Ready [Observatorio map + piso vegetacional overlay; 26 TCs from canonical stations.yaml; species classification via API; modular frontend per W41; CSS-Modules styling per W44; schema-drift visible at boot per S49] | Pending [`battery_voltage` curation in `ALLOWED_COLS` — surfaced by S49 drift check; cámaras trampa dashboard tab polish; fauna tab real data; Asistente real Claude API]
**Blockers/Notes:** Piso-vegetacional palette approved visually but Bosque Achaparrado / Bosque Abierto may need a stronger split if it blurs in field use — colors live at the top of `PisoVegetacionalLayer.jsx`, single edit. Deferred items unchanged: S58 needs Felipe's field records (TC-11 vs TC-18 SD-card duplication in stations.yaml); S76 re-opens when CI lands.

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
| Observatorio | Real data | Leaflet + Esri satellite, BP boundary polygon, 26 camera markers, piso vegetacional overlay (toggle, off by default) |
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
- `data/piso_vegetacional.geojson` — photointerpretation of vegetational floor (48 polygons, BIOTOPO + DISTRITO + Supe). Regenerable from the source shapefile (`data/piso_vegetacional_source/veg_foto_BP.*`) via `scripts/convert_piso_vegetacional.py`.
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
