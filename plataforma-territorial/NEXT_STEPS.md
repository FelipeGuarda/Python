# Plataforma Territorial — Next Steps

**Date:** 2026-03-13
**Scope:** Weather station dashboard + camera trap integration (approved for development)

---

## Two-Machine Architecture

| Machine | Role | OS | Connection |
|---|---|---|---|
| **Laptop** (PopOS Linux) | Code repos, DuckDB, React frontend, pipeline | Linux | Direct |
| **Office desktop** (Windows) | Raw image analysis: MegaDetector, CLIP, image review | Windows | Tailscale VPN |
| **Synology NAS** | Long-term image storage | — | Tailscale VPN |
| **CR800 datalogger** (Bosque Pehuén) | Live weather sensor data | — | Tailscale at `100.97.202.90:2000` |

**Split:** Raw data analysis (GPU-dependent, Windows-only tools like AddaxAI) runs on the office desktop. Once a reviewed CSV/DuckDB output is produced, everything downstream (pipeline ingestion, platform, reports) is cross-platform and runs on the laptop.

---

## Phase 1 — Data Foundations (can start now, on laptop)

### 1.1 Fix data-pipeline configuration

The pipeline is built but `config.yaml` has a Linux DB path. Update for current environment.

**Files:**
- `data-pipeline/config.yaml` — update `database.path`
- `data-pipeline/src/fetchers/cr800.py` — test connection to CR800 via Tailscale

**Test:** Run `python run_fetch.py` and verify new weather data lands in DuckDB.

### 1.2 Add boundary polygon and station coordinates — DONE

Completed 2026-03-18. KML files converted to GeoJSON, Excel field data parsed:
- `data/boundary.geojson` — reserve polygon (86 vertices)
- `data/camera_trap_stations.geojson` — 25 cameras with real field coordinates
- `data/stations.yaml` — weather station (centroid) + 25 cameras
- TC-26 excluded (erroneous coordinates in spreadsheet — TODO: get correct coords)
- **NOTE:** BP boundary delimitation is under review — confirm which polygon to use

### 1.3 Build FastAPI backend skeleton

The React frontend exists with mock data. The backend connects it to real data.

**Priority endpoints (weather + stations first):**
```
GET /api/stations              → reads stations.yaml, returns GeoJSON
GET /api/weather/history       → full CR800 timeline from DuckDB (weather_station table)
GET /api/weather/current       → latest readings
GET /api/weather/forecast      → Open-Meteo 7-day forecast (weather_forecast table)
```

**Files to create:**
- `plataforma-territorial/backend/main.py`
- `plataforma-territorial/backend/db.py`
- `plataforma-territorial/backend/routers/weather.py`
- `plataforma-territorial/backend/routers/stations.py`
- `plataforma-territorial/backend/requirements.txt`

### 1.4 Data backup architecture (CSV archive)

DuckDB is the single source of truth for downstream apps, but some data enters DuckDB with **no intermediate file** (CR800 live fetches, Open-Meteo forecasts, literature agent results). If the database corrupts, that data is gone. Additionally, management needs browseable backups in standard formats (CSV/Excel).

**Two mechanisms:**

**A. Archive-on-ingest** — every time data enters DuckDB, also save a dated CSV:
```
data-pipeline/data/archive/
├── weather_station/
│   ├── cr800_2026-03-07_to_2026-03-14.csv
│   ├── cr800_2026-03-14_to_2026-03-21.csv
│   └── ...
├── weather_forecast/
│   ├── open_meteo_2026-03-14.csv
│   └── ...
├── camera_traps/
│   ├── primavera_2025_reviewed.csv
│   ├── otono_2025_reviewed.csv
│   └── ...
└── literature/
    ├── week_2026-11.csv
    └── ...
```

Implementation: ~10 lines per fetcher — after the DuckDB upsert, `df.to_csv(archive_path)`. Files are plain CSV, openable in Excel, organized by date and data type.

**B. Periodic full export** — a script that dumps every DuckDB table to CSV:
```bash
python run_export.py
# → exports/weather_station_2026-03-13.csv
# → exports/ct_observations_2026-03-13.csv
# → exports/weather_forecast_2026-03-13.csv
# → ...
```

Run monthly as a "nuclear backup" — the complete database as plain files.

**Files to create/modify:**
- `data-pipeline/src/archive.py` — helper: `save_to_archive(df, category, date_label)`
- `data-pipeline/run_export.py` — full DuckDB → CSV dump
- Modify each fetcher (`cr800.py`, `open_meteo.py`) and ingestion step to call `save_to_archive` after upsert

**Long-term storage:** The `data/archive/` folder should live inside a Synology Drive–synced directory so backups automatically replicate to the NAS.

### 1.5 Schedule CR800 weekly fetch

Once `run_fetch.py` is verified working, schedule it:
- Windows: Task Scheduler (for now)
- Linux (after migration): cron

Already configured in `config.yaml` as `cr800_interval_minutes: 10080` (weekly).

---

## Phase 2 — Weather Dashboard (laptop, after Phase 1)

### 2.1 Read and catalog the consulting firm's Streamlit app

**Repo location:** `C:/Dev/Python/tres_hermanas_sitio/`

**Features to extract (confirmed):**
- Variable selector (choose which sensor variables to display)
- Full-period timeline (entire station history)
- Custom date range picker (zoom into specific periods)
- Date comparison (overlay two different date ranges on the same chart)

**Migration target:** React Dashboard tab in `plataforma-demo/src/App.jsx`, using Recharts.

### 2.2 Build weather dashboard tab in React — DONE (2026-03-19)

Implemented in `feature/weather-dashboard` branch.

**Delivered:**
- Variable selector (10 variables + wind toggle)
- Date range picker with Apply button (prevents excessive fetching)
- Resolution selector (15min, daily, monthly, seasonal)
- Per-variable line/bar charts (Recharts)
- Wind rose SVG (custom, 16 directions × 6 speed bins)
- Current conditions strip (latest reading from CR800)
- Stats table (mean, min, max per variable)

**Pending — Comparison mode** (intentionally deferred):
- Two-period overlay: select Period 1 and Period 2, show side-by-side charts
- Reason deferred: significant UI complexity, lower priority for v1
- When to tackle: after basic dashboard is validated with real users

### 2.3 Port fire risk dashboard to the platform

The fire risk logic lives at `Estacion meteorologica/Fire risk dashboard/`. Port the computation, not the UI.

**Backend:**
- `backend/routers/fire_risk.py` — new endpoint
- `backend/risk/calculator.py` — copy `risk_calculator.py` logic
- `backend/risk/fire_model.pkl` — copy trained model

**Endpoint:**
```
GET /api/fire-risk/current     → today's rule-based index + ML probability
GET /api/fire-risk/forecast    → 14-day forecast risk scores
```

**Frontend:** New "Riesgo de Incendio" tab in Dashboard:
- Dual gauge (rule-based index + ML probability) — recreate from `visualizations.py`
- Polar contribution plot (4 variables)
- 14-day forecast bar chart (color-coded by risk level)
- Wind direction compass
- Agreement indicator between rule-based and ML methods

**Reference files to read:**
- `Estacion meteorologica/Fire risk dashboard/app.py` (489 lines — main layout)
- `Estacion meteorologica/Fire risk dashboard/risk_calculator.py` (scoring logic)
- `Estacion meteorologica/Fire risk dashboard/visualizations.py` (chart definitions)
- `Estacion meteorologica/Fire risk dashboard/config.py` (risk bins, colors, geographic bounds)
- `Estacion meteorologica/Fire risk dashboard/map_utils.py` (regional wind map with pydeck)

---

## Phase 3 — Camera Trap Integration (requires office desktop)

### 3.1 Ingest reviewed CSVs into DuckDB

The bridge between `camera-traps/` and the platform. The DuckDB schema already has `ct_deployments`, `ct_media`, `ct_observations` tables (`data-pipeline/schema.sql`). What's missing is the ingestion code.

**Build:**
- `data-pipeline/src/parsers/camera_trap_reviewed.py` — reads `new_labeled_data_reviewed.csv`, maps columns to DuckDB schema
- Add to `data-pipeline/src/ingest.py` — orchestrate parse → validate → upsert
- Support multi-campaign: process all reviewed CSVs, deduplicate by `observationID`

**Workflow for each new campaign:**
1. Run MegaDetector → CLIP classification → human review (existing pipeline)
2. Drop `new_labeled_data_reviewed.csv` into `data/incoming/` (or run ingestion manually)
3. Pipeline parses and upserts into DuckDB
4. Platform sees new data immediately

**Must be tested on office desktop** with access to actual images and reviewed CSVs.

### 3.2 Thumbnail extraction pipeline

Camera trap images are large JPGs. The platform can't serve full-resolution images over Tailscale for map popups. Pre-extract thumbnails.

**Approach:**
- After ingestion, generate 300px-wide thumbnails for a subset of images per station (e.g., 3–5 best per species per station)
- Store in `plataforma-territorial/data/thumbnails/` (or a shared location on NAS)
- Reference paths stored in DuckDB `ct_media.thumbnail_path`

**Tailscale dependency:** For long-term, thumbnails are served from NAS via Tailscale. For development/demo, pre-extracted thumbnails in the repo are fine.

### 3.3 Camera stations on the Observatorio map

Add camera trap markers to the map (Leaflet or Mapbox GL JS).

**Interaction:**
- Markers at each station from `stations.yaml`
- Click → popup with:
  - Station name + deployment dates
  - Species detected (count per species)
  - 2–3 thumbnail images (most recent or most interesting detections)
  - Link to filter Dashboard by that station

**Depends on:** 3.1 (data in DuckDB) + 3.2 (thumbnails) + stations.yaml coordinates.

### 3.4 Camera trap Dashboard tab

Recreate CamtrapR-style analytics in React:

| Visualization | Description | Data source |
|---|---|---|
| Species frequency bar chart | Which species appear most/least | `ct_observations` |
| Diel activity patterns | 24-hour activity curves per species | `ct_observations` (timestamp hour) |
| Species accumulation curve | New species discovered over time | `ct_observations` (ordered by date) |
| Detection heatmap | Station × species matrix | `ct_observations` grouped |
| Temporal trends | Detections per month/season | `ct_observations` grouped |
| Sample images | Filterable by species/station | `ct_media` + thumbnails |

**Backend endpoint:**
```
GET /api/detections/summary    → aggregated stats for charts
GET /api/detections/recent     → latest N observations with thumbnails
GET /api/detections/by-station → observations grouped by station
GET /api/detections/by-species → observations grouped by species
```

### 3.5 Run Otoño 2025 through CLIP pipeline

Images are retrieved and on the office desktop. When at the office:
1. Fix unicode filenames if needed (`setup/fix_unicode_filenames.py`)
2. Flatten folder structure (`setup/flatten_for_camtrapdp.py`)
3. Run MegaDetector v5b via AddaxAI (junction already exists: `C:\ADDAX\Otono_2025`)
4. Run CLIP classification (`python run_classification.py`)
5. Human review (`streamlit run phase1_labeling/app.py`)
6. Ingest reviewed CSV into DuckDB (step 3.1)

---

## Phase 4 — Phase 2 Classifier (after enough reviewed data)

### 4.1 Evaluate data quantity

After Primavera 2025 + Otoño 2025 are both reviewed and ingested:
- Count reviewed images per species
- Species with ≥ 50 images → ready for training
- Rare species (Puma, Guiña, Monito del monte) → may need data augmentation or a "rare" catch-all class

### 4.2 Train EfficientNetV2

- Use all reviewed images as training data
- Compare accuracy against CLIP baseline
- If better → replace CLIP in the pipeline for future campaigns
- Details in `camera-traps/NEXT_SESSION.md`

---

## Dependencies and Blockers

```
Phase 1 ──→ Phase 2 (weather dashboard needs backend + data)
Phase 1 ──→ Phase 3 (camera integration needs backend + stations.yaml)

Phase 3.1–3.5 require office desktop (images)
Phase 2.1 ready — consulting firm's repo at `C:/Dev/Python/tres_hermanas_sitio/`
Phase 1.2 DONE — coordinates added from KML/Excel (2026-03-18)
Phase 1.4 archive-on-ingest should be built alongside 1.1 (same fetcher files)

Everything else can start immediately on the laptop.
```

---

## What to Provide (checklist for Felipe)

- [x] Consulting firm's Streamlit repo → available at `C:/Dev/Python/tres_hermanas_sitio/`
- [x] Bosque Pehuén boundary polygon → converted from areaBP.kml to GeoJSON (2026-03-18)
- [x] Camera trap station coordinates → parsed from Excel field data, 25 of 26 cameras (2026-03-18)
- [x] CR800 weather station coordinates → using park centroid -39.4417, -71.7420 (2026-03-18)
- [ ] TC-26 correct coordinates (grid 22, SD M23) → erroneous in spreadsheet
- [ ] Confirm BP boundary delimitation (under review)
- [ ] Flora plot coordinates (if available)
- [ ] Confirm Tailscale VPN to CR800 (`100.97.202.90:2000`) is active

---

## Starting Prompt for Next Session

> "We're working on `C:/Dev/Python/plataforma-territorial`. Read NEXT_STEPS.md first. The goal is Phase 1: fix the data-pipeline config, build the FastAPI backend skeleton, and add station coordinates. If the consulting firm's repo is downloaded, also start Phase 2.1."
