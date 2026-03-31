# FMA Project Status

**Last updated:** 2026-03-31
**Owner:** Felipe Guarda — Fundación Mar Adentro
**Field site:** Bosque Pehuén, La Araucanía, Chile (-39.61°, -71.71°)

---

## Two-Machine Architecture

| Machine | Role | Projects |
|---|---|---|
| **Laptop** (PopOS Linux) | Code, DuckDB, data pipeline, React frontend + FastAPI backend | `data-pipeline/`, `plataforma-territorial/`, `literatura-agent/`, `visualizaciones-artisticas/` |
| **Office desktop** (Windows) | GPU-dependent raw analysis: MegaDetector, CLIP, Timelapse2 review | `camera-traps/`, `plataforma-territorial/` Phase 3 only (camera tab) |

**Rule:** Platform repo is shared, but Phase 3 camera tasks = office, everything else = home. Always commit before switching machines.

**Handoff protocol:**
- Office → Home: commit reviewed CSV → pull at home → run ingestor
- Home → Office: commit platform code → pull before starting Phase 3 work

---

## Pending Decision: React vs Streamlit

The README for `plataforma-territorial/` describes a Streamlit app. What exists and works today is a React/Vite prototype (`plataforma-demo/`).

**Before building real modules, decide:** does the React prototype become the definitive version, or do we build the Streamlit version from the README?

- **React:** more visual control, better for the Observatorio (interactive map, animations), requires maintaining frontend + backend separately.
- **Streamlit:** faster to connect to Python/DuckDB, less friction for data modules, but visual limitations in the Observatorio.

---

## Dependency Chain

```
data-pipeline (writes DuckDB)
    ↓
plataforma-territorial/backend (reads DuckDB, serves API)
    ↓
plataforma-territorial/plataforma-demo (React frontend, consumes API)

camera-traps (produces reviewed CSVs)
    ↓
data-pipeline (ingests CSVs into ct_* tables)

visualizaciones-artisticas (reads DuckDB for art generation)
```

---

## Project Status

### 1. Data Pipeline (`data-pipeline/`) — OPERATIVO

Running as systemd service (`fma-pipeline.service`). Full pipeline with real data flowing.

**Live data:** 261,302 rows weather_station · 552 rows weather_forecast · 6,030 rows ct_observations (2,428 animals, 25 species) · 18,473 rows ct_media · 65 ct_deployments

| Component | Status | Notes |
|---|---|---|
| DuckDB schema (6 tables) | Done | `fma_data.duckdb` ~42 MB |
| Open-Meteo fetcher | Done | Hourly, 7-day forecast |
| Camera trap legacy parser | Done | Parses Timelapse2 CSV |
| Camtrap DP parser | Done | Awaiting test with real DP package |
| TOA5 parser (CR800 backfill) | Done | For historical .dat files |
| CR800 live fetcher | Done | Working via Tailscale VPN |
| File watcher daemon | Done | Monitors `data/incoming/` (not activated) |
| APScheduler daemon | Done | Open-Meteo hourly, CR800 weekly |
| systemd user service | **Done** | Enabled, starts on boot |

**Pending:**
- [ ] Tabla `literatura` pendiente de poblar (literatura-agent integration)
- [ ] Camtrap DP parser: test with real data
- [ ] Watcher de carpeta incoming: activate

---

### 2. Plataforma Territorial (`plataforma-territorial/`) — EN PROGRESO

React/Vite frontend with 4 pages. FastAPI backend operational with real endpoints.
**Access Linux:** `plataforma` alias → `http://localhost:8000` (systemd service).
**Access Windows:** `conda run -n plataforma-territorial uvicorn backend.main:app --port 8000`

**Two-machine data note:** DuckDB lives on Linux (written by data-pipeline service). On Windows, run `python bootstrap_windows_db.py` from `plataforma-territorial/` to seed a local DB with Open-Meteo data (90-day archive + 7-day forecast). Enough for Meteo and Riesgo tabs. No Tailscale needed.

**dist/ sync fix (2026-03-31):** Removed `dist` from `.gitignore`. Built frontend is now committed to git. Both machines get the same compiled UI via `git pull` — no per-machine rebuild needed.

| Component | Status | Notes |
|---|---|---|
| Observatorio (map) | Real data | Leaflet + Esri satellite + boundary.geojson + 25 cameras |
| Dashboard — Meteo tab | **Real data** | 90-day history, variable selector, wind rose, comparison mode |
| Dashboard — Fire risk tab | **Real data** | Rule-based FRI from real weather; forecast from Open-Meteo |
| Dashboard — Cameras/Fauna tabs | Mock data | Pending |
| Asistente (AI chat) | Mock data | Placeholder responses |
| Reportes (newsletter) | Mock data | Draft generator with typing animation |
| FastAPI backend | **Working** | Serves API + built frontend from single port 8000 |
| Deployment | **Done** | Linux: `fma-platform` systemd + `plataforma`/`plataforma-stop` aliases |
| Station coordinates | Done | `data/stations.yaml` + GeoJSON files |
| BP boundary polygon | Done | **Under review — confirm delimitation** |

**FastAPI endpoints live:**
- `GET /api/weather/current`, `/history`, `/forecast`
- `GET /api/fire-risk/current`, `/forecast`
- `GET /api/detections/recent`, `/species-summary`, `/stations`
- `GET /api/health`

**Priority 1 — Remaining:**
- [ ] Retrain `fire_model.pkl` with current scikit-learn (pickle incompatible → ml_probability returns null)
- [ ] Include ML index alongside rule-based index in fire risk view
- [ ] Replace mock data in cameras, fauna tabs with real API calls

**Priority 2 — Asistente with real Claude API:**
- [ ] Connect Asistente tab to Claude API (Sonnet + tool use)
- [ ] Implement DuckDB query tools: current risk, recent detections, trends
- [ ] Each response with calculated values must cite its formula and input data (methodological transparency)

**Priority 3 — Observatorio: real map layers:**
- [ ] Verify real coordinates for all camera stations and weather station
- [ ] Add optional layers: fire risk zones, historical fire perimeters

---

### 3. Camera Traps (`camera-traps/`) — FASE 1 OPERATIVA

CLIP classification pipeline and Streamlit review UI are production-quality. Primavera 2025 data processed. Otoño 2025 images retrieved but not classified.

| Component | Status | Notes |
|---|---|---|
| MegaDetector integration | Done | Via AddaxAI on Windows desktop |
| CLIP classification | Done | `run_classification.py` |
| Streamlit review UI | Done | `phase1_labeling/app.py` |
| GIS data (KML → GeoJSON) | Done | Boundary + 25 station coordinates |
| Otoño 2025 classification | **Done** | 694 animal obs reviewed, CSV backed up in git |
| Species image export | **Done** | Top 5/species/campaign via `export_best_images.py`; 131 images in `exports/` (gitignored) |
| EfficientNetV2 fine-tuning | Planned | Needs ≥50 reviewed images/species — now viable for common species |
| Otoño 2025 videos | Deferred | 2,593 videos, MegaDetector not run — process post-migration on Linux |

**Pre-migration GPU work: COMPLETE.** Both campaigns reviewed. Ready for OS migration.

**Post-migration next steps (Linux):** Run `export_best_images.py` after NAS re-download, then write DuckDB parser for reviewed CSVs.

Note: `config.yaml` and `NEXT_SESSION.md` have Windows paths — intentional (raw analysis runs on Windows desktop).

---

### 4. Literatura Agent (`literatura-agent/`) — DEPLOYED

Complete and running on weekly cron. Fetches papers from arXiv, SciELO, PMC, CORE, OpenAlex. Summarizes in Spanish via Claude Haiku. Sends HTML email.

**Status:** No action needed. Has its own improvement roadmap (relevance scoring, feedback loop).

---

### 5. Schedule Agent (`schedule-agent/`) — DEPLOYED

Monday scheduling: reads Google Tasks → Claude generates weekly plan → creates Google Calendar events → Flask approval UI.

**Status:** No action needed.

---

### 6. Visualizaciones Artísticas (`visualizaciones-artisticas/`) — EN ESPERA

Generative art from field data. Requires real DuckDB data. Volumetric bird songs visualization is complete.

- [ ] **Retrato Diario:** generative daily portrait of territory (risk + weather + detected species)
- [ ] **Constelación de Especies:** circular star map by activity time, distance by rarity
- [ ] **Río de Sonidos:** bird song visualization (requires audio files — see Acoustic Devices below)
- [ ] **Año Térmico:** circular calendar of annual temperature and risk

---

### 7. Dispositivos Acústicos — SIN CÓDIGO (datos aún no recuperados)

FMA has acoustic monitoring devices deployed in the field. Audio files not yet downloaded.

**Fase 1 — Recuperación e ingesta:**
- [ ] Download recordings from physical devices
- [ ] Define folder structure and naming convention (device, date, time)
- [ ] Add audio ingestor to data-pipeline: folder watcher → `acoustic` table in DuckDB (metadata only: device, timestamp, duration, file path)

**Fase 2 — Análisis de audio:**
- [ ] Species identification by vocalization — primary option: **BirdNET** (Cornell Lab, open source)
- [ ] Output: acoustic detections with species, confidence, timestamp → `acoustic_detections` table
- [ ] Integrate acoustic detections into pipeline alongside camera trap detections

**Fase 3 — Platform integration:**
- [ ] New "Acústica" tab in Dashboard or expand Fauna tab
- [ ] Acoustic device markers on Observatorio map
- [ ] Camera trap vs acoustic comparison for same species

Note: `visualizaciones-artisticas/` has the "Río de Sonidos" concept already designed, plus a reference project in `Volumetric bird songs/`. Audio files from this project feed those visualizations directly.

---

## Open Items

- [ ] **TC-26 coordinates** — grid 22, SD M23. Spreadsheet has wrong coords (30 km off). Get correct GPS from field team.
- [ ] **BP boundary delimitation** — polygon under review. Confirm which version to use.
- [x] **Otoño 2025 camera trap processing** — done. Both campaigns reviewed, backed up, species images exported.
- [ ] **Flora plot coordinates** — not yet available.
- [ ] **Aves en BP/** — contains bird list comparison notebooks and Excel files. No README. Appears to be taxonomic reference data for camera trap species list. Document before using in platform.
- [x] **Meteo tab** — label fixed to "Última medición", wind rose moved below charts (larger).
- [x] **Comparison mode** — implemented. Two-period comparison with stacked charts, side-by-side wind roses, dual stats table.
- [x] **Fire risk backend** — `fire_risk.py` ported to FastAPI with real DuckDB data.

---

## File Map

```
/home/fguarda/Dev/Python/                ← git repo root
├── PROJECT_STATUS.md                    ← THIS FILE
├── GIT_WORKFLOW_GUIDE.md                ← git workflow reference
├── fma_data.duckdb                      ← central database (~42 MB)
│
├── camera-traps/                        ← image analysis pipeline (Windows + Linux)
│   ├── README.md
│   ├── NEXT_SESSION.md                  ← Windows desktop session notes
│   ├── GIS/                             ← source KML/Excel files
│   └── old animal data DB.csv           ← legacy Timelapse2 export
│
├── data-pipeline/                       ← DuckDB ingestion service
│   ├── README.md
│   ├── BUILD_CONTEXT.md                 ← original build spec (reference)
│   ├── config.yaml
│   ├── schema.sql
│   ├── run_fetch.py                     ← scheduler/CLI entry point
│   ├── run_watcher.py                   ← file watcher entry point
│   └── src/
│
├── plataforma-territorial/              ← React platform + FastAPI backend
│   ├── README.md
│   ├── data/                            ← GeoJSON + stations.yaml
│   ├── backend/                         ← FastAPI (weather, fire risk, detections)
│   └── plataforma-demo/                 ← React/Vite app
│
├── literatura-agent/                    ← weekly paper summarizer (deployed)
├── schedule-agent/                      ← Monday scheduler (deployed)
└── visualizaciones-artisticas/          ← generative art pieces
```
