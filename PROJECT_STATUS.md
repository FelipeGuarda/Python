# FMA Project Status

**Last updated:** 2026-03-18
**Owner:** Felipe Guarda — Fundación Mar Adentro
**Field site:** Bosque Pehuén, La Araucanía, Chile (-39.44°, -71.74°)

---

## Two-Machine Architecture

| Machine | Role | OS |
|---|---|---|
| **Laptop** (PopOS Linux) | Code, DuckDB, React frontend, data pipeline, platform | Linux |
| **Office desktop** (Windows) | Raw image analysis: MegaDetector, CLIP, Timelapse2 review | Windows |

**The split:** GPU-dependent raw analysis (image classification, object detection) runs on the Windows desktop. Once a reviewed CSV or DuckDB file is produced, everything downstream is cross-platform and runs on the laptop. Camera-traps code has intentional Windows paths for this reason.

---

## Project Overview

| Project | Directory | Stage | What it does |
|---|---|---|---|
| Data Pipeline | `data-pipeline/` | Built, needs first production run | DuckDB ingestion: CR800 weather, Open-Meteo, camera trap CSVs |
| Plataforma Territorial | `plataforma-territorial/` | Frontend working, backend not started | React app: map, dashboard, AI chat, reports |
| Camera Traps | `camera-traps/` | Working pipeline, Otoño 2025 pending | CLIP classification + Streamlit review UI |
| Literatura Agent | `literatura-agent/` | Complete and deployed | Weekly paper fetch + Claude summary → email |
| Schedule Agent | `schedule-agent/` | Complete and deployed | Monday scheduling: Google Tasks → Claude → Calendar |
| Visualizaciones | `visualizaciones-artisticas/` | Volumetric bird songs done, others planned | Generative art from field data |

---

## Detailed Status by Project

### 1. Data Pipeline (`data-pipeline/`)

All 5 build phases complete. Code is production-ready.

| Component | Status | Notes |
|---|---|---|
| DuckDB schema (6 tables) | Done | `fma_data.duckdb` exists at 42 MB |
| Open-Meteo fetcher | Done | Working, schedulable every 60 min |
| Camera trap legacy parser | Done | Parses Timelapse2 CSV (18,473 rows) |
| Camtrap DP parser | Done | Awaiting test data from real DP package |
| TOA5 parser (CR800 backfill) | Done | For historical .dat files |
| CR800 live fetcher | Done | **Needs first connection test via Tailscale** |
| File watcher daemon | Done | Monitors `data/incoming/` |
| APScheduler daemon | Done | Open-Meteo hourly, CR800 weekly |

**Next action:** Test CR800 connection (`100.97.202.90:2000`) via Tailscale, run `list_tables()` to discover table names.

### 2. Plataforma Territorial (`plataforma-territorial/`)

React/Vite frontend with 4 pages. Observatorio page has a real Leaflet map with satellite imagery, boundary polygon, and 25 camera trap markers. Other 3 pages use mock data.

| Component | Status | Notes |
|---|---|---|
| Observatorio (map) | Real data | Leaflet + Esri satellite + boundary.geojson + 25 cameras |
| Dashboard (charts) | Mock data | Fire risk, meteo, cameras, fauna tabs |
| Asistente (AI chat) | Mock data | Placeholder responses |
| Reportes (newsletter) | Mock data | Draft generator with typing animation |
| FastAPI backend | Not started | 6 endpoints planned (see README) |
| Station coordinates | Done | `data/stations.yaml` + GeoJSON files |
| BP boundary polygon | Done | **Under review — confirm delimitation** |

**Next action:** Build FastAPI backend skeleton (Phase 1.3 in NEXT_STEPS.md), then connect weather dashboard to real DuckDB data.

**Branch:** `feature/real-map` has the Leaflet map implementation (ahead of `main`).

### 3. Camera Traps (`camera-traps/`)

CLIP classification pipeline and Streamlit review UI are production-quality. Primavera 2025 data has been processed. Otoño 2025 images retrieved but not yet classified.

| Component | Status | Notes |
|---|---|---|
| MegaDetector integration | Done | Via AddaxAI on Windows desktop |
| CLIP classification | Done | `run_classification.py` |
| Streamlit review UI | Done | `phase1_labeling/app.py` |
| GIS data (KML → GeoJSON) | Done | Boundary + 25 station coordinates extracted |
| Otoño 2025 classification | Pending | Images on desktop, needs MegaDetector → CLIP → review |
| EfficientNetV2 fine-tuning | Planned | Needs enough reviewed data (≥50 images/species) |

**Next action:** Process Otoño 2025 through the pipeline on the office desktop.

**Note:** `config.yaml` and `NEXT_SESSION.md` have Windows paths — this is intentional (raw analysis runs on Windows desktop).

### 4. Literatura Agent (`literatura-agent/`)

Complete and running on weekly cron. Fetches papers from arXiv, SciELO, PMC, CORE, OpenAlex. Summarizes in Spanish via Claude Haiku. Sends HTML email.

**Status:** Deployed, no action needed. Has its own improvement roadmap (relevance scoring, feedback loop).

### 5. Schedule Agent (`schedule-agent/`)

Complete and deployed. Monday scheduling: reads Google Tasks → Claude generates weekly plan → creates Google Calendar events → Flask approval UI.

**Status:** Deployed, no action needed.

### 6. Visualizaciones Artísticas (`visualizaciones-artisticas/`)

Generative art from field data. Volumetric bird songs visualization is complete. Other pieces (Retrato Diario, Constelación de Especies) are planned but awaiting real DuckDB data.

**Status:** Waiting on data pipeline + platform to provide real data.

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

## Open Items / Reminders

- [ ] **TC-26 coordinates** — grid 22, SD M23. Spreadsheet has wrong coords (30 km off). Get correct GPS from field team.
- [ ] **BP boundary delimitation** — polygon under review. Confirm which version to use.
- [ ] **CR800 Tailscale test** — first live connection to `100.97.202.90:2000`.
- [ ] **Otoño 2025 camera trap processing** — images on desktop, needs full pipeline run.
- [ ] **Flora plot coordinates** — not yet available.
- [ ] **FastAPI backend** — next major development task for the platform.
- [ ] **Fire risk dashboard port** — logic lives at `Estacion meteorologica/Fire risk dashboard/`, needs porting to FastAPI.

---

## File Map (what lives where)

```
/home/fguarda/Dev/Python/                ← git repo root
├── PROJECT_STATUS.md                    ← THIS FILE — consolidated status
├── GIT_WORKFLOW_GUIDE.md                ← git workflow reference
├── fma_data.duckdb                      ← central database (42 MB)
│
├── camera-traps/                        ← image analysis pipeline (Windows + Linux)
│   ├── README.md                        ← project docs
│   ├── NEXT_SESSION.md                  ← Windows desktop session notes
│   ├── GIS/                             ← source KML/Excel files
│   └── old animal data DB.csv           ← legacy Timelapse2 export (18,473 rows)
│
├── data-pipeline/                       ← DuckDB ingestion service
│   ├── README.md                        ← project docs (updated)
│   ├── BUILD_CONTEXT.md                 ← original build spec (reference)
│   ├── config.yaml                      ← runtime config
│   ├── schema.sql                       ← table definitions
│   ├── run_fetch.py                     ← scheduler/CLI entry point
│   ├── run_watcher.py                   ← file watcher entry point
│   └── src/                             ← all pipeline code
│
├── plataforma-territorial/              ← React platform
│   ├── README.md                        ← project docs (updated)
│   ├── NEXT_STEPS.md                    ← detailed phase plan
│   ├── data/                            ← GeoJSON + stations.yaml
│   └── plataforma-demo/                 ← React/Vite app
│
├── literatura-agent/                    ← weekly paper summarizer (deployed)
├── schedule-agent/                      ← Monday scheduler (deployed)
└── visualizaciones-artisticas/          ← generative art pieces
```
