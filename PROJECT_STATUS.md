# FMA Project Status

**Last updated:** 2026-06-16 (camera-traps: Phase-1 review UI overhauled. Streamlit review now shows a 3-col triptych per cell — `[anterior | actual | siguiente]` — sourced from the MD JSON so burst context (incl. empty triggers) is visible. Thumbnails switched from bbox crops to full frames at higher resolution (1280px / Q85) so habitat/scale context survives the expand-view. **CLIP classifier still uses the bbox crop** — only the human review UI changed. New resume loader rehydrates `confirmed` / `outcomes` from `new_labeled_data_reviewed.csv` on a fresh session and auto-jumps to the first incomplete species batch — the exported CSV is now the durable checkpoint. Added Vaca (*Bos taurus*) to `data-pipeline/species.yaml` (28 CLIP species now; was 27) — cows were being misclassified as Caballo on BP Mayo 2026. Renamed `crop_utils.py` → `cropping.py` (file was cohesive, `_utils` suffix unjustified). Fixed a latent `ModuleNotFoundError` when launching Streamlit from the project root via 2-line sys.path injection. Pending commit + push.)
**Owner:** Felipe Guarda — Fundación Mar Adentro
**Field site:** Bosque Pehuén, La Araucanía, Chile — reserve center -39.4417°, -71.7420° (canonical: `plataforma-territorial/data/stations.yaml` → `reserve.center`)

---

## 🏁 Milestone — First full code review complete (2026-05-11)

The independent code review of the three actively-developed projects (`data-pipeline`, `camera-traps`, `plataforma-territorial`) — started 2026-04-21 — is finished. **Every finding is closed or explicitly deferred with a re-open trigger.** No silently-open work remains.

- **Resolved:** 1 Critical · all Warnings · most Suggestions
- **Closed-rejected (with rationale):** S57, S47, S14, S72
- **Deferred (with re-open conditions):** S58 (field verification), S76 (Vitest tests, until CI exists)

Full narrative: `CHANGELOG.md` (top entry) + `~/Documents/Obsidian FG/SecondBrain/Sessions/2026-05-11-fma-ecosystem-code-review-tier-1-and-tier-2.md`. Per-project snapshots: `~/Documents/Obsidian FG/SecondBrain/Reviews/review-state-*.md`.

---

## Two-Machine Architecture

| Machine | Role | Projects |
|---|---|---|
| **Personal laptop** (PopOS Linux) | Code, DuckDB, data pipeline, React frontend + FastAPI backend | `data-pipeline/`, `plataforma-territorial/`, `literatura-agent/`, `visualizaciones-artisticas/` |
| **Office desktop** (Windows → future Linux) | GPU-dependent raw analysis: MegaDetector, CLIP, Timelapse2 review | `camera-traps/`, `plataforma-territorial/` Phase 3 only (camera tab) |

**Rule:** Platform repo is shared, but Phase 3 camera tasks = office, everything else = home. Always commit before switching machines.

**Handoff protocol:**
- Office → Home: commit reviewed CSV → pull at home → run ingestor
- Home → Office: commit platform code → pull before starting Phase 3 work

### ⚠️ Pending: Migration to Linux at office

The personal laptop (Linux) will NOT be left on permanently — it's a personal machine. This means:
- **DuckDB (`fma_data.duckdb`), data-pipeline service, and plataforma backend must migrate to the office machine once it switches to Linux.**
- Until then, the platform can only be used from the laptop itself or when it's manually on and the service running.
- `bootstrap_windows_db.py` is a temporary workaround for Windows office — it will not be needed post-migration.

**Migration checklist (when office switches to Linux):**
- [ ] Copy `fma_data.duckdb` to office Linux machine
- [ ] Set up conda environment (`data-pipeline`, `plataforma-territorial`)
- [ ] Enable `fma-pipeline.service` and `fma-platform.service` as systemd user services
- [ ] Update Tailscale or local network access so CR800 fetcher still reaches the weather station
- [ ] Update `literatura-agent` cron to run on office machine
- [ ] Verify `schedule-agent` still works (Google API credentials)

---

## Resolved Decision: React/Vite is canonical (2026-05-12)

React/Vite + FastAPI is the definitive platform stack. Streamlit references in `plataforma-territorial/README.md` are stale and should be cleaned up next time the README is touched.

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

**Canonical catalogs (2026-04-27):** `species.yaml` (31 entries — 27 CLIP + invasive/priority flags) is the single source of truth across the ecosystem. Sibling loaders in camera-traps and plataforma-territorial/backend read this same file. Pairs with `plataforma-territorial/data/stations.yaml` (also now consumed end-to-end after Track B).

**Code review Batch A+B (2026-04-27):** 9 warnings resolved — dead deps removed (W9), `_STATE_PATH` centralised to `src/paths.py` (W15), `cr800_session()` context manager (W18), Open-Meteo/CR800 fault isolation in `run_once()` (W19), `open_meteo.py` DST-safe `tz_localize` (W16), `_process_raw` made public (W12), algorithmic DST dates in `recover_dst_gaps.py` (W13), silent `count=1` default fixed (W14), `toa5.py` column-drop now logged (W17). **2026-05-06:** S12 (`recover_dst_gaps.py` moved to `scripts/`, path resolution updated), S13 (`run_watcher.py` connection cleanup in `finally` block). Remaining: C1 + W8 (Batch E, needs Opus).

**Live data:** 264,944 rows weather_station · 168 rows weather_forecast · 7,652 rows ct_observations · 20,095 ct_media · 106 ct_deployments (Otoño 2025 + Primavera-verano 2025-2026 ingested 2026-04-15)

| Component | Status | Notes |
|---|---|---|
| DuckDB schema (6 tables) | Done | `fma_data.duckdb` ~42 MB |
| Open-Meteo fetcher | Done | Hourly, 16-day forecast (extended 2026-04-21) |
| Camera trap legacy parser | Done | Parses Timelapse2 CSV |
| Camtrap DP parser | Done | Awaiting test with real DP package |
| TOA5 parser (CR800 backfill) | Done | Column names fixed 2026-04-07 (RH_Avg, WindDir_Avg, incomingSW_Avg) |
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

**Code review (2026-04-21 → 2026-05-07):** Full review of Blocks 3-5 complete; artifacts in `~/Documents/Obsidian FG/SecondBrain/Reviews/`. Track B closed the station-registry + species-catalog cross-project chains (W11/W23/W32/W33/W47/W51 map-center half). **Track C closed (2026-04-29):** W41 App.jsx 1805→37 line decomposition into 24 modules. New endpoints: `/api/config/geography`, `/api/config/species`. Track A (CR800 backfill safety) still queued. **2026-05-06:** S52, S54, S60, S61, S74. **2026-05-07 (early):** S67 + W44 chart slice via `src/styles/chart.js`. **2026-05-07 (Track K — same day):** W44 fully closed via CSS Modules pass — 246 → 25 inline `style={{}}` sites (90% reduction); 19 new `.module.css` files; `src/styles/vars.css` exposes 13 `--color-*` palette properties. Same change closed S65 (Card + SectionLabel accept className), S77 (Asistente :hover), S78 (Reportes @media print). Build 1.91s, CSS 36.5 kB / 10.8 kB gzipped.

| Component | Status | Notes |
|---|---|---|
| Observatorio (map) | **Real data** | 23 canonical stations from DuckDB, species counts + thumbnails in popups; piso vegetacional overlay (2026-06-02) — 48 polygons by BIOTOPO, off by default |
| Dashboard — Meteo tab | **Real data** | Year of history, variable selector, wind rose, comparison mode |
| Dashboard — Fire risk tab | **Real data** | All visuals use Open-Meteo exclusively. Polar chart color matches gauge. Fixed 3-week bar chart (no navigation). Wind compass from forecast. Freshness timestamps on all widgets. |
| Dashboard — Cameras tab | **Real data** | Diel activity chart, summary stats, station grid — all from DuckDB |
| Dashboard — Fauna tab | **Real data** | Species bar chart + stats + priority/invasive alerts — all real |
| Asistente (AI chat) | Mock data | Placeholder responses |
| Reportes (newsletter) | Mock data | Draft generator with typing animation |
| FastAPI backend | **Working** | Serves API + built frontend from single port 8000 |
| Deployment | **Done** | Linux: `fma-platform` systemd + `plataforma`/`plataforma-stop` aliases |
| Station coordinates | Done | `data/stations.yaml` + GeoJSON files |
| BP boundary polygon | Done | **Under review — confirm delimitation** |

**FastAPI endpoints live:**
- `GET /api/weather/current`, `/history`, `/forecast`
- `GET /api/fire-risk/current`, `/forecast`, `/history?days=`
- `GET /api/detections/recent`, `/species-summary`, `/stations`, `/station-summary`, `/station-images/{id}`
- `GET /ct-images/<campaign>/stations/<id>/<file>` (static mount)
- `GET /api/health`

**Priority 1 — Connect frontend to real endpoints:**
- [x] Replace mock data in fire risk tab with real API calls ← done 2026-03-30
- [x] Observatorio map stations from real DuckDB data ← done 2026-04-15
- [x] Cámaras trampa dashboard tab — diel activity, summary stats, station grid ← done 2026-04-17
- [x] Fauna tab: real stats + priority/invasive species alerts ← done 2026-04-17
- [x] Resize thumbnails (Pillow 1000px in export_best_images.py) + lightbox in Observatorio popups ← done 2026-04-16 (commit 22f6a08)
- [ ] Cámaras tab Phase 3.4 extensions: species×station heatmap, image gallery
- [ ] Retrain `fire_model.pkl` with current scikit-learn (pickle incompatible → ml_probability returns null)
- [ ] Include ML index alongside rule-based index in fire risk view
- [x] Extended Open-Meteo forecast from 7 to 16 days — bar chart next week now populated ← done 2026-04-21

**Priority 2 — Asistente with real Claude API:**
- [ ] Connect Asistente tab to Claude API (Sonnet + tool use)
- [ ] Implement DuckDB query tools: current risk, recent detections, trends
- [ ] Each response with calculated values must cite its formula and input data (methodological transparency)

**Priority 3 — Observatorio: real map layers:**
- [x] Verify real coordinates for all camera stations and weather station ← confirmed 2026-05-12
- [x] Piso vegetacional (fotointerpretación) — 48 BIOTOPO polygons, toggle layer ← done 2026-06-02
- [ ] BP boundary delimitation — final polygon version pending (carried from Open Items)
- [ ] DISTRITO (geomorphological) layer — same source GeoJSON, 4-color physiographic classification; queued
- ~~Optional layers: fire risk zones, historical fire perimeters~~ — dropped 2026-05-12 (out of scope)

---

### 3. Camera Traps (`camera-traps/`) — FASE 1 OPERATIVA · Informe Anual 2025 publicado y corregido

CLIP classification pipeline and Streamlit review UI are production-quality. Both campaigns (Otoño 2025 + Primavera-verano 2025-2026) classified and reviewed. Species list now sourced from canonical `data-pipeline/species.yaml` via sibling loader (Track B, 2026-04-27) — 27 CLIP species + 4 non-CLIP entries (31 total).

**Last Updated:** 2026-06-03
**What Changed:** Creado `Anual-reports/source_code_CT_2025/` — bundle autocontenido (2.7 MB, 33 archivos) para que un colega de FMA replique el informe desde Drive sin clonar el superrepo `Python/`. Incluye `inputs/` con copia inmutable de los 7 archivos de entrada externos, scripts con paths relativos al bundle, `requirements.txt` pinneado, `render.ps1` para Windows, y `GUIA_REPLICACION.md` exhaustiva en español. Durante el smoke-test se descubrió y corrigió bug latente UTF-8 en `2025/py/*.py` (Windows cp1252 crashea con `→` y `ñ` en stdout); fix backporteado a los 4 scripts originales.
**Integration Status:** Ready — bundle smoke-testeado end-to-end (419 eventos, 6 figuras regeneradas). Pendiente subir a Drive de FMA.
**Blockers/Notes:** Pendiente decidir si re-mapear despliegue `100EK113` a CT5 (sospecha confirmada por foto pero no re-ingestado). Pendiente correr `render.sh` en Linux para regenerar el `.docx` con las cifras nuevas para el jefe. Bundle untracked en git — decidir si commit + Drive o sólo Drive (el usuario indicó borrar local tras subir).

| Component | Status | Notes |
|---|---|---|
| MegaDetector integration | Done | Via AddaxAI on Windows desktop |
| CLIP classification | Done | `run_classification.py` — CSV-only workflow, no DB dependency |
| Streamlit review UI | Done | `phase1_labeling/app.py` — handles empty filePath column |
| GIS data (KML → GeoJSON) | Done | Boundary + 26 station coordinates (TC-26 fixed 2026-03-30) |
| Otoño 2025 classification | Done | 697 animal obs reviewed |
| Primavera-verano 2025-2026 | Done | 500 animal obs reviewed |
| Species image export | Done | `export_best_images.py`: auto-discovers campaigns; 155 species images + 103 station images in `exports/` (gitignored); filenames traceable to source |
| **Informe Anual 2025** | **Done (v2)** | `Anual-reports/2025/` — markdown source, 6 figuras, pipeline reproducible, revisión visual aplicada (2026-06-02). 419 eventos, 11 especies, 22/26 CTs con detecciones. |
| EfficientNetV2 fine-tuning | Planned | Needs ≥50 reviewed images/species — now viable for common species |
| Otoño 2025 videos | Deferred | 2,593 videos, MegaDetector not run — process post-migration on Linux |

**Pre-migration GPU work: COMPLETE.** Both campaigns reviewed and exported. Ready for OS migration.

**Post-migration next steps (Linux):** Ingest reviewed CSVs into DuckDB (Phase 3.1 in plataforma-territorial); station thumbnails in `exports/*/stations/` are ready for `plataforma-territorial/data/thumbnails/`.

Note: `config.yaml` and `NEXT_SESSION.md` have Windows paths — intentional (raw analysis runs on Windows desktop).

---

### 4. Literatura Agent (`literatura-agent/`) — DEPLOYED

Weekly cron script. Fetches from arXiv, OpenAlex, SciELO, Semantic Scholar (PubMed and CORE removed 2026-05-05). Claude Haiku scores each paper 1–5 for FMA relevance and drops scores < 3 before summarizing. Sends HTML email in Spanish.

**Last updated:** 2026-05-06 — `.gitignore` added; `papers_dump.csv` untracked; Semantic Scholar API key confirmed in `.env`.

**Pending:** Gmail app password in `.env` · end-to-end dry-run (`python run.py`).

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

- [x] **TC-26 coordinates** — grid 22, SD M23. Spreadsheet has wrong coords (30 km off). Get correct GPS from field team.
- [ ] **BP boundary delimitation** — polygon under review. Confirm which version to use.
- [x] **Otoño 2025 camera trap processing** — done. Both campaigns reviewed, backed up, species images exported.
- [x] **Informe Anual 2025 — revisión visual ciervo/güiña aplicada (2026-06-02).** Ciervo rojo 7 → 1 cámara, Güiña 7 → 6 cámaras. Pipeline ampliado (`apply_verdicts.py`), informe re-escrito con sec. 1.6 documentando el protocolo. Ver `camera-traps/Anual-reports/2025/data/correction_log.txt`.
- [ ] **100EK113 → CT5 re-mapping.** Sospecha confirmada visualmente (sesión 2026-05-27 + verdicts 2026-06-02) pero aún no re-ingestado en DuckDB ni en los CSVs de campaña.
- [ ] **Render del DOCX del Informe Anual 2025 v2.** Requiere `pandoc` en Linux; correr `bash camera-traps/Anual-reports/2025/render.sh`.
- [ ] **Subir bundle `source_code_CT_2025/` a Drive de FMA** (2026-06-03). Zip de 2.7 MB autocontenido, listo para entregar al colega. Decidir antes si se commitea o sólo Drive + borrar local.
- [ ] **Flora plot coordinates** — not yet available.
- [ ] **Aves en BP/** — contains bird list comparison notebooks and Excel files. No README. Appears to be taxonomic reference data for camera trap species list. Document before using in platform.
- [x] **Meteo tab** — label fixed to "Última medición", wind rose moved below charts (larger).
- [x] **Comparison mode** — implemented. Two-period comparison with stacked charts, side-by-side wind roses, dual stats table.
- [x] **Fire risk backend** — `fire_risk.py` ported to FastAPI with real DuckDB data.
- [x] **Fire risk frontend** — tab connected to real API; polar plot, gauge, compass, 3-week bar chart with today indicator.

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
│   └── GIS/                             ← source KML/Excel files
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
