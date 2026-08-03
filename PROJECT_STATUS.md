# FMA Project Status

**Last updated:** 2026-07-31 (camera-traps: **flatten now preserves capture order; `camtrap/clocks.py` implements the segment-aware repair rule.** Felipe resolved the two blockers from 2026-07-30: (1) **the Synology originals are untouched**, so every campaign except otoño 2026 can be re-downloaded with its DCIM subfolders intact — the ordering evidence is *not* lost; otoño 2026 was flattened before upload and has **no backup**; (2) the hard-fail export gate stays hard ("getting everything is more important than getting results now"). That put the flatten fix on the critical path, because re-downloading with the old script would destroy the evidence a second time. **Delivered:** `setup/flatten_for_camtrapdp.py` writes a `dcim_manifest.csv` sidecar (deployment, dcim_folder, original/flat name, size, mtime, action) — no renames, so existing `file_name` joins are unaffected — appended per deployment so an interrupted run still describes its moves, and including already-flat files so a *partial* manifest is visible instead of silently misleading. It **no longer treats same-name/same-size as a duplicate to skip**, which is precisely the signature of a reset-clock camera re-emitting `0101xxxx` names, and a conservation check now aborts the run if a deployment ends up short. `--dry-run` predicts renames correctly instead of reporting zero. **New `camtrap/clocks.py`** owns segments, order evidence (`dcim_manifest+counter` > `counter` > `none`), coherence, and the rule *a segment is repairable iff coherent AND containing ≥1 anchor*; adds `valid_effort` as a **station-level** flag (a camera dead at an unknown date leaves the effort *denominator*, not just the numerator); detects splits from capture order **or** the deployment window, which is what makes a forward jump visible where `year < 2024` was blind. Key design call: **failing the ordering precondition does not condemn a camera** — an in-window sequence whose filenames agree with their own stamps demonstrably never reset, which is what keeps otoño 2026's five flattened wrap cameras usable. **25 fixtures** (`tests/test_clocks.py`, stdlib unittest so it runs on both machines) cover scenarios A–G, both preconditions, forward jumps, video exclusion, ambiguous anchors and the partial manifest. **Validated on real data:** CT_18 otoño 2026 returns 5 segments (10/32/40/3/227) matching the hand analysis, with incoherence localised to segment 4 and the install anchor correctly rejected as falling inside no segment. Also removed the stale nested `camera-traps/.git` (April 2026, no remote) that made every `git` call inside that folder report the July work as untracked; its history is bundled in `~/Dev/_archive/` together with three files that existed nowhere else. **Next:** `timestamps.py` still uses `classify_epochs` and one offset per station — handoff steps 2–5. **Open:** the export gate's exact rule, since `unclassified` doubles as `empty` and would let today's category-less otoño 2026 export pass. Pending commit + push.)

**Prior — 2026-07-30** — later session (camera-traps: **CRITICAL — clock-reset diagnosis was structurally blind; CT18 contaminated the pehuen analysis.** Our protocol for deciding whether a broken camera clock can be repaired diagnosed resets from the **animal-only** CSV. Otoño 2026's all-images export (12,068 rows vs 1,785) shows **CT18 reset its clock 4 times, not once** — so the shipped `last_real_proxy` single offset (+3329 d) stamps segments 1–3 as starting 2026-02-12 when they actually occurred in Nov/Dec 2025, and 44.8 d is unaccounted. The anchor's own note ("2 real-time photos before reset") was itself written from the blind view — the camera ran correctly for 9.4 days / 10 photos. **CT18's install anchor is uncorroborated:** it asserts 2025-11-14 offset-zero, but the first image on the card is an animal at 2025-11-19 06:41, while every other camera's first frame is a midday `unclassified` install photo of the technician — invisible because nobody labels people. **Spill: annual report is CLEAN** (`REPORT_CAMPAIGNS` excludes otoño 2026), but **pehuen has 65 focal-species records with fabricated dates** (jabalí 33, culpeo 16, liebre 12, perro 2, puma 2 — exactly 1 trustworthy); `record_table`'s `valid_time_of_day` filter protected activity/overlap, but `valid_date` was never consumed, so `02_detection_summary` and `06_seasonal_detection_maps` are contaminated. **Blind spot covers all four campaigns** — only otoño 2026 has an all-images export, so every existing `unrepairable_pending` note (otoño 2025 CT15/CT16/CT19, primavera/pv CT16) is a lower bound from animal-only data. **Spec agreed** (Felipe rejected both of my first two criteria): two preconditions — ordering established, segment coherent — one rule — *a segment is repairable iff coherent AND containing ≥1 anchor* — and three independent flags `valid_date` / `valid_time_of_day` / **new `valid_effort`**. Full-category export (empty/animal/person/vehicle) becomes a hard ingest gate. **No code written**; complete handoff in `camera-traps/docs/HANDOFF-clock-repair.md`. **Blockers:** (1) DCIM-subfolder question — the filename counter is per-folder and wraps at 999, `RelativePath` keeps only the deployment name, and `flatten_for_camtrapdp.py:resolve_dest` discards the subfolder except on filename collision, so capture order may be unrecoverable for the five cameras with >999 images; (2) Felipe's field notebook for CT18's install/visit dates and whether the older campaigns have install photos at all. **Supersedes the next entry's claim** that CT15/CT16/CT19 need no field notebook — the filename-MMDD trick works only for year-only errors and must be verified per camera. Pending commit + push.)

**Prior — 2026-07-30** (camera-traps: **canonical observation table implemented; annual report rewired.** New `camtrap/` boundary package — `stations.py` owns the canonical station convention `CT01`–`CT27` (historical spellings in `data/campaigns/station_aliases.csv`, data not code) and `observations.py` owns the canonical table, emitted by `timestamps.py` as `observations.parquet` beside the existing `_corrected.csv`. `01_data_prep.py` now reads it via `read_campaigns()`; ~190 lines of duplicated clock repair, station parsing and species recovery deleted. **Report numbers changed: 419 → 369 events** — cross-campaign dedup removed 325 double-counted images (primavera_2025 is almost entirely superseded by pv_2025_2026, which re-reviewed the same SD cards), and 143 otoño-2025 CT15/CT16/CT19 records are now excluded because timestamps.py refuses the offset guess the old code made. A near-miss caught in the process: naive dedup would have kept the *earlier* campaign's labels and silently reverted 31 adjudicated species, including demoting a puma — precedence is now explicit in `CAMPAIGN_ORDER` and every label change is printed. Old figures preserved in `figures_pre_canonical/`. **Next:** add real anchors for CT15/CT16/CT19 to recover those 143 records (camera filenames encode MMDD, so no field notebook is strictly required); migrate pehuen + data-pipeline to `observations.parquet` and retire `_corrected.csv`. Pending commit + push.)

**Prior — 2026-07-29** (camera-traps: **architecture review — canonical observation schema agreed, implementation pending.** Added `## DESIGN_NOTES` to `camera-traps/README.md` recording external file-format leakage as the project's dominant decay risk (Timelapse2 CSV + MegaDetector JSON decoded independently at 6+ call sites). **Critical:** `Anual-reports/2025/py/01_data_prep.py` still reads `_reviewed.csv` and re-implements clock repair with different rules than `timestamps.py` — the `_corrected.csv` contract from 2026-06-25 is not honoured by the annual report, so report and pehuen can derive different timestamps from the same raw data. **Decision:** replace per-consumer normalisation with a canonical observation table (`camtrap/observations.py` → `observations.parquet`, keyed on `campaign, camera_num, file_name`) written once at ingest, so downstream repos need no shared code — just `read_parquet`. Canonical station ID `CT01`–`CT27`, enforced by a validation gate at the Synology folder level in `setup/flatten_for_camtrapdp.py`; the `_M##.2` grid suffix moves to a station registry (it is many-to-one — grid M15.2 holds cameras 11 and 18). Legacy station spellings go in `data/campaigns/station_aliases.csv` — data, not code. Also confirmed `100EK113` (Primavera 2025) is **camera 5**, an unrenamed SD-card folder duplicating `pv_2025_2026 / TC5_M9.2` (14 files, timestamps identical to the second); its 252 rows are dropped as unmappable today and no detections are lost, but only because the same images arrive via pv. **No code written this session** — fixing the report will change its figures, so it needs a figure diff. Pending commit + push.)

**Prior — 2026-06-17** (camera-traps: **Otoño 2026 campaign integrated** — May 2026 SD pull reviewed, 1785 observations, 25 deployments (CT_02 and CT_12 produced no animal triggers; timelapse parser is observation-centric so they're correctly absent from `ct_deployments`). Vaca payoff confirmed: 579 rows tagged Vaca (top species in campaign) that would have been mislabeled Caballo without yesterday's species addition. Added Quique (*Galictis cuja*) to `data-pipeline/species.yaml` with CLIP English prompt — 5 obs in this campaign, native mustelid, first project record. CSV staged at `camera-traps/data/campaigns/otono_2026/` and registered in `data-pipeline/config.yaml`. Zero overlap with otono_2025 / primavera_2025 / pv_2025_2026 — no dedup script needed. **Pending:** CT_18 timestamps — 135 rows currently dated 2017-01-01 (camera clock reverted to factory). Real deployment-start anchor pending from field notebook; do **not** run `--ct` on the Linux box until those are corrected (see comment in `config.yaml`). Pending commit + push.)
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

**Canonical catalogs (2026-04-27, updated 2026-06-17):** `species.yaml` (33 entries — 29 CLIP + 4 reviewer-discovered non-CLIP + invasive/priority flags) is the single source of truth across the ecosystem. Sibling loaders in camera-traps and plataforma-territorial/backend read this same file. Pairs with `plataforma-territorial/data/stations.yaml` (also now consumed end-to-end after Track B).

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

CLIP classification pipeline and Streamlit review UI are production-quality. Four campaigns reviewed: Otoño 2025, Primavera 2025, Primavera-verano 2025-2026, and Otoño 2026 (latest). Species list sourced from canonical `data-pipeline/species.yaml` via sibling loader (Track B, 2026-04-27) — 29 CLIP species + 4 non-CLIP entries (33 total).

**Last Updated:** 2026-06-17
**What Changed:** Otoño 2026 campaign reviewed and staged. 1785 obs / 25 deployments at `data/campaigns/otono_2026/`; registered in `data-pipeline/config.yaml`. Quique (*Galictis cuja*) added to species.yaml with CLIP English prompt (5 obs in this campaign, first project record). Vaca prompt (added 2026-06-16) validated: 579 rows in Otoño 2026 — would have been mislabeled Caballo without it. Zero cross-campaign overlap (verified via stdlib CSV-vs-CSV check).
**Integration Status:** Pending CT_18 timestamp fix. 135 rows show DateTime 2017-01-01 (camera clock reverted to factory). Real deployment-start anchor pending from field notebook. Linux ingestion (`python run_fetch.py --ct`) held until those rows are corrected to avoid landing wrong-year data in DuckDB.
**Blockers/Notes:** Pendiente CT_18 timestamp anchor — see config.yaml comment block. Pendiente decidir si re-mapear despliegue `100EK113` a CT5 (sospecha confirmada por foto pero no re-ingestado). Pendiente correr `render.sh` en Linux para regenerar el `.docx` del Informe Anual 2025. Bundle `source_code_CT_2025/` untracked en git — decidir si commit + Drive o sólo Drive.

| Component | Status | Notes |
|---|---|---|
| MegaDetector integration | Done | Via AddaxAI on Windows desktop |
| CLIP classification | Done | `run_classification.py` — CSV-only workflow, no DB dependency |
| Streamlit review UI | Done | `phase1_labeling/app.py` — handles empty filePath column |
| GIS data (KML → GeoJSON) | Done | Boundary + 26 station coordinates (TC-26 fixed 2026-03-30) |
| Otoño 2025 classification | Done | 697 animal obs reviewed |
| Primavera-verano 2025-2026 | Done | 500 animal obs reviewed |
| Otoño 2026 (May 2026 pull) | Reviewed; ingest pending | 1785 obs / 25 deployments; staged at `data/campaigns/otono_2026/`. CT_18 timestamp fix pending. |
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
- [ ] **Otoño 2026 — CT_18 timestamp fix (2026-06-17).** 135 rows con DateTime 2017-01-01 (reloj de la cámara reseteado a fábrica). Cuaderno de campo tiene la fecha real de despliegue; cuando esté a mano, re-timestampear y correr `python run_fetch.py --ct` en Linux. Mientras tanto, **no ingestar** Otoño 2026 (config.yaml ya lo deja anotado).
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
