# Changelog

All notable changes to the FMA Python ecosystem (data-pipeline, camera-traps, plataforma-territorial, literatura-agent, schedule-agent, visualizaciones-artisticas) will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely — dated sections, grouped by Added / Changed / Fixed / Deferred / Closed-rejected. Internal personal project, no public versioning.

---

## [2026-06-17] — Camera-traps: Otoño 2026 campaign integrated, +Quique

May 2026 SD pull (campaign name **Otoño 2026**, slug `otono_2026`) reviewed end-to-end and staged for ingestion. CSV registered in `data-pipeline/config.yaml`. New species — Quique (*Galictis cuja*) — added to the canonical catalog with a CLIP English prompt; first project record (5 obs in this campaign). Yesterday's Vaca addition validated: 579 rows tagged Vaca in this campaign, all of which would have been mislabeled Caballo. Ingestion itself is held until CT_18's clock-reset timestamps are corrected.

### Added
- **`data-pipeline/species.yaml`** — Quique (*Galictis cuja*). CLIP prompt: `"lesser grison small mustelid weasel"`. Native, no `is_invasive` / `is_priority` flag. 29 CLIP species + 4 reviewer-discovered non-CLIP entries = 33 total (was 28+4=32).
- **`camera-traps/data/campaigns/otono_2026/new_labeled_data_reviewed.csv`** — 1785 rows, 25 deployments (CT_02 and CT_12 produced no animal triggers; timelapse parser is observation-centric so they're correctly absent from `ct_deployments`). Date range covers 2025-? through 2026-05-15, except CT_18 which has 135 rows stuck at 2017-01-01 (see below).
- **`data-pipeline/config.yaml`** — 4th `camera_traps.campaigns` entry: `name: "Otoño 2026"`. Comment block immediately above the entry flags the CT_18 timestamp issue and instructs not to run `--ct` until corrected.

### Changed
- **`camera-traps/README.md`** — Status header rewritten for 2026-06-17; species table +Quique row + filled-in Invasive/Priority cells; CLIP species count 26 → 29 in Step 3; Campaign History table now includes Otoño 2026 and corrected paths for the prior three campaigns.
- **`PROJECT_STATUS.md`** — top "Last updated" line rewritten; section 1 species.yaml count 31 → 33; section 3 Last Updated/What Changed/Integration/Blockers refreshed; component table +Otoño 2026 row; Open Items: new CT_18 timestamp-fix entry.

### Notes
- **CT_18 clock reset**: 135 rows on CT_18 carry `DateTime` 2017-01-01 (camera clock reverted to factory default at some point during the deployment). Felipe has the real deployment-start date in his field notebook; until it's transcribed, `python run_fetch.py --ct` is held on the Linux box. Once the anchor is in hand, one re-stamp + re-ingest finishes the integration. Until then the `otono_2026` entry sits dormant in config.yaml behind a comment.
- **Zero cross-campaign overlap** verified via stdlib CSV-vs-CSV check against otono_2025 / primavera_2025 / pv_2025_2026. No dedup script needed (unlike the primavera_2025 case, this is a fresh pull, not a partial re-pull).
- **CLIP horse/cow confusion** that motivated yesterday's Vaca prompt is now quantifiable: 579 Vaca rows (#1 species in this campaign) vs 70 Caballo rows — strong evidence the prompt distinguishes correctly. Revisit `clip_confidence_threshold` (0.28) only if the false-positive rate on tightly-similar pairs (Vaca↔Caballo, side/rear shots) looks bad after ingestion.

Session log: `SecondBrain/Sessions/2026-06-17-camera-traps-otono-2026-ingest-prep-and-quique.md`.

---

## [2026-06-16] — Camera-traps: review UI burst context, full-frame display, resume loader, +Vaca

Reworked Phase-1 review UI to better support species disambiguation. Reviewers now see burst context (prev/current/next thumbnails sourced from the MD JSON, including empty triggers) and full frames instead of bbox crops. CLIP classifier untouched — it still receives the bbox crop, keeping its subject-isolation accuracy. Added a startup loader that rehydrates review progress from the previously-exported CSV, eliminating the in-memory-only footgun that bit during this session.

### Added
- **`phase1_labeling/app.py`** — `load_station_index()` builds `{station: [files sorted alphabetically]}` from the MD JSON (includes empty triggers, so reviewer sees what happened before/after the animal trigger). `neighbors(fp)` returns prev/next within station, `None` at deployment boundaries. **Resume loader**: on a fresh session with empty `confirmed`, reads `new_labeled_data_reviewed.csv` and rehydrates `st.session_state.confirmed` / `outcomes`, auto-jumping to the first species batch with unconfirmed images.
- **`data-pipeline/species.yaml`** — Vaca (*Bos taurus*). Aliases: `vaca, vacuno, ganado vacuno, bovino`. CLIP prompt: `"domestic cow cattle bovine"`. `is_invasive: true`. 28 CLIP species now (was 27). Cows were being misclassified as Caballo on BP Mayo 2026.

### Changed
- **`classify_campaign/crop_utils.py` → `cropping.py`** — the file is cohesive (only cropping code, no junk drawer), so the `_utils` suffix was lazy. Imports updated in `run_classification.py`; the now-unused `crop_to_bbox` import was dropped from `phase1_labeling/app.py`. General principle to apply elsewhere when `_utils` suffixes are spotted on cohesive modules.
- **Review UI grid** — was 5-col bbox-cropped thumbnails, now 3-col triptych grid `[anterior | actual | siguiente]` with full-frame thumbnails. Burst context (typically 2-3 frames per trigger) is now the primary species-disambiguation cue alongside the proposed species label.
- **`THUMB_SIZE`** 280 → 1280, **`JPEG_QUALITY`** 75 → 85 in `phase1_labeling/app.py`. Streamlit's expand-icon lightbox shows the cached JPEG as-is, so the cached resolution determines expand quality. In-grid display is unchanged (`use_container_width=True` clamps the visual width). Memory tradeoff acceptable for local desktop review (~10-18MB cache per page).
- **`camera-traps/README.md`** — project tree (`cropping.py`), Step 4 review-UI section rewritten to describe the burst triptych + full-frame display, species table now includes Vaca and the missing Invasive cells for Caballo / Gato doméstico.

### Fixed
- **Sibling import error when launching Streamlit** — `streamlit run phase1_labeling/app.py` was failing with `ModuleNotFoundError: No module named 'classify_campaign'` because Streamlit puts the script's directory on `sys.path`, not the project root. Added a 2-line `sys.path.insert(...)` at the top of `app.py`. This was a latent issue that newer Streamlit versions surfaced (older versions added CWD to sys.path more aggressively).

### Notes
- **Habit to keep**: hit "Exportar CSV revisado" periodically during long sessions — it's now the durable checkpoint and the resume loader will pick it up on next launch.
- **CLIP horse/cow confusion** may persist on side/rear shots even with the new Vaca prompt. If false-positive rate is high after re-classification, consider tightening `clip_confidence_threshold` (currently 0.28) — but only after seeing the data.

Session log: `SecondBrain/Sessions/2026-06-16-camera-traps-review-ui-context-strip-and-resume.md`.

---

## [2026-06-02] — Observatorio: piso vegetacional layer (plataforma-territorial)

New toggleable Leaflet layer on the Observatorio page: photointerpretation of Bosque Pehuén's vegetational floor (48 polygons). Off by default; click a polygon for `BIOTOPO / Distrito / Superficie (ha)`.

### Added
- **`plataforma-demo/src/components/PisoVegetacionalLayer.jsx`** — self-contained React layer: owns its own fetch, color map, and popup. `Observatorio.jsx` only adds 1 import, 1 `useState(false)`, 1 conditional render, and 1 checkbox. Future GIS layers should follow this shape.
- **`scripts/convert_piso_vegetacional.py`** — pure-Python shapefile → GeoJSON converter (`pyshp` + `pyproj`, no GDAL). Handles ESRI ring orientation (CW = outer), reprojection EPSG:32718 → EPSG:4326, and the actual UTF-8 dbf encoding. Writes the same payload to `data/`, `public/data/`, and `dist/data/`.
- **`data/piso_vegetacional.geojson`** (+ public/dist copies) — 48 features. Source shapefile preserved under `data/piso_vegetacional_source/veg_foto_BP.*` (outside the served frontend) so the script remains reproducible.

### Changed
- **Palette designed in two iterations.** First attempt (greens only, dark → light by density) collapsed visually to ~6 distinguishable swatches across 10 classes. Final palette groups by ecological type: greens for Bosque (4 classes), ochres for Renoval (3), blues/violets for Matorral / Pradera / Estepa.
- **No polygon borders** — `stroke: false`. White borders fought the Esri satellite imagery underneath.

### Fixed
- **Mojibake (`MesÃ³fito`) in popups.** First conversion used `encoding="latin-1"` based on a misleading test; the .dbf is genuinely UTF-8 (matching the .cpg). Codepoint inspection on Windows is authoritative — the cmd console can't render `ó` and shows `�`, which is not the same as a decode failure. Switched to `encoding="utf-8"`; also resolved a silent color-fallthrough where the mismatched key was hitting `FALLBACK_COLOR` for 4 polygons.

### Documentation
- README top block, "Four Pages" table, and "Real Data" list updated.
- `PROJECT_STATUS.md` Plataforma section: Observatorio row + Priority 3 list.

### Notes
- **`DISTRITO` field documented**: physiographic / terrain-relief classification (`Plano / Ondulado / Cerrano / Montano`) — geomorphological, independent of vegetation. Currently a popup property; could become a separate layer.
- **`ESPECIES_D` field omitted** from popups for now — Felipe doesn't have a key for the species codes (NF / ND / NA / NP / AA / SC). Re-enable when a legend surfaces.

Session log: `~/Documents/Obsidian FG/SecondBrain/Sessions/2026-06-02-plataforma-piso-vegetacional-layer.md`.

---

## [2026-05-11] — 🏁 First full code review complete

The independent code review of the FMA ecosystem (started 2026-04-21) is finished. **Every finding across data-pipeline, camera-traps, and plataforma-territorial is now closed or explicitly deferred with a re-open trigger.** No silently-open work remains.

**Headline numbers:**
- 1 Critical (C1 — CR800 state-before-commit) — resolved
- ~25 Warnings (W8…W52) — all resolved
- ~50 Suggestions (S8…S78) — resolved, closed-rejected, or deferred with re-open conditions

### Fixed (today's session — Tier 1 through Tier 4)
- **Tier 1 reliability** — S50 (FastAPI health 503), S37 (mtime-keyed Streamlit cache), S10 (`managed_conn` context manager), S11 (per-event DB connections in watcher).
- **Tier 2 high-value** — S44 (`backend/paths.py`), S48 (stations.yaml coords), S55 (DuckDB CTE for `days_without_rain`), S64 (`demo_report.js`), S66 (RiskGauge pure component), S39 (`classify_all` / `apply_classifications` split).
- **Bundle A — schema authority pass** — S49 (startup drift-check; surfaced 27 real extras in the DB including `battery_voltage` accidentally absent from `ALLOWED_COLS`).
- **Bundle B — API/hook ergonomics** — S53 (docstrings + `common_name` symmetry), S59 (`useAPI` `refetch`).
- **Bundle C — camera-traps `setup/` cleanup** — S29 (`crop_to_bbox` reuse), S30 + S32 (argparse), S36 (env + CLI for `CAMPAIGNS_BASE`), S43 (extracted `setup/_fileops.py`). Net **−80 lines** despite one new file.
- **Tier 4 finalization** — S35 (`AnimalRow` dataclass), S38 (UI strings standardized to Spanish).

### Closed-rejected (with documented rationale)
- **S57** — DuckDB pushdown for `strftime` (resample is pandas-side; pushdown would add round-trips).
- **S47** — bootstrap_windows_db.py inline SCHEMA (intentional for cross-machine portability; revisit at Windows→Linux migration).
- **S14** — DuckDB secondary indices (columnar engine + composite PKs already cover hot reads at current data scale).
- **S72** — react-router-dom (overkill for 4-page internal tool; migration path documented).

### Deferred (with re-open triggers)
- **S58** — `stations.yaml` TC-11 / TC-18 both list `sd_card: M15`. Re-open when field records produce an authoritative answer.
- **S76** — Vitest tests in `plataforma-demo/`. Re-open when CI exists.

### Process
Six batches in a single day. Full narrative in `~/Documents/Obsidian FG/SecondBrain/Sessions/2026-05-11-fma-ecosystem-code-review-tier-1-and-tier-2.md`. Review state snapshots in `~/Documents/Obsidian FG/SecondBrain/Reviews/review-state-{data-pipeline,camera-traps,plataforma-territorial}.md`.

---

## [Prior to changelog adoption]

Pre-2026-05-11 history is preserved in:
- `~/Documents/Obsidian FG/SecondBrain/Sessions/` — per-session narrative logs
- `~/Documents/Obsidian FG/SecondBrain/Reviews/review-plan-fma-ecosystem.md` — master review plan with Track A–K log
- Per-project README "Last Updated" sections and `PROJECT_STATUS.md`

This `CHANGELOG.md` is the new top-level history starting 2026-05-11. Going forward, every significant change lands a one-line entry.
