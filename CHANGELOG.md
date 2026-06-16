# Changelog

All notable changes to the FMA Python ecosystem (data-pipeline, camera-traps, plataforma-territorial, literatura-agent, schedule-agent, visualizaciones-artisticas) will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely — dated sections, grouped by Added / Changed / Fixed / Deferred / Closed-rejected. Internal personal project, no public versioning.

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
