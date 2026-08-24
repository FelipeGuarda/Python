# Pipeline de Datos — FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Status:** Built and operational. All 5 phases complete.
**Role in ecosystem:** Core plumbing. Every other project reads from the DuckDB database this pipeline maintains.

**Last Updated:** 2026-08-24 — the camera-trap path is rebuilt and the canonical contract is verified
**What Changed:** `ct_deployments` / `ct_media` / `ct_observations` rebuilt from camera-traps'
`observations.parquet` — **35,807 rows across 74 deployments** (21 + 26 + 27 stations, the real
deployment history), reconciling exactly against the published contract. They previously held 2,948
orphaned rows written by a parser deleted on 2026-08-20, under pre-flatten identity
(`oto_o_2025_CT07`) and keyed on Timelapse GUIDs, with `pv_2025_2026` still present as a phantom
campaign. New: `src/parsers/canonical_ct.py` (the projection, which re-derives nothing),
`src/canonical_gate.py` (the consumer half of the contract gate — `run_fetch.py --ct-check`), and
`src/recovery.py` (the irreplaceable weather tables as committed per-year Parquet). `literature`
dropped — 0 rows, no reader in this monorepo. **The Windows↔Linux question is answered
permanently:** any machine can rebuild this database from the repository alone.
**Integration Status:** `Ready` for weather and camera traps. `Pending [tests]` — see `docs/TEST-PLAN.md`.
**Blockers/Notes:** **this project has no test suite at all** (1,642 lines) while now carrying four
modules that guard the camera-traps boundary — designed in `docs/TEST-PLAN.md` and deferred. CR800
still offline since 2026-04-13. Timestamps in `ct_*` are **naive local wall time**, not TIMESTAMPTZ —
`schema.sql` explains why, and it is not an oversight.

**Prior — 2026-05-11** — code review complete
**What Changed:** Tier 1 review fixes — S10 (factored 9× connection-lifecycle boilerplate into `@contextmanager managed_conn(init=…)`) and S11 (`watcher.py` now opens a short-lived DB connection per filesystem event via a `connect_fn` factory; `run_watcher.py` does one-shot bootstrap init then releases the lock). S14 closed-rejected — no measured benefit from secondary indices at current data scale (columnar engine + composite PKs already cover the hot reads). **First full code review now complete:** every finding in this project is closed or has an explicit re-open trigger. See repo-root `CHANGELOG.md`.
**Integration Status:** Ready. Smoke-tested via `run_fetch.py --once` — Open-Meteo round-trip clean.
**Blockers/Notes:** CR800 still offline since 2026-04-13 (antenna replacement pending). C1 (state-before-commit) will be exercised in vivo on the 8-day backfill burst once the antenna is back.

---

## What This Project Does

A background service that ingests field data from multiple sources into a single local DuckDB database (`fma_data.duckdb`). Two ingestion modes:

1. **File watcher** (`run_watcher.py`): Monitors `data/incoming/` for new CSV/data exports and ingests them automatically.
2. **Scheduled fetch** (`run_fetch.py`): Connects to the CR800 datalogger via Tailscale VPN and pulls Open-Meteo forecasts on a schedule (APScheduler).

All downstream projects query DuckDB directly — this pipeline is the single source of truth.

---

## How to Run

```bash
conda activate data-pipeline

# Manual single fetch (Open-Meteo + CR800 if reachable)
python run_fetch.py --once

# Backfill a TOA5 .dat or CSV file
python run_fetch.py --backfill path/to/file.dat

# Start scheduled daemon (Open-Meteo every 60 min, CR800 weekly)
python run_fetch.py

# Start file watcher daemon
python run_watcher.py
```

---

## Architecture

```
Data Sources
    ↓
[File Watcher]   ←  drop files into data/incoming/
[Remote Fetcher] ←  APScheduler pull from CR800 via Tailscale
[API Fetcher]    ←  Open-Meteo hourly forecast
    ↓
Ingestion Layer (src/ingest.py)
    - schema validation
    - deduplication (INSERT OR REPLACE)
    - timestamp normalization (all → UTC)
    ↓
DuckDB (fma_data.duckdb)
    ├── weather_station       ← CR800 sensor readings
    ├── weather_forecast      ← Open-Meteo hourly/daily
    ├── ct_deployments        ← one row per station × campaign (74)
    ├── ct_media              ← one row per still (35,807), incl. those with no clock
    ├── ct_observations       ← media-level observations, 1:1 with ct_media
    └── ct_ingest_state       ← what was last ingested, for the contract gate

    (literature: removed 2026-08-24 — 0 rows, no reader; literatura-agent is standalone)

    ct_* are REBUILT WHOLE from camera-traps/data/campaigns/*/observations.parquet.
    Never hand-populated, never parsed from a Timelapse export.
    Committed recovery copies of the two irreplaceable tables live in data/recovery/.
    ↓
Downstream consumers (read-only)
    ├── Plataforma Territorial (React frontend via FastAPI)
    └── Ad-hoc analysis notebooks
```

---

## File Structure

```
data-pipeline/
├── .env.example              ← Template (DB path, CR800 credentials)
├── config.yaml               ← Runtime config (CR800, Open-Meteo, schedules)
├── schema.sql                ← DuckDB table definitions (6 tables)
├── environment.yml           ← Conda dependencies
├── run_fetch.py              ← Entry point: fetch daemon + backfill CLI
├── run_watcher.py            ← Entry point: file watcher daemon
└── src/
    ├── db.py                 ← DuckDB connection + schema + upsert
    ├── ingest.py             ← Orchestrator (6 ingest functions)
    ├── watcher.py            ← Watchdog FileSystemEventHandler
    ├── fetchers/
    │   ├── open_meteo.py     ← Fetch hourly weather forecast
    │   └── cr800.py          ← Connect + fetch CR800 via PakBus TCP
    └── parsers/
        ├── met_csv.py             ← Parse merged CR800 CSV exports
        └── toa5.py                ← Parse Campbell Scientific TOA5 files
```

---

## Data Sources

| Source | Format | Ingestion | Status |
|---|---|---|---|
| CR800 datalogger (Bosque Pehuén) | TOA5 ASCII / PakBus | Remote pull via Tailscale | Code complete, awaiting connection test |
| Open-Meteo API | JSON → DataFrame | Scheduled fetch (hourly) | Working |
| Camera trap | `observations.parquet` (camera-traps) | `run_fetch.py --ct` — full rebuild, gated on `CANONICAL_STATE.json` | **Working** since 2026-08-24. 35,807 rows / 74 deployments |

---

## Two-Machine Setup

Raw data analysis (MegaDetector, CLIP classification, image review) runs on the **Windows office desktop** with GPU + Synology NAS access. Once a reviewed CSV is exported, it can be ingested on **any OS** via this pipeline. The pipeline itself and all downstream code is cross-platform.

---

## Key Design Decisions

1. **DuckDB over PostgreSQL** — no server, single file, fast analytics
2. **Upsert over append** — `INSERT OR REPLACE` on all tables, always safe to re-run
3. **UTC everywhere** — timezone conversion at display time only
4. **APScheduler not cron** — cross-platform scheduling
5. **State tracking for CR800** — `data/cr800_state.json` enables incremental sync

---

## Status

**Last Updated:** 2026-04-27
**What Changed:** Code review Batch A+B fixes applied (9 warnings resolved). New `src/paths.py` centralises `_STATE_PATH` (W15). Dead deps removed from `environment.yml` — `httpx`, `pandera`, `openpyxl` (W9). `_process_raw` renamed to `process_raw` (W12). `open_meteo.py` `tz_localize` now has `ambiguous=False` to survive DST fall-back (W16). `cr800_session()` context manager added; 3 call sites updated to send PakBus Bye on exit (W18). `run_once()` isolates Open-Meteo and CR800 fetchers so a DNS failure no longer kills the CR800 fetch (W19). `recover_dst_gaps.py` DST dates now derived algorithmically via `_first_saturday_of_april()` — no longer need annual manual updates (W13). `timelapse_reviewed.py` missing `count` field now stores `None` instead of silent `1` (W14). `toa5.py` logs unrecognised columns instead of silently dropping them (W17).
**Integration Status:** Ready
**Blockers/Notes:** 8 Spanish display names changed to canonical form (e.g., "Güiña" → "Guiña", "Huet-huet" → "Chucao", "Lechuza del sur" → "Concón") — flag for biological review with Felipe before any user-facing release; species.yaml is the single edit point if any are biologically wrong. ~~Reviewed CSV is observation-centric — zero-animal stations absent from ct_deployments; platform map works around it with a TC-coords list in the backend.~~ **SOLVED 2026-08-19:** `observations.parquet` is now one row per still, so a station that recorded nothing is present with its true frame count and `valid_effort`. The per-campaign deployment manifest this note asked for is that file. **Consequence:** the TC-coords workaround in `plataforma-territorial/backend/routers/detections.py` is removable. ~~`occupancy_pct` divides by every station in `stations.yaml`~~ **FIXED 2026-08-24** — it now counts from `ct_deployments`, and returns `n_stations_deployed` so the denominator is visible; per-campaign filtering is still open. ~~Re-ingest pattern: DELETE rows for the campaign first~~ — no longer applies: `--ct` rebuilds the tables **whole** every run, which is what stops a retired campaign (`pv_2025_2026`) stranding rows. Remaining review items: C1 (CR800 state-before-commit), W8 (DST consolidation into tz_utils.py), W10, W20 — deferred to Batch E.

---

## CLI Reference

```bash
python run_fetch.py --once          # fetch once and exit
python run_fetch.py --export        # export weather_station to CSV now
python run_fetch.py --health        # health report (last fetch, row count, gaps)
python run_fetch.py --health --verbose  # health report with gap details
python run_fetch.py --backfill FILE            # backfill from .dat or met .csv
python run_fetch.py --fetch-range START END   # fetch explicit date range from CR800 (no state change)
python run_fetch.py                           # start scheduler daemon

# Camera traps — rebuilt whole from camera-traps' canonical parquets
python run_fetch.py --ct            # rebuild ct_deployments / ct_media / ct_observations
python run_fetch.py --ct-check      # is the DB current with CANONICAL_STATE.json? exits 1 if not

# The irreplaceable tables (weather only — everything else is regenerable)
python -m src.recovery export       # write data/recovery/<table>/<year>.parquet, then COMMIT them
python -m src.recovery restore      # rebuild the tables from the committed Parquet
python -m src.recovery verify       # do database and archive agree? exits 1 if not
```

**Setting up on a new machine:** clone, create the env, then
`python -m src.recovery restore && python run_fetch.py --ct`. That reconstructs the entire
database from the repository — nothing has to be copied by hand.

---

## Pending / Known Issues

- **No test suite.** 1,642 lines in `src/`, zero tests, and four modules added 2026-08-24
  (`canonical_ct`, `canonical_gate`, `recovery`, `_reconcile`) whose guarantees rest on one
  manual run each. Designed in `docs/TEST-PLAN.md`; stdlib `unittest`, no new dependencies
  (**pytest is installed in neither environment**)
- `duckdb_tables().estimated_size` is **not trustworthy** — it reported 24,665 rows for a
  2,948-row table on 2026-08-24. Use `COUNT(*)` when the number matters
- CR800 fetch fails silently on connection error with no alerting when zero rows returned
- `pycampbellcr1000` has no version pin in environment.yml
- Uses `print()` for logging — should migrate to `logging` module for production
- Annual ~60 min DST gap each April is a CR800 hardware behavior (logs ambiguous hour once, in standard time); not fixable in software
- C1: `cr800.py:fetch_since` saves state before upsert commits — silent data loss risk on interrupted run (Batch E)
- W8: 5 different DST/`ambiguous=` strategies across parsers — consolidate into `src/tz_utils.py` (Batch E)
