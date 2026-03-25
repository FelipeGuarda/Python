# Pipeline de Datos — FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Status:** Built and operational. All 5 phases complete.
**Role in ecosystem:** Core plumbing. Every other project reads from the DuckDB database this pipeline maintains.

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
    ├── ct_deployments        ← camera trap deployment metadata
    ├── ct_media              ← camera trap image/video files
    ├── ct_observations       ← wildlife detection records
    └── literature            ← paper metadata + summaries (written by literatura-agent)
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
        ├── camera_trap_legacy.py  ← Parse Timelapse2 CSV format
        ├── camtrap_dp.py          ← Parse Camtrap DP (TDWG) standard
        ├── met_csv.py             ← Parse merged CR800 CSV exports
        └── toa5.py                ← Parse Campbell Scientific TOA5 files
```

---

## Data Sources

| Source | Format | Ingestion | Status |
|---|---|---|---|
| CR800 datalogger (Bosque Pehuén) | TOA5 ASCII / PakBus | Remote pull via Tailscale | Code complete, awaiting connection test |
| Open-Meteo API | JSON → DataFrame | Scheduled fetch (hourly) | Working |
| Camera trap exports (Timelapse2) | CSV | File watcher | Working (legacy parser tested) |
| Camera trap (Camtrap DP) | 3 CSVs (TDWG standard) | File watcher | Code complete, awaiting test data |

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

## Pending / Known Issues

- CR800 live fetching needs first connection test (Tailscale to `100.97.202.90:2000`)
- `pycampbellcr1000` has no version pin in environment.yml
- Uses `print()` for logging — should migrate to `logging` module for production
- `pandera` and `openpyxl` are in environment.yml but never used (can be removed)
