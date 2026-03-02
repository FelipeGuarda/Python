# Pipeline de Datos — FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Status:** Planned / Not yet built.
**Role in ecosystem:** Core plumbing. Every other project (Plataforma Territorial, Dashboard, Asistente) reads from the DuckDB database this pipeline maintains.

---

## What This Project Does

A background service that ingests field data from multiple sources into a single local DuckDB database. It runs silently as a daemon/cron job and requires no manual intervention once configured.

Two ingestion modes:

1. **File watcher**: Monitors a folder for new CSV/data exports (e.g., camera trap exports, manually downloaded weather files) and ingests them automatically on arrival.
2. **Remote fetch** (optional): Connects to the CR800 Campbell Scientific datalogger at Bosque Pehuén via Tailscale VPN and pulls new records on a schedule.

All downstream projects query DuckDB directly — this pipeline is the single source of truth for all field data.

---

## Current State

- **Not yet built.** Architecture and data sources are defined.
- Existing data sits in flat CSVs and Excel files across project folders.
- The fire-risk dashboard currently fetches its own data from Open-Meteo API directly (will be replaced or complemented by this pipeline).

---

## Data Sources

| Source | Type | Format | Ingestion method |
|---|---|---|---|
| CR800 datalogger (Bosque Pehuén) | Weather station | TOA5 ASCII / CSV | Remote pull via Tailscale |
| Open-Meteo API | Modelled weather forecast | JSON → DataFrame | Scheduled fetch |
| Camera trap exports (Timelapse2) | Wildlife detection | CSV | File watcher |
| Megadetector outputs | AI detection results | JSON | File watcher |
| CONAF / external fire data | Historical fire records | GeoJSON / CSV | One-time or scheduled fetch |

---

## Planned Architecture

```
Data Sources
    ↓
[File Watcher]  ←  drop files into /data/incoming/
[Remote Fetcher] ← cron pull from CR800 via Tailscale
[API Fetcher]   ← Open-Meteo, CONAF, OpenAlex
    ↓
Ingestion Layer
    - schema validation
    - deduplication (skip already-ingested records)
    - timestamp normalization (all → UTC)
    ↓
DuckDB (local file: fma_data.duckdb)
    ├── weather_station       ← CR800 sensor readings
    ├── weather_forecast      ← Open-Meteo hourly/daily
    ├── camera_trap           ← wildlife detection records
    ├── fire_risk             ← computed daily risk index + ML prediction
    └── literature            ← paper metadata + summaries (from Agente de Literatura)
    ↓
Downstream consumers (read-only queries)
    ├── Plataforma Territorial (Streamlit)
    ├── Agente de Literatura (writes its own table)
    └── ad-hoc analysis notebooks
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Database | DuckDB | Local file, zero server, fast analytical queries |
| File watching | `watchdog` (Python) | Monitors incoming folder |
| Remote fetch | Tailscale + `paramiko` or `pylogger-cr` | CR800 via Tailscale SSH or LoggerNet protocol |
| Scheduling | cron or APScheduler | Periodic remote fetches |
| Data validation | `pandera` or simple pandas checks | Schema enforcement before insert |
| Language | Python 3.11 |  |

---

## CR800 Integration (Tailscale)

The CR800 is a Campbell Scientific datalogger physically at Bosque Pehuén. Access plan:

- Tailscale creates a VPN tunnel to the device (or to a Raspberry Pi bridging it)
- The pipeline connects on schedule and pulls new TOA5 records since the last fetch
- Records are parsed and inserted into `weather_station` table in DuckDB
- Duplicate protection: use `(station_id, timestamp)` as unique key

This is the "optional" but most valuable part — real sensor data vs. modelled data.

---

## File Structure (planned)

```
data-pipeline/
├── .env                      ← DB path, Tailscale credentials, API keys
├── config.yaml               ← source definitions, schedules, table schemas
├── requirements.txt
│
├── src/
│   ├── db.py                 ← DuckDB connection + schema creation + upsert helpers
│   ├── watcher.py            ← watchdog file watcher for /data/incoming/
│   ├── fetchers/
│   │   ├── cr800.py          ← CR800 remote pull via Tailscale
│   │   ├── open_meteo.py     ← Open-Meteo API fetch
│   │   └── conaf.py          ← CONAF fire data fetch
│   ├── parsers/
│   │   ├── toa5.py           ← CR800 TOA5 format parser
│   │   ├── camera_trap.py    ← Timelapse2 CSV parser
│   │   └── megadetector.py   ← Megadetector JSON parser
│   └── ingest.py             ← orchestrates parse → validate → upsert
│
├── run_watcher.py            ← entry point: file watcher daemon
├── run_fetch.py              ← entry point: manual or cron remote fetch
└── schema.sql                ← DuckDB table definitions
```

---

## Ideas & Future Features

- **Backfill command**: `python run_fetch.py --backfill 2025-01-01 2025-12-31` to reprocess historical files
- **Data quality alerts**: Email/Slack if new records contain sensor anomalies (e.g., temperature > 60°C, humidity = 0% for 48h)
- **Web admin UI**: Simple Streamlit page to view ingestion logs, last record timestamps per source, and trigger manual fetches

---

## Key Design Decisions

1. **DuckDB over PostgreSQL**: No server to manage. DuckDB reads/writes a single file, handles analytical queries fast, works perfectly for local + Streamlit Cloud use case.
2. **Upsert over append**: All inserts use `INSERT OR REPLACE` (or `ON CONFLICT DO NOTHING`) so re-running the pipeline on the same data is always safe.
3. **UTC everywhere**: All timestamps normalized to UTC at ingest time. Timezone conversion happens at display time in the platform.
4. **File watcher for camera trap data**: Manual exports from Timelapse2 are irregular — polling is cleaner than a push integration.

---

## Context for AI Sessions

When starting a new Claude session on this project:

1. This is the **data layer** for the entire FMA platform — nothing here has a UI
2. The central artifact is `fma_data.duckdb` — a single local file that all downstream apps open read-only
3. DuckDB Python API: `import duckdb; con = duckdb.connect("fma_data.duckdb")`
4. CR800 outputs TOA5 format (Campbell Scientific ASCII) — needs a dedicated parser
5. The Tailscale integration is the hardest part — may need a Pi at Bosque Pehuén as a bridge
6. Camera trap CSVs come from Timelapse2 exports — see camera-trap-analyzer/README.md for column schema
