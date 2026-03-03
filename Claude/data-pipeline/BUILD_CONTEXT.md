# Data Pipeline — Build Context for AI Sessions

This document contains everything a new AI session needs to build the FMA data pipeline
from scratch. Read this fully before writing any code.

---

## Project Role

This is the **data backbone** for Fundación Mar Adentro (FMA). It is a background ingestion
service that pulls from multiple field data sources and writes into a single DuckDB file.
Every other FMA project (Plataforma Territorial, Asistente, Reportes, Clasificador de Especies)
reads from that DuckDB file — nothing writes to it except this pipeline and the literatura-agent.

**Owner:** Felipe Guarda — Fundación Mar Adentro (FMA)
**Field site:** Bosque Pehuén, La Araucanía, Chile. Coordinates: lat -39.61°, lon -71.71°
**Timezone:** All timestamps stored as UTC. Converted to `America/Santiago` at display time.
**Language:** Code comments and logs in English. User-facing strings in Spanish.

---

## Environment

- **Conda env:** `data-pipeline` (Python 3.11)
- **OS:** Windows (current) → PopOS Linux (upcoming migration). Use cross-platform code only.
- **Scheduler:** APScheduler (not cron — must work on Windows)
- **Database file:** `/home/fguarda/Dev/Python/fma_data.duckdb`
  - This is the central file shared by all downstream projects
  - Downstream projects open it **read-only**
  - Only this pipeline (and literatura-agent for its own table) writes to it

---

## Data Sources — All Confirmed

### 1. CR800 Campbell Scientific Datalogger (Live)

| Parameter | Value |
|---|---|
| Protocol | PakBus over TCP |
| Tailscale IP | `100.97.202.90` |
| TCP Port | `2000` |
| PakBus address | `1` |
| Python library | `pycampbellcr6` |
| Access | Direct (no Pi bridge — CR800 has its own network interface on Tailscale) |

The CR800 has been running since 2018. It stores data in internal named tables (e.g.,
`Met_15min`, `Met_Daily` — exact names confirmed at first connection with `.listtables()`).

Historical backfill: the user has `.dat` TOA5 files + a master CSV covering 2018–present.
Parse these with the `toa5.py` parser at first run.

**Important:** The exact column names of `weather_station` table must be inferred from the
TOA5 `.dat` file headers at first run. Do not hardcode columns until you've seen the header.

### 2. Open-Meteo API (Forecast)

Free, no API key. Endpoint for Bosque Pehuén:

```
https://api.open-meteo.com/v1/forecast
  ?latitude=-39.61
  &longitude=-71.71
  &hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,
          wind_direction_10m,et0_fao_evapotranspiration
  &timezone=America%2FSantiago
  &past_days=7
```

Fetch hourly, store in `weather_forecast` table. Schedule: every 1 hour.
Upsert on `timestamp` — safe to re-run.

### 3. Camera Trap Data — Two Formats

Camera trap data exists in two formats that must both be supported:

#### Format A: Legacy Timelapse2 CSV (`old animal data DB.csv`)
One file, 18,473 rows, one row per image. **Already in the data-pipeline directory.**

Columns: `RootFolder, File, RelativePath, DateTime, Animal, Person, Especie, Counter0, Note0`

Key parsing rules:
- `RelativePath` example: `2022\Araucarias\CT10_06_12_22`
  - Area: `Araucarias` (second path segment)
  - Station/deploymentID: `CT10` (prefix before first `_` in last segment)
  - Service date: `06_12_22` (DD_MM_YY — camera collected Dec 6 2022)
- `DateTime` has a leading space — strip it. Format: `YYYY-MM-DD HH:MM:SS`. Treat as `America/Santiago`.
- `Note0` format: `burst_num:image_within_burst|total_images` (e.g. `2:1|3` = burst 2, image 1 of 3)
  - `eventID` = `{deploymentID}_{burst_num}`
- `Animal=Si` → `observationType='animal'`, `classificationMethod='human'`
- `Person=Si` → `observationType='human'`, `classificationMethod='human'`
- Both No → `observationType='blank'`
- `Especie` = Spanish common name (Puma, Zorro culpeo, Jabali, Perro, Caballo, etc.)
- `mediaID` = `legacy_{RelativePath_slug}_{File}` (make it unique and deterministic)
- `observationID` = `legacy_{mediaID}_obs`
- `source` = `'legacy'` in all three tables

One legacy CSV row → 1 `ct_media` row + 0 or 1 `ct_observations` row.

Species found in legacy data (25 unique):
Puma, Zorro culpeo, Jabali, Perro, Caballo, Liebre, Guiña, Chucao, Zorzal, Hued hued,
Bandurria, Gato, Ratón cola larga, Queltehue, Vison, Monito del monte, Tiuque, Carpintero,
Rayadito, Peuquito, Concón, Cometocino, Picaflor, Diucón, Traro

#### Format B: Camtrap DP (Camera Trap Data Package — TDWG/GBIF standard)
Three CSVs in a folder: `deployments.csv`, `media.csv`, `observations.csv`

`deployments.csv` columns:
```
deploymentID, locationID, locationName, latitude, longitude, coordinateUncertainty,
deploymentStart, deploymentEnd, setupBy, cameraID, cameraModel, cameraDelay, cameraHeight,
cameraDepth, cameraTilt, cameraHeading, detectionDistance, timestampIssues, baitUse,
featureType, habitat, deploymentGroups, deploymentTags, deploymentComments
```

`media.csv` columns:
```
mediaID, deploymentID, captureMethod, timestamp, filePath, filePublic, fileName,
fileMediatype, exifData, favorite, mediaComments
```

`observations.csv` columns:
```
observationID, deploymentID, mediaID, eventID, eventStart, eventEnd, observationLevel,
observationType, cameraSetupType, scientificName, count, lifeStage, sex, behavior,
individualID, individualPositionRadius, individualPositionAngle, individualSpeed,
bboxX, bboxY, bboxWidth, bboxHeight, classificationMethod, classifiedBy,
classificationTimestamp, classificationProbability, observationTags, observationComments
```

Key fields: `observationLevel` (event-level vs media-level), `classificationMethod`
(human vs machine), `classificationProbability` (AI confidence). Spec: https://camtrap-dp.tdwg.org

Load only the columns that map to our DuckDB schema. Set `source='camtrap_dp'`.

---

## Database Schema

All 5 tables in `/home/fguarda/Dev/Python/fma_data.duckdb`:

```sql
-- CR800 sensor readings
CREATE TABLE IF NOT EXISTS weather_station (
    station_id       TEXT NOT NULL,
    timestamp        TIMESTAMPTZ NOT NULL,
    temperature_air  DOUBLE,
    relative_humidity DOUBLE,
    wind_speed       DOUBLE,
    wind_direction   DOUBLE,
    precipitation    DOUBLE,
    solar_radiation  DOUBLE,
    battery_voltage  DOUBLE,
    PRIMARY KEY (station_id, timestamp)
    -- Additional columns added dynamically from TOA5 headers at first ingest
);

-- Open-Meteo hourly forecast
CREATE TABLE IF NOT EXISTS weather_forecast (
    timestamp                   TIMESTAMPTZ PRIMARY KEY,
    temperature_2m              DOUBLE,
    relative_humidity_2m        DOUBLE,
    precipitation               DOUBLE,
    wind_speed_10m              DOUBLE,
    wind_direction_10m          DOUBLE,
    et0_fao_evapotranspiration  DOUBLE,
    fetched_at                  TIMESTAMPTZ NOT NULL
);

-- Camera trap: one row per deployment (station + date range)
CREATE TABLE IF NOT EXISTS ct_deployments (
    deploymentID    TEXT PRIMARY KEY,
    locationID      TEXT,
    locationName    TEXT,
    latitude        DOUBLE,
    longitude       DOUBLE,
    deploymentStart TIMESTAMPTZ,
    deploymentEnd   TIMESTAMPTZ,
    cameraID        TEXT,
    cameraModel     TEXT,
    habitat         TEXT,
    source          TEXT NOT NULL   -- 'legacy' | 'camtrap_dp'
);

-- Camera trap: one row per image or video file
CREATE TABLE IF NOT EXISTS ct_media (
    mediaID         TEXT PRIMARY KEY,
    deploymentID    TEXT NOT NULL,
    timestamp       TIMESTAMPTZ,
    fileName        TEXT,
    filePath        TEXT,
    fileMediatype   TEXT,
    source          TEXT NOT NULL
);

-- Camera trap: one row per identified observation (event or media level)
CREATE TABLE IF NOT EXISTS ct_observations (
    observationID              TEXT PRIMARY KEY,
    deploymentID               TEXT NOT NULL,
    mediaID                    TEXT,
    eventID                    TEXT,
    eventStart                 TIMESTAMPTZ,
    eventEnd                   TIMESTAMPTZ,
    observationType            TEXT,   -- 'animal' | 'human' | 'blank' | 'unknown'
    scientificName             TEXT,   -- Spanish common name (legacy) or scientific name (camtrap_dp)
    count                      INTEGER,
    classificationMethod       TEXT,   -- 'human' | 'machine'
    classificationProbability  DOUBLE,
    source                     TEXT NOT NULL
);

-- Paper metadata + summaries written by the literatura-agent
CREATE TABLE IF NOT EXISTS literature (
    paperID        TEXT PRIMARY KEY,
    title          TEXT,
    authors        TEXT,
    published_date DATE,
    source         TEXT,
    url            TEXT,
    summary_es     TEXT,
    fetched_at     TIMESTAMPTZ,
    week_of        DATE
);
```

**Upsert strategy:** `INSERT OR REPLACE INTO` on all tables. Safe to re-run on same data.
For `weather_station`, also safe because `(station_id, timestamp)` is the PK.

---

## File Structure to Build

```
data-pipeline/
├── .env                          ← (never commit) secrets
├── .env.example                  ← template — create this
├── config.yaml                   ← paths, schedules, CR800 settings — create this
├── schema.sql                    ← all CREATE TABLE statements — create this
├── environment.yml               ← already exists — UPDATE (see below)
│
├── src/
│   ├── __init__.py
│   ├── db.py                     ← connect(), init_schema(), upsert_df()
│   ├── ingest.py                 ← orchestrator: route file type → parser → upsert
│   ├── watcher.py                ← watchdog FileSystemEventHandler
│   │
│   ├── fetchers/
│   │   ├── __init__.py
│   │   ├── open_meteo.py         ← fetch() → pd.DataFrame
│   │   └── cr800.py              ← connect(), list_tables(), fetch_since(table, dt)
│   │
│   └── parsers/
│       ├── __init__.py
│       ├── camera_trap_legacy.py ← parse(csv_path) → (deployments_df, media_df, obs_df)
│       ├── camtrap_dp.py         ← parse(folder_path) → (deployments_df, media_df, obs_df)
│       └── toa5.py               ← parse(dat_path) → pd.DataFrame (for CR800 backfill)
│
├── run_fetch.py                  ← entry point: manual run + APScheduler daemon
└── run_watcher.py                ← entry point: watchdog daemon
```

---

## environment.yml Changes Needed

Two packages must be replaced in the existing `environment.yml`:

| Remove | Add | Reason |
|---|---|---|
| `paramiko>=3.4.0` | `pycampbellcr6>=2.1.0` | CR800 uses PakBus, not SSH |
| `schedule>=1.2.0` | `apscheduler>=3.10.0` | APScheduler is cross-platform (Windows + Linux) |

After editing, run: `conda env update -f environment.yml --prune`

---

## Build Phases

Work through these in order. Each phase has a verification step — run it before moving on.

### Phase 1 — Foundation
**Files:** `schema.sql`, `src/db.py`, `config.yaml`, `.env.example`

`src/db.py` must expose:
- `connect() -> duckdb.DuckDBPyConnection` — opens `fma_data.duckdb` from path in config
- `init_schema(con)` — runs `schema.sql`, creates all tables if not exist
- `upsert_df(con, table: str, df: pd.DataFrame)` — `INSERT OR REPLACE INTO {table} SELECT * FROM df`

`config.yaml` structure:
```yaml
database:
  path: /home/fguarda/Dev/Python/fma_data.duckdb

cr800:
  host: 100.97.202.90
  port: 2000
  pakbus_address: 1
  station_id: bosque_pehuen

open_meteo:
  latitude: -39.61
  longitude: -71.71
  fetch_interval_minutes: 60

watcher:
  incoming_dir: data/incoming

schedules:
  open_meteo_interval_minutes: 60
  cr800_interval_minutes: 15
```

**Verify Phase 1:**
```bash
conda activate data-pipeline
python -c "from src.db import connect, init_schema; con = connect(); init_schema(con); print('OK')"
# Should print OK and create fma_data.duckdb
```

---

### Phase 2 — Open-Meteo (first live end-to-end)
**Files:** `src/fetchers/open_meteo.py`, `src/ingest.py`, `run_fetch.py` (minimal)

`open_meteo.py` must:
- Call the Open-Meteo API for Bosque Pehuén coordinates
- Return a `pd.DataFrame` with columns matching `weather_forecast` schema
- All timestamps as UTC (`pd.Timestamp`, tz-aware)

`ingest.py` must:
- `ingest_weather_forecast(con)` → calls fetcher → upserts into `weather_forecast`

`run_fetch.py` (minimal, single run):
```python
# run_fetch.py
import os, sys
from pathlib import Path
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from src.db import connect, init_schema
from src.ingest import ingest_weather_forecast

con = connect()
init_schema(con)
ingest_weather_forecast(con)
print("Done.")
```

**Verify Phase 2:**
```bash
python run_fetch.py
# Then check rows:
python -c "import duckdb; con = duckdb.connect('/home/fguarda/Dev/Python/fma_data.duckdb'); print(con.execute('SELECT COUNT(*) FROM weather_forecast').fetchone())"
```

---

### Phase 3 — Camera Trap Backfill
**Files:** `src/parsers/camera_trap_legacy.py`, `src/parsers/camtrap_dp.py`

Both parsers must return the same tuple: `(deployments_df, media_df, obs_df)`
with columns exactly matching the DuckDB schema column names.

`camera_trap_legacy.py` parses `old animal data DB.csv` (already in the project dir).
Must handle:
- Leading space in `DateTime`
- Building `deploymentID` from `RelativePath`
- Building `eventID` from `Note0`
- Generating deterministic `mediaID` and `observationID`
- Setting `source='legacy'` on all rows

`camtrap_dp.py` parses any folder dropped into `data/incoming/` that contains
`deployments.csv + media.csv + observations.csv`. Map Camtrap DP columns to our schema.
Set `source='camtrap_dp'`.

Add `ingest_camera_trap_legacy(con)` and `ingest_camtrap_dp(con, folder_path)` to `ingest.py`.

**Verify Phase 3:**
```bash
python -c "
from src.parsers.camera_trap_legacy import parse
from pathlib import Path
deps, media, obs = parse(Path('old animal data DB.csv'))
print(f'Deployments: {len(deps)}, Media: {len(media)}, Observations: {len(obs)}')
"
# Then run full ingest:
python -c "
from src.db import connect, init_schema
from src.ingest import ingest_camera_trap_legacy
con = connect(); init_schema(con)
ingest_camera_trap_legacy(con)
"
```

---

### Phase 4 — CR800
**Files:** `src/parsers/toa5.py`, `src/fetchers/cr800.py`

`toa5.py`:
- TOA5 is Campbell Scientific's ASCII data format. Files start with 4 header lines:
  - Line 1: `"TOA5",...` (file info)
  - Line 2: column names
  - Line 3: units
  - Line 4: processing type
  - Line 5+: data rows
- Parse the 4-line header to extract column names
- Return `pd.DataFrame` with `station_id` column added and `TIMESTAMP` converted to UTC
- This is used for one-time historical backfill of the `.dat` files

`cr800.py`:
- Use `pycampbellcr6` library
- `connect(host, port, pakbus_address) -> logger` — returns a connected logger object
- `list_tables(logger) -> list[str]` — discover available data tables
- `fetch_since(logger, table_name, last_timestamp) -> pd.DataFrame` — pull records newer than last fetch
- Track last fetch timestamp per table in a small JSON file `data/cr800_state.json`

Add `ingest_cr800_live(con)` and `ingest_cr800_backfill(con, dat_file_path)` to `ingest.py`.

**Verify Phase 4:**
```bash
# Test connection (CR800 must be reachable on Tailscale):
python -c "
from src.fetchers.cr800 import connect, list_tables
import yaml
cfg = yaml.safe_load(open('config.yaml'))['cr800']
logger = connect(cfg['host'], cfg['port'], cfg['pakbus_address'])
print('Tables:', list_tables(logger))
"
```

---

### Phase 5 — Automation
**Files:** `src/watcher.py`, `run_watcher.py`, updated `run_fetch.py`

`watcher.py`:
- `watchdog` `FileSystemEventHandler` subclass
- Watches `data/incoming/` for new files
- On new file: detect type by extension/contents:
  - `.dat` → TOA5 parser (CR800 backfill)
  - Folder with `deployments.csv` → Camtrap DP parser
  - `.csv` alone → try legacy camera trap parser
- Calls `ingest.py` accordingly

`run_watcher.py`:
- Starts watchdog observer
- Creates `data/incoming/` if missing
- Runs forever until Ctrl+C

`run_fetch.py` (full version):
- Adds APScheduler: Open-Meteo every 60 min, CR800 every 15 min
- Supports `--once` flag for manual single-run (no scheduler)
- Supports `--backfill` flag: `python run_fetch.py --backfill path/to/file.dat`

**Verify Phase 5:**
```bash
# Test watcher:
python run_watcher.py &
cp "old animal data DB.csv" data/incoming/test.csv
# Should auto-ingest within seconds

# Test scheduler:
python run_fetch.py  # runs scheduler loop
```

---

## Code Style (follow schedule-agent patterns)

```python
# Entry point boilerplate (every run_*.py):
import os, sys
from pathlib import Path
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

# Config loading:
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Env/secrets:
from dotenv import load_dotenv
import os
load_dotenv()

# Logging style (no logging module — use print with arrow prefix):
print("→ Fetching Open-Meteo data...")
print(f"  Inserted {n} rows into weather_forecast.")
print(f"  Warning: CR800 unreachable ({e}). Skipping.")
```

---

## .env.example

```dotenv
# Copy to .env and fill in values
DB_PATH=/home/fguarda/Dev/Python/fma_data.duckdb

# CR800 (values already in config.yaml — override here if needed)
CR800_HOST=100.97.202.90
CR800_PORT=2000
CR800_PAKBUS_ADDRESS=1
```

---

## What Already Exists

| Item | Status |
|---|---|
| `data-pipeline/environment.yml` | Exists — needs 2 package changes |
| `data-pipeline/README.md` | Exists — project overview |
| `data-pipeline/old animal data DB.csv` | Exists — 18,473 rows, Phase 3 backfill input |
| `fma_data.duckdb` | Does not exist yet — created in Phase 1 |
| All Python source files | Do not exist yet — build in phase order |

---

## Open Questions (resolve before or during Phase 4)

1. **CR800 internal table names** — run `list_tables()` at first connection, confirm names
   like `Met_15min`, `Met_Daily`. Schema columns finalized from actual TOA5 headers.
2. **Historical `.dat` file locations** — ask the user for the path to the `.dat` files
   before building `run_fetch.py --backfill`.
3. **Camtrap DP folder structure** — confirm with user whether full packages are placed
   as a folder in `data/incoming/` or as individual CSVs.
