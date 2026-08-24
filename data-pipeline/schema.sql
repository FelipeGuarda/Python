-- FMA Data Pipeline — DuckDB Schema
-- All timestamps stored as UTC (TIMESTAMPTZ). Convert to America/Santiago at display time.

-- CR800 Campbell Scientific sensor readings
CREATE TABLE IF NOT EXISTS weather_station (
    station_id        TEXT NOT NULL,
    timestamp         TIMESTAMPTZ NOT NULL,
    temperature_air   DOUBLE,
    relative_humidity DOUBLE,
    wind_speed        DOUBLE,
    wind_direction    DOUBLE,
    precipitation     DOUBLE,
    solar_radiation   DOUBLE,
    battery_voltage   DOUBLE,
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

-- ── Camera trap ──────────────────────────────────────────────────────────────
-- Rebuilt from camera-traps/data/campaigns/<campaign>/observations.parquet by
-- src/parsers/canonical_ct.py. NEVER hand-populated and never parsed from a
-- Timelapse export: the canonical parquet has already resolved species, review
-- verdict, effort validity and clock repair, and re-deriving any of them here is
-- what made the previous implementation wrong on 515 rows.
--
-- TIMESTAMPS ARE NAIVE LOCAL WALL TIME (America/Santiago), not TIMESTAMPTZ, and
-- that is deliberate. A camera clock reading is a wall-time reading of unknown
-- accuracy; there is no instant to recover, and 11% of rows have no datetime at
-- all. Storing TIMESTAMPTZ would force this table to invent a UTC offset per row
-- -- ambiguous twice a year at the DST boundary -- and would make HOUR(eventStart)
-- depend on the reader's session timezone rather than on what the camera saw.
-- The diel-activity figure needs the camera's local hour, so local is the honest
-- and the useful storage. Contrast weather_station above, which is TIMESTAMPTZ
-- because a datalogger reading IS a known instant.
-- See camera-traps/docs/V2-REVIEW.md 2.3 (which said UTC; this supersedes it) and
-- the horario-de-invierno decision: no DST correction, ever.

-- One row per deployment (station × campaign)
CREATE TABLE IF NOT EXISTS ct_deployments (
    deploymentID           TEXT PRIMARY KEY,   -- '<campaign>_<station>', e.g. otono_2025_CT01
    locationID             TEXT,               -- camera number as text; the platform parses int()
    locationName           TEXT,               -- canonical station, e.g. 'CT01'
    campaign               TEXT,               -- canonical slug, e.g. 'otono_2025'
    latitude               DOUBLE,
    longitude              DOUBLE,
    deploymentStart        TIMESTAMP,
    deploymentEnd          TIMESTAMP,
    deploymentWindowSource TEXT,               -- 'observed_media' | 'field_record'
    cameraID               TEXT,
    cameraModel            TEXT,
    habitat                TEXT,
    source                 TEXT NOT NULL       -- 'canonical_parquet'
);

-- One row per still. Every still in the gated export, not only the reviewed ones:
-- a station absent from this table is indistinguishable from one never deployed,
-- which is fine for a detection numerator and wrong for an effort denominator.
CREATE TABLE IF NOT EXISTS ct_media (
    mediaID       TEXT PRIMARY KEY,   -- derived from (campaign, station, fileName), never a Timelapse GUID
    deploymentID  TEXT NOT NULL,
    timestamp     TIMESTAMP,          -- NULL where the clock failed; the row still counts for presence
    fileName      TEXT,
    filePath      TEXT,
    fileMediatype TEXT,
    source        TEXT NOT NULL
);

-- One row per observation, media-level (one per still, matching ct_media 1:1)
CREATE TABLE IF NOT EXISTS ct_observations (
    observationID             TEXT PRIMARY KEY,
    deploymentID              TEXT NOT NULL,
    mediaID                   TEXT,
    eventID                   TEXT,   -- always NULL: measured empty in all three campaigns
    eventStart                TIMESTAMP,
    eventEnd                  TIMESTAMP,
    observationType           TEXT,   -- 'animal'|'human'|'blank'|'unknown'|'vehicle'
    scientificName            TEXT,   -- Latin binomial; NULL unless observationType='animal'
    count                     INTEGER,-- always NULL: measured empty in all three campaigns
    classificationMethod      TEXT,   -- 'human' where a reviewer ruled, else 'machine'
    classificationProbability DOUBLE,
    observationComments       TEXT,   -- the reviewer's verbatim text
    reviewOutcome             TEXT,   -- 'confirmed' | 'corrected' | NULL for sweep-only rows
    reviewResolution          TEXT,   -- which rule decided this row's verdict
    source                    TEXT NOT NULL
);

-- Camera-trap ingest state — the consumer half of the canonical contract gate.
-- camera-traps publishes CANONICAL_STATE.json; this records what was last ingested
-- from it, so a stale database is a detectable condition rather than a silent one.
CREATE TABLE IF NOT EXISTS ct_ingest_state (
    campaign     TEXT PRIMARY KEY,
    n_rows       BIGINT NOT NULL,
    n_stations   BIGINT,
    parquet_hash TEXT,
    ingested_at  TIMESTAMPTZ NOT NULL
);

-- REMOVED 2026-08-24: `literature`. It held 0 rows here and no code in this repo
-- read or wrote it -- literature-agent is standalone and mails its summaries. The
-- DDL is deleted rather than left in place because init_schema() runs on every
-- connect, so a CREATE IF NOT EXISTS would recreate the empty table forever.
-- Restore this block if literature-agent is ever pointed at the warehouse:
--   literature(paperID TEXT PRIMARY KEY, title, authors, published_date DATE,
--              source, url, summary_es, fetched_at TIMESTAMPTZ, week_of DATE)
