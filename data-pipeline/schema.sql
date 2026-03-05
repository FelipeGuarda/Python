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
    mediaID       TEXT PRIMARY KEY,
    deploymentID  TEXT NOT NULL,
    timestamp     TIMESTAMPTZ,
    fileName      TEXT,
    filePath      TEXT,
    fileMediatype TEXT,
    source        TEXT NOT NULL
);

-- Camera trap: one row per identified observation (event or media level)
CREATE TABLE IF NOT EXISTS ct_observations (
    observationID             TEXT PRIMARY KEY,
    deploymentID              TEXT NOT NULL,
    mediaID                   TEXT,
    eventID                   TEXT,
    eventStart                TIMESTAMPTZ,
    eventEnd                  TIMESTAMPTZ,
    observationType           TEXT,   -- 'animal' | 'human' | 'blank' | 'unknown'
    scientificName            TEXT,   -- Spanish common name (legacy) or scientific name (camtrap_dp)
    count                     INTEGER,
    classificationMethod      TEXT,   -- 'human' | 'machine'
    classificationProbability DOUBLE,
    source                    TEXT NOT NULL
);

-- Literature: paper metadata + summaries written by the literatura-agent
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
