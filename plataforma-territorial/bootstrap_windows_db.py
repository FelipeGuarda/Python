"""
Bootstrap a minimal fma_data.duckdb for Windows development.

Since the data-pipeline systemd service only runs on Linux, this script
seeds a local DuckDB with:
  1. Open-Meteo 7-day forecast → weather_forecast table
  2. Open-Meteo 90-day historical archive → weather_station table (proxy)

This lets the platform backend serve real data for the Meteo and Riesgo
de Incendio tabs without needing the CR800 datalogger or Tailscale VPN.

Usage (from plataforma-territorial/):
    conda run -n plataforma-territorial python bootstrap_windows_db.py
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

LAT = -39.61
LON = -71.71
STATION_ID = "bosque_pehuen"

# DB lives one level up (same location the backend expects)
DB_PATH = Path(__file__).resolve().parent.parent / "fma_data.duckdb"

SCHEMA = """
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
);

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
    source          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ct_media (
    mediaID       TEXT PRIMARY KEY,
    deploymentID  TEXT NOT NULL,
    timestamp     TIMESTAMPTZ,
    fileName      TEXT,
    filePath      TEXT,
    fileMediatype TEXT,
    source        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ct_observations (
    observationID             TEXT PRIMARY KEY,
    deploymentID              TEXT NOT NULL,
    mediaID                   TEXT,
    eventID                   TEXT,
    eventStart                TIMESTAMPTZ,
    eventEnd                  TIMESTAMPTZ,
    observationType           TEXT,
    scientificName            TEXT,
    count                     INTEGER,
    classificationMethod      TEXT,
    classificationProbability DOUBLE,
    source                    TEXT NOT NULL
);

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
"""

# ── Fetch helpers ──────────────────────────────────────────────────────────────

def fetch_forecast() -> pd.DataFrame:
    """Fetch 7-day hourly forecast from Open-Meteo."""
    print("→ Fetching Open-Meteo forecast (7 days)...")
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "et0_fao_evapotranspiration",
        ]),
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    now = datetime.now(timezone.utc)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"], utc=True),
        "temperature_2m": h["temperature_2m"],
        "relative_humidity_2m": h["relative_humidity_2m"],
        "precipitation": h["precipitation"],
        "wind_speed_10m": h["wind_speed_10m"],
        "wind_direction_10m": h["wind_direction_10m"],
        "et0_fao_evapotranspiration": h["et0_fao_evapotranspiration"],
        "fetched_at": now,
    })
    print(f"  Got {len(df)} rows.")
    return df


def fetch_historical(days: int = 90) -> pd.DataFrame:
    """
    Fetch historical hourly data from Open-Meteo archive API.
    Maps to weather_station schema. Wind speed stored in m/s
    to match CR800 convention (backend multiplies by 3.6 for display).
    """
    end = datetime.now().date() - timedelta(days=1)
    start = end - timedelta(days=days)
    print(f"→ Fetching Open-Meteo historical archive ({start} → {end})...")
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": str(start),
        "end_date": str(end),
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
        ]),
        "wind_speed_unit": "ms",  # store in m/s, matching CR800
        "timezone": "UTC",
    }
    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame({
        "station_id": STATION_ID,
        "timestamp": pd.to_datetime(h["time"], utc=True),
        "temperature_air": h["temperature_2m"],
        "relative_humidity": h["relative_humidity_2m"],
        "precipitation": h["precipitation"],
        "wind_speed": h["wind_speed_10m"],       # m/s
        "wind_direction": h["wind_direction_10m"],
        "solar_radiation": h["shortwave_radiation"],
        "battery_voltage": None,
    })
    print(f"  Got {len(df)} rows.")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nBootstrapping Windows DuckDB at: {DB_PATH}\n")

    con = duckdb.connect(str(DB_PATH))
    con.execute(SCHEMA)
    print("✓ Schema created.")

    # Forecast
    forecast_df = fetch_forecast()
    con.execute("DELETE FROM weather_forecast")
    con.register("forecast_df", forecast_df)
    con.execute("INSERT INTO weather_forecast SELECT * FROM forecast_df")
    count = con.execute("SELECT COUNT(*) FROM weather_forecast").fetchone()[0]
    print(f"✓ weather_forecast: {count} rows inserted.")

    # Historical (weather_station proxy)
    hist_df = fetch_historical(days=90)
    con.execute(f"DELETE FROM weather_station WHERE station_id = '{STATION_ID}'")
    con.register("hist_df", hist_df)
    con.execute("""
        INSERT OR REPLACE INTO weather_station
        SELECT station_id, timestamp, temperature_air, relative_humidity,
               wind_speed, wind_direction, precipitation, solar_radiation, battery_voltage
        FROM hist_df
    """)
    count = con.execute("SELECT COUNT(*) FROM weather_station").fetchone()[0]
    print(f"✓ weather_station: {count} rows inserted (Open-Meteo archive proxy).")

    con.close()
    print(f"\nDone. Restart the backend to pick up the new DB.\n")
    print("  conda run -n plataforma-territorial uvicorn backend.main:app --port 8000")


if __name__ == "__main__":
    main()
