"""Weather API endpoints — station data + forecast from DuckDB."""

from fastapi import APIRouter, Query

from ..db import get_connection

router = APIRouter(prefix="/api/weather", tags=["weather"])


@router.get("/current")
def weather_current():
    """Latest weather station reading."""
    with get_connection() as con:
        row = con.execute("""
            SELECT
                CAST(timestamp AS TEXT) as timestamp,
                temperature_air, relative_humidity,
                wind_speed, wind_direction,
                precipitation, solar_radiation, battery_voltage
            FROM weather_station
            ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
    if not row:
        return {"error": "No station data available"}
    cols = [
        "timestamp", "temperature_air", "relative_humidity",
        "wind_speed", "wind_direction", "precipitation",
        "solar_radiation", "battery_voltage",
    ]
    return dict(zip(cols, row))


@router.get("/history")
def weather_history(hours: int = Query(default=24, le=720)):
    """Station readings for the last N hours."""
    with get_connection() as con:
        rows = con.execute(f"""
            SELECT
                CAST(timestamp AS TEXT) as timestamp,
                temperature_air, relative_humidity,
                wind_speed, wind_direction, precipitation
            FROM weather_station
            WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
            ORDER BY timestamp ASC
        """).fetchall()
    cols = [
        "timestamp", "temperature_air", "relative_humidity",
        "wind_speed", "wind_direction", "precipitation",
    ]
    return [dict(zip(cols, r)) for r in rows]


@router.get("/forecast")
def weather_forecast(hours: int = Query(default=168, le=168)):
    """Open-Meteo hourly forecast (up to 7 days)."""
    with get_connection() as con:
        rows = con.execute(f"""
            SELECT
                CAST(timestamp AS TEXT) as timestamp,
                temperature_2m, relative_humidity_2m,
                precipitation, wind_speed_10m, wind_direction_10m,
                et0_fao_evapotranspiration
            FROM weather_forecast
            WHERE timestamp >= NOW()
            ORDER BY timestamp ASC
            LIMIT {hours}
        """).fetchall()
    cols = [
        "timestamp", "temperature_2m", "relative_humidity_2m",
        "precipitation", "wind_speed_10m", "wind_direction_10m",
        "et0_fao_evapotranspiration",
    ]
    return [dict(zip(cols, r)) for r in rows]
