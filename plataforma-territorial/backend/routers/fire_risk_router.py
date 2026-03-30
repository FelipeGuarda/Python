"""Fire risk API endpoints — rule-based FRI + ML model."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Query

from ..db import get_connection
from ..fire_risk import risk_components, ml_fire_probability

router = APIRouter(prefix="/api/fire-risk", tags=["fire-risk"])


def _compute_days_without_rain(con, threshold_mm: float = 2.0) -> int:
    """Count consecutive days without significant rain, looking back from today."""
    rows = con.execute("""
        SELECT
            CAST(timestamp AS DATE) as day,
            SUM(precipitation) as total_precip
        FROM weather_station
        WHERE timestamp >= NOW() - INTERVAL '60 days'
        GROUP BY CAST(timestamp AS DATE)
        ORDER BY day DESC
    """).fetchall()
    if not rows:
        return 0
    count = 0
    for day, precip in rows:
        if precip is not None and precip > threshold_mm:
            break
        count += 1
    return count


@router.get("/current")
def fire_risk_current():
    """Current fire risk based on latest weather station data."""
    with get_connection() as con:
        # Latest station reading
        row = con.execute("""
            SELECT
                CAST(timestamp AS TEXT) as timestamp,
                temperature_air, relative_humidity, wind_speed
            FROM weather_station
            ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
        if not row:
            return {"error": "No weather data available"}

        days_no_rain = _compute_days_without_rain(con)

    timestamp, temp_c, rh_pct, wind_ms = row
    wind_kmh = (wind_ms or 0) * 3.6

    # Rule-based index
    components = risk_components(temp_c or 0, rh_pct or 0, wind_kmh, days_no_rain)

    # ML probability
    ml_prob = ml_fire_probability(temp_c or 0, rh_pct or 0, wind_kmh, days_no_rain)

    return {
        "timestamp": timestamp,
        "weather": {
            "temperature_c": temp_c,
            "relative_humidity_pct": rh_pct,
            "wind_speed_kmh": round(wind_kmh, 1),
            "days_without_rain": days_no_rain,
        },
        "rule_based": components,
        "ml_probability": ml_prob,
        "methodology": {
            "rule_based": "FRI index (0-100): temp (0-25) + humidity (0-25 inv) + wind (0-15) + days without rain (0-35). Bins calibrated for La Araucanía.",
            "ml_model": "Random Forest (100 trees) trained on 1,232 fire/non-fire events from La Araucanía 1984-2018. ROC AUC: 0.94.",
        },
    }


@router.get("/forecast")
def fire_risk_forecast():
    """Fire risk forecast for the next 7 days based on Open-Meteo data."""
    with get_connection() as con:
        days_no_rain = _compute_days_without_rain(con)

        # Daily aggregates from forecast (avg of 14:00-16:00 peak hours)
        rows = con.execute("""
            SELECT
                CAST(timestamp AS DATE) as day,
                AVG(temperature_2m) as avg_temp,
                AVG(relative_humidity_2m) as avg_rh,
                AVG(wind_speed_10m) as avg_wind,
                SUM(precipitation) as total_precip
            FROM weather_forecast
            WHERE timestamp >= NOW()
            AND EXTRACT(HOUR FROM timestamp) BETWEEN 14 AND 16
            GROUP BY CAST(timestamp AS DATE)
            ORDER BY day ASC
        """).fetchall()

    result = []
    running_days_no_rain = days_no_rain
    for day, temp, rh, wind, precip in rows:
        if precip is not None and precip > 2.0:
            running_days_no_rain = 0
        else:
            running_days_no_rain += 1

        components = risk_components(temp or 0, rh or 0, (wind or 0) * 3.6, running_days_no_rain)
        ml_prob = ml_fire_probability(temp or 0, rh or 0, (wind or 0) * 3.6, running_days_no_rain)

        result.append({
            "date": str(day),
            "weather": {
                "temperature_c": round(temp, 1) if temp else None,
                "relative_humidity_pct": round(rh, 1) if rh else None,
                "wind_speed_kmh": round((wind or 0) * 3.6, 1),
                "precipitation_mm": round(precip, 1) if precip else 0,
                "days_without_rain": running_days_no_rain,
            },
            "rule_based": components,
            "ml_probability": ml_prob,
        })

    return result


@router.get("/history")
def fire_risk_history(days: int = Query(default=30, le=60)):
    """Historical fire risk for the past N days computed from weather station data.

    Uses afternoon peak hours (14–16 local time) to mirror the forecast methodology.
    Runs days_without_rain forward from 60 days earlier to give an accurate count.
    """
    with get_connection() as con:
        lookback = days + 60
        rows = con.execute(f"""
            SELECT
                CAST(timezone('America/Santiago', timestamp) AS DATE) as day,
                AVG(temperature_air)       as avg_temp,
                AVG(relative_humidity)     as avg_rh,
                AVG(wind_speed * 3.6)      as avg_wind_kmh,
                SUM(precipitation)         as total_precip
            FROM weather_station
            WHERE timestamp >= NOW() - INTERVAL '{lookback} days'
              AND EXTRACT(HOUR FROM timezone('America/Santiago', timestamp)) BETWEEN 14 AND 16
            GROUP BY CAST(timezone('America/Santiago', timestamp) AS DATE)
            ORDER BY day ASC
        """).fetchall()

    cutoff_date = (datetime.utcnow() - timedelta(days=days)).date()
    running_dnr = 0
    result = []

    for day, temp, rh, wind_kmh, precip in rows:
        if precip is not None and precip > 2.0:
            running_dnr = 0
        else:
            running_dnr += 1

        if day >= cutoff_date:
            components = risk_components(temp or 0, rh or 0, wind_kmh or 0, running_dnr)
            result.append({
                "date": str(day),
                "weather": {
                    "temperature_c": round(temp, 1) if temp else None,
                    "relative_humidity_pct": round(rh, 1) if rh else None,
                    "wind_speed_kmh": round(wind_kmh, 1) if wind_kmh else None,
                    "days_without_rain": running_dnr,
                },
                "rule_based": components,
                "ml_probability": ml_fire_probability(
                    temp or 0, rh or 0, wind_kmh or 0, running_dnr
                ),
                "is_historical": True,
            })

    return result
