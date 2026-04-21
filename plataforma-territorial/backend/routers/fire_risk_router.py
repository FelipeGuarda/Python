"""Fire risk API endpoints — rule-based FRI + ML model."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Query

from ..db import get_connection
from ..fire_risk import risk_components, ml_fire_probability

router = APIRouter(prefix="/api/fire-risk", tags=["fire-risk"])


def _compute_days_without_rain(con, threshold_mm: float = 2.0) -> int:
    """Count consecutive days without significant rain using Open-Meteo forecast data.

    A day counts as rainy if its total daily precipitation exceeds threshold_mm.
    Uses full daily totals (all hours), not just a peak-hour window.
    Includes today's full forecast but excludes future days.
    """
    rows = con.execute("""
        SELECT
            CAST(timestamp AS DATE) as day,
            SUM(precipitation) as total_precip
        FROM weather_forecast
        WHERE CAST(timestamp AS DATE) <= CURRENT_DATE
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
    """Current fire risk based on today's Open-Meteo forecast (14–16h peak window)."""
    with get_connection() as con:
        row = con.execute("""
            SELECT
                MAX(fetched_at)              AS fetched_at,
                AVG(temperature_2m)          AS avg_temp,
                AVG(relative_humidity_2m)    AS avg_rh,
                AVG(wind_speed_10m)          AS avg_wind,
                AVG(wind_direction_10m)      AS avg_wind_dir
            FROM weather_forecast
            WHERE CAST(timestamp AS DATE) = CURRENT_DATE
              AND EXTRACT(HOUR FROM timestamp) BETWEEN 14 AND 16
        """).fetchone()

        if not row or row[1] is None:
            return {"error": "No forecast data available for today"}

        days_no_rain = _compute_days_without_rain(con)

    fetched_at, temp_c, rh_pct, wind_ms, wind_dir = row
    wind_kmh = (wind_ms or 0) * 3.6

    components = risk_components(temp_c or 0, rh_pct or 0, wind_kmh, days_no_rain)
    ml_prob = ml_fire_probability(temp_c or 0, rh_pct or 0, wind_kmh, days_no_rain)

    return {
        "timestamp": str(fetched_at),
        "weather": {
            "temperature_c": round(temp_c, 1) if temp_c is not None else None,
            "relative_humidity_pct": round(rh_pct, 1) if rh_pct is not None else None,
            "wind_speed_kmh": round(wind_kmh, 1),
            "wind_direction": round(wind_dir, 1) if wind_dir is not None else None,
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
    """Historical fire risk for the past N days from Open-Meteo forecast data.

    Uses 14–16h peak-hour averages for weather conditions (consistent with /forecast).
    Uses full daily precipitation sum for days_without_rain running count.
    Fetches an extra 60-day warmup window so the dry-days counter starts accurately.
    """
    with get_connection() as con:
        lookback = days + 60
        rows = con.execute(f"""
            SELECT
                CAST(timestamp AS DATE)                                                       AS day,
                AVG(CASE WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 14 AND 16 THEN temperature_2m       END) AS avg_temp,
                AVG(CASE WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 14 AND 16 THEN relative_humidity_2m END) AS avg_rh,
                AVG(CASE WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 14 AND 16 THEN wind_speed_10m * 3.6 END) AS avg_wind_kmh,
                SUM(precipitation)                                                            AS total_precip
            FROM weather_forecast
            WHERE CAST(timestamp AS DATE) <= CURRENT_DATE
              AND CAST(timestamp AS DATE) >= CURRENT_DATE - INTERVAL '{lookback} days'
            GROUP BY 1
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
