from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from ..db import get_con

router = APIRouter()

STATION_ID = "bosque_pehuen"

# All columns we allow the frontend to request (validated to prevent SQL injection)
ALLOWED_COLS: set[str] = {
    "temperature_air",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "solar_radiation",
    "BP_mbar_Avg",
    "PtoRocio_Avg",
    "T107_10cm_Avg",
    "T107_50cm_Avg",
    "albedo_Avg",
    "DT_Avg",
}

RESAMPLE_FREQ = {
    "15min": "15min",
    "D": "D",
    "ME": "ME",
    "Q": "QE",
}

WIND_SPEED_BINS = [0, 3, 6, 9, 12, 15, float("inf")]
WIND_SPEED_LABELS = ["0–3", "3–6", "6–9", "9–12", "12–15", "≥15"]
DIR_LABELS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]


# ── /current ──────────────────────────────────────────────────────────────────

@router.get("/current")
def get_current() -> dict[str, Any]:
    con = get_con()
    cols = [
        "timestamp", "temperature_air", "relative_humidity",
        "wind_speed", "wind_direction", "precipitation",
        "solar_radiation", "BP_mbar_Avg", "PtoRocio_Avg",
        "T107_10cm_Avg", "T107_50cm_Avg", "albedo_Avg", "DT_Avg",
    ]
    row = con.execute(f"""
        SELECT {', '.join(cols)}
        FROM weather_station
        WHERE station_id = '{STATION_ID}'
        ORDER BY timestamp DESC
        LIMIT 1
    """).fetchone()
    con.close()

    if not row:
        return {}

    result: dict[str, Any] = dict(zip(cols, row))
    result["timestamp"] = result["timestamp"].isoformat()
    if result.get("wind_speed") is not None:
        result["wind_speed_kmh"] = round(float(result["wind_speed"]) * 3.6, 1)
    return result


# ── /history ──────────────────────────────────────────────────────────────────

@router.get("/history")
def get_history(
    start: date = Query(default=None),
    end: date = Query(default=None),
    resolution: str = Query(default="D"),
    variables: str = Query(default="temperature_air,relative_humidity,precipitation"),
) -> dict[str, Any]:
    # Validate and sanitise variable list
    requested = [v.strip() for v in variables.split(",")]
    valid_vars = [v for v in requested if v in ALLOWED_COLS]
    if not valid_vars:
        return {"data": [], "wind_rose": None, "stats": {}}

    # Always fetch both wind columns together for proper vector resampling
    need_wind = "wind_speed" in valid_vars or "wind_direction" in valid_vars
    fetch_cols = list(dict.fromkeys(
        valid_vars + (["wind_speed", "wind_direction"] if need_wind else [])
    ))

    con = get_con()

    # Build date filter — interpret user dates in Santiago local time
    where = [f"station_id = '{STATION_ID}'"]
    params: list[str] = []
    if start:
        where.append(
            "timestamp >= timezone('UTC', timezone('America/Santiago', CAST(? AS TIMESTAMP)))"
        )
        params.append(f"{start} 00:00:00")
    if end:
        where.append(
            "timestamp <= timezone('UTC', timezone('America/Santiago', CAST(? AS TIMESTAMP)))"
        )
        params.append(f"{end} 23:59:59")

    where_sql = " AND ".join(where)
    cols_sql = ", ".join(["timestamp"] + fetch_cols)

    df: pd.DataFrame = con.execute(
        f"SELECT {cols_sql} FROM weather_station WHERE {where_sql} ORDER BY timestamp",
        params,
    ).df()

    # Global DT_Avg max for snow depth derivation (over whole station history)
    dt_global_max: float | None = None
    if "DT_Avg" in fetch_cols:
        row = con.execute(
            f"SELECT MAX(DT_Avg) FROM weather_station WHERE station_id = '{STATION_ID}' AND DT_Avg IS NOT NULL"
        ).fetchone()
        if row:
            dt_global_max = row[0]

    con.close()

    if df.empty:
        return {"data": [], "wind_rose": None, "stats": {}}

    # Parse timestamps (DuckDB returns tz-aware Timestamps)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    # Derive snow depth: distance-sensor max − current distance, clipped to 0
    if "DT_Avg" in df.columns and dt_global_max is not None:
        df["DT_Avg"] = (dt_global_max - df["DT_Avg"]).clip(lower=0)

    # Wind rose from raw 15-min data (before resampling for better distribution)
    wind_rose_data = None
    if need_wind and "wind_speed" in df.columns and "wind_direction" in df.columns:
        ws_kmh = df["wind_speed"].dropna() * 3.6
        wd = df["wind_direction"].dropna()
        aligned = ws_kmh.align(wd, join="inner")
        wind_rose_data = _compute_wind_rose(aligned[1], aligned[0])

    # Resample
    freq = RESAMPLE_FREQ.get(resolution, "D")
    df_resampled = _resample(df, freq)

    # Convert wind speed m/s → km/h for display
    if "wind_speed" in df_resampled.columns:
        df_resampled["wind_speed"] = df_resampled["wind_speed"] * 3.6

    # Stats (exclude wind_direction — circular mean is not meaningful here)
    stat_cols = [v for v in valid_vars if v in df_resampled.columns and v != "wind_direction"]
    stats: dict[str, dict] = {}
    for col in stat_cols:
        s = df_resampled[col].describe()
        stats[col] = {k: round(float(v), 2) for k, v in s.items() if k in ("count", "mean", "std", "min", "max")}

    # Serialise — format timestamp based on resolution
    df_out = df_resampled.copy()
    df_out.index = df_out.index.tz_convert("America/Santiago")
    if freq == "15min":
        df_out.index = df_out.index.strftime("%Y-%m-%d %H:%M")
    else:
        df_out.index = df_out.index.strftime("%Y-%m-%d")
    df_out.index.name = "timestamp"
    df_out = df_out.reset_index()

    # Keep only the originally-requested valid columns
    out_cols = ["timestamp"] + [v for v in valid_vars if v in df_out.columns]
    df_out = df_out[out_cols]

    # Replace NaN with None so JSON serialises correctly
    data = df_out.where(df_out.notna(), other=None).to_dict(orient="records")

    return {"data": data, "wind_rose": wind_rose_data, "stats": stats}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample with wind vector decomposition and correct precipitation sum."""
    # Separate wind columns for special handling
    wind_present = "wind_speed" in df.columns and "wind_direction" in df.columns

    # Build aggregation rules for non-wind columns
    agg: dict[str, str] = {}
    for col in df.columns:
        if col in ("wind_speed", "wind_direction"):
            continue
        agg[col] = "sum" if col == "precipitation" else "mean"

    if agg:
        result = df.drop(
            columns=[c for c in ("wind_speed", "wind_direction") if c in df.columns]
        ).resample(freq).agg(agg)
    else:
        result = pd.DataFrame(index=df.resample(freq).mean().index)

    if wind_present:
        ws = df["wind_speed"].fillna(0)
        wd = df["wind_direction"]
        u = ws * np.sin(np.radians(wd))
        v = ws * np.cos(np.radians(wd))
        u_r = u.resample(freq).mean()
        v_r = v.resample(freq).mean()
        result["wind_speed"] = np.sqrt(u_r**2 + v_r**2)
        result["wind_direction"] = (np.degrees(np.arctan2(u_r, v_r)) % 360)

    return result


def _compute_wind_rose(
    direction: pd.Series, speed_kmh: pd.Series
) -> list[dict] | None:
    """Bin wind into 16 directions × 6 speed classes, return as % of total."""
    df = pd.DataFrame({"dir": direction.values, "spd": speed_kmh.values}).dropna()
    total = len(df)
    if total < 2:
        return None

    result = []
    for i, label in enumerate(DIR_LABELS):
        # North sector wraps around 360°
        if i == 0:
            mask = (df["dir"] >= 348.75) | (df["dir"] < 11.25)
        else:
            lo = i * 22.5 - 11.25
            hi = lo + 22.5
            mask = (df["dir"] >= lo) & (df["dir"] < hi)

        sector = df[mask]
        bins = []
        for j, sl in enumerate(WIND_SPEED_LABELS):
            lo_s = WIND_SPEED_BINS[j]
            hi_s = WIND_SPEED_BINS[j + 1]
            count = int(((sector["spd"] >= lo_s) & (sector["spd"] < hi_s)).sum())
            bins.append({"range": sl, "pct": round(count / total * 100, 2)})

        result.append({
            "direction": label,
            "angle": i * 22.5,
            "total_pct": round(len(sector) / total * 100, 2),
            "bins": bins,
        })

    return result
