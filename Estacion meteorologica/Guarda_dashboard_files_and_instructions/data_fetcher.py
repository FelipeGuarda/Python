# -*- coding: utf-8 -*-
"""
Data fetching functions for Open-Meteo API
"""

from typing import Dict
import requests
import pandas as pd
import streamlit as st

from config import TZ


@st.cache_data(ttl=1800)
def fetch_open_meteo(lat: float, lon: float, days_ahead: int = 7) -> Dict[str, pd.DataFrame]:
    """Fetch hourly & daily forecast + recent past from Open-Meteo."""
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
        ],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "forecast_days": days_ahead,
        "past_days": 60,
        "timezone": TZ,
    }
    
    q = params.copy()
    q["hourly"] = ",".join(params["hourly"]) 
    q["daily"] = ",".join(params["daily"]) 

    r = requests.get(base, params=q, timeout=30)
    r.raise_for_status()
    data = r.json()

    # hourly
    h = pd.DataFrame(data["hourly"]).rename(columns={
        "time": "timestamp",
        "temperature_2m": "temp_c",
        "relative_humidity_2m": "rh_pct",
        "wind_speed_10m": "wind_ms",
        "wind_direction_10m": "wind_dir",
        "precipitation": "precip_mm",
    })
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=False).dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert(TZ)
    h["date"] = h["timestamp"].dt.date
    h["wind_kmh"] = h["wind_ms"] * 3.6

    # daily
    d = pd.DataFrame(data["daily"]).rename(columns={
        "time": "date",
        "temperature_2m_max": "tmax_c",
        "temperature_2m_min": "tmin_c",
        "precipitation_sum": "precip_mm",
        "wind_speed_10m_max": "wind_max_ms",
    })
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d["wind_max_kmh"] = d["wind_max_ms"] * 3.6

    return {"hourly": h, "daily": d}


