# -*- coding: utf-8 -*-

"""
Fire-Risk Dashboard (MVP)
Stack: Streamlit + Plotly + pydeck + Open-Meteo

Notes
- Uses your 0–25 scoring tables (sum to 100) for Temperature, Humidity, Wind Speed, and Days without Rain.
- Days without rain: counts a day as rainy when daily precip sum > 2 mm (as per your rule), uses
  station history if provided; otherwise falls back to Open-Meteo past 60 days to build the counter.
- Regional map: samples a coarse lat/lon grid around the focal point and computes forecast-day risk.
- Animated polar: for MVP we render an interactive per-day Polar chart; a day slider gives an
  "animation" effect without heavy Matplotlib animations in Streamlit.

How to run
    pip install -r requirements.txt
    streamlit run app.py

Optional (for UTM conversion): pip install pyproj
"""

from __future__ import annotations
import os
import math
import json
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk

# ---------------------------
# Config
# ---------------------------
st.set_page_config(
    page_title="Fire Risk Dashboard — Bosque Pehuén",
    layout="wide",
)

TZ = "America/Santiago"
TODAY = pd.Timestamp.now(tz=TZ).normalize()

# Default approximate lat/lon for Bosque Pehuén / Palguín area (to be confirmed)
DEFAULT_LAT = -39.45
DEFAULT_LON = -71.80

# Score tables (as provided)
TEMP_BINS = [
    (-np.inf, 0, 2.7),
    (0, 5, 5.4),
    (6, 10, 8.1),
    (11, 15, 10.8),
    (16, 20, 13.5),
    (21, 25, 16.2),
    (26, 30, 18.9),
    (31, 35, 21.6),
    (35, np.inf, 25.0)
]
RH_BINS = [
    (0, 10, 25.0),
    (11, 20, 22.5),
    (21, 30, 20.0),
    (31, 40, 17.5),
    (41, 50, 15.0),
    (51, 60, 12.5),
    (61, 70, 10.0),
    (71, 80, 7.5),
    (81, 90, 5.0),
    (91, 100, 2.5)
]
WIND_BINS = [
    (-np.inf, 3.0, 1.5),
    (3.0, 5.9, 3.0),
    (6.0, 8.9, 4.5),
    (9.0, 11.9, 6.0),
    (12.0, 14.9, 7.5),
    (15.0, 17.9, 9.0),
    (18.0, 20.9, 10.5),
    (21.0, 23.9, 12.0),
    (24.0, 26.9, 13.5),
    (27.0, np.inf, 15.0),
]
DAYS_NR_BINS = [
    (0, 1, 3.5),
    (2, 4, 7.0),
    (5, 7, 10.5),
    (8, 10, 14.0),
    (11, 13, 17.5),
    (14, 16, 21.0),
    (17, 19, 24.5),
    (20, 22, 28.0),
    (23, 25, 31.5),
    (26, np.inf, 35.0),
]

RISK_COLORS = [
    (0.0, 19.999, "#2e7d32"),   # green
    (20.0, 39.999, "#c0ca33"),  # yellow-green
    (40.0, 59.999, "#fbc02d"),  # yellow 
    (60.0, 79.999, "#fb8c00"),  # orange
    (80.0, 89.999, "#e53935"),  # red-orange
    (90.0, 100.0, "#b71c1c"),   # dark red
]


# ---------------------------
# Utilities
# ---------------------------

def _bin_score(value: float, bins: List[Tuple[float, float, float]]) -> float:
    """Return score for value according to inclusive/exclusive bins.
    For ranges like 0-5, 6-10 etc., we'll treat lower bound inclusive, upper bound inclusive when it's the last numeric.
    We'll be careful with decimals.
    """
    for lo, hi, sc in bins:
        # inclusive on upper edge if hi == np.inf else inclusive lower, inclusive upper for integer-like ranges
        if lo == -np.inf and value < hi:
            return sc
        if hi == np.inf and value > lo:
            return sc
        if value >= lo and value <= hi:
            return sc
    return 0.0


def risk_components(temp_c: float, rh_pct: float, wind_kmh: float, days_no_rain: int) -> Dict[str, float]:
    t = _bin_score(temp_c, TEMP_BINS)
    h = _bin_score(rh_pct, RH_BINS)
    w = _bin_score(wind_kmh, WIND_BINS)
    d = _bin_score(days_no_rain, DAYS_NR_BINS)
    return {"temp": t, "rh": h, "wind": w, "days": d, "total": t + h + w + d}


def color_for_risk(total: float) -> str:
    """Return hex color for risk score, enforcing correct numeric mapping."""
    for lo, hi, col in sorted(RISK_COLORS, key=lambda x: x[0]):
        if lo <= total <= hi:
            return col
    return RISK_COLORS[-1][2]  # default highest (dark red)


# ---------------------------
# Data fetchers — Open-Meteo
# ---------------------------

@st.cache_data(ttl=1800)
def fetch_open_meteo(lat: float, lon: float, days_ahead: int = 7) -> Dict[str, pd.DataFrame]:
    """Fetch hourly & daily forecast + recent past from Open-Meteo (no key).
    We'll request: temp (2m), RH, wind speed, wind direction, precipitation.
    """
    # assemble params
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
    # requests allows comma-separated strings; convert lists
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
    # convert wind to km/h
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


def compute_days_without_rain(daily_df: pd.DataFrame, rain_threshold_mm: float = 2.0) -> pd.DataFrame:
    """Compute a running counter of consecutive days without rain using daily precip sums."""
    df = daily_df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    dry = []
    counter = 0
    for mm in df["precip_mm"]:
        if mm is not None and not pd.isna(mm) and mm > rain_threshold_mm:
            counter = 0
        else:
            counter += 1
        dry.append(counter)
    df["days_no_rain"] = dry
    return df


def best_hour_by_day(hourly: pd.DataFrame, days_nr_map: Dict[dt.date, int]) -> pd.DataFrame:
    """
    Compute risk scores for 14:00–16:00 hours and average them per day.
    Uses temp (°C), RH (%), wind (km/h), and daily days_without_rain from days_nr_map.
    Returns one row per date with averaged values and risk scores.
    """
    rows = []
    for d, sub in hourly.groupby("date"):
        if sub.empty:
            continue

        # ensure km/h exists and numeric
        sub = sub.copy()
        sub["wind_kmh"] = pd.to_numeric(sub["wind_ms"], errors="coerce") * 3.6
        sub["hour"] = pd.to_datetime(sub["timestamp"]).dt.hour

        # select 14–16 local hours (2–4 PM)
        sub_window = sub[(sub["hour"] >= 14) & (sub["hour"] <= 16)]
        if sub_window.empty:
            continue

        dnr = int(days_nr_map.get(d, 0))

        # compute risk components for each hour
        comps = sub_window.apply(
            lambda r: risk_components(
                float(r["temp_c"]),
                float(r["rh_pct"]),
                float(r["wind_kmh"]),
                dnr,
            ),
            axis=1,
        )
        comp_df = pd.DataFrame(list(comps))
        sub2 = pd.concat([sub_window.reset_index(drop=True), comp_df], axis=1)

        # average over 2–4 PM window
        mean_row = sub2.mean(numeric_only=True)
        rows.append({
            "date": d,
            "timestamp": sub2["timestamp"].iloc[0],  # representative start hour
            "temp_c": float(mean_row["temp_c"]),
            "rh_pct": float(mean_row["rh_pct"]),
            "wind_kmh": float(mean_row["wind_kmh"]),
            "days_no_rain": dnr,
            "temp": float(mean_row["temp"]),
            "rh": float(mean_row["rh"]),
            "wind": float(mean_row["wind"]),
            "days": float(mean_row["days"]),
            "total": float(mean_row["total"]),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------
# ---------------------------
# UI — Sidebar controls (fixed location from UTM 19S, WGS84)
# ---------------------------

# OPTION A: compute lat/lon from UTM (requires pyproj)
try:
    import pyproj  # pip install pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(263221.0, 5630634.0)
    lat, lon = float(lat), float(lon)
except Exception:
    # OPTION B: fallback (either defaults or hardcode decimals once you know them)
    lat, lon = DEFAULT_LAT, DEFAULT_LON
    st.sidebar.warning("pyproj not installed — using default lat/lon.")

st.sidebar.header("Settings")
st.sidebar.caption(f"Center: lat={lat:.5f}, lon={lon:.5f} (from UTM 19S)")

# Use st.* inside the context, not st.sidebar.*
with st.sidebar:
     st.subheader("Forecast horizon")
     days_ahead = st.slider("Days ahead", min_value=3, max_value=14, value=7)

     st.subheader("Map sampling radius (km)")
     radius_km = st.slider("Radius around point (km)", min_value=5, max_value=50, value=20, step=5)
     grid_step_km = st.slider("Grid step (km)", min_value=2, max_value=10, value=5, step=1)

# ---------------------------
# Fetch & prepare
# ---------------------------

with st.spinner("Fetching weather data…"):
    meteo = fetch_open_meteo(lat, lon, days_ahead=days_ahead)

hourly = meteo["hourly"]
daily = meteo["daily"]

# compute days without rain on daily
_daily = compute_days_without_rain(daily, rain_threshold_mm=2.0)

# derive best-hour-per-day hazard variables from hourly
# build days_no_rain mapping first
days_nr_map = dict(zip(_daily["date"], _daily["days_no_rain"]))

haz = best_hour_by_day(hourly, days_nr_map)

# Today-focused default selection
dates_sorted = haz["date"].tolist()
sel_idx = int(np.clip(np.searchsorted(dates_sorted, TODAY.date()), 0, len(dates_sorted)-1))

# Today-focused default selection
dates_sorted = haz["date"].tolist()
sel_idx = int(np.clip(np.searchsorted(dates_sorted, TODAY.date()), 0, len(dates_sorted)-1))

st.title("🔥 Fire-Risk Dashboard — Bosque Pehuén")
st.caption(f"Auto-updated • Source: Open-Meteo • Timezone: {TZ}")

# ---------------------------
# Top layout: Polar (left) + Score cards & Table (right)
# ---------------------------
colA, colB = st.columns([2, 1], gap="large")

with colA:
    st.subheader("Daily Polar — Variables & Risk")
day_idx = st.slider("Select day", min_value=0, max_value=len(dates_sorted)-1, value=sel_idx, format="%d")
sel_date = dates_sorted[day_idx]

row = haz.loc[haz["date"] == sel_date].iloc[0]
# Polar axes: Temperature, Humidity, Wind, Days since rain
axes = ["Temperature", "Humidity", "Wind", "Days w/o rain"]
values_real = [row["temp_c"], row["rh_pct"], row["wind_kmh"], row["days_no_rain"]]

# normalize with simple caps for visual balance
caps = [40, 100, 80, 45]
scores = [row["temp"], row["rh"], row["wind"], row["days"]]
values_norm = [min(max(s / 25.0, 0.0), 1.0) for s in scores]

theta = np.linspace(0, 2 * np.pi, num=len(axes)+1)
r = np.array(values_norm + [values_norm[0]])

# Plotly Polar (IRM discrete color mapping)
risk_color = color_for_risk(float(row["total"]))

# convert hex → rgba with ~50 % opacity
def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

fill_color = hex_to_rgba(risk_color, 0.5)

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=r,
    theta=[a for a in axes] + [axes[0]],
    fill="toself",
    name="Risk contribution (normalized to max 25)",
    line=dict(color=risk_color, width=3),
    fillcolor=fill_color,
))

fig.update_polars(
    radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#dddddd", gridwidth=0.5),
    angularaxis=dict(tickfont=dict(size=12), gridcolor="#eeeeee", gridwidth=0.5),
)

fig.update_layout(
    height=520,
    margin=dict(l=30, r=30, t=30, b=30),
    polar_bgcolor="#fafafa",
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

st.caption(f"Most dangerous hour for {sel_date}: {pd.to_datetime(row['timestamp']).strftime('%H:%M')} (local)")

with colB:
    total = float(row["total"]) if not pd.isna(row["total"]) else 0.0
    col = color_for_risk(total)
    st.subheader("Risk index")
    st.metric(label=str(sel_date), value=f"{total:0.0f} / 100")
    st.markdown(f"<div style='height:12px;width:100%;background:{col};border-radius:6px;'></div>", unsafe_allow_html=True)

    st.write("\n")
    st.subheader("Daily values (scored)")
    # Show table with real values and sub-scores
    tbl = pd.DataFrame({
    "Variable": ["Temperature (°C)", "Humidity (%)", "Wind speed (km/h)", "Days w/o rain"],
    "Value": [row["temp_c"], row["rh_pct"], row["wind_kmh"], row["days_no_rain"]],
    "Score (0–25)": [row["temp"], row["rh"], row["wind"], row["days"]],
    })
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ---------------------------
# Forecast tab: multi-day outlook
# ---------------------------
with st.expander("Forecast — next days"):
    # --- Risk bar chart (sorted by date, consistent discrete color mapping) ---
    haz_sorted = haz.sort_values("date")

    f1 = go.Figure()
    f1.add_trace(go.Bar(
        x=haz_sorted["date"],
        y=haz_sorted["total"],
        name="Risk (0–100)",
        marker_color=[color_for_risk(x) for x in haz_sorted["total"]],
    ))
    f1.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=30, b=30),
        yaxis_title="Risk (0–100)",
    )

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=haz["date"], y=haz["temp_c"], name="T max (\u00B0C)"))
    f2.add_trace(go.Scatter(x=haz["date"], y=haz["rh_pct"], name="RH min (%)"))
    f2.add_trace(go.Scatter(x=haz["date"], y=haz["wind_kmh"], name="Wind max (km/h)"))
    f2.update_layout(height=260, margin=dict(l=30, r=30, t=30, b=30), yaxis_title="Value")

    st.plotly_chart(f1, use_container_width=True)
    st.plotly_chart(f2, use_container_width=True)

# ---------------------------
# Regional map: compute risk on a coarse grid for the selected day
# ---------------------------
# ---------------------------
# Regional map: optional "risk grid" overlay (hexagonal)
# ---------------------------
st.subheader("Regional risk map")

show_overlay = st.toggle("Show risk grid overlay", value=False)

# Build coordinate grid
def km_to_deg_lat(km: float) -> float:
    return km / 110.574

def km_to_deg_lon(km: float, at_lat: float) -> float:
    import math
    return km / (111.320 * math.cos(math.radians(abs(at_lat))))

lat_delta = km_to_deg_lat(radius_km)
lon_delta = km_to_deg_lon(radius_km, lat)

n_steps = max(2, int((radius_km * 2) / grid_step_km))
lat_steps = np.linspace(lat - lat_delta, lat + lat_delta, n_steps)
lon_steps = np.linspace(lon - lon_delta, lon + lon_delta, n_steps)

points = [(float(la), float(lo)) for la in lat_steps for lo in lon_steps]

@st.cache_data(ttl=1800)
def grid_forecast(points, target_date: dt.date) -> pd.DataFrame:
    rows = []
    for la, lo in points:
        try:
            dct = fetch_open_meteo(la, lo, days_ahead=days_ahead)
            h = dct["hourly"]
            h_day = h[h["date"] == target_date].copy()
            if h_day.empty:
                continue

            # Single conversion
            h_day["wind_kmh"] = pd.to_numeric(h_day["wind_ms"], errors="coerce") * 3.6

            dd = dct["daily"]
            dd2 = compute_days_without_rain(dd, 2.0)
            rec = dd2.loc[dd2["date"] == target_date]
            if rec.empty:
                continue
            days_nr = int(rec["days_no_rain"].iloc[0])

            comps = h_day.apply(
                lambda r: risk_components(
                    float(r["temp_c"]),
                    float(r["rh_pct"]),
                    float(r["wind_kmh"]),
                    days_nr,
                ),
                axis=1,
            )
            comp_df = pd.DataFrame(list(comps))
            h2 = pd.concat([h_day.reset_index(drop=True), comp_df], axis=1)
            best = h2.loc[h2["total"].idxmax()]

            rows.append({
                "lat": float(la),
                "lon": float(lo),
                "risk": float(best["total"]),
                "timestamp": best["timestamp"],
                "temp_c": float(best["temp_c"]),
                "rh_pct": float(best["rh_pct"]),
                "wind_kmh": float(best["wind_kmh"]),
                "days_no_rain": days_nr,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

if show_overlay:
    with st.spinner("Computing regional risk grid..."):
        grid_df = grid_forecast(points, sel_date)

    if grid_df.empty:
        st.info("No grid data for the selected day.")
    else:
        grid_df["color_hex"] = grid_df["risk"].apply(color_for_risk)
        grid_df["timestamp"] = pd.to_datetime(grid_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

        # Convert hex colors to RGB lists
        def hex_to_rgb_list(hex_color: str):
            hex_color = hex_color.lstrip("#")
            return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

        grid_df["color"] = grid_df["color_hex"].apply(hex_to_rgb_list)

        layer = pdk.Layer(
            "HexagonLayer",
            data=grid_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            radius=5000,           # adjust for grid smoothness (~5 km cells)
            opacity=0.55,          # semi-transparent
            extruded=False,
            pickable=True,
        )

        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=9)

        tooltip = {
            "html": (
                "<b>Risk:</b> {risk}<br/>"
                "<b>Most dangerous hour:</b> {timestamp}<br/>"
                "<b>T:</b> {temp_c}&deg;C<br/>"
                "<b>RH:</b> {rh_pct}%<br/>"
                "<b>Wind:</b> {wind_kmh} km/h<br/>"
                "<b>Days w/o rain:</b> {days_no_rain}"
            ),
            "style": {"backgroundColor": "#222", "color": "white"},
        }

        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

else:
    st.caption("Overlay disabled — base map only (toggle above to view regional risk grid).")

# ---------------------------
# Data download
# ---------------------------
st.subheader("Download data")

csv = haz.copy()
csv["date"] = csv["date"].astype(str)
st.download_button("Download daily risk CSV", data=csv.to_csv(index=False), file_name="daily_risk.csv", mime="text/csv")

# ---------------------------
# Footer
# ---------------------------
st.caption("Risk scoring: FMA / Bosque Pehuén scheme (T(\u00B0C), RH, Wind, Days no rain → 0–100). This dashboard is for situational awareness — not a substitute for official warnings.")
