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

import requests # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import streamlit as st # type: ignore
import plotly.graph_objects as go # type: ignore
import pydeck as pdk # type: ignore

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
    import pyproj  # type: ignore # pip install pyproj
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
     st.subheader("Data Fetching")
     days_ahead = st.slider("Forecast days to fetch", min_value=7, max_value=14, value=14, 
                           help="Number of forecast days to retrieve from API. Higher values provide more forecast data but may take longer to load.")
     
     st.subheader("Map Settings")
     st.caption("Map covers entire Araucania region")
     st.info("💡 Toggle 'Show risk grid overlay' on the map to view risk hexagons and wind currents")

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

# Initialize session state for date synchronization
dates_sorted = haz["date"].tolist()
if "selected_date" not in st.session_state:
    today_idx = int(np.clip(np.searchsorted(dates_sorted, TODAY.date()), 0, len(dates_sorted)-1))
    st.session_state.selected_date = dates_sorted[today_idx] if today_idx < len(dates_sorted) else dates_sorted[-1]

st.title("🔥 Fire-Risk Dashboard — Bosque Pehuén")
st.caption(f"Auto-updated • Source: Open-Meteo • Timezone: {TZ}")

# ---------------------------
# Top layout: Polar (left) + Score cards & Table (right)
# ---------------------------
colA, colB = st.columns([2, 1], gap="large")

with colA:
    st.subheader("Daily Polar — Variables & Risk")
    
    # Date selector with quick buttons
    col_date1, col_date2, col_date3, col_date4 = st.columns(4)
    
    with col_date1:
        if st.button("Today", use_container_width=True):
            today_idx = int(np.clip(np.searchsorted(dates_sorted, TODAY.date()), 0, len(dates_sorted)-1))
            if today_idx < len(dates_sorted):
                st.session_state.selected_date = dates_sorted[today_idx]
                st.rerun()
    
    with col_date2:
        if st.button("Yesterday", use_container_width=True):
            yesterday = (TODAY - pd.Timedelta(days=1)).date()
            yest_idx = int(np.clip(np.searchsorted(dates_sorted, yesterday), 0, len(dates_sorted)-1))
            if yest_idx < len(dates_sorted):
                # Find closest date to yesterday
                if dates_sorted[yest_idx] == yesterday:
                    st.session_state.selected_date = dates_sorted[yest_idx]
                elif yest_idx > 0 and abs((dates_sorted[yest_idx-1] - yesterday).days) < abs((dates_sorted[yest_idx] - yesterday).days):
                    st.session_state.selected_date = dates_sorted[yest_idx-1]
                else:
                    st.session_state.selected_date = dates_sorted[yest_idx]
                st.rerun()
    
    with col_date3:
        if st.button("Last Week", use_container_width=True):
            last_week = (TODAY - pd.Timedelta(days=7)).date()
            lw_idx = int(np.clip(np.searchsorted(dates_sorted, last_week), 0, len(dates_sorted)-1))
            if lw_idx < len(dates_sorted):
                st.session_state.selected_date = dates_sorted[lw_idx]
                st.rerun()
    
    with col_date4:
        if st.button("Next 7 Days", use_container_width=True):
            next_week = (TODAY + pd.Timedelta(days=7)).date()
            nw_idx = int(np.clip(np.searchsorted(dates_sorted, next_week), 0, len(dates_sorted)-1))
            if nw_idx < len(dates_sorted):
                st.session_state.selected_date = dates_sorted[nw_idx]
                st.rerun()
    
    # Date picker - centered and shorter
    col_picker1, col_picker2, col_picker3 = st.columns([1, 3, 1])
    with col_picker2:
        min_date = dates_sorted[0] if dates_sorted else TODAY.date()
        max_date = dates_sorted[-1] if dates_sorted else TODAY.date()
        
        selected_date_input = st.date_input(
            "Select date",
            value=st.session_state.selected_date if st.session_state.selected_date in dates_sorted else dates_sorted[0] if dates_sorted else TODAY.date(),
            min_value=min_date,
            max_value=max_date,
            key="date_picker"
        )
    
    # Update session state if date picker changed
    if selected_date_input in dates_sorted:
        st.session_state.selected_date = selected_date_input
        sel_date = selected_date_input
    else:
        # Find closest date
        closest_idx = int(np.clip(np.searchsorted(dates_sorted, selected_date_input), 0, len(dates_sorted)-1))
        if closest_idx < len(dates_sorted):
            st.session_state.selected_date = dates_sorted[closest_idx]
            sel_date = dates_sorted[closest_idx]
        else:
            sel_date = st.session_state.selected_date

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

st.plotly_chart(fig, width='stretch')

# Show forecast/historical indicator
is_forecast = sel_date > TODAY.date()
indicator_text = "🔮 Forecast" if is_forecast else "📊 Historical"
indicator_color = "#2196F3" if is_forecast else "#757575"
st.markdown(f"<div style='text-align:center;padding:8px;background:{indicator_color}20;border-radius:6px;color:{indicator_color};font-weight:bold;'>{indicator_text}</div>", unsafe_allow_html=True)

st.caption(f"Most dangerous hour for {sel_date}: {pd.to_datetime(row['timestamp']).strftime('%H:%M')} (local)")

with colB:
    total = float(row["total"]) if not pd.isna(row["total"]) else 0.0
    col = color_for_risk(total)
    st.subheader("Risk index")
    st.metric(label=str(sel_date), value=f"{total:0.0f} / 100")
    st.markdown(f"<div style='height:12px;width:100%;background:{col};border-radius:6px;'></div>", unsafe_allow_html=True)
    
    # Wind direction compass
    st.write("\n")
    st.subheader("Wind direction")
    
    # Get wind direction for selected date (from hourly data)
    hourly_sel = hourly[hourly["date"] == sel_date]
    if not hourly_sel.empty:
        # Get wind direction from the same hour window (14-16)
        hourly_sel_copy = hourly_sel.copy()
        hourly_sel_copy["hour"] = pd.to_datetime(hourly_sel_copy["timestamp"]).dt.hour
        wind_window = hourly_sel_copy[(hourly_sel_copy["hour"] >= 14) & (hourly_sel_copy["hour"] <= 16)]
        
        if not wind_window.empty:
            avg_wind_dir = float(wind_window["wind_dir"].mean())
            avg_wind_speed = float(wind_window["wind_kmh"].mean())
            
            # Create compass visualization using polar plot
            compass_fig = go.Figure()
            
            # Draw compass circle (outer ring)
            angles_circle = np.linspace(0, 360, 100)
            compass_fig.add_trace(go.Scatterpolar(
                r=[1]*100,
                theta=angles_circle,
                mode='lines',
                line=dict(color='#333', width=2),
                showlegend=False,
                hoverinfo='skip',
                fill='none'
            ))
            
            # Draw inner grid circles
            for r_val in [0.25, 0.5, 0.75]:
                compass_fig.add_trace(go.Scatterpolar(
                    r=[r_val]*100,
                    theta=angles_circle,
                    mode='lines',
                    line=dict(color='#ddd', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip',
                    fill='none'
                ))
            
            # Draw cardinal direction markers
            for direction, angle in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
                # Outer marker
                compass_fig.add_trace(go.Scatterpolar(
                    r=[0.9, 1.0],
                    theta=[angle, angle],
                    mode='lines',
                    line=dict(color='#666', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                # Label (using text annotation on regular plot)
                compass_fig.add_annotation(
                    text=direction,
                    x=0.5 + 0.45 * np.cos(np.radians(angle)),
                    y=0.5 + 0.45 * np.sin(np.radians(angle)),
                    showarrow=False,
                    font=dict(size=16, color='#333', family="Arial Black"),
                    xref="paper",
                    yref="paper"
                )
            
            # Draw wind arrow (wind direction is where wind comes FROM, so arrow points opposite)
            # For display, we show where wind is going (opposite of wind_dir)
            arrow_angle_deg = (avg_wind_dir + 180) % 360  # Reverse direction
            arrow_length = 0.75
            
            # Main arrow line
            compass_fig.add_trace(go.Scatterpolar(
                r=[0.05, arrow_length],
                theta=[arrow_angle_deg, arrow_angle_deg],
                mode='lines',
                line=dict(color='#e53935', width=4),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Arrow head - create a small triangle
            arrow_head_angle = np.radians(arrow_angle_deg)
            head_size = 0.1
            # Three points for arrow head triangle
            head_center_r = arrow_length
            head_center_theta = arrow_angle_deg
            
            # Points of the triangle
            tip_r = arrow_length + head_size
            tip_theta = arrow_angle_deg
            
            left_r = arrow_length - head_size * 0.3
            left_theta = (arrow_angle_deg - 30) % 360
            
            right_r = arrow_length - head_size * 0.3
            right_theta = (arrow_angle_deg + 30) % 360
            
            # Draw arrow head as filled shape
            compass_fig.add_trace(go.Scatterpolar(
                r=[left_r, tip_r, right_r, left_r],
                theta=[left_theta, tip_theta, right_theta, left_theta],
                mode='lines',
                fill='toself',
                fillcolor='#e53935',
                line=dict(color='#e53935', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            compass_fig.update_polars(
                radialaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, showline=False),
                angularaxis=dict(
                    showticklabels=True,
                    tickmode='array',
                    tickvals=[0, 90, 180, 270],
                    ticktext=['N', 'E', 'S', 'W'],
                    tickfont=dict(size=12),
                    showgrid=True,
                    gridcolor='#ddd',
                    gridwidth=1,
                    rotation=90,
                    direction='counterclockwise'
                )
            )
            
            compass_fig.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                polar_bgcolor="#fafafa",
                showlegend=False,
            )
            
            st.plotly_chart(compass_fig, use_container_width=True, config={'displayModeBar': False})
            st.caption(f"{avg_wind_dir:.0f}° ({avg_wind_speed:.1f} km/h)")
        else:
            st.info("Wind data not available for this date")
    else:
        st.info("No hourly data for this date")

    st.write("\n")
    st.subheader("Daily values (scored)")
    # Show table with real values and sub-scores
    tbl = pd.DataFrame({
    "Variable": ["Temperature (°C)", "Humidity (%)", "Wind speed (km/h)", "Days w/o rain"],
    "Value": [row["temp_c"], row["rh_pct"], row["wind_kmh"], row["days_no_rain"]],
    "Score (0–25)": [row["temp"], row["rh"], row["wind"], row["days"]],
    })
st.dataframe(tbl, width='stretch', hide_index=True)

# ---------------------------
# Forecast tab: multi-day outlook
# ---------------------------
st.subheader("Forecast — Risk Evolution")

# Initialize forecast view state
if "forecast_expanded" not in st.session_state:
    st.session_state.forecast_expanded = False

# Forecast view toggle
col_fc1, col_fc2 = st.columns([1, 10])
with col_fc1:
    if st.button("Expand to 2 weeks" if not st.session_state.forecast_expanded else "Show 7+7 days"):
        st.session_state.forecast_expanded = not st.session_state.forecast_expanded
        st.rerun()

# Filter data based on view
haz_sorted = haz.sort_values("date").copy()
today_date = TODAY.date()

if st.session_state.forecast_expanded:
    # Last 7 days + next 14 days
    start_date = today_date - dt.timedelta(days=7)
    end_date = today_date + dt.timedelta(days=14)
    view_label = "Last 7 days + Next 14 days"
else:
    # Last 7 days + next 7 days (default)
    start_date = today_date - dt.timedelta(days=7)
    end_date = today_date + dt.timedelta(days=7)
    view_label = "Last 7 days + Next 7 days"

# Filter by date (ensure date column is date type for comparison)
haz_filtered = haz_sorted[
    (pd.to_datetime(haz_sorted["date"]).dt.date >= start_date) & 
    (pd.to_datetime(haz_sorted["date"]).dt.date <= end_date)
].copy()

if not haz_filtered.empty:
    st.caption(view_label)
    
    # Always show charts (removed expander)
    # --- Risk bar chart (sorted by date, consistent discrete color mapping) ---
    # Convert dates to datetime for proper x-axis handling
    haz_dates_dt = pd.to_datetime(haz_filtered["date"])
    
    f1 = go.Figure()
    f1.add_trace(go.Bar(
        x=haz_dates_dt,
        y=haz_filtered["total"],
        name="Risk (0–100)",
        marker_color=[color_for_risk(x) for x in haz_filtered["total"]],
    ))
    
    # Add vertical line for today
    # Check if today is in the filtered dates
    haz_dates = haz_dates_dt.dt.date
    if (haz_dates == today_date).any():
        # Use add_shape instead of add_vline to avoid Timestamp arithmetic issues
        today_datetime = pd.Timestamp(today_date)
        f1.add_shape(
            type="line",
            x0=today_datetime,
            x1=today_datetime,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
        )
        # Add annotation for "Today" label
        f1.add_annotation(
            x=today_datetime,
            y=1,
            yref="paper",
            text="Today",
            showarrow=False,
            font=dict(color="red"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1,
            xanchor="center",
            yanchor="bottom"
        )
    
    f1.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=30, b=30),
        yaxis_title="Risk (0–100)",
        xaxis_title="Date",
    )

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=haz_dates_dt, y=haz_filtered["temp_c"], name="T max (\u00B0C)", mode='lines+markers'))
    f2.add_trace(go.Scatter(x=haz_dates_dt, y=haz_filtered["rh_pct"], name="RH min (%)", mode='lines+markers'))
    f2.add_trace(go.Scatter(x=haz_dates_dt, y=haz_filtered["wind_kmh"], name="Wind max (km/h)", mode='lines+markers'))
    f2.add_trace(go.Scatter(x=haz_dates_dt, y=haz_filtered["days_no_rain"], name="Days without rain", mode='lines+markers'))
    
    # Add vertical line for today
    # Check if today is in the filtered dates
    if (haz_dates == today_date).any():
        # Use add_shape instead of add_vline to avoid Timestamp arithmetic issues
        today_datetime = pd.Timestamp(today_date)
        f2.add_shape(
            type="line",
            x0=today_datetime,
            x1=today_datetime,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
        )
        # Add annotation for "Today" label
        f2.add_annotation(
            x=today_datetime,
            y=1,
            yref="paper",
            text="Today",
            showarrow=False,
            font=dict(color="red"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1,
            xanchor="center",
            yanchor="bottom"
        )
    
    f2.update_layout(height=260, margin=dict(l=30, r=30, t=30, b=30), yaxis_title="Value", xaxis_title="Date")

    st.plotly_chart(f1, width='stretch')
    st.plotly_chart(f2, width='stretch')
else:
    st.info("No forecast data available for the selected range.")

# ---------------------------
# Regional map: compute risk on a coarse grid for the selected day
# ---------------------------
st.subheader("Regional risk map")
st.caption(f"Showing risk for: {st.session_state.selected_date}")

# Toggle for overlay (map always visible)
col_map1, col_map2 = st.columns([1, 10])
with col_map1:
    show_overlay = st.toggle("Show risk grid overlay", value=False)

# Build coordinate grid for Araucania region (~38-40°S, 71-73°W)
def km_to_deg_lat(km: float) -> float:
    return km / 110.574

def km_to_deg_lon(km: float, at_lat: float) -> float:
    import math
    return km / (111.320 * math.cos(math.radians(abs(at_lat))))

# Araucania region bounds
araucania_lat_min = -40.0
araucania_lat_max = -38.0
araucania_lon_min = -73.0
araucania_lon_max = -71.0

# Use 0.1 degree steps to match Open-Meteo granularity (~11 km per cell)
step_deg = 0.1

lat_steps = np.arange(araucania_lat_min, araucania_lat_max + step_deg, step_deg)
lon_steps = np.arange(araucania_lon_min, araucania_lon_max + step_deg, step_deg)

points = [(float(la), float(lo)) for la in lat_steps for lo in lon_steps]

@st.cache_data(ttl=1800)
def grid_forecast(points, target_date: dt.date, days_ahead: int) -> pd.DataFrame:
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

            # Get wind direction from best row (h2 includes all columns from h_day including wind_dir)
            wind_dir_val = float(best.get("wind_dir", 0.0)) if "wind_dir" in best.index else 0.0
            
            rows.append({
                "lat": float(la),
                "lon": float(lo),
                "risk": float(best["total"]),
                "timestamp": best["timestamp"],
                "temp_c": float(best["temp_c"]),
                "rh_pct": float(best["rh_pct"]),
                "wind_kmh": float(best["wind_kmh"]),
                "wind_dir": wind_dir_val,
                "days_no_rain": days_nr,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

# Always show base map, overlay is optional
layers = []

# Base map is always visible (pydeck shows base map by default)

# Add risk grid overlay if toggled
if show_overlay:
    with st.spinner("Computing regional risk grid..."):
        grid_df = grid_forecast(points, st.session_state.selected_date, days_ahead)

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

        # Risk hexagon layer
        risk_layer = pdk.Layer(
            "HexagonLayer",
            data=grid_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            radius=5500,
            opacity=0.25,
            extruded=False,
            pickable=True,
        )
        layers.append(risk_layer)
        
        # Wind currents layer (arrows showing wind direction)
        # Sample every 3rd point to avoid overcrowding
        wind_df = grid_df.iloc[::3].copy()
        
        # Create wind vectors (wind direction is where wind comes FROM)
        # For visualization, we'll show arrows pointing in the direction wind is going
        wind_vectors = []
        for idx, row in wind_df.iterrows():
            wind_dir_rad = np.radians(row["wind_dir"] + 180)  # Reverse direction
            wind_speed = row["wind_kmh"]
            # Scale arrow length based on wind speed (max ~50 km/h = 0.05 degrees)
            arrow_length_deg = min(wind_speed / 1000.0, 0.05)
            
            # Calculate end point
            end_lat = row["lat"] + arrow_length_deg * np.cos(wind_dir_rad)
            end_lon = row["lon"] + arrow_length_deg * np.sin(wind_dir_rad) / np.cos(np.radians(row["lat"]))
            
            wind_vectors.append({
                "start_lat": row["lat"],
                "start_lon": row["lon"],
                "end_lat": end_lat,
                "end_lon": end_lon,
                "wind_speed": wind_speed,
                "wind_dir": row["wind_dir"]
            })
        
        if wind_vectors:
            wind_df_vectors = pd.DataFrame(wind_vectors)
            
            # Create line layer for wind vectors
            wind_layer = pdk.Layer(
                "LineLayer",
                data=wind_df_vectors,
                get_source_position="[start_lon, start_lat]",
                get_target_position="[end_lon, end_lat]",
                get_color=[255, 100, 100, 180],  # Red with transparency
                get_width=2,
                pickable=True,
            )
            layers.append(wind_layer)

# Bosque Pehuen highlight (always visible)
# Create circle around Bosque Pehuen
circle_points = []
num_points = 64
for i in range(num_points + 1):
    angle = 2 * np.pi * i / num_points
    # Circle radius ~5km
    radius_deg = 5.0 / 111.0  # ~5km in degrees
    circle_lat = lat + radius_deg * np.cos(angle)
    circle_lon = lon + radius_deg * np.sin(angle) / np.cos(np.radians(lat))
    circle_points.append([circle_lon, circle_lat])

# Close the circle
circle_points.append(circle_points[0])

bosque_highlight = pdk.Layer(
    "PolygonLayer",
    data=[{"coordinates": [circle_points]}],
    get_polygon="coordinates",
    get_fill_color=[255, 215, 0, 60],  # Gold with transparency
    get_line_color=[255, 140, 0, 255],  # Orange border
    get_line_width=3,
    pickable=False,
)
layers.append(bosque_highlight)

# Marker for Bosque Pehuen center
bosque_marker = pdk.Layer(
    "ScatterplotLayer",
    data=[{"lon": lon, "lat": lat, "name": "Bosque Pehuén"}],
    get_position="[lon, lat]",
    get_color=[255, 140, 0, 255],  # Orange
    get_radius=200,
    pickable=True,
)
layers.append(bosque_marker)

# View state centered on Araucania region
center_lat = (araucania_lat_min + araucania_lat_max) / 2
center_lon = (araucania_lon_min + araucania_lon_max) / 2

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=8,
    pitch=0,
    bearing=0
)

tooltip = {
    "html": (
        "<b>Risk:</b> {risk}<br>"
        "<b>T:</b> {temp_c}°C<br>"
        "<b>RH:</b> {rh_pct}%<br>"
        "<b>Wind:</b> {wind_kmh} km/h<br>"
        "<b>Days no rain:</b> {days_no_rain}"
    ),
    "style": {"backgroundColor": "#222", "color": "white"},
}

# Render map (always visible, layers added conditionally)
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v9"))

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
