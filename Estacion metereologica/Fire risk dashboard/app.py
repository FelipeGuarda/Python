# -*- coding: utf-8 -*-
"""
Fire-Risk Dashboard — Main Application
Stack: Streamlit + Plotly + pydeck + Open-Meteo

How to run
    pip install -r requirements.txt
    streamlit run app.py

Optional (for UTM conversion): pip install pyproj
"""

from __future__ import annotations
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st  # pyright: ignore[reportMissingImports]
import pydeck as pdk  # pyright: ignore[reportMissingImports]

# Local imports
from config import TZ, TODAY, DEFAULT_LAT, DEFAULT_LON
from data_fetcher import fetch_open_meteo
from risk_calculator import compute_days_without_rain, best_hour_by_day, color_for_risk
from visualizations import create_polar_plot, create_wind_compass, create_forecast_charts
from map_utils import create_araucania_grid, grid_forecast, create_map_layers, create_map_view_state

# Page config
st.set_page_config(
    page_title="Fire Risk Dashboard — Bosque Pehuén",
    layout="wide",
)

# ---------------------------
# Sidebar controls
# ---------------------------
try:
    import pyproj  # pyright: ignore[reportMissingImports]
    transformer = pyproj.Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(263221.0, 5630634.0)
    lat, lon = float(lat), float(lon)
except Exception:
    lat, lon = DEFAULT_LAT, DEFAULT_LON
    st.sidebar.warning("pyproj not installed — using default lat/lon.")

st.sidebar.header("Settings")
st.sidebar.caption(f"Center: lat={lat:.5f}, lon={lon:.5f} (from UTM 19S)")

with st.sidebar:
    st.subheader("Data Fetching")
    days_ahead = st.slider("Forecast days to fetch", min_value=7, max_value=14, value=14, 
                          help="Number of forecast days to retrieve from API.")
    
    st.subheader("Map Settings")
    st.caption("Map covers entire Araucania region")
    st.info("Toggle 'Show risk grid overlay' on the map to view risk hexagons and wind currents")

# ---------------------------
# Fetch & prepare data
# ---------------------------
with st.spinner("Fetching weather data…"):
    meteo = fetch_open_meteo(lat, lon, days_ahead=days_ahead)

hourly = meteo["hourly"]
daily = meteo["daily"]

_daily = compute_days_without_rain(daily, rain_threshold_mm=2.0)
days_nr_map = dict(zip(_daily["date"], _daily["days_no_rain"]))
haz = best_hour_by_day(hourly, days_nr_map)

# Initialize session state
dates_sorted = haz["date"].tolist()
if "selected_date" not in st.session_state:
    today_idx = int(np.clip(np.searchsorted(dates_sorted, TODAY.date()), 0, len(dates_sorted)-1))
    st.session_state.selected_date = dates_sorted[today_idx] if today_idx < len(dates_sorted) else dates_sorted[-1]

st.title("Fire-Risk Dashboard — Bosque Pehuén")
st.caption(f"Auto-updated • Source: Open-Meteo • Timezone: {TZ}")

# ---------------------------
# Polar Plot Section
# ---------------------------
colA, colB = st.columns([2, 1], gap="large")

with colA:
    st.subheader("Daily Polar — Variables & Risk")
    
    # Date selector buttons
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
    
    # Date picker
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
    
    if selected_date_input in dates_sorted:
        st.session_state.selected_date = selected_date_input
        sel_date = selected_date_input
    else:
        closest_idx = int(np.clip(np.searchsorted(dates_sorted, selected_date_input), 0, len(dates_sorted)-1))
        if closest_idx < len(dates_sorted):
            st.session_state.selected_date = dates_sorted[closest_idx]
            sel_date = dates_sorted[closest_idx]
        else:
            sel_date = st.session_state.selected_date

row = haz.loc[haz["date"] == sel_date].iloc[0]

# Create and display polar plot
fig = create_polar_plot(row)
st.plotly_chart(fig, width='stretch')

# Forecast/historical indicator
is_forecast = sel_date > TODAY.date()
indicator_text = "Forecast" if is_forecast else "Historical"
indicator_color = "#2196F3" if is_forecast else "#757575"
st.markdown(f"<div style='text-align:center;padding:8px;background:{indicator_color}20;border-radius:6px;color:{indicator_color};font-weight:bold;'>{indicator_text}</div>", unsafe_allow_html=True)

st.caption(f"Most dangerous hour for {sel_date}: {pd.to_datetime(row['timestamp']).strftime('%H:%M')} (local)")

# Right column: Risk index and wind compass
with colB:
    total = float(row["total"]) if not pd.isna(row["total"]) else 0.0
    col = color_for_risk(total)
    st.subheader("Risk index")
    st.metric(label=str(sel_date), value=f"{total:0.0f} / 100")
    st.markdown(f"<div style='height:12px;width:100%;background:{col};border-radius:6px;'></div>", unsafe_allow_html=True)
    
    # Wind compass
    st.write("\n")
    st.subheader("Wind direction")
    
    hourly_sel = hourly[hourly["date"] == sel_date]
    if not hourly_sel.empty:
        hourly_sel_copy = hourly_sel.copy()
        hourly_sel_copy["hour"] = pd.to_datetime(hourly_sel_copy["timestamp"]).dt.hour
        wind_window = hourly_sel_copy[(hourly_sel_copy["hour"] >= 14) & (hourly_sel_copy["hour"] <= 16)]
        
        if not wind_window.empty:
            avg_wind_dir = float(wind_window["wind_dir"].mean())
            avg_wind_speed = float(wind_window["wind_kmh"].mean())
            
            compass_fig = create_wind_compass(avg_wind_dir, avg_wind_speed)
            st.plotly_chart(compass_fig, use_container_width=True, config={'displayModeBar': False})
            st.caption(f"{avg_wind_dir:.0f}° ({avg_wind_speed:.1f} km/h)")
        else:
            st.info("Wind data not available for this date")
    else:
        st.info("No hourly data for this date")

    st.write("\n")
    st.subheader("Daily values (scored)")
    tbl = pd.DataFrame({
        "Variable": ["Temperature (°C)", "Humidity (%)", "Wind speed (km/h)", "Days w/o rain"],
        "Value": [row["temp_c"], row["rh_pct"], row["wind_kmh"], row["days_no_rain"]],
        "Score (0–25)": [row["temp"], row["rh"], row["wind"], row["days"]],
    })
    st.dataframe(tbl, width='stretch', hide_index=True)

# ---------------------------
# Forecast Section
# ---------------------------
st.subheader("Forecast — Risk Evolution")

if "forecast_expanded" not in st.session_state:
    st.session_state.forecast_expanded = False

col_fc1, col_fc2 = st.columns([1, 10])
with col_fc1:
    if st.button("Expand to 2 weeks" if not st.session_state.forecast_expanded else "Show 7+7 days"):
        st.session_state.forecast_expanded = not st.session_state.forecast_expanded
        st.rerun()

haz_sorted = haz.sort_values("date").copy()
today_date = TODAY.date()

if st.session_state.forecast_expanded:
    start_date = today_date - dt.timedelta(days=7)
    end_date = today_date + dt.timedelta(days=14)
    view_label = "Last 7 days + Next 14 days"
else:
    start_date = today_date - dt.timedelta(days=7)
    end_date = today_date + dt.timedelta(days=7)
    view_label = "Last 7 days + Next 7 days"

haz_filtered = haz_sorted[
    (pd.to_datetime(haz_sorted["date"]).dt.date >= start_date) & 
    (pd.to_datetime(haz_sorted["date"]).dt.date <= end_date)
].copy()

if not haz_filtered.empty:
    st.caption(view_label)
    f1, f2 = create_forecast_charts(haz_filtered, today_date)
    st.plotly_chart(f1, width='stretch')
    st.plotly_chart(f2, width='stretch')
else:
    st.info("No forecast data available for the selected range.")

# ---------------------------
# Regional Map
# ---------------------------
st.subheader("Regional risk map")
st.caption(f"Showing risk for: {st.session_state.selected_date}")

col_map1, col_map2 = st.columns([1, 10])
with col_map1:
    show_overlay = st.toggle("Show risk grid overlay", value=False)

points = create_araucania_grid()
layers = []

if show_overlay:
    with st.spinner("Computing regional risk grid..."):
        grid_df = grid_forecast(points, st.session_state.selected_date, days_ahead)

    if grid_df.empty:
        st.info("No grid data for the selected day.")
    else:
        grid_df["timestamp"] = pd.to_datetime(grid_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        layers = create_map_layers(grid_df, lat, lon)
else:
    # Still show Bosque Pehuen highlight even without overlay
    layers = create_map_layers(pd.DataFrame(), lat, lon)

view_state = create_map_view_state()

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

st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip, map_style=pdk.map_styles.LIGHT))

# ---------------------------
# Data Download
# ---------------------------
st.subheader("Download data")
csv = haz.copy()
csv["date"] = csv["date"].astype(str)
st.download_button("Download daily risk CSV", data=csv.to_csv(index=False), file_name="daily_risk.csv", mime="text/csv")

# ---------------------------
# Footer
# ---------------------------
st.caption("Risk scoring: FMA / Bosque Pehuén scheme (T(°C), RH, Wind, Days no rain → 0–100). This dashboard is for situational awareness — not a substitute for official warnings.")
