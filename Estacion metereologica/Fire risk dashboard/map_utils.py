# -*- coding: utf-8 -*-
"""
Map utility functions for regional risk visualization
"""

import math
import datetime as dt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from data_fetcher import fetch_open_meteo
from risk_calculator import risk_components, compute_days_without_rain, color_for_risk
from config import ARAUCANIA_LAT_MIN, ARAUCANIA_LAT_MAX, ARAUCANIA_LON_MIN, ARAUCANIA_LON_MAX, MAP_GRID_STEP_DEG, RISK_COLORS


def km_to_deg_lat(km: float) -> float:
    """Convert kilometers to degrees latitude."""
    return km / 110.574


def km_to_deg_lon(km: float, at_lat: float) -> float:
    """Convert kilometers to degrees longitude."""
    return km / (111.320 * math.cos(math.radians(abs(at_lat))))


def create_araucania_grid() -> list:
    """Create coordinate grid for Araucania region."""
    lat_steps = np.arange(ARAUCANIA_LAT_MIN, ARAUCANIA_LAT_MAX + MAP_GRID_STEP_DEG, MAP_GRID_STEP_DEG)
    lon_steps = np.arange(ARAUCANIA_LON_MIN, ARAUCANIA_LON_MAX + MAP_GRID_STEP_DEG, MAP_GRID_STEP_DEG)
    return [(float(la), float(lo)) for la in lat_steps for lo in lon_steps]


@st.cache_data(ttl=1800)
def grid_forecast(points: list, target_date: dt.date, days_ahead: int) -> pd.DataFrame:
    """Compute risk forecast for a grid of points."""
    rows = []
    for la, lo in points:
        try:
            dct = fetch_open_meteo(la, lo, days_ahead=days_ahead)
            h = dct["hourly"]
            h_day = h[h["date"] == target_date].copy()
            if h_day.empty:
                continue

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


def hex_to_rgb_list(hex_color: str) -> list:
    """Convert hex color to RGB list."""
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]


def create_risk_color_scale() -> list:
    """Create color scale for heatmap from RISK_COLORS."""
    # Extract colors from RISK_COLORS and convert to RGB tuples
    color_scale = []
    for lo, hi, hex_col in sorted(RISK_COLORS, key=lambda x: x[0]):
        rgb = hex_to_rgb_list(hex_col)
        color_scale.append(rgb)
    return color_scale


def create_map_layers(grid_df: pd.DataFrame, lat: float, lon: float) -> list:
    """Create map layers for risk visualization."""
    layers = []
    
    if not grid_df.empty:
        # Risk visualization using ScatterplotLayer with individual colors per point
        # Add color column based on actual risk values - each point gets its own color
        grid_df = grid_df.copy()
        grid_df["color_hex"] = grid_df["risk"].apply(color_for_risk)
        grid_df["color_rgb"] = grid_df["color_hex"].apply(lambda x: hex_to_rgb_list(x) + [180])  # RGBA with alpha
        
        # ScatterplotLayer with large radius for heatmap-like appearance
        # Using radius in meters for zoom-independent sizing (~20km per point for good overlap)
        radius_meters = 20000
        
        risk_layer = pdk.Layer(
            "ScatterplotLayer",
            data=grid_df,
            get_position="[lon, lat]",
            get_fill_color="color_rgb",
            get_radius=radius_meters,
            radius_scale=1,
            radius_min_pixels=3,
            radius_max_pixels=200,
            line_width_min_pixels=0,
            opacity=0.6,
            pickable=True,
            auto_highlight=True,
        )
        layers.append(risk_layer)

    # Bosque Pehuen highlight (always visible)
    circle_points = []
    num_points = 64
    for i in range(num_points + 1):
        angle = 2 * np.pi * i / num_points
        radius_deg = 5.0 / 111.0
        circle_lat = lat + radius_deg * np.cos(angle)
        circle_lon = lon + radius_deg * np.sin(angle) / np.cos(np.radians(lat))
        circle_points.append([circle_lon, circle_lat])
    circle_points.append(circle_points[0])

    bosque_highlight = pdk.Layer(
        "PolygonLayer",
        data=[{"coordinates": [circle_points]}],
        get_polygon="coordinates",
        get_fill_color=[255, 215, 0, 60],
        get_line_color=[255, 140, 0, 255],
        get_line_width=3,
        pickable=False,
    )
    layers.append(bosque_highlight)

    bosque_marker = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lon": lon, "lat": lat, "name": "Bosque Pehuén"}],
        get_position="[lon, lat]",
        get_color=[255, 140, 0, 255],
        get_radius=200,
        pickable=True,
    )
    layers.append(bosque_marker)

    return layers


def create_map_view_state() -> pdk.ViewState:
    """Create view state centered on Araucania region."""
    center_lat = (ARAUCANIA_LAT_MIN + ARAUCANIA_LAT_MAX) / 2
    center_lon = (ARAUCANIA_LON_MIN + ARAUCANIA_LON_MAX) / 2
    
    return pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=8,
        pitch=0,
        bearing=0
    )


