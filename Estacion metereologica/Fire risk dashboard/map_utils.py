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
from scipy.interpolate import griddata

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


def interpolate_risk_grid(grid_df: pd.DataFrame, resolution: float = 0.1) -> pd.DataFrame:
    """
    Interpolate sparse risk grid to create smooth heatmap.
    
    Args:
        grid_df: DataFrame with lat, lon, risk columns
        resolution: Grid resolution in degrees for interpolated points
    
    Returns:
        DataFrame with interpolated points
    """
    if grid_df.empty:
        return grid_df
    
    # Original sparse points
    points = grid_df[['lon', 'lat']].values
    values = grid_df['risk'].values
    
    # Create dense interpolation grid
    lon_range = np.arange(ARAUCANIA_LON_MIN, ARAUCANIA_LON_MAX + resolution, resolution)
    lat_range = np.arange(ARAUCANIA_LAT_MIN, ARAUCANIA_LAT_MAX + resolution, resolution)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    
    # Interpolate using cubic method for smooth transitions
    risk_interpolated = griddata(points, values, (lon_grid, lat_grid), method='cubic', fill_value=0)
    
    # Clip negative values from cubic interpolation
    risk_interpolated = np.clip(risk_interpolated, 0, 100)
    
    # Flatten and create DataFrame
    interpolated_df = pd.DataFrame({
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten(),
        'risk': risk_interpolated.flatten()
    })
    
    # Remove points with very low risk (likely extrapolation artifacts)
    interpolated_df = interpolated_df[interpolated_df['risk'] > 1.0]
    
    return interpolated_df


def wind_speed_to_color(speed_kmh: float) -> list:
    """Convert wind speed to RGBA color."""
    if speed_kmh < 10:
        return [100, 200, 255, 180]  # Light cyan - calm
    elif speed_kmh < 20:
        return [200, 255, 100, 180]  # Lime - moderate
    elif speed_kmh < 30:
        return [255, 165, 0, 180]    # Orange - high
    else:
        return [255, 50, 50, 180]     # Red - extreme


def create_wind_flow_field(wind_data: dict) -> list:
    """
    Create flow field lines showing wind patterns with directional fading.
    
    Args:
        wind_data: Dict with 'wind_dir', 'wind_speed', 'lat', 'lon'
    
    Returns:
        List of pydeck layers for wind visualization
    """
    if not wind_data or 'wind_dir' not in wind_data:
        return []
    
    wind_dir = wind_data['wind_dir']
    wind_speed = wind_data['wind_speed']
    
    # Convert wind direction to radians (meteorological: direction wind comes FROM)
    # We want to show where it's blowing TO, so add 180°
    flow_direction_rad = np.radians((wind_dir + 180) % 360)
    
    # Create grid of starting points distributed across the entire region
    # Fewer streaks with more spacing for cleaner look
    num_streaks_lat = 4  # Vertical distribution
    num_streaks_lon = 6  # Horizontal distribution (reduced from 12)
    
    # Distribute evenly across the region with margins
    lat_positions = np.linspace(ARAUCANIA_LAT_MIN + 0.3, ARAUCANIA_LAT_MAX - 0.3, num_streaks_lat)
    lon_positions = np.linspace(ARAUCANIA_LON_MIN + 0.3, ARAUCANIA_LON_MAX - 0.3, num_streaks_lon)
    
    streamlines = []
    base_color = wind_speed_to_color(wind_speed)
    
    # Generate streamlines from grid points
    for start_lat in lat_positions:
        for start_lon in lon_positions:
            current_lat = start_lat
            current_lon = start_lon
            
            # Shorter streaks: 6-8 segments, smaller steps
            num_segments = 7
            step_size_deg = 0.04  # ~4km per step = ~28km total length
            
            for step in range(num_segments):
                # Calculate next point in wind direction
                dlat = step_size_deg * np.cos(flow_direction_rad)
                dlon = step_size_deg * np.sin(flow_direction_rad) / np.cos(np.radians(current_lat))
                
                next_lat = current_lat + dlat
                next_lon = current_lon + dlon
                
                # Create opacity gradient: fade from transparent to opaque (shows direction)
                # Start: low alpha (transparent), End: high alpha (opaque) - shows where wind is heading
                opacity_start = 20 + int((step / num_segments) * 180)
                opacity_end = 20 + int(((step + 1) / num_segments) * 180)
                
                # Store segment with gradient colors
                color_start = base_color[:3] + [opacity_start]
                color_end = base_color[:3] + [opacity_end]
                
                streamlines.append({
                    'start_lat': current_lat,
                    'start_lon': current_lon,
                    'end_lat': next_lat,
                    'end_lon': next_lon,
                    'color_start': color_start,
                    'color_end': color_end,
                    'wind_speed': wind_speed,
                })
                
                current_lat = next_lat
                current_lon = next_lon
    
    # Create DataFrame for pydeck
    if not streamlines:
        return []
    
    streamlines_df = pd.DataFrame(streamlines)
    
    # Create flow field layer with gradient effect
    # Use source color for start and target color for end
    flow_layer = pdk.Layer(
        "LineLayer",
        data=streamlines_df,
        get_source_position="[start_lon, start_lat]",
        get_target_position="[end_lon, end_lat]",
        get_color="color_start",
        get_width=2.5,
        width_min_pixels=1.5,
        width_max_pixels=3,
        pickable=False,
    )
    
    return [flow_layer]


def create_map_layers(wind_data: dict, lat: float, lon: float) -> list:
    """Create map layers with wind flow field visualization."""
    layers = []
    
    # Add wind flow field if wind data is available
    if wind_data:
        wind_layers = create_wind_flow_field(wind_data)
        layers.extend(wind_layers)

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

    # Add wind data to marker for tooltip
    marker_data = [{"lon": lon, "lat": lat, "name": "Bosque Pehuén"}]
    if wind_data:
        marker_data[0]["wind_speed"] = wind_data.get('wind_speed', 0)
        marker_data[0]["wind_dir"] = wind_data.get('wind_dir', 0)
    
    bosque_marker = pdk.Layer(
        "ScatterplotLayer",
        data=marker_data,
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


