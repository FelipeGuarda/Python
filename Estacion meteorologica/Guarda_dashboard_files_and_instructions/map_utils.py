# -*- coding: utf-8 -*-
"""
Map utility functions for regional risk visualization
"""

import numpy as np
import pandas as pd
import pydeck as pdk

from config import ARAUCANIA_LAT_MIN, ARAUCANIA_LAT_MAX, ARAUCANIA_LON_MIN, ARAUCANIA_LON_MAX


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
    # More vertical lines, extended horizontal coverage to include ocean on the west
    num_streaks_lat = 11  # Vertical distribution (11 latitude lines)
    num_streaks_lon = 6   # Horizontal distribution (6 streaks per line, extended to west)
    # Total: 66 streaks evenly distributed
    
    # Distribute evenly across the region
    lat_positions = np.linspace(ARAUCANIA_LAT_MIN + 0.2, ARAUCANIA_LAT_MAX - 0.2, num_streaks_lat)
    # Extend west from ocean (-73.0) to east edge (-71.0)
    # Start at -73.0 (western ocean edge) and extend across land mass
    lon_positions = np.linspace(-73.0, -71.0, num_streaks_lon)
    
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


def create_map_view_state(center_lat: float = None, center_lon: float = None) -> pdk.ViewState:
    """Create view state centered on specified location or Araucania region."""
    # Use provided coordinates or default to region center
    if center_lat is None or center_lon is None:
        center_lat = (ARAUCANIA_LAT_MIN + ARAUCANIA_LAT_MAX) / 2
        center_lon = (ARAUCANIA_LON_MIN + ARAUCANIA_LON_MAX) / 2
    
    return pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=8.5,     # Default zoom - balanced view showing Bosque Pehuén and wind patterns
        min_zoom=7,   # Limit zoom out (prevents seeing too much)
        max_zoom=12,  # Limit zoom in (keeps context visible)
        pitch=0,
        bearing=0
    )


