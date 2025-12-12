# -*- coding: utf-8 -*-
"""
Visualization functions for charts and plots
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple

from risk_calculator import color_for_risk
from config import TODAY


def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
    """Convert hex color to rgba string."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def create_polar_plot(row: pd.Series) -> go.Figure:
    """Create polar plot for daily risk variables."""
    axes = ["Temperature", "Humidity", "Wind", "Days w/o rain"]
    scores = [row["temp"], row["rh"], row["wind"], row["days"]]
    values_norm = [min(max(s / 25.0, 0.0), 1.0) for s in scores]

    theta = np.linspace(0, 2 * np.pi, num=len(axes)+1)
    r = np.array(values_norm + [values_norm[0]])

    risk_color = color_for_risk(float(row["total"]))
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
    
    return fig


def create_wind_compass(avg_wind_dir: float, avg_wind_speed: float) -> go.Figure:
    """Create wind direction compass visualization."""
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
    
    # Draw cardinal direction markers (lines only, labels come from ticklabels)
    for direction, angle in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
        compass_fig.add_trace(go.Scatterpolar(
            r=[0.9, 1.0],
            theta=[angle, angle],
            mode='lines',
            line=dict(color='#666', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Draw wind arrow - using standard meteorological convention (wind coming FROM)
    # In Plotly polar plots, 0° is at top (North), angles increase counterclockwise
    arrow_angle_deg = (avg_wind_dir + 180) % 360  # Wind direction (where wind is coming from)
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
    
    # Arrow head triangle
    head_size = 0.1
    tip_r = arrow_length + head_size
    tip_theta = arrow_angle_deg
    left_r = arrow_length - head_size * 0.3
    left_theta = (arrow_angle_deg - 30) % 360
    right_r = arrow_length - head_size * 0.3
    right_theta = (arrow_angle_deg + 30) % 360
    
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
            tickfont=dict(size=12, weight='bold'),
            showgrid=True,
            gridcolor='#ddd',
            gridwidth=1,
            rotation=90,
            direction='counterclockwise'
        )
    )
    
    compass_fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=40),
        polar_bgcolor="#fafafa",
        showlegend=False,
    )
    
    # Add clarifying text below the plot
    compass_fig.add_annotation(
        text="Shows direction wind is coming from",
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=10, color='#666'),
        xanchor="center"
    )
    
    return compass_fig


def create_forecast_charts(haz_filtered: pd.DataFrame, today_date) -> Tuple[go.Figure, go.Figure]:
    """Create forecast risk bar chart and variables line chart."""
    haz_dates_dt = pd.to_datetime(haz_filtered["date"])
    
    # Risk bar chart
    f1 = go.Figure()
    f1.add_trace(go.Bar(
        x=haz_dates_dt,
        y=haz_filtered["total"],
        name="Risk (0–100)",
        marker_color=[color_for_risk(x) for x in haz_filtered["total"]],
    ))
    
    # Add vertical line for today
    haz_dates = haz_dates_dt.dt.date
    if (haz_dates == today_date).any():
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

    # Variables line chart
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=haz_dates_dt, y=haz_filtered["temp_c"], name="T max (°C)", mode='lines+markers'))
    f2.add_trace(go.Scatter(x=haz_dates_dt, y=haz_filtered["rh_pct"], name="RH min (%)", mode='lines+markers'))
    f2.add_trace(go.Scatter(x=haz_dates_dt, y=haz_filtered["wind_kmh"], name="Wind max (km/h)", mode='lines+markers'))
    f2.add_trace(go.Scatter(x=haz_dates_dt, y=haz_filtered["days_no_rain"], name="Days without rain", mode='lines+markers'))
    
    if (haz_dates == today_date).any():
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

    return f1, f2


