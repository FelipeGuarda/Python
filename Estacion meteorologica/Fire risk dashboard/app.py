# -*- coding: utf-8 -*-
"""
Fire-Risk Dashboard — Main Application
Stack: Streamlit + Plotly + pydeck + Open-Meteo

How to run
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st  # pyright: ignore[reportMissingImports]
import pydeck as pdk  # pyright: ignore[reportMissingImports]
import joblib
from pathlib import Path

# Local imports
from config import TZ, TODAY, RISK_COLORS
from data_fetcher import fetch_open_meteo
from risk_calculator import compute_days_without_rain, best_hour_by_day, color_for_risk
from visualizations import create_polar_plot, create_wind_compass, create_forecast_charts, create_dual_gauge, load_validation_results, get_validation_modal_content
from map_utils import create_map_layers, create_map_view_state

# ---------------------------
# ML Model Loading
# ---------------------------
@st.cache_resource
def load_ml_model():
    """Load trained ML fire prediction model (cached)."""
    model_path = Path("ml_model") / "fire_model.pkl"
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.warning(f"Could not load ML model: {e}")
            return None
    return None

def predict_fire_probability(model, temp_c: float, rh_pct: float, wind_kmh: float, days_no_rain: int) -> float:
    """Predict fire probability using ML model. Returns probability as percentage (0-100)."""
    if model is None:
        return None
    try:
        # Create DataFrame with feature names matching training data
        X = pd.DataFrame({
            'temp_c': [temp_c],
            'rh_pct': [rh_pct],
            'wind_kmh': [wind_kmh],
            'days_no_rain': [days_no_rain]
        })
        proba = model.predict_proba(X)[0][1]  # Probability of fire class (class 1)
        return proba * 100  # Convert to percentage
    except Exception as e:
        st.warning(f"ML prediction error: {e}")
        return None

# Page config
st.set_page_config(
    page_title="Fire Risk Dashboard — Bosque Pehuén",
    layout="wide",
)

# Load ML model
ml_model = load_ml_model()

# ---------------------------
# Modal dialog for validation details
# ---------------------------
@st.dialog("Statistical Validation Results")
def show_validation_modal(selected_date, rule_based_risk, ml_probability):
    """Display statistical validation details in a modal dialog."""
    validation, has_plot = get_validation_modal_content()
    
    if validation is None:
        st.warning("Validation results not available. Run `python ml_model/validate_model_agreement.py` to generate.")
        return
    
    # Header with selected date
    st.markdown(f"### Comparison for {selected_date}")
    
    # Current day's comparison
    difference = rule_based_risk - ml_probability
    abs_difference = abs(difference)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Rule-Based Risk", f"{rule_based_risk:.1f}")
    with col_b:
        st.metric("ML Probability", f"{ml_probability:.1f}%")
    with col_c:
        st.metric("Difference", f"{difference:+.1f} pts")
    
    # Check if within limits
    ba = validation['bland_altman']
    within_limits = (difference >= ba['lower_limit']) and (difference <= ba['upper_limit'])
    
    if within_limits:
        st.success(f"✓ Within expected range (95% limits: {ba['lower_limit']:.1f} to {ba['upper_limit']:.1f} pts)")
    else:
        st.warning(f"! Outside expected range (95% limits: {ba['lower_limit']:.1f} to {ba['upper_limit']:.1f} pts)")
    
    st.markdown("---")
    
    # Overall validation statistics
    st.markdown("### Overall Statistical Validation")
    
    if validation.get('is_mock_data', False):
        st.info("Note: Currently showing demonstration data. Run validation with real training data for actual results.")
    
    # Key statistics
    st.markdown("#### Statistical Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "McNemar's Test p-value",
            f"{validation['mcnemar']['p_value']:.3f}",
            help="Tests for systematic disagreement. p > 0.05 means no significant difference."
        )
        interpretation = "No significant difference" if validation['mcnemar']['p_value'] > 0.05 else "Significant difference"
        st.caption(f"**{interpretation}**")
    
    with col2:
        ccc = validation['concordance_coefficient']
        st.metric(
            "Concordance Coefficient",
            f"{ccc:.3f}",
            help="Measures agreement strength. >0.90=strong, 0.75-0.90=moderate, <0.75=poor"
        )
        if ccc > 0.90:
            strength = "Strong agreement"
        elif ccc > 0.75:
            strength = "Moderate agreement"
        else:
            strength = "Limited agreement"
        st.caption(f"**{strength}**")
    
    # Bland-Altman results
    st.markdown("#### Bland-Altman Limits")
    st.caption("Expected range of differences across all predictions")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Mean Difference", f"{ba['mean_diff']:.1f} pts")
    with col4:
        st.metric("Lower Limit (-1.96 SD)", f"{ba['lower_limit']:.1f} pts")
    with col5:
        st.metric("Upper Limit (+1.96 SD)", f"{ba['upper_limit']:.1f} pts")
    
    st.caption(f"{ba['pct_within_limits']:.1f}% of predictions fall within these limits")
    
    # Show plot if available
    if has_plot:
        st.markdown("#### Bland-Altman Plot")
        st.image("ml_model/plots/bland_altman.png", use_container_width=True)
    
    # Interpretation
    st.markdown("#### Interpretation")
    st.markdown(validation['interpretation'])
    
    # Sample size
    st.caption(f"Based on {validation['n_samples']:,} historical samples ({validation['n_fires']:,} fires, {validation['n_samples']-validation['n_fires']:,} non-fires)")
    
    # Close button
    if st.button("Close", type="primary", use_container_width=True):
        st.rerun()

# ---------------------------
# Default settings (previously in sidebar)
# ---------------------------
# Center: lat=-39.44132, lon=-71.75140 (from UTM 19S)
lat = -39.44132
lon = -71.75140

# Forecast days to fetch: 14
days_ahead = 14

# Map covers entire Araucania region

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
        if st.button("Today", width='stretch'):
            today_idx = int(np.clip(np.searchsorted(dates_sorted, TODAY.date()), 0, len(dates_sorted)-1))
            if today_idx < len(dates_sorted):
                st.session_state.selected_date = dates_sorted[today_idx]
                st.rerun()
    
    with col_date2:
        if st.button("Tomorrow", width='stretch'):
            tomorrow = (TODAY + pd.Timedelta(days=1)).date()
            tom_idx = int(np.clip(np.searchsorted(dates_sorted, tomorrow), 0, len(dates_sorted)-1))
            if tom_idx < len(dates_sorted):
                if dates_sorted[tom_idx] == tomorrow:
                    st.session_state.selected_date = dates_sorted[tom_idx]
                elif tom_idx > 0 and abs((dates_sorted[tom_idx-1] - tomorrow).days) < abs((dates_sorted[tom_idx] - tomorrow).days):
                    st.session_state.selected_date = dates_sorted[tom_idx-1]
                else:
                    st.session_state.selected_date = dates_sorted[tom_idx]
                st.rerun()
    
    with col_date3:
        if st.button("Next 3 days", width='stretch'):
            next_3_days = (TODAY + pd.Timedelta(days=3)).date()
            n3_idx = int(np.clip(np.searchsorted(dates_sorted, next_3_days), 0, len(dates_sorted)-1))
            if n3_idx < len(dates_sorted):
                st.session_state.selected_date = dates_sorted[n3_idx]
                st.rerun()
    
    with col_date4:
        if st.button("Next 7 days", width='stretch'):
            next_week = (TODAY + pd.Timedelta(days=7)).date()
            nw_idx = int(np.clip(np.searchsorted(dates_sorted, next_week), 0, len(dates_sorted)-1))
            if nw_idx < len(dates_sorted):
                st.session_state.selected_date = dates_sorted[nw_idx]
                st.rerun()
    
    # Display selected date prominently
    sel_date = st.session_state.selected_date
    st.markdown(f"<div style='text-align:center;font-size:24px;font-weight:bold;padding:12px;margin:10px 0;'>{sel_date}</div>", unsafe_allow_html=True)

    row = haz.loc[haz["date"] == sel_date].iloc[0]

    # Create and display polar plot with legend
    fig = create_polar_plot(row)
    
    # Display plot and legend side by side
    plot_col, legend_col = st.columns([5, 1])
    with plot_col:
        st.plotly_chart(fig, width='stretch')
    
    with legend_col:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        st.markdown("**Risk Color Legend**", unsafe_allow_html=True)
        legend_html = "<div style='font-size:15px;'>"
        for lo, hi, col in sorted(RISK_COLORS, key=lambda x: x[0]):
            if hi == 100.0:
                label = f"{lo:.0f}+"
            else:
                label = f"{lo:.0f}-{hi:.0f}"
            legend_html += f"<div style='display:flex;align-items:center;margin-bottom:6px;'><div style='width:20px;height:12px;background:{col};border-radius:3px;margin-right:8px;border:1px solid #ccc;'></div><span>{label}</span></div>"
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    # Forecast/historical indicator
    is_forecast = sel_date > TODAY.date()
    indicator_text = "Forecast" if is_forecast else "Historical"
    indicator_color = "#2196F3" if is_forecast else "#757575"
    st.markdown(f"<div style='text-align:center;padding:8px;background:{indicator_color}20;border-radius:6px;color:{indicator_color};font-weight:bold;'>{indicator_text}</div>", unsafe_allow_html=True)

    st.caption(f"Most dangerous hour for {sel_date}: {pd.to_datetime(row['timestamp']).strftime('%H:%M')} (local)")

# Right column: Risk index and wind compass
with colB:
    row = haz.loc[haz["date"] == sel_date].iloc[0]
    total = float(row["total"]) if not pd.isna(row["total"]) else 0.0
    col = color_for_risk(total)
    
    # Dual gauge visualization (rule-based risk + ML probability)
    if ml_model is not None:
        ml_prob = predict_fire_probability(
            ml_model,
            float(row["temp_c"]),
            float(row["rh_pct"]),
            float(row["wind_kmh"]),
            int(row["days_no_rain"])
        )
        
        if ml_prob is not None:
            st.subheader("Risk Comparison")
            gauge_fig = create_dual_gauge(total, ml_prob, sel_date)
            st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
            
            # Captions under gauges
            col_cap1, col_cap2 = st.columns(2)
            with col_cap1:
                st.caption("Based on Chilean fire danger standards")
            with col_cap2:
                st.caption("Trained on 20 years of Chilean fire data")
            
            # Agreement indicator based on statistical validation
            validation = load_validation_results()
            difference = abs(total - ml_prob)
            
            # Use Bland-Altman limits if available
            if 'bland_altman' in validation:
                upper_limit = abs(validation['bland_altman']['upper_limit'])
                agreement = difference <= upper_limit
                agreement_text = "Methods agree (within statistical limits)" if agreement else "Methods differ (outside expected range)"
            else:
                # Fallback to simple threshold
                agreement = difference <= 15
                agreement_text = "Methods agree" if agreement else "Methods differ"
            
            agreement_color = "#4caf50" if agreement else "#ff9800"
            
            # Agreement indicator with info button
            col_agree, col_info = st.columns([4, 1])
            with col_agree:
                st.markdown(
                    f"<div style='padding:8px;background:{agreement_color}20;border-radius:6px;color:{agreement_color};text-align:center;'>"
                    f"{agreement_text}</div>",
                    unsafe_allow_html=True
                )
            
            with col_info:
                if st.button("ℹ️", key="validation_info", help="View statistical validation details"):
                    show_validation_modal(sel_date, total, ml_prob)
        else:
            # Fallback if ML prediction fails
            st.subheader("Rule-Based Risk Index")
            st.metric(label=str(sel_date), value=f"{total:0.0f} / 100")
            st.markdown(f"<div style='height:12px;width:{total}%;background:{col};border-radius:6px;'></div>", unsafe_allow_html=True)
            st.caption("Based on Chilean fire danger standards")
    else:
        # If ML model not loaded, show only rule-based risk
        st.subheader("Rule-Based Risk Index")
        st.metric(label=str(sel_date), value=f"{total:0.0f} / 100")
        st.markdown(f"<div style='height:12px;width:{total}%;background:{col};border-radius:6px;'></div>", unsafe_allow_html=True)
        st.caption("Based on Chilean fire danger standards")
    
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
            
            compass_fig = create_wind_compass(avg_wind_dir, avg_wind_speed, risk_color=col)
            st.plotly_chart(compass_fig, width='stretch', config={'displayModeBar': False})
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
    f1, f2 = create_forecast_charts(haz_filtered, st.session_state.selected_date)
    st.plotly_chart(f1, width='stretch')
    st.plotly_chart(f2, width='stretch')
else:
    st.info("No forecast data available for the selected range.")

# ---------------------------
# Regional Map
# ---------------------------
st.subheader("Regional wind map")
st.caption(f"Showing wind patterns for: {st.session_state.selected_date}")

# Get wind data for selected date
sel_date = st.session_state.selected_date
row = haz.loc[haz["date"] == sel_date].iloc[0]
hourly_sel = hourly[hourly["date"] == sel_date]

wind_data = {}
if not hourly_sel.empty:
    hourly_sel_copy = hourly_sel.copy()
    hourly_sel_copy["hour"] = pd.to_datetime(hourly_sel_copy["timestamp"]).dt.hour
    wind_window = hourly_sel_copy[(hourly_sel_copy["hour"] >= 14) & (hourly_sel_copy["hour"] <= 16)]
    
    if not wind_window.empty:
        wind_data = {
            'wind_dir': float(wind_window["wind_dir"].mean()),
            'wind_speed': float(wind_window["wind_kmh"].mean()),
            'lat': lat,
            'lon': lon
        }

# Create map layers with wind flow field
layers = create_map_layers(wind_data, lat, lon)

view_state = create_map_view_state(center_lat=lat, center_lon=lon)

tooltip = {
    "html": "<b>Bosque Pehuén</b><br>Wind: {wind_speed:.1f} km/h<br>Direction: {wind_dir:.0f}°",
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
