# -*- coding: utf-8 -*-
"""
Risk calculation functions for fire risk assessment
"""

from typing import Dict, List, Tuple
import datetime as dt
import pandas as pd
import numpy as np

from config import TEMP_BINS, RH_BINS, WIND_BINS, DAYS_NR_BINS, RISK_COLORS


def _bin_score(value: float, bins: List[Tuple[float, float, float]]) -> float:
    """Return score for value according to inclusive/exclusive bins."""
    for lo, hi, sc in bins:
        if lo == -np.inf and value < hi:
            return sc
        if hi == np.inf and value > lo:
            return sc
        if value >= lo and value <= hi:
            return sc
    return 0.0


def risk_components(temp_c: float, rh_pct: float, wind_kmh: float, days_no_rain: int) -> Dict[str, float]:
    """Calculate risk components for given weather variables."""
    t = _bin_score(temp_c, TEMP_BINS)
    h = _bin_score(rh_pct, RH_BINS)
    w = _bin_score(wind_kmh, WIND_BINS)
    d = _bin_score(days_no_rain, DAYS_NR_BINS)
    return {"temp": t, "rh": h, "wind": w, "days": d, "total": t + h + w + d}


def color_for_risk(total: float) -> str:
    """Return hex color for risk score."""
    for lo, hi, col in sorted(RISK_COLORS, key=lambda x: x[0]):
        if lo <= total <= hi:
            return col
    return RISK_COLORS[-1][2]  # default highest (dark red)


def compute_days_without_rain(daily_df: pd.DataFrame, rain_threshold_mm: float = 2.0) -> pd.DataFrame:
    """Compute a running counter of consecutive days without rain."""
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
    Returns one row per date with averaged values and risk scores.
    """
    rows = []
    for d, sub in hourly.groupby("date"):
        if sub.empty:
            continue

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
            "timestamp": sub2["timestamp"].iloc[0],
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

