"""
Fire risk calculation — ported from Estacion meteorologica/Fire risk dashboard/.
Rule-based FRI (Fire Risk Index) + ML Random Forest model.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# ── Scoring bins (sum to 100 max) ────────────────────────────────────────

TEMP_BINS = [
    (-np.inf, 0, 2.7), (0, 5, 5.4), (6, 10, 8.1), (11, 15, 10.8),
    (16, 20, 13.5), (21, 25, 16.2), (26, 30, 18.9), (31, 35, 21.6),
    (35, np.inf, 25.0),
]

RH_BINS = [
    (0, 10, 25.0), (11, 20, 22.5), (21, 30, 20.0), (31, 40, 17.5),
    (41, 50, 15.0), (51, 60, 12.5), (61, 70, 10.0), (71, 80, 7.5),
    (81, 90, 5.0), (91, 100, 2.5),
]

WIND_BINS = [
    (-np.inf, 3.0, 1.5), (3.0, 5.9, 3.0), (6.0, 8.9, 4.5),
    (9.0, 11.9, 6.0), (12.0, 14.9, 7.5), (15.0, 17.9, 9.0),
    (18.0, 20.9, 10.5), (21.0, 23.9, 12.0), (24.0, 26.9, 13.5),
    (27.0, np.inf, 15.0),
]

DAYS_NR_BINS = [
    (0, 1, 3.5), (2, 4, 7.0), (5, 7, 10.5), (8, 10, 14.0),
    (11, 13, 17.5), (14, 16, 21.0), (17, 19, 24.5), (20, 22, 28.0),
    (23, 25, 31.5), (26, np.inf, 35.0),
]

RISK_COLORS = [
    (0.0, 19.999, "#2e7d32"),   # green
    (20.0, 39.999, "#c0ca33"),  # yellow-green
    (40.0, 59.999, "#fbc02d"),  # yellow
    (60.0, 79.999, "#fb8c00"),  # orange
    (80.0, 89.999, "#e53935"),  # red-orange
    (90.0, 100.0, "#b71c1c"),   # dark red
]

RISK_LABELS = {
    "#2e7d32": "Bajo",
    "#c0ca33": "Moderado-Bajo",
    "#fbc02d": "Moderado",
    "#fb8c00": "Alto",
    "#e53935": "Muy Alto",
    "#b71c1c": "Extremo",
}

# ── ML model (lazy load) ────────────────────────────────────────────────

_ml_model = None
_ML_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "Estacion meteorologica" / "Fire risk dashboard" / "ml_model" / "fire_model.pkl"
)


def _get_ml_model():
    global _ml_model
    if _ml_model is None and _ML_MODEL_PATH.exists():
        try:
            with open(_ML_MODEL_PATH, "rb") as f:
                _ml_model = pickle.load(f)
        except Exception:
            # Model may be pickled with a different scikit-learn version
            _ml_model = False  # sentinel: tried and failed
    return _ml_model if _ml_model is not False else None


# ── Core functions ───────────────────────────────────────────────────────

def _bin_score(value: float, bins: list) -> float:
    for lo, hi, sc in bins:
        if lo == -np.inf and value < hi:
            return sc
        if hi == np.inf and value > lo:
            return sc
        if value >= lo and value <= hi:
            return sc
    return 0.0


def risk_components(
    temp_c: float, rh_pct: float, wind_kmh: float, days_no_rain: int
) -> dict:
    """Rule-based fire risk index. Returns component scores and total (0-100)."""
    t = _bin_score(temp_c, TEMP_BINS)
    h = _bin_score(rh_pct, RH_BINS)
    w = _bin_score(wind_kmh, WIND_BINS)
    d = _bin_score(days_no_rain, DAYS_NR_BINS)
    total = t + h + w + d
    return {
        "temp_score": t,
        "rh_score": h,
        "wind_score": w,
        "days_score": d,
        "total": total,
        "color": color_for_risk(total),
        "label": label_for_risk(total),
    }


def color_for_risk(total: float) -> str:
    for lo, hi, col in sorted(RISK_COLORS, key=lambda x: x[0]):
        if lo <= total <= hi:
            return col
    return RISK_COLORS[-1][2]


def label_for_risk(total: float) -> str:
    return RISK_LABELS.get(color_for_risk(total), "Desconocido")


def ml_fire_probability(
    temp_c: float, rh_pct: float, wind_kmh: float, days_no_rain: int
) -> Optional[float]:
    """Return ML model's fire probability (0-1), or None if model unavailable."""
    model = _get_ml_model()
    if model is None:
        return None
    features = np.array([[temp_c, rh_pct, wind_kmh, days_no_rain]])
    proba = model.predict_proba(features)[0][1]
    return float(proba)
