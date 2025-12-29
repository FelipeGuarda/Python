# -*- coding: utf-8 -*-
"""
Configuration constants for Fire Risk Dashboard
"""

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

# Timezone
TZ = "America/Santiago"
TODAY = pd.Timestamp.now(tz=TZ).normalize()

# Score tables (0-25 points each, sum to 100)
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

# Araucania region bounds
ARAUCANIA_LAT_MIN = -40.0
ARAUCANIA_LAT_MAX = -38.0
ARAUCANIA_LON_MIN = -73.0
ARAUCANIA_LON_MAX = -71.0


