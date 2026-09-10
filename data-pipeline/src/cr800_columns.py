"""How a CR800 channel name becomes a `weather_station` column.

WHAT THIS OWNS. The mapping from the datalogger's own column names to the
warehouse's, and the fact that **the record counter arrives under two different
names**: `RECORD` in TOA5 files and in `merged_timeline.csv`, `RecNbr` over
PakBus. Both are the same thing and both land in `record`.

WHY IT EXISTS. Until 2026-09-10 this mapping lived in three copies —
`parsers/met_csv.py`, `parsers/toa5.py` and `fetchers/cr800.py` — and all three
independently discarded the record counter:

    met_csv.py   _DROP_COLS = {"RECORD"}          # "internal / not useful in DB"
    toa5.py      result = df[schema_cols]         # RECORD not among the 9
    cr800.py     df.drop(columns=["RecNbr", ...])  # the antenna delivers it; dropped

The counter is the opposite of not useful: it is the only column that makes the
record's continuity provable and the only one that resolves a repeated
timestamp. Reading the 2018-2025 record through it showed that the "12-hour gap"
of 2023-07-12/13 is a clock jump and not lost data, that there are exactly two
clock events in seven years (+11:45:00 at RECORD 179607, -11:45:00 at RECORD
187673) and that they cancel exactly. None of that is derivable from the
timestamp axis, which is non-monotonic in two windows. See
`Estacion meteorologica/GeoMountains/build_registry.py`, which had to reconstruct
the counter from the raw dumps because the warehouse had thrown it away.

A CAVEAT THE COUNTER EXPOSES BUT DOES NOT FIX. `weather_station`'s primary key is
(station_id, timestamp), so where the logger stamped two records with the same
time — 47 of them on 2023-10-04/05, when the clock was set back — the warehouse
still keeps only one, and `record` will be discontinuous there. The counter lets a
consumer *detect* that; it does not let the table hold both.
"""

from __future__ import annotations

import pandas as pd

#: Warehouse column for the datalogger's record counter.
RECORD_COLUMN = "record"

#: Every name the record counter arrives under. TOA5 files and the merged CSV use
#: the first; a PakBus table read uses the second.
RECORD_ALIASES: tuple[str, ...] = ("RECORD", "RecNbr")

#: Channel → warehouse column, for the columns `schema.sql` declares by name.
#: Every other channel keeps its native name and is added by `db.ensure_columns`.
CHANNEL_RENAME: dict[str, str] = {
    "TIMESTAMP": "timestamp",
    "AirTC_Avg": "temperature_air",
    "RH_Avg": "relative_humidity",
    "WS_ms_Avg": "wind_speed",
    "WindDir_Avg": "wind_direction",
    "Rain_mm_Tot": "precipitation",
    "incomingSW_Avg": "solar_radiation",
    "BattV_Min": "battery_voltage",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CR800 channels to warehouse columns and type the record counter.

    Returns a new frame. Columns not in the mapping are left untouched, which is
    deliberate: the extra sensor channels reach the warehouse under their native
    TOA5 names and `db.ensure_columns` creates them.

    `record` comes back as nullable `Int64`. It is a counter, so a float column
    would be wrong even when it round-trips, and the rows already in the
    warehouse from before this change have no counter at all.
    """
    renames = {src: dst for src, dst in CHANNEL_RENAME.items() if src in df.columns}
    renames.update({alias: RECORD_COLUMN for alias in RECORD_ALIASES if alias in df.columns})

    out = df.rename(columns=renames)
    if RECORD_COLUMN in out.columns:
        out[RECORD_COLUMN] = pd.to_numeric(out[RECORD_COLUMN], errors="coerce").astype("Int64")
    return out
