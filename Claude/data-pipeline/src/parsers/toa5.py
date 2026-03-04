"""Parse Campbell Scientific TOA5 ASCII data files for historical CR800 backfill."""

from pathlib import Path

import pandas as pd
import pytz


def parse(dat_path: Path, station_id: str = "bosque_pehuen") -> pd.DataFrame:
    """
    Parse a TOA5 .dat file.

    TOA5 format:
      Line 1: "TOA5",station_name,logger_type,...   (file info)
      Line 2: column names
      Line 3: units
      Line 4: processing type
      Line 5+: data rows

    Returns:
        pd.DataFrame with station_id column + all sensor columns.
        TIMESTAMP is parsed as UTC.
    """
    dat_path = Path(dat_path)
    print(f"→ Parsing TOA5 file: {dat_path}")

    with open(dat_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Extract column names from line 2 (0-indexed: line index 1)
    col_line = lines[1].strip()
    columns = [c.strip('"') for c in col_line.split(",")]

    # Data starts at line 5 (index 4)
    from io import StringIO
    data_text = "".join(lines[4:])
    df = pd.read_csv(
        StringIO(data_text),
        header=None,
        names=columns,
        na_values=["NAN", "nan", "NaN", ""],
    )

    if "TIMESTAMP" not in df.columns:
        raise ValueError(f"No TIMESTAMP column found in {dat_path}. Columns: {list(df.columns)}")

    # Parse TIMESTAMP (Campbell format: "YYYY-MM-DD HH:MM:SS") as America/Santiago → UTC
    santiago = pytz.timezone("America/Santiago")
    df["TIMESTAMP"] = (
        pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        .dt.tz_localize(santiago, ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    # Map known columns to weather_station schema
    rename_map = {
        "TIMESTAMP": "timestamp",
        "AirTC_Avg": "temperature_air",
        "RH": "relative_humidity",
        "WS_ms_Avg": "wind_speed",
        "WindDir": "wind_direction",
        "Rain_mm_Tot": "precipitation",
        "SlrW_Avg": "solar_radiation",
        "BattV_Min": "battery_voltage",
    }

    schema_cols = ["station_id", "timestamp", "temperature_air", "relative_humidity",
                   "wind_speed", "wind_direction", "precipitation", "solar_radiation",
                   "battery_voltage"]

    df = df.rename(columns=rename_map)
    df["station_id"] = station_id

    # Keep schema columns that exist; leave others as None
    for col in schema_cols:
        if col not in df.columns:
            df[col] = None

    result = df[schema_cols].dropna(subset=["timestamp"])
    print(f"  Parsed {len(result)} rows from {dat_path.name}.")
    return result
