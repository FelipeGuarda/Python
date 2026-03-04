"""Parse the merged_timeline.csv pre-merged meteorological data export from CR800."""

from pathlib import Path

import pandas as pd

# Map CSV column names → weather_station schema column names (core 8)
_CORE_RENAME = {
    "AirTC_Avg": "temperature_air",
    "RH_Avg": "relative_humidity",
    "WS_ms_Avg": "wind_speed",
    "WindDir_Avg": "wind_direction",
    "Rain_mm_Tot": "precipitation",
    "incomingSW_Avg": "solar_radiation",
    "BattV_Min": "battery_voltage",
}

# Columns to drop entirely (internal / not useful in DB)
_DROP_COLS = {"RECORD"}


def parse(csv_path: Path, station_id: str = "bosque_pehuen") -> pd.DataFrame:
    """
    Parse the merged CR800 CSV export.

    Returns a DataFrame with:
      - Core schema columns (station_id, timestamp, temperature_air, ...)
      - All extra CR800 columns kept under their original names
    """
    csv_path = Path(csv_path)
    print(f"→ Parsing met CSV: {csv_path} ...")

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Drop unwanted columns
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    # Parse timestamp (America/Santiago → UTC)
    df["TIMESTAMP"] = df["TIMESTAMP"].str.strip()
    df["timestamp"] = (
        pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        .dt.tz_localize("America/Santiago", ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    df = df.drop(columns=["TIMESTAMP"])

    # Rename core columns
    df = df.rename(columns=_CORE_RENAME)

    # Convert all remaining text columns to numeric where possible
    non_text = [c for c in df.columns if c not in ("timestamp", "source_file", "station_id")]
    for col in non_text:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add station_id as first column
    df.insert(0, "station_id", station_id)

    # Drop rows with no valid timestamp
    df = df.dropna(subset=["timestamp"])

    print(f"  Parsed {len(df)} rows, {len(df.columns)} columns.")
    return df
