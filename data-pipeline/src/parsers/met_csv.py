"""Parse the merged_timeline.csv pre-merged meteorological data export from CR800."""

from pathlib import Path

import pandas as pd

from src.cr800_columns import RECORD_COLUMN, normalize_columns
from src.tz_utils import localize_santiago_to_utc


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

    # Parse timestamp (America/Santiago → UTC)
    df["TIMESTAMP"] = df["TIMESTAMP"].str.strip()
    naive_ts = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df["timestamp"] = localize_santiago_to_utc(naive_ts)
    df = df.drop(columns=["TIMESTAMP"])

    df = normalize_columns(df)

    # Convert all remaining text columns to numeric where possible. `record` is
    # already typed by normalize_columns and must not be coerced back to float.
    skip = ("timestamp", "source_file", "station_id", RECORD_COLUMN)
    for col in [c for c in df.columns if c not in skip]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add station_id as first column
    df.insert(0, "station_id", station_id)

    # Drop rows with no valid timestamp
    df = df.dropna(subset=["timestamp"])

    print(f"  Parsed {len(df)} rows, {len(df.columns)} columns.")
    return df
