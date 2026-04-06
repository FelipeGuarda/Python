"""
One-shot script to recover the ~8 records dropped each year during Chile's DST fall-back.
Fetches a narrow 2-hour window around each DST transition directly from the CR800,
processes through the fixed tz_localize (ambiguous="infer"), and upserts into DuckDB.

Run once after applying the ambiguous="infer" fix in src/fetchers/cr800.py.
Safe to re-run — all upserts are idempotent.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.db import connect, init_schema, upsert_df, ensure_columns
from src.fetchers.cr800 import connect as cr800_connect, _process_raw

# Chile DST fall-back nights (first Saturday of April each year).
# Fetch window: 22:30 → 00:30+1day in naive local Santiago time,
# wide enough to give pandas context on both sides of the transition.
DST_NIGHTS = [
    datetime(2019, 4, 6, 22, 30),
    datetime(2020, 4, 4, 22, 30),
    datetime(2021, 4, 3, 22, 30),
    datetime(2022, 4, 2, 22, 30),
    datetime(2023, 4, 1, 22, 30),
    datetime(2024, 4, 6, 22, 30),
    datetime(2025, 4, 5, 22, 30),
    datetime(2026, 4, 4, 22, 30),
]
WINDOW = timedelta(hours=2)  # 22:30 → 00:30+1

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

cr_cfg = cfg["cr800"]

print("→ Connecting to CR800...")
logger = cr800_connect(cr_cfg["host"], cr_cfg["port"], cr_cfg["pakbus_address"])
station_id = cr_cfg["station_id"]

con = connect()
init_schema(con)

total_recovered = 0

for start in DST_NIGHTS:
    end = start + WINDOW
    label = start.strftime("%Y-%m-%d")
    print(f"  Fetching DST window {label}  ({start.strftime('%H:%M')} → {end.strftime('%H:%M +1d')})...", end=" ")

    try:
        data = logger.get_data("Table1", start, end)
    except Exception as e:
        print(f"ERROR: {e}")
        continue

    if not data:
        print("no data returned")
        continue

    df = _process_raw(data, station_id)
    if df.empty:
        print("0 rows after processing")
        continue

    ensure_columns(con, "weather_station", df)
    n = upsert_df(con, "weather_station", df)
    total_recovered += n
    print(f"{n} rows upserted")

con.close()
print(f"\n→ Done. Total rows recovered: {total_recovered}")
