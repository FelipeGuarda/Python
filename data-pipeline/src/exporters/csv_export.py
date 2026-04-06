"""CSV export and health check for weather_station."""

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

_STATE_PATH = Path(__file__).parent.parent.parent / "data" / "cr800_state.json"
_INTERVAL_MINUTES = 15
_GAP_THRESHOLD_MINUTES = 16  # 15-min interval + 1 min tolerance


def export_weather_station(con: duckdb.DuckDBPyConnection, output_dir: Path) -> Path:
    """Export full weather_station table to a dated CSV. Returns the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = output_dir / f"weather_station_{date_str}.csv"

    df = con.execute("SELECT * FROM weather_station ORDER BY timestamp").df()
    df.to_csv(out_path, index=False)
    print(f"  Exported {len(df):,} rows → {out_path.name}")
    return out_path


def health_check(con: duckdb.DuckDBPyConnection, output_dir: Path, verbose: bool = False) -> None:
    """Print a health report for weather_station data."""
    print("→ CR800 / weather_station Health Check")
    print()

    # Last fetch from CR800 state file
    if _STATE_PATH.exists():
        with open(_STATE_PATH) as f:
            state = json.load(f)
        last_ts_str = max(state.values(), default=None) if state else None
        if last_ts_str:
            last_ts = datetime.fromisoformat(last_ts_str)
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - last_ts
            days, remainder = divmod(int(age.total_seconds()), 86400)
            hours = remainder // 3600
            age_str = f"{days}d {hours}h ago" if days else f"{hours}h ago"
            print(f"  Last CR800 fetch:  {last_ts.strftime('%Y-%m-%d %H:%M')} UTC  ({age_str})")
        else:
            print("  Last CR800 fetch:  unknown (state file empty)")
    else:
        print("  Last CR800 fetch:  unknown (no state file — CR800 never fetched)")

    # DuckDB row count and date range
    try:
        db_count, ts_min, ts_max = con.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM weather_station"
        ).fetchone()
        print(f"  DuckDB rows:       {db_count:,}")
        if ts_min and ts_max:
            print(f"  Date range:        {ts_min} → {ts_max}")
        else:
            print("  Date range:        table is empty")
    except Exception as e:
        print(f"  DuckDB rows:       error ({e})")
        db_count = None

    # Latest CSV snapshot
    csvs = sorted(output_dir.glob("weather_station_*.csv")) if output_dir.exists() else []
    if csvs:
        latest_csv = csvs[-1]
        csv_df = pd.read_csv(latest_csv)
        csv_count = len(csv_df)
        if db_count is not None:
            match_str = "✓ match" if csv_count == db_count else f"✗ MISMATCH (DB={db_count:,})"
        else:
            match_str = ""
        print(f"  Latest CSV:        {latest_csv.name}")
        print(f"  CSV rows:          {csv_count:,}  {match_str}")
    else:
        print(f"  Latest CSV:        none found in {output_dir}")

    # Gap detection
    if db_count and db_count > 1:
        gaps = con.execute(f"""
            WITH ordered AS (
                SELECT timestamp,
                       LAG(timestamp) OVER (ORDER BY timestamp) AS prev_ts
                FROM weather_station
            )
            SELECT prev_ts, timestamp
            FROM ordered
            WHERE prev_ts IS NOT NULL
              AND timestamp - prev_ts > INTERVAL '{_GAP_THRESHOLD_MINUTES} minutes'
            ORDER BY timestamp
        """).fetchall()

        if not gaps:
            print("  Gaps (>15 min):    none")
        else:
            print(f"  Gaps (>15 min):    {len(gaps)}", end="")
            if not verbose:
                print("  (run --health --verbose to list)")
            else:
                print()
                print()
                print("  Gap details:")
                for prev_ts, ts in gaps:
                    missing = round((ts - prev_ts).total_seconds() / 60) - _INTERVAL_MINUTES
                    print(f"    {prev_ts}  →  {ts}  (~{missing} min missing)")

    print()
