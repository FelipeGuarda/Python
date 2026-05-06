"""CR800 Campbell Scientific datalogger fetcher via PakBus over TCP."""

import json
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
from pycampbellcr1000 import CR1000

from src.paths import _STATE_PATH
from src.tz_utils import localize_santiago_to_utc


def _load_state() -> dict:
    if _STATE_PATH.exists():
        with open(_STATE_PATH) as f:
            return json.load(f)
    return {}


def _save_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def connect(host: str, port: int, pakbus_address: int = 1):
    """Connect to CR800 via PakBus TCP. Returns a CR1000 logger object."""
    return CR1000.from_url(f"tcp:{host}:{port}", dest_addr=pakbus_address)


@contextmanager
def cr800_session(host: str, port: int, pakbus_address: int = 1):
    """Connect to CR800 and send a PakBus Bye command on exit."""
    logger = connect(host, port, pakbus_address)
    try:
        yield logger
    finally:
        logger.bye()


def _decode_name(name) -> str:
    """Normalize a table/column name: decode bytes or strip b'...' string artifacts."""
    if isinstance(name, bytes):
        return name.decode("utf-8")
    s = str(name)
    if s.startswith("b'") and s.endswith("'"):
        return s[2:-1]
    return s


def list_tables(logger) -> list:
    """List available data tables on the logger (returns plain strings)."""
    return [_decode_name(t) for t in logger.list_tables()]


def process_raw(data: list, station_id: str) -> pd.DataFrame:
    """Convert a raw pycampbellcr1000 result list into a clean DataFrame."""
    df = pd.DataFrame(data)
    df.columns = [_decode_name(c) for c in df.columns]

    ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    naive_ts = pd.to_datetime(df[ts_col], errors="coerce")
    df["timestamp"] = localize_santiago_to_utc(naive_ts)
    df["station_id"] = station_id

    rename_map = {
        "AirTC_Avg": "temperature_air",
        "RH_Avg": "relative_humidity",
        "WS_ms_Avg": "wind_speed",
        "WindDir_Avg": "wind_direction",
        "Rain_mm_Tot": "precipitation",
        "incomingSW_Avg": "solar_radiation",
        "BattV_Min": "battery_voltage",
    }
    df = df.rename(columns=rename_map)
    df = df.drop(columns=[c for c in ("RecNbr", ts_col) if c in df.columns])

    schema_cols = ["station_id", "timestamp", "temperature_air", "relative_humidity",
                   "wind_speed", "wind_direction", "precipitation", "solar_radiation",
                   "battery_voltage"]
    for col in schema_cols:
        if col not in df.columns:
            df[col] = None

    return df.dropna(subset=["timestamp"])


def fetch_since(logger, station_id: str, table_name: str = None):
    """
    Fetch all records since last saved timestamp, in 24-hour chunks.

    Yields ``(df, commit)`` per chunk. The caller MUST call ``commit()`` only
    after the chunk has been successfully persisted (e.g. upsert returned
    without raising). State is advanced inside ``commit()`` — never before the
    yield — so a failed upsert leaves the state pointing at the last
    successfully-committed chunk and the next run replays from there. Upserts
    are idempotent, so replay is safe.
    """
    from datetime import timedelta
    import pytz

    state = _load_state()

    if table_name is None:
        table_name = "Table1"

    state_key = f"{station_id}:{table_name}"
    last_ts_str = state.get(state_key)

    santiago = pytz.timezone("America/Santiago")
    now_local = datetime.now(santiago).replace(tzinfo=None)

    if last_ts_str:
        last_ts_utc = datetime.fromisoformat(last_ts_str)
        chunk_start = last_ts_utc.astimezone(santiago).replace(tzinfo=None)
    else:
        # No state: start 24 hours back (historical data already loaded via backfill)
        chunk_start = now_local - timedelta(hours=24)

    chunk_size = timedelta(hours=24)
    total_rows = 0

    while chunk_start < now_local:
        chunk_end = min(chunk_start + chunk_size, now_local)
        print(f"  Chunk {chunk_start.strftime('%Y-%m-%d %H:%M')} → "
              f"{chunk_end.strftime('%Y-%m-%d %H:%M')} (Santiago)...")

        data = logger.get_data(table_name, chunk_start, chunk_end)

        if data:
            df = process_raw(data, station_id)
            if not df.empty:
                latest_iso = df["timestamp"].max().isoformat()

                def commit(_iso=latest_iso):
                    state[state_key] = _iso
                    _save_state(state)

                total_rows += len(df)
                yield df, commit

        chunk_start = chunk_end

    if total_rows:
        print(f"  Total fetched: {total_rows} rows.")


def fetch_range(logger, station_id: str, start: str, end: str, table_name: str = "Table1"):
    """
    Fetch records between two dates (inclusive) without reading or updating state.

    start/end: ISO date strings, e.g. "2026-03-04" or "2026-03-04T14:00".
    Yields one DataFrame per 24-hour chunk. Safe to run alongside the normal
    scheduler — state is never touched.
    """
    from datetime import timedelta
    import pytz

    santiago = pytz.timezone("America/Santiago")

    def _parse(s):
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = santiago.localize(dt)
        return dt.astimezone(santiago).replace(tzinfo=None)

    chunk_start = _parse(start)
    range_end = _parse(end)
    # Include the full end day if only a date was given
    if "T" not in end and ":" not in end:
        range_end = range_end.replace(hour=23, minute=59, second=59)

    chunk_size = timedelta(hours=24)
    total_rows = 0

    while chunk_start <= range_end:
        chunk_end = min(chunk_start + chunk_size, range_end)
        print(f"  Range chunk {chunk_start.strftime('%Y-%m-%d %H:%M')} → "
              f"{chunk_end.strftime('%Y-%m-%d %H:%M')} (Santiago)...")

        data = logger.get_data(table_name, chunk_start, chunk_end)
        if data:
            df = process_raw(data, station_id)
            if not df.empty:
                total_rows += len(df)
                yield df

        chunk_start = chunk_end + timedelta(seconds=1)

    print(f"  Range fetch total: {total_rows} rows.")
