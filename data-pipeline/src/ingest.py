"""Orchestrator: routes data sources → parsers/fetchers → upsert into DuckDB."""

import duckdb
from pathlib import Path

from src.db import upsert_df, ensure_columns
from src.fetchers.open_meteo import fetch as fetch_open_meteo


def ingest_weather_forecast(con: duckdb.DuckDBPyConnection) -> None:
    df = fetch_open_meteo()
    n = upsert_df(con, "weather_forecast", df)
    print(f"  Upserted {n} rows into weather_forecast.")


class CameraTrapIngestNotRebuilt(NotImplementedError):
    """Raised by `ingest_all_ct_campaigns` — the camera-trap path was retired, not fixed."""


def ingest_all_ct_campaigns(con: duckdb.DuckDBPyConnection) -> None:
    """Refuse, loudly. Retired 2026-08-20; the replacement is V2-REVIEW 2.3.

    Three functions used to live here — `ingest_camtrap_dp`, `ingest_timelapse_reviewed`
    and this one iterating `config.yaml`'s `camera_traps.campaigns`. All three are gone,
    along with both parsers, because the path was not merely stale but actively wrong:

    * `timelapse_reviewed.py` re-derived FIVE decisions `camtrap.observations` owns —
      station->camera number, coordinates, Spanish->Latin, Santiago->UTC, and the
      review-comment resolution. The last disagreed on 515 live rows: it knew four
      comment strings and only ever demoted to `blank`, with no rule producing `human`,
      `vehicle` or `unknown`. Ingesting would have rebuilt the 815-row defect that
      V2-REVIEW 1.3 closed.
    * `camtrap_dp.py` parsed a Camtrap DP folder. No such folder has ever existed in this
      monorepo. Its column mapping is preserved in V2-REVIEW 2.3.
    * The campaign list named `primavera_2025`'s CSV as `...reviewed.dedup.csv`, which does
      not exist on disk — so the loop skipped primavera with a warning and ingested
      `pv_2025_2026`, a retired review pass, AS a campaign.

    Failing here is deliberate. The old code would have run and produced a wrong table.
    """
    raise CameraTrapIngestNotRebuilt(
        "Camera-trap ingest is not implemented. It must be rebuilt from "
        "camera-traps/data/campaigns/<campaign>/observations.parquet, which already "
        "carries the resolved observationType, species, effort validity and repair "
        "provenance -- see camera-traps/docs/V2-REVIEW.md sections 2.3 and 2.4. "
        "The previous implementation was deleted on 2026-08-20 because it re-derived "
        "the review resolution and disagreed with the canonical table on 515 rows."
    )


def ingest_cr800_live(con: duckdb.DuckDBPyConnection) -> None:
    import yaml, os
    from dotenv import load_dotenv
    load_dotenv()
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["cr800"]

    host = os.getenv("CR800_HOST") or cfg["host"]
    port = int(os.getenv("CR800_PORT") or cfg["port"])
    addr = int(os.getenv("CR800_PAKBUS_ADDRESS") or cfg["pakbus_address"])
    station_id = cfg["station_id"]

    print(f"→ Connecting to CR800 at {host}:{port}...")
    try:
        from src.fetchers.cr800 import cr800_session, fetch_since
        total = 0
        first_chunk = True
        with cr800_session(host, port, addr) as logger:
            for df, commit in fetch_since(logger, station_id):
                if first_chunk:
                    ensure_columns(con, "weather_station", df)
                    first_chunk = False
                total += upsert_df(con, "weather_station", df)
                # State advances only after the upsert above succeeds.
                # If upsert raises, commit() never runs and the next run
                # replays this chunk (idempotent via PK upsert).
                commit()
        if total:
            print(f"  Upserted {total} rows into weather_station.")
        else:
            print("  No new CR800 data.")
    except Exception as e:
        print(f"  Warning: CR800 unavailable ({e}). Skipping.")


def ingest_cr800_range(con: duckdb.DuckDBPyConnection, start: str, end: str) -> None:
    import yaml, os
    from dotenv import load_dotenv
    load_dotenv()
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["cr800"]

    host = os.getenv("CR800_HOST") or cfg["host"]
    port = int(os.getenv("CR800_PORT") or cfg["port"])
    addr = int(os.getenv("CR800_PAKBUS_ADDRESS") or cfg["pakbus_address"])
    station_id = cfg["station_id"]

    print(f"→ Connecting to CR800 at {host}:{port} for range {start} → {end}...")
    from src.fetchers.cr800 import cr800_session, fetch_range
    total = 0
    first_chunk = True
    with cr800_session(host, port, addr) as logger:
        for df in fetch_range(logger, station_id, start, end):
            if first_chunk:
                ensure_columns(con, "weather_station", df)
                first_chunk = False
            total += upsert_df(con, "weather_station", df)
    print(f"  Upserted {total} rows into weather_station from range fetch.")


def ingest_cr800_backfill(con: duckdb.DuckDBPyConnection, dat_file_path: Path) -> None:
    from src.parsers.toa5 import parse
    import yaml
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        station_id = yaml.safe_load(f)["cr800"]["station_id"]
    print(f"→ Parsing TOA5 file: {dat_file_path}")
    df = parse(dat_file_path, station_id=station_id)
    ensure_columns(con, "weather_station", df)
    n = upsert_df(con, "weather_station", df)
    print(f"  Upserted {n} rows into weather_station from backfill.")


def export_weather_station(con: duckdb.DuckDBPyConnection) -> None:
    import yaml
    from src.exporters.csv_export import export_weather_station as _export
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["exports"]
    output_dir = Path(__file__).parent.parent / cfg["output_dir"]
    print("→ Exporting weather_station to CSV...")
    _export(con, output_dir)


def ingest_met_csv(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    from src.parsers.met_csv import parse
    import yaml
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        station_id = yaml.safe_load(f)["cr800"]["station_id"]
    df = parse(csv_path, station_id=station_id)
    ensure_columns(con, "weather_station", df)
    n = upsert_df(con, "weather_station", df)
    print(f"  Upserted {n} rows into weather_station from met CSV.")
