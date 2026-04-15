"""Orchestrator: routes data sources → parsers/fetchers → upsert into DuckDB."""

import duckdb
from pathlib import Path

from src.db import upsert_df, ensure_columns
from src.fetchers.open_meteo import fetch as fetch_open_meteo


def ingest_weather_forecast(con: duckdb.DuckDBPyConnection) -> None:
    df = fetch_open_meteo()
    n = upsert_df(con, "weather_forecast", df)
    print(f"  Upserted {n} rows into weather_forecast.")


def ingest_camera_trap_legacy(con: duckdb.DuckDBPyConnection, csv_path: Path = None) -> None:
    from src.parsers.camera_trap_legacy import parse
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "old animal data DB.csv"
    deployments, media, obs = parse(csv_path)
    nd = upsert_df(con, "ct_deployments", deployments)
    nm = upsert_df(con, "ct_media", media)
    no = upsert_df(con, "ct_observations", obs)
    print(f"  Legacy camera trap: {nd} deployments, {nm} media, {no} observations upserted.")


def ingest_camtrap_dp(con: duckdb.DuckDBPyConnection, folder_path: Path) -> None:
    from src.parsers.camtrap_dp import parse
    deployments, media, obs = parse(folder_path)
    nd = upsert_df(con, "ct_deployments", deployments)
    nm = upsert_df(con, "ct_media", media)
    no = upsert_df(con, "ct_observations", obs)
    print(f"  Camtrap DP: {nd} deployments, {nm} media, {no} observations upserted.")


def ingest_timelapse_reviewed(
    con: duckdb.DuckDBPyConnection,
    csv_path: Path,
    campaign_name: str,
) -> None:
    """Ingest a new_labeled_data_reviewed.csv from a CLIP+human-review campaign."""
    from src.parsers.timelapse_reviewed import parse
    dep_df, med_df, obs_df = parse(csv_path, campaign_name)
    # Add columns not in the base schema (campaign, observationComments, reviewOutcome)
    ensure_columns(con, "ct_deployments", dep_df)
    ensure_columns(con, "ct_observations", obs_df)
    nd = upsert_df(con, "ct_deployments", dep_df)
    nm = upsert_df(con, "ct_media", med_df)
    no = upsert_df(con, "ct_observations", obs_df)
    print(f"  {campaign_name}: {nd} deployments, {nm} media, {no} observations upserted.")


def ingest_all_ct_campaigns(con: duckdb.DuckDBPyConnection) -> None:
    """
    Ingest all camera trap campaigns configured in config.yaml under
    camera_traps.campaigns. Each entry needs csv (path) and name (campaign name).
    Paths are resolved relative to the data-pipeline repo root.
    """
    import yaml
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    campaigns = cfg.get("camera_traps", {}).get("campaigns", [])
    if not campaigns:
        print("  No camera_traps.campaigns entries in config.yaml — nothing to ingest.")
        return

    repo_root = Path(__file__).parent.parent
    for entry in campaigns:
        csv_path = (repo_root / entry["csv"]).resolve()
        campaign_name = entry["name"]
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found — skipping {campaign_name!r}")
            continue
        ingest_timelapse_reviewed(con, csv_path, campaign_name)


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
        from src.fetchers.cr800 import connect as cr800_connect, fetch_since
        logger = cr800_connect(host, port, addr)
        total = 0
        first_chunk = True
        for df in fetch_since(logger, station_id):
            if first_chunk:
                ensure_columns(con, "weather_station", df)
                first_chunk = False
            total += upsert_df(con, "weather_station", df)
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
    from src.fetchers.cr800 import connect as cr800_connect, fetch_range
    logger = cr800_connect(host, port, addr)
    total = 0
    first_chunk = True
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
