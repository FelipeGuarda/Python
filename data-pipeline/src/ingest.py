"""Orchestrator: routes data sources → parsers/fetchers → upsert into DuckDB."""

import duckdb
import pandas as pd
from pathlib import Path

from src import canonical_gate
from src.db import upsert_df, ensure_columns
from src.fetchers.open_meteo import fetch as fetch_open_meteo
from src.parsers import canonical_ct


def ingest_weather_forecast(con: duckdb.DuckDBPyConnection) -> None:
    df = fetch_open_meteo()
    n = upsert_df(con, "weather_forecast", df)
    print(f"  Upserted {n} rows into weather_forecast.")


class CameraTrapIngestError(RuntimeError):
    """The camera-trap rebuild could not be completed, or could not be trusted."""


#: data-pipeline/src/ingest.py -> parents[2] = monorepo root
_CAMPAIGNS_ROOT = (
    Path(__file__).resolve().parents[2] / "camera-traps" / "data" / "campaigns"
)


def _read_canonical(campaign: str, state: dict) -> pd.DataFrame:
    """One campaign's canonical table, checked against the contract before it is used."""
    path = _CAMPAIGNS_ROOT / campaign / "observations.parquet"
    if not path.exists():
        raise CameraTrapIngestError(
            f"{path} not found, but CANONICAL_STATE.json declares campaign "
            f"{campaign!r}. The contract and the files on disk disagree."
        )
    frame = pd.read_parquet(path)

    missing = [c for c in state["columns"] if c not in frame.columns]
    if missing:
        raise CameraTrapIngestError(
            f"{path} is missing canonical column(s) {missing}. It predates the "
            f"published contract — re-run camera-traps' ingest for this campaign."
        )
    declared = state["campaigns"][campaign]["n_rows"]
    if len(frame) != declared:
        raise CameraTrapIngestError(
            f"{campaign}: the parquet holds {len(frame)} rows and the contract declares "
            f"{declared}. Re-publish CANONICAL_STATE.json, or find out who rewrote the "
            f"parquet without publishing."
        )
    return frame


def ingest_all_ct_campaigns(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Rebuild ct_deployments / ct_media / ct_observations from the canonical parquets.

    REBUILT WHOLE, NEVER INCREMENTALLY. The canonical table is 35,807 rows and the
    parquets are the entire truth, so a full replace costs nothing and cannot strand a
    row from a campaign that shrank or was retired — which is how `pv_2025_2026` lived
    on in this database as a phantom campaign after being dropped upstream.

    The gate runs FIRST and the state is stamped LAST, so an interrupted run reports as
    stale rather than as finished.

    This replaces the implementation deleted on 2026-08-20, which re-derived five
    decisions `camtrap.observations` already owned and disagreed with it on 515 rows.
    Nothing here re-derives anything — see src/parsers/canonical_ct.py.
    """
    state = canonical_gate.load_published()
    campaigns = list(state["campaigns"])
    print(f"→ Canonical contract v{state['schema_version']}: "
          f"{len(campaigns)} campaign(s), {state['n_rows_total']} rows declared.")

    frames = {c: _read_canonical(c, state) for c in campaigns}
    tables = canonical_ct.to_tables(frames)

    written: dict[str, int] = {}
    for table, df in tables.items():
        con.execute(f"DELETE FROM {table}")
        ensure_columns(con, table, df)
        written[table] = upsert_df(con, table, df)
        print(f"  {table}: {written[table]} rows.")

    _reconcile(con, state)
    canonical_gate.record(con, state)
    return written


def _reconcile(con: duckdb.DuckDBPyConnection, state: dict) -> None:
    """V2-REVIEW 2.8: database row counts equal parquet row counts, per campaign.

    Read back from the DATABASE, not from the frames just built. The frames are what we
    meant to write; a reconciliation that trusts them checks nothing.
    """
    problems = []
    for campaign, declared in state["campaigns"].items():
        actual = con.execute(
            "SELECT COUNT(*) FROM ct_observations o "
            "JOIN ct_deployments d ON o.deploymentID = d.deploymentID "
            "WHERE d.campaign = ?", [campaign]
        ).fetchone()[0]
        if actual != declared["n_rows"]:
            problems.append(
                f"{campaign}: {actual} observation rows in the database, "
                f"{declared['n_rows']} declared"
            )
        stations = con.execute(
            "SELECT COUNT(DISTINCT locationName) FROM ct_deployments WHERE campaign = ?",
            [campaign]
        ).fetchone()[0]
        if stations != declared["n_stations"]:
            problems.append(
                f"{campaign}: {stations} stations in the database, "
                f"{declared['n_stations']} declared"
            )

    media = con.execute("SELECT COUNT(*) FROM ct_media").fetchone()[0]
    obs = con.execute("SELECT COUNT(*) FROM ct_observations").fetchone()[0]
    if media != obs:
        problems.append(
            f"ct_media has {media} rows and ct_observations {obs}; these are "
            f"media-level observations and must be 1:1"
        )

    if problems:
        raise CameraTrapIngestError(
            "Rebuild finished but did not reconcile:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
    print(f"  Reconciled: {obs} observations across {len(state['campaigns'])} campaigns.")


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
