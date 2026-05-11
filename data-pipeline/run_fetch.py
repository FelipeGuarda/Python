import os
import sys
import argparse
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.db import managed_conn
from src.ingest import (
    ingest_weather_forecast,
    ingest_cr800_live,
    ingest_cr800_backfill,
    ingest_cr800_range,
    ingest_met_csv,
    export_weather_station,
    ingest_all_ct_campaigns,
)


def run_once():
    """Connect, fetch, disconnect — so the DB lock is released between cycles."""
    with managed_conn() as con:
        try:
            ingest_weather_forecast(con)
        except Exception as e:
            print(f"  Warning: Open-Meteo fetch failed ({e}). Skipping.")
        try:
            ingest_cr800_live(con)
        except Exception as e:
            print(f"  Warning: CR800 fetch failed ({e}). Skipping.")


def run_backfill(path: Path):
    with managed_conn() as con:
        if path.suffix.lower() == ".dat":
            ingest_cr800_backfill(con, path)
        elif path.suffix.lower() == ".csv":
            ingest_met_csv(con, path)
        else:
            print(f"Unknown backfill file type: {path.suffix}. Expected .dat or .csv")


def run_fetch_range(start: str, end: str):
    with managed_conn() as con:
        ingest_cr800_range(con, start, end)


def run_ingest_ct():
    """Ingest all configured camera trap campaigns into DuckDB."""
    with managed_conn() as con:
        ingest_all_ct_campaigns(con)


def run_export():
    with managed_conn(init=False) as con:
        export_weather_station(con)


def run_health(verbose: bool = False):
    import yaml
    from src.exporters.csv_export import health_check
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["exports"]
    output_dir = Path(cfg["output_dir"])
    with managed_conn(init=False) as con:
        health_check(con, output_dir, verbose=verbose)


def scheduled_open_meteo():
    with managed_conn(init=False) as con:
        ingest_weather_forecast(con)


def scheduled_cr800():
    with managed_conn(init=False) as con:
        ingest_cr800_live(con)


def scheduled_export():
    with managed_conn(init=False) as con:
        export_weather_station(con)


def main():
    parser = argparse.ArgumentParser(description="FMA data pipeline fetch runner")
    parser.add_argument("--once", action="store_true", help="Run once and exit (no scheduler)")
    parser.add_argument("--backfill", metavar="FILE", help="Backfill weather_station from a TOA5 .dat or met .csv file")
    parser.add_argument("--fetch-range", nargs=2, metavar=("START", "END"),
                        help="Fetch a specific date range from the CR800 without touching state. "
                             "Dates: YYYY-MM-DD or YYYY-MM-DDTHH:MM. Example: --fetch-range 2026-03-04 2026-03-19")
    parser.add_argument("--export", action="store_true", help="Export weather_station to CSV and exit")
    parser.add_argument("--health", action="store_true", help="Print health report and exit")
    parser.add_argument("--verbose", action="store_true", help="Show gap details with --health")
    parser.add_argument("--ct", action="store_true",
                        help="Ingest camera trap campaigns from config.yaml camera_traps.campaigns")
    args = parser.parse_args()

    if args.ct:
        run_ingest_ct()
        return

    if args.backfill:
        run_backfill(Path(args.backfill))
        return

    if args.fetch_range:
        run_fetch_range(args.fetch_range[0], args.fetch_range[1])
        return

    if args.export:
        run_export()
        return

    if args.health:
        run_health(verbose=args.verbose)
        return

    if args.once:
        run_once()
        print("Done.")
        return

    # APScheduler daemon — each job opens/closes its own connection
    # so the DB is unlocked between fetch cycles and other processes
    # (e.g. the FastAPI backend) can read from it.
    from apscheduler.schedulers.blocking import BlockingScheduler

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["schedules"]

    scheduler = BlockingScheduler()
    scheduler.add_job(
        scheduled_open_meteo,
        "interval",
        minutes=cfg["open_meteo_interval_minutes"],
        id="open_meteo",
    )
    scheduler.add_job(
        scheduled_cr800,
        "interval",
        minutes=cfg["cr800_interval_minutes"],
        id="cr800",
    )
    # Monthly full snapshot: 1st of each month at 02:00 UTC
    scheduler.add_job(
        scheduled_export,
        "cron",
        day=1,
        hour=2,
        minute=0,
        id="csv_export",
    )

    print("→ Scheduler started. Open-Meteo every "
          f"{cfg['open_meteo_interval_minutes']} min, "
          f"CR800 every {cfg['cr800_interval_minutes']} min, "
          "CSV export on the 1st of each month.")
    print("  Press Ctrl+C to stop.")

    # Run fetch immediately on start; export only on schedule
    run_once()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n→ Scheduler stopped.")


if __name__ == "__main__":
    main()
