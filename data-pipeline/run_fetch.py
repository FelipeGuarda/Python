import os
import sys
import argparse
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.db import connect, init_schema
from src.ingest import (
    ingest_weather_forecast,
    ingest_cr800_live,
    ingest_cr800_backfill,
    ingest_met_csv,
)


def run_once():
    """Connect, fetch, disconnect — so the DB lock is released between cycles."""
    con = connect()
    init_schema(con)
    try:
        ingest_weather_forecast(con)
        ingest_cr800_live(con)
    finally:
        con.close()


def run_backfill(path: Path):
    con = connect()
    init_schema(con)
    try:
        if path.suffix.lower() == ".dat":
            ingest_cr800_backfill(con, path)
        elif path.suffix.lower() == ".csv":
            ingest_met_csv(con, path)
        else:
            print(f"Unknown backfill file type: {path.suffix}. Expected .dat or .csv")
    finally:
        con.close()


def scheduled_open_meteo():
    con = connect()
    try:
        ingest_weather_forecast(con)
    finally:
        con.close()


def scheduled_cr800():
    con = connect()
    try:
        ingest_cr800_live(con)
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description="FMA data pipeline fetch runner")
    parser.add_argument("--once", action="store_true", help="Run once and exit (no scheduler)")
    parser.add_argument("--backfill", metavar="DAT_FILE", help="Backfill weather_station from a TOA5 .dat file")
    args = parser.parse_args()

    if args.backfill:
        run_backfill(Path(args.backfill))
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

    print("→ Scheduler started. Open-Meteo every "
          f"{cfg['open_meteo_interval_minutes']} min, "
          f"CR800 every {cfg['cr800_interval_minutes']} min.")
    print("  Press Ctrl+C to stop.")

    # Run immediately on start
    run_once()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n→ Scheduler stopped.")


if __name__ == "__main__":
    main()
