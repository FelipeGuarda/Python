"""Watchdog FileSystemEventHandler for data/incoming/ drop folder."""

import time
from pathlib import Path
from typing import Callable

import duckdb
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


def _detect_and_ingest(path: Path, connect_fn: Callable[[], duckdb.DuckDBPyConnection]) -> None:
    """Detect file type and call appropriate ingest function. Opens its own DB connection."""
    from src.ingest import (
        ingest_camtrap_dp,
        ingest_cr800_backfill,
    )

    if path.is_dir():
        if not (path / "deployments.csv").exists():
            print(f"  Skipping folder (no deployments.csv): {path}")
            return
        print(f"→ Detected Camtrap DP folder: {path}")
        con = connect_fn()
        try:
            ingest_camtrap_dp(con, path)
        finally:
            con.close()
        return

    suffix = path.suffix.lower()

    if suffix == ".dat":
        print(f"→ Detected TOA5 .dat file: {path}")
        con = connect_fn()
        try:
            ingest_cr800_backfill(con, path)
        finally:
            con.close()

    elif suffix == ".csv":
        print(f"→ Detected CSV file: {path}")
        try:
            import pandas as pd
            cols = set(pd.read_csv(path, nrows=0).columns)
            if "deploymentID" in cols and "locationID" in cols:
                print("  Looks like a Camtrap DP CSV but needs full folder — skipping.")
            else:
                print(f"  Unknown CSV format, columns: {list(cols)[:8]}...")
        except Exception as e:
            print(f"  Could not parse CSV ({e}). Skipping.")

    else:
        print(f"  Ignoring unrecognized file type: {path.name}")


class IncomingHandler(FileSystemEventHandler):
    def __init__(self, connect_fn: Callable[[], duckdb.DuckDBPyConnection]):
        self.connect_fn = connect_fn

    def on_created(self, event):
        path = Path(event.src_path)
        # Brief wait so file is fully written before parsing
        time.sleep(1)
        try:
            _detect_and_ingest(path, self.connect_fn)
        except Exception as e:
            print(f"  Error ingesting {path.name}: {e}")


def start_watcher(incoming_dir: Path, connect_fn: Callable[[], duckdb.DuckDBPyConnection]) -> Observer:
    incoming_dir = Path(incoming_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    handler = IncomingHandler(connect_fn)
    observer = Observer()
    observer.schedule(handler, str(incoming_dir), recursive=True)
    observer.start()
    print(f"→ Watching {incoming_dir} for new files...")
    return observer
