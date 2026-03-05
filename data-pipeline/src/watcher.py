"""Watchdog FileSystemEventHandler for data/incoming/ drop folder."""

import time
from pathlib import Path

import duckdb
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


def _detect_and_ingest(path: Path, con: duckdb.DuckDBPyConnection) -> None:
    """Detect file type and call appropriate ingest function."""
    from src.ingest import (
        ingest_camera_trap_legacy,
        ingest_camtrap_dp,
        ingest_cr800_backfill,
    )

    if path.is_dir():
        # Check if it's a Camtrap DP package
        if (path / "deployments.csv").exists():
            print(f"→ Detected Camtrap DP folder: {path}")
            ingest_camtrap_dp(con, path)
        else:
            print(f"  Skipping folder (no deployments.csv): {path}")
        return

    suffix = path.suffix.lower()

    if suffix == ".dat":
        print(f"→ Detected TOA5 .dat file: {path}")
        ingest_cr800_backfill(con, path)

    elif suffix == ".csv":
        print(f"→ Detected CSV file: {path}")
        # Try to detect if it's a camtrap DP piece or legacy format
        try:
            import pandas as pd
            header = pd.read_csv(path, nrows=0)
            cols = set(header.columns)
            if "RootFolder" in cols and "RelativePath" in cols:
                ingest_camera_trap_legacy(con, path)
            elif "deploymentID" in cols and "locationID" in cols:
                print("  Looks like a Camtrap DP CSV but needs full folder — skipping.")
            else:
                print(f"  Unknown CSV format, columns: {list(cols)[:8]}...")
        except Exception as e:
            print(f"  Could not parse CSV ({e}). Skipping.")

    else:
        print(f"  Ignoring unrecognized file type: {path.name}")


class IncomingHandler(FileSystemEventHandler):
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def on_created(self, event):
        path = Path(event.src_path)
        # Brief wait so file is fully written before parsing
        time.sleep(1)
        try:
            _detect_and_ingest(path, self.con)
        except Exception as e:
            print(f"  Error ingesting {path.name}: {e}")


def start_watcher(incoming_dir: Path, con: duckdb.DuckDBPyConnection) -> Observer:
    incoming_dir = Path(incoming_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    handler = IncomingHandler(con)
    observer = Observer()
    observer.schedule(handler, str(incoming_dir), recursive=True)
    observer.start()
    print(f"→ Watching {incoming_dir} for new files...")
    return observer
