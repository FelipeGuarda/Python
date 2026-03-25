"""
Database connection for the platform backend.
Connects read-only to the shared fma_data.duckdb written by data-pipeline.
"""

import os
from contextlib import contextmanager
from pathlib import Path

import duckdb

DB_PATH = os.getenv(
    "FMA_DB_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "fma_data.duckdb"),
)


@contextmanager
def get_connection():
    """Yield a short-lived read-only DuckDB connection.

    The data-pipeline daemon writes to this DB on a schedule (every 60 min).
    Between writes the lock is released, so read-only connections work.
    If the pipeline is mid-write, this will briefly block then succeed.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        yield con
    finally:
        con.close()
