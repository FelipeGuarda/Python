"""
Database connection for the platform backend.
Connects read-only to the shared fma_data.duckdb written by data-pipeline.
"""

import os
import time
from contextlib import contextmanager
from pathlib import Path

import duckdb

DB_PATH = os.getenv(
    "FMA_DB_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "fma_data.duckdb"),
)

_OPEN_BACKOFF = (0.05, 0.15, 0.5)  # seconds; 4 attempts total


@contextmanager
def get_connection():
    """Yield a short-lived read-only DuckDB connection.

    The data-pipeline daemon writes to this DB on a schedule (every 60 min).
    Between writes the lock is released, so read-only connections work.
    If a write lock is held, retries with exponential back-off
    (50 ms → 150 ms → 500 ms) before raising IOException.
    """
    for delay in _OPEN_BACKOFF:
        try:
            con = duckdb.connect(DB_PATH, read_only=True)
            break
        except duckdb.IOException:
            time.sleep(delay)
    else:
        # All retries exhausted — final attempt, let any exception propagate.
        con = duckdb.connect(DB_PATH, read_only=True)
    try:
        yield con
    finally:
        con.close()
