import os
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

_config_path = Path(__file__).parent.parent / "config.yaml"
with open(_config_path) as f:
    _config = yaml.safe_load(f)

_schema_path = Path(__file__).parent.parent / "schema.sql"

# data-pipeline/src/db.py -> parents[2] = monorepo root. Derived from THIS FILE's location,
# so it is correct on any machine and any OS with no configuration at all.
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "fma_data.duckdb"


def db_path() -> Path:
    """Where the warehouse lives. Env var wins, then config, then repo-relative.

    THE REPO-RELATIVE DEFAULT IS THE POINT. Until 2026-08-24 `config.yaml` carried a
    committed absolute path (`/home/fguarda/Dev/Python/fma_data.duckdb`) — correct on
    exactly one machine, silently wrong everywhere else, and the same failure mode that
    made `campaign_dir` a required argument in camera-traps: a machine-specific value with
    a committed default goes stale, and a run against the wrong path looks completely
    normal. Every other cross-project path in this monorepo already resolves this way
    (`src/stations.py`, `src/canonical_gate.py`, `backend/paths.py`).

    `DB_PATH` still overrides, for containers and one-off runs.
    """
    override = os.getenv("DB_PATH") or (_config.get("database") or {}).get("path")
    return Path(override) if override else _DEFAULT_DB_PATH


def connect() -> duckdb.DuckDBPyConnection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    sql = _schema_path.read_text()
    con.execute(sql)


@contextmanager
def managed_conn(init: bool = True):
    """Open a DuckDB connection, optionally init the schema, and guarantee close on exit."""
    con = connect()
    if init:
        init_schema(con)
    try:
        yield con
    finally:
        con.close()


def ensure_columns(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    """Add any columns present in df that don't yet exist in the table."""
    existing = {row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()}

    def _sql_type(dtype) -> str:
        if pd.api.types.is_integer_dtype(dtype):
            return "BIGINT"
        if pd.api.types.is_float_dtype(dtype):
            return "DOUBLE"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMPTZ"
        return "TEXT"

    for col in df.columns:
        if col not in existing:
            sql_type = _sql_type(df[col].dtype)
            con.execute(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS "{col}" {sql_type}')
            print(f"  Added column {col} ({sql_type}) to {table}.")


def upsert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """INSERT OR REPLACE INTO {table} with explicit column names. Returns row count."""
    if df.empty:
        return 0
    con.register("_upsert_tmp", df)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM _upsert_tmp")
    con.unregister("_upsert_tmp")
    return len(df)
