import os
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


def connect() -> duckdb.DuckDBPyConnection:
    db_path = os.getenv("DB_PATH") or _config["database"]["path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(db_path)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    sql = _schema_path.read_text()
    con.execute(sql)


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
