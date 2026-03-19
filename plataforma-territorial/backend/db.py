import duckdb
from pathlib import Path

# Resolves to /home/fguarda/Dev/Python/fma_data.duckdb (3 parents up from this file)
_DEFAULT_DB = Path(__file__).parent.parent.parent / "fma_data.duckdb"


def get_con() -> duckdb.DuckDBPyConnection:
    import os
    path = os.getenv("FMA_DB_PATH", str(_DEFAULT_DB))
    return duckdb.connect(path, read_only=True)
