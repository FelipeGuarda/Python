"""The tables that cannot be refetched, and the form in which they travel between machines.

WHAT THIS OWNS. Which tables are irreplaceable, and how they are carried across a machine
boundary without depending on anyone copying a database file correctly.

WHY IT EXISTS. `fma_data.duckdb` is gitignored and lives on one machine at a time, which
made "which box holds the real database" a standing question and the Windows->Linux
migration a blocker. The answer is a split by REGENERABILITY, not a file move:

    ct_deployments / ct_media / ct_observations   regenerable from observations.parquet,
                                                  which is in git on both machines
    weather_station / weather_forecast            NOT regenerable. A CR800 pull is a
                                                  point-in-time read of a datalogger, and
                                                  Open-Meteo serves a forecast horizon,
                                                  not an archive. Neither can be asked
                                                  again for a date that has passed.

So the camera-trap tables are rebuilt and never migrated, and the weather tables travel
as committed Parquet. Once that is in git the migration is finished, permanently: any
machine can reconstruct the database from the repository alone.

PARTITIONED BY YEAR, DELIBERATELY. Parquet is opaque to git, so a single 16.8 MB blob
would be stored again in full on every export -- and this runs whenever the station is
polled. Weather for a year that has ended never changes, so per-year files mean git only
ever stores a new blob for the current year.

EXPORT AND RESTORE ARE SEPARATE VERBS, and there is no `sync`. Guessing the direction on
irreplaceable data is how the empty copy overwrites the good one -- the exact hazard
V2-REVIEW 2.1 names when it says to compare row counts before deciding which copy to
keep. `restore` refuses by default to shrink a table; see `restore`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from src.db import ensure_columns, managed_conn, upsert_df

#: Tables that cannot be refetched for a date that has passed. Ordered for reporting.
IRREPLACEABLE: tuple[str, ...] = ("weather_station", "weather_forecast")

#: The column each table is partitioned on. Both are the reading's instant.
_PARTITION_COLUMN: dict[str, str] = {
    "weather_station": "timestamp",
    "weather_forecast": "timestamp",
}

_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "recovery"


class RecoveryError(RuntimeError):
    """A recovery operation would lose data, or cannot prove that it would not."""


def _table_dir(root: Path, table: str) -> Path:
    return root / table


def _year_files(root: Path, table: str) -> list[Path]:
    return sorted(_table_dir(root, table).glob("*.parquet"))


def export(con: duckdb.DuckDBPyConnection, root: Path | None = None) -> dict[str, list[Path]]:
    """Write each irreplaceable table to one Parquet file per calendar year.

    Rewrites only the years whose contents changed, so a poll that adds today's rows
    touches exactly one file. Years that vanished from the table are NOT deleted --
    see the guard below; a table that shrank is a symptom, not an instruction.
    """
    root = root or _DEFAULT_ROOT
    written: dict[str, list[Path]] = {}

    for table in IRREPLACEABLE:
        col = _PARTITION_COLUMN[table]
        out_dir = _table_dir(root, table)
        out_dir.mkdir(parents=True, exist_ok=True)

        years = [
            r[0] for r in con.execute(
                f'SELECT DISTINCT year("{col}") AS y FROM {table} '
                f'WHERE "{col}" IS NOT NULL ORDER BY y'
            ).fetchall()
        ]
        paths = []
        for year in years:
            path = out_dir / f"{year}.parquet"
            # zstd over snappy: this is archival data read rarely and stored in git,
            # so size on disk matters more than decompression speed.
            con.execute(
                f'COPY (SELECT * FROM {table} WHERE year("{col}") = {int(year)} '
                f'ORDER BY "{col}") TO \'{path}\' (FORMAT PARQUET, COMPRESSION ZSTD)'
            )
            paths.append(path)

        orphaned = set(_year_files(root, table)) - set(paths)
        if orphaned:
            raise RecoveryError(
                f"{table}: {sorted(p.name for p in orphaned)} exist on disk but the "
                f"table has no rows for those years. Refusing to export -- the archive "
                f"holds data the database has lost. Restore first, or delete the files "
                f"deliberately if the years were genuinely dropped."
            )
        written[table] = paths

    return written


def restore(
    con: duckdb.DuckDBPyConnection,
    root: Path | None = None,
    allow_shrink: bool = False,
) -> dict[str, int]:
    """Load the committed Parquet back into the database. Returns rows written per table.

    Upserts on the primary key, so running it against a populated database is safe and
    idempotent: it fills gaps and never duplicates.

    REFUSES BY DEFAULT WHEN THE ARCHIVE IS SMALLER THAN THE TABLE. That means this
    machine holds readings the archive has never seen, and restoring would be the
    wrong direction -- export first. `allow_shrink` is for deliberately rewinding to
    the archive, which is not a thing to do by accident.
    """
    root = root or _DEFAULT_ROOT
    written: dict[str, int] = {}

    for table in IRREPLACEABLE:
        files = _year_files(root, table)
        if not files:
            raise RecoveryError(
                f"{table}: no Parquet under {_table_dir(root, table)}. Nothing to "
                f"restore -- run `export` on the machine that holds the data."
            )

        frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        existing = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if existing > len(frame) and not allow_shrink:
            raise RecoveryError(
                f"{table}: the database has {existing} rows and the archive has "
                f"{len(frame)}. This machine holds readings the archive does not. "
                f"Run `export` here instead, or pass allow_shrink to overrule."
            )

        ensure_columns(con, table, frame)
        written[table] = upsert_df(con, table, frame)

    return written


def verify(con: duckdb.DuckDBPyConnection, root: Path | None = None) -> list[str]:
    """Discrepancies between database and archive. Empty list means they agree.

    Returns findings rather than raising: the caller is a report, and a caller that
    wants to abort can check for truth. Every finding names both counts, because
    "they differ" without the direction does not tell you which way to fix it.
    """
    root = root or _DEFAULT_ROOT
    findings: list[str] = []

    for table in IRREPLACEABLE:
        files = _year_files(root, table)
        if not files:
            findings.append(f"{table}: no archive under {_table_dir(root, table)}")
            continue
        archived = sum(
            con.execute(f"SELECT COUNT(*) FROM read_parquet('{p}')").fetchone()[0]
            for p in files
        )
        live = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if archived != live:
            findings.append(
                f"{table}: database {live} rows, archive {archived} "
                f"({'export' if live > archived else 'restore'} to reconcile)"
            )

    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export/restore the irreplaceable tables.")
    ap.add_argument("action", choices=("export", "restore", "verify"))
    ap.add_argument("--allow-shrink", action="store_true",
                    help="restore even when the database has more rows than the archive")
    args = ap.parse_args(argv)

    with managed_conn() as con:
        if args.action == "export":
            for table, paths in export(con).items():
                total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                size = sum(p.stat().st_size for p in paths) / 1e6
                print(f"{table}: {total} rows -> {len(paths)} file(s), {size:.2f} MB")
        elif args.action == "restore":
            for table, n in restore(con, allow_shrink=args.allow_shrink).items():
                print(f"{table}: restored {n} rows")
        else:
            findings = verify(con)
            if not findings:
                print("Database and archive agree.")
                return 0
            for f in findings:
                print(f"DRIFT: {f}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
