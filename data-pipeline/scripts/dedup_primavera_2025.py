"""
dedup_primavera_2025.py — build a de-duplicated Primavera 2025 campaign CSV.

Why this exists
---------------
The `primavera_2025` campaign was never ingested into DuckDB (it was missing
from config.yaml). When we went to add it we found two problems:

1. `pv_2025_2026` is a *partial re-pull* of the same SD cards: 396 of
   Primavera 2025's images are physically identical to records already in
   DuckDB (matched on File + DateTime). `mediaID`/`observationID` are
   per-campaign UUIDs with zero cross-campaign overlap, so the INSERT OR
   REPLACE upsert would NOT collapse them — it would double-insert.
2. The `100EK113` deployment folder does not map to a CT number. We have
   evidence it is physically CT5, but that is not yet confirmed from the
   photos, so for now we EXCLUDE all 100EK113 records (consistent with the
   annual report's "despliegues no mapeables descartados" rule). They can be
   re-added later once the camera is verified.

What it does
------------
Reads the raw Primavera 2025 CSV and writes `new_labeled_data_reviewed.dedup.csv`
containing only rows that are (a) not already present in the otoño/pv CSVs
(matched on File + DateTime) and (b) not on deployment 100EK113. Row content is
copied verbatim (string-level) so the existing timelapse_reviewed parser reads
it identically.

Run: python scripts/dedup_primavera_2025.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parent.parent.parent  # …/Dev/Python
CAMPAIGNS = REPO / "camera-traps" / "data" / "campaigns"

OTONO = CAMPAIGNS / "otono_2025" / "new_labeled_data_reviewed.csv"
PV = CAMPAIGNS / "pv_2025_2026" / "new_labeled_data_reviewed.csv"
PRIMAVERA = CAMPAIGNS / "primavera_2025" / "new_labeled_data_reviewed.csv"
OUT = CAMPAIGNS / "primavera_2025" / "new_labeled_data_reviewed.dedup.csv"

EXCLUDE_DEPLOYMENTS = {"100EK113"}


def keep_media_ids() -> set[str]:
    """mediaIDs to retain: not already in otoño/pv (by File+DateTime), not excluded deployment."""
    con = duckdb.connect()
    rows = con.execute(
        """
        WITH prim AS (SELECT * FROM read_csv_auto(?, sample_size=-1)),
             existing AS (
                 SELECT "File" f, DateTime dt FROM read_csv_auto(?, sample_size=-1)
                 UNION
                 SELECT "File" f, DateTime dt FROM read_csv_auto(?, sample_size=-1)
             )
        SELECT prim.mediaID
        FROM prim
        WHERE prim."Deployments" NOT IN ('100EK113')
          AND NOT EXISTS (
              SELECT 1 FROM existing e
              WHERE e.f = prim."File" AND e.dt = prim.DateTime
          )
        """,
        [str(PRIMAVERA), str(OTONO), str(PV)],
    ).fetchall()
    con.close()
    return {r[0] for r in rows}


def main() -> None:
    keep = keep_media_ids()

    with open(PRIMAVERA, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    kept = [r for r in all_rows if r.get("mediaID", "").strip() in keep]
    excluded_100ek = sum(1 for r in all_rows if r.get("Deployments", "").strip() in EXCLUDE_DEPLOYMENTS)

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Source rows           : {len(all_rows)}")
    print(f"  on 100EK113 (excl.) : {excluded_100ek}")
    print(f"  duplicate of pv/otoño: {len(all_rows) - excluded_100ek - len(kept)}")
    print(f"Kept (written)        : {len(kept)}")
    print(f"Wrote → {OUT}")


if __name__ == "__main__":
    main()
