"""
setup/add_malfunction_columns.py — ONE-TIME: the record takes the form's new shape.

RUN ONCE, ON 2026-09-09, AND NEVER AGAIN

    `camtrap/visit_schema.py` gained three columns — `aim_intact`, `stop_reason` and
    `last_known_working` — so that the malfunction checks Silva-Rodríguez et al.
    (2025) put before classification are asked at the visit instead of surviving by
    luck in free text. `field_notes.csv` accumulates and predates them.

    This adds the three columns, blank, in the position the form declares, and
    changes nothing else. It is NOT an analytical change: no date moves, no
    coordinate changes, no deployment window changes, no parquet changes. Every
    legacy row keeps its `notes`, `data_flags` and `source_sheet` verbatim —
    including the two CT27 reconstructions that exist in no field sheet and cannot
    be regenerated.

WHY THE COLUMNS STAY BLANK, AND WHY THAT IS THE HONEST ANSWER

    The observations were never collected, so there is nothing to backfill. Two
    exist in prose and are deliberately NOT mined into the new columns here: a
    defective SD card recorded in a camera folder's name, and a camera found
    pointing upward, noted in comments. Reading a structured value out of a comment
    written for a human is exactly the kind of silent reinterpretation `visit_form`
    refuses at ingest; if those two are worth promoting, that is a curated edit with
    its reason written into `data_flags`, made by a person, not a regex run here.

IT REFUSES TO RUN TWICE

    The guard is code and not a comment because the cost of being wrong is the
    record. A file that already has the three columns is left untouched.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camtrap.visit_form import FIELD_NOTES_COLUMNS  # noqa: E402

NEW_COLUMNS = ('aim_intact', 'stop_reason', 'last_known_working')
RECORD = Path('data/campaigns/field_notes.csv')


def migrate(path: Path) -> int:
    with path.open(encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        before = tuple(reader.fieldnames or ())
        rows = list(reader)

    if before == FIELD_NOTES_COLUMNS:
        print(f'{path} ya tiene la forma del formulario. No se hace nada.')
        return 0

    expected_before = tuple(c for c in FIELD_NOTES_COLUMNS if c not in NEW_COLUMNS)
    if before != expected_before:
        raise SystemExit(
            f'{path} no tiene la forma previa esperada.\n'
            f'  esperaba: {expected_before}\n'
            f'  encontró: {before}\n'
            'Este script migra exactamente un cambio de forma y no adivina otro.')

    backup = path.with_suffix('.csv.pre-malfunction')
    shutil.copy2(path, backup)

    for row in rows:
        for column in NEW_COLUMNS:
            row[column] = ''

    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELD_NOTES_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    print(f'{path}: {len(rows)} filas, {len(before)} → {len(FIELD_NOTES_COLUMNS)} columnas')
    print(f'respaldo en {backup}')
    return len(rows)


if __name__ == '__main__':
    migrate(Path(sys.argv[1]) if len(sys.argv) > 1 else RECORD)
