"""
setup/reshape_field_notes.py — ONE-TIME: field_notes.csv takes the form's shape.

RUN ONCE, ON 2026-08-26, AND NEVER AGAIN

    `setup/build_visit_template.py` renders the visit form from
    `camtrap/visit_schema.py`, and `camtrap/visit_form.py` reads a filled one back.
    Until this script ran, the form and its destination disagreed: the workbook
    collects 20 columns, `field_notes.csv` held 28, and five of the form's columns
    had no home at all. This converts the record, once, so that a loaded visit and a
    legacy visit are the same kind of row.

    It refuses to run twice. Re-running it is not merely redundant — the file it
    reads no longer exists in the shape it expects — and the guard is here rather
    than in a comment because the cost of being wrong is the 107 rows below.

    THIS IS NOT AN ANALYTICAL CHANGE. Every legacy row keeps its dates, its
    coordinates, its notes and its flags; the five new columns are blank in all 107
    because that information was never collected. No deployment window moves, no
    anchor changes, no parquet changes. What it buys is that the NEXT salida lands in
    the same file without a translation layer.

WHAT IS DROPPED, AND WHY IT IS SAFE TO DROP

    campaign_closed                 derived from the visit sequence — anchors.py
    clock_state, camera_replaced    loaded onto `Visit` and never read by anything
    clock_action, clock_offset_hours  verdicts the new form refuses to collect
    sd_out, sd_in                   the `M##` grid tag, dropped 2026-08-24
    waypoint, gps_device            GPS bookkeeping nothing reads
    grid_id, elevation_m            estaciones.csv owns station identity

    `notes`, `data_flags` and `source_sheet` are carried VERBATIM and asserted
    identical before anything is written: they hold the whole curated corpus — 17
    distinct flag types over 58 rows — including the two CT27 reconstructions that
    exist in no field sheet and cannot be regenerated from the workbook.

THE ONE VALUE THIS DELETES

    CT27's 2025-12-11 row recorded `campaign_closed=primavera_2025`. CT27 has no
    primavera deployment in `deployments.csv`, none in the canonical table, and no
    prior visit for that campaign to have opened. Felipe's call, 2026-08-26: the
    assertion is wrong, delete it. Where the dropped value and the derivation
    disagree, the row says so in `data_flags` — deleting the reason is how the
    mistake comes back.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap.anchors import FIELD_NOTES_FILENAME, FieldRecord, RECORDED_CLOSE_COLUMN
from camtrap.visit_form import FIELD_NOTES_COLUMNS

REPO = Path(__file__).resolve().parents[1]
CAMPAIGNS = REPO / 'data' / 'campaigns'
LIVE_CSV = CAMPAIGNS / FIELD_NOTES_FILENAME
SNAPSHOT = CAMPAIGNS / 'legacy' / 'field_notes (HISTORICO 2024-2026 - NO LLENAR).csv'

#: Columns carried across untouched. The curation lives here.
VERBATIM = ('notes', 'data_flags', 'source_sheet')

#: The legacy vocabulary the form renamed. `unrecorded` has no form equivalent: it
#: marks a visit the workbook never recorded (CT27's install) and stays as it is.
VISIT_TYPE_RENAMES = {'install': 'instalacion'}


def _reshaped(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """The legacy frame in the new shape, plus a mask of the rows renamed to `CAM-`."""
    out = pd.DataFrame('', index=df.index, columns=list(FIELD_NOTES_COLUMNS))
    for column in FIELD_NOTES_COLUMNS:
        if column in df.columns:
            out[column] = df[column].fillna('').astype(str).str.strip()

    out['visit_type'] = out['visit_type'].replace(VISIT_TYPE_RENAMES)

    # The form writes HH:MM; the legacy migration wrote HH:MM:SS. Truncating is
    # lossless only if no visit was ever recorded to the second, so it is checked.
    seconds = out['visit_time'].str.extract(r':(\d\d)$')[0].dropna()
    assert set(seconds) <= {'00'}, f'a visit time carries seconds: {sorted(set(seconds))}'
    out['visit_time'] = out['visit_time'].str.replace(r'^(\d\d:\d\d):00$', r'\1', regex=True)

    # `CAM-` is the prefix the form requires, and these are the two rows it was
    # designed for: May 2026, when CT23 and CT18 each received a different body.
    unit = out['camera_unit_id'].str.strip()
    renamed = unit.str.fullmatch(r'\d+')
    out.loc[renamed, 'camera_unit_id'] = 'CAM-' + unit[renamed]
    return out, renamed


def _note_dropped_closings(out: pd.DataFrame, recorded: pd.Series,
                           tmp: Path) -> list[str]:
    """Flag every row where the dropped value and the derivation disagree.

    The derivation is not reimplemented here: the reshaped rows are written out and
    read back through `FieldRecord`, so this compares the legacy file against the
    one implementation that will serve every future reader.
    """
    out.to_csv(tmp, index=False, encoding='utf-8')
    visits = FieldRecord.load(tmp)._visits
    assert len(visits) == len(out), 'FieldRecord dropped or reordered rows'

    messages = []
    for position, (index, visit) in enumerate(zip(out.index, visits)):
        was = recorded.iloc[position].strip()
        now = visit.campaign_closed.strip()
        if was == now:
            continue
        note = (f'campaign_closed_dropped 2026-08-26: the legacy file recorded '
                f'{was or "nothing"!r} for this visit; derived from '
                f'{visit.station_id}\'s own visit sequence it closes '
                f'{now or "nothing"}. Felipe 2026-08-26: the recorded value is '
                f'wrong — the station has no such deployment in deployments.csv, '
                f'none in the canonical table, and no earlier visit to have opened '
                f'it. Kept here because deleting the reason is how it returns.')
        existing = out.at[index, 'data_flags'].strip()
        out.at[index, 'data_flags'] = f'{existing}; {note}' if existing else note
        messages.append(f'{visit.station_id} {visit.visit_date:%Y-%m-%d}: '
                        f'{was or "(blank)"} -> {now or "(nothing)"}')
    return messages


def _verify(before: pd.DataFrame, after: pd.DataFrame, flagged: int) -> None:
    """Refuse to write unless the curation survived exactly."""
    assert len(after) == len(before), f'{len(before)} rows in, {len(after)} out'
    for column in VERBATIM:
        left = before[column].fillna('').astype(str).str.strip()
        right = after[column].astype(str).str.strip()
        if column == 'data_flags':
            differing = (left != right).sum()
            assert differing == flagged, (
                f'data_flags changed on {differing} row(s), expected {flagged}')
            assert all(right[left != right].str.contains('campaign_closed_dropped')), \
                'a data_flags value changed for a reason other than the dropped closing'
            continue
        assert left.equals(right), f'{column} was not carried verbatim'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would change; write nothing')
    args = ap.parse_args(argv)

    df = pd.read_csv(LIVE_CSV, dtype=str, keep_default_na=False)
    if RECORDED_CLOSE_COLUMN not in df.columns:
        print(f'{LIVE_CSV} is already in the new shape '
              f'({len(df.columns)} columns). Nothing to do; this script runs once.')
        return 1
    if SNAPSHOT.exists() and not args.dry_run:
        print(f'{SNAPSHOT} already exists — this script has run before. Refusing.')
        return 1

    out, renamed = _reshaped(df)
    tmp = LIVE_CSV.with_suffix('.reshape-check.csv')
    try:
        dropped = _note_dropped_closings(out, df[RECORDED_CLOSE_COLUMN], tmp)
    finally:
        tmp.unlink(missing_ok=True)
    _verify(df, out, len(dropped))

    print(f'{len(df)} row(s), {len(df.columns)} columns -> {len(out.columns)} columns')
    print(f'  dropped : {", ".join(c for c in df.columns if c not in FIELD_NOTES_COLUMNS)}')
    print(f'  added   : {", ".join(c for c in FIELD_NOTES_COLUMNS if c not in df.columns)}')
    print(f'  camera_unit_id given the CAM- prefix on {int(renamed.sum())} row(s)')
    for line in dropped:
        print(f'  closing dropped and flagged: {line}')
    if args.dry_run:
        print('\n--dry-run: nothing written.')
        return 0

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LIVE_CSV, SNAPSHOT)
    out.to_csv(LIVE_CSV, index=False, encoding='utf-8')
    print(f'\nsnapshot: {SNAPSHOT.relative_to(REPO)}')
    print(f'rewrote : {LIVE_CSV.relative_to(REPO)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
