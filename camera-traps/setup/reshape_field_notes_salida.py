"""
setup/reshape_field_notes_salida.py — ONE-TIME: the record stores the salida.

RUN ONCE, ON 2026-11-03, AND NEVER AGAIN

    Visit form 2.0 asks for the salida a visit belongs to ("Campaña de mantención")
    instead of the campaign it opens, because the team knows the first at the tree
    and found the second confusing (it named a salida that had not happened yet).
    `field_notes.csv` follows: `campaign_opened` is dropped, `maintenance_campaign`
    takes its place, and `anchors._derive_campaigns` derives both the campaign a
    visit closes and the one it opens. Two witness-photo columns are added, blank on
    every legacy row because they were never collected.

    THIS IS NOT AN ANALYTICAL CHANGE, and the script proves it before writing: the
    converted file is read back through the one live implementation
    (`FieldRecord`), and every row must open and close exactly the campaigns the old
    file did. One mismatch and nothing is written.

WHY THE OLD RULE IS RESTATED HERE

    `_legacy_closings` is a frozen copy of the derivation retired today. The live
    `FieldRecord` refuses the old shape by design, so the old file cannot be read
    through it; this copy exists only to state what the old file meant, and dies
    with this script.

HOW THE SALIDA IS RECOVERED

    Campaigns are named for the salida that retrieves their cards, so a visit that
    closed campaign X belongs to salida X. Visits that close nothing (the 2024-25
    installs, CT27's undated placeholder) belonged to no retrieval salida; they keep
    a blank. A blank there is history, not a gap: the 2.0 form requires the field.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camtrap import visit_schema
from camtrap.anchors import FIELD_NOTES_FILENAME, FieldRecord, RECORDED_OPEN_COLUMN
from camtrap.visit_form import FIELD_NOTES_COLUMNS

REPO = Path(__file__).resolve().parents[1]
CAMPAIGNS = REPO / 'data' / 'campaigns'
LIVE_CSV = CAMPAIGNS / FIELD_NOTES_FILENAME
SNAPSHOT = CAMPAIGNS / 'legacy' / 'field_notes (campaign_opened, hasta 2026-11-03).csv'


def _legacy_closings(df: pd.DataFrame) -> pd.Series:
    """The retired rule: a visit closes what the previous card-leaving visit opened."""
    closed = pd.Series('', index=df.index)
    dated = df[df['visit_date'] != '']
    for _, rows in dated.groupby('station_id'):
        in_ground = ''
        for i in rows.sort_values('visit_date', kind='stable').index:
            kind = df.at[i, 'visit_type']
            if kind in visit_schema.CLOSES_CAMPAIGN:
                closed.at[i] = in_ground
            if kind in visit_schema.OPENS_CAMPAIGN:
                in_ground = df.at[i, RECORDED_OPEN_COLUMN]
            elif kind in visit_schema.CLOSES_CAMPAIGN:
                in_ground = ''
    return closed


def _converted(df: pd.DataFrame, closed: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame('', index=df.index, columns=list(FIELD_NOTES_COLUMNS))
    for column in FIELD_NOTES_COLUMNS:
        if column in df.columns:
            out[column] = df[column]
    out['maintenance_campaign'] = closed
    return out


def _verify(df: pd.DataFrame, closed: pd.Series, out: pd.DataFrame, tmp: Path) -> None:
    """Refuse to write unless every opening and closing survives exactly."""
    assert not (df['visit_type'] == 'mantencion').any(), \
        'a row uses `mantencion`; rename it to `inspeccion` deliberately first'
    kept = [c for c in df.columns if c != RECORDED_OPEN_COLUMN]
    for column in kept:
        assert out[column].equals(df[column]), f'{column} was not carried verbatim'

    out.to_csv(tmp, index=False, encoding='utf-8')
    visits = FieldRecord.load(tmp)._visits
    assert len(visits) == len(df), 'FieldRecord dropped or reordered rows'
    problems = []
    for position, v in enumerate(visits):
        i = df.index[position]
        was_open, was_closed = df.at[i, RECORDED_OPEN_COLUMN], closed.at[i]
        if v.campaign_opened != was_open or v.campaign_closed != was_closed:
            problems.append(
                f'{v.station_id} {df.at[i, "visit_date"] or "(undated)"} '
                f'{v.visit_type}: opened {was_open!r} -> {v.campaign_opened!r}, '
                f'closed {was_closed!r} -> {v.campaign_closed!r}')
    if problems:
        raise SystemExit('Refusing to write; the derivation would change:\n  '
                         + '\n  '.join(problems))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--dry-run', action='store_true',
                    help='verify and report; write nothing')
    args = ap.parse_args(argv)

    df = pd.read_csv(LIVE_CSV, dtype=str, keep_default_na=False)
    if RECORDED_OPEN_COLUMN not in df.columns:
        print(f'{LIVE_CSV.name} already stores the salida. Nothing to do; '
              'this script runs once.')
        return 1
    if SNAPSHOT.exists() and not args.dry_run:
        print(f'{SNAPSHOT} already exists — this script has run before. Refusing.')
        return 1

    closed = _legacy_closings(df)
    out = _converted(df, closed)
    tmp = LIVE_CSV.with_suffix('.reshape-check.csv')
    try:
        _verify(df, closed, out, tmp)
    finally:
        tmp.unlink(missing_ok=True)

    print(f'{len(df)} row(s): every opening and closing reproduced exactly.')
    print(f'  dropped : {RECORDED_OPEN_COLUMN}')
    print(f'  added   : {", ".join(c for c in FIELD_NOTES_COLUMNS if c not in df.columns)}')
    print(f'  salida recovered on {int((closed != "").sum())} row(s); '
          f'{int((closed == "").sum())} closed nothing and stay blank')
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
