"""
propose_anchors.py — turn the field visit record into reviewable anchor rows.

An anchor pairs a wall-clock moment with what the camera's clock said at that
moment. `field_notes.csv` holds the first half for every visit; `anchor_candidates.csv`
holds the second half for every frame worth opening. This script joins them and says,
per segment, whether the pair can be made — and refuses, in writing, where it cannot.

WHAT IT WILL NOT DO

    It will not propose an anchor for a camera whose clock is fine. The notebook
    records a visit DATE, not a clock reading, so forcing it onto a coherent camera
    would apply the notebook's imprecision as an offset to correct data. CT01 is the
    worked example: notebook 2025-11-24 → 2026-05-13, frames 2025-11-26 → 2026-05-14,
    one coherent segment, no reset. `clocks.repair_plan` already returns `clock_clean`
    for it, which is the right answer and needs no help from here.

    It will not guess. A segment it cannot pair becomes an `unrepairable_pending` row
    rather than no row at all, so promoting the file refuses the station explicitly.
    A station missing from the anchor file and a station known to be unanchorable look
    identical downstream, and only one of them is a decision anybody made.

USAGE
    python propose_anchors.py --campaign otono_2026
    python propose_anchors.py --campaign otono_2026 --write

OUTPUT
    data/campaigns/<name>/anchor_proposals.csv   every proposal with its status
    Nothing is promoted into deployment_anchors.csv automatically. Review the file,
    open the frames it names, then move the rows you accept across by hand — the
    anchor file is the one place a human signature still means something.

    `--write` appends only READY rows, and refuses to touch a station that already
    has an anchor on file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Force UTF-8 on stdout/stderr so accented species names and box-drawing characters
# do not raise UnicodeEncodeError on a default Windows console (cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from camtrap import anchors, exports, stations
from camtrap.anchors import (
    ANCHOR_FILENAME,
    ANCHOR_WRITE_COLUMNS,
    FIELD_NOTES_FILENAME,
    NEEDS_REVIEW,
    NOT_NEEDED,
    READY,
    FieldRecord,
    load_anchors,
)
from timestamps import diagnose_campaign, prepare_total

OUTPUT_FILENAME  = 'anchor_proposals.csv'
CANDIDATES_FILENAME = 'anchor_candidates.csv'


def load_candidates(path: Path) -> pd.DataFrame:
    """The anchor_candidates report, with its two datetime columns parsed.

    Empty rather than fatal when absent: the proposer can still say WHICH segments
    need an anchor and why, which is most of the value when no candidates exist yet.
    """
    cols = ['station', 'camera_num', 'file_name', 'camera_datetime',
            'candidate_kind', 'clock_segment']
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    df['camera_datetime'] = pd.to_datetime(df['camera_datetime'], errors='coerce')
    df['clock_segment'] = pd.to_numeric(df['clock_segment'], errors='coerce')
    return df


def render(df: pd.DataFrame) -> str:
    if df.empty:
        return 'No stations diagnosed.'
    lines: list[str] = []
    for status in (READY, NEEDS_REVIEW, NOT_NEEDED):
        rows = df[df['status'] == status]
        if rows.empty:
            continue
        lines.append(f'\n{status} — {len(rows)} row(s)')
        if status == NOT_NEEDED:
            # One line for the whole group: naming 26 clean stations individually
            # buries the handful that need attention. The unverified ones are called
            # out separately — their clean verdict rests on a test that never ran.
            checked = rows[rows['evidence'] == anchors.VERIFIED]
            lines.append(f'    verified against a deployment window: '
                         f'{", ".join(sorted(checked["station_id"]))}')
            blind = rows[rows['evidence'] == anchors.UNVERIFIED]
            if len(blind):
                lines.append(
                    f'    ⚠ NO deployment window, so a forward jump would be '
                    f'invisible: {", ".join(sorted(blind["station_id"]))}')
                lines.append(f'        {blind.iloc[0]["why"]}')
            continue
        for _, r in rows.iterrows():
            seg = '' if r['segment_index'] == '' else f' seg {r["segment_index"]}'
            lines.append(f'    {r["station_id"]}{seg}  [{r["anchor_type"]}]'
                         + (f'  {r["evidence"]}' if r['evidence'] else ''))
            lines.append(f'        {r["why"]}')
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='Propose deployment_anchors.csv rows from the field visit '
                    'record, and refuse in writing where none can be made.')
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--data-root', default='data/campaigns')
    ap.add_argument('--write', action='store_true',
                    help='Append READY rows to deployment_anchors.csv. Stations '
                         'that already have an anchor are left untouched.')
    args = ap.parse_args(argv)

    campaign_dir = Path(args.data_root) / args.campaign
    if not campaign_dir.is_dir():
        print(f'ERROR: campaign dir not found: {campaign_dir}', file=sys.stderr)
        return 2

    total_csv = campaign_dir / exports.TOTAL_EXPORT_FILENAME
    if not total_csv.exists():
        print(f'ERROR: {total_csv} not found — an anchor is placed inside a clock '
              f'segment, and segments come from the all-images export.',
              file=sys.stderr)
        return 2

    print(f'Reading all-images export : {total_csv}')
    total = pd.read_csv(total_csv, dtype=str, keep_default_na=False, low_memory=False)
    audit = exports.audit_categories(total[exports.OBSERVATION_TYPE_COLUMN])
    print(f'  {len(total)} row(s); export gate verdict: {audit.verdict}')
    total = prepare_total(total, campaign_dir)

    field_path = Path(args.data_root) / FIELD_NOTES_FILENAME
    field = FieldRecord.load(field_path)
    if not len(field):
        print(f'ERROR: {field_path} not found or empty — this script has nothing to '
              f'propose from. Build it with setup/build_field_notes.py.',
              file=sys.stderr)
        return 2
    print(f'Reading field notes       : {field_path}\n  {len(field)} visit(s)')

    on_file = load_anchors(campaign_dir / ANCHOR_FILENAME)
    print(f'  {len(on_file)} anchor row(s) already on file')

    candidates = load_candidates(campaign_dir / CANDIDATES_FILENAME)
    print(f'  {len(candidates)} candidate frame(s) on file')

    try:
        diagnoses = diagnose_campaign(total, on_file, args.campaign, field)
    except stations.UnknownStation as exc:
        print(f'\nERROR: {exc}', file=sys.stderr)
        return 3

    by_camera = anchors.anchors_by_camera(on_file, args.campaign)
    proposals = []
    for camera_num, sd in sorted(diagnoses.items()):
        proposals.extend(anchors.propose(
            sd.diagnosis,
            args.campaign,
            field,
            candidates[candidates['camera_num'] == camera_num],
            by_camera.get(camera_num, []),
        ))

    df = anchors.to_frame(proposals)
    print(render(df))

    out_path = campaign_dir / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)
    print(f'\nWrote: {out_path}')

    ready = df[df['status'] == READY]
    if args.write and len(ready):
        anchored = {a.station_id for a in on_file}
        fresh = ready[~ready['station_id'].isin(anchored)]
        if len(fresh) < len(ready):
            skipped = sorted(set(ready['station_id']) - set(fresh['station_id']))
            print(f'  {len(ready) - len(fresh)} READY row(s) skipped — these '
                  f'stations already carry a hand-written anchor: {skipped}')
        if len(fresh):
            path = campaign_dir / ANCHOR_FILENAME
            combined = pd.concat(
                [pd.read_csv(path) if path.exists() else pd.DataFrame(),
                 fresh[ANCHOR_WRITE_COLUMNS]],
                ignore_index=True,
            )
            combined.to_csv(path, index=False)
            print(f'  appended {len(fresh)} row(s) to {path}')
    elif len(ready):
        print(f'  {len(ready)} READY row(s) NOT written — pass --write to append '
              f'them to {ANCHOR_FILENAME}.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
