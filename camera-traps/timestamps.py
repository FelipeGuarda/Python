"""
timestamps.py — detect + repair camera-clock-reset issues in reviewed camera-trap CSVs.

A camera-trap RTC occasionally reverts to a factory epoch (e.g. 2017-01-01)
mid-deployment. The reset corrupts the EXIF DateTime, filename, and filesystem
mtime identically — there is no independent date source in the raw files. This
module recovers what it can from field-provided anchor data.

USAGE
    python timestamps.py --campaign otono_2026
    python timestamps.py --campaign otono_2026 --dry-run

INPUT
    data/campaigns/<name>/new_labeled_data_reviewed.csv   reviewer output (immutable)
    data/campaigns/<name>/deployment_anchors.csv          field ground truth

OUTPUT
    data/campaigns/<name>/new_labeled_data_corrected.csv  reviewed + 5 new columns
    data/campaigns/<name>/timestamps_audit.log            human-readable report

ANCHOR CSV SCHEMA
    station_id, anchor_type, real_datetime, camera_datetime, source, notes

    station_id        — matches the Deployments column in the reviewed CSV verbatim
                        (e.g. "CT_18" for otono_2026, "CT18" for otono_2025).
    anchor_type       — one of:
        install            EXACT anchor at install. Use a trigger photo + wall clock.
        mid_visit          EXACT anchor at a mid-deployment maintenance visit.
        retrieval          EXACT anchor at retrieval (camera fired at the visit).
        last_real_proxy    APPROXIMATE anchor used ONLY when camera was not firing
                           at retrieval. camera_datetime = last bogus photo's stamp;
                           real_datetime ≈ retrieval time. valid_time_of_day will
                           be FALSE for repaired rows.
        unrepairable_pending
                           Known clock issue, no anchor data yet. Real/camera
                           datetimes may be empty. Photos in the bogus cluster get
                           valid_date=FALSE, valid_time_of_day=FALSE until field
                           info arrives.
    real_datetime     — true wall-clock at the anchor moment (YYYY-MM-DD HH:MM:SS)
    camera_datetime   — what the camera's clock said at that moment (= the EXIF
                        stamp on the trigger photo). Equals real_datetime if the
                        clock was correct at the anchor moment.
    source            — provenance: 'field_notebook', 'trigger_photo', 'last_real_proxy',
                        'pending_field_info', etc.
    notes             — free text

ADDED COLUMNS in new_labeled_data_corrected.csv
    datetime_corrected     offset-adjusted datetime (NaT if unrepairable)
    valid_date             can downstream trust the date?
    valid_time_of_day      can downstream trust the time-of-day?
    repair_method          'none' | 'offset_from_<anchor_type>' | 'unrepairable_<reason>'
                           | 'unparseable_datetime'
    repair_anchor_source   the 'source' field of the anchor row used (or '')

FIELD PROTOCOL (recommended for all future maintenance visits)
    At install, mid-visit, and retrieval:
      1. Note the wall-clock time on your phone or watch — to the minute.
      2. Trigger the camera deliberately (wave hand in front of PIR, or open
         the case to fire the wakeup photo).
      3. Add a row to deployment_anchors.csv with anchor_type ∈ {install,
         mid_visit, retrieval}. The trigger photo's EXIF stamp IS the
         camera_datetime; the wall clock IS the real_datetime.
    If the two match, no clock issue. If they differ, the offset is exactly
    real_datetime − camera_datetime — both date and time-of-day fully
    recoverable.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# 1. Schema & constants
# =============================================================================

BOGUS_YEAR_THRESHOLD = 2024   # photos with year < this are considered bogus

ANCHOR_TYPES_EXACT       = {'install', 'mid_visit', 'retrieval'}
ANCHOR_TYPES_APPROXIMATE = {'last_real_proxy'}
ANCHOR_TYPES_UNREPAIRABLE = {'unrepairable_pending'}
ALL_ANCHOR_TYPES = (
    ANCHOR_TYPES_EXACT | ANCHOR_TYPES_APPROXIMATE | ANCHOR_TYPES_UNREPAIRABLE
)

ANCHOR_REQUIRED_COLS = {
    'station_id', 'anchor_type', 'real_datetime',
    'camera_datetime', 'source', 'notes',
}


@dataclass(frozen=True)
class Anchor:
    station_id: str
    anchor_type: str
    real_datetime: Optional[datetime]    # nullable for unrepairable_pending
    camera_datetime: Optional[datetime]  # nullable for unrepairable_pending
    source: str
    notes: str


@dataclass
class RepairReport:
    campaign: str
    n_photos_total: int = 0
    n_photos_clean: int = 0
    n_photos_repaired_exact: int = 0
    n_photos_repaired_approximate: int = 0
    n_photos_unrepairable: int = 0
    n_photos_unparseable: int = 0
    per_station: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


# =============================================================================
# 2. Load + validate anchors
# =============================================================================

def _parse_datetime(s: str) -> Optional[datetime]:
    """Parse an anchor CSV datetime. Returns None for empty / NA / NaN / NULL."""
    s = (s or '').strip()
    if not s or s.upper() in ('NA', 'NAN', 'NULL', 'NONE'):
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M', '%Y-%m-%d'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f'cannot parse datetime: {s!r}')


def load_anchors(anchor_csv: Path) -> list[Anchor]:
    """Read deployment_anchors.csv; return list of validated Anchor records.
    Returns empty list if the file does not exist."""
    if not anchor_csv.exists():
        return []

    df = pd.read_csv(anchor_csv, dtype=str, keep_default_na=False)

    missing = ANCHOR_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f'{anchor_csv}: missing columns: {sorted(missing)}')

    out: list[Anchor] = []
    for i, row in df.iterrows():
        anchor_type = row['anchor_type'].strip()
        if anchor_type not in ALL_ANCHOR_TYPES:
            raise ValueError(
                f'{anchor_csv} row {i + 2}: unknown anchor_type {anchor_type!r}; '
                f'must be one of {sorted(ALL_ANCHOR_TYPES)}'
            )

        real_dt = _parse_datetime(row['real_datetime'])
        cam_dt  = _parse_datetime(row['camera_datetime'])

        if anchor_type in (ANCHOR_TYPES_EXACT | ANCHOR_TYPES_APPROXIMATE):
            if real_dt is None or cam_dt is None:
                raise ValueError(
                    f'{anchor_csv} row {i + 2}: anchor_type={anchor_type} '
                    f'requires both real_datetime and camera_datetime'
                )

        out.append(Anchor(
            station_id=row['station_id'].strip(),
            anchor_type=anchor_type,
            real_datetime=real_dt,
            camera_datetime=cam_dt,
            source=row['source'].strip(),
            notes=row['notes'].strip(),
        ))
    return out


# =============================================================================
# 3. Load reviewed CSV + classify epochs
# =============================================================================

def load_reviewed(csv: Path) -> pd.DataFrame:
    """Read new_labeled_data_reviewed.csv. Coalesces timestamp + DateTime cols
    and parses into a tz-naive datetime column `_datetime_parsed`."""
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    # Older campaigns populate `timestamp`; newer ones leave it blank and use `DateTime`.
    def _coalesce(r):
        for col in ('timestamp', 'DateTime'):
            v = (r.get(col) or '').strip()
            if v:
                return v
        return ''
    df['_datetime_raw'] = df.apply(_coalesce, axis=1)
    df['_datetime_parsed'] = pd.to_datetime(df['_datetime_raw'], errors='coerce')
    return df


def classify_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'epoch_cluster' ∈ {'real', 'bogus', 'unparseable'}. Pure."""
    df = df.copy()
    def _cluster(dt):
        if pd.isna(dt):
            return 'unparseable'
        return 'bogus' if dt.year < BOGUS_YEAR_THRESHOLD else 'real'
    df['epoch_cluster'] = df['_datetime_parsed'].apply(_cluster)
    return df


# =============================================================================
# 4. Repair
# =============================================================================

def repair_campaign(
    photos: pd.DataFrame,
    anchors: list[Anchor],
    campaign: str,
) -> tuple[pd.DataFrame, RepairReport]:
    """Apply offset repair using the supplied anchors. Adds five columns to the
    photos DataFrame. The original DateTime/timestamp columns are not modified."""
    photos = photos.copy()
    report = RepairReport(campaign=campaign)

    photos['datetime_corrected']   = photos['_datetime_parsed']
    photos['valid_date']           = False
    photos['valid_time_of_day']    = False
    photos['repair_method']        = ''
    photos['repair_anchor_source'] = ''

    anchors_by_station: dict[str, list[Anchor]] = {}
    for a in anchors:
        anchors_by_station.setdefault(a.station_id, []).append(a)

    for station in photos['Deployments'].unique():
        mask_station = photos['Deployments'] == station
        station_anchors = anchors_by_station.get(station, [])

        per_st = {'clusters': {}, 'methods': set()}

        # --- Clean (real epoch) rows ----------------------------------------
        mask_real = mask_station & (photos['epoch_cluster'] == 'real')
        n_real = int(mask_real.sum())
        if n_real:
            photos.loc[mask_real, 'valid_date']           = True
            photos.loc[mask_real, 'valid_time_of_day']    = True
            photos.loc[mask_real, 'repair_method']        = 'none'
            report.n_photos_clean += n_real
            per_st['clusters']['real'] = n_real

        # --- Bogus rows: need an anchor -------------------------------------
        mask_bogus = mask_station & (photos['epoch_cluster'] == 'bogus')
        n_bogus = int(mask_bogus.sum())
        if n_bogus:
            unrepairable_anchors = [
                a for a in station_anchors
                if a.anchor_type in ANCHOR_TYPES_UNREPAIRABLE
            ]
            bogus_anchors = [
                a for a in station_anchors
                if a.camera_datetime is not None
                and a.camera_datetime.year < BOGUS_YEAR_THRESHOLD
                and a.anchor_type not in ANCHOR_TYPES_UNREPAIRABLE
            ]

            if unrepairable_anchors:
                photos.loc[mask_bogus, 'valid_date']           = False
                photos.loc[mask_bogus, 'valid_time_of_day']    = False
                photos.loc[mask_bogus, 'repair_method']        = 'unrepairable_pending_anchor'
                photos.loc[mask_bogus, 'repair_anchor_source'] = unrepairable_anchors[0].source
                photos.loc[mask_bogus, 'datetime_corrected']   = pd.NaT
                report.n_photos_unrepairable += n_bogus
                per_st['clusters']['bogus'] = n_bogus
                per_st['methods'].add('unrepairable_pending_anchor')
                report.warnings.append(
                    f'{station}: {n_bogus} bogus photos marked unrepairable_pending '
                    f'(awaiting field info — notes: {unrepairable_anchors[0].notes!r})'
                )
            elif not bogus_anchors:
                photos.loc[mask_bogus, 'valid_date']           = False
                photos.loc[mask_bogus, 'valid_time_of_day']    = False
                photos.loc[mask_bogus, 'repair_method']        = 'unrepairable_no_anchor'
                photos.loc[mask_bogus, 'datetime_corrected']   = pd.NaT
                report.n_photos_unrepairable += n_bogus
                per_st['clusters']['bogus'] = n_bogus
                per_st['methods'].add('unrepairable_no_anchor')
                report.warnings.append(
                    f'{station}: {n_bogus} bogus photos but NO anchor in '
                    f'deployment_anchors.csv. Add a row with anchor_type one of '
                    f'[retrieval, last_real_proxy, unrepairable_pending].'
                )
            else:
                # Prefer EXACT anchor over APPROXIMATE
                exact = [a for a in bogus_anchors if a.anchor_type in ANCHOR_TYPES_EXACT]
                chosen = exact[0] if exact else bogus_anchors[0]
                offset = chosen.real_datetime - chosen.camera_datetime

                photos.loc[mask_bogus, 'datetime_corrected'] = (
                    photos.loc[mask_bogus, '_datetime_parsed'] + offset
                )
                photos.loc[mask_bogus, 'valid_date']           = True
                photos.loc[mask_bogus, 'repair_method']        = f'offset_from_{chosen.anchor_type}'
                photos.loc[mask_bogus, 'repair_anchor_source'] = chosen.source

                if chosen.anchor_type in ANCHOR_TYPES_EXACT:
                    photos.loc[mask_bogus, 'valid_time_of_day'] = True
                    report.n_photos_repaired_exact += n_bogus
                else:
                    photos.loc[mask_bogus, 'valid_time_of_day'] = False
                    report.n_photos_repaired_approximate += n_bogus

                per_st['clusters']['bogus'] = n_bogus
                per_st['methods'].add(f'offset_from_{chosen.anchor_type}')
                per_st['offset'] = str(offset)
                per_st['anchor_source'] = chosen.source

        # --- Unparseable datetime rows --------------------------------------
        mask_unp = mask_station & (photos['epoch_cluster'] == 'unparseable')
        n_unp = int(mask_unp.sum())
        if n_unp:
            photos.loc[mask_unp, 'valid_date']           = False
            photos.loc[mask_unp, 'valid_time_of_day']    = False
            photos.loc[mask_unp, 'repair_method']        = 'unparseable_datetime'
            photos.loc[mask_unp, 'datetime_corrected']   = pd.NaT
            report.n_photos_unparseable += n_unp
            per_st['clusters']['unparseable'] = n_unp
            per_st['methods'].add('unparseable_datetime')

        report.per_station[station] = per_st

    report.n_photos_total = len(photos)

    # Drop internal scratch columns before returning
    return (
        photos.drop(columns=['_datetime_raw', '_datetime_parsed', 'epoch_cluster']),
        report,
    )


# =============================================================================
# 5. Audit / render
# =============================================================================

def render_report(report: RepairReport) -> str:
    lines = []
    lines.append(f'=== Timestamp audit: campaign {report.campaign} ===')
    lines.append(f'Total photos:                       {report.n_photos_total}')
    lines.append(f'  Clean (no repair needed):         {report.n_photos_clean}')
    lines.append(f'  Repaired (exact anchor):          {report.n_photos_repaired_exact}')
    lines.append(f'  Repaired (approximate anchor):    {report.n_photos_repaired_approximate}')
    lines.append(f'  Unrepairable:                     {report.n_photos_unrepairable}')
    lines.append(f'  Unparseable datetime:             {report.n_photos_unparseable}')
    lines.append('')

    issue_stations = sorted([
        (st, info) for st, info in report.per_station.items()
        if info['clusters'].get('bogus', 0) > 0
        or info['clusters'].get('unparseable', 0) > 0
    ])
    if issue_stations:
        lines.append('Stations with issues:')
        for st, info in issue_stations:
            clusters = info['clusters']
            methods  = sorted(info['methods'])
            lines.append(f'  {st}: clusters={clusters}, methods={methods}')
            if 'offset' in info:
                lines.append(f'    applied offset: {info["offset"]}')
            if 'anchor_source' in info:
                lines.append(f'    anchor source: {info["anchor_source"]}')
        lines.append('')

    clean_stations = sorted([
        st for st, info in report.per_station.items()
        if info['clusters'].get('bogus', 0) == 0
        and info['clusters'].get('unparseable', 0) == 0
    ])
    if clean_stations:
        lines.append(f'Stations with clean clocks ({len(clean_stations)}): '
                     f'{", ".join(clean_stations)}')
        lines.append('')

    if report.warnings:
        lines.append('Warnings:')
        for w in report.warnings:
            lines.append(f'  ! {w}')
        lines.append('')

    return '\n'.join(lines)


# =============================================================================
# 6. CLI
# =============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='Detect + repair camera-clock-reset issues in a reviewed '
                    'camera-trap CSV. Reads new_labeled_data_reviewed.csv and '
                    'deployment_anchors.csv; writes new_labeled_data_corrected.csv '
                    'and timestamps_audit.log.',
    )
    ap.add_argument('--campaign', required=True,
                    help='Campaign directory name, e.g. otono_2026')
    ap.add_argument('--data-root', default='data/campaigns',
                    help='Root directory containing campaign dirs (default: data/campaigns)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print the audit but do not write output files.')
    args = ap.parse_args(argv)

    campaign_dir = Path(args.data_root) / args.campaign
    if not campaign_dir.is_dir():
        print(f'ERROR: campaign dir not found: {campaign_dir}', file=sys.stderr)
        return 2

    reviewed_csv  = campaign_dir / 'new_labeled_data_reviewed.csv'
    anchor_csv    = campaign_dir / 'deployment_anchors.csv'
    corrected_csv = campaign_dir / 'new_labeled_data_corrected.csv'
    audit_log     = campaign_dir / 'timestamps_audit.log'

    if not reviewed_csv.exists():
        print(f'ERROR: reviewed CSV not found: {reviewed_csv}', file=sys.stderr)
        return 2

    print(f'Reading reviewed CSV : {reviewed_csv}')
    photos = load_reviewed(reviewed_csv)
    photos = classify_epochs(photos)
    print(f'  {len(photos)} rows; epoch counts: '
          f'{photos["epoch_cluster"].value_counts().to_dict()}')

    print(f'Reading anchors      : {anchor_csv}')
    anchors = load_anchors(anchor_csv)
    print(f'  {len(anchors)} anchor row(s) loaded')

    print('Applying repair...')
    corrected, report = repair_campaign(photos, anchors, args.campaign)

    audit_text = render_report(report)
    print()
    print(audit_text)

    if args.dry_run:
        print('--dry-run: no files written.')
        return 0

    corrected.to_csv(corrected_csv, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f'Wrote: {corrected_csv}  ({len(corrected)} rows, +5 columns)')

    audit_log.write_text(audit_text, encoding='utf-8')
    print(f'Wrote: {audit_log}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
