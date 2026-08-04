"""
timestamps.py — apply segment-aware clock repair to a reviewed camera-trap campaign.

A camera-trap RTC occasionally reverts to a factory epoch (e.g. 2017-01-01)
mid-deployment. The reset corrupts the EXIF DateTime, the filename and the
filesystem mtime identically — there is no independent date source in the raw
files. This module recovers what field anchors allow, and refuses the rest.

WHAT THIS MODULE OWNS, AND WHAT IT DOES NOT

    `camtrap/clocks.py` decides what a camera's clock did and what may be concluded
    from it: segments, capture order, coherence, and the rule *a segment is
    repairable iff it is coherent AND contains at least one anchor*. This file does
    the I/O and the arithmetic around that decision — anchor CSV, the two Timelapse2
    exports, the DCIM manifest, offset application, the audit log, the CLI.

    That split is not cosmetic. Until 2026-07-31 the repairability rule lived here,
    as `classify_epochs` plus one offset per station, and it could not see more than
    one reset per camera. Otoño 2026 CT_18 had FOUR; a single offset was applied
    across all of them and 65 fabricated dates reached the pehuen analysis. Do not
    move clock reasoning back into this file.

TWO EXPORTS, TWO JOBS

    ImageData_total.csv   ALL images with every category assigned. The clock is
                          diagnosed from this and nothing else, because a reset
                          between two animal photos is invisible in an animal-only
                          file. Gated by `camtrap/exports.py`; a campaign without a
                          valid one HARD-FAILS here rather than falling back.
    new_labeled_data_reviewed.csv   The reviewer's species output. Supplies the rows
                          that get written out, and never the chronology.

USAGE
    python timestamps.py --campaign otono_2026
    python timestamps.py --campaign otono_2026 --dry-run

INPUT
    data/campaigns/<name>/ImageData_total.csv             all images, all categories
    data/campaigns/<name>/new_labeled_data_reviewed.csv   reviewer output (immutable)
    data/campaigns/<name>/deployment_anchors.csv          field ground truth
    data/campaigns/<name>/dcim_manifest.csv               optional; capture order

OUTPUT
    data/campaigns/<name>/new_labeled_data_corrected.csv  reviewed + 7 new columns
    data/campaigns/<name>/observations.parquet            canonical table
    data/campaigns/<name>/timestamps_audit.log            human-readable report

ANCHOR CSV SCHEMA
    station_id, anchor_type, real_datetime, camera_datetime, source, notes
    [, segment_index]

    station_id        — canonical station ID (CT01..CT27). Matched to photos by
                        resolved camera number via `camtrap.stations`, so it does NOT
                        need to match the campaign's raw Deployments spelling.
    anchor_type       — one of:
        install            EXACT anchor at install. Use a trigger photo + wall clock.
        mid_visit          EXACT anchor at a mid-deployment maintenance visit. This
                           is the one that rescues an INTERIOR segment; on a camera
                           that reset four times, install and retrieval between them
                           recover only two of five segments.
        retrieval          EXACT anchor at retrieval (camera fired at the visit).
        last_real_proxy    APPROXIMATE anchor used ONLY when the camera was not
                           firing at retrieval. camera_datetime = last bogus photo's
                           stamp; real_datetime ≈ retrieval time. valid_time_of_day
                           is FALSE for repaired rows.
        unrepairable_pending
                           Known clock issue, no anchor data yet. Real/camera
                           datetimes may be empty. Every row of that station is
                           refused until field info arrives.
    real_datetime     — true wall-clock at the anchor moment (YYYY-MM-DD HH:MM:SS)
    camera_datetime   — what the camera's clock said at that moment (= the EXIF
                        stamp on the trigger photo). Equals real_datetime if the
                        clock was correct at the anchor moment.
    source            — provenance: 'field_notebook', 'trigger_photo', etc.
    notes             — free text
    segment_index     — OPTIONAL. Normally the anchor finds its own segment by
                        strict containment of camera_datetime. Set this only to
                        resolve a case containment cannot: an anchor that falls
                        inside no segment (CT_18's install anchor asserts camera-time
                        2025-11-14 14:00 but the first frame on the card is 11-19
                        06:41), or inside several overlapping ones (CT_18's segments
                        1, 2 and 3 all begin 2017-01-01). It is an assertion that
                        someone checked by eye — record who, in `notes`.

ADDED COLUMNS in new_labeled_data_corrected.csv
    datetime_corrected     offset-adjusted datetime (NaT if unrepairable)
    valid_date             can downstream trust the date?
    valid_time_of_day      can downstream trust the time-of-day?
    valid_effort           are this STATION's trap-nights knowable? Station-level:
                           FALSE excludes the camera from rate DENOMINATORS as well
                           as numerators, for every row, including rows whose own
                           date is fine.
    clock_segment          which segment the row was assigned to (blank if none)
    repair_method          the rule that fired — 'clock_clean' | 'offset_from_<type>'
                           | 'segment_incoherent:<why>' | 'no_anchor_in_segment' |
                           'ordering_unrecoverable' | 'unrepairable_pending_anchor' |
                           'unsegmented_row' | 'unparseable_datetime' |
                           'not_in_total_export'
    repair_anchor_source   the 'source' field of the anchor used (or '')

FIELD PROTOCOL (mandatory for every visit from 2026-08 on)
    At install, at every mid-deployment visit, and at retrieval:
      1. Note the wall-clock time on your phone — to the minute.
      2. Note what the CAMERA's own screen says at install. This is the only thing
         that distinguishes "clock was right and later reset" from "clock was
         already wrong when installed", and it costs five seconds.
      3. Trigger the camera deliberately (wave a hand at the PIR, or open the case
         to fire the wakeup photo) so a person frame exists in the sequence.
      4. Add a row to deployment_anchors.csv.
    Each anchor buys back one whole segment of a broken clock, and a mid-visit
    anchor is the only way to buy back an interior one.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from camtrap import clocks, exports, stations
from camtrap.clocks import (
    ALL_ANCHOR_TYPES,
    ANCHOR_TYPES_APPROXIMATE,
    ANCHOR_TYPES_EXACT,
    ANCHOR_TYPES_UNREPAIRABLE,
    Anchor,
    ClockDiagnosis,
    SegmentRepair,
)
from camtrap.observations import CANONICAL_FILENAME, write_canonical


# =============================================================================
# 1. Schema & constants
# =============================================================================

ANCHOR_REQUIRED_COLS = {
    'station_id', 'anchor_type', 'real_datetime',
    'camera_datetime', 'source', 'notes',
}
ANCHOR_OPTIONAL_COLS = {'segment_index'}

# The deployment window is taken from the anchors' wall-clock times, and a frame
# outside it is evidence of a clock jump. Anchors are recorded to the minute by hand
# while a technician may well trigger the camera before writing the time down, so a
# little slack keeps that sloppiness from manufacturing a segment boundary. It stays
# small: a 2017 or a 2030 stamp is still hours away from any tolerance.
WINDOW_TOLERANCE = timedelta(hours=1)

# Rows that never reached the clock diagnosis at all.
METHOD_UNPARSEABLE  = 'unparseable_datetime'
METHOD_UNSEGMENTED  = 'unsegmented_row'
METHOD_NOT_IN_TOTAL = 'not_in_total_export'

# Key on which a reviewed row is matched to its row in the total export. Filenames
# repeat across deployments (12,068 otoño 2026 rows are unique on this pair and NOT
# on File alone), so the camera must be part of the key.
MATCH_KEY = ['_camera_num', '_file_name']


@dataclass
class StationDiagnosis:
    """One camera's diagnosis, its per-segment verdicts, and the row bookkeeping."""
    station_label: str
    camera_num: int
    diagnosis: ClockDiagnosis
    repairs: list[SegmentRepair]
    window: tuple[datetime, datetime] | None = None
    notes: list[str] = field(default_factory=list)
    n_anchors: int = 0
    row_methods: dict = field(default_factory=dict)   # method -> n reviewed rows

    @property
    def by_segment(self) -> dict[int, SegmentRepair]:
        return {r.segment_index: r for r in self.repairs}

    @property
    def valid_effort(self) -> bool:
        return bool(self.repairs) and all(r.valid_effort for r in self.repairs)


@dataclass
class RepairReport:
    campaign: str
    n_photos_total: int = 0
    n_photos_clean: int = 0
    n_photos_repaired_exact: int = 0
    n_photos_repaired_approximate: int = 0
    n_photos_unrepairable: int = 0
    n_photos_unparseable: int = 0
    n_stations_no_effort: int = 0
    export_audit: Optional[exports.CategoryAudit] = None
    order_evidence: dict = field(default_factory=dict)     # station -> evidence
    per_station: dict = field(default_factory=dict)        # station -> StationDiagnosis
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


def _parse_segment_index(s: str, where: str) -> Optional[int]:
    s = (s or '').strip()
    if not s or s.upper() in ('NA', 'NAN', 'NULL', 'NONE'):
        return None
    try:
        return int(s)
    except ValueError:
        raise ValueError(
            f'{where}: segment_index must be an integer or empty, got {s!r}'
        ) from None


def load_anchors(anchor_csv: Path) -> list[Anchor]:
    """Read deployment_anchors.csv; return validated `camtrap.clocks.Anchor` rows.
    Returns an empty list if the file does not exist."""
    if not anchor_csv.exists():
        return []

    df = pd.read_csv(anchor_csv, dtype=str, keep_default_na=False)

    missing = ANCHOR_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f'{anchor_csv}: missing columns: {sorted(missing)}')

    unexpected = set(df.columns) - ANCHOR_REQUIRED_COLS - ANCHOR_OPTIONAL_COLS
    if unexpected:
        # Not fatal, but a misspelled `segment_idx` would be silently ignored, and
        # silently ignoring an assertion about which segment an anchor belongs to is
        # exactly the kind of quiet failure this pipeline keeps being bitten by.
        print(f'  WARNING: {anchor_csv} has unrecognised column(s) '
              f'{sorted(unexpected)} — ignored')

    out: list[Anchor] = []
    for i, row in df.iterrows():
        where = f'{anchor_csv} row {i + 2}'
        anchor_type = row['anchor_type'].strip()
        if anchor_type not in ALL_ANCHOR_TYPES:
            raise ValueError(
                f'{where}: unknown anchor_type {anchor_type!r}; '
                f'must be one of {sorted(ALL_ANCHOR_TYPES)}'
            )

        real_dt = _parse_datetime(row['real_datetime'])
        cam_dt  = _parse_datetime(row['camera_datetime'])

        if anchor_type in (ANCHOR_TYPES_EXACT | ANCHOR_TYPES_APPROXIMATE):
            if real_dt is None or cam_dt is None:
                raise ValueError(
                    f'{where}: anchor_type={anchor_type} requires both '
                    f'real_datetime and camera_datetime'
                )

        out.append(Anchor(
            station_id=row['station_id'].strip(),
            anchor_type=anchor_type,
            real_datetime=real_dt,
            camera_datetime=cam_dt,
            source=row['source'].strip(),
            notes=row['notes'].strip(),
            segment_index=_parse_segment_index(row.get('segment_index', ''), where),
        ))
    return out


def anchors_by_camera(anchors: list[Anchor], campaign: str) -> dict[int, list[Anchor]]:
    """Group anchors by resolved camera number.

    Anchors and photos are matched on the camera number, not on the raw station
    string, so an anchor file can use the canonical ID (CT16) regardless of how the
    campaign's Timelapse2 export spelled it (CT16 / TC16_M13.2 / CT_16).
    """
    grouped: dict[int, list[Anchor]] = {}
    for a in anchors:
        grouped.setdefault(stations.resolve(a.station_id, campaign), []).append(a)
    return grouped


def deployment_window(anchors: list[Anchor]) -> tuple[datetime, datetime] | None:
    """The real-time window this station was deployed for, from its anchors.

    Supplying a window to `clocks.diagnose` is what lets a FORWARD jump be seen: a
    clock set to 2030 keeps every delta positive and is invisible to
    backwards-step detection. Needs two distinct wall-clock times to be a window at
    all, so a station with a single anchor gets None rather than a zero-width guess.
    """
    reals = sorted({a.real_datetime for a in anchors if a.real_datetime is not None})
    if len(reals) < 2:
        return None
    return reals[0] - WINDOW_TOLERANCE, reals[-1] + WINDOW_TOLERANCE


# =============================================================================
# 3. Load the two exports and the manifest
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


def load_manifest(campaign_dir: Path) -> Optional[pd.DataFrame]:
    """Read dcim_manifest.csv if the campaign has one.

    Written by setup/flatten_for_camtrapdp.py. Its absence is not an error — otoño
    2026 was flattened before the manifest existed and can never have one — but it
    is the difference between being able to locate a reset and only being able to
    rule one out.
    """
    path = campaign_dir / clocks.DCIM_MANIFEST_FILENAME
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(clocks.DCIM_MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    return df


def attach_dcim_folder(total: pd.DataFrame, manifest: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add a `dcim_folder` column to the total export from the manifest.

    Joined on (deployment, flat_name) — the flat name is what Timelapse2 sees after
    flattening, so it is the only key the two files share. A frame the manifest does
    not describe gets an empty folder, which `clocks.establish_order` treats as a
    partially-described deployment and refuses to order. That is deliberate: a
    manifest covering some frames would otherwise sort those under their folder and
    pool the rest, yielding a confident wrong order.
    """
    df = total.copy()
    if manifest is None or manifest.empty:
        return df

    lookup = manifest[['deployment', 'flat_name', 'dcim_folder']].drop_duplicates(
        subset=['deployment', 'flat_name'], keep='first'
    )
    df = df.merge(
        lookup.rename(columns={'deployment': 'Deployments', 'flat_name': 'File'}),
        on=['Deployments', 'File'], how='left',
    )
    df['dcim_folder'] = df['dcim_folder'].fillna('')

    n_undescribed = int(df['dcim_folder'].eq('').sum())
    if n_undescribed:
        print(f'  WARNING: {n_undescribed} export row(s) are absent from '
              f'{clocks.DCIM_MANIFEST_FILENAME} — those deployments cannot be ordered')
    return df


def prepare_total(total: pd.DataFrame, campaign_dir: Path) -> pd.DataFrame:
    """Attach capture-order evidence and a parsed stamp to a loaded export.

    Split out from load_total so `anchor_candidates.py` can prepare the same frame
    WITHOUT the gate: that report exists to help Felipe find the anchors a rejected
    campaign is missing, so gating it would lock the door on the room holding the key.
    Enforcement stays in exports.read_total_export, which is the ingest path.
    """
    total = attach_dcim_folder(total, load_manifest(campaign_dir))
    total['_datetime_parsed'] = pd.to_datetime(
        total['DateTime'].astype(str).str.strip(), errors='coerce'
    )
    return total


def load_total(campaign_dir: Path) -> tuple[pd.DataFrame, exports.CategoryAudit]:
    """The all-images export, gated, with camera numbers and capture order attached."""
    total, audit = exports.read_total_export(campaign_dir)
    return prepare_total(total, campaign_dir), audit


# =============================================================================
# 4. Diagnose every camera from the total export
# =============================================================================

def diagnose_campaign(
    total: pd.DataFrame,
    anchors: list[Anchor],
    campaign: str,
) -> dict[int, StationDiagnosis]:
    """Run clocks.diagnose + clocks.repair_plan once per camera.

    Keyed by camera number rather than by the raw Deployments string: two folders can
    belong to one card (primavera 2025 has camera 5 under both `CT05` and the
    unrenamed `100EK113`), and they are one chronology, so splitting them would
    invent a reset at the folder boundary.
    """
    stations.validate(total['Deployments'].astype(str), campaign)
    total['_camera_num'] = [
        stations.resolve(str(s), campaign) for s in total['Deployments']
    ]

    by_camera = anchors_by_camera(anchors, campaign)
    out: dict[int, StationDiagnosis] = {}

    for camera_num, frames in total.groupby('_camera_num', sort=True):
        station_anchors = by_camera.get(camera_num, [])
        window = deployment_window(station_anchors)

        images = pd.DataFrame({
            'file_name': frames['File'].astype(str).str.strip(),
            'camera_datetime': frames['_datetime_parsed'],
        }, index=frames.index)
        if 'dcim_folder' in frames.columns:
            images['dcim_folder'] = frames['dcim_folder']

        label = stations.canonical_id(camera_num)
        diagnosis = clocks.diagnose(images, label, window=window)
        repairs, notes = clocks.repair_plan(diagnosis, station_anchors)

        out[camera_num] = StationDiagnosis(
            station_label=label,
            camera_num=camera_num,
            diagnosis=diagnosis,
            repairs=repairs,
            window=window,
            # repair_plan() already prefixes its notes with the station; the audit
            # prefixes them again when rendering, so strip it here.
            notes=list(diagnosis.notes) + [n.removeprefix(f'{label}: ') for n in notes],
            n_anchors=len(station_anchors),
        )

    return out


def segment_lookup(
    total: pd.DataFrame,
    diagnoses: dict[int, StationDiagnosis],
) -> pd.DataFrame:
    """Build the (camera, file) -> segment table the reviewed rows are joined onto.

    A (camera, file) pair whose rows disagree about their segment is left unresolved
    rather than given one of the two answers. It happens when the same filename
    appears twice on one card — which is precisely what a reset-clock camera emits,
    so guessing here would corrupt the case this pipeline exists to handle.
    """
    frames = []
    for camera_num, sd in diagnoses.items():
        rows = total[total['_camera_num'] == camera_num]
        seg = clocks.segment_for_rows(sd.diagnosis, rows['_datetime_parsed'])
        frames.append(pd.DataFrame({
            '_camera_num': camera_num,
            '_file_name': rows['File'].astype(str).str.strip().values,
            '_segment': seg.values,
        }))

    if not frames:
        return pd.DataFrame(columns=MATCH_KEY + ['_segment'])

    table = pd.concat(frames, ignore_index=True)
    agg = table.groupby(MATCH_KEY, dropna=False)['_segment'].agg(
        _segment='first', _n_distinct=lambda s: s.dropna().nunique(),
    ).reset_index()
    ambiguous = agg['_n_distinct'] > 1
    agg.loc[ambiguous, '_segment'] = pd.NA
    return agg.drop(columns='_n_distinct')


# =============================================================================
# 5. Apply the plan to the reviewed rows
# =============================================================================

def repair_campaign(
    photos: pd.DataFrame,
    total: pd.DataFrame,
    diagnoses: dict[int, StationDiagnosis],
    campaign: str,
    *,
    allow_unmatched: bool = False,
) -> tuple[pd.DataFrame, RepairReport]:
    """Apply the per-segment plan to the reviewed rows. Adds seven columns.

    The original DateTime/timestamp columns are never modified.
    """
    photos = photos.copy()
    report = RepairReport(campaign=campaign)

    photos['_camera_num'] = [
        stations.resolve(str(s), campaign) for s in photos['Deployments']
    ]
    photos['_file_name'] = photos['File'].astype(str).str.strip()

    lookup = segment_lookup(total, diagnoses)
    photos = photos.merge(lookup, on=MATCH_KEY, how='left', indicator='_matched')

    # A reviewed row absent from the all-images export means the export is not, in
    # fact, all the images — the one thing the gate exists to guarantee. Stopping is
    # the point; --allow-unmatched exists so a mismatch does not block a campaign
    # outright, but it marks every such row unusable rather than guessing.
    unmatched = photos['_matched'].eq('left_only')
    if unmatched.any():
        by_station = (
            photos.loc[unmatched]
            .groupby('Deployments', observed=True)
            .size().sort_values(ascending=False)
        )
        detail = ', '.join(f'{st}: {n}' for st, n in by_station.items())
        examples = ', '.join(photos.loc[unmatched, '_file_name'].head(5))
        msg = (
            f'{int(unmatched.sum())} reviewed row(s) do not appear in '
            f'{exports.TOTAL_EXPORT_FILENAME} ({detail}). Examples: {examples}. '
            f'The total export must cover every image the reviewer saw, or the '
            f'clock diagnosis was run on a different set of frames than the rows '
            f'being written.'
        )
        if not allow_unmatched:
            raise ValueError(
                f'{msg}\n  Re-export all images from the same Timelapse2 project, '
                f'or re-run with --allow-unmatched to write those rows as '
                f'{METHOD_NOT_IN_TOTAL} (unusable).'
            )
        report.warnings.append(msg)

    photos['datetime_corrected']   = photos['_datetime_parsed']
    photos['valid_date']           = False
    photos['valid_time_of_day']    = False
    photos['valid_effort']         = False
    photos['clock_segment']        = pd.Series(pd.NA, index=photos.index, dtype='Int64')
    photos['repair_method']        = ''
    photos['repair_anchor_source'] = ''

    for camera_num, sd in diagnoses.items():
        mask_station = photos['_camera_num'] == camera_num
        if not mask_station.any():
            continue
        by_segment = sd.by_segment

        for seg_index, repair in by_segment.items():
            mask = mask_station & photos['_segment'].eq(seg_index)
            if not mask.any():
                continue
            photos.loc[mask, 'clock_segment']        = seg_index
            photos.loc[mask, 'repair_method']        = repair.reason
            photos.loc[mask, 'repair_anchor_source'] = repair.anchor_source
            photos.loc[mask, 'valid_date']           = repair.valid_date
            photos.loc[mask, 'valid_time_of_day']    = repair.valid_time_of_day
            photos.loc[mask, 'valid_effort']         = repair.valid_effort

            if repair.valid_date and repair.offset is not None:
                photos.loc[mask, 'datetime_corrected'] = (
                    photos.loc[mask, '_datetime_parsed'] + repair.offset
                )
            elif not repair.valid_date:
                photos.loc[mask, 'datetime_corrected'] = pd.NaT

        # Rows of a diagnosed station that no segment claims. On a multi-segment
        # camera this is an honest "we cannot say which reset this frame is on".
        mask_unseg = mask_station & photos['_segment'].isna() & ~unmatched
        if mask_unseg.any():
            photos.loc[mask_unseg, 'repair_method']      = METHOD_UNSEGMENTED
            photos.loc[mask_unseg, 'datetime_corrected'] = pd.NaT
            report.warnings.append(
                f'{sd.station_label}: {int(mask_unseg.sum())} row(s) fall inside no '
                f'single segment of a {len(sd.diagnosis.segments)}-segment camera — '
                f'marked {METHOD_UNSEGMENTED}'
            )

    if unmatched.any():
        photos.loc[unmatched, 'repair_method']      = METHOD_NOT_IN_TOTAL
        photos.loc[unmatched, 'datetime_corrected'] = pd.NaT
        photos.loc[unmatched, ['valid_date', 'valid_time_of_day', 'valid_effort']] = False

    # An unparseable stamp cannot be repaired by any offset, whatever its segment
    # says. Checked last so it overrides a clean verdict inherited from the station.
    mask_unp = photos['_datetime_parsed'].isna()
    if mask_unp.any():
        photos.loc[mask_unp, 'repair_method']      = METHOD_UNPARSEABLE
        photos.loc[mask_unp, 'datetime_corrected'] = pd.NaT
        photos.loc[mask_unp, ['valid_date', 'valid_time_of_day']] = False

    # ── Counters ──────────────────────────────────────────────────────────────
    exact_methods = {f'offset_from_{t}' for t in ANCHOR_TYPES_EXACT}
    approx_methods = {f'offset_from_{t}' for t in ANCHOR_TYPES_APPROXIMATE}
    method = photos['repair_method']

    report.n_photos_total                 = len(photos)
    report.n_photos_clean                 = int((method == 'clock_clean').sum())
    report.n_photos_repaired_exact        = int(method.isin(exact_methods).sum())
    report.n_photos_repaired_approximate  = int(method.isin(approx_methods).sum())
    report.n_photos_unparseable           = int((method == METHOD_UNPARSEABLE).sum())
    invalid_date = ~photos['valid_date'].astype(bool)
    report.n_photos_unrepairable = int(
        (invalid_date & (method != METHOD_UNPARSEABLE)).sum()
    )

    for camera_num, sd in diagnoses.items():
        rows = photos[photos['_camera_num'] == camera_num]
        sd.row_methods = rows['repair_method'].value_counts().to_dict()
        report.per_station[sd.station_label] = sd
        report.order_evidence[sd.station_label] = sd.diagnosis.order_evidence
        report.warnings.extend(f'{sd.station_label}: {n}' for n in sd.notes)
        if not sd.valid_effort:
            report.n_stations_no_effort += 1

    return (
        photos.drop(columns=[
            '_datetime_raw', '_datetime_parsed', '_camera_num', '_file_name',
            '_segment', '_matched',
        ]),
        report,
    )


# =============================================================================
# 6. Audit / render
# =============================================================================

def _fmt(dt) -> str:
    return '—' if dt is None or pd.isna(dt) else pd.Timestamp(dt).strftime('%Y-%m-%d %H:%M')


def render_report(report: RepairReport) -> str:
    lines = []
    lines.append(f'=== Timestamp audit: campaign {report.campaign} ===')
    if report.export_audit is not None:
        lines.append('Export gate:')
        lines.extend(f'  {ln}' for ln in report.export_audit.describe().splitlines())
        lines.append('')

    lines.append(f'Reviewed rows:                      {report.n_photos_total}')
    lines.append(f'  Clean (no repair needed):         {report.n_photos_clean}')
    lines.append(f'  Repaired (exact anchor):          {report.n_photos_repaired_exact}')
    lines.append(f'  Repaired (approximate anchor):    {report.n_photos_repaired_approximate}')
    lines.append(f'  Unrepairable:                     {report.n_photos_unrepairable}')
    lines.append(f'  Unparseable datetime:             {report.n_photos_unparseable}')
    lines.append(f'  Stations with unknowable effort:  {report.n_stations_no_effort}')
    lines.append('')

    troubled = [sd for sd in report.per_station.values()
                if sd.diagnosis.has_clock_failure or not sd.valid_effort]
    healthy = [sd for sd in report.per_station.values() if sd not in troubled]

    if troubled:
        lines.append('── Stations with a clock failure ' + '─' * 46)
        for sd in sorted(troubled, key=lambda s: s.camera_num):
            d = sd.diagnosis
            lines.append(
                f'  {sd.station_label} — {d.n_stills} still(s), '
                f'{d.n_videos_excluded} video(s) excluded, '
                f'{d.n_unparseable} unparseable, {sd.n_anchors} anchor(s)'
            )
            lines.append(
                f'    capture order: {"established" if d.ordered else "NOT established"} '
                f'(evidence: {d.order_evidence})'
            )
            if sd.window:
                lines.append(f'    deployment window: {_fmt(sd.window[0])} → {_fmt(sd.window[1])}')
            lines.append(f'    segments: {len(d.segments)}')
            by_segment = sd.by_segment
            for s in d.segments:
                r = by_segment.get(s.index)
                flags = (
                    f'date={"T" if r.valid_date else "F"} '
                    f'tod={"T" if r.valid_time_of_day else "F"}'
                    if r else 'no verdict'
                )
                offset = '' if not r or r.offset is None else f'  offset={r.offset}'
                lines.append(
                    f'      [{s.index}] {s.n_images:>5} frame(s)  '
                    f'{_fmt(s.camera_start)} → {_fmt(s.camera_end)}  '
                    f'{"coherent" if s.coherent else "INCOHERENT"}  '
                    f'{"" if s.in_window is None else ("in-window" if s.in_window else "OUT-OF-WINDOW")}'
                )
                lines.append(
                    f'            → {r.reason if r else "—"}  {flags}{offset}'
                    + (f'  anchor={r.anchor_source}' if r and r.anchor_source else '')
                )
            if d.unaccounted_days:
                lines.append(
                    f'    unaccounted_days: {d.unaccounted_days} '
                    f'(AUDIT DIAGNOSTIC ONLY — never a criterion)'
                )
            lines.append(
                f'    valid_effort: {"TRUE" if sd.valid_effort else "FALSE"}'
                + ('' if sd.valid_effort else
                   ' — exclude this station from rate DENOMINATORS as well as numerators')
            )
            if sd.row_methods:
                lines.append(f'    reviewed rows by method: {sd.row_methods}')
        lines.append('')

    if healthy:
        lines.append(
            f'Stations with a clean clock ({len(healthy)}): '
            + ', '.join(sorted(sd.station_label for sd in healthy))
        )
        unordered = [sd.station_label for sd in healthy if not sd.diagnosis.ordered]
        if unordered:
            lines.append(
                f'  of which {len(unordered)} could not be ordered but show no clock '
                f'failure ({", ".join(sorted(unordered))}) — every frame is in-window '
                f'and agrees with its own filename, so there is no reset to attribute'
            )
        lines.append('')

    if report.warnings:
        lines.append('Notes and warnings:')
        for w in report.warnings:
            lines.append(f'  ! {w}')
        lines.append('')

    return '\n'.join(lines)


# =============================================================================
# 7. CLI
# =============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='Apply segment-aware camera-clock repair to a reviewed '
                    'camera-trap campaign. Diagnoses every clock from the '
                    'all-images export (ImageData_total.csv, gated), applies '
                    'per-segment offsets from deployment_anchors.csv, and writes '
                    'new_labeled_data_corrected.csv, observations.parquet and '
                    'timestamps_audit.log.',
    )
    ap.add_argument('--campaign', required=True,
                    help='Campaign directory name, e.g. otono_2026')
    ap.add_argument('--data-root', default='data/campaigns',
                    help='Root directory containing campaign dirs (default: data/campaigns)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print the audit but do not write output files.')
    ap.add_argument('--allow-unmatched', action='store_true',
                    help='Do not abort when a reviewed row is missing from the '
                         'total export; mark those rows unusable instead.')
    args = ap.parse_args(argv)

    campaign_dir = Path(args.data_root) / args.campaign
    if not campaign_dir.is_dir():
        print(f'ERROR: campaign dir not found: {campaign_dir}', file=sys.stderr)
        return 2

    reviewed_csv  = campaign_dir / 'new_labeled_data_reviewed.csv'
    anchor_csv    = campaign_dir / 'deployment_anchors.csv'
    corrected_csv = campaign_dir / 'new_labeled_data_corrected.csv'
    canonical_pq  = campaign_dir / CANONICAL_FILENAME
    audit_log     = campaign_dir / 'timestamps_audit.log'

    if not reviewed_csv.exists():
        print(f'ERROR: reviewed CSV not found: {reviewed_csv}', file=sys.stderr)
        return 2

    print(f'Reading all-images export : {campaign_dir / exports.TOTAL_EXPORT_FILENAME}')
    try:
        total, export_audit = load_total(campaign_dir)
    except exports.ExportGateError as exc:
        print(f'\nERROR: {exc}', file=sys.stderr)
        return 3
    print(f'  {len(total)} row(s); {export_audit.verdict}')

    print(f'Reading reviewed CSV      : {reviewed_csv}')
    photos = load_reviewed(reviewed_csv)
    print(f'  {len(photos)} row(s)')

    print(f'Reading anchors           : {anchor_csv}')
    anchors = load_anchors(anchor_csv)
    print(f'  {len(anchors)} anchor row(s) loaded')

    print('Diagnosing clocks from the all-images export...')
    try:
        diagnoses = diagnose_campaign(total, anchors, args.campaign)
    except stations.UnknownStation as exc:
        print(f'\nERROR: {exc}', file=sys.stderr)
        return 3
    n_failed = sum(1 for sd in diagnoses.values() if sd.diagnosis.has_clock_failure)
    print(f'  {len(diagnoses)} camera(s); {n_failed} with a clock failure')

    print('Applying repair...')
    try:
        corrected, report = repair_campaign(
            photos, total, diagnoses, args.campaign,
            allow_unmatched=args.allow_unmatched,
        )
    except ValueError as exc:
        print(f'\nERROR: {exc}', file=sys.stderr)
        return 3
    report.export_audit = export_audit

    audit_text = render_report(report)
    print()
    print(audit_text)

    if args.dry_run:
        print('--dry-run: no files written.')
        return 0

    corrected.to_csv(corrected_csv, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f'Wrote: {corrected_csv}  ({len(corrected)} rows, +7 columns)')

    # Canonical observation table — the shape every downstream consumer reads.
    # The _corrected.csv above stays for consumers not yet migrated (pehuen).
    n_canonical = write_canonical(corrected, args.campaign, canonical_pq)
    print(f'Wrote: {canonical_pq}  ({n_canonical} rows, canonical schema)')

    audit_log.write_text(audit_text, encoding='utf-8')
    print(f'Wrote: {audit_log}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
