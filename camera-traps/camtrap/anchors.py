"""
camtrap/anchors.py — what the field record asserts about a camera's clock.

WHAT THIS OWNS

    Two different assertions, both sourced from the field, and the boundary between
    them is the whole point of this module:

    1. THE DEPLOYMENT WINDOW — when the camera was physically in the ground. Comes
       from the visit record and needs no photograph at all. It is what makes a
       FORWARD clock jump visible, because a clock set to 2030 keeps every capture
       delta positive and is invisible to backwards-step detection.

    2. AN ANCHOR — a wall-clock moment paired with what the camera's clock said at
       that moment. Requires a frame someone can date. This is the only thing that
       can repair a broken clock.

    Keeping them apart matters because they have different preconditions. Every
    station has a window; only a station whose clock actually failed needs an anchor.

A VISIT IS NOT AN ANCHOR (learned from CT01, 2026-08-12)

    The tempting move — "we know when the technician visited, so use that date" — is
    wrong for a camera whose clock is fine. CT01's notebook says the deployment
    opened 2025-11-24 and closed 2026-05-13, while its frames run 2025-11-26 13:39
    to 2026-05-14 13:35 across ONE coherent segment with no reset. Forcing the
    notebook's dates on as an anchor would apply a two-day offset to a clock that was
    never wrong, importing the notebook's own imprecision into clean data.

    So: an anchor is proposed only where `repair_plan` would otherwise refuse the
    segment. A clean camera gets NOT_NEEDED and no row. `clocks.repair_plan` already
    returns `clock_clean` for it, which is the correct outcome and needs no help.

WHY A VISIT-DERIVED WINDOW NEEDS ITS OWN TOLERANCE

    `WINDOW_TOLERANCE` (1 h) is calibrated for an anchor recorded to the minute
    against the frame it describes. A notebook visit date is a different kind of
    measurement: it has day precision, and the visit itself spans several days across
    27 stations. Applying the 1 h tolerance to it would have called CT01 broken.

    The bound is measured, not guessed. Across the 20 otoño 2026 stations that are
    provably coherent from capture order alone — so any gap to the notebook is the
    NOTEBOOK's imprecision, not the camera's — the largest excursion past a recorded
    visit date is +1.67 d (CT02's last frame). VISIT_WINDOW_TOLERANCE is set to 3 d,
    above that observed maximum and still some three orders of magnitude tighter than
    the failures it must catch: CT18's stamps sit eight YEARS outside its window.
    Verified 2026-08-12: with this tolerance, 26 of 27 stations gain a window they
    never had and not one verdict changes.

THE WINDOW IS A BRACKET, NOT A BAND

    A frame before the opening visit is impossible — the camera was not in the
    ground. A frame after the closing visit is impossible — the card had been pulled.
    A quiet stretch INSIDE the window is evidence of nothing: CT06 and CT11 went 35
    and 41 days from install to first trigger, and CT19 stopped firing 91 days before
    retrieval. Both are ordinary low-traffic behaviour, not clock failure. Since the
    window is only ever tested at its two edges, `[opening - tol, closing + tol]`
    expresses this directly and `clocks.diagnose` needs no change.

DATE-ONLY VISITS

    All 27 otoño 2026 opening visits record a date and no time; all 26 closing visits
    record a time to the minute. A date-only visit therefore cannot support an EXACT
    anchor — asserting an hour nobody wrote down is how CT18's existing install anchor
    came to claim `14:00:00` against a notebook that says only `2025-11-14`. It
    becomes a `visit_date_only` anchor instead, which is APPROXIMATE: the date is
    recovered, `valid_time_of_day` is False, and activity analysis never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from camtrap import clocks, stations, visit_schema
from camtrap.clocks import (
    ALL_ANCHOR_TYPES,
    ANCHOR_TYPES_APPROXIMATE,
    ANCHOR_TYPES_EXACT,
    Anchor,
    ClockDiagnosis,
)

# =============================================================================
# 1. The anchor CSV
# =============================================================================

ANCHOR_FILENAME = 'deployment_anchors.csv'

ANCHOR_REQUIRED_COLS = {
    'station_id', 'anchor_type', 'real_datetime',
    'camera_datetime', 'source', 'notes',
}
ANCHOR_OPTIONAL_COLS = {'segment_index'}
ANCHOR_WRITE_COLUMNS = [
    'station_id', 'anchor_type', 'real_datetime', 'camera_datetime',
    'source', 'notes', 'segment_index',
]

# The deployment window is taken from the anchors' wall-clock times, and a frame
# outside it is evidence of a clock jump. Anchors are recorded to the minute by hand
# while a technician may well trigger the camera before writing the time down, so a
# little slack keeps that sloppiness from manufacturing a segment boundary. It stays
# small: a 2017 or a 2030 stamp is still hours away from any tolerance.
WINDOW_TOLERANCE = timedelta(hours=1)


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


# =============================================================================
# 2. The field visit record
# =============================================================================

FIELD_NOTES_FILENAME = 'field_notes.csv'

# Above this observed maximum excursion of +1.67 d; see the module docstring.
VISIT_WINDOW_TOLERANCE = timedelta(days=3)

# The hour assumed for a visit whose date was recorded without a time. Noon is the
# minimax choice: it bounds the date error at ±12 h whatever the true hour was.
# Recorded visit times, where they exist at all, run 10:05 to 17:20. The assumption
# never reaches an output — a date-only visit yields an APPROXIMATE anchor, whose
# `valid_time_of_day` is False precisely so nobody reads the hour back.
ASSUMED_VISIT_HOUR = time(12, 0)


@dataclass(frozen=True)
class Visit:
    """One physical visit to a station, as `field_notes.csv` records it.

    A visit is an event, not a property of a campaign: at Bosque Pehuén every
    revision swaps the card, so one visit CLOSES one campaign and OPENS the next.
    """
    station_id: str
    visit_type: str                 # visit_schema.VISIT_TYPES, or legacy 'unrecorded'
    campaign_closed: str            # DERIVED, never read from the file — see _derive_closings
    campaign_opened: str
    visit_date: Optional[datetime]  # date at midnight, or None
    visit_time_recorded: bool       # False => the hour is ASSUMED_VISIT_HOUR
    flags: str
    source_sheet: str

    @property
    def real_datetime(self) -> Optional[datetime]:
        return self.visit_date

    @property
    def datable(self) -> bool:
        """Can this visit date anything at all? A flagged date cannot."""
        return self.visit_date is not None and not self.ambiguous

    @property
    def ambiguous(self) -> bool:
        """The builder flags a date it could not settle; such a date may not anchor.

        `date_ambiguous` and `date_conflict` mean two readings are equally plausible
        — CT27's only visit is 2025-11-12 or 2025-12-11, a month apart. Anchoring on
        either would be a coin flip recorded as a fact.
        """
        return 'date_ambiguous' in self.flags or 'date_conflict' in self.flags


#: Retired 2026-08-26. Present only in a pre-reshape `field_notes.csv`.
RECORDED_CLOSE_COLUMN = 'campaign_closed'


def _derive_closings(visits: list[Visit]) -> list[Visit]:
    """Fill `campaign_closed` from each station's own visit sequence.

    The field form deliberately does not collect it: a visit always closes the
    campaign the previous visit to that station opened, so recording it a second time
    only creates a cell that can disagree with the rest of the sheet. Measured
    against the 107 legacy rows before the column was dropped, the derivation
    reproduced 105 of the 106 dated values. The single exception was CT27's
    2025-12-11 row, which claimed to close `primavera_2025` for a station that has no
    primavera deployment in `deployments.csv`, none in the canonical table and no
    prior visit at all — i.e. the derivation was right and the recorded value wrong.

    An undated visit cannot be placed in the sequence, so it closes nothing and does
    not advance the carry. CT27's `unrecorded` placeholder is the live case.

    `retiro` resets the carry because the card comes out for good: without the reset,
    a station lifted and later reinstalled would derive its reinstall as closing the
    campaign that ended before the gap. There are no `retiro` rows in the record yet,
    so this branch is held by a fixture rather than by data.
    """
    closed = [''] * len(visits)
    by_station: dict[str, list[int]] = {}
    for i, v in enumerate(visits):
        by_station.setdefault(v.station_id, []).append(i)

    for idxs in by_station.values():
        ordered = sorted(idxs, key=lambda i: (visits[i].visit_date is None,
                                              visits[i].visit_date or datetime.min,
                                              i))
        in_ground = ''
        for i in ordered:
            v = visits[i]
            if v.visit_type in visit_schema.CLOSES_CAMPAIGN:
                closed[i] = in_ground
            if v.visit_type in visit_schema.OPENS_CAMPAIGN:
                in_ground = v.campaign_opened
            elif v.visit_type in visit_schema.CLOSES_CAMPAIGN:
                in_ground = ''          # retiro: nothing left in the ground

    return [replace(v, campaign_closed=c) for v, c in zip(visits, closed)]


class FieldRecord:
    """Every visit in `field_notes.csv`, queryable by station and campaign.

    Hides the CSV's shape and the campaign_opened/closed convention. Callers ask
    "when did this station's deployment open?" and never touch a column name.
    """

    def __init__(self, visits: list[Visit]):
        self._visits = visits

    @classmethod
    def load(cls, path: Path) -> 'FieldRecord':
        """Returns an empty record if the file does not exist, so a campaign with no
        field notes degrades to the anchor-derived window rather than failing.

        Refuses a file that still carries `campaign_closed`. That column was
        authoritative until 2026-08-26 and is now derived, so a file holding it is
        either a pre-reshape copy or a `build_field_notes.py` run that reverted the
        curated rows — and reading it while ignoring the column would reinterpret it
        silently. `setup/reshape_field_notes.py` performs the one-time conversion.
        """
        if not path.exists():
            return cls([])

        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if RECORDED_CLOSE_COLUMN in df.columns:
            raise ValueError(
                f'{path} still carries the retired `{RECORDED_CLOSE_COLUMN}` column. '
                'The campaign a visit closes is derived from the station\'s visit '
                'sequence since 2026-08-26; a file that records it is a pre-reshape '
                'copy. Run setup/reshape_field_notes.py, or fix the fixture.'
            )
        visits: list[Visit] = []
        for _, r in df.iterrows():
            raw_date = (r.get('visit_date') or '').strip()
            raw_time = (r.get('visit_time') or '').strip()
            when = _parse_datetime(raw_date) if raw_date else None
            has_time = bool(raw_time)
            if when is not None:
                clock = _parse_datetime(f'2000-01-01 {raw_time}') if has_time else None
                when = when.replace(
                    hour=clock.hour if clock else ASSUMED_VISIT_HOUR.hour,
                    minute=clock.minute if clock else ASSUMED_VISIT_HOUR.minute,
                    second=clock.second if clock else 0,
                )
            visits.append(Visit(
                station_id=(r.get('station_id') or '').strip(),
                visit_type=(r.get('visit_type') or '').strip(),
                campaign_closed='',        # filled by _derive_closings below
                campaign_opened=(r.get('campaign_opened') or '').strip(),
                visit_date=when,
                visit_time_recorded=has_time,
                flags=(r.get('data_flags') or '').strip(),
                source_sheet=(r.get('source_sheet') or '').strip(),
            ))
        return cls(_derive_closings(visits))

    def __len__(self) -> int:
        return len(self._visits)

    def stations(self) -> set[str]:
        return {v.station_id for v in self._visits}

    def opening(self, station_id: str, campaign: str) -> Optional[Visit]:
        """The visit that put this station's card in the ground for `campaign`."""
        return self._one(v for v in self._visits
                         if v.station_id == station_id and v.campaign_opened == campaign)

    def closing(self, station_id: str, campaign: str) -> Optional[Visit]:
        """The visit that pulled it out again."""
        return self._one(v for v in self._visits
                         if v.station_id == station_id and v.campaign_closed == campaign)

    @staticmethod
    def _one(matches) -> Optional[Visit]:
        """The earliest dated match. One station opens a campaign once, so more than
        one match is a workbook error; taking the earliest keeps it deterministic
        rather than dependent on row order."""
        found = sorted((v for v in matches if v.visit_date is not None),
                       key=lambda v: v.visit_date)
        return found[0] if found else None

    def window(self, station_id: str, campaign: str) -> Optional[tuple[datetime, datetime]]:
        """`[opening - tol, closing + tol]`, or None if either end is missing.

        Both ends are required: a half-open window would test one edge and silently
        pass everything beyond the other, which reads as a check while being none.
        """
        o, c = self.opening(station_id, campaign), self.closing(station_id, campaign)
        if o is None or c is None or o.visit_date is None or c.visit_date is None:
            return None
        return (o.visit_date - VISIT_WINDOW_TOLERANCE,
                c.visit_date + VISIT_WINDOW_TOLERANCE)


def deployment_window(
    station_id: str,
    campaign: str,
    anchors: list[Anchor],
    field: Optional[FieldRecord] = None,
) -> Optional[tuple[datetime, datetime]]:
    """The real-time window this station was deployed for.

    The field record is preferred and the anchors are the fallback, because the visit
    record covers every station while anchors exist only where a clock already broke.
    Before field notes, a station with fewer than two anchors got no window at all and
    its forward jumps were invisible — which was every station but CT18.
    """
    if field is not None:
        from_visits = field.window(station_id, campaign)
        if from_visits is not None:
            return from_visits

    reals = sorted({a.real_datetime for a in anchors if a.real_datetime is not None})
    if len(reals) < 2:
        return None
    return reals[0] - WINDOW_TOLERANCE, reals[-1] + WINDOW_TOLERANCE


# =============================================================================
# 3. Candidate evidence
# =============================================================================

# What kind of frame a candidate is. Defined here rather than in the report that
# emits them, because ranking them is an anchor decision and the two must not drift.
#
# NOTE THE TWO VOCABULARIES. `human_labelled` comes from `observationType` in the
# swept Timelapse2 export, which is Camtrap DP's controlled vocabulary and says
# `human`. `person_detection` comes from MegaDetector's own `detection_categories`
# map, which says `person`. They are different vocabularies owned by different
# modules (`camtrap/exports.py` and `camtrap/detections.py`); spelling them alike
# would claim an agreement that does not exist.
EVIDENCE_HUMAN_LABELLED    = 'human_labelled'
EVIDENCE_VEHICLE_LABELLED  = 'vehicle_labelled'
EVIDENCE_PERSON_DETECTION  = 'person_detection'
EVIDENCE_VEHICLE_DETECTION = 'vehicle_detection'
EVIDENCE_COUNTER_0001      = 'counter_0001'
EVIDENCE_SEGMENT_EDGE      = 'segment_edge'

# Strongest first. A `human` label is a person who looked at the frame and said so;
# a MegaDetector box is a guess that no one has confirmed. Ranking the confirmed
# evidence above the unconfirmed is the entire reason the sweep was worth doing.
#
# A person outranks a vehicle within each tier: the technician is always in the frame
# at a visit, whereas a vehicle may be anyone passing the road.
EVIDENCE_RANK = [
    EVIDENCE_HUMAN_LABELLED,
    EVIDENCE_VEHICLE_LABELLED,
    EVIDENCE_PERSON_DETECTION,
    EVIDENCE_VEHICLE_DETECTION,
    EVIDENCE_COUNTER_0001,
    EVIDENCE_SEGMENT_EDGE,
]

# WITNESS vs NAVIGATIONAL evidence — the distinction that keeps a visit date off a
# frame the technician was not in.
#
# A witness frame shows someone at the camera, so pairing it with a visit says "the
# clock read X at a moment we can date". A counter-0001 frame or a segment edge only
# says WHERE TO LOOK: it is the first file on a card, or the boundary of a run, and
# nothing about it asserts a person was present when it was taken.
#
# CT18 segment 0 is why this is a set and not a preference. Its only candidate is
# `11190001.JPG`, camera-time 2025-11-19 06:41, counter 0001 — and the install visit
# is 2025-11-14. Pairing them yields a -5 day offset, which would be applied to 10
# frames whose clock was CORRECT; the camera simply did not trigger for five days
# after install. That reproduces the 2026-08-03 finding mechanically: no photo
# corroborates CT18's install anchor, so segment 0 stays refused.
EVIDENCE_WITNESS = frozenset({
    EVIDENCE_HUMAN_LABELLED,
    EVIDENCE_VEHICLE_LABELLED,
    EVIDENCE_PERSON_DETECTION,
    EVIDENCE_VEHICLE_DETECTION,
})


def _rank(kind: str) -> int:
    return EVIDENCE_RANK.index(kind) if kind in EVIDENCE_RANK else len(EVIDENCE_RANK)


# =============================================================================
# 4. Proposals
# =============================================================================

READY        = 'READY'          # write it as-is
NEEDS_REVIEW = 'NEEDS_REVIEW'   # a human must settle it; refused meanwhile
NOT_NEEDED   = 'NOT_NEEDED'     # the clock is fine; an anchor would do harm

# How much the NOT_NEEDED verdict is worth: whether the in-window test actually ran.
VERIFIED   = 'window_checked'
UNVERIFIED = 'no_window_available'

PROPOSAL_COLUMNS = ANCHOR_WRITE_COLUMNS + ['status', 'evidence', 'why']


@dataclass(frozen=True)
class Proposal:
    """A candidate row for deployment_anchors.csv, plus why it is what it is.

    NEEDS_REVIEW proposals carry anchor_type `unrepairable_pending`, so promoting the
    whole file refuses the station explicitly rather than leaving it silently absent
    — a station missing from the anchor file and a station known to be unanchorable
    look identical downstream otherwise, and only one of them is a decision.
    """
    station_id: str
    anchor_type: str
    real_datetime: Optional[datetime]
    camera_datetime: Optional[datetime]
    source: str
    notes: str
    segment_index: Optional[int]
    status: str
    evidence: str
    why: str

    def as_row(self) -> dict:
        def fmt(d):
            return '' if d is None else d.strftime('%Y-%m-%d %H:%M:%S')
        return {
            'station_id': self.station_id,
            'anchor_type': self.anchor_type,
            'real_datetime': fmt(self.real_datetime),
            'camera_datetime': fmt(self.camera_datetime),
            'source': self.source,
            'notes': self.notes,
            'segment_index': '' if self.segment_index is None else self.segment_index,
            'status': self.status,
            'evidence': self.evidence,
            'why': self.why,
        }


def _pending(station_id: str, why: str, segment_index: Optional[int] = None) -> Proposal:
    return Proposal(
        station_id=station_id, anchor_type='unrepairable_pending',
        real_datetime=None, camera_datetime=None,
        source='field_notes.csv', notes=why, segment_index=segment_index,
        status=NEEDS_REVIEW, evidence='', why=why,
    )


def _best_candidate(candidates: pd.DataFrame, segment_index: int, *, earliest: bool):
    """The strongest WITNESS frame in a segment, tie-broken by position in time.

    Restricted to EVIDENCE_WITNESS: a frame that does not show someone at the camera
    cannot date a visit, however conveniently it sits at the start of a segment.
    `earliest` picks the install-side frame, otherwise the retrieval-side one. Ties
    within the best evidence kind are broken by time rather than left to row order,
    because row order is not a decision anyone made.
    """
    in_seg = candidates[candidates['clock_segment'] == segment_index]
    in_seg = in_seg[in_seg['camera_datetime'].notna()]
    in_seg = in_seg[in_seg['candidate_kind'].isin(EVIDENCE_WITNESS)]
    if in_seg.empty:
        return None
    ranked = in_seg.assign(_r=in_seg['candidate_kind'].map(_rank))
    best = ranked[ranked['_r'] == ranked['_r'].min()]
    best = best.sort_values('camera_datetime', ascending=earliest)
    return best.iloc[0]


def propose(
    diagnosis: ClockDiagnosis,
    campaign: str,
    field: FieldRecord,
    candidates: pd.DataFrame,
    existing: list[Anchor],
) -> list[Proposal]:
    """What anchors, if any, this station needs — and whether we can supply them.

    `candidates` is the anchor_candidates report filtered to THIS station. `existing`
    is what deployment_anchors.csv already says, so a segment already anchored by
    hand is left alone rather than re-proposed.
    """
    sid = diagnosis.station

    # A clean clock needs no anchor, and giving it one would apply the notebook's
    # imprecision as an offset. See the module docstring (CT01).
    if not diagnosis.has_clock_failure:
        # "Clean" and "clean as far as we could check" are different claims. Without a
        # deployment window the in-window test never ran, and a forward jump — a clock
        # set ahead — keeps every capture delta positive and shows up nowhere else.
        # Saying so is the difference between a verdict and an absence of evidence.
        checked = any(s.in_window is not None for s in diagnosis.segments)
        return [Proposal(
            station_id=sid, anchor_type='', real_datetime=None, camera_datetime=None,
            source='', notes='', segment_index=None, status=NOT_NEEDED,
            evidence=VERIFIED if checked else UNVERIFIED,
            why=(f'clock_clean: {len(diagnosis.segments)} coherent segment(s), no '
                 f'reset and nothing outside the deployment window' if checked else
                 f'clock_clean: {len(diagnosis.segments)} coherent segment(s) and no '
                 f'reset — but NO deployment window could be built for {campaign}, so '
                 f'a forward jump would not have been detected. Record this station\'s '
                 f'install and retrieval dates in {FIELD_NOTES_FILENAME} to close the '
                 f'gap'),
        )]

    if not diagnosis.ordered:
        return [_pending(
            sid,
            f'capture order not established (evidence={diagnosis.order_evidence}), so '
            f'no anchor can be placed in a segment — repair_plan refuses this station '
            f'whatever the field record says',
        )]

    opening = field.opening(sid, campaign)
    closing = field.closing(sid, campaign)
    if opening is None and closing is None:
        return [_pending(
            sid,
            f'{sid} has a clock failure and no visit recorded for {campaign} in '
            f'{FIELD_NOTES_FILENAME}; its install and retrieval dates must be '
            f'reconstructed before it can carry an anchor',
        )]

    # Which segments are already spoken for. Resolved through assign_anchors rather
    # than by reading `segment_index`, because most anchors leave it blank and find
    # their segment by containment — CT18's two hand-written rows both do, and
    # trusting the column alone would re-propose segments that are already anchored.
    already, _ = clocks.assign_anchors(diagnosis, existing)
    anchored = {i for i, rows in already.items() if rows}
    n = len(diagnosis.segments)
    out: list[Proposal] = []

    for s in diagnosis.segments:
        if not s.coherent:
            out.append(_pending(
                sid,
                f'segment {s.index} is incoherent ({s.reason}); an anchor repairs a '
                f'run of the clock, not a run that contradicts itself',
                segment_index=s.index,
            ))
            continue
        if s.index in anchored:
            continue

        # Which visit can date this segment. Only the two ends of the deployment are
        # known moments: the first segment starts at the opening visit and the last
        # ends at the closing one. An INTERIOR segment began at a reset nobody
        # witnessed, so only a mid-deployment visit could date it, and the workbook
        # records none — this is why §8.1's interior segments stay refused.
        if s.index == 0:
            visit, earliest, kind = opening, True, 'install'
        elif s.index == n - 1:
            visit, earliest, kind = closing, False, 'retrieval'
        else:
            out.append(_pending(
                sid,
                f'segment {s.index} of {n} is interior: it begins at an unwitnessed '
                f'reset, and only a mid-deployment visit could date it. None is '
                f'recorded for {campaign}',
                segment_index=s.index,
            ))
            continue

        if visit is None or not visit.datable:
            reason = ('no visit recorded' if visit is None or visit.visit_date is None
                      else f'the visit date is unsettled — {visit.flags}')
            out.append(_pending(
                sid,
                f'segment {s.index} needs the {kind} moment but {reason}',
                segment_index=s.index,
            ))
            continue

        frame = _best_candidate(candidates, s.index, earliest=earliest)
        if frame is None:
            out.append(_pending(
                sid,
                f'segment {s.index} has a datable {kind} visit '
                f'({visit.visit_date:%Y-%m-%d}) but no frame WITNESSING it — no '
                f'labelled human or vehicle inside the segment. What the clock read '
                f'at that visit is therefore unknown, and a counter-0001 frame does '
                f'not supply it: it marks where to look, not when it was taken',
                segment_index=s.index,
            ))
            continue

        # A date-only visit cannot support an exact anchor; see the module docstring.
        exact = visit.visit_time_recorded
        anchor_type = kind if exact else 'visit_date_only'
        precision = ('recorded to the minute' if exact
                     else f'date only — hour assumed {ASSUMED_VISIT_HOUR:%H:%M}, so '
                          f'valid_time_of_day stays FALSE')

        out.append(Proposal(
            station_id=sid,
            anchor_type=anchor_type,
            real_datetime=visit.visit_date,
            camera_datetime=frame['camera_datetime'].to_pydatetime(),
            source=f'{FIELD_NOTES_FILENAME} ({visit.source_sheet}) + {frame["file_name"]}',
            notes=(f'{kind} visit {visit.visit_date:%Y-%m-%d}, {precision}; paired '
                   f'with {frame["file_name"]} ({frame["candidate_kind"]}). '
                   f'VERIFY THE FRAME BY EYE before promoting.'),
            segment_index=s.index,
            status=NEEDS_REVIEW if not exact else READY,
            evidence=str(frame['candidate_kind']),
            why=(f'segment {s.index} of {n} paired with the {kind} visit; '
                 f'offset {visit.visit_date - frame["camera_datetime"].to_pydatetime()}'),
        ))

    return out or [_pending(sid, 'clock failure present but every segment is already '
                                 'anchored or refused')]


def to_frame(proposals: list[Proposal]) -> pd.DataFrame:
    """Proposals as a reviewable table, in PROPOSAL_COLUMNS order."""
    if not proposals:
        return pd.DataFrame(columns=PROPOSAL_COLUMNS)
    return pd.DataFrame([p.as_row() for p in proposals])[PROPOSAL_COLUMNS]
