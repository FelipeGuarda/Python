"""How a camera's clock failure is characterised, and what may be concluded from it.

This module owns four questions and nothing else:

    1. What evidence establishes the order in which frames were captured?  (P1)
    2. Does a camera's datetime sequence tick coherently?                  (P2)
    3. Where are the segment boundaries — i.e. where did the clock reset?
    4. Which segments can be repaired, and what may downstream trust?

It deliberately does NOT own anchor storage, CSV/parquet I/O, offset application or
audit rendering — `timestamps.py` keeps those. The split is by knowledge, not by
time: an earlier design detected resets in one place and repaired them in another,
and the repairability rule leaked into the caller. That is precisely how otoño 2026
CT_18 came to have a single offset applied across four separate resets, fabricating
most of its datetimes and putting 65 wrong-dated records into the pehuen analysis.

THE RULE (agreed 2026-07-30, after two weaker criteria were rejected)

    A segment is repairable if and only if it is COHERENT and contains AT LEAST ONE
    ANCHOR. The number of repairable segments equals the number of segments an
    anchor falls inside.

    Anchors come from the install photo, every mid-deployment maintenance visit, and
    the retrieval photo — each needing a real wall-clock datetime recorded in the
    field AND an identifiable frame in the image sequence. The lever this exposes is
    that anchors are cheap and each one buys back a whole segment.

    Explicitly rejected as criteria:
      - Segment count. A 2-segment camera bracketed by install and retrieval anchors
        is fully repairable, so counting segments says nothing.
      - "Slack" S = window − Σ(segment durations). It assumes the camera rebooted
        promptly after each power loss, which is unprovable. `unaccounted_days`
        survives here as an AUDIT DIAGNOSTIC ONLY and must never gate validity.

PRECONDITIONS — both fail closed

    P1 ordering established. Order comes from the DCIM folder plus the filename
       counter. The counter alone is not enough: it is per-folder and wraps at 999,
       so the five otoño 2026 cameras with >999 images have colliding counters, and
       sorting on the counter alone once produced 987 phantom resets for CT_14.
    P2 segment coherence. Inside a segment the filename's MMDD must agree with its
       own DateTime. CT_18 fails this on 166 of 312 frames — its month/day registers
       are corrupt, so its datetimes may not tick coherently even within a segment.
       A camera that fails P2 is not repairable from an offset, however many anchors
       it has, which is what distinguishes it from otoño 2025 CT15's clean +8 year
       error where the filename still encodes the true month and day.

    Note the asymmetry: failing P1 does NOT by itself condemn a camera. A camera
    whose datetimes all sit inside the deployment window and whose filenames agree
    with their DateTime has demonstrably not reset, whether or not we can order its
    frames — ordering is needed to ATTRIBUTE a split, not to rule one out. This is
    what keeps otoño 2026's five wrap cameras usable; they were flattened before
    anyone knew a manifest was needed and can never satisfy P1.

THREE INDEPENDENT FLAGS

    valid_date          is the date trustworthy?
    valid_time_of_day   is the time-of-day trustworthy?
    valid_effort        are this station's trap-nights knowable? (station-level)

    They must stay independent. A pure year error (2017 for 2025, same MM-DD
    HH:MM:SS) preserves time-of-day exactly, so those frames remain valid for
    activity and overlap analysis before anyone fixes the year; conversely a
    `last_real_proxy` offset rotates time-of-day while roughly preserving date
    order. A single usable/not-usable switch would throw away recoverable data.

    `valid_effort` is station-level on purpose. A camera that died at an unknown
    date has unknowable trap-nights, so it must leave the effort DENOMINATOR as
    well as the numerator — it contributes no effort for any species at that
    station, including for the segments whose dates are fine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

# ── The DCIM manifest ─────────────────────────────────────────────────────────
# Written by setup/flatten_for_camtrapdp.py, read here. The schema lives with the
# reader because this module is the only thing that cares what it means.

DCIM_MANIFEST_FILENAME = 'dcim_manifest.csv'
DCIM_MANIFEST_COLUMNS = [
    'deployment', 'dcim_folder', 'original_name', 'original_relpath',
    'flat_name', 'size_bytes', 'mtime', 'action',
]

# ── Order evidence ────────────────────────────────────────────────────────────

ORDER_MANIFEST = 'dcim_manifest+counter'   # strongest: folder then counter
ORDER_COUNTER  = 'counter'                 # filename counter alone
ORDER_NONE     = 'none'                    # no usable evidence

# ── Anchors ───────────────────────────────────────────────────────────────────

ANCHOR_TYPES_EXACT        = {'install', 'mid_visit', 'retrieval'}
ANCHOR_TYPES_APPROXIMATE  = {'last_real_proxy'}
ANCHOR_TYPES_UNREPAIRABLE = {'unrepairable_pending'}
ALL_ANCHOR_TYPES = (
    ANCHOR_TYPES_EXACT | ANCHOR_TYPES_APPROXIMATE | ANCHOR_TYPES_UNREPAIRABLE
)

# Retained for the legacy audit trail only. This was the whole of the old
# detection rule ("year < 2024 is bogus") and it cannot see a forward jump — a
# clock set to 2030, or a subtle 2025→2024 slip. Detection now works off capture
# order and the deployment window instead. Do not reintroduce it as a criterion.
BOGUS_YEAR_THRESHOLD = 2024

# ── Filename grammar ──────────────────────────────────────────────────────────
# `01190313.JPG` = MMDD 01-19, counter 0313. Collision-renamed frames carry the
# DCIM folder as a prefix (`102EK113_01190313.JPG`), so anchor the match at the end.
_FILENAME_RE = re.compile(r'(?P<mmdd>\d{4})(?P<counter>\d{4})$')

# Videos are excluded from every chronology decision: their DateTime is stamped an
# hour off from the paired JPG and some carry the file-copy date instead. Including
# them produced 61 phantom resets for CT_18 against a true count of 4. Both .MP4
# and .MOV occur in otoño 2026, so filter TO stills rather than excluding one
# extension.
STILL_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'})


# =============================================================================
# Value types
# =============================================================================

@dataclass(frozen=True)
class Anchor:
    """What one field observation asserts about a camera's clock.

    `camera_datetime` is what the clock said at the anchor moment — the EXIF stamp
    of the trigger frame. `real_datetime` is the wall clock. Both are None for
    `unrepairable_pending`, which records a known problem with no field data yet.
    """
    station_id: str
    anchor_type: str
    real_datetime: datetime | None
    camera_datetime: datetime | None
    source: str = ''
    notes: str = ''
    segment_index: int | None = None   # explicit override; see assign_anchors()

    @property
    def exact(self) -> bool:
        """False => the date may be recoverable but the time-of-day is not."""
        return self.anchor_type in ANCHOR_TYPES_EXACT

    @property
    def offset(self) -> timedelta | None:
        if self.real_datetime is None or self.camera_datetime is None:
            return None
        return self.real_datetime - self.camera_datetime


@dataclass(frozen=True)
class Segment:
    """A run of frames over which the clock ticked without resetting.

    `camera_start`/`camera_end` are in CAMERA time, so two segments of the same
    camera can overlap or even coincide — otoño 2026 CT_18 has three segments that
    all begin at 2017-01-01. That is why contains() can match more than one
    segment and why an ambiguous anchor repairs nothing.
    """
    index: int
    n_images: int
    camera_start: datetime
    camera_end: datetime
    coherent: bool                    # precondition P2
    in_window: bool | None = None     # None => no deployment window was supplied
    reason: str = ''                  # why it is incoherent, when it is

    def contains(self, camera_dt: datetime) -> bool:
        return self.camera_start <= camera_dt <= self.camera_end

    @property
    def duration_days(self) -> float:
        return (self.camera_end - self.camera_start).total_seconds() / 86400.0


@dataclass(frozen=True)
class ClockDiagnosis:
    """Facts about one camera's clock. No verdicts — see repair_plan()."""
    station: str
    ordered: bool                     # precondition P1
    order_evidence: str               # ORDER_MANIFEST | ORDER_COUNTER | ORDER_NONE
    segments: list[Segment]
    unaccounted_days: float | None = None   # AUDIT DIAGNOSTIC ONLY — never a criterion
    n_stills: int = 0
    n_videos_excluded: int = 0
    n_unparseable: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def has_clock_failure(self) -> bool:
        return (
            len(self.segments) > 1
            or any(not s.coherent for s in self.segments)
            or any(s.in_window is False for s in self.segments)
        )


@dataclass(frozen=True)
class SegmentRepair:
    """The decision for one segment.

    `offset is None` means "apply nothing" — either because the clock was already
    correct or because the segment is unrepairable. The flags carry the verdict;
    `reason` names the rule that fired.
    """
    segment_index: int
    offset: timedelta | None
    valid_date: bool
    valid_time_of_day: bool
    valid_effort: bool
    reason: str


# =============================================================================
# Filename grammar and capture order (P1)
# =============================================================================

def parse_filename(name: str) -> tuple[str | None, int | None]:
    """`01190313.JPG` -> ('0119', 313). Returns (None, None) if it does not match.

    Hand-renamed frames exist (`01060117_fiscalizador.JPG` in otoño 2026 CT_27)
    and are not a convention we can rely on, so they simply fail to parse.
    """
    stem = name.rsplit('.', 1)[0] if '.' in name else name
    m = _FILENAME_RE.search(stem)
    if not m:
        return None, None
    return m.group('mmdd'), int(m.group('counter'))


def is_still(name: str) -> bool:
    ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
    return f'.{ext}' in STILL_EXTENSIONS


def establish_order(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, str, list[str]]:
    """Sort frames into capture order and report how strongly that order is known.

    Expects columns `_mmdd`, `_counter` and optionally `dcim_folder`.
    Returns (sorted_df, ordered, evidence, notes).

    With a DCIM folder the counter only has to be unique WITHIN each folder, which
    is what makes >999-image cameras orderable at all. Without one, a repeated
    counter means the order is genuinely unknown and we say so rather than
    inventing a sequence — sorting CT_14's wrapped counters once yielded 987
    phantom resets.
    """
    notes: list[str] = []
    n_unparsed = int(df['_counter'].isna().sum())

    has_folder = False
    if 'dcim_folder' in df.columns:
        folder = df['dcim_folder'].fillna('').astype(str).str.strip()
        folder = folder.mask(folder.isin({'nan', 'None'}), '')
        has_folder = bool(folder.ne('').any())
        if has_folder:
            # A manifest that covers only some frames is worse than none: the
            # described frames would sort under their folder and the rest would pool
            # together, producing a confident but wrong order. Frames flattened
            # before the manifest existed are recorded with an empty dcim_folder
            # precisely so this check can see them.
            n_missing = int(folder.eq('').sum())
            if n_missing:
                notes.append(
                    f'{n_missing} frame(s) have no DCIM folder while others do — the '
                    f'manifest describes this deployment only partially, so capture '
                    f'order cannot be trusted'
                )
                return df, False, ORDER_MANIFEST, notes

    if n_unparsed:
        # Even one unparseable counter means the sequence has a hole we cannot
        # place, so order is not established. Return unsorted — a partially
        # sortable column would only invite someone to trust the result.
        notes.append(
            f'{n_unparsed} filename(s) do not match the MMDD+counter grammar, '
            f'so no counter order can be derived from them'
        )
        if has_folder:
            evidence = ORDER_MANIFEST
        else:
            evidence = ORDER_NONE if n_unparsed == len(df) else ORDER_COUNTER
        return df, False, evidence, notes

    if has_folder:
        ordered_df = df.sort_values(['dcim_folder', '_counter'], kind='stable')
        dupes = int(ordered_df.duplicated(subset=['dcim_folder', '_counter']).sum())
        ordered = (n_unparsed == 0) and dupes == 0
        if dupes:
            notes.append(f'{dupes} (dcim_folder, counter) pair(s) collide')
        return ordered_df, ordered, ORDER_MANIFEST, notes

    ordered_df = df.sort_values('_counter', kind='stable')
    dupes = int(ordered_df.duplicated(subset=['_counter']).sum())
    if dupes:
        notes.append(
            f'{dupes} colliding filename counter(s) and no DCIM manifest — the '
            f'counter is per-folder and wraps at 999, so capture order is not '
            f'recoverable from filenames alone'
        )
    ordered = (n_unparsed == 0) and dupes == 0
    return ordered_df, ordered, ORDER_COUNTER, notes


# =============================================================================
# Diagnosis
# =============================================================================

def diagnose(
    images: pd.DataFrame,
    station: str,
    window: tuple[datetime, datetime] | None = None,
) -> ClockDiagnosis:
    """Characterise one camera's clock from its frames.

    `images` needs two columns and may carry a third:
        file_name        as written on the card (post-flatten is fine)
        camera_datetime  parsed datetime as the camera stamped it (NaT allowed)
        dcim_folder      from dcim_manifest.csv, when one exists

    `window` is the real-time deployment window (install, retrieval). Supplying it
    is what allows a FORWARD clock jump to be detected: a clock set to 2030 keeps
    every delta positive and is invisible to backwards-step detection alone.

    Pass frames for ONE station. Column naming is the caller's job — this module
    does not decode Timelapse2 or any other vendor format.
    """
    notes: list[str] = []
    df = images.copy()

    for col in ('file_name', 'camera_datetime'):
        if col not in df.columns:
            raise ValueError(f'diagnose() requires a {col!r} column')

    # Trap 1 — videos never take part in a chronology decision.
    mask_still = df['file_name'].astype(str).map(is_still)
    n_videos = int((~mask_still).sum())
    df = df[mask_still]

    df['camera_datetime'] = pd.to_datetime(df['camera_datetime'], errors='coerce')
    n_unparseable = int(df['camera_datetime'].isna().sum())
    if n_unparseable:
        notes.append(f'{n_unparseable} frame(s) have an unparseable datetime and are excluded')
    df = df[df['camera_datetime'].notna()]

    n_stills = len(df)
    if n_stills == 0:
        notes.append('no still frames with a parseable datetime — nothing to diagnose')
        return ClockDiagnosis(
            station=station, ordered=False, order_evidence=ORDER_NONE, segments=[],
            n_stills=0, n_videos_excluded=n_videos, n_unparseable=n_unparseable,
            notes=notes,
        )

    parsed = df['file_name'].astype(str).map(parse_filename)
    df['_mmdd']    = [p[0] for p in parsed]
    df['_counter'] = [p[1] for p in parsed]

    df, ordered, evidence, order_notes = establish_order(df)
    notes.extend(order_notes)

    # P2 — the filename's own MMDD must agree with the DateTime it carries.
    stamp_mmdd = df['camera_datetime'].dt.strftime('%m%d')
    mismatch = df['_mmdd'].notna() & (df['_mmdd'] != stamp_mmdd)
    n_mismatch = int(mismatch.sum())
    if n_mismatch:
        notes.append(
            f'{n_mismatch}/{n_stills} frame(s) disagree between filename MMDD and '
            f'DateTime — the date registers are corrupt, so the clock may not tick '
            f'coherently even within a segment'
        )

    in_window = None
    if window is not None:
        start, end = window
        in_window = (df['camera_datetime'] >= start) & (df['camera_datetime'] <= end)
        n_out = int((~in_window).sum())
        if n_out:
            notes.append(f'{n_out} frame(s) fall outside the deployment window')

    # ── Split detection (§5.3) ────────────────────────────────────────────────
    if ordered:
        # A split is a datetime that moves backwards relative to capture order, OR
        # a crossing of the deployment-window boundary (the forward-jump case).
        breaks = df['camera_datetime'].diff() < pd.Timedelta(0)
        if in_window is not None:
            crossing = in_window.ne(in_window.shift())
            crossing.iloc[0] = False        # the first frame cannot be a break
            breaks = breaks | crossing
        seg_ids = breaks.cumsum()
    else:
        # Order unknown, so a split cannot be located. We can still rule one OUT:
        # every frame inside the window with a filename that agrees with its own
        # stamp is a camera that demonstrably never reset.
        clean = (n_mismatch == 0) and (in_window is None or bool(in_window.all()))
        seg_ids = pd.Series(0, index=df.index)
        if clean:
            notes.append(
                'capture order not established, but no clock failure is detectable: '
                'every frame is in-window and agrees with its own filename'
            )
        else:
            notes.append(
                'capture order not established AND the datetimes are suspect — the '
                'split cannot be located, so the camera fails closed'
            )

    segments: list[Segment] = []
    for pos, (seg_id, grp) in enumerate(df.groupby(seg_ids, sort=True)):
        seg_mismatch = int(mismatch.loc[grp.index].sum())
        seg_in_window = (
            None if in_window is None else bool(in_window.loc[grp.index].all())
        )
        if not ordered and not (
            seg_mismatch == 0 and (seg_in_window is None or seg_in_window)
        ):
            coherent, reason = False, 'ordering_unrecoverable'
        elif seg_mismatch:
            coherent, reason = False, 'filename_mmdd_disagrees_with_datetime'
        else:
            coherent, reason = True, ''

        segments.append(Segment(
            index=pos,
            n_images=len(grp),
            camera_start=grp['camera_datetime'].min().to_pydatetime(),
            camera_end=grp['camera_datetime'].max().to_pydatetime(),
            coherent=coherent,
            in_window=seg_in_window,
            reason=reason,
        ))

    # Audit diagnostic only (§5.6). Reported, never used to decide anything.
    unaccounted = None
    if window is not None:
        span = (window[1] - window[0]).total_seconds() / 86400.0
        unaccounted = round(span - sum(s.duration_days for s in segments), 2)

    return ClockDiagnosis(
        station=station,
        ordered=ordered,
        order_evidence=evidence,
        segments=segments,
        unaccounted_days=unaccounted,
        n_stills=n_stills,
        n_videos_excluded=n_videos,
        n_unparseable=n_unparseable,
        notes=notes,
    )


# =============================================================================
# Repair planning
# =============================================================================

def assign_anchors(
    d: ClockDiagnosis,
    anchors: list[Anchor],
) -> tuple[dict[int, list[Anchor]], list[str]]:
    """Map each anchor to the segment it falls inside. Strict containment.

    An anchor that matches no segment, or more than one, is unusable and repairs
    nothing. Both cases are real:

      - CT_18's install anchor asserts camera-time 2025-11-14 14:00, but the first
        frame on the card is 2025-11-19 06:41, so it falls inside no segment. That
        is the honest answer: the anchor is uncorroborated by any photo, which is
        why CT_18 has zero verified anchors.
      - CT_18's segments 1, 2 and 3 all start at 2017-01-01, so an anchor stamped
        in early January matches several of them at once.

    An anchor row may name its segment explicitly via `segment_index`, which is the
    escape hatch for both cases once someone has verified which segment it belongs
    to by eye.
    """
    by_segment: dict[int, list[Anchor]] = {}
    notes: list[str] = []

    for a in anchors:
        if a.segment_index is not None:
            if any(s.index == a.segment_index for s in d.segments):
                by_segment.setdefault(a.segment_index, []).append(a)
            else:
                notes.append(
                    f'anchor {a.anchor_type} names segment_index={a.segment_index}, '
                    f'which does not exist ({len(d.segments)} segment(s) found)'
                )
            continue

        if a.camera_datetime is None:
            continue   # unrepairable_pending — handled by repair_plan()

        matches = [s for s in d.segments if s.contains(a.camera_datetime)]
        if len(matches) == 1:
            by_segment.setdefault(matches[0].index, []).append(a)
        elif not matches:
            notes.append(
                f'anchor {a.anchor_type} (camera time {a.camera_datetime}) falls '
                f'inside no segment, so it corroborates nothing'
            )
        else:
            notes.append(
                f'anchor {a.anchor_type} (camera time {a.camera_datetime}) is '
                f'ambiguous — it falls inside segments '
                f'{[s.index for s in matches]}, which overlap in camera time. '
                f'Set segment_index on the anchor row to resolve it.'
            )

    return by_segment, notes


def _choose(anchors: list[Anchor]) -> Anchor:
    """Prefer an exact anchor over an approximate one; otherwise first wins."""
    for a in anchors:
        if a.exact:
            return a
    return anchors[0]


def repair_plan(
    d: ClockDiagnosis,
    anchors: list[Anchor],
) -> tuple[list[SegmentRepair], list[str]]:
    """Decide, per segment, what may be applied and what downstream may trust.

    Returns (repairs, notes). Repairs are returned in segment order; a caller maps
    them back onto rows via the segment each frame belongs to.
    """
    notes: list[str] = []

    if not d.segments:
        return [], ['no segments to repair']

    def _all(reason: str, *, ok: bool = False) -> list[SegmentRepair]:
        return [
            SegmentRepair(
                segment_index=s.index, offset=None,
                valid_date=ok, valid_time_of_day=ok, valid_effort=ok,
                reason=reason,
            )
            for s in d.segments
        ]

    # A pending anchor is an explicit statement that the field data is not in yet.
    # It outranks everything, including a clean-looking sequence.
    pending = [a for a in anchors if a.anchor_type in ANCHOR_TYPES_UNREPAIRABLE]
    if pending:
        notes.append(
            f'{d.station}: unrepairable_pending anchor present '
            f'({pending[0].notes or pending[0].source or "no note"}) — awaiting field info'
        )
        return _all('unrepairable_pending_anchor'), notes

    # The common case, and the one that keeps the flattened wrap cameras usable:
    # nothing is wrong, so no anchor is needed and ordering is irrelevant.
    if not d.has_clock_failure:
        return _all('clock_clean', ok=True), notes

    if not d.ordered:
        notes.append(
            f'{d.station}: clock failure present but capture order is not '
            f'established (evidence={d.order_evidence}) — no repair reasoning applies'
        )
        return _all('ordering_unrecoverable'), notes

    by_segment, assign_notes = assign_anchors(d, anchors)
    notes.extend(f'{d.station}: {n}' for n in assign_notes)

    repairs: list[SegmentRepair] = []
    for s in d.segments:
        if not s.coherent:
            repairs.append(SegmentRepair(
                segment_index=s.index, offset=None,
                valid_date=False, valid_time_of_day=False, valid_effort=False,
                reason=f'segment_incoherent:{s.reason}',
            ))
            continue

        seg_anchors = by_segment.get(s.index, [])
        if not seg_anchors:
            repairs.append(SegmentRepair(
                segment_index=s.index, offset=None,
                valid_date=False, valid_time_of_day=False, valid_effort=False,
                reason='no_anchor_in_segment',
            ))
            continue

        chosen = _choose(seg_anchors)
        repairs.append(SegmentRepair(
            segment_index=s.index,
            offset=chosen.offset,
            valid_date=True,
            # An approximate anchor (`last_real_proxy`) pins the date roughly but
            # rotates the time-of-day, so activity analysis must not see it.
            valid_time_of_day=chosen.exact,
            valid_effort=False,          # replaced below, once every segment is known
            reason=f'offset_from_{chosen.anchor_type}',
        ))

    # valid_effort is a property of the STATION, not of a segment: if any segment's
    # dates are unknown then the camera's operating period is unknown, so its
    # trap-nights are unknowable and it must leave the effort denominator too —
    # even for the segments whose own dates are fine.
    effort_ok = all(r.valid_date for r in repairs)
    if not effort_ok:
        unknown = [r.segment_index for r in repairs if not r.valid_date]
        notes.append(
            f'{d.station}: segment(s) {unknown} have unknown dates, so the station '
            f'contributes no effort at all — exclude it from rate denominators, '
            f'not just numerators'
        )
    if d.unaccounted_days:
        notes.append(
            f'{d.station}: {d.unaccounted_days} day(s) of the deployment window are '
            f'unaccounted for by any segment (diagnostic only)'
        )

    return (
        [
            SegmentRepair(
                segment_index=r.segment_index, offset=r.offset,
                valid_date=r.valid_date, valid_time_of_day=r.valid_time_of_day,
                valid_effort=effort_ok, reason=r.reason,
            )
            for r in repairs
        ],
        notes,
    )
