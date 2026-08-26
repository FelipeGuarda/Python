"""
camtrap/episodes.py — when two detections are one event, and when they are two.

WHAT THIS OWNS

    The independence rule: how long a gap has to be before the same species at the
    same station counts as a second detection rather than a continuation of the
    first. That is one decision with four parts — the threshold, the key, where the
    gap is measured from, and where an episode is forbidden to continue — and it is
    the last preprocessing decision that was still being made downstream.

WHY IT MOVED UPSTREAM

    It existed three times: `Anual-reports/2025/py/01_data_prep.py`,
    `apply_verdicts.py`, and pehuen's `R/00_admissibility.R`. On 2026-08-26 two of
    those three disagreed — `01_data_prep.py` measures the gap from the last RETAINED
    detection (fixed 2026-08-20 to match camtrapR and pehuen), `apply_verdicts.py`
    still compared each row against its immediate predecessor. Measured over the
    canonical animal rows: **523 events the old way against 696 the correct way, a
    33% undercount**, in the script that writes `events_clean.parquet`.

    A rule that lives in its consumers drifts silently, because nothing compares the
    copies. Here it is computed once, at ingest, and travels in the table.

THE GAP IS MEASURED FROM THE LAST RETAINED DETECTION

    Detections at 0, 20 and 40 minutes are TWO events (0 and 40), not one. Comparing
    each row against its predecessor makes them one, because neither step exceeds 30
    minutes. This is the camtrapR definition and the one pehuen already uses.

AN EPISODE MAY NOT CROSS A CLOCK SEGMENT

    A segment boundary is a clock reset, so the interval across it is arithmetic on
    two different clocks and means nothing. Twenty minutes measured across a reset is
    not twenty minutes. Segments are known at ingest and are NOT in the canonical
    table, which is the other half of why this cannot be computed by a consumer.

WHY THE THRESHOLD IS IN THE COLUMN NAME

    `episode_30min`, not `episode_id`. A second threshold is a second column, never a
    parameter at read time, so two analyses cannot quietly compare figures built on
    different definitions of one event. Felipe's decision, 2026-08-24.

WHAT GETS NO EPISODE

    A row with no identified species, and a row with no clock — 419 animal rows carry
    a station and no datetime, and they are valid PRESENCE records. `pd.NA`, so a
    consumer counting `nunique()` cannot mistake them for an event of their own.

    Rows whose time-of-day is untrustworthy DO get episodes (33 rows, all
    `offset_from_last_real_proxy`). A whole-segment offset is constant, so relative
    spacing — all this rule uses — survives it even though absolute time-of-day does
    not. Excluding them would drop real detections from every count to protect a
    property the rule never reads. Felipe's decision, 2026-08-26.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

#: In the column name as well, deliberately. See the module docstring.
GAP_MINUTES = 30
GAP = timedelta(minutes=GAP_MINUTES)

COLUMN = f'episode_{GAP_MINUTES}min'

#: Everything the id is made of. The campaign is in there because `read_campaigns`
#: concatenates campaigns and `CT14|Puma concolor|1` would otherwise name a different
#: event in each one, making a cross-campaign `nunique()` undercount. Measured
#: 2026-08-26: keying WITH the campaign gives the same 696 events as keying without
#: it, because no two campaigns at one station are within 30 minutes of each other —
#: so this costs nothing and it is the only key computable at ingest, where one
#: campaign's table is written alone.
KEY = ('campaign', 'station_canonical', 'species_latin')


def label(frame: pd.DataFrame, segments: pd.Series | None = None) -> pd.Series:
    """One episode id per row — `otono_2026|CT14|Puma concolor|3` — or `pd.NA`.

    `segments` is the per-row clock segment, from timestamps.py. Passing None treats
    the frame as one segment, which is correct only for a frame with no clock
    failures; the ingest always passes it.
    """
    ids = pd.Series(pd.NA, index=frame.index, dtype='string')

    species = frame['species_latin'].fillna('').astype(str).str.strip()
    when = pd.to_datetime(frame['datetime'], errors='coerce')
    if segments is None:
        segment = pd.Series('', index=frame.index, dtype=object)
    else:
        segment = pd.Series(segments, index=frame.index).fillna('').astype(str).str.strip()

    eligible = species.ne('') & when.notna()
    if not eligible.any():
        return ids

    work = pd.DataFrame({
        'campaign': frame['campaign'].astype(str),
        'station_canonical': frame['station_canonical'].astype(str),
        'species_latin': species,
        'segment': segment,
        'when': when,
    })[eligible]

    # The counter runs per KEY and not per segment, so ids stay unique when a station
    # resets its clock: two segments would otherwise both produce episode 1.
    seen: dict[tuple, int] = {}
    for group, rows in work.groupby(list(KEY) + ['segment'], sort=False, observed=True):
        key = group[:len(KEY)]
        n = seen.get(key, 0)
        retained = None
        for index, moment in rows.sort_values('when')['when'].items():
            if retained is None or (moment - retained) >= GAP:
                n += 1
                retained = moment
            ids.at[index] = '|'.join(str(part) for part in key) + f'|{n}'
        seen[key] = n

    return ids


def count(frame: pd.DataFrame) -> int:
    """Independent detections in `frame`. Rows with no episode are not events."""
    return int(frame[COLUMN].nunique(dropna=True))
