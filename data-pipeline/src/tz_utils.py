"""Single canonical strategy for localizing America/Santiago timestamps to UTC.

DST policy:
  - nonexistent="shift_forward" everywhere (spring-forward gap → first valid time).
  - For sequences (Series / DatetimeIndex / array): try ambiguous="infer"; on
    AmbiguousTimeError fall back to ambiguous=False (= standard time, second
    occurrence after fall-back).
  - For scalar Timestamp: always ambiguous=False (pandas can't infer from a
    single point).

Replaces the five inconsistent `ambiguous=` strategies that previously lived
across cr800.py, open_meteo.py, toa5.py, met_csv.py, and timelapse_reviewed.py.
"""

from __future__ import annotations

import pandas as pd

_SANTIAGO = "America/Santiago"


def localize_santiago_to_utc(naive):
    """
    Localize a naive timestamp to America/Santiago, then convert to UTC.

    Accepts: pd.Timestamp | pd.Series | pd.DatetimeIndex | array-like.
    Returns the same container type, in UTC.
    """
    if isinstance(naive, pd.Timestamp):
        return (
            naive.tz_localize(_SANTIAGO, ambiguous=False, nonexistent="shift_forward")
            .tz_convert("UTC")
        )

    if isinstance(naive, pd.Series):
        idx = pd.DatetimeIndex(pd.to_datetime(naive, errors="coerce"))
        utc = _localize_index(idx)
        return pd.Series(utc, index=naive.index)

    if isinstance(naive, pd.DatetimeIndex):
        return _localize_index(naive)

    idx = pd.DatetimeIndex(pd.to_datetime(naive, errors="coerce"))
    return _localize_index(idx)


def _localize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    try:
        localized = idx.tz_localize(
            _SANTIAGO, ambiguous="infer", nonexistent="shift_forward"
        )
    except Exception:
        localized = idx.tz_localize(
            _SANTIAGO, ambiguous=False, nonexistent="shift_forward"
        )
    return localized.tz_convert("UTC")
