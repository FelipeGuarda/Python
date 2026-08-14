"""The FMA camera-station naming convention.

Canonical form is `CT01` .. `CT27` — the letters `CT`, then the camera number
zero-padded to two digits. Nothing else. The monitoring-grid ID is NOT part of a
station name (grid `M15.2` holds cameras 11 and 18, so it identifies a place, not a
camera); it lives in `station_aliases.csv` for the historical campaigns and belongs in
a station registry going forward.

Timelapse2 derives its `Deployments` column from the image folder name, so this
convention is really a *folder* convention — enforce it on Synology at Step 1b
(`setup/flatten_for_camtrapdp.py --check-stations`) and every downstream consumer gets
canonical stations for free.

Historical campaigns predate the convention and used three other spellings
(`CT_01`, `TC1_M7.2`, plus one unrenamed SD-card folder `100EK113`). Those are
resolved through `data/campaigns/station_aliases.csv` — a frozen data file, not code.
New campaigns must be canonical; `resolve()` raises rather than guessing, because a
station silently dropped for being unrecognised is exactly how 252 rows of camera 5
went missing from the 2025 annual report for a year.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

CANONICAL_PATTERN = r"^CT\d{2}$"
_CANONICAL_RE = re.compile(CANONICAL_PATTERN)

_ALIAS_CSV = Path(__file__).resolve().parents[1] / "data" / "campaigns" / "station_aliases.csv"


class UnknownStation(ValueError):
    """A station name is neither canonical nor a known historical alias."""


def canonical_id(camera_num: int) -> str:
    """5 -> 'CT05'. The one place the canonical spelling is constructed."""
    return f"CT{camera_num:02d}"


def is_canonical(station_raw: str) -> bool:
    return bool(_CANONICAL_RE.match(station_raw.strip()))


# Every spelling this project has used for a station, as a SHAPE: canonical `CT23`,
# otoño 2026's `CT_23`, and the TC-with-grid form `TC23_M20.2`. Recognition only —
# see names_a_station.
_STATION_SHAPE_RE = re.compile(r'^(?:CT|TC)[_-]?\d{1,2}(?:_M[0-9.]+)?$', re.IGNORECASE)


def names_a_station(folder_name: str) -> bool:
    """Does this folder name name a camera station, in any spelling we have used?

    RECOGNISES, NEVER RESOLVES. It returns a bool precisely so it cannot become a
    second route from a name to a camera number — `resolve()` and
    `station_aliases.csv` remain the only one, because a name this matches by shape
    may still belong to no camera we know.

    Matched by SHAPE rather than by membership in the alias table, for one concrete
    reason: `100EK113` IS an alias row (an unrenamed SD-card folder that became
    primavera_2025's camera 5), so a membership test would call every DCIM folder a
    station and refuse every deployment that contains one. The shape test excludes
    it without needing to know what a DCF folder is — that stays owned by
    `clocks.dcim_folder_key`.

    DELIBERATELY NARROW, AND NOT THE ONLY GUARD. Recognising by shape means it knows
    the three spellings this project has used and no others: a folder called
    `Camara 23` or `Cam23` walks straight past it. That is an accepted limit rather
    than a gap, because `camtrap/provenance.multiple_capture_stories()` sits behind
    it and recognises a second camera from its FRAMES, enumerating nothing. The two
    are complementary — this one can say *which folder to move*, which the general
    check cannot; the general one catches names nobody has thought of, which this
    one cannot. Do not widen this regex to chase a name the other check already
    covers.

    Used by `setup/flatten_for_camtrapdp.py` to refuse a deployment with a station
    folder nested inside it. Primavera 2025 had `TC23_M20.2` — 2,460 files, a whole
    camera — sitting inside `TC22_M19.2`; flattening would have attributed all of
    them to camera 22, at camera 22's coordinates. Nothing caught it: the two use
    different filename schemes, so there were zero collisions and the conservation
    check passed. Conservation and ordering were checked; attribution was not.
    """
    return bool(_STATION_SHAPE_RE.match(folder_name.strip()))


@lru_cache(maxsize=1)
def _aliases() -> dict[tuple[str, str], int]:
    """{(campaign, station_raw): camera_num} for the pre-convention campaigns."""
    if not _ALIAS_CSV.exists():
        return {}
    with open(_ALIAS_CSV, encoding="utf-8", newline="") as f:
        return {
            (row["campaign"].strip(), row["station_raw"].strip()): int(row["camera_num"])
            for row in csv.DictReader(f)
            if row.get("camera_num", "").strip()
        }


def resolve(station_raw: str, campaign: str) -> int:
    """Station name (canonical or historical) -> camera number.

    Raises UnknownStation if it is neither. Callers must not swallow this: an
    unrecognised station means real detections are about to go missing.
    """
    station = station_raw.strip()
    m = _CANONICAL_RE.match(station)
    if m:
        return int(station[2:])
    alias = _aliases().get((campaign, station))
    if alias is not None:
        return alias
    raise UnknownStation(
        f"station {station_raw!r} in campaign {campaign!r} is neither canonical "
        f"({CANONICAL_PATTERN}) nor listed in {_ALIAS_CSV.name}. "
        f"Rename the source folder to canonical form, or add an alias row."
    )


def validate(station_raws: Iterable[str], campaign: str) -> None:
    """Raise once, listing every offender, instead of failing on the first."""
    offenders = sorted(
        {
            s.strip()
            for s in station_raws
            if s.strip() and not is_canonical(s) and (campaign, s.strip()) not in _aliases()
        }
    )
    if offenders:
        raise UnknownStation(
            f"{len(offenders)} unrecognised station(s) in campaign {campaign!r}: "
            f"{offenders}. Expected {CANONICAL_PATTERN}."
        )


def non_canonical(station_raws: Iterable[str]) -> list[str]:
    """Stations that resolve but do not yet follow the convention — for warnings."""
    return sorted({s.strip() for s in station_raws if s.strip() and not is_canonical(s)})
