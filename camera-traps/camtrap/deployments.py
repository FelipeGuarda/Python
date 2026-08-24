"""How long each camera was actually in the ground, per campaign.

WHAT THIS OWNS
    One decision: what a deployment window IS. It pairs the visit that put a card in
    the ground with the visit that pulled it out, and publishes the result next to the
    campaign's canonical table so a consumer never has to open `field_notes.csv` and
    re-derive it.

WHY IT IS PUBLISHED RATHER THAN COMPUTED DOWNSTREAM
    A deployment window is the DENOMINATOR of every detection rate. A wrong row count
    is visible -- a species appears or it does not. A wrong denominator silently
    rescales every rate in a report and nothing looks broken. It is therefore decided
    once, here, in the repository that owns the field record, and copied verbatim by
    everyone else.

THE OBSERVED WINDOW IS NOT THE FIELD WINDOW, AND THAT IS THE WHOLE POINT
    Before this file existed, consumers inferred "how long was it watching" from the
    first and last photograph. That is circular: a camera whose battery died after two
    months looks like it was DEPLOYED for two months, so its detection rate comes out
    inflated. Measured on otono_2025: CT12 was in the ground 219 days and photographed
    across 61 of them -- a 3.6x overstatement -- and CT08 and CT10 have no observed
    window at all because their clocks failed, while the field record dates both.

DO NOT USE FieldRecord.window() HERE
    That method returns `[opening - 3d, closing + 3d]`. The padding exists so a clock
    anchor can be validated against a window it might sit just outside of; applied to
    effort it would add six days to every camera. This module uses the raw
    `opening()` / `closing()` visit dates and nothing else.

STATIONS WITH A WINDOW AND NO IMAGES ARE PUBLISHED, NOT DROPPED
    otono_2025 records CT21, CT22, CT24, CT25 and CT26 as installed in February 2025
    and collected in June, and not one of them appears in that campaign's DCIM
    manifest, image data or reviewed CSV. Whether the cards were never downloaded or
    the cameras never fired is unresolved (Felipe is checking the NAS, 2026-08-24).
    Publishing them with `has_media=false` keeps roughly 620 camera-days visible
    instead of letting them read as stations that never existed. A consumer that wants
    only real deployments filters on `has_media`; a consumer that wants effort has to
    make that choice deliberately, which is the point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from camtrap.anchors import FIELD_NOTES_FILENAME, FieldRecord
from camtrap.observations import CAMPAIGNS_ROOT, CANONICAL_FILENAME

#: Written next to `observations.parquet`, one file per campaign.
DEPLOYMENTS_FILENAME = "deployments.csv"

COLUMNS = (
    "campaign",
    "station_id",
    "field_start",     # date the card went in, from the field record
    "field_end",       # date it came out
    "field_days",      # field_end - field_start, in days
    "has_media",       # does this station appear in the campaign's canonical table?
    "note",
)


def _media_stations(campaign: str, root: Path) -> set[str]:
    parquet = root / campaign / CANONICAL_FILENAME
    if not parquet.exists():
        return set()
    df = pd.read_parquet(parquet, columns=["station_canonical"])
    return set(df["station_canonical"].dropna().unique())


def build(campaign: str, *, root: Path = CAMPAIGNS_ROOT) -> pd.DataFrame:
    """One row per station deployed in `campaign`, by field record or by images.

    A station appears if the field record dates BOTH ends of its deployment, or if it
    has images, or both. Both ends are required for a window: a half-open window would
    silently invent an end date, and an invented denominator is worse than none.
    """
    field = FieldRecord.load(root / FIELD_NOTES_FILENAME)
    media = _media_stations(campaign, root)

    windowed = {
        s for s in field.stations()
        if field.opening(s, campaign) is not None and field.closing(s, campaign) is not None
    }

    rows = []
    for station in sorted(media | windowed):
        opening = field.opening(station, campaign)
        closing = field.closing(station, campaign)
        has_media = station in media

        if station in windowed:
            # DATES, not datetimes. `FieldRecord` stamps a visit with no recorded time
            # at ASSUMED_VISIT_HOUR, so subtracting datetimes truncates whenever the two
            # ends disagree about the hour: CT01's install is timed 15:13 and its
            # retrieval is not, which reads as 168 days instead of 169. Camera-days are
            # a date-scale quantity and the assumed hour must not reach them.
            start, end = opening.visit_date.date(), closing.visit_date.date()
            days = (end - start).days
            note = "" if has_media else (
                "deployed per the field record, no images in the canonical table"
            )
        else:
            start = end = days = None
            note = "no field record dates both ends of this deployment"

        rows.append({
            "campaign": campaign,
            "station_id": station,
            "field_start": None if start is None else start.isoformat(),
            "field_end": None if end is None else end.isoformat(),
            "field_days": days,
            "has_media": has_media,
            "note": note,
        })

    return pd.DataFrame(rows, columns=list(COLUMNS))


def publish(*, root: Path = CAMPAIGNS_ROOT) -> list[Path]:
    """Write `<campaign>/deployments.csv` for every published campaign."""
    from camtrap.canonical_state import PUBLISHED_CAMPAIGNS

    written = []
    for campaign in PUBLISHED_CAMPAIGNS:
        frame = build(campaign, root=root)
        path = root / campaign / DEPLOYMENTS_FILENAME
        frame.to_csv(path, index=False, lineterminator="\n")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    from camtrap.canonical_state import PUBLISHED_CAMPAIGNS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--publish", action="store_true",
                    help="write the deployments.csv files (default: report only)")
    args = ap.parse_args(argv)

    for campaign in PUBLISHED_CAMPAIGNS:
        frame = build(campaign)
        dated = frame[frame["field_days"].notna()]
        print(f"{campaign}:")
        print(f"  deployments      : {len(frame)} "
              f"({int(frame['has_media'].sum())} with images)")
        print(f"  field-dated      : {len(dated)}")
        if len(dated):
            print(f"  camera-days      : {int(dated['field_days'].sum())} "
                  f"(median {int(dated['field_days'].median())} d per station)")
        for _, r in frame[frame["note"] != ""].iterrows():
            print(f"    {r['station_id']}: {r['note']}")

    if args.publish:
        for path in publish():
            print(f"wrote {path}")
    else:
        print("\n(nothing written; re-run with --publish)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
