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
    manifest, image data or reviewed CSV -- about 620 camera-days. Publishing them
    keeps that visible instead of letting them read as stations that never existed.

    RESOLVED 2026-08-25 (Felipe, from the NAS), and the answer inverted the
    assumption this module was built on. Four of the five WERE recording: their media
    is VIDEO, held for this campaign in a separate tree outside the pipeline. Only
    CT21 recorded nothing, and its own field note says why -- "SD vacia", the camera
    dead at retrieval. So `has_media=false` never meant "saw nothing", and a
    consumer that read it that way would have been wrong four times out of five.

    Hence `media_status`: the measurement and the REASON are now separate columns,
    because they answer different questions and only the reason decides a
    denominator. See MEDIA_ABSENCE_REASONS below.

THE DATES ARE THE FIELD RECORD'S, NOT THE MEDIA'S
    CT22 and CT25 were found failed at retrieval (humidity, dead screen), so each
    stopped sampling at some point before the recorded end date. Felipe's ruling,
    2026-08-25: the install and stop dates stand AS REGISTERED -- they were written
    down in the field and the video is not being re-read to second-guess them. The
    residual caveat is therefore stated, not silently corrected: for those two
    stations the true sampling period is at most `field_days` and its end is
    unknown. Anyone computing a rate from their video must read that as a ceiling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from camtrap.anchors import FIELD_NOTES_FILENAME, FieldRecord
from camtrap.observations import CAMPAIGNS_ROOT, CANONICAL_FILENAME

#: Written next to `observations.parquet`, one file per campaign.
DEPLOYMENTS_FILENAME = "deployments.csv"

#: Why a deployed station contributes no rows to the canonical table. One row per
#: station-campaign; see MEDIA_ABSENCE_REASONS for what each verdict licenses.
MEDIA_ABSENCE_FILENAME = "media_absence.csv"

COLUMNS = (
    "campaign",
    "station_id",
    "field_start",     # date the card went in, from the field record
    "field_end",       # date it came out
    "field_days",      # field_end - field_start, in days
    "has_media",       # does this station appear in the campaign's canonical table?
    "media_status",    # WHY not, when it does not -- see MEDIA_ABSENCE_REASONS
    "note",
)

# ── Why a deployment has no stills, and what each answer licenses ─────────────
#
# `has_media` is a MEASUREMENT: are there rows in the canonical table? The reason
# there are none is a separate fact, it is not measurable from the table, and
# conflating the two is what made this file assert something false. Until
# 2026-08-25 otono_2025's five station-campaigns without stills were published as
# "deployed per the field record, no images in the canonical table" -- true but
# useless, because it reads as "the camera saw nothing" when four of the five had
# been recording video the whole time, stored outside this pipeline.
#
# The distinction decides a DENOMINATOR, which is why it is worth a column:
#
#   in_canonical        stills are here. The only rows any rate computed from
#                       `observations.parquet` may divide by.
#   video_only_offline  the camera WAS sampling; its media is video held outside
#                       the pipeline. Real effort, and detections that exist and
#                       are unreadable here -- so these camera-days belong in an
#                       occupancy/presence denominator and MUST be excluded from
#                       any stills-based rate. Counting them there biases every
#                       rate downwards by a plausible-looking amount, which is
#                       DATA-HEALTH-MANUAL 6.3's "a plausible number from two
#                       mistakes".
#   card_failure        the camera recorded nothing at all. Contributes no effort
#                       to any question.
#   unexplained         no stills and nobody has said why. NOT a synonym for the
#                       others: an unexplained gap is a question that has not been
#                       asked, and it must not be silently absorbed into an
#                       effort figure. This is 4E.3's refusal-recording rule
#                       applied to media instead of anchors.
#   no_field_dates      the field record does not date both ends, so there is no
#                       window to speak of. Says nothing about media.
MEDIA_ABSENCE_REASONS = ("video_only_offline", "card_failure")

STATUS_IN_CANONICAL = "in_canonical"
STATUS_UNEXPLAINED = "unexplained"
STATUS_NO_FIELD_DATES = "no_field_dates"


def _media_stations(campaign: str, root: Path) -> set[str]:
    parquet = root / campaign / CANONICAL_FILENAME
    if not parquet.exists():
        return set()
    df = pd.read_parquet(parquet, columns=["station_canonical"])
    return set(df["station_canonical"].dropna().unique())


def _absence_reasons(campaign: str, root: Path) -> dict[str, str]:
    """station_id -> reason, for stations declared to have no stills in `campaign`.

    An unreadable or missing file yields no reasons rather than raising: every
    station then reports `unexplained`, which is the fail-visible direction. A
    reason this module does not know is also refused -- a typo in the CSV must not
    read as a licence to count the camera-days.
    """
    path = root / MEDIA_ABSENCE_FILENAME
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for _, r in df.iterrows():
        if r["campaign"].strip() != campaign:
            continue
        reason = r["reason"].strip()
        if reason not in MEDIA_ABSENCE_REASONS:
            raise ValueError(
                f"{path}: {r['station_id']} carries reason {reason!r}, which is not "
                f"one of {MEDIA_ABSENCE_REASONS}. Add it to MEDIA_ABSENCE_REASONS "
                f"with a note on what denominator it licenses, or fix the spelling."
            )
        out[r["station_id"].strip()] = reason
    return out


_NOTES = {
    "video_only_offline":
        "deployed and recording; media is video held outside this pipeline — real "
        "effort, but EXCLUDE from any stills-based rate denominator",
    "card_failure":
        "deployed but recorded nothing; contributes no effort to any question",
    STATUS_UNEXPLAINED:
        "deployed per the field record, no stills and no recorded reason — declare "
        "it in media_absence.csv before using this campaign's effort",
}


def _media_status(station: str, has_media: bool, absence: dict[str, str]) -> tuple[str, str]:
    """The station's media verdict and the sentence a reader needs, as one decision.

    Returned together because they cannot disagree if they are produced together;
    the previous free-text `note` drifted from the facts precisely because nothing
    tied the two.
    """
    if has_media:
        return STATUS_IN_CANONICAL, ""
    reason = absence.get(station, STATUS_UNEXPLAINED)
    return reason, _NOTES[reason]


def build(campaign: str, *, root: Path = CAMPAIGNS_ROOT) -> pd.DataFrame:
    """One row per station deployed in `campaign`, by field record or by images.

    A station appears if the field record dates BOTH ends of its deployment, or if it
    has images, or both. Both ends are required for a window: a half-open window would
    silently invent an end date, and an invented denominator is worse than none.
    """
    field = FieldRecord.load(root / FIELD_NOTES_FILENAME)
    media = _media_stations(campaign, root)
    absence = _absence_reasons(campaign, root)

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
            status, note = _media_status(station, has_media, absence)
        else:
            start = end = days = None
            status = STATUS_NO_FIELD_DATES
            note = "no field record dates both ends of this deployment"

        rows.append({
            "campaign": campaign,
            "station_id": station,
            "field_start": None if start is None else start.isoformat(),
            "field_end": None if end is None else end.isoformat(),
            "field_days": days,
            "has_media": has_media,
            "media_status": status,
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
        stills = dated[dated["media_status"] == STATUS_IN_CANONICAL]
        # Two denominators, both named. Printing one number called "camera-days"
        # beside a larger deployment count is how a reader silently picks the wrong
        # one -- and the wrong one here understates every rate.
        sampling = dated[dated["media_status"] != "card_failure"]
        print(f"{campaign}:")
        print(f"  deployments        : {len(frame)} "
              f"({int(frame['has_media'].sum())} with stills)")
        print(f"  field-dated        : {len(dated)}")
        if len(dated):
            print(f"  camera-days, stills: {int(stills['field_days'].sum())} "
                  f"over {len(stills)} station(s)  <- for rates from observations.parquet")
            print(f"  camera-days, all   : {int(sampling['field_days'].sum())} "
                  f"over {len(sampling)} station(s)  <- includes video-only deployments")
        for status, g in frame[frame["note"] != ""].groupby("media_status"):
            print(f"    {status}: {', '.join(g['station_id'])}")
            print(f"      {_NOTES.get(status, '')}")

    if args.publish:
        for path in publish():
            print(f"wrote {path}")
    else:
        print("\n(nothing written; re-run with --publish)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
