"""The canonical camera-trap table, projected into the warehouse's relational triple.

WHAT THIS OWNS. One decision: how one-row-per-still becomes deployment / media /
observation rows, and how each row's identity is derived. Nothing else.

WHAT IT MUST NEVER OWN, and this is the whole reason the previous implementation was
deleted rather than repaired. `camera-traps/camtrap/observations.py` has ALREADY decided:

    station -> camera number        review comment -> observationType
    Spanish common name -> Latin    which rows are effort-valid
    clock repair and its provenance

`timelapse_reviewed.py` re-derived all five and disagreed with the canonical table on
515 live rows -- it knew four comment strings, only ever demoted to `blank`, and had no
rule producing `human`, `vehicle` or `unknown`. Ingesting it would have rebuilt the
815-row defect V2-REVIEW 1.3 closed. Every field below is COPIED from the parquet or
derived from the station registry. If you find yourself writing a lookup table here,
you are writing the bug again.

IDENTITY IS DERIVED FROM THE IMAGE, NEVER INHERITED FROM TIMELAPSE. Timelapse mints
`mediaID`/`observationID` GUIDs per PROJECT, not per image: joining primavera's legacy
project against its current one gives 2,387 shared filenames and 0 shared mediaIDs. A
rebuilt project therefore forks the table silently, orphaning every row keyed on the old
GUID. Keys here come from (campaign, station, fileName).

A CORRECTION TO V2-REVIEW 2.8, MEASURED 2026-08-24. It specifies `DEDUP_KEY =
[camera_num, file_name, datetime]` as the natural key. `datetime` is NULL in 4,013 of
35,807 rows -- the clock-failure stills, which must stay in the table because presence
needs a station and not a clock -- so DEDUP_KEY cannot be a primary key. Measured across
all three campaigns, `(campaign, camera_num, file_name)` is unique for all 35,807 rows
and never null. That is the key used here.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from src.stations import tc_coords

#: Names the producer in every row, so a table can always say where it came from.
SOURCE = "canonical_parquet"

#: Where deploymentStart/End came from. The observed window is the min/max media
#: timestamp; it is NOT the field window, which needs a visit record. When the visit
#: form's loader lands (V2-REVIEW 1.14), field-recorded windows arrive as
#: 'field_record' and consumers can tell the two apart instead of guessing.
WINDOW_OBSERVED = "observed_media"

_MEDIATYPE = "image/jpeg"


def _row_id(kind: str, campaign: str, station: str, file_name: str) -> str:
    """A stable, content-free identifier for one still.

    Derived, not random: re-running the rebuild must produce the SAME id for the same
    photograph, or `INSERT OR REPLACE` duplicates instead of replacing. `kind`
    separates the media and observation namespaces so the two ids never collide.
    """
    digest = hashlib.sha1(
        f"{kind}|{campaign}|{station}|{file_name}".encode("utf-8")
    ).hexdigest()
    return f"{kind}_{digest[:24]}"


def deployment_id(campaign: str, station: str) -> str:
    """'otono_2025', 'CT01' -> 'otono_2025_CT01'.

    ASCII by construction. The retired implementation produced `oto_o_2025_CT07`,
    having mangled the ñ, which is the kind of identity that silently fails to join.
    """
    return f"{campaign}_{station}"


def _blank_to_none(series: pd.Series) -> pd.Series:
    """Empty string means NOT RECORDED in the canonical table; SQL spells that NULL."""
    out = series.astype("object").where(series.notna(), None)
    return out.map(lambda v: None if v is None or str(v).strip() == "" else str(v))


def to_tables(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Canonical frames, keyed by campaign, -> {'ct_deployments', 'ct_media',
    'ct_observations'}.

    Pure: reads the frames and the station registry, touches no database. That is what
    lets the reconciliation in `ingest` compare row counts against the parquet without
    the projection having already written anything.
    """
    if not frames:
        raise ValueError("no campaigns supplied; refusing to build empty ct_* tables")

    df = pd.concat(frames.values(), ignore_index=True)
    df = df.assign(
        _deployment=[
            deployment_id(c, s) for c, s in zip(df["campaign"], df["station_canonical"])
        ],
        _media=[
            _row_id("med", c, s, f)
            for c, s, f in zip(df["campaign"], df["station_canonical"], df["file_name"])
        ],
        _observation=[
            _row_id("obs", c, s, f)
            for c, s, f in zip(df["campaign"], df["station_canonical"], df["file_name"])
        ],
    )

    return {
        "ct_deployments": _deployments(df),
        "ct_media": _media(df),
        "ct_observations": _observations(df),
    }


def _deployments(df: pd.DataFrame) -> pd.DataFrame:
    coords = tc_coords()

    grouped = df.groupby(["_deployment", "campaign", "station_canonical", "camera_num"],
                         dropna=False, observed=True)
    # min/max skip NaT, so a station whose clock failed entirely yields NaT for both
    # rather than dropping out of the table. It was still deployed.
    window = grouped["datetime"].agg(["min", "max"]).reset_index()

    cam = window["camera_num"].astype("Int64")
    return pd.DataFrame({
        "deploymentID": window["_deployment"],
        # Text, but must parse as int(): the platform buckets stations by
        # int(locationID) and logs-and-skips anything else.
        "locationID": cam.map(lambda n: None if pd.isna(n) else str(int(n))),
        "locationName": window["station_canonical"],
        "campaign": window["campaign"],
        "latitude": cam.map(lambda n: coords.get(int(n), (None, None))[0]
                            if not pd.isna(n) else None),
        "longitude": cam.map(lambda n: coords.get(int(n), (None, None))[1]
                             if not pd.isna(n) else None),
        "deploymentStart": window["min"],
        "deploymentEnd": window["max"],
        "deploymentWindowSource": WINDOW_OBSERVED,
        # Not recorded anywhere yet. estaciones.csv reserves camera_unit_id,
        # height_m, bearing_deg and detection_distance_m; all are blank for all 27
        # stations because the field record never captured them. They are the punch
        # list for the next salida, not data to impute here.
        "cameraID": None,
        "cameraModel": None,
        "habitat": None,
        "source": SOURCE,
    })


def _media(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "mediaID": df["_media"],
        "deploymentID": df["_deployment"],
        "timestamp": df["datetime"],
        "fileName": df["file_name"],
        "filePath": df["rel_path"],
        "fileMediatype": _MEDIATYPE,
        "source": SOURCE,
    })


def _observations(df: pd.DataFrame) -> pd.DataFrame:
    # A reviewer ruled on every row that is not sweep-only; those carry the classifier's
    # verdict. This is a RESTATEMENT of review_resolution, not a second opinion about it.
    method = df["review_resolution"].map(
        lambda r: "machine" if (pd.isna(r) or r == "sweep_only") else "human"
    )
    species = _blank_to_none(df["species_latin"])
    return pd.DataFrame({
        "observationID": df["_observation"],
        "deploymentID": df["_deployment"],
        "mediaID": df["_media"],
        # Empty in all three campaigns (measured 2026-08-20). Left NULL rather than
        # invented; independent events are computed by the consumers that need them,
        # each of which states its own gap rule.
        "eventID": None,
        "eventStart": df["datetime"],
        "eventEnd": df["datetime"],
        "observationType": df["observation_type"],
        "scientificName": species.where(df["observation_type"] == "animal", None),
        "count": pd.Series([None] * len(df), dtype="Int64"),
        "classificationMethod": method,
        "classificationProbability": df["classification_probability"].astype("Float64"),
        "observationComments": _blank_to_none(df["observation_comments"]),
        "reviewOutcome": _blank_to_none(df["review_outcome"]),
        "reviewResolution": _blank_to_none(df["review_resolution"]),
        "source": SOURCE,
    })
