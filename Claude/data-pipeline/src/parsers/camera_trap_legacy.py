"""Parse the legacy Timelapse2 CSV ('old animal data DB.csv')."""

import re
from pathlib import Path

import pandas as pd


def _parse_note0(note: str):
    """Parse Note0 field: 'burst_num:image_within_burst|total_images'. Returns burst_num str."""
    if not isinstance(note, str) or not note.strip():
        return "1"
    m = re.match(r"(\d+):", note.strip())
    return m.group(1) if m else "1"


def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s)).strip("_")


def parse(csv_path: Path):
    """
    Parse legacy Timelapse2 CSV.

    Returns:
        (deployments_df, media_df, obs_df) — DataFrames matching ct_* schema columns.
    """
    print(f"→ Parsing legacy camera trap CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)

    # Strip trailing comma that creates an empty last column
    df = df.loc[:, ~df.columns.str.fullmatch(r"\s*")]

    # Strip whitespace from DateTime (leading space in source)
    df["DateTime"] = df["DateTime"].str.strip()

    # Parse timestamp as America/Santiago → UTC
    df["timestamp"] = (
        pd.to_datetime(df["DateTime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        .dt.tz_localize("America/Santiago", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    # --- Build deployment fields from RelativePath ---
    # RelativePath example: 2022\Araucarias\CT10_06_12_22
    def _parse_relpath(rp):
        parts = str(rp).replace("\\", "/").split("/")
        area = parts[1] if len(parts) >= 2 else ""
        station_segment = parts[-1] if parts else ""
        station_id = station_segment.split("_")[0]  # e.g. CT10
        return area, station_segment, station_id

    parsed = df["RelativePath"].apply(_parse_relpath)
    df["_area"] = [p[0] for p in parsed]
    df["_station_segment"] = [p[1] for p in parsed]
    df["_station_id"] = [p[2] for p in parsed]

    # deploymentID = RelativePath slug (unique per camera deployment)
    df["deploymentID"] = df["RelativePath"].apply(
        lambda rp: "legacy_" + _slugify(rp)
    )

    # --- Build IDs ---
    df["_rp_slug"] = df["RelativePath"].apply(_slugify)
    df["mediaID"] = "legacy_" + df["_rp_slug"] + "_" + df["File"].apply(_slugify)
    df["observationID"] = "legacy_" + df["mediaID"] + "_obs"

    df["_burst_num"] = df["Note0"].apply(_parse_note0)
    df["eventID"] = df["deploymentID"] + "_burst" + df["_burst_num"]

    # --- observationType ---
    def _obs_type(row):
        animal = str(row.get("Animal", "")).strip().lower()
        person = str(row.get("Person", "")).strip().lower()
        if animal == "si":
            return "animal"
        if person == "si":
            return "human"
        return "blank"

    df["observationType"] = df.apply(_obs_type, axis=1)

    # --- DEPLOYMENTS ---
    dep_cols = ["deploymentID", "locationID", "locationName", "latitude", "longitude",
                "deploymentStart", "deploymentEnd", "cameraID", "cameraModel", "habitat", "source"]

    dep_agg = df.groupby("deploymentID").agg(
        locationName=("_area", "first"),
        deploymentStart=("timestamp", "min"),
        deploymentEnd=("timestamp", "max"),
        cameraID=("_station_id", "first"),
    ).reset_index()
    dep_agg["locationID"] = dep_agg["cameraID"]
    dep_agg["latitude"] = None
    dep_agg["longitude"] = None
    dep_agg["cameraModel"] = None
    dep_agg["habitat"] = None
    dep_agg["source"] = "legacy"
    deployments_df = dep_agg[dep_cols]

    # --- MEDIA ---
    media_df = df[["mediaID", "deploymentID", "timestamp", "File",
                   "RelativePath"]].copy()
    media_df = media_df.rename(columns={"File": "fileName", "RelativePath": "filePath"})
    media_df["fileMediatype"] = "image/jpeg"
    media_df["source"] = "legacy"
    media_df = media_df[["mediaID", "deploymentID", "timestamp", "fileName", "filePath",
                          "fileMediatype", "source"]]

    # --- OBSERVATIONS (only animal or human rows) ---
    obs_mask = df["observationType"].isin(["animal", "human"])
    obs_df = df.loc[obs_mask, ["observationID", "deploymentID", "mediaID", "eventID",
                               "timestamp", "Especie", "observationType"]].copy()
    obs_df = obs_df.rename(columns={"timestamp": "eventStart", "Especie": "scientificName"})
    obs_df["eventEnd"] = obs_df["eventStart"]
    obs_df["count"] = 1
    obs_df["classificationMethod"] = "human"
    obs_df["classificationProbability"] = None
    obs_df["source"] = "legacy"
    obs_df = obs_df[["observationID", "deploymentID", "mediaID", "eventID",
                     "eventStart", "eventEnd", "observationType", "scientificName",
                     "count", "classificationMethod", "classificationProbability", "source"]]

    print(f"  Deployments: {len(deployments_df)}, Media: {len(media_df)}, Observations: {len(obs_df)}")
    return deployments_df, media_df, obs_df
