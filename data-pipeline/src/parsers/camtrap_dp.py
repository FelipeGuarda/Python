"""Parse Camtrap DP (Camera Trap Data Package — TDWG/GBIF standard) folder."""

from pathlib import Path

import pandas as pd


def parse(folder_path: Path):
    """
    Parse a Camtrap DP folder containing deployments.csv, media.csv, observations.csv.

    Returns:
        (deployments_df, media_df, obs_df) — DataFrames matching ct_* schema columns.
    """
    folder_path = Path(folder_path)
    print(f"→ Parsing Camtrap DP folder: {folder_path}")

    dep_path = folder_path / "deployments.csv"
    med_path = folder_path / "media.csv"
    obs_path = folder_path / "observations.csv"

    for p in (dep_path, med_path, obs_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing Camtrap DP file: {p}")

    # --- DEPLOYMENTS ---
    dep_raw = pd.read_csv(dep_path, dtype=str)
    dep_df = pd.DataFrame()
    dep_df["deploymentID"] = dep_raw["deploymentID"]
    dep_df["locationID"] = dep_raw.get("locationID")
    dep_df["locationName"] = dep_raw.get("locationName")
    dep_df["latitude"] = pd.to_numeric(dep_raw.get("latitude"), errors="coerce")
    dep_df["longitude"] = pd.to_numeric(dep_raw.get("longitude"), errors="coerce")
    dep_df["deploymentStart"] = pd.to_datetime(
        dep_raw.get("deploymentStart"), utc=True, errors="coerce"
    )
    dep_df["deploymentEnd"] = pd.to_datetime(
        dep_raw.get("deploymentEnd"), utc=True, errors="coerce"
    )
    dep_df["cameraID"] = dep_raw.get("cameraID")
    dep_df["cameraModel"] = dep_raw.get("cameraModel")
    dep_df["habitat"] = dep_raw.get("habitat")
    dep_df["source"] = "camtrap_dp"

    # --- MEDIA ---
    med_raw = pd.read_csv(med_path, dtype=str)
    med_df = pd.DataFrame()
    med_df["mediaID"] = med_raw["mediaID"]
    med_df["deploymentID"] = med_raw["deploymentID"]
    med_df["timestamp"] = pd.to_datetime(
        med_raw.get("timestamp"), utc=True, errors="coerce"
    )
    med_df["fileName"] = med_raw.get("fileName")
    med_df["filePath"] = med_raw.get("filePath")
    med_df["fileMediatype"] = med_raw.get("fileMediatype")
    med_df["source"] = "camtrap_dp"

    # --- OBSERVATIONS ---
    obs_raw = pd.read_csv(obs_path, dtype=str)
    obs_df = pd.DataFrame()
    obs_df["observationID"] = obs_raw["observationID"]
    obs_df["deploymentID"] = obs_raw["deploymentID"]
    obs_df["mediaID"] = obs_raw.get("mediaID")
    obs_df["eventID"] = obs_raw.get("eventID")
    obs_df["eventStart"] = pd.to_datetime(
        obs_raw.get("eventStart"), utc=True, errors="coerce"
    )
    obs_df["eventEnd"] = pd.to_datetime(
        obs_raw.get("eventEnd"), utc=True, errors="coerce"
    )
    obs_df["observationType"] = obs_raw.get("observationType")
    obs_df["scientificName"] = obs_raw.get("scientificName")
    obs_df["count"] = pd.to_numeric(obs_raw.get("count"), errors="coerce").astype("Int64")
    obs_df["classificationMethod"] = obs_raw.get("classificationMethod")
    obs_df["classificationProbability"] = pd.to_numeric(
        obs_raw.get("classificationProbability"), errors="coerce"
    )
    obs_df["source"] = "camtrap_dp"

    print(f"  Deployments: {len(dep_df)}, Media: {len(med_df)}, Observations: {len(obs_df)}")
    return dep_df, med_df, obs_df
