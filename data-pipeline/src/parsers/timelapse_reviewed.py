"""
Parse new_labeled_data_reviewed.csv — the flat Timelapse2 export after CLIP
classification and human review — into (deployments_df, media_df, obs_df).

Station naming conventions handled:
  CT01, CT02 … CT26   — Otoño 2025 style  (number = TC camera number)
  TC10_M3.2           — Primavera-verano style (digits after TC = camera number)

Returns DataFrames matching the ct_* DuckDB schema, with two extra columns on
obs_df that are added dynamically by ensure_columns on first ingest:
  observationComments  — Spanish common name written by the review UI
  reviewOutcome        — 'confirmed' | 'reclassified_p2' | …
And one extra column on dep_df:
  campaign             — human-readable campaign name (e.g. "Otoño 2025")
"""

import csv
import re
from pathlib import Path

import pandas as pd

from ..species import spanish_to_latin as _load_spanish_to_latin
from ..stations import tc_coords as _load_tc_coords

# TC camera number (1..N) → (lat, lon). Canonical source: plataforma-territorial/data/stations.yaml.
_TC_COORDS: dict[int, tuple[float, float]] = _load_tc_coords()

# Spanish common name (or alias) → scientific name. Canonical source: data-pipeline/species.yaml.
# Used when reviewer typed a Spanish name in observationComments via
# "Otro (especificar)" but scientificName was never filled.
SPANISH_TO_LATIN: dict[str, str] = _load_spanish_to_latin()


def _station_to_tc_number(station: str) -> int | None:
    """
    Extract TC camera number from station name for coordinate lookup.

    CT01   → 1    (strip CT, parse int)
    TC10_M3.2 → 10  (digits immediately after TC)
    """
    m = re.match(r"CT0*(\d+)$", station, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.match(r"TC0*(\d+)(?:_|$)", station, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s)).strip("_").lower()


# observationComments values (lowercased) that override the automated
# observationType='animal' — the reviewer explicitly said the image is not
# an animal. Demoted to 'blank' so counts reflect the human review.
NON_ANIMAL_COMMENTS: set[str] = {
    "no es un animal",
    "error de imagen",
    "no aparece imagen",
    "no aparece imagewn",  # typo in reviewer data
}


def _parse_timestamp(dt_str: str) -> pd.Timestamp:
    """Parse Timelapse2 DateTime string to UTC. Returns NaT on failure."""
    try:
        ts = pd.Timestamp(dt_str.strip())
        # tz_localize on a scalar: ambiguous must be bool, not "infer"
        # Try non-fold (DST) first; fall back to fold if the time is ambiguous.
        try:
            return ts.tz_localize(
                "America/Santiago", ambiguous=False, nonexistent="shift_forward"
            ).tz_convert("UTC")
        except Exception:
            return ts.tz_localize(
                "America/Santiago", ambiguous=True, nonexistent="shift_forward"
            ).tz_convert("UTC")
    except Exception:
        return pd.NaT


def parse(csv_path: Path, campaign_name: str):
    """
    Parse new_labeled_data_reviewed.csv.

    Args:
        csv_path:      Path to the reviewed CSV file.
        campaign_name: Human-readable name, e.g. "Otoño 2025".

    Returns:
        (deployments_df, media_df, obs_df)
    """
    campaign_slug = _slugify(campaign_name)
    csv_path = Path(csv_path)
    print(f"Parsing timelapse reviewed CSV: {csv_path}")

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"  {len(rows)} rows from {csv_path.name}")

    # ── Build per-station timestamp bounds ────────────────────────────────────
    station_times: dict[str, dict[str, pd.Timestamp]] = {}
    for row in rows:
        station = row["RelativePath"].strip()
        ts = _parse_timestamp(row.get("DateTime", ""))
        if pd.isna(ts):
            continue
        if station not in station_times:
            station_times[station] = {"min": ts, "max": ts}
        else:
            if ts < station_times[station]["min"]:
                station_times[station]["min"] = ts
            if ts > station_times[station]["max"]:
                station_times[station]["max"] = ts

    # ── DEPLOYMENTS ───────────────────────────────────────────────────────────
    dep_records = []
    for station, times in sorted(station_times.items()):
        tc_num = _station_to_tc_number(station)
        coords = _TC_COORDS.get(tc_num) if tc_num is not None else None
        if coords is None:
            print(f"  WARNING: no coordinates for {station!r} (TC={tc_num})")
        lat = coords[0] if coords else None
        lon = coords[1] if coords else None
        dep_records.append({
            "deploymentID":    f"{campaign_slug}_{station}",
            "locationID":      str(tc_num) if tc_num is not None else station,
            "locationName":    station,
            "latitude":        lat,
            "longitude":       lon,
            "deploymentStart": times["min"],
            "deploymentEnd":   times["max"],
            "cameraID":        None,
            "cameraModel":     None,
            "habitat":         None,
            "campaign":        campaign_name,
            "source":          "timelapse_reviewed",
        })

    dep_df = pd.DataFrame(dep_records)

    # ── MEDIA & OBSERVATIONS ──────────────────────────────────────────────────
    med_records = []
    obs_records = []

    for row in rows:
        station      = row["RelativePath"].strip()
        deployment_id = f"{campaign_slug}_{station}"
        ts            = _parse_timestamp(row.get("DateTime", ""))
        media_id      = row.get("mediaID", "").strip()

        if media_id:
            med_records.append({
                "mediaID":       media_id,
                "deploymentID":  deployment_id,
                "timestamp":     ts,
                "fileName":      row.get("File", "").strip() or None,
                "filePath":      row.get("filePath", "").strip() or None,
                "fileMediatype": row.get("fileMediatype", "").strip() or None,
                "source":        "timelapse_reviewed",
            })

        obs_id = row.get("observationID", "").strip()
        if obs_id:
            prob_str = row.get("classificationProbability", "").strip()
            prob = float(prob_str) if prob_str else None

            count_str = row.get("count", "").strip()
            count = int(count_str) if count_str else 1

            obs_type = row.get("observationType", "").strip() or None
            sci_name = row.get("scientificName", "").strip() or None
            comment  = row.get("observationComments", "").strip() or None
            comment_lc = comment.lower() if comment else ""

            if obs_type == "animal" and comment_lc in NON_ANIMAL_COMMENTS:
                obs_type = "blank"
            elif obs_type == "animal" and not sci_name and comment_lc:
                sci_name = SPANISH_TO_LATIN.get(comment_lc) or sci_name

            obs_records.append({
                "observationID":             obs_id,
                "deploymentID":              deployment_id,
                "mediaID":                   media_id or None,
                "eventID":                   row.get("eventID", "").strip() or None,
                "eventStart":                ts,
                "eventEnd":                  pd.NaT,
                "observationType":           obs_type,
                "scientificName":            sci_name,
                "count":                     count,
                "classificationMethod":      row.get("classificationMethod", "").strip() or None,
                "classificationProbability": prob,
                "observationComments":       comment,
                "reviewOutcome":             row.get("reviewOutcome", "").strip() or None,
                "source":                    "timelapse_reviewed",
            })

    med_df = pd.DataFrame(med_records)
    obs_df = pd.DataFrame(obs_records)

    print(f"  {len(dep_df)} deployments, {len(med_df)} media, {len(obs_df)} observations")
    return dep_df, med_df, obs_df
