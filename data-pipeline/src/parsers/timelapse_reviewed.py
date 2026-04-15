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

# ── Station coordinates ───────────────────────────────────────────────────────
# TC camera number (1–26) → (latitude, longitude)
# Source: GIS/CT ID and coordinates.xlsx, columns TC / S / W
# S and W stored as positive decimal degrees; negated here (southern/western hemisphere).
_TC_COORDS: dict[int, tuple[float, float]] = {
     1: (-39.45183, -71.72707),
     2: (-39.45163, -71.73252),
     3: (-39.43796, -71.74220),
     4: (-39.43774, -71.74829),
     5: (-39.45579, -71.73253),
     6: (-39.42418, -71.75465),
     7: (-39.44718, -71.73228),
     8: (-39.44286, -71.74485),
     9: (-39.42885, -71.75478),
    10: (-39.44675, -71.74390),
    11: (-39.45130, -71.74873),
    12: (-39.45905, -71.75111),
    13: (-39.45556, -71.75083),
    14: (-39.44709, -71.73836),
    15: (-39.45142, -71.74398),
    16: (-39.45524, -71.74502),
    17: (-39.45180, -71.73987),
    18: (-39.43320, -71.74338),
    19: (-39.43286, -71.75497),
    20: (-39.43752, -71.76076),
    21: (-39.44299, -71.73801),
    22: (-39.43296, -71.74903),
    23: (-39.43704, -71.73900),
    24: (-39.43359, -71.73804),
    25: (-39.42929, -71.74325),
    26: (-39.25447, -71.44562),
}


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

            obs_records.append({
                "observationID":             obs_id,
                "deploymentID":              deployment_id,
                "mediaID":                   media_id or None,
                "eventID":                   row.get("eventID", "").strip() or None,
                "eventStart":                ts,
                "eventEnd":                  pd.NaT,
                "observationType":           row.get("observationType", "").strip() or None,
                "scientificName":            row.get("scientificName", "").strip() or None,
                "count":                     count,
                "classificationMethod":      row.get("classificationMethod", "").strip() or None,
                "classificationProbability": prob,
                "observationComments":       row.get("observationComments", "").strip() or None,
                "reviewOutcome":             row.get("reviewOutcome", "").strip() or None,
                "source":                    "timelapse_reviewed",
            })

    med_df = pd.DataFrame(med_records)
    obs_df = pd.DataFrame(obs_records)

    print(f"  {len(dep_df)} deployments, {len(med_df)} media, {len(obs_df)} observations")
    return dep_df, med_df, obs_df
