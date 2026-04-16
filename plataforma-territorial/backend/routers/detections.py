"""Camera trap detection endpoints."""

import os
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..db import get_connection

router = APIRouter(prefix="/api/detections", tags=["detections"])

# Station image exports live in the camera-traps repo (gitignored, large files).
# Default: sibling repo layout on Linux — /home/fguarda/Dev/Python/camera-traps/exports
# Override with CT_EXPORTS_DIR env var if the path differs.
_DEFAULT_EXPORTS = Path(__file__).resolve().parents[3] / "camera-traps" / "exports"
_CT_EXPORTS_DIR = Path(os.getenv("CT_EXPORTS_DIR", str(_DEFAULT_EXPORTS)))

# Canonical TC camera locations (1..26). Canonical copy lives in
# data-pipeline/src/parsers/timelapse_reviewed.py (_TC_COORDS); keep in sync.
# Used so the Observatorio map shows all 26 deployed cameras even if the
# reviewed CSV for a given campaign has no rows for that TC (Timelapse only
# exports rows with animal images, so zero-animal stations are absent).
_TC_COORDS: dict[int, tuple[float, float]] = {
     1: (-39.45183, -71.72707),  2: (-39.45163, -71.73252),
     3: (-39.43796, -71.74220),  4: (-39.43774, -71.74829),
     5: (-39.45579, -71.73253),  6: (-39.42418, -71.75465),
     7: (-39.44718, -71.73228),  8: (-39.44286, -71.74485),
     9: (-39.42885, -71.75478), 10: (-39.44675, -71.74390),
    11: (-39.45130, -71.74873), 12: (-39.45905, -71.75111),
    13: (-39.45556, -71.75083), 14: (-39.44709, -71.73836),
    15: (-39.45142, -71.74398), 16: (-39.45524, -71.74502),
    17: (-39.45180, -71.73987), 18: (-39.43320, -71.74338),
    19: (-39.43286, -71.75497), 20: (-39.43752, -71.76076),
    21: (-39.44299, -71.73801), 22: (-39.43296, -71.74903),
    23: (-39.43704, -71.73900), 24: (-39.43359, -71.73804),
    25: (-39.42929, -71.74325), 26: (-39.42908, -71.74894),
}


@router.get("/recent")
def recent_detections(limit: int = Query(default=20, le=100)):
    """Most recent animal detections."""
    with get_connection() as con:
        rows = con.execute(f"""
            SELECT
                o.scientificName as species,
                o.count,
                CAST(o.eventStart AS TEXT) as timestamp,
                d.locationName as station,
                d.latitude, d.longitude,
                o.classificationMethod,
                o.classificationProbability
            FROM ct_observations o
            JOIN ct_deployments d ON o.deploymentID = d.deploymentID
            WHERE o.observationType = 'animal'
              AND o.scientificName IS NOT NULL
            ORDER BY o.eventStart DESC
            LIMIT {limit}
        """).fetchall()
    cols = [
        "species", "count", "timestamp", "station",
        "latitude", "longitude",
        "classification_method", "classification_probability",
    ]
    return [dict(zip(cols, r)) for r in rows]


@router.get("/species-summary")
def species_summary():
    """Detection counts by species."""
    with get_connection() as con:
        rows = con.execute("""
            SELECT
                scientificName as species,
                COUNT(*) as total_detections,
                SUM(count) as total_individuals,
                CAST(MAX(eventStart) AS TEXT) as last_seen
            FROM ct_observations
            WHERE observationType = 'animal'
              AND scientificName IS NOT NULL
            GROUP BY scientificName
            ORDER BY total_detections DESC
        """).fetchall()
    cols = ["species", "total_detections", "total_individuals", "last_seen"]
    return [dict(zip(cols, r)) for r in rows]


@router.get("/stations")
def camera_stations():
    """Camera trap deployment locations."""
    with get_connection() as con:
        rows = con.execute("""
            SELECT
                deploymentID, locationName, locationID,
                latitude, longitude,
                cameraID, cameraModel, habitat,
                CAST(deploymentStart AS TEXT) as start,
                CAST(deploymentEnd AS TEXT) as end,
                (SELECT COUNT(*) FROM ct_observations o
                 WHERE o.deploymentID = d.deploymentID
                   AND o.observationType = 'animal') as animal_count
            FROM ct_deployments d
            ORDER BY locationName
        """).fetchall()
    cols = [
        "deployment_id", "location_name", "location_id",
        "latitude", "longitude",
        "camera_id", "camera_model", "habitat",
        "start", "end", "animal_count",
    ]
    return [dict(zip(cols, r)) for r in rows]


@router.get("/station-images/{station_id}")
def station_images(station_id: str):
    """
    Return image filenames exported for a given station across all campaigns.

    Images live in: <CT_EXPORTS_DIR>/<campaign>/stations/<station_id>/
    They are served as static files mounted at /ct-images/ in main.py.

    Response: { "station": str, "images": [{ "campaign": str, "url": str }] }
    """
    if not _CT_EXPORTS_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Camera trap exports directory not found: {_CT_EXPORTS_DIR}",
        )

    results = []
    for campaign_dir in sorted(_CT_EXPORTS_DIR.iterdir()):
        if not campaign_dir.is_dir():
            continue
        station_dir = campaign_dir / "stations" / station_id
        if not station_dir.is_dir():
            continue
        for img in sorted(station_dir.glob("*.jpg")):
            # URL path mirrors the static mount: /ct-images/<campaign>/stations/<station>/<file>
            url = f"/ct-images/{campaign_dir.name}/stations/{station_id}/{img.name}"
            results.append({"campaign": campaign_dir.name, "url": url})

    return {"station": station_id, "images": results}


@router.get("/station-summary")
def station_summary():
    """
    All 26 TC camera locations with species breakdown and image URLs.

    TC_COORDS is the ground truth for where cameras exist — the reviewed CSV
    only contains rows for images classified as animals, so a station with no
    animal detections in a campaign is absent from ct_deployments for that
    campaign. This endpoint joins DB data onto the full TC list so the map
    always shows all 26 markers.
    """
    with get_connection() as con:
        rows = con.execute("""
            SELECT
                d.locationID,
                d.locationName,
                o.scientificName,
                COUNT(*) FILTER (WHERE o.observationType = 'animal'
                                   AND o.scientificName IS NOT NULL) AS obs_count
            FROM ct_deployments d
            LEFT JOIN ct_observations o ON o.deploymentID = d.deploymentID
            WHERE d.campaign IS NOT NULL
            GROUP BY d.locationID, d.locationName, o.scientificName
        """).fetchall()

    # Bucket DB rows by TC number. locationID is the TC number as text.
    by_tc: dict[int, dict] = {}
    for loc_id, loc_name, species, count in rows:
        try:
            tc = int(loc_id)
        except (TypeError, ValueError):
            continue
        slot = by_tc.setdefault(tc, {"location_names": [], "species": defaultdict(int)})
        if loc_name and loc_name not in slot["location_names"]:
            slot["location_names"].append(loc_name)
        if species and count:
            slot["species"][species] += count

    result = []
    for tc, (lat, lon) in sorted(_TC_COORDS.items()):
        slot = by_tc.get(tc, {"location_names": [], "species": defaultdict(int)})
        location_names = slot["location_names"]

        images = []
        for loc_name in location_names:
            export_id = loc_name.replace(".", "_")
            if _CT_EXPORTS_DIR.exists():
                for campaign_dir in sorted(_CT_EXPORTS_DIR.iterdir()):
                    if not campaign_dir.is_dir():
                        continue
                    station_dir = campaign_dir / "stations" / export_id
                    if not station_dir.is_dir():
                        continue
                    for img in sorted(station_dir.glob("*.jpg"))[:2]:
                        images.append({
                            "campaign": campaign_dir.name,
                            "url": f"/ct-images/{campaign_dir.name}/stations/{export_id}/{img.name}",
                        })
                    if len(images) >= 3:
                        break
            if len(images) >= 3:
                break

        species_list = sorted(slot["species"].items(), key=lambda x: -x[1])
        canonical = next((n for n in location_names if n.startswith("CT")),
                        location_names[0] if location_names else f"TC{tc:02d}")

        result.append({
            "canonical_name": canonical,
            "tc_number": tc,
            "latitude": lat,
            "longitude": lon,
            "total_observations": sum(slot["species"].values()),
            "species": [{"name": k, "count": v} for k, v in species_list[:8]],
            "images": images[:3],
            "has_data": bool(location_names),
        })

    result.sort(key=lambda x: x["tc_number"])
    return result
