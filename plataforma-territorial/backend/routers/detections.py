"""Camera trap detection endpoints."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..db import get_connection

router = APIRouter(prefix="/api/detections", tags=["detections"])

# Station image exports live in the camera-traps repo (gitignored, large files).
# Default: sibling repo layout on Linux — /home/fguarda/Dev/Python/camera-traps/exports
# Override with CT_EXPORTS_DIR env var if the path differs.
_DEFAULT_EXPORTS = Path(__file__).resolve().parents[3] / "camera-traps" / "exports"
_CT_EXPORTS_DIR = Path(os.getenv("CT_EXPORTS_DIR", str(_DEFAULT_EXPORTS)))


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
