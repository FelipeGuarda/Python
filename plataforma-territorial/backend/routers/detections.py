"""Camera trap detection endpoints."""

from fastapi import APIRouter, Query

from ..db import get_connection

router = APIRouter(prefix="/api/detections", tags=["detections"])


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
