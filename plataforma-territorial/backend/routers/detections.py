"""Camera trap detection endpoints."""

import math
import os
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..db import get_connection

router = APIRouter(prefix="/api/detections", tags=["detections"])

_COMMON_NAMES: dict[str, str] = {
    "Puma concolor": "Puma",
    "Leopardus guigna": "Güiña",
    "Lycalopex culpaeus": "Zorro culpeo",
    "Sus scrofa": "Jabalí",
    "Lepus europaeus": "Liebre europea",
    "Canis lupus familiaris": "Perro doméstico",
    "Pteroptochos tectus": "Turca",
    "Equus caballus": "Caballo",
    "Turdus falcklandii": "Zorzal",
    "Caracara plancus": "Traro",
    "Milvago chimango": "Tiuque",
    "Dromiciops gliroides": "Monito del monte",
    "Cervus elaphus": "Ciervo rojo",
    "Felis catus": "Gato doméstico",
    "Pudu puda": "Pudú",
    "Sephanoides sephaniodes": "Picaflor",
    "Scelorchilus rubecula": "Huet-huet",
    "Conepatus chinga": "Chingue",
    "Strix rufipes": "Lechuza del sur",
    "Enicognathus ferrugineus": "Choroy",
    "Phrygilus gayi": "Cometocino",
    "Aphrastura spinicauda": "Rayadito",
    "Elaenia albiceps": "Fío-fío",
    "Abrothrix longipilis": "Ratón de pelo largo",
}

_TOTAL_TC_STATIONS = 26

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


@router.get("/diel-activity")
def diel_activity():
    """Detection counts by hour of day (0–23), all campaigns combined."""
    with get_connection() as con:
        rows = con.execute("""
            SELECT HOUR(eventStart) as hour, COUNT(*) as count
            FROM ct_observations
            WHERE observationType = 'animal' AND scientificName IS NOT NULL
            GROUP BY HOUR(eventStart)
            ORDER BY hour
        """).fetchall()
    hour_map = {r[0]: r[1] for r in rows}
    return [{"hour": h, "count": hour_map.get(h, 0)} for h in range(24)]


@router.get("/summary-stats")
def summary_stats():
    """Overall camera trap summary statistics."""
    with get_connection() as con:
        total = con.execute("""
            SELECT COUNT(*) FROM ct_observations
            WHERE observationType = 'animal' AND scientificName IS NOT NULL
        """).fetchone()[0]
        unique_species = con.execute("""
            SELECT COUNT(DISTINCT scientificName) FROM ct_observations
            WHERE observationType = 'animal' AND scientificName IS NOT NULL
        """).fetchone()[0]
        active_stations = con.execute("""
            SELECT COUNT(DISTINCT locationID) FROM ct_deployments
        """).fetchone()[0]
        campaigns = con.execute("""
            SELECT DISTINCT campaign FROM ct_deployments
            WHERE campaign IS NOT NULL ORDER BY campaign
        """).fetchall()
        days_sampled = con.execute("""
            SELECT COUNT(DISTINCT CAST(eventStart AS DATE))
            FROM ct_observations WHERE observationType = 'animal'
        """).fetchone()[0]
        date_range = con.execute("""
            SELECT CAST(MIN(eventStart) AS TEXT), CAST(MAX(eventStart) AS TEXT)
            FROM ct_observations WHERE observationType = 'animal'
        """).fetchone()
    return {
        "total_detections": total,
        "unique_species": unique_species,
        "active_stations": active_stations,
        "campaign_count": len(campaigns),
        "campaigns": [r[0] for r in campaigns],
        "days_sampled": days_sampled,
        "date_range_start": date_range[0],
        "date_range_end": date_range[1],
    }


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

        last_rows = con.execute("""
            WITH ranked AS (
                SELECT
                    d.locationID,
                    o.scientificName,
                    o.eventStart,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.locationID ORDER BY o.eventStart DESC
                    ) AS rn
                FROM ct_deployments d
                JOIN ct_observations o ON o.deploymentID = d.deploymentID
                WHERE o.observationType = 'animal' AND o.scientificName IS NOT NULL
            )
            SELECT locationID, scientificName, CAST(eventStart AS TEXT) AS last_time
            FROM ranked WHERE rn = 1
        """).fetchall()

    last_by_tc: dict[int, dict] = {}
    for loc_id, species, t in last_rows:
        try:
            last_by_tc[int(loc_id)] = {"last_species": species, "last_time": t}
        except (TypeError, ValueError):
            pass

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
                        f"CT{tc:02d}")

        last = last_by_tc.get(tc, {})
        result.append({
            "canonical_name": canonical,
            "tc_number": tc,
            "latitude": lat,
            "longitude": lon,
            "total_observations": sum(slot["species"].values()),
            "species": [{"name": k, "count": v} for k, v in species_list[:8]],
            "images": images[:3],
            "has_data": bool(location_names),
            "last_species": last.get("last_species"),
            "last_time": last.get("last_time"),
        })

    result.sort(key=lambda x: x["tc_number"])
    return result


@router.get("/species-list")
def species_list_with_occupancy():
    """All detected species with total detections and naive occupancy across TC stations."""
    with get_connection() as con:
        rows = con.execute("""
            SELECT
                o.scientificName,
                COUNT(*) AS total_detections,
                COUNT(DISTINCT d.locationID) AS n_stations
            FROM ct_observations o
            JOIN ct_deployments d ON o.deploymentID = d.deploymentID
            WHERE o.observationType = 'animal' AND o.scientificName IS NOT NULL
            GROUP BY o.scientificName
            ORDER BY total_detections DESC
        """).fetchall()
    return [
        {
            "scientific_name": r[0],
            "common_name": _COMMON_NAMES.get(r[0], r[0]),
            "total_detections": r[1],
            "n_stations": r[2],
            "occupancy_pct": round(r[2] / _TOTAL_TC_STATIONS * 100, 1),
        }
        for r in rows
    ]


@router.get("/overlap")
def species_overlap(sp1: str = Query(...), sp2: str = Query(...)):
    """
    Pairwise comparison for two species: 24h activity distributions, per-station
    detection counts, and a normalized overlap coefficient (0–1, Dhat1-style).
    """
    with get_connection() as con:
        hourly = con.execute("""
            SELECT o.scientificName, HOUR(o.eventStart) AS hour, COUNT(*) AS count
            FROM ct_observations o
            JOIN ct_deployments d ON o.deploymentID = d.deploymentID
            WHERE o.observationType = 'animal' AND o.scientificName IN (?, ?)
            GROUP BY o.scientificName, HOUR(o.eventStart)
        """, [sp1, sp2]).fetchall()

        station_counts = con.execute("""
            SELECT d.locationID, o.scientificName, COUNT(*) AS count
            FROM ct_observations o
            JOIN ct_deployments d ON o.deploymentID = d.deploymentID
            WHERE o.observationType = 'animal' AND o.scientificName IN (?, ?)
            GROUP BY d.locationID, o.scientificName
        """, [sp1, sp2]).fetchall()

        occupancy = con.execute("""
            SELECT o.scientificName, COUNT(DISTINCT d.locationID) AS n_stations
            FROM ct_observations o
            JOIN ct_deployments d ON o.deploymentID = d.deploymentID
            WHERE o.observationType = 'animal' AND o.scientificName IN (?, ?)
            GROUP BY o.scientificName
        """, [sp1, sp2]).fetchall()

    # Hourly distributions (fill 0s for missing hours)
    hourly_by_sp: dict[str, list[int]] = {sp1: [0] * 24, sp2: [0] * 24}
    for sp, hour, count in hourly:
        if sp in hourly_by_sp and hour is not None:
            hourly_by_sp[sp][hour] = count

    # Per-station counts keyed by TC number
    station_by_sp: dict[str, dict[int, int]] = {sp1: {}, sp2: {}}
    for loc_id, sp, count in station_counts:
        try:
            tc = int(loc_id)
            if sp in station_by_sp:
                station_by_sp[sp][tc] = count
        except (TypeError, ValueError):
            pass

    occ_by_sp = {sp: n for sp, n in occupancy}

    # Dhat1-style overlap coefficient: sum of min of normalized hourly proportions
    v1 = hourly_by_sp[sp1]
    v2 = hourly_by_sp[sp2]
    total1 = max(sum(v1), 1)
    total2 = max(sum(v2), 1)
    p1 = [v / total1 for v in v1]
    p2 = [v / total2 for v in v2]
    overlap_coeff = round(sum(min(p1[i], p2[i]) for i in range(24)), 3)

    def station_markers(sp: str) -> list[dict]:
        return [
            {"tc": tc, "lat": lat, "lon": lon, "count": station_by_sp[sp].get(tc, 0)}
            for tc, (lat, lon) in _TC_COORDS.items()
        ]

    return {
        "chart": [{"hour": h, "sp1": v1[h], "sp2": v2[h]} for h in range(24)],
        "stations_sp1": station_markers(sp1),
        "stations_sp2": station_markers(sp2),
        "occupancy_sp1": occ_by_sp.get(sp1, 0),
        "occupancy_sp2": occ_by_sp.get(sp2, 0),
        "total_sp1": sum(v1),
        "total_sp2": sum(v2),
        "overlap_coeff": overlap_coeff,
        "sp1_name": _COMMON_NAMES.get(sp1, sp1),
        "sp2_name": _COMMON_NAMES.get(sp2, sp2),
    }
