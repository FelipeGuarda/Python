"""Frontend configuration endpoints — geography, station counts."""

from fastapi import APIRouter

from ..stations import load_stations, reserve, tc_coords, weather_station

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/geography")
def geography():
    """
    Geography metadata for frontend map initialization.

    Returns reserve center/zoom for the default map view, the weather
    station location and metadata, and the camera-trap station count.
    The frontend uses this in place of hardcoded coordinates and the
    `26 estaciones` literal.
    """
    data = load_stations()
    ws = weather_station()
    return {
        "reserve": reserve(),
        "weather_station": {
            "id": ws["id"],
            "name": ws["name"],
            "lat": ws["lat"],
            "lon": ws["lon"],
            "model": ws.get("model"),
        },
        "camera_trap_count": len(data["camera_traps"]),
    }
