"""
Station registry loader for the plataforma-territorial backend.

Two sources, one dict:

- `reserve` and `weather` come from this project's own `data/stations.yaml`. They are
  not camera-trap stations and no other project owns them.
- `camera_traps` come from the producer's registry, `camera-traps/data/campaigns/
  estaciones.csv`, read directly. That CSV owns station identity and coordinates; this
  module reads it and derives nothing the producer already decided. Until 2026-09-03 the
  producer spliced a generated `camera_traps:` section into our YAML, which meant it knew
  our file's layout and path -- the direction of knowledge the data-health manual forbids.
  Now the knowledge runs the right way: we know where the registry is, it does not know we
  exist.

Stations are spelled `CT01`..`CT27` here as everywhere else. They were `TC-01`
until 2026-08-24, which gave the project two names for one thing.

Override the YAML with FMA_STATIONS_YAML and the registry with CT_STATION_REGISTRY when
running outside the repo layout.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

import yaml

from .paths import ct_station_registry

_ENV_VAR = "FMA_STATIONS_YAML"

# backend/stations.py → parents[1] = plataforma-territorial/
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "stations.yaml"


def stations_yaml_path() -> Path:
    override = os.getenv(_ENV_VAR)
    return Path(override) if override else _DEFAULT_PATH


def _camera_traps_from_registry(path: Path) -> list[dict]:
    """One entry per registry row, in canonical order, with the numeric `tc` the map needs.

    An empty coordinate cell is NOT RECORDED in the registry; a station without a
    position cannot be a marker, so it is refused here rather than drawn at (0, 0).
    """
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("station_id") or "").strip()]
    entries = []
    for r in rows:
        sid = r["station_id"].strip()
        if not (r.get("lat") or "").strip() or not (r.get("lon") or "").strip():
            raise ValueError(f"{path.name}: station {sid} has no coordinates")
        entries.append({
            "id": sid,
            "tc": int(sid[2:]),
            "grid_id": (r.get("grid_id") or "").strip() or None,
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "altitude_m": (r.get("elevation_m") or "").strip() or None,
        })
    entries.sort(key=lambda e: e["tc"])
    return entries


@lru_cache(maxsize=1)
def load_stations() -> dict:
    with open(stations_yaml_path(), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["camera_traps"] = _camera_traps_from_registry(ct_station_registry())
    return data


def tc_coords() -> dict[int, tuple[float, float]]:
    """TC camera number (1..N) → (lat, lon)."""
    return {cam["tc"]: (cam["lat"], cam["lon"]) for cam in load_stations()["camera_traps"]}


def weather_station() -> dict:
    """First weather station in the registry (today always WS-01)."""
    return load_stations()["weather"][0]


def reserve() -> dict:
    """Reserve metadata: name, center [lat, lon], zoom, timezone."""
    return load_stations()["reserve"]
