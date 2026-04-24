"""
Station registry loader for the plataforma-territorial backend.

Loads `plataforma-territorial/data/stations.yaml` — the single source of
truth for Bosque Pehuén monitoring stations. Override with the
FMA_STATIONS_YAML env var when running outside the repo layout.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_ENV_VAR = "FMA_STATIONS_YAML"

# backend/stations.py → parents[1] = plataforma-territorial/
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "stations.yaml"


def stations_yaml_path() -> Path:
    override = os.getenv(_ENV_VAR)
    return Path(override) if override else _DEFAULT_PATH


@lru_cache(maxsize=1)
def load_stations() -> dict:
    path = stations_yaml_path()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tc_coords() -> dict[int, tuple[float, float]]:
    """TC camera number (1..N) → (lat, lon)."""
    data = load_stations()
    return {int(cam["tc"]): (float(cam["lat"]), float(cam["lon"])) for cam in data["camera_traps"]}


def weather_station() -> dict:
    """First weather station in the registry (today always WS-01)."""
    return load_stations()["weather"][0]


def reserve() -> dict:
    """Reserve metadata: name, center [lat, lon], zoom, timezone."""
    return load_stations()["reserve"]
