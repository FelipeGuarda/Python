"""
Station registry loader.

Reads `plataforma-territorial/data/stations.yaml`, which is GENERATED from
`camera-traps/data/campaigns/estaciones.csv` by
`camera-traps/setup/build_station_registry.py`. That CSV owns station identity;
this file reads a projection of it, and the YAML must not be hand-edited.

Until 2026-08-24 this docstring called stations.yaml "the single source of truth"
while it held 26 stations against the other two registries' 27 — so CT27's otoño
2026 images ingested with no coordinates and nothing raised. A file three modules
call canonical is only canonical if something checks; that check now lives in
`camera-traps/tests/test_station_registry.py`.

Default resolution assumes the FMA monorepo layout (data-pipeline/ and
plataforma-territorial/ as siblings under one parent). Override via
FMA_STATIONS_YAML env var when the layout differs (e.g. container deployments).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_ENV_VAR = "FMA_STATIONS_YAML"

# data-pipeline/src/stations.py → parents[2] = monorepo root
_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "plataforma-territorial" / "data" / "stations.yaml"
)


def stations_yaml_path() -> Path:
    """Resolve the stations.yaml location. Env var wins over default."""
    override = os.getenv(_ENV_VAR)
    return Path(override) if override else _DEFAULT_PATH


@lru_cache(maxsize=1)
def load_stations() -> dict:
    """Parsed stations.yaml as a dict. Cached for process lifetime."""
    path = stations_yaml_path()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tc_coords() -> dict[int, tuple[float, float]]:
    """TC camera number (1..N) → (lat, lon)."""
    data = load_stations()
    return {int(cam["tc"]): (float(cam["lat"]), float(cam["lon"])) for cam in data["camera_traps"]}
