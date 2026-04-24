"""
Canonical station registry loader.

Reads `plataforma-territorial/data/stations.yaml` — the single source of
truth for Bosque Pehuén monitoring stations. Default resolution assumes
the FMA monorepo layout (data-pipeline/ and plataforma-territorial/ as
siblings under one parent). Override via FMA_STATIONS_YAML env var when
the layout differs (e.g. container deployments).
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
