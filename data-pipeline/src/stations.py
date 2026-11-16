"""
Station registry loader.

Camera-trap coordinates come from `camera-traps/data/campaigns/estaciones.csv`, read
directly: that CSV owns station identity and position. Same source and same refusal
rule as `plataforma-territorial/backend/stations.py`, so the warehouse and the map
cannot disagree about where a camera is.

Until 2026-09-03 the registry builder spliced a `camera_traps:` section into
`plataforma-territorial/data/stations.yaml` and this module read it from there. The
splice was removed (it made the producer know a consumer's file layout); the platform
moved to the CSV the same day, this module did not, and `run_fetch.py --ct` failed
with KeyError 'camera_traps' until 2026-11-16. stations.yaml now holds only `reserve`
and `weather`.

NOT VERIFIED against the contract: CANONICAL_STATE.json hashes estaciones.geojson,
not this CSV. See SUPPORTED_SCHEMA_VERSION in src/canonical_gate.py.

Override with CT_STATION_REGISTRY (registry) or FMA_STATIONS_YAML (yaml) when the
layout differs (e.g. container deployments).
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

import yaml

_ENV_VAR = "FMA_STATIONS_YAML"

# data-pipeline/src/stations.py → parents[2] = monorepo root
_REGISTRY_ENV_VAR = "CT_STATION_REGISTRY"
_DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[2] / "camera-traps" / "data" / "campaigns" / "estaciones.csv"
)
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


@lru_cache(maxsize=1)
def tc_coords() -> dict[int, tuple[float, float]]:
    """Camera number (CT07 -> 7) → (lat, lon), from the producer's registry.

    An empty coordinate cell is NOT RECORDED; a station without a position is refused
    rather than written to the warehouse as NULL or (0, 0).
    """
    path = Path(os.getenv(_REGISTRY_ENV_VAR, str(_DEFAULT_REGISTRY)))
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("station_id") or "").strip()]
    coords = {}
    for r in rows:
        sid = r["station_id"].strip()
        if not (r.get("lat") or "").strip() or not (r.get("lon") or "").strip():
            raise ValueError(f"{path.name}: station {sid} has no coordinates")
        coords[int(sid[2:])] = (float(r["lat"]), float(r["lon"]))
    return coords
