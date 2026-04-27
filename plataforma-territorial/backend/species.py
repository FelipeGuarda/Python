"""
Species catalog loader for the plataforma-territorial backend.

Reads `data-pipeline/species.yaml` — the single source of truth shared
across the FMA ecosystem (data-pipeline + camera-traps + plataforma).
Sibling of `data-pipeline/src/species.py` and
`camera-traps/classify_campaign/species.py`. Override the path with the
FMA_SPECIES_YAML env var when running outside the monorepo layout.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_ENV_VAR = "FMA_SPECIES_YAML"

# backend/species.py → parents[2] = monorepo root
_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "data-pipeline" / "species.yaml"
)


def species_yaml_path() -> Path:
    override = os.getenv(_ENV_VAR)
    return Path(override) if override else _DEFAULT_PATH


@lru_cache(maxsize=1)
def load_species() -> list[dict]:
    path = species_yaml_path()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["species"]


@lru_cache(maxsize=1)
def invasive_latins() -> frozenset[str]:
    return frozenset(sp["latin"] for sp in load_species() if sp.get("is_invasive"))


@lru_cache(maxsize=1)
def priority_latins() -> frozenset[str]:
    return frozenset(sp["latin"] for sp in load_species() if sp.get("is_priority"))
