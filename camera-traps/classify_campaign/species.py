"""
Canonical species catalog loader (camera-traps local copy).

Reads `data-pipeline/species.yaml` — the single source of truth shared
across the FMA ecosystem. This module is a thin sibling of
`data-pipeline/src/species.py`; duplicated because camera-traps runs on
Windows in its own conda env and cannot rely on data-pipeline being on
PYTHONPATH. If you edit one loader, mirror the change in the other.

Default path assumes the monorepo layout (camera-traps/ and data-pipeline/
as siblings). Override via FMA_SPECIES_YAML for relocated deployments.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_ENV_VAR = "FMA_SPECIES_YAML"

# camera-traps/classify_campaign/species.py → parents[2] = monorepo root
_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "data-pipeline" / "species.yaml"
)


def species_yaml_path() -> Path:
    """Resolve the species.yaml location. Env var wins over default."""
    override = os.getenv(_ENV_VAR)
    return Path(override) if override else _DEFAULT_PATH


@lru_cache(maxsize=1)
def load_species() -> list[dict]:
    """Parsed species list as a list of dicts. Cached for process lifetime."""
    path = species_yaml_path()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["species"]


@lru_cache(maxsize=1)
def spanish_to_latin() -> dict[str, str]:
    """Lowercased Spanish common name (or alias) → scientific name."""
    mapping: dict[str, str] = {}
    for sp in load_species():
        mapping[sp["spanish"].lower()] = sp["latin"]
        for alias in sp.get("spanish_aliases", []) or []:
            mapping[alias.lower()] = sp["latin"]
    return mapping


@lru_cache(maxsize=1)
def common_names() -> dict[str, str]:
    """Scientific name → canonical Spanish common name."""
    return {sp["latin"]: sp["spanish"] for sp in load_species()}


@lru_cache(maxsize=1)
def clip_species() -> list[dict]:
    """Species that carry an english CLIP prompt — the classifier's input set."""
    return [sp for sp in load_species() if sp.get("english")]
