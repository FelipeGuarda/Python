"""
Canonical species catalog loader.

Reads `data-pipeline/species.yaml` — the single source of truth for
Bosque Pehuén fauna (27 CLIP-classified species + 4 reviewer-added extras).

Mirrors the pattern in `stations.py`: default path assumes FMA monorepo
layout (data-pipeline/ at siblings under one parent); override via
FMA_SPECIES_YAML env var for container / relocated deployments.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_ENV_VAR = "FMA_SPECIES_YAML"

# data-pipeline/src/species.py → parents[1] = data-pipeline root
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "species.yaml"


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
    """
    Lowercased Spanish common name (or alias) → scientific name.

    Used to recover scientificName for rows reviewed via "Otro (especificar)"
    where the app didn't fill in a latin name at review time. Aliases cover
    accent-stripped forms (pudu ↔ pudú) and synonyms (zorro ↔ zorro culpeo).
    """
    mapping: dict[str, str] = {}
    for sp in load_species():
        mapping[sp["spanish"].lower()] = sp["latin"]
        for alias in sp.get("spanish_aliases", []) or []:
            mapping[alias.lower()] = sp["latin"]
    return mapping


@lru_cache(maxsize=1)
def common_names() -> dict[str, str]:
    """Scientific name → canonical Spanish common name. Used for display + dir slugging."""
    return {sp["latin"]: sp["spanish"] for sp in load_species()}


@lru_cache(maxsize=1)
def clip_species() -> list[dict]:
    """Species that carry an english CLIP prompt — the classifier's input set."""
    return [sp for sp in load_species() if sp.get("english")]


@lru_cache(maxsize=1)
def invasive_latins() -> frozenset[str]:
    """Set of scientific names flagged is_invasive: true."""
    return frozenset(sp["latin"] for sp in load_species() if sp.get("is_invasive"))


@lru_cache(maxsize=1)
def priority_latins() -> frozenset[str]:
    """Set of scientific names flagged is_priority: true (conservation)."""
    return frozenset(sp["latin"] for sp in load_species() if sp.get("is_priority"))
