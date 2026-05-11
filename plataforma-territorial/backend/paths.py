"""Shared path helpers for the backend.

Centralizes filesystem paths that multiple modules used to compute independently
(with subtly different `parents[N]` indices), so the canonical location stays
in one place.
"""

import os
from pathlib import Path

# backend/paths.py → parents[1] = plataforma-territorial/, parents[2] = repo root
_DEFAULT_CT_EXPORTS = Path(__file__).resolve().parents[2] / "camera-traps" / "exports"


def ct_exports_dir() -> Path:
    """Camera-trap image export tree. Override with CT_EXPORTS_DIR env var."""
    return Path(os.getenv("CT_EXPORTS_DIR", str(_DEFAULT_CT_EXPORTS)))
