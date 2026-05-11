"""
FMA Plataforma Territorial — FastAPI Backend

Serves weather, fire risk, and camera trap data from fma_data.duckdb.
The data-pipeline systemd service writes to the DB on a schedule;
this backend reads from it.

Normal use (single process, serves UI + API):
    uvicorn backend.main:app --port 8000
    → open http://localhost:8000

Frontend development (hot reload):
    uvicorn backend.main:app --port 8000   (terminal 1)
    cd plataforma-demo && npm run dev      (terminal 2)
    → open http://localhost:5173
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import config_router, detections, fire_risk_router, weather

logger = logging.getLogger(__name__)

DIST_DIR = Path(__file__).resolve().parent.parent / "plataforma-demo" / "dist"

# Camera trap image exports — served at /ct-images/<campaign>/stations/<station>/<file>
# Same default path logic as detections.py; override with CT_EXPORTS_DIR env var.
_DEFAULT_CT_EXPORTS = Path(__file__).resolve().parents[2] / "camera-traps" / "exports"
CT_EXPORTS_DIR = Path(os.getenv("CT_EXPORTS_DIR", str(_DEFAULT_CT_EXPORTS)))

app = FastAPI(
    title="FMA Plataforma Territorial API",
    description="Backend API for Bosque Pehuén territorial platform",
    version="0.1.0",
)

_DEFAULT_CORS = "http://localhost:5173,http://localhost:4173,http://localhost:3000,http://127.0.0.1:5173"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", _DEFAULT_CORS).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(weather.router)
app.include_router(fire_risk_router.router)
app.include_router(detections.router)
app.include_router(config_router.router)


@app.get("/api/health")
def health():
    """Health check — also verifies DB connectivity. Returns 503 if DB unreachable."""
    from .db import get_connection
    try:
        with get_connection() as con:
            count = con.execute("SELECT COUNT(*) FROM weather_station").fetchone()[0]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")
    return {"status": "ok", "weather_station_rows": count}


# Serve camera trap station images — must come before the catch-all React mount.
# Images are large files (gitignored); they live in the camera-traps repo on the host.
if CT_EXPORTS_DIR.exists():
    app.mount("/ct-images", StaticFiles(directory=CT_EXPORTS_DIR), name="ct-images")

# Serve the built React app — must be mounted AFTER all API routes.
# Run `cd plataforma-demo && npm run build` to update after frontend changes.
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
else:
    logger.warning("Frontend dist/ not found at %s — React UI unavailable. Run: cd plataforma-demo && npm run build", DIST_DIR)
