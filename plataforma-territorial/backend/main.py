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

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import weather, fire_risk_router, detections

DIST_DIR = Path(__file__).resolve().parent.parent / "plataforma-demo" / "dist"

app = FastAPI(
    title="FMA Plataforma Territorial API",
    description="Backend API for Bosque Pehuén territorial platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(weather.router)
app.include_router(fire_risk_router.router)
app.include_router(detections.router)


@app.get("/api/health")
def health():
    """Health check — also verifies DB connectivity."""
    from .db import get_connection
    try:
        with get_connection() as con:
            count = con.execute("SELECT COUNT(*) FROM weather_station").fetchone()[0]
        return {"status": "ok", "weather_station_rows": count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# Serve the built React app — must be mounted AFTER all API routes.
# Run `cd plataforma-demo && npm run build` to update after frontend changes.
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
