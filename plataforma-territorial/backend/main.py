"""
FMA Plataforma Territorial — FastAPI Backend

Serves weather, fire risk, and camera trap data from fma_data.duckdb.
The data-pipeline systemd service writes to the DB on a schedule;
this backend reads from it.

Run:
    uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import weather, fire_risk_router, detections

app = FastAPI(
    title="FMA Plataforma Territorial API",
    description="Backend API for Bosque Pehuén territorial platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
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
