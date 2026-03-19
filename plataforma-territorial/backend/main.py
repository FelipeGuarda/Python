from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import weather

app = FastAPI(title="Plataforma Territorial FMA — API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(weather.router, prefix="/api/weather", tags=["weather"])


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
