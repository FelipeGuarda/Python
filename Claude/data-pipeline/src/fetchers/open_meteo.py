import pandas as pd
import requests
import yaml
from pathlib import Path
from datetime import datetime, timezone

_config_path = Path(__file__).parent.parent.parent / "config.yaml"
with open(_config_path) as f:
    _cfg = yaml.safe_load(f)["open_meteo"]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch() -> pd.DataFrame:
    """Fetch hourly forecast for Bosque Pehuén. Returns DataFrame matching weather_forecast schema."""
    print("→ Fetching Open-Meteo data...")

    params = {
        "latitude": _cfg["latitude"],
        "longitude": _cfg["longitude"],
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "et0_fao_evapotranspiration",
        ]),
        "timezone": "America/Santiago",
    }

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]).tz_localize("America/Santiago").tz_convert("UTC"),
        "temperature_2m": hourly["temperature_2m"],
        "relative_humidity_2m": hourly["relative_humidity_2m"],
        "precipitation": hourly["precipitation"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "wind_direction_10m": hourly["wind_direction_10m"],
        "et0_fao_evapotranspiration": hourly["et0_fao_evapotranspiration"],
        "fetched_at": datetime.now(timezone.utc),
    })

    print(f"  Got {len(df)} hourly rows from Open-Meteo.")
    return df
