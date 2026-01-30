"""
Prepare training dataset for fire prediction ML model.

This script:
1. Loads historical fire data (GeoJSON)
2. Filters to Araucania region and recent years
3. For each fire, fetches historical weather data from Open-Meteo
4. Creates negative samples (non-fire days from same region/season)
5. Computes the 4 features: temp_c, rh_pct, wind_kmh, days_no_rain
6. Saves balanced training dataset as CSV

Usage:
    python prepare_training_data.py
"""

import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Dict, List, Tuple
import random

# Add parent directory to path to import dashboard modules
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import our existing risk calculator for days_no_rain computation
from risk_calculator import compute_days_without_rain

# Configuration
ARAUCANIA_LAT_MIN = -40.0
ARAUCANIA_LAT_MAX = -38.0
ARAUCANIA_LON_MIN = -73.0
ARAUCANIA_LON_MAX = -71.0

MIN_FIRE_SIZE_HA = 10  # Only use fires >= 10 hectares
START_YEAR = 2015  # Focus on recent years (Open-Meteo historical data available 2015+)
END_YEAR = 2024

# Sampling (set to None to use all data, or set a number for faster testing)
MAX_SAMPLES = None  # e.g., 200 for quick test, None for full dataset

# File paths (relative to ml_model directory)
FIRE_GEOJSON_PATH = Path(__file__).parent / "data" / "cicatrices_incendios_resumen.geojson"
OUTPUT_CSV_PATH = Path(__file__).parent / "training_data.csv"

# Open-Meteo Historical Weather API
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_fire_data() -> pd.DataFrame:
    """Load and filter fire dataset."""
    print("Loading fire dataset...")
    with open(FIRE_GEOJSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract features to DataFrame
    records = []
    for feature in data['features']:
        props = feature.get('properties', {})
        records.append(props)
    
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} fires total")
    
    # Parse dates
    df['ign_date'] = pd.to_datetime(df['ign_date_conaf'], errors='coerce')
    df['year'] = df['ign_date'].dt.year
    
    # Filter to valid data
    df = df[df['ign_date'].notna()].copy()
    df = df[df['lat'].notna()].copy()
    df = df[df['lon'].notna()].copy()
    
    # Filter to Araucania region (using region name - more accurate than lat/lon bounds)
    df = df[df['region_conaf'].str.contains('Araucan', na=False, case=False)].copy()
    print(f"  Fires in Araucanía: {len(df)}")
    
    # Filter to recent years (Open-Meteo historical data availability)
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)].copy()
    print(f"  Fires {START_YEAR}-{END_YEAR}: {len(df)}")
    
    # Filter to significant fires
    df = df[df['area_conaf'] >= MIN_FIRE_SIZE_HA].copy()
    print(f"  Fires >= {MIN_FIRE_SIZE_HA} ha: {len(df)}")
    
    return df


def fetch_historical_weather(lat: float, lon: float, date: datetime) -> Dict:
    """
    Fetch historical weather for a specific location and date.
    Returns afternoon (14:00-16:00) averages for fire risk variables.
    """
    # Open-Meteo requires date range
    start_date = (date - timedelta(days=30)).strftime("%Y-%m-%d")  # Need context for days_no_rain
    end_date = date.strftime("%Y-%m-%d")
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m',
        'daily': 'precipitation_sum',
        'timezone': 'America/Santiago'
    }
    
    try:
        response = requests.get(HISTORICAL_WEATHER_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Extract target date weather (afternoon 14:00-16:00)
        hourly = pd.DataFrame({
            'time': pd.to_datetime(data['hourly']['time']),
            'temp_c': data['hourly']['temperature_2m'],
            'rh_pct': data['hourly']['relative_humidity_2m'],
            'wind_ms': data['hourly']['wind_speed_10m']
        })
        
        # Filter to target date, afternoon hours
        target_date = pd.Timestamp(date).normalize()
        afternoon = hourly[
            (hourly['time'].dt.date == target_date.date()) &
            (hourly['time'].dt.hour >= 14) &
            (hourly['time'].dt.hour <= 16)
        ]
        
        if afternoon.empty:
            return None
        
        # Compute afternoon averages
        temp_c = afternoon['temp_c'].mean()
        rh_pct = afternoon['rh_pct'].mean()
        wind_kmh = afternoon['wind_ms'].mean() * 3.6  # Convert m/s to km/h
        
        # Compute days without rain (using daily precipitation data)
        daily = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'precip_mm': data['daily']['precipitation_sum']
        })
        
        # Use our existing function to compute days_no_rain
        daily_with_dry = compute_days_without_rain(daily, rain_threshold_mm=2.0)
        target_row = daily_with_dry[daily_with_dry['date'].dt.date == target_date.date()]
        
        if target_row.empty:
            return None
        
        days_no_rain = int(target_row.iloc[0]['days_no_rain'])
        
        return {
            'temp_c': temp_c,
            'rh_pct': rh_pct,
            'wind_kmh': wind_kmh,
            'days_no_rain': days_no_rain
        }
    
    except Exception as e:
        print(f"    Error fetching weather for {date.date()}: {e}")
        return None


def generate_negative_samples(fire_df: pd.DataFrame, n_samples: int) -> List[Tuple[float, float, datetime]]:
    """
    Generate negative samples (non-fire days) from same region and seasons as fires.
    Returns list of (lat, lon, date) tuples.
    """
    print(f"\nGenerating {n_samples} negative samples...")
    
    # Extract fire months (to match seasonality)
    fire_months = fire_df['ign_date'].dt.month.unique()
    
    negatives = []
    attempts = 0
    max_attempts = n_samples * 10
    
    while len(negatives) < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Random location in Araucania
        lat = random.uniform(ARAUCANIA_LAT_MIN, ARAUCANIA_LAT_MAX)
        lon = random.uniform(ARAUCANIA_LON_MIN, ARAUCANIA_LON_MAX)
        
        # Random date in same years and months as fires
        year = random.randint(START_YEAR, END_YEAR)
        month = random.choice(fire_months)
        day = random.randint(1, 28)  # Safe for all months
        
        try:
            date = datetime(year, month, day)
        except ValueError:
            continue
        
        # Check this date isn't within 7 days of any actual fire
        is_near_fire = any(
            abs((date - fire_date).days) <= 7
            for fire_date in fire_df['ign_date']
        )
        
        if not is_near_fire:
            negatives.append((lat, lon, date))
            
            if len(negatives) % 100 == 0:
                print(f"  Generated {len(negatives)}/{n_samples} negative samples...")
    
    print(f"  ✓ Generated {len(negatives)} negative samples (attempts: {attempts})")
    return negatives


def build_training_dataset(fire_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build complete training dataset with positive (fire) and negative (non-fire) samples.
    """
    print("\n" + "="*60)
    print("BUILDING TRAINING DATASET")
    print("="*60)
    
    # Sample if requested
    if MAX_SAMPLES is not None and len(fire_df) > MAX_SAMPLES:
        print(f"\nSampling {MAX_SAMPLES} fires from {len(fire_df)} total (for faster processing)...")
        fire_df = fire_df.sample(n=MAX_SAMPLES, random_state=42).copy()
    
    training_records = []
    
    # Positive samples (fires)
    print(f"\nProcessing {len(fire_df)} fire samples...")
    
    for idx, row in fire_df.iterrows():
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(fire_df)} fires processed...")
        
        lat = row['lat']
        lon = row['lon']
        date = row['ign_date']
        
        # Fetch weather data
        weather = fetch_historical_weather(lat, lon, date)
        
        if weather is None:
            continue
        
        training_records.append({
            'lat': lat,
            'lon': lon,
            'date': date.strftime('%Y-%m-%d'),
            'year': date.year,
            'month': date.month,
            'temp_c': weather['temp_c'],
            'rh_pct': weather['rh_pct'],
            'wind_kmh': weather['wind_kmh'],
            'days_no_rain': weather['days_no_rain'],
            'fire': 1,  # Target label
            'fire_size_ha': row['area_conaf']
        })
        
        # Rate limiting (be nice to Open-Meteo API)
        time.sleep(0.1)
    
    print(f"  ✓ Successfully processed {len(training_records)} fire samples")
    
    # Negative samples (non-fires)
    n_negatives = len(training_records)  # 1:1 ratio
    negative_locations = generate_negative_samples(fire_df, n_negatives)
    
    print(f"\nProcessing {len(negative_locations)} non-fire samples...")
    
    for idx, (lat, lon, date) in enumerate(negative_locations):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(negative_locations)} non-fires processed...")
        
        weather = fetch_historical_weather(lat, lon, date)
        
        if weather is None:
            continue
        
        training_records.append({
            'lat': lat,
            'lon': lon,
            'date': date.strftime('%Y-%m-%d'),
            'year': date.year,
            'month': date.month,
            'temp_c': weather['temp_c'],
            'rh_pct': weather['rh_pct'],
            'wind_kmh': weather['wind_kmh'],
            'days_no_rain': weather['days_no_rain'],
            'fire': 0,  # Target label
            'fire_size_ha': 0.0
        })
        
        # Rate limiting
        time.sleep(0.1)
    
    print(f"  ✓ Successfully processed {sum(1 for r in training_records if r['fire'] == 0)} non-fire samples")
    
    # Convert to DataFrame
    df = pd.DataFrame(training_records)
    
    print(f"\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"  Fire samples: {(df['fire'] == 1).sum()}")
    print(f"  Non-fire samples: {(df['fire'] == 0).sum()}")
    print(f"\nFeature statistics:")
    print(df[['temp_c', 'rh_pct', 'wind_kmh', 'days_no_rain']].describe())
    
    return df


def main():
    """Main execution."""
    print("="*60)
    print("FIRE PREDICTION MODEL - DATA PREPARATION")
    print("="*60)
    
    # Load fire data
    fire_df = load_fire_data()
    
    if len(fire_df) == 0:
        print("\nERROR: No fires found matching criteria!")
        print("Try adjusting START_YEAR, MIN_FIRE_SIZE_HA, or region bounds")
        return 1
    
    # Build training dataset
    training_df = build_training_dataset(fire_df)
    
    # Save to CSV
    training_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✓ Training dataset saved to: {OUTPUT_CSV_PATH}")
    print(f"  Shape: {training_df.shape}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review training_data.csv to verify data quality")
    print("2. Run: python train_fire_model.py")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

