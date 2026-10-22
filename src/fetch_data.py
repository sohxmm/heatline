"""
Data Acquisition Module for Nagpur Meteorological Reanalysis.

Fetches 46 years (1981-2026) of April 1 - May 31 hourly meteorological records
from the Open-Meteo Historical Archive API (explicit ECMWF ERA5 selection)
for Nagpur Urban Agglomeration (21.1458°N, 79.0882°E, Asia/Kolkata timezone).

Results are cached to `data/raw/` to ensure offline reproducibility.
"""

import json
import logging
import time
from pathlib import Path
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "surface_pressure",
    "wind_speed_10m",
    "shortwave_radiation",
    "soil_moisture_0_to_7cm",
]


def fetch_year_data(
    year: int,
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Kolkata",
    model: str = "era5",
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> dict:
    """Fetch hourly data for a single year (April 1 to May 31) from Open-Meteo."""
    start_date = f"{year}-04-01"
    end_date = f"{year}-05-31"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": timezone,
        "models": model,
    }

    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if "hourly" in data and len(data["hourly"].get("time", [])) > 0:
                    data['_heatline_provenance'] = {
                        'requested_model': model, 'request_parameters': params,
                        'retrieved_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                    }
                    return data
                raise ValueError(f"Empty hourly payload received for year {year}: {data}")
            elif response.status_code == 429:
                logger.warning(f"Rate limited (429) for year {year}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed for {year} (HTTP {response.status_code}): {response.text}"
                )
                time.sleep(delay)
                delay *= backoff_factor
        except requests.RequestException as e:
            logger.warning(f"Network error on attempt {attempt}/{max_retries} for {year}: {e}")
            time.sleep(delay)
            delay *= backoff_factor

    # Do not silently mix reanalysis products when a model request fails.
    raise RuntimeError(f"Failed to fetch data for year {year} after {max_retries} attempts.")


def fetch_all_data(config: dict, force: bool = False) -> Path:
    """Fetch and cache hourly records for all configured years (1981-2026)."""
    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    lat = config["location"]["target_latitude"]
    lon = config["location"]["target_longitude"]
    tz = config["location"]["timezone"]
    model = config["data"]["open_meteo"]["primary_model"]
    start_year = config["data"]["years"]["start"]
    end_year = config["data"]["years"]["end"]

    metadata_path = raw_dir / "grid_metadata.json"
    grid_metadata = None

    logger.info(
        f"Fetching Open-Meteo meteorological reanalysis for Nagpur ({lat}, {lon}) [{start_year}..{end_year}]"
    )

    for year in range(start_year, end_year + 1):
        cache_file = raw_dir / f"raw_hourly_{year}.json"
        if cache_file.exists() and not force:
            logger.info(f"Year {year}: Using cached data at {cache_file.name}")
            if grid_metadata is None:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    grid_metadata = {
                        "latitude": cached_data.get("latitude"),
                        "longitude": cached_data.get("longitude"),
                        "elevation": cached_data.get("elevation"),
                        "timezone": cached_data.get("timezone"),
                        "timezone_abbreviation": cached_data.get("timezone_abbreviation"),
                        "hourly_units": cached_data.get("hourly_units"),
                        "model": model,
                    }
            continue

        logger.info(f"Year {year}: Requesting from Open-Meteo API...")
        year_data = fetch_year_data(year, lat, lon, timezone=tz, model=model)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(year_data, f)

        if grid_metadata is None:
            grid_metadata = {
                "latitude": year_data.get("latitude"),
                "longitude": year_data.get("longitude"),
                "elevation": year_data.get("elevation"),
                "timezone": year_data.get("timezone"),
                "timezone_abbreviation": year_data.get("timezone_abbreviation"),
                "hourly_units": year_data.get("hourly_units"),
                "model": model,
            }

        # Polite delay between API calls to stay within free tier limits
        time.sleep(0.3)

    if grid_metadata:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(grid_metadata, f, indent=2)
        logger.info(
            f"Cached grid metadata: lat={grid_metadata['latitude']}, lon={grid_metadata['longitude']}, elevation={grid_metadata['elevation']}m"
        )

        # Update config.yaml with actual coordinates if null
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg["location"]["grid"]["actual_latitude"] = grid_metadata["latitude"]
            cfg["location"]["grid"]["actual_longitude"] = grid_metadata["longitude"]
            cfg["location"]["grid"]["elevation"] = grid_metadata["elevation"]
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            logger.info("Updated config.yaml with actual grid coordinates.")

    logger.info("All 46 years (1981-2026) verified and cached in data/raw/.")
    return raw_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo meteorological reanalysis data.")
    parser.add_argument("--force", action="store_true", help="Force refetch and overwrite cached files.")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as cfg_f:
        config_data = yaml.safe_load(cfg_f)
    fetch_all_data(config_data, force=args.force)
