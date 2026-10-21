"""
Hourly-to-Daily Meteorological Aggregation Module.

Processes 46 years of raw hourly Open-Meteo reanalysis files in `data/raw/`
into local calendar day (Asia/Kolkata) records for April 1 - May 31 (1981-2026).

Produces `data/processed/daily.csv` containing:
- date, year, month, day, day_of_year
- tmax, tmin, tmean
- rh_mean, rh_min
- dewpoint_mean
- apparent_tmax
- wind_mean
- pressure_mean
- solar_sum
- soil_moisture_mean
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def aggregate_year_hourly_to_daily(year_json_path: Path) -> pd.DataFrame:
    """Aggregate a single year's hourly JSON file into daily summary records."""
    with open(year_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError(f"Invalid hourly data in {year_json_path}")

    df_hourly = pd.DataFrame(hourly)
    df_hourly["time"] = pd.to_datetime(df_hourly["time"])
    if data.get('timezone') != 'Asia/Kolkata':
        raise ValueError('Hourly input must use Asia/Kolkata local calendar time')
    if df_hourly['time'].duplicated().any() or not (df_hourly['time'].diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError('Hourly observations must be unique and contiguous')
    if df_hourly.drop(columns=['time']).isna().any().any():
        raise ValueError('Missing hourly measurements cannot be silently averaged')
    df_hourly["date"] = df_hourly["time"].dt.date

    # Group by local calendar date (Asia/Kolkata)
    daily_records = []
    for day_date, group in df_hourly.groupby("date"):
        if len(group) != 24:
            raise ValueError(f'Date {day_date} has {len(group)} hours; expected 24')

        tmax = group["temperature_2m"].max()
        tmin = group["temperature_2m"].min()
        tmean = group["temperature_2m"].mean()
        rh_mean = group["relative_humidity_2m"].mean()
        rh_min = group["relative_humidity_2m"].min()
        dewpoint_mean = group["dew_point_2m"].mean()
        apparent_tmax = group["apparent_temperature"].max()
        wind_mean = group["wind_speed_10m"].mean()
        pressure_mean = group["surface_pressure"].mean()
        # Hourly mean irradiance W/m² integrated over 1 h gives Wh/m²/day.
        solar_sum = group["shortwave_radiation"].sum()
        soil_moisture_mean = (
            group["soil_moisture_0_to_7cm"].mean()
            if "soil_moisture_0_to_7cm" in group.columns
            else np.nan
        )

        daily_records.append({
            "date": str(day_date),
            "year": day_date.year,
            "month": day_date.month,
            "day": day_date.day,
            "day_of_year": day_date.timetuple().tm_yday,
            "tmax": round(float(tmax), 2),
            "tmin": round(float(tmin), 2),
            "tmean": round(float(tmean), 2),
            "rh_mean": round(float(rh_mean), 2),
            "rh_min": round(float(rh_min), 2),
            "dewpoint_mean": round(float(dewpoint_mean), 2),
            "apparent_tmax": round(float(apparent_tmax), 2),
            "wind_mean": round(float(wind_mean), 2),
            "pressure_mean": round(float(pressure_mean), 2),
            "solar_sum": round(float(solar_sum), 2),
            "soil_moisture_mean": round(float(soil_moisture_mean), 4) if not np.isnan(soil_moisture_mean) else np.nan,
        })

    return pd.DataFrame(daily_records)


def process_hourly_to_daily(config: dict) -> pd.DataFrame:
    """Process all cached hourly files in raw_dir and save consolidated daily.csv."""
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(config["data"]["daily_file"])

    start_year = config["data"]["years"]["start"]
    end_year = config["data"]["years"]["end"]

    all_daily_dfs = []
    logger.info(f"Aggregating hourly data for years {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        year_file = raw_dir / f"raw_hourly_{year}.json"
        if not year_file.exists():
            raise FileNotFoundError(
                f"Raw hourly file missing for {year}: {year_file}. Run fetch_data first."
            )
        df_year = aggregate_year_hourly_to_daily(year_file)
        all_daily_dfs.append(df_year)

    df_daily = pd.concat(all_daily_dfs, ignore_index=True)
    df_daily.sort_values(by="date", inplace=True)
    df_daily.reset_index(drop=True, inplace=True)

    # Sanity checks
    expected_rows = (end_year - start_year + 1) * 61  # 30 April days + 31 May days = 61 days
    logger.info(f"Total aggregated daily rows: {len(df_daily)} (Expected: {expected_rows})")
    assert len(df_daily) == expected_rows, f"Row count mismatch: got {len(df_daily)}, expected {expected_rows}"

    may_rows = len(df_daily[df_daily["month"] == 5])
    expected_may_rows = (end_year - start_year + 1) * 31
    logger.info(f"Total May rows: {may_rows} (Expected: {expected_may_rows})")
    assert may_rows == expected_may_rows, f"May row count mismatch: got {may_rows}, expected {expected_may_rows}"

    # Missing value audit
    null_counts = df_daily.isnull().sum()
    logger.info(f"Null value counts across all columns:\n{null_counts[null_counts > 0]}")
    assert null_counts.sum() == 0, f"Found unexpected null values in daily aggregation:\n{null_counts}"

    # Save to processed CSV
    df_daily.to_csv(output_csv, index=False)
    logger.info(f"Successfully saved clean daily dataset to {output_csv} ({len(df_daily)} rows, {len(df_daily.columns)} columns)")

    return df_daily


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as cfg_f:
        config_data = yaml.safe_load(cfg_f)
    process_hourly_to_daily(config_data)
