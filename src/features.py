"""
Feature Engineering Module for Nagpur May Heatwave Prediction.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).

Soft Computing Connection:
- In Neural Networks (McCulloch-Pitts, Perceptron, ADALINE, and EBPA), input
  representation defines whether classes are linearly separable or require
  nonlinear hidden projections.
- Excitatory inputs (high lagged temperatures, extreme anomalies, hot streaks)
  and inhibitory inputs (high humidity, cooling wind) mimic biological dendrites
  as modeled in McCulloch-Pitts neurons (Exp 1).
- Time-series lags (L1, L2, L3) and rolling windows capture atmospheric inertia
  and thermal accumulation, essential for Error Backpropagation ANNs (Exp 2, 6, 7).
- Linguistic antecedents for Mamdani Fuzzy Inference (Module 4) rely on lagged
  temperature, departure anomaly, and relative humidity.

Crucial Temporal Integrity Rule:
To strictly prevent same-day data leakage, any prediction for May day T must ONLY
use meteorological information observed up to day T-1, plus deterministic
calendar and WMO 1991-2020 climatology normals for day T itself.
"""

import logging
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# List of 26 engineered meteorological, rolling, trend, and calendar features
FEATURE_COLS = [
    # Lag 1 Atmospheric Predictors (Day T-1)
    "tmax_lag1",
    "tmin_lag1",
    "tmean_lag1",
    "rh_mean_lag1",
    "rh_min_lag1",
    "dewpoint_mean_lag1",
    "apparent_tmax_lag1",
    "wind_mean_lag1",
    "pressure_mean_lag1",
    "solar_sum_lag1",
    "soil_moisture_lag1",
    # Multi-day Lag Predictors (Days T-2 and T-3)
    "tmax_lag2",
    "tmax_lag3",
    "apparent_tmax_lag2",
    "apparent_tmax_lag3",
    # Multi-day Thermal & Solar Accumulation (Rolling Windows T-3..T-1 and T-7..T-1)
    "tmax_roll3",
    "tmax_roll7",
    "solar_roll3",
    # Atmospheric Tendencies, Diurnal Range & Persistence
    "tmax_trend3",
    "tmax_anomaly_lag1",
    "dtr_lag1",
    "pressure_tendency",
    "hot_streak_len",
    # Deterministic Climatology & Cyclical Solar Calendar
    "normal_tmax_T",
    "day_sin",
    "day_cos",
]

METADATA_COLS = ["date", "year", "month", "day"]
TARGET_COLS = [
    "heatwave_day",
    "imd_reanalysis_plains",
    "climatology_p85_42c",
    "imd_strict_raw",
]


def compute_features(
    df_daily: pd.DataFrame,
    df_normals: Optional[pd.DataFrame] = None,
    normals_path: str = "data/processed/climatology_normals.csv",
) -> pd.DataFrame:
    """
    Compute ~24-26 predictive features for each May day T using ONLY information
    from days <= T-1 and deterministic calendar/climatology for day T.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily aggregated meteorological records (must include April days for lags).
    df_normals : pd.DataFrame, optional
        Preloaded WMO 1991-2020 climatology normals.
    normals_path : str
        File path to climatology normals if df_normals is not provided.

    Returns
    -------
    pd.DataFrame
        Engineered feature table containing only May rows (month == 5),
        with metadata, features, and target labels.
    """
    if df_normals is None:
        p = Path(normals_path)
        if not p.exists():
            raise FileNotFoundError(f"Climatology normals file not found at: {normals_path}")
        df_normals = pd.read_csv(p)

    # Build calendar lookup for deterministic normal_tmax of day T
    norm_lookup = {
        (int(row["month"]), int(row["day"])): float(row["normal_tmax"])
        for _, row in df_normals.iterrows()
    }

    # Ensure temporal ordering
    df = df_daily.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date_dt"] = pd.to_datetime(df["date"])
    else:
        df["date_dt"] = df["date"]

    df = df.sort_values("date_dt").reset_index(drop=True)

    yearly_processed = []

    # Process each year independently to prevent cross-year leakage
    # (e.g., late May of year Y should not bleed into early April of year Y+1)
    for year, grp in df.groupby("year", sort=True):
        g = grp.sort_values("date_dt").copy()

        # 1. Lag 1 features (Day T-1 meteorological state)
        g["tmax_lag1"] = g["tmax"].shift(1)
        g["tmin_lag1"] = g["tmin"].shift(1)
        g["tmean_lag1"] = g["tmean"].shift(1)
        g["rh_mean_lag1"] = g["rh_mean"].shift(1)
        g["rh_min_lag1"] = g["rh_min"].shift(1)
        g["dewpoint_mean_lag1"] = g["dewpoint_mean"].shift(1)
        g["apparent_tmax_lag1"] = g["apparent_tmax"].shift(1)
        g["wind_mean_lag1"] = g["wind_mean"].shift(1)
        g["pressure_mean_lag1"] = g["pressure_mean"].shift(1)
        g["solar_sum_lag1"] = g["solar_sum"].shift(1)
        g["soil_moisture_lag1"] = g["soil_moisture_mean"].shift(1)

        # 2. Lag 2 and Lag 3 features
        g["tmax_lag2"] = g["tmax"].shift(2)
        g["tmax_lag3"] = g["tmax"].shift(3)
        g["apparent_tmax_lag2"] = g["apparent_tmax"].shift(2)
        g["apparent_tmax_lag3"] = g["apparent_tmax"].shift(3)

        # 3. Multi-day thermal & solar accumulation (strictly T-1 backwards)
        # Shift 1 then roll 3 covers [T-3, T-2, T-1]
        g["tmax_roll3"] = np.round(
            g["tmax"].shift(1).rolling(window=3, min_periods=3).mean(), 4
        )
        # Shift 1 then roll 7 covers [T-7..T-1]
        g["tmax_roll7"] = np.round(
            g["tmax"].shift(1).rolling(window=7, min_periods=7).mean(), 4
        )
        g["solar_roll3"] = np.round(
            g["solar_sum"].shift(1).rolling(window=3, min_periods=3).mean(), 4
        )

        # 4. Atmospheric Trends, Diurnal Range & Anomaly
        # 3-day warming trend up to T-1
        g["tmax_trend3"] = np.round(g["tmax"].shift(1) - g["tmax"].shift(3), 2)
        # Climatological departure at T-1
        g["tmax_anomaly_lag1"] = np.round(g["tmax"].shift(1) - g["normal_tmax"].shift(1), 2)
        # Diurnal Temperature Range (DTR) at T-1
        g["dtr_lag1"] = np.round(g["tmax"].shift(1) - g["tmin"].shift(1), 2)
        # 3-day barometric pressure tendency (falling pressure precedes thermal lows)
        g["pressure_tendency"] = np.round(g["pressure_mean"].shift(1) - g["pressure_mean"].shift(3), 2)

        # 5. Hot streak length: consecutive days up to T-1 where heatwave_day == 1
        heatwave_vals = g["heatwave_day"].tolist()
        streak_ending_at = []
        running_streak = 0
        for val in heatwave_vals:
            if val == 1:
                running_streak += 1
            else:
                running_streak = 0
            streak_ending_at.append(running_streak)

        # hot_streak_len for day T is streak ending at day T-1
        g["hot_streak_len"] = pd.Series(streak_ending_at, index=g.index).shift(1).fillna(0).astype(int)

        # 6. Calendar & Climatology for day T itself (strictly deterministic, zero leakage)
        g["normal_tmax_T"] = [
            norm_lookup.get((int(m), int(d)), np.nan) for m, d in zip(g["month"], g["day"])
        ]
        # Day of year cyclical harmonic encoding (Soft Computing time-feature representation)
        doy = g["date_dt"].dt.dayofyear
        g["day_sin"] = np.round(np.sin(2.0 * np.pi * doy / 365.0), 6)
        g["day_cos"] = np.round(np.cos(2.0 * np.pi * doy / 365.0), 6)

        yearly_processed.append(g)

    df_full = pd.concat(yearly_processed, ignore_index=True)

    # Filter strictly for May (month == 5) prediction targets
    # April rows served their sole purpose of priming lags and 7-day rolling windows
    df_may = df_full[df_full["month"] == 5].copy().reset_index(drop=True)

    # Select and order final columns: metadata + features + targets
    output_cols = METADATA_COLS + FEATURE_COLS + TARGET_COLS
    df_out = df_may[output_cols].copy()

    # Drop temporary column if present
    if "date_dt" in df_out.columns:
        df_out = df_out.drop(columns=["date_dt"])

    return df_out


def build_and_save_features(
    config: Optional[dict] = None,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """
    Main function to load daily data, generate engineered features, and save features.csv.
    """
    if config is None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    daily_path = config["data"].get("daily_file", "data/processed/daily.csv")
    features_path = config["data"].get("features_file", "data/processed/features.csv")
    normals_path = "data/processed/climatology_normals.csv"

    logger.info("Loading daily meteorological records from: %s", daily_path)
    df_daily = pd.read_csv(daily_path)

    logger.info("Engineering 26 lag, rolling, trend, and climatological features...")
    df_features = compute_features(df_daily=df_daily, normals_path=normals_path)

    # Verification checks
    null_counts = df_features[FEATURE_COLS].isnull().sum().sum()
    if null_counts > 0:
        raise ValueError(f"Feature matrix contains {null_counts} null values!")

    out_file = Path(features_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out_file, index=False)

    logger.info(
        "Successfully saved engineered features to %s (Rows: %d, Columns: %d)",
        features_path,
        len(df_features),
        len(df_features.columns),
    )
    return df_features


if __name__ == "__main__":
    build_and_save_features()
