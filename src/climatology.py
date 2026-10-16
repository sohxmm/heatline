"""
WMO 1991-2020 Climatological Baseline & Percentiles Module.

Computes standard WMO 30-year (1991-2020) daily climatological normals
and percentile thresholds using a +/- 7-calendar-day centered rolling window.

Outputs `data/processed/climatology_normals.csv` containing:
- month, day, day_of_year
- normal_tmax, std_tmax, p85_tmax, p90_tmax
- normal_tmin, normal_apparent_tmax, normal_rh_mean
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_wmo_climatology(config: dict) -> pd.DataFrame:
    """Compute WMO 1991-2020 baseline daily normals and percentiles for April & May."""
    daily_csv = Path(config["data"]["daily_file"])
    if not daily_csv.exists():
        raise FileNotFoundError(f"Daily dataset not found at {daily_csv}. Run daily.py first.")

    df = pd.read_csv(daily_csv)
    df["date"] = pd.to_datetime(df["date"])

    b_start = config["climatology"]["baseline_start"]
    b_end = config["climatology"]["baseline_end"]
    window_days = config["climatology"]["window_days"]

    # Filter baseline period (1991 to 2020)
    baseline_df = df[(df["year"] >= b_start) & (df["year"] <= b_end)].copy()
    logger.info(
        f"Computing climatology over baseline {b_start}-{b_end} ({len(baseline_df)} days, window +/- {window_days} days)..."
    )

    # Reference year for day-of-year distance (non-leap year)
    ref_year = 2001

    results = []
    # Process all unique (month, day) pairs in the dataset (April 1 to May 31)
    unique_dates = df[["month", "day"]].drop_duplicates().sort_values(by=["month", "day"])

    for _, row in unique_dates.iterrows():
        m = int(row["month"])
        d = int(row["day"])
        target_ts = pd.Timestamp(year=ref_year, month=m, day=d)

        # Collect observations across baseline years within +/- window_days
        window_tmax = []
        window_tmin = []
        window_apparent = []
        window_rh = []

        for _, base_row in baseline_df.iterrows():
            bm = int(base_row["month"])
            bd = int(base_row["day"])
            base_ts = pd.Timestamp(year=ref_year, month=bm, day=bd)
            diff = abs((base_ts - target_ts).days)
            if diff <= window_days:
                window_tmax.append(base_row["tmax"])
                window_tmin.append(base_row["tmin"])
                window_apparent.append(base_row["apparent_tmax"])
                window_rh.append(base_row["rh_mean"])

        arr_tmax = np.array(window_tmax)
        arr_tmin = np.array(window_tmin)
        arr_apparent = np.array(window_apparent)
        arr_rh = np.array(window_rh)

        results.append({
            "month": m,
            "day": d,
            "day_of_year": target_ts.timetuple().tm_yday,
            "sample_count": len(arr_tmax),
            "normal_tmax": round(float(np.mean(arr_tmax)), 2),
            "std_tmax": round(float(np.std(arr_tmax, ddof=1)), 2),
            "p85_tmax": round(float(np.percentile(arr_tmax, 85)), 2),
            "p90_tmax": round(float(np.percentile(arr_tmax, 90)), 2),
            "normal_tmin": round(float(np.mean(arr_tmin)), 2),
            "normal_apparent_tmax": round(float(np.mean(arr_apparent)), 2),
            "normal_rh_mean": round(float(np.mean(arr_rh)), 2),
        })

    df_normals = pd.DataFrame(results)
    output_path = Path(config["data"]["processed_dir"]) / "climatology_normals.csv"
    df_normals.to_csv(output_path, index=False)
    logger.info(f"Saved WMO climatology normals to {output_path} ({len(df_normals)} calendar days).")

    return df_normals


def load_climatology_lookup(config: dict) -> dict:
    """Load climatology normals into a fast lookup dictionary keyed by (month, day)."""
    normals_path = Path(config["data"]["processed_dir"]) / "climatology_normals.csv"
    if not normals_path.exists():
        compute_wmo_climatology(config)
    df_normals = pd.read_csv(normals_path)
    lookup = {}
    for _, row in df_normals.iterrows():
        lookup[(int(row["month"]), int(row["day"]))] = row.to_dict()
    return lookup


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    compute_wmo_climatology(cfg)
