"""
Unit Test: Same-Day Data Leakage Prevention.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).

Concept:
In time-series neural network modeling (Error Backpropagation ANN, Perceptron,
ADALINE), any reliance on future or same-day observations (Day T) violates
causality and causes catastrophic model failure in operational deployment.
Features used to predict whether Day T will be a heatwave MUST be computable
strictly at Day T-1 (end of previous day) plus deterministic calendar climatology.

Test Strategy:
1. Load features.csv and daily.csv.
2. Select a representative sample of May days across Train, Val, and Test splits.
3. For each test day T, perturb ALL weather observation columns in daily.csv by +50.0.
4. Recompute the engineered features for day T.
5. Assert that the resulting feature vector is 100% BITWISE IDENTICAL to the original,
   proving zero same-day data leakage.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features import compute_features, FEATURE_COLS

WEATHER_COLUMNS = [
    "tmax",
    "tmin",
    "tmean",
    "rh_mean",
    "rh_min",
    "dewpoint_mean",
    "apparent_tmax",
    "wind_mean",
    "pressure_mean",
    "solar_sum",
    "soil_moisture_mean",
    "heatwave_day",
    "imd_reanalysis_plains",
    "climatology_p85_42c",
    "imd_strict_raw",
]


@pytest.fixture(scope="module")
def datasets():
    daily_path = Path("data/processed/daily.csv")
    features_path = Path("data/processed/features.csv")
    assert daily_path.exists(), f"Missing {daily_path}"
    assert features_path.exists(), f"Missing {features_path}"

    df_daily = pd.read_csv(daily_path)
    df_features = pd.read_csv(features_path)
    return df_daily, df_features


def test_no_same_day_leakage_on_may_sample(datasets):
    """
    Perturbing weather columns on Day T by +50.0 must NOT alter any feature for Day T.
    """
    df_daily, df_features = datasets

    # Diverse sample: early May, mid May, late May across Train, Val, and Test eras
    sample_dates = [
        "1981-05-01",  # First day of earliest train year
        "1989-05-04",  # Early train
        "1998-05-15",  # Mid train
        "2010-05-31",  # Last day of train split
        "2011-05-01",  # First day of validation split
        "2015-05-18",  # Mid validation
        "2018-05-31",  # Last day of validation split
        "2019-05-01",  # First day of test split
        "2023-05-15",  # Test era
        "2026-05-31",  # Final day of test split
    ]

    for sdate in sample_dates:
        # Clone daily table
        df_perturbed = df_daily.copy()
        matching_idx = df_perturbed[df_perturbed["date"] == sdate].index

        assert len(matching_idx) == 1, f"Date {sdate} not found uniquely in daily.csv"
        row_idx = matching_idx[0]

        # Drastically perturb all weather observations on day T
        df_perturbed.loc[row_idx, WEATHER_COLUMNS] += 50.0

        # Recompute feature table
        df_recomputed = compute_features(df_perturbed)

        # Extract Day T feature vectors
        orig_row = df_features[df_features["date"] == sdate][FEATURE_COLS].to_numpy().flatten()
        pert_row = df_recomputed[df_recomputed["date"] == sdate][FEATURE_COLS].to_numpy().flatten()

        # Bitwise identical assertion
        assert np.array_equal(orig_row, pert_row), (
            f"Leakage detected on {sdate}! Feature vectors differ after day T weather perturbation."
        )


def test_features_row_count_and_columns(datasets):
    """Assert features.csv has exactly 1,426 rows (31 May days * 46 years) and all columns."""
    _, df_features = datasets
    assert len(df_features) == 1426, f"Expected 1426 May rows, found {len(df_features)}"
    assert (df_features["month"] == 5).all(), "features.csv should strictly contain May records"

    for col in FEATURE_COLS:
        assert col in df_features.columns, f"Missing feature: {col}"
        assert not df_features[col].isnull().any(), f"Found nulls in feature column: {col}"
