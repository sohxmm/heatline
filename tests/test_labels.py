"""
Unit Test: Meteorological Heat Wave Label Definitions and Formulations.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).

Concept:
In binary classification with soft computing models (McCulloch-Pitts, Perceptrons,
Error Backpropagation ANN), the ground truth label definition establishes the
decision boundary separating positive (heatwave) and negative (normal) days.

Ground Truth Rules for Nagpur (Vidarbha Interior Plains):
1. `imd_reanalysis_plains` (Candidate A & Primary Target):
   tmax >= 43.5°C OR (tmax >= 40.0°C AND anomaly >= 3.5°C)
2. `climatology_p85_42c` (Candidate B):
   tmax >= max(42.0°C, P85_tmax_1991_2020)
3. `imd_strict_raw` (Candidate Ref):
   tmax >= 45.0°C OR (tmax >= 40.0°C AND anomaly >= 4.5°C)

Where anomaly = round(tmax - normal_tmax, 2).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def dataset():
    daily_path = Path("data/processed/daily.csv")
    features_path = Path("data/processed/features.csv")
    assert daily_path.exists(), f"Missing {daily_path}"
    assert features_path.exists(), f"Missing {features_path}"

    df_daily = pd.read_csv(daily_path)
    df_features = pd.read_csv(features_path)
    return df_daily, df_features


def test_imd_reanalysis_plains_formula(dataset):
    """
    Verify imd_reanalysis_plains definition:
    tmax >= 43.5 OR (tmax >= 40.0 AND anomaly >= 3.5)
    """
    df_daily, df_features = dataset
    anomaly = np.round(df_daily["tmax"] - df_daily["normal_tmax"], 2)
    expected = (
        (df_daily["tmax"] >= 43.5)
        | ((df_daily["tmax"] >= 40.0) & (anomaly >= 3.5))
    ).astype(int)

    assert (df_daily["imd_reanalysis_plains"] == expected).all(), (
        "imd_reanalysis_plains in daily.csv does not match official formula!"
    )

    # Also assert equality in features.csv
    assert (df_features["heatwave_day"] == df_features["imd_reanalysis_plains"]).all(), (
        "Primary heatwave_day in features.csv must equal imd_reanalysis_plains!"
    )


def test_climatology_p85_42c_formula(dataset):
    """
    Verify climatology_p85_42c definition:
    tmax >= max(42.0, p85_tmax)
    """
    df_daily, _ = dataset
    p85_effective = np.maximum(42.0, df_daily["p85_tmax"])
    expected = (df_daily["tmax"] >= p85_effective).astype(int)

    assert (df_daily["climatology_p85_42c"] == expected).all(), (
        "climatology_p85_42c in daily.csv does not match official formula!"
    )


def test_imd_strict_raw_formula(dataset):
    """
    Verify imd_strict_raw definition:
    tmax >= 45.0 OR (tmax >= 40.0 AND anomaly >= 4.5)
    """
    df_daily, _ = dataset
    anomaly = np.round(df_daily["tmax"] - df_daily["normal_tmax"], 2)
    expected = (
        (df_daily["tmax"] >= 45.0)
        | ((df_daily["tmax"] >= 40.0) & (anomaly >= 4.5))
    ).astype(int)

    assert (df_daily["imd_strict_raw"] == expected).all(), (
        "imd_strict_raw in daily.csv does not match official formula!"
    )
