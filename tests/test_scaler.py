"""
Unit Test: Train-Only StandardScaler Integrity and Non-Leakage.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).

Concept:
In Error Backpropagation Neural Networks (EBPA, Exp 2, 6, 7) and Perceptrons (Exp 4),
feature scaling (standardization to zero mean and unit variance) is essential to:
1. Prevent sigmoid activation function saturation where f'(z) = f(z)(1-f(z)) -> 0.
2. Condition the gradient error surface for stable Delta Rule weight updates.

StandardScaler Non-Leakage Rule:
Mean (mu) and standard deviation (sigma) MUST be calculated solely on the
historical training partition (1981-2010). Validation (2011-2018) and Test
(2019-2026) must be transformed using mu_train and sigma_train. Fitting the
scaler on the combined dataset leaks future distribution moments.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.split import get_splits, load_config
from src.features import FEATURE_COLS


@pytest.fixture(scope="module")
def split_bundle():
    config = load_config("config.yaml")
    return get_splits(config=config, save_scaler=True, scaler_out_path="models/scaler.json")


def test_scaler_fit_strictly_on_train(split_bundle):
    """
    Assert scaler statistics (mean_ and var_) match independent fit on Train split ONLY.
    """
    scaler = split_bundle["scaler"]
    X_train_raw = split_bundle["X_train_raw"]
    X_val_raw = split_bundle["X_val_raw"]
    X_test_raw = split_bundle["X_test_raw"]

    # Fit an independent benchmark scaler on training split
    benchmark_scaler = StandardScaler()
    benchmark_scaler.fit(X_train_raw)

    assert np.allclose(scaler.mean_, benchmark_scaler.mean_, atol=1e-10), (
        "Scaler mean_ does not match training set mean!"
    )
    assert np.allclose(scaler.scale_, benchmark_scaler.scale_, atol=1e-10), (
        "Scaler scale_ (std) does not match training set std!"
    )
    assert scaler.n_samples_seen_ == 930, (
        f"Scaler was fit on {scaler.n_samples_seen_} samples instead of Train count (930)!"
    )

    # Verify that fitting on all data produces distinctly different statistics
    all_raw = np.vstack([X_train_raw, X_val_raw, X_test_raw])
    leak_scaler = StandardScaler().fit(all_raw)
    assert not np.allclose(scaler.mean_, leak_scaler.mean_, atol=1e-4), (
        "Suspicious: Scaler matches all-data mean! Check for leakage."
    )


def test_val_and_test_transformed_with_train_statistics(split_bundle):
    """
    Assert X_val and X_test are transformed using (X_raw - mu_train) / sigma_train.
    """
    scaler = split_bundle["scaler"]
    X_train = split_bundle["X_train"]
    X_val = split_bundle["X_val"]
    X_test = split_bundle["X_test"]

    X_val_raw = split_bundle["X_val_raw"]
    X_test_raw = split_bundle["X_test_raw"]

    # Manual transformation using training parameters
    expected_val = (X_val_raw - scaler.mean_) / scaler.scale_
    expected_test = (X_test_raw - scaler.mean_) / scaler.scale_

    assert np.allclose(X_val, expected_val, atol=1e-10), (
        "X_val was not properly transformed with Train mean/scale!"
    )
    assert np.allclose(X_test, expected_test, atol=1e-10), (
        "X_test was not properly transformed with Train mean/scale!"
    )

    # Train standardized features must have zero mean and unit variance
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-10), "Train features mean is not zero!"
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-10), "Train features std is not one!"

    # Validation and Test should NOT have exactly zero mean or unit variance
    # due to observed climate distribution shift
    assert not np.allclose(X_val.mean(axis=0), 0.0, atol=1e-2), (
        "X_val mean should reflect distribution shift relative to 1981-2010"
    )
    assert not np.allclose(X_test.mean(axis=0), 0.0, atol=1e-2), (
        "X_test mean should reflect distribution shift relative to 1981-2010"
    )


def test_exported_scaler_json():
    """Verify that models/scaler.json contains valid parameters matching scaler object."""
    scaler_file = Path("models/scaler.json")
    assert scaler_file.exists(), f"Scaler export JSON missing at: {scaler_file}"

    with open(scaler_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    assert "mean" in params
    assert "scale" in params
    assert "feature_names" in params
    assert len(params["feature_names"]) == len(FEATURE_COLS)
    assert len(params["mean"]) == len(FEATURE_COLS)
    assert len(params["scale"]) == len(FEATURE_COLS)
    assert params["n_samples_seen"] == 930
    assert params["train_years"] == [1981, 2010]
