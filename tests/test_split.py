"""
Unit Test: Chronological Dataset Splitting.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).

Concept:
In time-series machine learning, random shuffling or random k-fold cross-validation
creates temporal leakage and pseudo-replication due to strong autocorrelation
between adjacent weather days. A strictly chronological split (Train 1981-2010,
Val 2011-2018, Test 2019-2026) is required for realistic evaluation of neural models
(Perceptron, ADALINE, Error Backpropagation ANN, and Neuro-Fuzzy hybrids).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.split import get_splits, load_config
from src.features import FEATURE_COLS


@pytest.fixture(scope="module")
def split_data():
    config = load_config("config.yaml")
    return get_splits(config=config, save_scaler=False)


def test_split_years_non_overlapping(split_data):
    """Assert no overlapping years between Train, Validation, and Test sets."""
    train_years = set(split_data["train_df"]["year"].unique())
    val_years = set(split_data["val_df"]["year"].unique())
    test_years = set(split_data["test_df"]["year"].unique())

    # Check pairwise non-overlap
    assert train_years.isdisjoint(val_years), "Train and Validation years must not overlap!"
    assert val_years.isdisjoint(test_years), "Validation and Test years must not overlap!"
    assert train_years.isdisjoint(test_years), "Train and Test years must not overlap!"


def test_split_year_boundaries(split_data):
    """Assert exact expected year ranges for Train (1981-2010), Val (2011-2018), Test (2019-2026)."""
    train_years = sorted(split_data["train_df"]["year"].unique())
    val_years = sorted(split_data["val_df"]["year"].unique())
    test_years = sorted(split_data["test_df"]["year"].unique())

    assert train_years == list(range(1981, 2011)), "Train split must be 1981-2010 (30 years)"
    assert val_years == list(range(2011, 2019)), "Validation split must be 2011-2018 (8 years)"
    assert test_years == list(range(2019, 2027)), "Test split must be 2019-2026 (8 years)"


def test_split_row_counts(split_data):
    """
    Assert exact row counts for May:
    Train: 30 years * 31 days = 930 rows
    Val: 8 years * 31 days = 248 rows
    Test: 8 years * 31 days = 248 rows
    Total: 1,426 rows
    """
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_val = split_data["X_val"]
    y_val = split_data["y_val"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    assert len(X_train) == 930, f"Expected 930 train samples, got {len(X_train)}"
    assert len(y_train) == 930
    assert len(X_val) == 248, f"Expected 248 val samples, got {len(X_val)}"
    assert len(y_val) == 248
    assert len(X_test) == 248, f"Expected 248 test samples, got {len(X_test)}"
    assert len(y_test) == 248

    total = len(X_train) + len(X_val) + len(X_test)
    assert total == 1426, f"Total samples across splits must be 1426, got {total}"


def test_split_feature_shapes_and_types(split_data):
    """Assert numpy shapes and data types match model expectations."""
    num_features = len(FEATURE_COLS)

    assert split_data["X_train"].shape == (930, num_features)
    assert split_data["X_val"].shape == (248, num_features)
    assert split_data["X_test"].shape == (248, num_features)

    assert split_data["X_train"].dtype in [np.float64, np.float32]
    assert split_data["y_train"].dtype in [np.int64, np.int32]
    assert set(np.unique(split_data["y_train"])).issubset({0, 1})
