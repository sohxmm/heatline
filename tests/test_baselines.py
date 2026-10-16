"""
Unit Tests for Soft Computing Baselines and Statistical Benchmarks.
Course: Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.sc_baselines import (
    McCullochPittsNeuron,
    PerceptronRuleClassifier,
    DeltaRuleADALINE,
)
from src.baselines import (
    PersistenceBaseline,
    fit_logistic_regression,
    evaluate_all_baselines,
)
from src.split import get_splits


def test_mcculloch_pitts_logic():
    """Verify biological excitatory threshold logic and inhibitory veto mechanism."""
    mp = McCullochPittsNeuron(
        tmax_threshold=42.0,
        anomaly_threshold=3.5,
        rh_min_threshold=25.0,
        wind_threshold=15.0,
        theta=2.0,
        inhibitory_weight=2.0,
    )
    mp.fit()

    # Case 1: 3 excitatory active, 0 inhibitory -> net = 3 >= 2 -> Fire (1)
    df_fire = pd.DataFrame([{
        "tmax_lag1": 43.5,
        "tmax_anomaly_lag1": 4.0,
        "rh_min_lag1": 18.0,
        "wind_mean_lag1": 10.0,
    }])
    assert mp.predict(df_fire)[0] == 1
    assert mp.compute_net_input(df_fire)[0] == 3.0

    # Case 2: 3 excitatory active, BUT strong inhibitory wind active (veto) -> net = 3 - 2 = 1 < 2 -> Quiescent (0)
    df_inhibited = pd.DataFrame([{
        "tmax_lag1": 43.5,
        "tmax_anomaly_lag1": 4.0,
        "rh_min_lag1": 18.0,
        "wind_mean_lag1": 18.0,  # >= 15.0 km/h triggers inhibitory synapse
    }])
    assert mp.predict(df_inhibited)[0] == 0
    assert mp.compute_net_input(df_inhibited)[0] == 1.0

    # Case 3: Only 1 excitatory active -> net = 1 < 2 -> Quiescent (0)
    df_cool = pd.DataFrame([{
        "tmax_lag1": 39.0,
        "tmax_anomaly_lag1": 0.5,
        "rh_min_lag1": 20.0,  # only rh_min meets condition
        "wind_mean_lag1": 10.0,
    }])
    assert mp.predict(df_cool)[0] == 0
    assert mp.compute_net_input(df_cool)[0] == 1.0

    # Check probabilities are well-behaved in [0, 1]
    probs = mp.predict_proba(df_fire)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)


def test_perceptron_classifier():
    """Verify Perceptron learning rule with bipolar targets on scaled split features."""
    splits = get_splits()
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]

    clf = PerceptronRuleClassifier(eta=0.01, max_epochs=20, random_state=42)
    clf.fit(X_train, y_train)

    assert clf.w_ is not None
    assert clf.w_.shape[0] == X_train.shape[1]
    assert len(clf.errors_per_epoch_) == 20

    val_preds = clf.predict(X_val)
    val_probs = clf.predict_proba(X_val)

    assert val_preds.shape == (len(X_val),)
    assert val_probs.shape == (len(X_val), 2)
    assert set(np.unique(val_preds)).issubset({0, 1})
    assert np.all((val_probs >= 0.0) & (val_probs <= 1.0))


def test_adaline_delta_rule():
    """Verify ADALINE continuous Widrow-Hoff LMS gradient descent decreases loss."""
    splits = get_splits()
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]

    ada = DeltaRuleADALINE(eta=0.001, max_epochs=25, random_state=42)
    ada.fit(X_train, y_train)

    assert ada.w_ is not None
    assert ada.w_.shape[0] == X_train.shape[1]
    assert len(ada.mse_history_) == 25
    # MSE should decrease over training
    assert ada.mse_history_[-1] < ada.mse_history_[0]

    val_preds = ada.predict(X_val)
    val_probs = ada.predict_proba(X_val)

    assert val_preds.shape == (len(X_val),)
    assert val_probs.shape == (len(X_val), 2)
    assert set(np.unique(val_preds)).issubset({0, 1})


def test_persistence_baseline():
    """Verify atmospheric persistence rule based on hot_streak_len."""
    persistence = PersistenceBaseline()
    df = pd.DataFrame({
        "hot_streak_len": [0, 1, 3, 0, 2],
    })
    preds = persistence.predict(df)
    assert np.array_equal(preds, np.array([0, 1, 1, 0, 1]))

    probs = persistence.predict_proba(df)
    assert probs.shape == (5, 2)
    assert np.array_equal(probs[:, 1], np.array([0.0, 1.0, 1.0, 0.0, 1.0]))


def test_baseline_comparison_table():
    """Verify the generated baseline comparison table structure and metrics validity."""
    table_path = Path("results/tables/baseline_comparison.csv")
    assert table_path.exists(), "baseline_comparison.csv was not generated."

    df = pd.read_csv(table_path)
    expected_cols = ["Model", "Split", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column '{col}' in comparison table."

    # Check both splits are present
    assert set(df["Split"].unique()) == {"Validation", "Test"}

    # Check that all 5 models are present
    expected_models = {
        "Persistence",
        "McCulloch-Pitts (Exp 1)",
        "Perceptron Rule (Exp 4)",
        "ADALINE Delta Rule (Exp 5)",
        "Logistic Regression (Balanced)",
        "Random Forest (Balanced)",
    }
    assert set(df["Model"].unique()) == expected_models

    # Metrics should be valid probabilities/percentages in [0, 1]
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    for m in metric_cols:
        assert (df[m] >= 0.0).all() and (df[m] <= 1.0).all()
