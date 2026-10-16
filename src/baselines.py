"""
Baseline Classifiers and Statistical Benchmarks for Nagpur Heat Wave Prediction.
Course: Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).
Location: Nagpur Urban Agglomeration (Vidarbha, Central India Core Heatwave Zone).

This module implements statistical and physical reference baselines:
1. PersistenceBaseline:
   Atmospheric inertia benchmark predicting that Day T's heatwave status equals Day T-1's
   observed state. Derived directly from hot_streak_len > 0 on raw feature data.
2. Class-Weighted Logistic Regression:
   Standard statistical linear benchmark utilizing inverse class frequency weighting
   (class_weight='balanced') to address May heatwave imbalance (~24.3% in train).

Evaluation Framework:
Computes and formats a comparative benchmark across all 5 initial models:
- Persistence Baseline
- McCulloch-Pitts Neuron (Exp 1)
- Single-Layer Perceptron (Exp 4)
- Continuous Delta Rule ADALINE (Exp 5)
- Class-Weighted Logistic Regression
Results are exported to results/tables/baseline_comparison.csv.
"""

import logging
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional, Union, Tuple

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import yaml

from src.sc_baselines import (
    McCullochPittsNeuron,
    PerceptronRuleClassifier,
    DeltaRuleADALINE,
)
from src.split import get_splits, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PersistenceBaseline(BaseEstimator, ClassifierMixin):
    """
    Atmospheric Persistence Baseline.

    Physical Rationale:
    Synoptic heatwaves in Central India are governed by semi-stationary mid-tropospheric
    anticyclonic ridges and strong surface sensible heat flux over dry landmasses.
    Due to atmospheric thermal inertia, weather conditions exhibit strong autocorrelation:
        Y_hat(T) = Y(T - 1)

    In our feature pipeline, `hot_streak_len` for day T denotes the consecutive run of
    heatwave days ending at day T-1. Therefore:
        If hot_streak_len > 0: Y(T - 1) == 1  -->  Predict 1
        If hot_streak_len == 0: Y(T - 1) == 0 -->  Predict 0

    IMPORTANT: Operates on unscaled raw feature data (e.g. val_df['hot_streak_len']
    or features.csv), since standardized features have non-zero means.
    """

    def __init__(self):
        self.classes_ = np.array([0, 1], dtype=np.int64)

    def fit(self, X: Optional[Any] = None, y: Optional[Any] = None) -> "PersistenceBaseline":
        """No fitting required for persistence rule."""
        return self

    def _extract_streak(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """Extract hot_streak_len array from DataFrame, Series, or 2D array."""
        if isinstance(X, pd.DataFrame):
            if "hot_streak_len" in X.columns:
                return X["hot_streak_len"].to_numpy(dtype=np.float64)
            elif "persistence_pred" in X.columns:
                return X["persistence_pred"].to_numpy(dtype=np.float64)
            raise KeyError("Neither 'hot_streak_len' nor 'persistence_pred' found in DataFrame columns.")

        if isinstance(X, pd.Series):
            return X.to_numpy(dtype=np.float64)

        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X.astype(np.float64)
            elif X.ndim == 2:
                # Column index 22 corresponds to hot_streak_len in FEATURE_COLS
                if X.shape[1] >= 23:
                    return X[:, 22].astype(np.float64)
                elif X.shape[1] == 1:
                    return X[:, 0].astype(np.float64)
                raise ValueError(f"Cannot identify hot_streak_len column in array of shape {X.shape}")

        raise TypeError(f"Unsupported input type for PersistenceBaseline: {type(X)}")

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """
        Predict binary heatwave state using previous day's state:
        Predict 1 if hot_streak_len > 0 else 0.
        """
        streak = self._extract_streak(X)
        return (streak > 0.0).astype(np.int64)

    def predict_proba(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """
        Return pseudo-probabilities corresponding to binary persistence state:
        [1.0, 0.0] if previous day was non-heatwave, [0.0, 1.0] if previous day was heatwave.
        """
        preds = self.predict(X).astype(np.float64)
        return np.column_stack([1.0 - preds, preds])


def fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
) -> LogisticRegression:
    """
    Train a Class-Weighted Logistic Regression model on standardized training features.

    Soft Computing & Statistical Connection:
    - Logistic Regression is mathematically isomorphic to a single-layer neural network
      with a sigmoid output activation, optimized via L2-regularized maximum likelihood
      estimation (cross-entropy / log-loss) using L-BFGS or coordinate descent.
    - Specifying `class_weight='balanced'` assigns sample weights inversely proportional
      to class frequencies:
          w_k = N / (2 * N_k)
      This offsets the ~3.1:1 majority-to-minority class imbalance in the 1981-2010 training set.

    Parameters
    ----------
    X_train : np.ndarray
        Standardized training features.
    y_train : np.ndarray
        Binary training labels {0, 1}.
    random_state : int, default=42
        Reproducibility seed.
    max_iter : int, default=1000
        Maximum solver convergence iterations.
    C : float, default=1.0
        Inverse regularization parameter.

    Returns
    -------
    LogisticRegression
        Fitted scikit-learn LogisticRegression model.
    """
    logger.info(
        "Fitting Class-Weighted Logistic Regression: N=%d, D=%d, C=%.2f, class_weight='balanced'",
        X_train.shape[0], X_train.shape[1], C
    )
    lr = LogisticRegression(
        class_weight="balanced",
        random_state=random_state,
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    return lr


def fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 6,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Fit Class-Weighted Random Forest Classifier on standardized training features.

    Parameters
    ----------
    X_train : np.ndarray
        Standardized training input array.
    y_train : np.ndarray
        Ground-truth binary training labels.
    n_estimators : int, default=100
        Number of decision trees in the ensemble forest.
    max_depth : int, default=6
        Maximum tree depth constraint to prevent boundary-layer overfitting.
    random_state : int, default=42
        Seed for reproducibility.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_model_on_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute standard evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels {0, 1}.
    y_pred : np.ndarray
        Binary class predictions at decision threshold {0, 1}.
    y_prob : np.ndarray, optional
        Continuous probability or decision scores for ranking metrics (ROC-AUC / PR-AUC).
        If omitted, binary predictions are used.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    score_for_auc = y_prob if y_prob is not None else y_pred
    roc = roc_auc_score(y_true, score_for_auc)
    pr = average_precision_score(y_true, score_for_auc)

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "ROC_AUC": float(roc),
        "PR_AUC": float(pr),
    }


def evaluate_all_baselines(
    config: Optional[dict] = None,
    save_csv: bool = True,
    out_path: str = "results/tables/baseline_comparison.csv",
) -> pd.DataFrame:
    """
    Train and evaluate all soft computing and statistical baseline models on Validation and Test sets.

    Models Evaluated:
    1. Persistence Baseline: Previous day state (hot_streak_len > 0)
    2. McCulloch-Pitts Neuron (Exp 1): Expert biological threshold logic on raw features
    3. Perceptron Rule Classifier (Exp 4): Discrete error step update with bipolar targets
    4. ADALINE Delta Rule (Exp 5): Continuous Widrow-Hoff LMS gradient descent with sigmoid
    5. Class-Weighted Logistic Regression: Balanced L2-regularized logistic regression

    Returns
    -------
    pd.DataFrame
        Formatted comparison table containing all evaluation metrics across both splits.
    """
    if config is None:
        config = load_config()

    splits = get_splits(config)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    val_df = splits["val_df"]
    test_df = splits["test_df"]

    logger.info("Initializing and training baseline classifiers...")

    # 1. Persistence Baseline
    persistence = PersistenceBaseline()
    persistence_val_pred = persistence.predict(val_df)
    persistence_test_pred = persistence.predict(test_df)

    # 2. McCulloch-Pitts Neuron (Exp 1) - Raw Features
    mp = McCullochPittsNeuron(
        tmax_threshold=42.0,
        anomaly_threshold=3.5,
        rh_min_threshold=25.0,
        wind_threshold=15.0,
        theta=2.0,
        inhibitory_weight=2.0,
    )
    mp.fit()
    mp_val_pred = mp.predict(val_df)
    mp_val_prob = mp.predict_proba(val_df)[:, 1]
    mp_test_pred = mp.predict(test_df)
    mp_test_prob = mp.predict_proba(test_df)[:, 1]

    # 3. Single-Layer Perceptron (Exp 4) - Scaled Features
    perceptron = PerceptronRuleClassifier(eta=0.01, max_epochs=100, random_state=42)
    perceptron.fit(X_train, y_train)
    perc_val_pred = perceptron.predict(X_val)
    perc_val_prob = perceptron.predict_proba(X_val)[:, 1]
    perc_test_pred = perceptron.predict(X_test)
    perc_test_prob = perceptron.predict_proba(X_test)[:, 1]

    # 4. Continuous Delta Rule ADALINE (Exp 5) - Scaled Features
    adaline = DeltaRuleADALINE(eta=0.001, max_epochs=200, random_state=42)
    adaline.fit(X_train, y_train)
    ada_val_pred = adaline.predict(X_val)
    ada_val_prob = adaline.predict_proba(X_val)[:, 1]
    ada_test_pred = adaline.predict(X_test)
    ada_test_prob = adaline.predict_proba(X_test)[:, 1]

    # 5. Class-Weighted Logistic Regression - Scaled Features
    lr = fit_logistic_regression(X_train, y_train, random_state=42)
    lr_val_prob = lr.predict_proba(X_val)[:, 1]
    lr_val_pred = (lr_val_prob >= 0.5).astype(np.int64)
    lr_test_prob = lr.predict_proba(X_test)[:, 1]
    lr_test_pred = (lr_test_prob >= 0.5).astype(np.int64)

    # 6. Class-Weighted Random Forest - Scaled Features
    rf = fit_random_forest(X_train, y_train, random_state=42)
    rf_val_prob = rf.predict_proba(X_val)[:, 1]
    rf_val_pred = (rf_val_prob >= 0.5).astype(np.int64)
    rf_test_prob = rf.predict_proba(X_test)[:, 1]
    rf_test_pred = (rf_test_prob >= 0.5).astype(np.int64)

    # Compile evaluation records
    models_config = [
        (
            "Persistence",
            persistence_val_pred,
            None,
            persistence_test_pred,
            None,
        ),
        (
            "McCulloch-Pitts (Exp 1)",
            mp_val_pred,
            mp_val_prob,
            mp_test_pred,
            mp_test_prob,
        ),
        (
            "Perceptron Rule (Exp 4)",
            perc_val_pred,
            perc_val_prob,
            perc_test_pred,
            perc_test_prob,
        ),
        (
            "ADALINE Delta Rule (Exp 5)",
            ada_val_pred,
            ada_val_prob,
            ada_test_pred,
            ada_test_prob,
        ),
        (
            "Logistic Regression (Balanced)",
            lr_val_pred,
            lr_val_prob,
            lr_test_pred,
            lr_test_prob,
        ),
        (
            "Random Forest (Balanced)",
            rf_val_pred,
            rf_val_prob,
            rf_test_pred,
            rf_test_prob,
        ),
    ]

    records = []
    for model_name, v_pred, v_prob, t_pred, t_prob in models_config:
        # Validation Split Evaluation
        val_metrics = evaluate_model_on_split(y_val, v_pred, v_prob)
        records.append({
            "Model": model_name,
            "Split": "Validation",
            **{k: round(v, 4) for k, v in val_metrics.items()},
        })

        # Test Split Evaluation
        test_metrics = evaluate_model_on_split(y_test, t_pred, t_prob)
        records.append({
            "Model": model_name,
            "Split": "Test",
            **{k: round(v, 4) for k, v in test_metrics.items()},
        })

    df_results = pd.DataFrame(records)

    # Save to disk
    if save_csv:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(dest, index=False)
        logger.info("Saved baseline comparison table to: %s", out_path)

    return df_results


if __name__ == "__main__":
    print("=" * 80)
    print(" Nagpur Heat Wave Prediction: Baseline Evaluation (Soft Computing Lab)")
    print("=" * 80)
    results_df = evaluate_all_baselines()
    print("\nBaseline Comparison Summary Table:")
    print(results_df.to_string(index=False))
