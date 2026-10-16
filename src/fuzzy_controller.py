"""
Mamdani Fuzzy Inference System for Heat Wave Risk Prediction.
KJSCE Soft Computing Lab — Module 4 (CO4).

Soft Computing Concepts:
- Linguistic Variables & Antecedents:
  1. tmax_lag1: Normal, Hot, Extreme
  2. tmax_anomaly_lag1: Normal, Elevated, Severe
  3. rh_min_lag1: Dry, Moderate, Humid
- Consequent: Heatwave Risk Score in [0, 100] (Low, Watch, Warning)
- Operators:
  - Fuzzification: Triangular and Trapezoidal Membership Functions
  - Fuzzy Conjunction (AND): Minimum operator (T-norm: min(mu_A, mu_B))
  - Implication: Mamdani Minimum Clipping
  - Rule Aggregation: Maximum operator (S-norm: max_k(mu_k))
  - Defuzzification: Centroid / Center of Gravity (COG) defuzzification
"""

import logging
from pathlib import Path
import sys
from typing import Dict, Any, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Membership Functions
# ---------------------------------------------------------------------------

def trapmf(x: Union[float, np.ndarray], abcd: Tuple[float, float, float, float]) -> Union[float, np.ndarray]:
    """Trapezoidal membership function with shoulder support."""
    a, b, c, d = abcd
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    # Left slope
    if b > a:
        mask1 = (x >= a) & (x < b)
        y[mask1] = (x[mask1] - a) / (b - a)
    else:
        # Left shoulder (a == -inf or a == b)
        mask1 = x < b
        y[mask1] = 1.0

    # Plateau
    mask2 = (x >= b) & (x <= c)
    y[mask2] = 1.0

    # Right slope
    if d > c:
        mask3 = (x > c) & (x <= d)
        y[mask3] = (d - x[mask3]) / (d - c)
    else:
        # Right shoulder (d == inf or c == d)
        mask3 = x > c
        y[mask3] = 1.0

    return float(y) if y.ndim == 0 else y


def trimf(x: Union[float, np.ndarray], abc: Tuple[float, float, float]) -> Union[float, np.ndarray]:
    """Triangular membership function."""
    a, b, c = abc
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    mask1 = (x >= a) & (x < b)
    if b > a:
        y[mask1] = (x[mask1] - a) / (b - a)

    mask2 = (x >= b) & (x <= c)
    if c > b:
        y[mask2] = (c - x[mask2]) / (c - b)

    return float(y) if y.ndim == 0 else y


# ---------------------------------------------------------------------------
# Mamdani Fuzzy Controller
# ---------------------------------------------------------------------------

class MamdaniFuzzyController:
    """
    Expert Mamdani Fuzzy Logic Controller modeling Vidarbha regional thermodynamics.
    KJSCE Soft Computing Lab Experiment CO4.
    """

    def __init__(self, resolution: int = 101):
        self.resolution = resolution
        self.risk_universe = np.linspace(0.0, 100.0, resolution)

        # Output Consequent Sets on [0, 100]
        self.consequent_low = trapmf(self.risk_universe, (-10.0, -10.0, 15.0, 35.0))
        self.consequent_watch = trimf(self.risk_universe, (25.0, 50.0, 75.0))
        self.consequent_warning = trapmf(self.risk_universe, (65.0, 85.0, 110.0, 110.0))

    def evaluate_sample(self, tmax_lag1: float, tmax_anomaly_lag1: float, rh_min_lag1: float) -> Tuple[float, float]:
        """
        Evaluate fuzzy risk for a single day's atmospheric conditions.
        Returns: (centroid_risk_score in [0, 100], probability in [0, 1])
        """
        # 1. Fuzzification
        # Antecedent 1: tmax_lag1 (Nagpur plateau May temperature)
        mu_tmax_norm = trapmf(tmax_lag1, (-100.0, -100.0, 38.0, 40.5))
        mu_tmax_hot = trimf(tmax_lag1, (39.0, 41.5, 43.5))
        mu_tmax_ext = trapmf(tmax_lag1, (42.0, 44.0, 100.0, 100.0))

        # Antecedent 2: tmax_anomaly_lag1 (Climatological departure)
        mu_anom_norm = trapmf(tmax_anomaly_lag1, (-100.0, -100.0, 1.0, 2.0))
        mu_anom_elev = trimf(tmax_anomaly_lag1, (1.0, 2.5, 4.0))
        mu_anom_sev = trapmf(tmax_anomaly_lag1, (3.0, 4.5, 100.0, 100.0))

        # Antecedent 3: rh_min_lag1 (Dry continental advection index)
        mu_rh_dry = trapmf(rh_min_lag1, (-100.0, -100.0, 12.0, 20.0))
        mu_rh_mod = trimf(rh_min_lag1, (15.0, 25.0, 38.0))
        mu_rh_hum = trapmf(rh_min_lag1, (30.0, 45.0, 200.0, 200.0))

        # 2. Rule Evaluation (Mamdani Min operator)
        rule_warning_strengths = [
            min(mu_tmax_ext, mu_anom_sev),                    # Rule 1
            min(mu_tmax_ext, mu_rh_dry),                      # Rule 2
            min(mu_tmax_ext, mu_rh_mod),                      # Rule 3
            min(mu_tmax_ext, mu_anom_elev),                   # Rule 4
            min(mu_tmax_hot, mu_anom_sev, mu_rh_dry),         # Rule 5
            min(mu_tmax_hot, mu_anom_sev, mu_rh_mod),         # Rule 6
        ]

        rule_watch_strengths = [
            min(mu_tmax_hot, mu_anom_elev, mu_rh_dry),        # Rule 7
            min(mu_tmax_hot, mu_anom_elev, mu_rh_mod),        # Rule 8
            min(mu_tmax_ext, mu_rh_hum),                      # Rule 9: Extreme heat dampened by humidity
            min(mu_tmax_hot, mu_anom_sev, mu_rh_hum),         # Rule 10
        ]

        rule_low_strengths = [
            mu_tmax_norm,                                     # Rule 11: Normal temp
            min(mu_tmax_hot, mu_anom_norm),                   # Rule 12: Hot but no departure
            min(mu_tmax_hot, mu_rh_hum),                      # Rule 13: Hot but high humidity
            min(mu_anom_norm, mu_rh_hum),                     # Rule 14: Cool and humid
        ]

        # 3. Rule Aggregation (Max S-norm)
        alpha_warning = max(rule_warning_strengths)
        alpha_watch = max(rule_watch_strengths)
        alpha_low = max(rule_low_strengths)

        # 4. Implication (Min clipping of consequent sets)
        clipped_warning = np.minimum(alpha_warning, self.consequent_warning)
        clipped_watch = np.minimum(alpha_watch, self.consequent_watch)
        clipped_low = np.minimum(alpha_low, self.consequent_low)

        aggregated_output = np.maximum(clipped_low, np.maximum(clipped_watch, clipped_warning))

        # 5. Defuzzification (Centroid method)
        total_area = np.sum(aggregated_output)
        if total_area > 1e-9:
            centroid_score = float(np.sum(self.risk_universe * aggregated_output) / total_area)
        else:
            centroid_score = 15.0  # Default to low risk

        prob = float(np.clip(centroid_score / 100.0, 0.0, 1.0))
        return centroid_score, prob

    def predict_proba(self, df_or_X: Union[pd.DataFrame, np.ndarray], feature_names: list = None) -> np.ndarray:
        """
        Generate continuous probability predictions for an entire dataset.
        Accepts either a DataFrame (with raw columns) or an unscaled/scaled array with feature_names.
        """
        if isinstance(df_or_X, pd.DataFrame):
            tmax_l = df_or_X["tmax_lag1"].values
            anom_l = df_or_X["tmax_anomaly_lag1"].values
            rh_l = df_or_X["rh_min_lag1"].values
        else:
            # Array form: require feature_names to extract correct columns
            if feature_names is None:
                raise ValueError("feature_names required when passing ndarray to MamdaniFuzzyController")
            idx_tmax = feature_names.index("tmax_lag1")
            idx_anom = feature_names.index("tmax_anomaly_lag1")
            idx_rh = feature_names.index("rh_min_lag1")
            
            tmax_l = df_or_X[:, idx_tmax]
            anom_l = df_or_X[:, idx_anom]
            rh_l = df_or_X[:, idx_rh]

        probs = np.zeros(len(tmax_l), dtype=float)
        for i in range(len(tmax_l)):
            _, p = self.evaluate_sample(float(tmax_l[i]), float(anom_l[i]), float(rh_l[i]))
            probs[i] = p

        return probs

    def predict(self, df_or_X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5, feature_names: list = None) -> np.ndarray:
        """Binary classification predictions based on decision threshold."""
        probs = self.predict_proba(df_or_X, feature_names=feature_names)
        return (probs >= threshold).astype(int)


def evaluate_fuzzy_controller(config: dict) -> Dict[str, Any]:
    """Evaluate Mamdani Fuzzy Controller on raw features across Train, Val, and Test splits."""
    features_csv = Path(config["data"]["features_file"])
    df = pd.read_csv(features_csv)

    val_df = df[(df["year"] >= config["splits"]["val"]["start_year"]) & (df["year"] <= config["splits"]["val"]["end_year"])]
    test_df = df[(df["year"] >= config["splits"]["test"]["start_year"]) & (df["year"] <= config["splits"]["test"]["end_year"])]

    controller = MamdaniFuzzyController()

    val_probs = controller.predict_proba(val_df)
    test_probs = controller.predict_proba(test_df)

    y_val = val_df["heatwave_day"].values
    y_test = test_df["heatwave_day"].values

    # Validation metrics
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    val_roc_auc = float(roc_auc_score(y_val, val_probs))
    val_f1 = float(f1_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0))
    val_prec = float(precision_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0))
    val_rec = float(recall_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0))

    # Test metrics
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    test_roc_auc = float(roc_auc_score(y_test, test_probs))
    test_f1 = float(f1_score(y_test, (test_probs >= 0.5).astype(int), zero_division=0))
    test_prec = float(precision_score(y_test, (test_probs >= 0.5).astype(int), zero_division=0))
    test_rec = float(recall_score(y_test, (test_probs >= 0.5).astype(int), zero_division=0))

    logger.info("--- Mamdani Fuzzy Logic Controller Performance (Threshold 0.50) ---")
    logger.info(f"Validation: PR-AUC={val_pr_auc:.4f}, ROC-AUC={val_roc_auc:.4f}, F1={val_f1:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}")
    logger.info(f"Test:       PR-AUC={test_pr_auc:.4f}, ROC-AUC={test_roc_auc:.4f}, F1={test_f1:.4f}, Prec={test_prec:.4f}, Rec={test_rec:.4f}")

    return {
        "controller": controller,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "val_metrics": {"pr_auc": val_pr_auc, "roc_auc": val_roc_auc, "f1": val_f1, "prec": val_prec, "rec": val_rec},
        "test_metrics": {"pr_auc": test_pr_auc, "roc_auc": test_roc_auc, "f1": test_f1, "prec": test_prec, "rec": test_rec},
    }


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    evaluate_fuzzy_controller(cfg)
