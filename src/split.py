"""
Chronological Splitting and Train-Only Standardization Module.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).

Soft Computing Connection:
- In Error Backpropagation Neural Networks (EBPA, Exp 2, 6, 7), unscaled inputs
  with varying dynamic ranges cause gradient imbalance: features with large
  variances dominate weight updates (Delta rule: delta_w = eta * delta * x),
  driving sigmoid/tanh activations into flat saturation regions where f'(x) -> 0.
- In Single-Layer Perceptron (Exp 4) and ADALINE Delta Rule (Exp 5), standardizing
  features to zero mean and unit variance conditions the Hessian error surface,
  preventing oscillation and ensuring smooth gradient descent.
- Temporal Integrity: We strictly use chronological splitting (Train 1981-2010,
  Validation 2011-2018, Test 2019-2026). Random splits (e.g. k-fold or random shuffle)
  would cause autocorrelated weather patterns to leak between sets, violating
  causality.
- Scaler Leakage Prevention: StandardScaler is fit SOLELY on the Training split
  (1981-2010). Validation and Test splits are transformed using training statistics
  (mu_train, sigma_train).
"""

import json
import logging
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List, Tuple

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml

from src.features import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project YAML configuration."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_scaler_params(
    scaler: StandardScaler,
    feature_names: List[str],
    out_path: str = "models/scaler.json",
    train_years: Tuple[int, int] = (1981, 2010),
) -> None:
    """
    Export fitted StandardScaler parameters (mean, scale, variance) to JSON.
    Enables pure-NumPy inference in deployment without scikit-learn dependency.
    """
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "feature_names": list(feature_names),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": int(scaler.n_samples_seen_),
        "train_years": list(train_years),
    }

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    logger.info("Saved fitted scaler parameters to: %s", out_path)


def get_splits(
    config: Optional[dict] = None,
    features_path: Optional[str] = None,
    target_col: Optional[str] = None,
    scale: bool = True,
    save_scaler: bool = True,
    scaler_out_path: str = "models/scaler.json",
) -> Dict[str, Any]:
    """
    Split the May dataset chronologically into Train, Validation, and Test partitions.
    Fit a StandardScaler strictly on the Training partition, and transform all partitions.

    Splits:
    - Train: 1981–2010 (30 years, 930 May days, ~24.3% positive rate)
    - Validation: 2011–2018 (8 years, 248 May days, ~35.5% positive rate)
    - Test: 2019–2026 (8 years, 248 May days, ~14.9% positive rate)

    Parameters
    ----------
    config : dict, optional
        Project configuration dictionary.
    features_path : str, optional
        Path to features.csv. Defaults to config['data']['features_file'].
    target_col : str, optional
        Target label column name. Defaults to config['labels']['primary'].
    scale : bool
        Whether to standardize features using StandardScaler fit on Train.
    save_scaler : bool
        Whether to save fitted scaler parameters to models/scaler.json.
    scaler_out_path : str
        Output path for exported scaler parameters JSON.

    Returns
    -------
    dict
        Contains:
        - X_train, y_train: Training inputs (scaled) and binary targets
        - X_val, y_val: Validation inputs (scaled) and binary targets
        - X_test, y_test: Test inputs (scaled) and binary targets
        - scaler: Fitted sklearn StandardScaler instance
        - feature_names: List of feature column names
        - train_df, val_df, test_df: Unscaled DataFrames for reference
    """
    if config is None:
        config = load_config()

    if features_path is None:
        features_path = config["data"].get("features_file", "data/processed/features.csv")

    if target_col is None:
        target_col = config.get("labels", {}).get("primary", "imd_reanalysis_plains")

    p = Path(features_path)
    if not p.exists():
        raise FileNotFoundError(f"Features file not found at: {features_path}. Run features.py first.")

    df = pd.read_csv(p)

    # Validate required columns
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise KeyError(f"Required feature column missing from {features_path}: {col}")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' missing from {features_path}")

    # Extract split year boundaries from config
    train_cfg = config["splits"]["train"]
    val_cfg = config["splits"]["val"]
    test_cfg = config["splits"]["test"]

    train_y0, train_y1 = train_cfg["start_year"], train_cfg["end_year"]
    val_y0, val_y1 = val_cfg["start_year"], val_cfg["end_year"]
    test_y0, test_y1 = test_cfg["start_year"], test_cfg["end_year"]

    # Filter chronologically
    train_df = df[(df["year"] >= train_y0) & (df["year"] <= train_y1)].copy().reset_index(drop=True)
    val_df = df[(df["year"] >= val_y0) & (df["year"] <= val_y1)].copy().reset_index(drop=True)
    test_df = df[(df["year"] >= test_y0) & (df["year"] <= test_y1)].copy().reset_index(drop=True)

    logger.info(
        "Dataset Partition Summary: Train=%d (%d-%d), Val=%d (%d-%d), Test=%d (%d-%d)",
        len(train_df), train_y0, train_y1,
        len(val_df), val_y0, val_y1,
        len(test_df), test_y0, test_y1,
    )

    # Extract raw numpy arrays
    X_train_raw = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train = train_df[target_col].to_numpy(dtype=np.int64)

    X_val_raw = val_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_val = val_df[target_col].to_numpy(dtype=np.int64)

    X_test_raw = test_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_test = test_df[target_col].to_numpy(dtype=np.int64)

    # Fit scaler strictly on Train split
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    if scale:
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train = X_train_raw.copy()
        X_val = X_val_raw.copy()
        X_test = X_test_raw.copy()

    if save_scaler:
        save_scaler_params(
            scaler=scaler,
            feature_names=FEATURE_COLS,
            out_path=scaler_out_path,
            train_years=(train_y0, train_y1),
        )

    # Log class distribution shift
    logger.info("Split Positive Rates (%s):", target_col)
    logger.info("  Train (1981-2010): %d/%d (%.1f%%)", y_train.sum(), len(y_train), 100.0 * y_train.mean())
    logger.info("  Val   (2011-2018): %d/%d (%.1f%%)", y_val.sum(), len(y_val), 100.0 * y_val.mean())
    logger.info("  Test  (2019-2026): %d/%d (%.1f%%)", y_test.sum(), len(y_test), 100.0 * y_test.mean())

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
    }


if __name__ == "__main__":
    splits = get_splits()
    print("Splits successfully computed and verified.")
