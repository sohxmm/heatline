# LEGACY/OPTIONAL: original experiment only. Use src.experiment and frontend/ for Heatline.
# Old weights, label definitions and reported metrics are not current assessed evidence.
"""
Deep Multilayer Perceptron (MLP) Ensemble Training Module.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).

Soft Computing Concepts:
- Architecture: 2-Hidden-Layer Feedforward Neural Network with ReLU activations,
  Batch Normalization, and Dropout regularization:
  Input(26) -> Linear(units1) -> BatchNorm1d -> ReLU -> Dropout(p) ->
  Linear(units2) -> ReLU -> Dropout(p) -> Linear(1) -> Sigmoid.
- Loss Function: Class-weighted Binary Cross-Entropy with Logits:
  pos_weight = N_neg / N_pos from Training split to penalize false negatives
  under class imbalance (~24.3% positive rate).
- Optimizer: AdamW with decoupled weight decay (L2 regularization).
- Ensemble Method: 10-seed bagging (seeds 42..51) to reduce model variance
  and provide well-calibrated posterior probability estimates.
"""

import json
import logging
from pathlib import Path
import random
import sys
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
import torch
import torch.nn as nn
import yaml

from src.split import get_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Seed all random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HeatwaveMLP(nn.Module):
    """
    2-Hidden-Layer Regularized Multilayer Perceptron for Meteorological Extreme Prediction.
    
    Layers:
    1. Fully-Connected Linear projection (n_features -> units1)
    2. Batch Normalization 1D (stabilizes internal covariate shift across mini-batches)
    3. ReLU activation (piecewise linear, eliminates vanishing gradients)
    4. Dropout (regularizes hidden node co-adaptation)
    5. Fully-Connected Linear projection (units1 -> units2)
    6. ReLU activation
    7. Dropout
    8. Output Linear projection (units2 -> 1 logit)
    """

    def __init__(self, n_features: int = 26, units1: int = 64, units2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_features, units1)
        self.bn1 = nn.BatchNorm1d(units1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(units1, units2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(units2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation producing raw logit."""
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        logits = self.fc3(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode forward pass producing calibrated probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Precision-Recall Area Under Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def train_single_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float = 3.115,
    max_epochs: int = 300,
    patience: int = 25,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a single MLP model with Early Stopping on Validation PR-AUC."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    best_val_pr_auc = -1.0
    best_weights = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
    }

    y_val_np = y_val.cpu().numpy().ravel()

    for epoch in range(1, max_epochs + 1):
        # Training pass
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()

        val_pr_auc = compute_pr_auc(y_val_np, val_probs)

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss))
        history["val_pr_auc"].append(float(val_pr_auc))

        # Check early stopping criterion (PR-AUC on validation)
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def grid_search_hyperparameters(
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    pos_weight: float,
    seeds: List[int] = [42, 43, 44, 45, 46],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Search over MLP hyperparameter grid on Validation PR-AUC across 5 seeds:
    - units1 in {32, 64}
    - units2 in {16, 32}
    - dropout in {0.2, 0.3}
    - lr in {1e-3, 3e-3}
    """
    units1_opts = [32, 64]
    units2_opts = [16, 32]
    dropout_opts = [0.2, 0.3]
    lr_opts = [1e-3, 3e-3]

    logger.info("Starting hyperparameter grid search across 16 configurations (5 seeds each)...")
    results = []

    for u1 in units1_opts:
        for u2 in units2_opts:
            for drop in dropout_opts:
                for lr in lr_opts:
                    val_pr_aucs = []
                    val_f1s = []
                    val_losses = []

                    for seed in seeds:
                        set_seed(seed)
                        model = HeatwaveMLP(n_features=X_train_t.shape[1], units1=u1, units2=u2, dropout=drop)
                        model, hist = train_single_model(
                            model, X_train_t, y_train_t, X_val_t, y_val_t,
                            lr=lr, pos_weight=pos_weight, max_epochs=200, patience=20
                        )
                        probs = model.predict_proba(X_val_t).cpu().numpy().ravel()
                        y_val_np = y_val_t.cpu().numpy().ravel()
                        
                        pr_auc = compute_pr_auc(y_val_np, probs)
                        f1 = f1_score(y_val_np, (probs >= 0.5).astype(int), zero_division=0)
                        
                        val_pr_aucs.append(pr_auc)
                        val_f1s.append(f1)
                        val_losses.append(hist["val_loss"][-1])

                    mean_pr_auc = float(np.mean(val_pr_aucs))
                    std_pr_auc = float(np.std(val_pr_aucs))
                    mean_f1 = float(np.mean(val_f1s))
                    mean_loss = float(np.mean(val_losses))

                    results.append({
                        "units1": u1,
                        "units2": u2,
                        "dropout": drop,
                        "lr": lr,
                        "mean_val_pr_auc": round(mean_pr_auc, 4),
                        "std_val_pr_auc": round(std_pr_auc, 4),
                        "mean_val_f1": round(mean_f1, 4),
                        "mean_val_loss": round(mean_loss, 4),
                    })

    df_grid = pd.DataFrame(results).sort_values(by="mean_val_pr_auc", ascending=False).reset_index(drop=True)
    best_config = {
        "units1": int(df_grid.iloc[0]["units1"]),
        "units2": int(df_grid.iloc[0]["units2"]),
        "dropout": float(df_grid.iloc[0]["dropout"]),
        "lr": float(df_grid.iloc[0]["lr"]),
    }
    logger.info(f"Winning Architecture: {best_config} with Val PR-AUC = {df_grid.iloc[0]['mean_val_pr_auc']:.4f}")
    return best_config, df_grid


def train_mlp_ensemble(
    config: dict,
    best_params: Dict[str, Any] = None,
    seeds: List[int] = list(range(42, 52)),  # 10 seeds: 42..51
) -> Dict[str, Any]:
    """Train the 10-seed Deep MLP Ensemble and persist all checkpoints and training curves."""
    splits = get_splits(config)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    pos_weight = float(n_neg / n_pos)
    logger.info(f"Class distribution in Train: {n_pos} positives, {n_neg} negatives (pos_weight={pos_weight:.3f})")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # 1. Grid search if best_params not pre-specified
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    if best_params is None:
        best_params, df_grid = grid_search_hyperparameters(X_train_t, y_train_t, X_val_t, y_val_t, pos_weight)
        df_grid.to_csv(tables_dir / "grid_search_results.csv", index=False)
        logger.info(f"Saved grid search results to {tables_dir / 'grid_search_results.csv'}")

    # 2. Train winning configuration across 10 seeds (42..51)
    logger.info(f"Retraining winning configuration across {len(seeds)} seeds ({seeds[0]}..{seeds[-1]})...")
    models = []
    histories = []
    val_probs_all = []
    test_probs_all = []
    train_probs_all = []

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        set_seed(seed)
        model = HeatwaveMLP(
            n_features=X_train.shape[1],
            units1=best_params["units1"],
            units2=best_params["units2"],
            dropout=best_params["dropout"],
        )
        model, hist = train_single_model(
            model, X_train_t, y_train_t, X_val_t, y_val_t,
            lr=best_params["lr"], pos_weight=pos_weight, max_epochs=300, patience=25
        )
        models.append(model)
        histories.append(hist)

        p_train = model.predict_proba(X_train_t).cpu().numpy().ravel()
        p_val = model.predict_proba(X_val_t).cpu().numpy().ravel()
        p_test = model.predict_proba(X_test_t).cpu().numpy().ravel()

        train_probs_all.append(p_train)
        val_probs_all.append(p_val)
        test_probs_all.append(p_test)

    # Compute ensemble average probabilities
    ensemble_val_probs = np.mean(val_probs_all, axis=0)
    ensemble_test_probs = np.mean(test_probs_all, axis=0)
    ensemble_train_probs = np.mean(train_probs_all, axis=0)

    # Compute validation metrics
    val_pr_auc = compute_pr_auc(y_val, ensemble_val_probs)
    val_roc_auc = float(roc_auc_score(y_val, ensemble_val_probs))
    val_brier = float(brier_score_loss(y_val, ensemble_val_probs))
    
    # Evaluate at threshold = 0.5 and search optimal threshold on Val
    val_preds_05 = (ensemble_val_probs >= 0.5).astype(int)
    f1_05 = float(f1_score(y_val, val_preds_05, zero_division=0))
    prec_05 = float(precision_score(y_val, val_preds_05, zero_division=0))
    rec_05 = float(recall_score(y_val, val_preds_05, zero_division=0))

    logger.info("--- 10-Seed Deep MLP Ensemble Validation Performance ---")
    logger.info(f"PR-AUC: {val_pr_auc:.4f} | ROC-AUC: {val_roc_auc:.4f} | Brier: {val_brier:.4f}")
    logger.info(f"Threshold 0.50: F1={f1_05:.4f}, Precision={prec_05:.4f}, Recall={rec_05:.4f}")

    # Save validation metrics to CSV
    val_metrics_df = pd.DataFrame([{
        "model": "10-Seed Deep MLP Ensemble",
        "n_seeds": len(seeds),
        "architecture": f"Input({X_train.shape[1]}) -> Dense({best_params['units1']}, ReLU, BN, Drop({best_params['dropout']})) -> Dense({best_params['units2']}, ReLU, Drop) -> Dense(1, Sigmoid)",
        "val_pr_auc": round(val_pr_auc, 4),
        "val_roc_auc": round(val_roc_auc, 4),
        "val_brier_score": round(val_brier, 4),
        "val_f1_thresh_05": round(f1_05, 4),
        "val_precision_thresh_05": round(prec_05, 4),
        "val_recall_thresh_05": round(rec_05, 4),
    }])
    val_metrics_df.to_csv(tables_dir / "ann_ensemble_val_metrics.csv", index=False)

    # Save PyTorch checkpoints for all seeds
    torch_save_dict = {
        "best_params": best_params,
        "seeds": seeds,
        "state_dicts": [m.state_dict() for m in models],
        "feature_names": splits["feature_names"],
    }
    torch.save(torch_save_dict, models_dir / "mlp_ensemble_checkpoints.pt")
    logger.info(f"Saved PyTorch ensemble checkpoints to {models_dir / 'mlp_ensemble_checkpoints.pt'}")

    # Save training loss history for fig05 plot
    history_export = {
        "seeds": seeds,
        "histories": histories,
    }
    with open(models_dir / "ensemble_train_history.json", "w", encoding="utf-8") as f:
        json.dump(history_export, f)
    logger.info(f"Saved ensemble training histories to {models_dir / 'ensemble_train_history.json'}")

    return {
        "models": models,
        "best_params": best_params,
        "ensemble_train_probs": ensemble_train_probs,
        "ensemble_val_probs": ensemble_val_probs,
        "ensemble_test_probs": ensemble_test_probs,
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "val_f1_05": f1_05,
        "histories": histories,
    }


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_mlp_ensemble(cfg)
