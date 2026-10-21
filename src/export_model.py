# LEGACY/OPTIONAL: original experiment only. Use src.experiment and frontend/ for Heatline.
# Old weights, label definitions and reported metrics are not current assessed evidence.
"""
Lightweight Model Export & Pure-NumPy Inference Module.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).

Purpose:
- Exports PyTorch 10-seed MLP ensemble parameters into `models/ensemble_weights.npz`.
- Implements `NumpyMLPEnsemble` for 100% dependency-free, zero-cloud-overhead
  inference in Streamlit and embedded platforms (no PyTorch/TensorFlow wheels needed).
- Formally verifies numerical equivalence between PyTorch and pure-NumPy forward passes.
"""

import json
import logging
from pathlib import Path
import sys
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
try:
    import torch
except ImportError:
    torch = None
import yaml

from src.split import get_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NumpyMLPEnsemble:
    """
    100% Pure-NumPy Feedforward Inference Engine for the 10-Seed Deep MLP Ensemble.
    Eliminates all PyTorch/CUDA dependencies for lightweight deployment.
    """

    def __init__(self, weights_path: str = "models/ensemble_weights.npz"):
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"Weights file not found at: {weights_path}")
        
        data = np.load(p)
        self.n_seeds = int(data["n_seeds"])
        self.feature_names = list(data["feature_names"])
        self.eps = 1e-5

        # Unpack parameters for each seed
        self.seed_params = []
        for s in range(self.n_seeds):
            self.seed_params.append({
                "w1": data[f"w1_{s}"],
                "b1": data[f"b1_{s}"],
                "gamma": data[f"gamma_{s}"],
                "beta": data[f"beta_{s}"],
                "mean": data[f"mean_{s}"],
                "var": data[f"var_{s}"],
                "w2": data[f"w2_{s}"],
                "b2": data[f"b2_{s}"],
                "w3": data[f"w3_{s}"],
                "b3": data[f"b3_{s}"],
            })

    def forward_seed(self, X: np.ndarray, s_idx: int) -> np.ndarray:
        """Single model feedforward pass in pure NumPy."""
        p = self.seed_params[s_idx]
        
        # Layer 1 + BatchNorm
        x1 = X @ p["w1"].T + p["b1"]
        bn1 = (x1 - p["mean"]) / np.sqrt(p["var"] + self.eps) * p["gamma"] + p["beta"]
        h1 = np.maximum(0.0, bn1)  # ReLU
        
        # Layer 2
        x2 = h1 @ p["w2"].T + p["b2"]
        h2 = np.maximum(0.0, x2)  # ReLU
        
        # Output Layer (Binary Sigmoid)
        logits = h2 @ p["w3"].T + p["b3"]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        return probs.ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble average predicted probability across all 10 seeds."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        seed_probs = np.zeros((self.n_seeds, len(X)), dtype=float)
        for s in range(self.n_seeds):
            seed_probs[s] = self.forward_seed(X, s)

        return np.mean(seed_probs, axis=0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary classification decision based on threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)


def export_torch_ensemble_to_numpy(
    torch_checkpoint_path: str = "models/mlp_ensemble_checkpoints.pt",
    output_npz_path: str = "models/ensemble_weights.npz",
) -> None:
    """Extract PyTorch parameters from checkpoint and save to NPZ archive."""
    chkpt_path = Path(torch_checkpoint_path)
    if not chkpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {chkpt_path}")

    chkpt = torch.load(chkpt_path, weights_only=False)
    state_dicts = chkpt["state_dicts"]
    feature_names = chkpt["feature_names"]
    n_seeds = len(state_dicts)

    export_dict = {
        "n_seeds": np.array(n_seeds),
        "feature_names": np.array(feature_names),
    }

    for s, sd in enumerate(state_dicts):
        export_dict[f"w1_{s}"] = sd["fc1.weight"].cpu().numpy()
        export_dict[f"b1_{s}"] = sd["fc1.bias"].cpu().numpy()
        export_dict[f"gamma_{s}"] = sd["bn1.weight"].cpu().numpy()
        export_dict[f"beta_{s}"] = sd["bn1.bias"].cpu().numpy()
        export_dict[f"mean_{s}"] = sd["bn1.running_mean"].cpu().numpy()
        export_dict[f"var_{s}"] = sd["bn1.running_var"].cpu().numpy()
        export_dict[f"w2_{s}"] = sd["fc2.weight"].cpu().numpy()
        export_dict[f"b2_{s}"] = sd["fc2.bias"].cpu().numpy()
        export_dict[f"w3_{s}"] = sd["fc3.weight"].cpu().numpy()
        export_dict[f"b3_{s}"] = sd["fc3.bias"].cpu().numpy()

    np.savez_compressed(output_npz_path, **export_dict)
    logger.info(f"Successfully exported 10-seed ensemble weights to: {output_npz_path}")


def verify_inference_equivalence(config: dict) -> None:
    """Formally assert bitwise / numerical equivalence between PyTorch and pure-NumPy inference."""
    splits = get_splits(config)
    X_val = splits["X_val"]

    # 1. Load PyTorch model
    from src.train_ann import HeatwaveMLP
    chkpt = torch.load("models/mlp_ensemble_checkpoints.pt", weights_only=False)
    params = chkpt["best_params"]
    state_dicts = chkpt["state_dicts"]
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    pt_probs_all = []
    for sd in state_dicts:
        m = HeatwaveMLP(n_features=X_val.shape[1], units1=params["units1"], units2=params["units2"], dropout=params["dropout"])
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            pt_probs_all.append(torch.sigmoid(m(X_val_t)).cpu().numpy().ravel())
    pt_ensemble_probs = np.mean(pt_probs_all, axis=0)

    # 2. Load pure-NumPy ensemble
    np_ensemble = NumpyMLPEnsemble("models/ensemble_weights.npz")
    np_ensemble_probs = np_ensemble.predict_proba(X_val)

    # 3. Assert tolerance
    max_diff = float(np.max(np.abs(pt_ensemble_probs - np_ensemble_probs)))
    mean_diff = float(np.mean(np.abs(pt_ensemble_probs - np_ensemble_probs)))
    logger.info(f"Equivalence Verification: Max Diff = {max_diff:.2e}, Mean Diff = {mean_diff:.2e}")
    assert max_diff < 1e-5, f"Inference mismatch: max diff {max_diff} exceeds 1e-5"
    logger.info("Verification Passed: Pure-NumPy forward pass is identical to PyTorch (< 1e-5).")


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    export_torch_ensemble_to_numpy()
    verify_inference_equivalence(cfg)
