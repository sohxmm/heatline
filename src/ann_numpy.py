"""
Error Backpropagation Neural Network (EBPA) in Pure NumPy.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).

Theoretical & Pedagogical Framework:
-------------------------------------
1. McCulloch-Pitts Neuron (Exp 1, 1943):
   - First computational model of a biological neuron with binary inputs/outputs
     and fixed threshold logic: y = 1 if sum(w_i * x_i) >= theta else 0.
   - Limitation: Fixed weights and thresholds; no learning mechanism.

2. Activation Functions (Exp 2):
   - Binary Sigmoid (Unipolar): f(x) = 1 / (1 + exp(-x)), range (0, 1)
     Derivative: f'(x) = f(x) * (1 - f(x))
     Textbook equation: df/dx = f(x) * (1 - f(x))
     Smooth, differentiable, suitable for probability output. Suffers from vanishing
     gradient for large |x|.
   - Bipolar Sigmoid: f(x) = 2 / (1 + exp(-x)) - 1 = tanh(x / 2), range (-1, 1)
     Derivative: f'(x) = 0.5 * (1 + f(x)) * (1 - f(x))
     Textbook equation: df/dx = 0.5 * (1 + f(x)) * (1 - f(x))
     Zero-mean activation property reduces gradient zig-zagging during backpropagation.
   - Rectified Linear Unit (ReLU): f(x) = max(0, x), range [0, inf)
     Derivative: f'(x) = 1 if x > 0 else 0
     Textbook equation: df/dx = H(x) (Heaviside step function)
     Mitigates vanishing gradient for positive activations; requires gradient
     clipping to prevent exploding gradients.

3. Single-Layer Perceptron (Exp 4, Rosenblatt 1958) & ADALINE (Exp 5, Widrow-Hoff 1960):
   - Perceptron Learning Rule updates weights on discrete classification error.
   - ADALINE introduces the continuous Delta Rule (Least Mean Squares / LMS):
     Delta_w = eta * (t - y_in) * x, minimizing Mean Squared Error (MSE) via
     gradient descent on a quadratic error surface.
   - Limitation: Minsky & Papert (1969) proved single-layer networks cannot solve
     linearly non-separable problems (e.g. XOR logic, multi-variable climate extremes).

4. Multi-Layer Perceptron & Error Backpropagation Algorithm (Exp 6 & 7):
   - Rumelhart, Hinton, & Williams (1986); Laurene Fausett (1994).
   - Introduces hidden layers with continuous non-linear activation functions to form
     arbitrary internal representations and non-linear decision boundaries.
   - Generalized Delta Rule backpropagates error signals from output layer backwards:
     Forward Pass:
       z_in_j = v_0j + sum_i(x_i * v_ij)       [hidden pre-activation]
       z_j    = f(z_in_j)                       [hidden activation]
       y_in_k = w_0k + sum_j(z_j * w_jk)       [output pre-activation]
       y_k    = sigmoid(y_in_k)                 [output activation, binary sigmoid]
     Output Error Signal:
       delta_k = (t_k - y_k) * f'(y_in_k)       [Fausett Eq. 6.6]
     Hidden Error Signal:
       delta_in_j = sum_k(delta_k * w_jk)       [error backpropagated to hidden layer]
       delta_j    = delta_in_j * f'(z_in_j)     [Fausett Eq. 6.8]
     Weight Updates with Momentum (alpha):
       Delta_w_jk(t) = eta * delta_k * z_j + alpha * Delta_w_jk(t - 1)
       Delta_w_0k(t) = eta * delta_k       + alpha * Delta_w_0k(t - 1)
       Delta_v_ij(t) = eta * delta_j * x_i + alpha * Delta_v_ij(t - 1)
       Delta_v_0j(t) = eta * delta_j       + alpha * Delta_v_0j(t - 1)
     Weights adjustment:
       w_jk <- w_jk + Delta_w_jk(t)
       v_ij <- v_ij + Delta_v_ij(t)
"""

import json
import logging
from pathlib import Path
import sys
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.split import get_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EBPANetwork:
    """
    Two-layer Error Backpropagation Artificial Neural Network in 100% Pure NumPy.
    Implements KJSCE Soft Computing Lab Experiments 2, 6, and 7.

    Architecture:
      Input (n_features) -> Hidden (n_hidden) -> Output (1, Binary Sigmoid)

    Parameters
    ----------
    n_input : int
        Number of input meteorological features (26).
    n_hidden : int, default=32
        Number of hidden layer neurons (z_j).
    activation : {'bipolar_sigmoid', 'binary_sigmoid', 'relu'}, default='bipolar_sigmoid'
        Hidden layer activation function (Exp 2). Bipolar sigmoid is default
        as demonstrated by the activation comparison study.
    learning_rate : float, default=0.01
        Learning rate (eta) for gradient updates.
    momentum : float, default=0.9
        Momentum factor (alpha) to damp oscillations and accelerate convergence.
    seed : int, default=42
        Random seed for reproducible weight initialization.
    """

    SUPPORTED_ACTIVATIONS = ("binary_sigmoid", "bipolar_sigmoid", "relu")

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 32,
        activation: str = "bipolar_sigmoid",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        seed: int = 42,
    ):
        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. Must be one of {self.SUPPORTED_ACTIVATIONS}"
            )

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = 1  # Binary classification (Heatwave Day vs Non-Heatwave Day)
        self.activation = activation
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.seed = seed

        # Initialize network weights and biases
        self._init_weights()

        # Allocate momentum buffers: Delta_w(t - 1) and Delta_v(t - 1)
        self.delta_W_prev = np.zeros((self.n_hidden, self.n_output), dtype=np.float64)
        self.delta_w0_prev = np.zeros((1, self.n_output), dtype=np.float64)
        self.delta_V_prev = np.zeros((self.n_input, self.n_hidden), dtype=np.float64)
        self.delta_v0_prev = np.zeros((1, self.n_hidden), dtype=np.float64)

        # Cache for forward activations (used during backpropagation)
        self.z_in: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.y_in: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None

        # Checkpointing
        self.best_weights: Optional[Dict[str, np.ndarray]] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "epoch": []}

    def _init_weights(self) -> None:
        """
        Weight Initialization (Exp 6 & 7):
        - Xavier / Glorot Initialization for Sigmoid activations:
          Maintains activation variance across layers, avoiding premature saturation:
          V ~ Uniform(-sqrt(6 / (n_in + n_hid)), sqrt(6 / (n_in + n_hid)))
          W ~ Uniform(-sqrt(6 / (n_hid + n_out)), sqrt(6 / (n_hid + n_out)))
        - He / Kaiming Initialization for ReLU:
          Accounts for half the neurons being zeroed out:
          V ~ Normal(0, sqrt(2 / n_in))
          W ~ Normal(0, sqrt(2 / n_hid))
        - Biases initialized to zero.
        """
        rng = np.random.RandomState(self.seed)

        if self.activation == "relu":
            # He Normal Initialization
            std_v = np.sqrt(2.0 / self.n_input)
            self.V = rng.randn(self.n_input, self.n_hidden).astype(np.float64) * std_v
            std_w = np.sqrt(2.0 / self.n_hidden)
            self.W = rng.randn(self.n_hidden, self.n_output).astype(np.float64) * std_w
        else:
            # Xavier Uniform Initialization
            limit_v = np.sqrt(6.0 / (self.n_input + self.n_hidden))
            self.V = rng.uniform(-limit_v, limit_v, (self.n_input, self.n_hidden)).astype(np.float64)
            limit_w = np.sqrt(6.0 / (self.n_hidden + self.n_output))
            self.W = rng.uniform(-limit_w, limit_w, (self.n_hidden, self.n_output)).astype(np.float64)

        # Biases initialized to zero
        self.v0 = np.zeros((1, self.n_hidden), dtype=np.float64)
        self.w0 = np.zeros((1, self.n_output), dtype=np.float64)

    def _activation_fn(self, z_in: np.ndarray) -> np.ndarray:
        """
        Hidden Layer Activation Function f(z_in) [KJSCE Exp 2]:
        - binary_sigmoid:  f(x) = 1 / (1 + exp(-x))
        - bipolar_sigmoid: f(x) = 2 / (1 + exp(-x)) - 1 = tanh(x / 2)
        - relu:            f(x) = max(0, x)
        """
        clipped = np.clip(z_in, -30.0, 30.0)
        if self.activation == "binary_sigmoid":
            return 1.0 / (1.0 + np.exp(-clipped))
        elif self.activation == "bipolar_sigmoid":
            return 2.0 / (1.0 + np.exp(-clipped)) - 1.0
        elif self.activation == "relu":
            return np.maximum(0.0, z_in)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_derivative(self, z_in: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Analytical Derivative of Hidden Layer Activation f'(z_in) [KJSCE Exp 2]:
        - binary_sigmoid:  f'(x) = f(x) * (1 - f(x)) = z * (1 - z)
        - bipolar_sigmoid: f'(x) = 0.5 * (1 + f(x)) * (1 - f(x)) = 0.5 * (1 + z) * (1 - z)
        - relu:            f'(x) = 1 if x > 0 else 0
        """
        if self.activation == "binary_sigmoid":
            return z * (1.0 - z)
        elif self.activation == "bipolar_sigmoid":
            return 0.5 * (1.0 + z) * (1.0 - z)
        elif self.activation == "relu":
            return (z_in > 0.0).astype(np.float64)
        raise ValueError(f"Unknown activation: {self.activation}")

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Output Layer Binary Sigmoid Activation [KJSCE Exp 2]:
        sigmoid(x) = 1 / (1 + exp(-x))
        Maps output pre-activation y_in to posterior probability P(Heatwave=1|X) in (0, 1).
        """
        clipped = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward Pass (KJSCE Exp 6 & 7):
        Computes layer-by-layer activations through the network:

        Hidden Layer:
          z_in_j = v_0j + sum_i(x_i * v_ij)   [hidden pre-activation]
          z_j    = f(z_in_j)                  [hidden activation]

        Output Layer:
          y_in_k = w_0k + sum_j(z_j * w_jk)   [output pre-activation]
          y_k    = sigmoid(y_in_k)            [output activation, always binary sigmoid]

        Parameters
        ----------
        X : np.ndarray of shape (N, n_input)
            Standardized input feature matrix.

        Returns
        -------
        np.ndarray of shape (N, 1)
            Predicted probabilities y_k in [0, 1].
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Hidden pre-activation: z_in_j = v_0j + sum_i(x_i * v_ij)
        self.z_in = X_arr @ self.V + self.v0

        # Hidden activation: z_j = f(z_in_j)
        self.z = self._activation_fn(self.z_in)

        # Output pre-activation: y_in_k = w_0k + sum_j(z_j * w_jk)
        self.y_in = self.z @ self.W + self.w0

        # Output activation: y_k = sigmoid(y_in_k)
        self.y_pred = self._sigmoid(self.y_in)

        return self.y_pred

    def backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        clip_value: float = 5.0,
    ) -> Dict[str, np.ndarray]:
        """
        Backward Pass & Weight Updates (KJSCE Exp 6 & 7):
        Computes generalized delta error signals and applies updates with momentum:

        Output Error Signal:
          delta_k = (t_k - y_k) * f'(y_in_k)
          where f'(y_in_k) = y_k * (1 - y_k)   [binary sigmoid derivative]

        Hidden Error Signal:
          delta_in_j = sum_k(delta_k * w_jk)   [backpropagated error from output layer]
          delta_j    = delta_in_j * f'(z_in_j) [scaled by hidden activation derivative]

        Weight Updates with Learning Rate (eta) and Momentum (alpha):
          Delta_w_jk(t) = eta * (1/B) * (z^T @ delta_k) + alpha * Delta_w_jk(t - 1)
          Delta_w_0k(t) = eta * (1/B) * sum(delta_k)    + alpha * Delta_w_0k(t - 1)
          Delta_v_ij(t) = eta * (1/B) * (X^T @ delta_j) + alpha * Delta_v_ij(t - 1)
          Delta_v_0j(t) = eta * (1/B) * sum(delta_j)    + alpha * Delta_v_0j(t - 1)

        Weights Adjustment:
          W  <- W  + Delta_w_jk(t)
          w0 <- w0 + Delta_w_0k(t)
          V  <- V  + Delta_v_ij(t)
          v0 <- v0 + Delta_v_0j(t)

        Parameters
        ----------
        X : np.ndarray of shape (B, n_input)
            Batch input features.
        y_true : np.ndarray of shape (B, 1) or (B,)
            Batch ground-truth targets (t_k in {0, 1}).
        y_pred : np.ndarray of shape (B, 1)
            Forward output probabilities y_k.
        clip_value : float, default=5.0
            Maximum gradient magnitude for clipping (essential for ReLU stability).

        Returns
        -------
        dict
            Contains delta_k, delta_j, grad_W, grad_V.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        t = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
        batch_size = len(t)

        # 1. Output error signal: delta_k = (t_k - y_k) * f'(y_in_k)
        # For binary sigmoid output: f'(y_in_k) = y_k * (1.0 - y_k)
        f_prime_out = y * (1.0 - y)
        delta_k = (t - y) * f_prime_out  # shape: (B, 1)

        # 2. Backpropagate error to hidden layer: delta_in_j = sum_k(delta_k * w_jk)
        delta_in_j = delta_k @ self.W.T  # shape: (B, n_hidden)

        # 3. Hidden error signal: delta_j = delta_in_j * f'(z_in_j)
        f_prime_hid = self._activation_derivative(self.z_in, self.z)  # shape: (B, n_hidden)
        delta_j = delta_in_j * f_prime_hid

        # 4. Compute mini-batch average gradients
        grad_W = (self.z.T @ delta_k) / batch_size
        grad_w0 = np.sum(delta_k, axis=0, keepdims=True) / batch_size
        grad_V = (X_arr.T @ delta_j) / batch_size
        grad_v0 = np.sum(delta_j, axis=0, keepdims=True) / batch_size

        # 5. Gradient clipping to prevent exploding gradients (particularly with ReLU)
        if clip_value is not None and clip_value > 0:
            grad_W = np.clip(grad_W, -clip_value, clip_value)
            grad_w0 = np.clip(grad_w0, -clip_value, clip_value)
            grad_V = np.clip(grad_V, -clip_value, clip_value)
            grad_v0 = np.clip(grad_v0, -clip_value, clip_value)

        # 6. Weight updates with learning rate (eta) and momentum (alpha):
        # Delta_w_jk(t) = eta * delta_k * z_j + alpha * Delta_w_jk(t - 1)
        delta_W = self.learning_rate * grad_W + self.momentum * self.delta_W_prev
        delta_w0 = self.learning_rate * grad_w0 + self.momentum * self.delta_w0_prev
        delta_V = self.learning_rate * grad_V + self.momentum * self.delta_V_prev
        delta_v0 = self.learning_rate * grad_v0 + self.momentum * self.delta_v0_prev

        # 7. Apply weight adjustments
        self.W += delta_W
        self.w0 += delta_w0
        self.V += delta_V
        self.v0 += delta_v0

        # 8. Update momentum memory for subsequent iteration
        self.delta_W_prev = delta_W
        self.delta_w0_prev = delta_w0
        self.delta_V_prev = delta_V
        self.delta_v0_prev = delta_v0

        return {
            "delta_k": delta_k,
            "delta_j": delta_j,
            "grad_W": grad_W,
            "grad_V": grad_V,
        }

    @staticmethod
    def compute_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pos_weight: float = 1.0,
        eps: float = 1e-15,
    ) -> float:
        """
        Class-Weighted Binary Cross-Entropy (BCE) Loss:
        L = - (1/N) * sum_i [ pos_weight * t_i * log(y_i) + (1 - t_i) * log(1 - y_i) ]

        Compensates for class imbalance in Nagpur heatwave days (~24.3% in train).
        pos_weight = n_neg / n_pos.
        """
        t = np.asarray(y_true, dtype=np.float64).ravel()
        y = np.clip(np.asarray(y_pred, dtype=np.float64).ravel(), eps, 1.0 - eps)
        loss = -np.mean(pos_weight * t * np.log(y) + (1.0 - t) * np.log(1.0 - y))
        return float(loss)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_epochs: int = 500,
        patience: int = 30,
        pos_weight: Optional[float] = None,
        batch_size: int = 32,
        clip_value: float = 5.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the EBPA Network using Mini-Batch SGD with Momentum and Early Stopping.

        Parameters
        ----------
        X_train : np.ndarray
            Standardized training inputs (1981-2010).
        y_train : np.ndarray
            Binary training labels (1 = Heatwave, 0 = Normal).
        X_val : np.ndarray, optional
            Standardized validation inputs (2011-2018).
        y_val : np.ndarray, optional
            Binary validation labels.
        max_epochs : int, default=500
            Maximum number of complete passes over the training dataset.
        patience : int, default=30
            Early stopping patience on validation loss to prevent overfitting.
        pos_weight : float, optional
            Class weight for positive class. Defaults to n_neg / n_pos from training set.
        batch_size : int, default=32
            Mini-batch size for stochastic gradient descent.
        clip_value : float, default=5.0
            Gradient clipping bound.
        verbose : bool, default=False
            Whether to log progress every 20 epochs.

        Returns
        -------
        dict
            Training history and summary metrics.
        """
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
        n_samples = len(X_tr)

        if pos_weight is None:
            n_pos = float(np.sum(y_tr == 1.0))
            n_neg = float(np.sum(y_tr == 0.0))
            pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.float64).reshape(-1, 1)

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0
        self.history = {"train_loss": [], "val_loss": [], "epoch": []}

        # Save initial checkpoint
        self.best_weights = {
            "V": self.V.copy(),
            "v0": self.v0.copy(),
            "W": self.W.copy(),
            "w0": self.w0.copy(),
        }

        rng = np.random.RandomState(self.seed)

        for epoch in range(max_epochs):
            # Mini-batch shuffling
            indices = rng.permutation(n_samples)
            X_shuffled = X_tr[indices]
            y_shuffled = y_tr[indices]

            n_batches = int(np.ceil(n_samples / batch_size))
            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                xb = X_shuffled[start_idx:end_idx]
                yb = y_shuffled[start_idx:end_idx]

                # Forward pass on mini-batch
                yb_pred = self.forward(xb)

                # Backward pass & weight update on mini-batch
                self.backward(xb, yb, yb_pred, clip_value=clip_value)

            # Evaluate training loss after full epoch
            y_tr_pred = self.forward(X_tr)
            train_loss = self.compute_loss(y_tr, y_tr_pred, pos_weight=pos_weight)
            self.history["train_loss"].append(train_loss)
            self.history["epoch"].append(epoch)

            # Evaluate validation loss
            if has_val:
                y_v_pred = self.forward(X_v)
                val_loss = self.compute_loss(y_v, y_v_pred, pos_weight=pos_weight)
                self.history["val_loss"].append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    self.best_weights = {
                        "V": self.V.copy(),
                        "v0": self.v0.copy(),
                        "W": self.W.copy(),
                        "w0": self.w0.copy(),
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            logger.info(
                                "Early stopping triggered at epoch %d (best epoch: %d, val_loss: %.4f)",
                                epoch,
                                best_epoch,
                                best_val_loss,
                            )
                        break

            if verbose and (epoch % 20 == 0 or epoch == max_epochs - 1):
                val_str = f", val_loss={self.history['val_loss'][-1]:.4f}" if has_val else ""
                logger.info("Epoch %3d: train_loss=%.4f%s", epoch, train_loss, val_str)

        # Restore optimal weights from best validation checkpoint
        if has_val and self.best_weights is not None:
            self.V = self.best_weights["V"].copy()
            self.v0 = self.best_weights["v0"].copy()
            self.W = self.best_weights["W"].copy()
            self.w0 = self.best_weights["w0"].copy()

        return {
            "best_epoch": best_epoch,
            "epochs_trained": len(self.history["train_loss"]),
            "best_val_loss": best_val_loss if has_val else train_loss,
            "final_train_loss": self.history["train_loss"][best_epoch] if has_val else train_loss,
            "history": self.history,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predicted heatwave probability P(y=1|X) for input instances.

        Parameters
        ----------
        X : np.ndarray of shape (N, n_input)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (N,)
            Predicted probabilities in [0, 1].
        """
        y_pred = self.forward(X)
        return y_pred.ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary heatwave predictions (1 = Heatwave, 0 = Normal)
        based on a classification decision threshold.

        Parameters
        ----------
        X : np.ndarray of shape (N, n_input)
            Input feature matrix.
        threshold : float, default=0.5
            Decision threshold on predicted probability.

        Returns
        -------
        np.ndarray of shape (N,)
            Binary class predictions in {0, 1}.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.int64)

    def save_weights(self, filepath: str) -> None:
        """Export model weights and architecture configuration to JSON."""
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n_input": self.n_input,
            "n_hidden": self.n_hidden,
            "n_output": self.n_output,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "seed": self.seed,
            "V": self.V.tolist(),
            "v0": self.v0.tolist(),
            "W": self.W.tolist(),
            "w0": self.w0.tolist(),
        }
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved EBPANetwork model to %s", filepath)

    def load_weights(self, filepath: str) -> None:
        """Load model weights and architecture configuration from JSON."""
        dest = Path(filepath)
        if not dest.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(dest, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.n_input = data["n_input"]
        self.n_hidden = data["n_hidden"]
        self.n_output = data["n_output"]
        self.activation = data["activation"]
        self.learning_rate = data["learning_rate"]
        self.momentum = data["momentum"]
        self.seed = data.get("seed", 42)
        self.V = np.array(data["V"], dtype=np.float64)
        self.v0 = np.array(data["v0"], dtype=np.float64)
        self.W = np.array(data["W"], dtype=np.float64)
        self.w0 = np.array(data["w0"], dtype=np.float64)
        logger.info("Loaded EBPANetwork model from %s", filepath)


def run_activation_study(
    config_path: str = "config.yaml",
    out_table_path: str = "results/tables/exp2_activation_comparison.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    KJSCE Soft Computing Lab Experiment 2: Activation Function Comparison Study.

    Trains the 2-layer EBPA network with each of the three hidden layer activations:
      1. Binary Sigmoid:  f(x) = 1 / (1 + exp(-x))
      2. Bipolar Sigmoid: f(x) = 2 / (1 + exp(-x)) - 1 = tanh(x / 2)
      3. ReLU:            f(x) = max(0, x)

    For each activation, records:
      - final train loss
      - val loss
      - val F1
      - val PR-AUC
      - val ROC-AUC
      - epochs trained

    Saves results to results/tables/exp2_activation_comparison.csv.
    """
    splits = get_splits()
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]

    activations = ["binary_sigmoid", "bipolar_sigmoid", "relu"]
    records = []

    pos_weight = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    logger.info("Starting Activation Function Study (Exp 2) with pos_weight=%.4f", pos_weight)

    for act in activations:
        logger.info("Evaluating activation: %s", act)
        net = EBPANetwork(
            n_input=X_train.shape[1],
            n_hidden=32,
            activation=act,
            learning_rate=0.01,
            momentum=0.9,
            seed=seed,
        )

        train_info = net.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_epochs=500,
            patience=30,
            pos_weight=pos_weight,
            batch_size=32,
            clip_value=5.0,
            verbose=False,
        )

        # Evaluate validation metrics with restored best weights
        val_proba = net.predict_proba(X_val)
        val_pred = net.predict(X_val, threshold=0.5)

        val_f1 = f1_score(y_val, val_pred)
        val_pr_auc = average_precision_score(y_val, val_proba)
        val_roc_auc = roc_auc_score(y_val, val_proba)

        rec = {
            "activation": act,
            "final_train_loss": round(train_info["final_train_loss"], 4),
            "val_loss": round(train_info["best_val_loss"], 4),
            "val_f1": round(val_f1, 4),
            "val_pr_auc": round(val_pr_auc, 4),
            "val_roc_auc": round(val_roc_auc, 4),
            "epochs_trained": train_info["epochs_trained"],
        }
        records.append(rec)
        logger.info(
            "[%s] Ep=%d | TrainLoss=%.4f | ValLoss=%.4f | ValF1=%.4f | PR-AUC=%.4f | ROC-AUC=%.4f",
            act,
            train_info["epochs_trained"],
            rec["final_train_loss"],
            rec["val_loss"],
            rec["val_f1"],
            rec["val_pr_auc"],
            rec["val_roc_auc"],
        )

    df_results = pd.DataFrame(records)

    out_file = Path(out_table_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_file, index=False)
    logger.info("Saved activation comparison table to: %s", out_table_path)

    # Determine winning activation based on lowest validation loss (and PR-AUC / ROC-AUC)
    best_row = df_results.sort_values(by=["val_loss", "val_pr_auc"], ascending=[True, False]).iloc[0]
    logger.info(
        "Best performing activation: %s (Val Loss: %.4f, Val F1: %.4f, Val ROC-AUC: %.4f)",
        best_row["activation"],
        best_row["val_loss"],
        best_row["val_f1"],
        best_row["val_roc_auc"],
    )

    return df_results


if __name__ == "__main__":
    df_comp = run_activation_study()
    print("\n" + "=" * 70)
    print("KJSCE Soft Computing Lab Exp 2: Activation Function Comparison Table")
    print("=" * 70)
    print(df_comp.to_string(index=False))
    print("=" * 70)
