"""
KJSCE Soft Computing Lab — Foundational Neural Baseline Classifiers.
Course: Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).
Location: Nagpur Urban Agglomeration (Vidarbha, Central India Core Heatwave Zone).

This module implements three classical connectionist architectures:
1. McCulloch-Pitts Neuron (Exp 1):
   Handcrafted threshold logic unit with excitatory and inhibitory synapses
   operating on raw (unscaled) meteorological threshold conditions.
2. Single-Layer Perceptron (Exp 4):
   Rosenblatt's step-activation learning rule with bipolar targets {-1, +1}
   trained on standardized multi-dimensional atmospheric predictors.
3. ADALINE with Continuous Delta Rule (Exp 5):
   Widrow-Hoff Least Mean Squares (LMS) gradient descent on continuous
   sigmoid activations, contrasting against discrete perceptron step updates.

Atmospheric Physics & Soft Computing Synthesis:
- Excitatory inputs reflect thermal accumulation (Tmax >= 42°C), severe positive
  departures from 1991-2020 WMO climatology (+3.5°C), and low relative humidity (RH <= 25%).
- Inhibitory inputs model turbulent wind ventilation (Wind >= 15 km/h), breaking
  boundary-layer heat traps and advecting cooler regional air.
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

from src.features import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable logistic sigmoid activation function.
    f(z) = 1 / (1 + exp(-z))
    Maps net input to continuous probabilities in (0, 1).
    """
    # Clip z to prevent float64 overflow in exp(-z)
    z_clipped = np.clip(z, -250.0, 250.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


class McCullochPittsNeuron(BaseEstimator, ClassifierMixin):
    """
    McCulloch-Pitts Neuron Classifier (KJSCE Soft Computing Lab Exp 1).

    Biological Analogy:
    Proposed by Warren McCulloch and Walter Pitts in 1943, this model represents
    the earliest mathematical abstraction of biological neural computation:
    - Dendrites receive electrochemical signals through excitatory synapses (which
      depolarize the cell membrane toward firing) or inhibitory synapses (which
      hyperpolarize the membrane, preventing action potential generation).
    - Soma performs spatial summation: net = sum(excitatory) - w_inh * inhibitory.
    - Axon Hillock enforces an all-or-none firing threshold:
      y = 1 if net >= theta else 0.
    - Absolute Inhibitory Veto: In biological neurons, an active inhibitory synapse
      can completely suppress firing regardless of excitatory excitation.

    Meteorological Operationalization for Nagpur Heatwaves:
    Heatwaves over Vidarbha are driven by strong solar insolation and dry advection,
    while strong surface winds produce turbulent boundary-layer mixing that suppresses
    heat accumulation. We encode this physical understanding into binary synapses:
    1. Excitatory 1: tmax_lag1 >= 42.0°C (Absolute thermal threshold)
    2. Excitatory 2: tmax_anomaly_lag1 >= 3.5°C (Departure from 1991-2020 normal)
    3. Excitatory 3: rh_min_lag1 <= 25.0% (Desiccated continental air mass)
    4. Inhibitory:   wind_mean_lag1 >= 15.0 km/h (Boundary-layer turbulent ventilation)

    IMPORTANT: This model operates strictly on RAW (unscaled) meteorological
    features (e.g. from features.csv or splits['val_df']), not standardized X arrays.

    Parameters
    ----------
    tmax_threshold : float, default=42.0
        Threshold for previous day's maximum temperature (°C).
    anomaly_threshold : float, default=3.5
        Threshold for previous day's climatological temperature anomaly (°C).
    rh_min_threshold : float, default=25.0
        Upper bound threshold for previous day's minimum relative humidity (%).
    wind_threshold : float, default=15.0
        Lower bound threshold for wind speed (km/h) triggering inhibitory suppression.
    theta : float, default=2.0
        Firing threshold: neuron discharges when net >= theta.
    inhibitory_weight : float, default=2.0
        Suppression penalty applied when wind_mean_lag1 >= wind_threshold.
        Since max excitatory sum is 3, w_inh >= 2 ensures that net <= 3 - 2 = 1 < theta (2),
        biologically vetoing firing on windy days.
    """

    def __init__(
        self,
        tmax_threshold: float = 42.0,
        anomaly_threshold: float = 3.5,
        rh_min_threshold: float = 25.0,
        wind_threshold: float = 15.0,
        theta: float = 2.0,
        inhibitory_weight: float = 2.0,
    ):
        self.tmax_threshold = float(tmax_threshold)
        self.anomaly_threshold = float(anomaly_threshold)
        self.rh_min_threshold = float(rh_min_threshold)
        self.wind_threshold = float(wind_threshold)
        self.theta = float(theta)
        self.inhibitory_weight = float(inhibitory_weight)
        self.classes_ = np.array([0, 1], dtype=np.int64)

    def fit(self, X: Optional[Any] = None, y: Optional[Any] = None) -> "McCullochPittsNeuron":
        """
        Analytical fitting for McCulloch-Pitts Neuron.
        In Exp 1, synaptic weights and firing thresholds are fixed analytically
        by expert domain knowledge rather than adjusted via numerical optimization.
        """
        return self

    def _extract_raw_features(
        self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract required meteorological predictors from DataFrame, array, or dict.
        Required columns: tmax_lag1, tmax_anomaly_lag1, rh_min_lag1, wind_mean_lag1.
        """
        if isinstance(X, pd.DataFrame):
            tmax_lag1 = X["tmax_lag1"].to_numpy(dtype=np.float64)
            anomaly = X["tmax_anomaly_lag1"].to_numpy(dtype=np.float64)
            rh_min = X["rh_min_lag1"].to_numpy(dtype=np.float64)
            wind = X["wind_mean_lag1"].to_numpy(dtype=np.float64)
            return tmax_lag1, anomaly, rh_min, wind

        if isinstance(X, dict):
            tmax_lag1 = np.asarray(X["tmax_lag1"], dtype=np.float64)
            anomaly = np.asarray(X["tmax_anomaly_lag1"], dtype=np.float64)
            rh_min = np.asarray(X["rh_min_lag1"], dtype=np.float64)
            wind = np.asarray(X["wind_mean_lag1"], dtype=np.float64)
            return tmax_lag1, anomaly, rh_min, wind

        if isinstance(X, np.ndarray):
            # If 2D array matching FEATURE_COLS:
            # 0: tmax_lag1, 4: rh_min_lag1, 7: wind_mean_lag1, 19: tmax_anomaly_lag1
            if X.ndim == 2 and X.shape[1] >= 20:
                tmax_lag1 = X[:, 0].astype(np.float64)
                rh_min = X[:, 4].astype(np.float64)
                wind = X[:, 7].astype(np.float64)
                anomaly = X[:, 19].astype(np.float64)
                return tmax_lag1, anomaly, rh_min, wind
            elif X.ndim == 2 and X.shape[1] == 4:
                # Direct 4-column raw input: [tmax, anomaly, rh_min, wind]
                return X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            elif X.ndim == 1 and len(X) == 4:
                return (
                    np.array([X[0]], dtype=np.float64),
                    np.array([X[1]], dtype=np.float64),
                    np.array([X[2]], dtype=np.float64),
                    np.array([X[3]], dtype=np.float64),
                )
            else:
                raise ValueError(
                    f"Array shape {X.shape} incompatible with McCulloch-Pitts feature extraction."
                )

        raise TypeError(f"Unsupported input type for McCullochPittsNeuron: {type(X)}")

    def compute_net_input(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Compute aggregate membrane potential (net input) across samples:
        net = (e1 + e2 + e3) - w_inh * inhibitory
        """
        tmax, anomaly, rh_min, wind = self._extract_raw_features(X)

        # Excitatory synaptic activations (1 if condition met, 0 otherwise)
        e1 = (tmax >= self.tmax_threshold).astype(np.float64)
        e2 = (anomaly >= self.anomaly_threshold).astype(np.float64)
        e3 = (rh_min <= self.rh_min_threshold).astype(np.float64)

        # Inhibitory synaptic activation
        inh = (wind >= self.wind_threshold).astype(np.float64)

        # Spatial dendritic summation
        net = (e1 + e2 + e3) - (self.inhibitory_weight * inh)
        return net

    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Binary prediction: all-or-none firing rule.
        Fires action potential (output=1) when net >= theta; remains quiescent (output=0) otherwise.
        """
        net = self.compute_net_input(X)
        return (net >= self.theta).astype(np.int64)

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Continuous decision score: distance of aggregate potential from threshold theta.
        score = net - theta
        """
        return self.compute_net_input(X) - self.theta

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Continuous risk confidence scores for ROC-AUC / PR-AUC ranking.
        Maps the distance from firing threshold (net - theta) through a logistic sigmoid.
        Returns 2D array [P(no heatwave), P(heatwave)].
        """
        score = self.decision_function(X)
        p1 = sigmoid(score)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class PerceptronRuleClassifier(BaseEstimator, ClassifierMixin):
    """
    Single-Layer Perceptron Classifier (KJSCE Soft Computing Lab Exp 4).

    Historical & Theoretical Foundation:
    Introduced by Frank Rosenblatt (1958) at Cornell Aeronautical Laboratory,
    the Perceptron is the pioneering learnable artificial neural network.
    - Architecture: A single linear summation layer with bias followed by a
      bipolar step activation function (Signum).
    - Inputs: All scaled atmospheric predictors x in R^D.
    - Targets: Bipolar representation t in {-1, +1}, where +1 denotes Heatwave Day
      and -1 denotes Non-Heatwave Day.
    - Forward Propagation:
      y_in = w^T x + b
      y = +1 if y_in >= 0 else -1

    Perceptron Learning Rule:
    For every training vector x_i presented sequentially:
    If predicted output matches target (y == t):
        No weight change: Delta w = 0, Delta b = 0.
    If misclassified (y != t):
        Delta w = eta * t * x_i
        Delta b = eta * t
        w <- w + Delta w
        b <- b + Delta b

    Novikoff's Perceptron Convergence Theorem (1962):
    If the training set is linearly separable with a positive margin gamma > 0,
    the perceptron learning algorithm is guaranteed to converge to a separating
    hyperplane in a finite number of correction steps k <= (R / gamma)^2,
    where R = max_i ||x_i|| is the radius of the input data sphere.
    However, real meteorological features exhibit non-linear boundaries and atmospheric
    turbulence, rendering the problem linearly non-separable. Hence, the algorithm
    oscillates without full convergence; we restrict training to max_epochs=100.

    Parameters
    ----------
    eta : float, default=0.01
        Learning rate parameter (step size for weight updates).
    max_epochs : int, default=100
        Maximum training iterations across the full training dataset.
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        eta: float = 0.01,
        max_epochs: int = 100,
        random_state: int = 42,
    ):
        self.eta = float(eta)
        self.max_epochs = int(max_epochs)
        self.random_state = int(random_state)
        self.classes_ = np.array([0, 1], dtype=np.int64)

        # Learned parameters
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.errors_per_epoch_: List[int] = []
        self.accuracies_per_epoch_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerceptronRuleClassifier":
        """
        Fit the single-layer perceptron on standardized training data.

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Standardized input features (e.g., X_train).
        y : np.ndarray of shape (N,)
            Binary target labels {0, 1}. Converted internally to bipolar {-1, +1}.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        n_samples, n_features = X.shape

        # Convert binary labels {0, 1} to bipolar targets {-1, +1}
        # KJSCE Exp 4 standard: t = +1 for positive class, -1 for negative class
        t = np.where(y == 1, 1, -1).astype(np.int64)

        # Initialize weights and bias to zeros (canonical Rosenblatt formulation)
        self.w_ = np.zeros(n_features, dtype=np.float64)
        self.b_ = 0.0
        self.errors_per_epoch_ = []
        self.accuracies_per_epoch_ = []

        logger.info(
            "Training PerceptronRuleClassifier (Exp 4): N=%d, D=%d, eta=%.4f, max_epochs=%d",
            n_samples, n_features, self.eta, self.max_epochs
        )

        for epoch in range(1, self.max_epochs + 1):
            epoch_errors = 0

            # Online sample-by-sample update rule
            for i in range(n_samples):
                xi = X[i]
                ti = t[i]

                # Net input calculation
                y_in = np.dot(xi, self.w_) + self.b_

                # Bipolar step activation function
                y_pred = 1 if y_in >= 0.0 else -1

                # Update weights only when an error occurs (y_pred != target)
                if y_pred != ti:
                    # Delta w = eta * t * x
                    # Delta b = eta * t
                    self.w_ += self.eta * ti * xi
                    self.b_ += self.eta * ti
                    epoch_errors += 1

            acc = 1.0 - (epoch_errors / n_samples)
            self.errors_per_epoch_.append(epoch_errors)
            self.accuracies_per_epoch_.append(acc)

            # Early stopping if linearly separable dataset converged perfectly
            if epoch_errors == 0:
                logger.info("Perceptron converged to zero error at epoch %d!", epoch)
                break

        logger.info(
            "Perceptron training finished. Final Epoch %d Errors: %d/%d (Acc: %.2f%%)",
            len(self.errors_per_epoch_), self.errors_per_epoch_[-1], n_samples,
            100.0 * self.accuracies_per_epoch_[-1]
        )
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear net input (distance from separating hyperplane).
        y_in = X @ w + b
        """
        if self.w_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.w_) + self.b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous class probability estimates via logistic sigmoid mapping:
        P(y=1|x) = sigmoid(y_in) = 1 / (1 + exp(-y_in))
        Returns 2D array [P(no heatwave), P(heatwave)].
        """
        y_in = self.decision_function(X)
        p1 = sigmoid(y_in)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary heatwave occurrence {0, 1}.
        Default threshold 0.5 on probability corresponds to y_in >= 0.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(np.int64)


class DeltaRuleADALINE(BaseEstimator, ClassifierMixin):
    """
    ADALINE with Continuous Delta Rule / Widrow-Hoff LMS (KJSCE Soft Computing Lab Exp 5).

    Theoretical Foundation:
    Developed by Bernard Widrow and Ted Hoff (1960) at Stanford University,
    ADALINE (Adaptive Linear Neuron) introduced the foundational Delta Rule,
    also known as the Least Mean Squares (LMS) learning algorithm.

    ADALINE / LMS Gradient Descent vs Perceptron Learning Rule:
    - In Rosenblatt's Perceptron (Exp 4), weights are updated ONLY when a discrete
      misclassification occurs after hard thresholding (y != t). If the data is
      non-separable, weights oscillate erratically across the decision boundary.
    - In ADALINE (Exp 5), the objective is to minimize a continuous Mean Squared
      Error (MSE) loss function over continuous activation values:
      E = 0.5 * sum_i (t_i - y_i)^2
      where y_i = f(y_in, i) is the continuous activation (here, logistic sigmoid).
    - Because the error surface is a smooth quadratic paraboloid in weight space,
      gradient descent slides smoothly toward the global minimum, maintaining
      stability even on linearly non-separable real-world weather patterns.

    Continuous Delta Rule Mathematics:
    - Net input:
      y_in = w^T x + b
    - Continuous activation function (Logistic Sigmoid):
      y = f(y_in) = 1 / (1 + exp(-y_in))
    - Activation derivative:
      f'(y_in) = y * (1 - y)
    - Gradient computation via chain rule:
      dE / dw_j = dE/dy * dy/dy_in * dy_in/dw_j
                = -(t - y) * f'(y_in) * x_j
    - Weight update equation (Gradient Descent):
      Delta w_j = -eta * (dE / dw_j) = eta * (t - y) * f'(y_in) * x_j
      Delta b   = eta * (t - y) * f'(y_in)

    Parameters
    ----------
    eta : float, default=0.001
        Learning rate parameter for LMS gradient descent.
    max_epochs : int, default=200
        Number of training epochs across the full training dataset.
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        eta: float = 0.001,
        max_epochs: int = 200,
        random_state: int = 42,
    ):
        self.eta = float(eta)
        self.max_epochs = int(max_epochs)
        self.random_state = int(random_state)
        self.classes_ = np.array([0, 1], dtype=np.int64)

        # Learned parameters
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.mse_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeltaRuleADALINE":
        """
        Fit ADALINE using continuous Widrow-Hoff LMS gradient descent on scaled inputs.

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Standardized input features (e.g. X_train).
        y : np.ndarray of shape (N,)
            Target class labels in {0, 1}.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.w_ = np.zeros(n_features, dtype=np.float64)
        self.b_ = 0.0
        self.mse_history_ = []

        logger.info(
            "Training DeltaRuleADALINE (Exp 5): N=%d, D=%d, eta=%.4f, max_epochs=%d",
            n_samples, n_features, self.eta, self.max_epochs
        )

        for epoch in range(1, self.max_epochs + 1):
            epoch_loss = 0.0

            # Online continuous delta rule updates
            for i in range(n_samples):
                xi = X[i]
                ti = y[i]

                # 1. Forward linear combination (net input)
                y_in = np.dot(xi, self.w_) + self.b_

                # 2. Continuous sigmoid activation
                y_pred = sigmoid(np.array([y_in]))[0]

                # 3. Continuous error signal
                err = ti - y_pred

                # 4. Derivative of sigmoid: f'(y_in) = y * (1 - y)
                f_prime = y_pred * (1.0 - y_pred)

                # 5. Continuous Delta Rule: delta = (t - y) * f'(y_in)
                delta = err * f_prime

                # 6. Weight and bias updates via gradient descent
                self.w_ += self.eta * delta * xi
                self.b_ += self.eta * delta

                epoch_loss += 0.5 * (err ** 2)

            mean_mse = epoch_loss / n_samples
            self.mse_history_.append(mean_mse)

        logger.info(
            "ADALINE training finished. Initial MSE: %.4f -> Final MSE: %.4f (Epoch %d)",
            self.mse_history_[0], self.mse_history_[-1], len(self.mse_history_)
        )
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute linear net input: y_in = X @ w + b.
        """
        if self.w_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.w_) + self.b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous output probabilities via logistic sigmoid activation:
        y = sigmoid(y_in) = 1 / (1 + exp(-y_in))
        Returns 2D array [P(no heatwave), P(heatwave)].
        """
        y_in = self.decision_function(X)
        p1 = sigmoid(y_in)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary heatwave occurrence {0, 1} at probability threshold.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(np.int64)


if __name__ == "__main__":
    from src.split import get_splits
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

    print("--- Verifying KJSCE Soft Computing Baselines ---")
    splits = get_splits()
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    val_df = splits["val_df"]

    # 1. McCulloch-Pitts Neuron (Raw Features)
    mp = McCullochPittsNeuron()
    mp.fit()
    mp_preds = mp.predict(val_df)
    mp_probs = mp.predict_proba(val_df)[:, 1]
    print(f"McCulloch-Pitts Val: Acc={accuracy_score(y_val, mp_preds):.4f}, F1={f1_score(y_val, mp_preds):.4f}, PR-AUC={average_precision_score(y_val, mp_probs):.4f}")

    # 2. Perceptron Rule (Scaled Features)
    perc = PerceptronRuleClassifier(eta=0.01, max_epochs=100)
    perc.fit(X_train, y_train)
    perc_preds = perc.predict(X_val)
    perc_probs = perc.predict_proba(X_val)[:, 1]
    print(f"Perceptron Val:      Acc={accuracy_score(y_val, perc_preds):.4f}, F1={f1_score(y_val, perc_preds):.4f}, ROC-AUC={roc_auc_score(y_val, perc_probs):.4f}")

    # 3. ADALINE Delta Rule (Scaled Features)
    adaline = DeltaRuleADALINE(eta=0.001, max_epochs=200)
    adaline.fit(X_train, y_train)
    ada_preds = adaline.predict(X_val)
    ada_probs = adaline.predict_proba(X_val)[:, 1]
    print(f"ADALINE Val:         Acc={accuracy_score(y_val, ada_preds):.4f}, F1={f1_score(y_val, ada_preds):.4f}, ROC-AUC={roc_auc_score(y_val, ada_probs):.4f}")
