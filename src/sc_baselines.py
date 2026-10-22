"""Foundational neural comparators for the assessed ANN.

Lab mapping: McCulloch–Pitts = Experiment 2 / CO1; perceptron = Experiment 4 /
CO2; sigmoid delta learning = Experiment 5 / CO2. Sigmoid delta learning is
nonlinear squared-error gradient descent, NOT linear LMS/ADALINE. Sigmoid
scores returned by the sklearn-style interface are not automatically calibrated.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from src.features import FEATURE_COLS


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -z))


class McCullochPittsNeuron(BaseEstimator, ClassifierMixin):
    """Legacy fixed-threshold weather illustration (Experiment 2, CO1).

    These handcrafted thresholds are an illustrative neuron, not a validated
    physical model or official heatwave criterion. NAND/NOR and a memory cell
    are demonstrated separately in src.lab_examples with explicit weights.
    Raw, unscaled inputs are required.
    """
    def __init__(self, tmax_threshold=42.0, anomaly_threshold=3.5,
                 rh_min_threshold=25.0, wind_threshold=15.0,
                 theta=2.0, inhibitory_weight=2.0):
        self.tmax_threshold = tmax_threshold
        self.anomaly_threshold = anomaly_threshold
        self.rh_min_threshold = rh_min_threshold
        self.wind_threshold = wind_threshold
        self.theta = theta
        self.inhibitory_weight = inhibitory_weight
        self.classes_ = np.array([0, 1])

    def fit(self, X=None, y=None):
        return self

    def _extract_raw_features(self, X):
        names = ('tmax_lag1', 'tmax_anomaly_lag1', 'rh_min_lag1', 'wind_mean_lag1')
        if isinstance(X, (pd.DataFrame, dict)):
            return tuple(np.atleast_1d(np.asarray(X[name], dtype=np.float64)) for name in names)
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] == 4:
            return tuple(X[:, i] for i in range(4))
        if X.shape[1] == len(FEATURE_COLS):
            return tuple(X[:, FEATURE_COLS.index(name)] for name in names)
        raise ValueError('Expected four raw weather features or the full feature matrix')

    def compute_net_input(self, X):
        tmax, anomaly, rh_min, wind = self._extract_raw_features(X)
        return ((tmax >= self.tmax_threshold).astype(float)
                + (anomaly >= self.anomaly_threshold).astype(float)
                + (rh_min <= self.rh_min_threshold).astype(float)
                - self.inhibitory_weight * (wind >= self.wind_threshold))

    def decision_function(self, X):
        return self.compute_net_input(X) - self.theta

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    def predict_proba(self, X):
        """Uncalibrated ranking score obtained from distance to fixed threshold."""
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])


class PerceptronRuleClassifier(BaseEstimator, ClassifierMixin):
    """Bipolar-step perceptron, Experiment 4 / CO2.

    For an incorrect output, Δw = ηtx and Δb = ηt, t ∈ {-1,+1}.
    XOR demonstrates nonseparability. Any advantage of a multilayer model on
    weather data must be measured; a failed run is not a proof of nonseparability.
    ``fit`` takes binary 0/1 labels and maps them internally to bipolar targets.
    """
    def __init__(self, eta=0.01, max_epochs=100, random_state=42):
        self.eta, self.max_epochs, self.random_state = eta, max_epochs, random_state
        self.classes_ = np.array([0, 1])
        self.w_, self.b_ = None, 0.0
        self.errors_per_epoch_, self.accuracies_per_epoch_ = [], []

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y).ravel()
        if len(X) == 0 or len(X) != len(y) or not np.isin(y, [0, 1]).all():
            raise ValueError('Nonempty matching inputs and binary targets required')
        t = 2 * y - 1
        self.w_, self.b_ = np.zeros(X.shape[1]), 0.0
        self.errors_per_epoch_, self.accuracies_per_epoch_ = [], []
        # Fixed presentation order is deterministic; no random initialization.
        for _ in range(self.max_epochs):
            errors = 0
            for xi, ti in zip(X, t):
                prediction = 1 if xi @ self.w_ + self.b_ >= 0 else -1
                if prediction != ti:
                    self.w_ += self.eta * ti * xi
                    self.b_ += self.eta * ti
                    errors += 1
            self.errors_per_epoch_.append(errors)
            # Evaluate the final epoch weights, not online pre-update mistakes.
            self.accuracies_per_epoch_.append(float(np.mean(self.predict(X) == y)))
            if errors == 0:
                break
        return self

    def decision_function(self, X):
        if self.w_ is None:
            raise ValueError('Call fit before prediction')
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict_proba(self, X):
        """Uncalibrated sigmoid of the perceptron's linear decision score."""
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class SigmoidDeltaRuleClassifier(BaseEstimator, ClassifierMixin):
    """Single sigmoid neuron trained with the delta rule (Experiment 5, CO2).

    E = 1/2(t-y)^2; Δw = η(t-y)f'(net)x; Δb = η(t-y)f'(net).
    Binary sigmoid uses t∈{0,1} and f'=y(1-y). Bipolar sigmoid uses t∈{-1,+1}
    and f'=0.5(1+y)(1-y). The public fit API accepts binary labels for both.
    Linear LMS/ADALINE instead uses y=net and f'=1, giving a quadratic loss in
    weights. The sigmoid objective here is not quadratic and has no general
    guarantee of convergence to a global minimum.
    """
    def __init__(self, eta=0.001, max_epochs=200, random_state=42,
                 activation='binary_sigmoid'):
        self.eta, self.max_epochs, self.random_state = eta, max_epochs, random_state
        self.activation = activation
        self.classes_ = np.array([0, 1])
        self.w_, self.b_ = None, 0.0
        self.mse_history_ = []

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
        if self.activation not in ('binary_sigmoid', 'bipolar_sigmoid'):
            raise ValueError('Choose binary_sigmoid or bipolar_sigmoid')
        if len(X) == 0 or len(X) != len(y) or not np.isin(y, [0, 1]).all():
            raise ValueError('Nonempty matching inputs and binary targets required')
        targets = y if self.activation == 'binary_sigmoid' else 2 * y - 1
        self.w_, self.b_, self.mse_history_ = np.zeros(X.shape[1]), 0.0, []
        for _ in range(self.max_epochs):
            for xi, ti in zip(X, targets):
                net = float(xi @ self.w_ + self.b_)
                if self.activation == 'binary_sigmoid':
                    prediction = float(sigmoid(net))
                    derivative = prediction * (1 - prediction)
                else:
                    prediction = float(np.tanh(net / 2))
                    derivative = 0.5 * (1 + prediction) * (1 - prediction)
                delta = (ti - prediction) * derivative
                self.w_ += self.eta * delta * xi
                self.b_ += self.eta * delta
            net = X @ self.w_ + self.b_
            output = sigmoid(net) if self.activation == 'binary_sigmoid' else np.tanh(net / 2)
            self.mse_history_.append(float(np.mean(0.5 * (targets - output) ** 2)))
        return self

    def decision_function(self, X):
        if self.w_ is None:
            raise ValueError('Call fit before prediction')
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict_proba(self, X):
        """Uncalibrated score; bipolar output mapped from [-1,1] to [0,1]."""
        # (tanh(net/2)+1)/2 = sigmoid(net), so both conventions have this score.
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# Compatibility for archived notebooks/imports; the old name was mathematically
# incorrect. New reports and code must use SigmoidDeltaRuleClassifier.
DeltaRuleADALINE = SigmoidDeltaRuleClassifier
