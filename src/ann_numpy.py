"""A one-hidden-layer NumPy ANN for lab Experiments 1, 6 and 7 (CO1, CO2).

The textbook objective is mean half-squared error. Binary cross-entropy is an
explicit applied extension. Both forward computation and all weight/bias
gradients are written here; no automatic differentiation is used for training.
Learning rate is eta (η), momentum is mu (μ). A sigmoid output is a score;
calibration must be assessed separately before interpreting it as a probability.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EBPANetwork:
    """Input → one hidden layer → binary sigmoid output.

    ``loss='half_squared_error'`` is the Experiment 7 reference. For BCE, the
    sigmoid derivative cancels algebraically with the derivative of log loss.
    ``pos_weight`` multiplies positive-example losses AND their gradients.
    ``grad_*`` values are conventional derivatives; ``delta_*`` are the
    textbook negative derivatives (error signals). Updates subtract gradients.
    """

    SUPPORTED_ACTIVATIONS = ('binary_sigmoid', 'bipolar_sigmoid', 'relu')
    SUPPORTED_LOSSES = ('half_squared_error', 'binary_cross_entropy')
    LOSS_ALIASES = {'mse': 'half_squared_error', 'bce': 'binary_cross_entropy'}

    def __init__(self, n_input: int, n_hidden: int = 16,
                 activation: str = 'bipolar_sigmoid', learning_rate: float = 0.01,
                 momentum: float = 0.9, seed: int = 42,
                 loss: str = 'half_squared_error'):
        loss = self.LOSS_ALIASES.get(loss, loss)
        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(f'Unsupported activation: {activation}')
        if loss not in self.SUPPORTED_LOSSES:
            raise ValueError(f'Unsupported loss: {loss}')
        if n_input < 1 or n_hidden < 1 or learning_rate <= 0 or not 0 <= momentum < 1:
            raise ValueError('Positive layer sizes/learning rate and 0 <= momentum < 1 required')
        self.n_input, self.n_hidden, self.n_output = n_input, n_hidden, 1
        self.activation, self.loss = activation, loss
        self.learning_rate, self.momentum, self.seed = learning_rate, momentum, seed
        self.legacy_inconsistent_loss_reporting = False
        self._init_weights()
        self._reset_momentum()
        self.z_in = self.z = self.y_in = self.y_pred = None
        self.best_weights = None
        self.history = {'train_loss': [], 'val_loss': [], 'epoch': []}

    def _init_weights(self):
        rng = np.random.RandomState(self.seed)
        if self.activation == 'relu':
            self.V = rng.randn(self.n_input, self.n_hidden) * np.sqrt(2 / self.n_input)
            self.W = rng.randn(self.n_hidden, 1) * np.sqrt(2 / self.n_hidden)
        else:
            v_limit = np.sqrt(6 / (self.n_input + self.n_hidden))
            w_limit = np.sqrt(6 / (self.n_hidden + 1))
            self.V = rng.uniform(-v_limit, v_limit, (self.n_input, self.n_hidden))
            self.W = rng.uniform(-w_limit, w_limit, (self.n_hidden, 1))
        self.v0 = np.zeros((1, self.n_hidden), dtype=np.float64)
        self.w0 = np.zeros((1, 1), dtype=np.float64)

    def _reset_momentum(self):
        for name in ('V', 'v0', 'W', 'w0'):
            setattr(self, f'delta_{name}_prev', np.zeros_like(getattr(self, name)))

    @staticmethod
    def _sigmoid(x):
        """Overflow-safe sigmoid without clipping the mathematical function."""
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        positive = x >= 0
        y[positive] = 1 / (1 + np.exp(-x[positive]))
        e = np.exp(x[~positive])
        y[~positive] = e / (1 + e)
        return y

    def _activation_fn(self, z_in):
        if self.activation == 'binary_sigmoid':
            return self._sigmoid(z_in)
        if self.activation == 'bipolar_sigmoid':
            return np.tanh(z_in / 2)
        return np.maximum(0, z_in)

    def _activation_derivative(self, z_in, z):
        if self.activation == 'binary_sigmoid':
            return z * (1 - z)
        if self.activation == 'bipolar_sigmoid':
            return 0.5 * (1 + z) * (1 - z)
        return (z_in > 0).astype(np.float64)

    def _inputs(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_input or len(X) == 0:
            raise ValueError(f'Expected nonempty (N, {self.n_input}) inputs')
        if not np.isfinite(X).all():
            raise ValueError('Inputs must be finite')
        return X

    @staticmethod
    def _targets(y, n):
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if len(y) != n or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError('One finite binary target per sample required')
        return y

    def forward(self, X):
        """Explicit sums, bias additions and activations (Experiments 6–7, CO2)."""
        X = self._inputs(X)
        self.z_in = X @ self.V + self.v0
        self.z = self._activation_fn(self.z_in)
        self.y_in = self.z @ self.W + self.w0
        self.y_pred = self._sigmoid(self.y_in)
        return self.y_pred

    def compute_loss(self, y_true, y_pred, pos_weight=1.0, eps=1e-15, *, logits=None):
        """Mean loss with positive weights; optional logits give stable exact BCE.

        Half-squared-error: mean(0.5 * a(t) * (y-t)^2).
        BCE: mean(-p*t*log(y) - (1-t)*log(1-y)), with a(t)=1+(p-1)t.
        For reporting BCE on saturated outputs, pass the forward-pass logits.
        """
        y = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
        t = self._targets(y_true, len(y))
        if not np.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError('pos_weight must be finite and positive')
        if self.loss == 'half_squared_error':
            return float(np.mean(0.5 * (1 + (pos_weight - 1) * t) * (y - t) ** 2))
        if logits is not None:
            z = np.asarray(logits, dtype=np.float64).reshape(y.shape)
            return float(np.mean(pos_weight * t * np.logaddexp(0, -z)
                                 + (1 - t) * np.logaddexp(0, z)))
        p = np.clip(y, eps, 1 - eps)
        return float(-np.mean(pos_weight * t * np.log(p) + (1 - t) * np.log1p(-p)))

    def loss_on_batch(self, X, y, pos_weight=1.0):
        prediction = self.forward(X)
        return self.compute_loss(y, prediction, pos_weight, logits=self.y_in)

    def gradients(self, X, y_true, pos_weight=1.0):
        """Return exact derivatives without changing parameters (Experiment 7).

        dL/ds = a(t)*(y-t)*y*(1-y) for half-squared-error;
        dL/ds = a(t)*y - p*t for BCE, where s is the output logit.
        Average over the batch once, after propagating per-example signals.
        """
        X = self._inputs(X)
        t = self._targets(y_true, len(X))
        if not np.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError('pos_weight must be finite and positive')
        y = self.forward(X)
        weight = 1 + (pos_weight - 1) * t
        if self.loss == 'half_squared_error':
            d_output = weight * (y - t) * y * (1 - y)
        else:
            d_output = weight * y - pos_weight * t
        d_hidden = (d_output @ self.W.T) * self._activation_derivative(self.z_in, self.z)
        return {
            'delta_k': -d_output, 'delta_j': -d_hidden,
            'grad_W': self.z.T @ d_output / len(X),
            'grad_w0': d_output.mean(axis=0, keepdims=True),
            'grad_V': X.T @ d_hidden / len(X),
            'grad_v0': d_hidden.mean(axis=0, keepdims=True),
        }

    def backward(self, X, y_true, y_pred=None, clip_value=5.0, pos_weight=1.0):
        """Apply Δθ = -η∂L/∂θ + μΔθ_previous and return unclipped derivatives.

        ``y_pred`` is retained for old callers; the forward pass is recomputed to
        prevent stale caches from yielding incorrect gradients. Momentum is μ,
        never the worksheet's α (which denotes η=0.25 in Experiment 7).
        """
        result = self.gradients(X, y_true, pos_weight)
        for name in ('V', 'v0', 'W', 'w0'):
            grad = result[f'grad_{name}']
            if clip_value is not None and clip_value > 0:
                grad = np.clip(grad, -clip_value, clip_value)
            change = -self.learning_rate * grad + self.momentum * getattr(self, f'delta_{name}_prev')
            setattr(self, name, getattr(self, name) + change)
            setattr(self, f'delta_{name}_prev', change)
        return result

    def train(self, X_train, y_train, X_val=None, y_val=None, max_epochs=500,
              patience=30, pos_weight=None, batch_size=32, clip_value=5.0, verbose=False):
        """Seeded mini-batch SGD, with early stopping on the declared same loss.

        ``pos_weight=None`` means 1.0. Class weighting is an explicit experiment,
        not an automatic prevalence correction. Validation must be the model
        selection period (2011–2014), not calibration/evaluation observations.
        This call starts fresh momentum buffers while retaining current weights.
        """
        X = self._inputs(X_train)
        y = self._targets(y_train, len(X))
        weight = 1.0 if pos_weight is None else float(pos_weight)
        if max_epochs < 1 or batch_size < 1 or patience < 1:
            raise ValueError('max_epochs, batch_size and patience must be positive')
        if (X_val is None) != (y_val is None):
            raise ValueError('Provide both validation inputs and targets')
        has_val = X_val is not None
        if has_val:
            X_v = self._inputs(X_val)
            y_v = self._targets(y_val, len(X_v))
        self._reset_momentum()
        self.history = {'train_loss': [], 'val_loss': [], 'epoch': []}
        best_loss, best_epoch, stalled = float('inf'), 0, 0
        self.best_weights = None
        rng = np.random.RandomState(self.seed)
        for epoch in range(max_epochs):
            indices = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = indices[start:start + batch_size]
                self.backward(X[idx], y[idx], clip_value=clip_value, pos_weight=weight)
            train_loss = self.loss_on_batch(X, y, weight)
            self.history['train_loss'].append(train_loss)
            self.history['epoch'].append(epoch)
            criterion = self.loss_on_batch(X_v, y_v, weight) if has_val else train_loss
            if has_val:
                self.history['val_loss'].append(criterion)
            if criterion < best_loss:
                best_loss, best_epoch, stalled = criterion, epoch, 0
                self.best_weights = {name: getattr(self, name).copy() for name in ('V', 'v0', 'W', 'w0')}
            else:
                stalled += 1
            if verbose and epoch % 20 == 0:
                logger.info('epoch=%d train_loss=%.6f selection_loss=%.6f', epoch, train_loss, criterion)
            if has_val and stalled >= patience:
                break
        if has_val:
            for name, value in self.best_weights.items():
                setattr(self, name, value.copy())
            self._reset_momentum()
        return {
            'best_epoch': best_epoch, 'epochs_trained': len(self.history['epoch']),
            'best_val_loss': best_loss,
            'final_train_loss': self.loss_on_batch(X, y, weight),
            'loss': self.loss, 'pos_weight': weight, 'history': self.history,
        }

    def predict_proba(self, X):
        """Uncalibrated sigmoid scores in [0,1]; name retained for compatibility."""
        return self.forward(X).ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def save_weights(self, filepath):
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        metadata = {name: getattr(self, name) for name in
                    ('n_input', 'n_hidden', 'n_output', 'activation', 'learning_rate', 'momentum', 'seed', 'loss')}
        metadata.update(format_version=2, output_semantics='uncalibrated_sigmoid_score',
                        legacy_inconsistent_loss_reporting=self.legacy_inconsistent_loss_reporting)
        weights = {name: getattr(self, name) for name in ('V', 'v0', 'W', 'w0')}
        if dest.suffix == '.npz':
            np.savez_compressed(dest, **metadata, **weights)
        else:
            dest.write_text(json.dumps({**metadata, **{k: v.tolist() for k, v in weights.items()}}, indent=2), encoding='utf-8')

    def load_weights(self, filepath):
        """Load v2 and legacy v1 weights; legacy math was squared-error updates.

        Files record inference weights, not an optimizer checkpoint. Loading
        resets momentum and invalidates caches, including when shapes change.
        """
        dest = Path(filepath)
        if dest.suffix == '.npz':
            with np.load(dest, allow_pickle=False) as source:
                data = {key: source[key].tolist() for key in source.files}
        else:
            data = json.loads(dest.read_text(encoding='utf-8'))
        self.__init__(n_input=int(data['n_input']), n_hidden=int(data['n_hidden']),
                      activation=data['activation'], learning_rate=float(data['learning_rate']),
                      momentum=float(data['momentum']), seed=int(data.get('seed', 42)),
                      loss=data.get('loss', 'half_squared_error'))
        for name in ('V', 'v0', 'W', 'w0'):
            value = np.asarray(data[name], dtype=np.float64)
            if value.shape != getattr(self, name).shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid {name} weights')
            setattr(self, name, value)
        self.legacy_inconsistent_loss_reporting = bool(data.get('legacy_inconsistent_loss_reporting', 'loss' not in data))
        self._reset_momentum()


def run_activation_study(config_path='config.yaml',
                         out_table_path='results/tables/exp1_activation_comparison.csv', seed=42):
    """Compatibility entry point; use the reproducible Heatline experiment runner.

    This small comparison uses only 2011–2014 for model selection, and does not
    overwrite scaler artifacts. ReLU is a supplementary activation, outside the
    supplied Experiment 1 list. Lab mapping: activation functions → CO1.
    """
    import pandas as pd
    import yaml
    from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
    from src.split import get_splits
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    splits = get_splits(config=config, save_scaler=False)
    select = splits['val_df']['year'].to_numpy() <= 2014
    X, y = splits['X_val'][select], splits['y_val'][select]
    records = []
    for activation in EBPANetwork.SUPPORTED_ACTIVATIONS:
        net = EBPANetwork(splits['X_train'].shape[1], activation=activation, seed=seed)
        info = net.train(splits['X_train'], splits['y_train'], X, y)
        score = net.predict_proba(X)
        records.append({'activation': activation, 'loss': net.loss,
                        'final_train_loss': info['final_train_loss'],
                        'val_loss': info['best_val_loss'], 'val_f1': f1_score(y, score >= 0.5, zero_division=0),
                        'val_pr_auc': average_precision_score(y, score),
                        'val_roc_auc': roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                        'epochs_trained': info['epochs_trained']})
    frame = pd.DataFrame(records)
    Path(out_table_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_table_path, index=False)
    return frame


if __name__ == '__main__':
    print(run_activation_study().to_string(index=False))
