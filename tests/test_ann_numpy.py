"""
Unit Tests for Pure-NumPy Error Backpropagation Artificial Neural Network (EBPA).
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.ann_numpy import EBPANetwork


def test_ebpa_initialization():
    """Verify weight shapes and initialization bounds."""
    n_in = 26
    n_hid = 32
    net = EBPANetwork(n_input=n_in, n_hidden=n_hid, activation="bipolar_sigmoid", seed=42)

    assert net.V.shape == (n_in, n_hid)
    assert net.v0.shape == (1, n_hid)
    assert net.W.shape == (n_hid, 1)
    assert net.w0.shape == (1, 1)

    # Biases should be initialized to zero
    np.testing.assert_array_equal(net.v0, np.zeros((1, n_hid)))
    np.testing.assert_array_equal(net.w0, np.zeros((1, 1)))


def test_forward_activations_and_shapes():
    """Test forward pass shapes and mathematical bounds for all three activations."""
    X = np.random.randn(10, 26)

    for act in ["binary_sigmoid", "bipolar_sigmoid", "relu"]:
        net = EBPANetwork(n_input=26, n_hidden=32, activation=act, seed=42)
        y_pred = net.forward(X)

        assert y_pred.shape == (10, 1)
        # Output should always be in (0, 1) due to output binary sigmoid
        assert np.all((y_pred >= 0.0) & (y_pred <= 1.0))

        # Check hidden activation ranges
        if act == "binary_sigmoid":
            assert np.all((net.z >= 0.0) & (net.z <= 1.0))
        elif act == "bipolar_sigmoid":
            assert np.all((net.z >= -1.0) & (net.z <= 1.0))
        elif act == "relu":
            assert np.all(net.z >= 0.0)


def test_backward_and_weight_updates():
    """Verify that backward pass computes gradients and updates weights with momentum."""
    X = np.random.randn(8, 26)
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0]).reshape(-1, 1)

    net = EBPANetwork(n_input=26, n_hidden=32, activation="bipolar_sigmoid", seed=42)
    V_initial = net.V.copy()
    W_initial = net.W.copy()

    y_pred = net.forward(X)
    grad_dict = net.backward(X, y, y_pred)

    assert "delta_k" in grad_dict
    assert "delta_j" in grad_dict
    assert grad_dict["delta_k"].shape == (8, 1)
    assert grad_dict["delta_j"].shape == (8, 32)

    # Weights must change after backward pass
    assert not np.allclose(net.V, V_initial)
    assert not np.allclose(net.W, W_initial)


def test_ebpa_synthetic_learning():
    """Verify that network reduces loss over multiple training steps on synthetic data."""
    np.random.seed(42)
    X = np.random.randn(100, 26)
    # Simple linear threshold target
    y = (X[:, 0] + X[:, 1] > 0.0).astype(int)

    net = EBPANetwork(n_input=26, n_hidden=16, activation="bipolar_sigmoid", learning_rate=0.05, seed=42)
    history = net.train(X, y, X, y, max_epochs=40, patience=20, batch_size=25)

    assert len(history["history"]["train_loss"]) > 1
    # Train loss should decrease from epoch 0
    assert history["history"]["train_loss"][-1] < history["history"]["train_loss"][0]


def test_predict_and_predict_proba():
    """Verify prediction functions format and thresholding behavior."""
    net = EBPANetwork(n_input=26, n_hidden=16, seed=42)
    X = np.random.randn(5, 26)

    probas = net.predict_proba(X)
    assert probas.shape == (5,)
    assert np.all((probas >= 0.0) & (probas <= 1.0))

    preds = net.predict(X, threshold=0.5)
    assert preds.shape == (5,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_save_and_load_weights(tmp_path):
    """Verify weight serialization and deserialization."""
    net = EBPANetwork(n_input=26, n_hidden=16, activation="relu", seed=42)
    X = np.random.randn(5, 26)
    orig_proba = net.predict_proba(X)

    save_path = tmp_path / "model_test.json"
    net.save_weights(str(save_path))

    net_loaded = EBPANetwork(n_input=26, n_hidden=16, activation="relu", seed=999)
    net_loaded.load_weights(str(save_path))

    loaded_proba = net_loaded.predict_proba(X)
    np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-12)
