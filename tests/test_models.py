"""
Comprehensive Model Integration & Inference Equivalence Tests.
KJSCE Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP).
"""

from pathlib import Path
import json
import numpy as np
import pytest
import yaml

from src.split import get_splits
from src.export_model import NumpyMLPEnsemble
from src.ga_optimizer import GeneticAlgorithmOptimizer


def test_numpy_mlp_ensemble_inference():
    """Verify that pure-NumPy MLP ensemble loads and produces valid probabilities in [0, 1]."""
    weights_path = Path("models/ensemble_weights.npz")
    assert weights_path.exists(), "ensemble_weights.npz must exist"

    ensemble = NumpyMLPEnsemble(str(weights_path))
    assert ensemble.n_seeds == 10
    assert len(ensemble.feature_names) == 26

    # Test dummy single sample
    x_dummy = np.zeros(26)
    p_dummy = ensemble.predict_proba(x_dummy)
    assert 0.0 <= p_dummy[0] <= 1.0

    # Test batch predictions
    X_batch = np.random.randn(15, 26)
    p_batch = ensemble.predict_proba(X_batch)
    assert len(p_batch) == 15
    assert np.all((p_batch >= 0.0) & (p_batch <= 1.0))


def test_ga_threshold_file():
    """Verify threshold.json exists and contains valid blending weights and threshold."""
    thresh_path = Path("models/threshold.json")
    assert thresh_path.exists(), "threshold.json must exist"

    with open(thresh_path, "r") as f:
        data = json.load(f)

    assert "w_ann" in data
    assert "w_ebpa" in data
    assert "w_fuzzy" in data
    assert "threshold" in data

    total_weight = data["w_ann"] + data["w_ebpa"] + data["w_fuzzy"]
    assert total_weight == pytest.approx(1.0, rel=1e-3)
    assert 0.45 <= data["threshold"] <= 0.80


def test_ga_optimizer_chromosome_decoding():
    """Verify chromosome decoding respects weight normalization and threshold range."""
    ga = GeneticAlgorithmOptimizer()
    chrom = np.array([2.0, 1.0, 1.0, 0.55])
    w1, w2, w3, tau = ga.decode_chromosome(chrom)

    assert w1 == pytest.approx(0.50)
    assert w2 == pytest.approx(0.25)
    assert w3 == pytest.approx(0.25)
    assert tau == pytest.approx(0.55)
