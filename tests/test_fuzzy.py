"""
Unit tests for Mamdani Fuzzy Logic Controller (CO4).
"""

import numpy as np
import pytest
from src.fuzzy_controller import MamdaniFuzzyController, trimf, trapmf


def test_membership_functions():
    """Verify triangular and trapezoidal membership bounds."""
    # Triangular at peak
    assert trimf(41.5, (39.0, 41.5, 43.5)) == pytest.approx(1.0)
    assert trimf(38.0, (39.0, 41.5, 43.5)) == pytest.approx(0.0)
    assert trimf(44.0, (39.0, 41.5, 43.5)) == pytest.approx(0.0)

    # Trapezoid shoulder
    assert trapmf(35.0, (-100.0, -100.0, 38.0, 40.5)) == pytest.approx(1.0)
    assert trapmf(45.0, (42.0, 44.0, 100.0, 100.0)) == pytest.approx(1.0)


def test_fuzzy_controller_extremes():
    """Verify extreme cold/humid and extreme hot/dry inputs map correctly."""
    controller = MamdaniFuzzyController()

    # Extreme heatwave condition: Tmax=46C, Anomaly=+5C, RH_min=8%
    score_hot, prob_hot = controller.evaluate_sample(46.0, 5.0, 8.0)
    assert prob_hot > 0.70, f"Expected high risk for extreme heat, got {prob_hot}"

    # Cool benign condition: Tmax=34C, Anomaly=-3C, RH_min=60%
    score_cool, prob_cool = controller.evaluate_sample(34.0, -3.0, 60.0)
    assert prob_cool < 0.35, f"Expected low risk for cool weather, got {prob_cool}"


def test_fuzzy_controller_monotonicity():
    """Higher temperatures with lower humidity should yield higher risk scores."""
    controller = MamdaniFuzzyController()
    _, p1 = controller.evaluate_sample(39.0, 0.5, 40.0)
    _, p2 = controller.evaluate_sample(42.5, 2.5, 20.0)
    _, p3 = controller.evaluate_sample(45.0, 4.5, 10.0)

    assert p1 < p2 < p3
