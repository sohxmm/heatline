import pytest

from scripts.export_metric_context import classification_context


def test_precision_is_not_accuracy_and_negative_baseline_is_explicit():
    result = classification_context(dict(tp=5, fp=12, fn=3, tn=228))
    assert result["precision"] == pytest.approx(5 / 17)
    assert result["accuracy"] == pytest.approx(233 / 248)
    assert result["balanced_accuracy"] == pytest.approx(0.7875)
    assert result["always_negative"]["accuracy"] == pytest.approx(240 / 248)
    assert result["always_negative"]["recall"] == 0
    assert result["fair_coin"]["expected_accuracy"] == 0.5


def test_empty_confusion_counts_are_not_a_perfect_score():
    with pytest.raises(ValueError):
        classification_context(dict(tp=0, tn=0, fp=0, fn=0))
