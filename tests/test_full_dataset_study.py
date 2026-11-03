"""Held-out chronology, causal inputs and honest metric names for full coverage."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.full_dataset_study import (FEATURES, calibrated_scores, expected_coin,
    fold_plan, make_features, measured_metrics)


@pytest.fixture
def daily():
    frame = pd.read_csv('data/processed/seasonal_daily.csv')
    return frame[frame.year.le(1997)].copy()


def test_whole_year_folds_are_disjoint_and_strictly_forward():
    evaluated = []
    for fold in fold_plan():
        fit, selection, calibration, test = [fold[f'{part}_years']
            for part in ['fit', 'selection', 'calibration', 'evaluation']]
        assert fit[0] == 1981
        assert fit[1] < selection[0] <= selection[1] < calibration[0] <= calibration[1] < test[0]
        evaluated.extend(range(test[0], test[1]+1))
    assert evaluated == list(range(1996, 2027))
    assert len(evaluated) == len(set(evaluated))


def test_future_weather_cannot_change_reference_or_earlier_features(daily):
    original = make_features(daily, reference_end=1991)
    changed = daily.copy()
    changed.loc[changed.year.ge(1996), ['tmax', 'tmin', 'tmean', 'rh_mean']] += 100
    result = make_features(changed, reference_end=1991)
    earlier = original.year.lt(1996)
    np.testing.assert_array_equal(original.loc[earlier, FEATURES], result.loc[earlier, FEATURES])
    np.testing.assert_array_equal(original.normal_tmax, result.normal_tmax)
    np.testing.assert_array_equal(original.loc[earlier, 'heatwave_day'], result.loc[earlier, 'heatwave_day'])


def test_target_day_weather_is_not_an_input_to_its_prediction(daily):
    original = make_features(daily, reference_end=1991)
    changed = daily.copy()
    date = '1996-05-20'
    changed.loc[changed.date.eq(date), ['tmax', 'tmin', 'tmean', 'rh_mean', 'wind_mean', 'pressure_mean']] += 100
    result = make_features(changed, reference_end=1991)
    np.testing.assert_array_equal(original.loc[original.date.eq(date), FEATURES], result.loc[result.date.eq(date), FEATURES])
    assert len(FEATURES) == 52


def test_incoming_saved_labels_and_normals_are_ignored(daily):
    original = make_features(daily, reference_end=1991)
    daily['heatwave_day'] = 1-daily.heatwave_day
    daily['normal_tmax'] = -500
    changed = make_features(daily, reference_end=1991)
    np.testing.assert_array_equal(original[FEATURES+['heatwave_day', 'normal_tmax']], changed[FEATURES+['heatwave_day', 'normal_tmax']])


def test_missing_history_fails_instead_of_interpolating(daily):
    daily = daily[~daily.date.eq('1996-05-19')]
    with pytest.raises(ValueError, match='Missing weather history'):
        make_features(daily, reference_end=1991)


def test_accuracy_is_not_alert_precision():
    # The published v2 counts: 228 TN, 12 FP, 3 FN, 5 TP.
    actual = np.r_[np.zeros(240), np.ones(8)]
    predicted = np.r_[np.zeros(228), np.ones(12), np.zeros(3), np.ones(5)]
    result = measured_metrics(actual, predicted, predicted)
    assert result['accuracy'] == pytest.approx(233/248)
    assert result['precision'] == pytest.approx(5/17)
    assert result['recall'] == pytest.approx(5/8)
    assert result['accuracy'] > .9 and result['precision'] < .3


def test_expected_coin_uses_prevalence_and_no_lucky_seed():
    y = np.r_[np.zeros(90), np.ones(10)]
    result = expected_coin(y)
    assert result['accuracy'] == result['balanced_accuracy'] == .5
    assert result['precision'] == result['pr_auc'] == .1
    assert result['brier'] == .25
    assert result['tp'] == result['fn'] == 5
    assert result['tn'] == result['fp'] == 45


def test_one_class_calibration_does_not_borrow_test_labels():
    calibration = np.array([.1, .2, .3])
    target = np.array([.5, .7])
    cal, score, metadata = calibrated_scores(np.zeros(3), calibration, target)
    np.testing.assert_array_equal(cal, calibration)
    np.testing.assert_array_equal(score, target)
    assert metadata is None


def test_saved_full_refit_is_separate_from_held_out_evidence():
    path = Path('artifacts/heatline/full_dataset_study.json')
    if not path.exists():
        pytest.skip('Run python -m src.full_dataset_study to build research artifacts')
    data = json.loads(path.read_text())
    assert data['model_version'] == 'heatline-walk-forward-v1'
    for source in data['sources']:
        assert source['coverage']['total_seasonal_days'] == 4186
        assert source['coverage']['held_out_may_days'] == 31*31
        assert source['final_fit']['days'] == 4186
        assert source['final_fit']['mode'] == 'full-data refit'
        assert len(source['records']) == len({r['date'] for r in source['records']}) == 961
        for fold in source['folds']:
            assert fold['reference_years'][1] <= fold['fit_years'][1]
            assert fold['scaler_fit_years'] == fold['fit_years']
