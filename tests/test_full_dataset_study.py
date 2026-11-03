"""Held-out chronology, causal inputs and honest metric names for full coverage."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.full_dataset_study import (calibrated_scores, expected_coin,
    fold_plan, make_features, measured_metrics, source_feature_names)

FEATURES = source_feature_names('era5')


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
    assert len(FEATURES) == 26


def test_incoming_saved_labels_and_normals_are_ignored(daily):
    original = make_features(daily, reference_end=1991)
    daily['heatwave_day'] = 1-daily.heatwave_day
    daily['normal_tmax'] = -500
    changed = make_features(daily, reference_end=1991)
    np.testing.assert_array_equal(original[FEATURES+['heatwave_day', 'normal_tmax']], changed[FEATURES+['heatwave_day', 'normal_tmax']])


def test_missing_history_fails_instead_of_interpolating(daily):
    daily = daily[~daily.date.eq('1996-05-19')]
    with pytest.raises(ValueError, match='Missing.*history'):
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
    assert data['model_version'] == 'heatline-walk-forward-v2'
    for source in data['sources']:
        assert source['coverage']['total_seasonal_days'] == 4186
        assert source['coverage']['held_out_may_days'] == 31*31
        assert source['final_fit']['days'] == 4186
        assert source['final_fit']['mode'] == 'full-data refit'
        assert len(source['records']) == len({r['date'] for r in source['records']}) == 961
        for fold in source['folds']:
            assert fold['reference_years'][1] <= fold['fit_years'][1]
            assert fold['scaler_fit_years'] == fold['fit_years']


def test_nasa_keeps_its_own_52_input_representation():
    daily = pd.read_csv('data/processed/nasa_daily.csv')
    frame = make_features(daily, reference_end=1991, source='nasa_power')
    names = source_feature_names('nasa_power')
    assert len(names) == 52 and len(frame) == 4186
    assert np.isfinite(frame[names]).all().all()
    assert names != source_feature_names('era5')


def test_current_forward_trace_reproduces_held_out_prediction():
    from src.ann_numpy import EBPANetwork
    from src.experiment import logit
    for source, width in [('era5', 26), ('nasa_power', 52)]:
        path = Path(f'artifacts/heatline/study/{source}/2026.json')
        if not path.exists():
            pytest.skip('Regenerate the v2 study first')
        trace = json.loads(path.read_text())
        assert trace['model_version'] == 'heatline-walk-forward-v2'
        assert trace['source_id'] == source and trace['selected_year'] == 2026
        assert trace['architecture']['inputs'] == width
        day = trace['records'][0]
        inputs = np.array([day['inputs'][name] for name in trace['feature_names']])
        model = trace['fold_model']
        expected_x = (inputs-np.array(model['scaler_mean'])) / np.array(model['scaler_scale'])
        actual_x = np.array([day['normalized_inputs'][name] for name in trace['feature_names']])
        np.testing.assert_allclose(actual_x, expected_x, rtol=1e-12)
        scores = []
        for weights, observed in zip(model['members'], day['members']):
            net = EBPANetwork(width, 16, activation='bipolar_sigmoid', seed=weights['seed'])
            for key in ['V', 'v0', 'W', 'w0']:
                setattr(net, key, np.array(weights[key]))
            score = float(net.forward(actual_x)[0, 0])
            assert observed['score'] == pytest.approx(score, abs=1e-14)
            np.testing.assert_allclose(observed['hidden_activations'], net.z[0], atol=1e-14)
            scores.append(score)
        raw = float(np.mean(scores))
        calibration = day['calibration']
        expected_probability = raw if calibration is None else 1/(1+np.exp(-(float(logit([raw])[0, 0])*calibration['coefficient']+calibration['intercept'])))
        assert day['score'] == pytest.approx(raw, abs=1e-14)
        assert day['probability'] == pytest.approx(expected_probability, abs=1e-14)
        assert day['prediction'] == int(expected_probability >= day['threshold'])


def test_learning_curves_describe_actual_member_histories():
    path = Path('artifacts/heatline/full_dataset_study.json')
    data = json.loads(path.read_text())
    if data['model_version'] != 'heatline-walk-forward-v2':
        pytest.skip('Regenerate v2 first')
    for source in data['sources']:
        width = 26 if source['id'] == 'era5' else 52
        assert source['architecture']['inputs'] == len(source['feature_names']) == width
        assert source['initialization']['input_hidden_limit'] == pytest.approx(np.sqrt(6/(width+16)))
        for group in [*source['folds'], source['final_fit']]:
            members = group['members']
            history = group['training_history']
            assert history[0]['epoch'] == 1
            assert len(history) == min(len(member['training_history']) for member in members)
            assert history[0]['train_loss'] == pytest.approx(np.mean([member['training_history'][0]['train_loss'] for member in members]))
        assert 'val_loss' not in source['final_fit']['training_history'][0]


def test_common_input_predecessor_is_preserved_as_a_scientific_archive():
    from zipfile import ZipFile
    with ZipFile('legacy/common-input-study-v1.zip') as archive:
        data = json.loads(archive.read('artifacts/heatline/full_dataset_study.json'))
        assert data['model_version'] == 'heatline-walk-forward-v1'
        assert all(source['architecture']['inputs'] == 52 for source in data['sources'])
        assert b"VERSION = 'heatline-walk-forward-v1'" in archive.read('src/full_dataset_study.py')


def test_partitioned_traces_match_source_year_and_published_decisions():
    path = Path('artifacts/heatline/full_dataset_study.json')
    data = json.loads(path.read_text())
    if data['model_version'] != 'heatline-walk-forward-v2':
        pytest.skip('Regenerate v2 first')
    assert path.read_bytes() == Path('frontend/public/data/full_dataset_study.json').read_bytes()
    for source in data['sources']:
        expected = {record['date']: record for record in source['records']}
        dates = []
        for year in source['trace_years']:
            filename = Path(f"study/{source['id']}/{year}.json")
            trace_path = Path('artifacts/heatline')/filename
            trace = json.loads(trace_path.read_text())
            assert trace_path.read_bytes() == (Path('frontend/public/data')/filename).read_bytes()
            assert trace['model_version'] == data['model_version']
            assert trace['source_id'] == source['id'] and trace['selected_year'] == year
            assert trace['feature_names'] == source['feature_names']
            assert trace['fold_model']['fit_years'][1] < year
            assert len(trace['records']) == 31
            for record in trace['records']:
                date = record['date']; dates.append(date)
                assert date.startswith(f'{year}-05-')
                assert record['prediction'] == expected[date]['prediction']
                assert record['actual'] == expected[date]['actual']
                assert record['probability'] == pytest.approx(expected[date]['probability'], abs=1e-14)
                assert record['threshold_margin'] == pytest.approx(record['probability']-record['threshold'])
        assert len(dates) == len(set(dates)) == len(expected)
