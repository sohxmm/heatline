"""The primary lab workflow uses every row and labels its scores as in-sample."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from api.main import create_app
from src.ann_numpy import EBPANetwork
from src.full_dataset_fit import FILES, VERSION, fit_score_transform
from src.full_dataset_study import make_features, measured_metrics
from src.experiment import logit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def fit():
    return json.loads((ROOT/'artifacts/heatline/full_dataset_fit.json').read_text(encoding='utf-8'))


def test_no_rows_reserved_for_another_stage(fit):
    assert fit['model_version'] == VERSION
    assert fit['evaluation_kind'] == 'in-sample'
    assert fit['protocol']['no_partitions'] is True
    assert not fit['protocol']['hyperparameters']['early_stopping']
    for source in fit['sources']:
        run = source['final_fit']
        assert run['no_partitions'] and not source['folds']
        assert {run[k] for k in ['preprocessing_rows', 'training_rows', 'calibration_rows', 'threshold_rows', 'checking_rows']} == {4186}
        assert run['epochs'] == 300
        assert len(source['records']) == 1426
        assert run['in_sample_metrics']['days'] == 1426
        assert run['all_season_metrics']['days'] == 4186
        for member in run['members']:
            assert len(member['training_history']) == 300
            assert all('val_loss' not in h for h in member['training_history'])


@pytest.mark.parametrize('source_id,width', [('era5', 26), ('nasa_power', 52)])
def test_all_data_scaler_calibration_threshold_and_scores_reproduce(fit, source_id, width):
    source = next(s for s in fit['sources'] if s['id'] == source_id)
    frame = make_features(pd.read_csv(FILES[source_id]), 2010, source_id)
    names = source['feature_names']; assert len(names) == width
    scaler = StandardScaler().fit(frame[names]); X = scaler.transform(frame[names])
    metadata = json.loads((ROOT/f'artifacts/full-dataset-fit/{source_id}_preprocessing.json').read_text())
    np.testing.assert_allclose(metadata['mean'], scaler.mean_, atol=1e-14)
    np.testing.assert_allclose(metadata['scale'], scaler.scale_, atol=1e-14)
    scores = []
    for seed in [7, 21, 42]:
        net = EBPANetwork(width, 16)
        net.load_weights(ROOT/f'artifacts/full-dataset-fit/{source_id}_seed{seed}.npz')
        scores.append(net.predict_proba(X))
    raw = np.mean(scores, axis=0)
    p, calibration, threshold = fit_score_transform(frame.heatwave_day.to_numpy(), raw)
    assert metadata['calibration'] == calibration
    assert metadata['threshold'] == threshold
    may = frame.month.eq(5).to_numpy()
    metrics = measured_metrics(frame.heatwave_day.to_numpy()[may], p[may], p[may] >= threshold)
    assert {k: source['metrics'][0][k] for k in metrics} == metrics
    np.testing.assert_allclose([r['probability'] for r in source['records']], p[may], atol=1e-14)


@pytest.mark.parametrize('source_id', ['era5', 'nasa_power'])
def test_every_calendar_trace_matches_source_version_and_saved_weights(fit, source_id):
    source = next(s for s in fit['sources'] if s['id'] == source_id)
    expected = {r['date']: r for r in source['records']}
    seen = []
    for year in range(1981, 2027):
        path = Path(f'full_fit/{source_id}/{year}.json')
        data = (ROOT/'artifacts/heatline'/path).read_bytes()
        assert data == (ROOT/'frontend/public/data'/path).read_bytes()
        trace = json.loads(data)
        assert trace['evaluation_kind'] == 'in-sample'
        assert trace['model_version'] == VERSION and trace['selected_year'] == year
        assert trace['source_id'] == source_id and len(trace['records']) == 31
        for record in trace['records']:
            assert record['probability'] == expected[record['date']]['probability']
            assert record['actual'] == expected[record['date']]['actual']
            seen.append(record['date'])
        if year == 2026:
            r = trace['records'][21]; model = trace['fit_model']
            x = np.array([r['inputs'][f] for f in trace['feature_names']])
            x = (x-np.array(model['scaler_mean']))/np.array(model['scaler_scale'])
            observed = []
            for weights, member in zip(model['members'], r['members']):
                net = EBPANetwork(len(x), 16, activation='bipolar_sigmoid', seed=weights['seed'])
                for key in ['V', 'v0', 'W', 'w0']:
                    setattr(net, key, np.array(weights[key]))
                observed.append(float(net.forward(x)[0, 0]))
                assert observed[-1] == pytest.approx(member['score'], abs=1e-14)
            raw = float(np.mean(observed)); cal = model['calibration']
            p = 1/(1+np.exp(-(float(logit([raw])[0, 0])*cal['coefficient']+cal['intercept'])))
            assert p == pytest.approx(r['probability'], abs=1e-14)
    assert len(seen) == len(set(seen)) == 1426


def test_primary_fit_api_is_read_only_bounded_and_separate(fit):
    with TestClient(create_app()) as client:
        response = client.get('/api/full_dataset_fit')
        assert response.status_code == 200 and response.json()['evaluation_kind'] == 'in-sample'
        assert client.get('/api/full_fit/era5/1981').json()['selected_year'] == 1981
        assert client.get('/api/full_fit/nasa_power/2026').json()['source_id'] == 'nasa_power'
        assert client.get('/api/full_fit/era5/1980').status_code == 422
        assert client.get('/api/full_fit/other/2026').status_code == 404
        assert client.post('/api/full_dataset_fit').status_code == 405
        assert client.get('/api/full_dataset_study').json()['model_version'] == 'heatline-walk-forward-v2'
