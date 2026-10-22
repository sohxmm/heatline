"""Saved-model traces must reproduce the day-level results shown in the browser."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.ann_numpy import EBPANetwork
from src.features import FEATURE_COLS
from src.replay import export_replay_details

ROOT = Path('artifacts/heatline')


def test_export_matches_weights_scaler_and_dates(tmp_path):
    history = json.loads((ROOT/'history.json').read_text())
    history['records'] = [row for row in history['records'] if row['date'] in ['2019-05-27','2026-05-20']]
    scaler = json.loads((ROOT/'scaler.json').read_text())
    network = EBPANetwork(26); network.load_weights(ROOT/'ann_weights.npz')
    features = pd.read_csv('data/processed/features.csv')
    result = export_replay_details(history, features, network, scaler['mean'], scaler['scale'], [tmp_path])
    assert set(result) == {2019,2026}
    exported = json.loads((tmp_path/'2026.json').read_text())
    day = exported['records'][0]
    assert day['date'] == '2026-05-20' and day['input_cutoff'] == '2026-05-19'
    assert len(day['inputs']) == 26 and len(day['hidden_activations']) == 8
    original = features[features.date == day['date']].iloc[0]
    for feature, mean, scale in zip(FEATURE_COLS,scaler['mean'],scaler['scale']):
        assert day['inputs'][feature] == original[feature]
        assert day['normalized_inputs'][feature] == pytest.approx((original[feature]-mean)/scale)
    logit = np.asarray(day['hidden_activations']) @ network.W[:,0] + network.w0.item()
    assert logit == pytest.approx(day['output_logit'])
    assert network._sigmoid(np.array([logit])).item() == pytest.approx(day['score'])
    assert day['prediction'] == int(day['probability'] >= day['threshold'])


def test_rejects_model_mismatch_before_export(tmp_path):
    history = json.loads((ROOT/'history.json').read_text())
    history['records'] = history['records'][:1]
    network = EBPANetwork(26,8,seed=123)
    scaler = json.loads((ROOT/'scaler.json').read_text())
    with pytest.raises(ValueError,match='reproduce'):
        export_replay_details(history,pd.read_csv('data/processed/features.csv'),network,scaler['mean'],scaler['scale'],[tmp_path])
    assert not list(tmp_path.glob('*.json'))
