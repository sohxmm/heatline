"""Integrity checks for expanded training and validation-only alert decisions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score
from src.features import compute_features,FEATURE_COLS
from src.seasonal_experiment import operating_threshold,year_bootstrap
from src.replay import export_ensemble_replay
from src.ann_numpy import EBPANetwork


def test_expanded_training_preserves_may_inputs_and_targets():
    seasonal=pd.read_csv('data/processed/seasonal_features.csv')
    original=pd.read_csv('data/processed/features.csv')
    may=seasonal[seasonal.month==5].reset_index(drop=True)
    np.testing.assert_array_equal(may.heatwave_day,original.heatwave_day)
    np.testing.assert_allclose(may[FEATURE_COLS],original[FEATURE_COLS],atol=1e-8)
    train=seasonal[seasonal.year<=2010]
    assert len(train)==2730 and train.heatwave_day.sum()>original[original.year<=2010].heatwave_day.sum()


def test_future_june_cannot_change_may_feature_history():
    daily=pd.read_csv('data/processed/seasonal_daily.csv');daily=daily[daily.year==2000].copy()
    normals=pd.read_csv('data/processed/climatology_normals.csv')
    # Limit output to May; June records are not its preceding observations.
    original=compute_features(daily,normals)
    daily.loc[daily.month==6,['tmax','tmin','heatwave_day']]+=100
    changed=compute_features(daily,normals)
    np.testing.assert_array_equal(original[FEATURE_COLS],changed[FEATURE_COLS])


def test_f2_threshold_optimizes_only_supplied_calibration_examples():
    y=np.array([0,0,0,0,1,1,1]);score=np.array([.05,.15,.35,.8,.2,.6,.9])
    threshold=operating_threshold(y,score,2)
    actual=fbeta_score(y,score>=threshold,beta=2)
    assert actual==max(fbeta_score(y,score>=t,beta=2,zero_division=0) for t in np.r_[0,score,np.nextafter(1.,2.)])
    # Neither function accepts evaluation labels or a desired event count.
    assert operating_threshold(y,score,2)==threshold


def test_bootstrap_resamples_whole_years_deterministically():
    years=np.array([2019,2019,2020,2020]);y=np.array([0,1,0,1]);score=np.array([.1,.8,.2,.9])
    assert year_bootstrap(years,y,score,.5,30)==year_bootstrap(years,y,score,.5,30)


def test_exported_mean_uses_member_sigmoid_scores(tmp_path):
    members=[EBPANetwork(26,8,seed=seed) for seed in [7,21,42]]
    f=pd.read_csv('data/processed/features.csv').iloc[:2].copy()
    X=f[FEATURE_COLS].to_numpy();mean=X.mean(axis=0);scale=np.ones(26)
    scores=np.mean([net.predict_proba(X-mean) for net in members],axis=0)
    history={'model_version':'test','records':[{'date':date,'score':float(score)} for date,score in zip(f.date,scores)]}
    export_ensemble_replay(history,f,members,mean,scale,[tmp_path])
    day=json.loads((tmp_path/'1981.json').read_text())['records'][0]
    assert len(day['members'])==3
    assert day['score']==pytest.approx(np.mean([m['score'] for m in day['members']]))
    for m in day['members']:
        assert m['score']==pytest.approx(1/(1+np.exp(-m['output_logit'])))
    assert day['member_score']==day['members'][-1]['score']
