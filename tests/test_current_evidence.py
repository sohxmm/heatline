"""Verify published v2 results against the saved model and fixed evaluation days."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix,average_precision_score,fbeta_score
from src.ann_numpy import EBPANetwork
from src.provider_benchmark import prepare_nasa

ART=Path('artifacts/heatline')


def test_saved_ensemble_and_calibration_reproduce_all_published_may_scores():
    model=json.loads((ART/'model.json').read_text());history=json.loads((ART/'history.json').read_text())
    frame=pd.read_csv('data/processed/features.csv');scaler=json.loads((ART/'scaler.json').read_text())
    X=(frame[scaler['feature_names']].to_numpy()-scaler['mean'])/scaler['scale']
    scores=[]
    for filename in json.loads((ART/'ensemble.json').read_text())['members']:
        network=EBPANetwork(26);network.load_weights(ART/filename);scores.append(network.predict_proba(X))
    mean=np.mean(scores,axis=0);records=pd.DataFrame(history['records'])
    np.testing.assert_allclose(mean,records.score,atol=1e-12,rtol=1e-12)
    c=model['calibration'];logits=np.log(np.clip(mean,1e-7,1-1e-7)/np.clip(1-mean,1e-7,1))
    transformed=1/(1+np.exp(-(c['coefficient']*logits+c['intercept'])))
    np.testing.assert_allclose(transformed,records.probability,atol=1e-12,rtol=1e-12)
    np.testing.assert_array_equal(records.prediction,transformed>=model['threshold'])
    test=records[records.split=='retrospective'];result=model['metrics'][0]
    assert len(test)==248 and test.actual.sum()==8
    assert confusion_matrix(test.actual,test.prediction,labels=[0,1]).ravel().tolist()==[result[k] for k in ['tn','fp','fn','tp']]
    assert average_precision_score(test.actual,test.probability)==pytest.approx(result['pr_auc'])
    assert fbeta_score(test.actual,test.prediction,beta=2)==pytest.approx(result['f2'])


def test_nasa_benchmark_has_independent_labels_and_complete_history():
    _,frame,features=prepare_nasa()
    benchmark=json.loads((ART/'provider_benchmark.json').read_text())
    assert len(features)==52
    test=frame[(frame.year>=2019)&(frame.month==5)]
    assert len(test)==248 and test.heatwave_day.sum()==benchmark['evaluation']['positive_days']
    records=pd.DataFrame(benchmark['records'])
    np.testing.assert_array_equal(records.actual,test.heatwave_day)
    result=benchmark['metrics'][0]
    assert confusion_matrix(records.actual,records.balanced_prediction,labels=[0,1]).ravel().tolist()==[result['balanced'][k] for k in ['tn','fp','fn','tp']]
    assert benchmark['evaluation']['positive_days']!=8


def test_source_audit_reports_unchanged_primary_target_and_hashes():
    audit=json.loads((ART/'source_audit.json').read_text())
    assert audit['era5_overlap_days']==2806 and audit['may_label_changes']==0
    assert audit['max_daily_tmax_difference']==0
    assert len(audit['station_samples'])==3
    assert all(len(file['sha256'])==64 for file in audit['files'])
