"""Full-dataset lab experiment; all reported fit scores are resubstitution.

The same seasonal rows fit preprocessing, neural weights, score calibration
and the F1 threshold. May scores are a subset of those already-seen rows.
Chronological evidence remains in full_dataset_study.py and is not overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.experiment import write_json
from src.full_dataset_study import (ROOT, SEEDS, calibrated_scores, expected_coin,
    initialization_metadata, make_features, mean_training_history, measured_metrics,
    member_history, new_network, source_feature_names)
from src.sc_baselines import PerceptronRuleClassifier, SigmoidDeltaRuleClassifier
from src.seasonal_experiment import operating_threshold

VERSION = 'heatline-full-dataset-v3'
OUT = ROOT / 'artifacts/heatline'
WEIGHTS = ROOT / 'artifacts/full-dataset-fit'
EPOCHS = 300
FILES = {'era5': ROOT/'data/processed/seasonal_daily.csv',
         'nasa_power': ROOT/'data/processed/nasa_daily.csv'}


def fit_score_transform(y, raw):
    """Fit the calibration map and decision threshold on every supplied row."""
    _, scores, calibration = calibrated_scores(y, raw, raw)
    threshold = operating_threshold(y, scores, beta=1)
    return scores, calibration, float(threshold)


def export_traces(source, frame, X, nets, scaler, calibration, threshold, scores):
    """Save all May dates with the exact full-data model computation."""
    names = source_feature_names(source)
    architecture = dict(inputs=len(names), hidden=16, output=1,
        activation='bipolar_sigmoid', loss='half_squared_error', ensemble_size=len(nets))
    model = dict(fit_years=[1981, 2026], calibration_years=[1981, 2026],
        reference_years=[1981, 2010], scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(), calibration=calibration, threshold=threshold,
        members=[dict(seed=net.seed, **{k: getattr(net, k).tolist()
            for k in ('V', 'v0', 'W', 'w0')}) for net in nets])
    records = []
    for index in np.flatnonzero(frame.month.eq(5).to_numpy()):
        members = []
        for net in nets:
            raw = float(net.forward(X[index:index+1])[0, 0])
            members.append(dict(seed=net.seed, hidden_net=net.z_in.ravel().tolist(),
                hidden_activations=net.z.ravel().tolist(),
                output_logit=float(net.y_in[0, 0]), score=raw))
        primary = members[-1]
        records.append(dict(date=frame.iloc[index].date, model_version=VERSION,
            source_id=source, evaluation_kind='in-sample',
            inputs={name: float(frame.iloc[index][name]) for name in names},
            normalized_inputs=dict(zip(names, X[index].tolist())), members=members,
            hidden_activations=primary['hidden_activations'], hidden_net=primary['hidden_net'],
            output_logit=primary['output_logit'], member_score=primary['score'], seed=primary['seed'],
            score=float(np.mean([m['score'] for m in members])), calibration=calibration,
            probability=float(scores[index]), threshold=threshold,
            threshold_margin=float(scores[index]-threshold),
            prediction=int(scores[index] >= threshold), actual=int(frame.iloc[index].heatwave_day),
            normal_tmax=float(frame.iloc[index].normal_tmax), tmax=float(frame.iloc[index].tmax)))
    for year in range(1981, 2027):
        payload = dict(schema_version='3.0', model_version=VERSION, source_id=source,
            source='ERA5 via Open-Meteo' if source == 'era5' else 'NASA POWER / MERRA-2',
            mode='full-dataset training replay', evaluation_kind='in-sample',
            availability='available', issued_at=datetime.now(timezone.utc).isoformat(),
            target_date=None, selected_year=year, year=year,
            timezone='Asia/Kolkata' if source == 'era5' else 'UTC', architecture=architecture,
            feature_names=names, fit_model=model,
            records=[r for r in records if r['date'].startswith(str(year))])
        for root in (OUT/'full_fit', ROOT/'frontend/public/data/full_fit'):
            path = root/source/f'{year}.json'; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return records


def source_fit(source, daily, epochs=EPOCHS, seeds=SEEDS):
    names = source_feature_names(source)
    frame = make_features(daily, reference_end=2010, source=source)
    y = frame.heatwave_day.to_numpy(int)
    scaler = StandardScaler().fit(frame[names])
    X = scaler.transform(frame[names])
    weight = float(np.sum(y == 0)/np.sum(y == 1))
    nets = []
    for seed in seeds:
        net = new_network(seed, len(names))
        net.train(X, y, max_epochs=epochs, batch_size=128, pos_weight=weight)
        net.save_weights(WEIGHTS/f'{source}_seed{seed}.npz'); nets.append(net)
    raw = np.mean([net.predict_proba(X) for net in nets], axis=0)
    score, calibration, threshold = fit_score_transform(y, raw)
    arrays = {'NumPy ANN': (score, score >= threshold)}
    baselines = [
        ('Logistic regression', LogisticRegression(C=1, class_weight='balanced', max_iter=2000, random_state=42)),
        ('Perceptron', PerceptronRuleClassifier(eta=.01, max_epochs=100)),
        ('Sigmoid delta rule', SigmoidDeltaRuleClassifier(eta=.001, max_epochs=200))]
    baseline_models = []
    for label, model in baselines:
        model.fit(X, y)
        p, cal, cut = fit_score_transform(y, model.predict_proba(X)[:, 1])
        arrays[label] = p, p >= cut
        baseline_models.append(dict(model=label, threshold=cut, calibration=cal,
            weights=(model.coef_.ravel() if hasattr(model, 'coef_') else model.w_).tolist(),
            bias=float(model.intercept_[0] if hasattr(model, 'intercept_') else model.b_)))
    persistence = frame.previous_proxy.to_numpy()
    arrays['Persistence'] = persistence, persistence.astype(bool)
    # Same calendar-day event frequency over all seasonal labels, not future evidence.
    climatology = frame.groupby(['month', 'day']).heatwave_day.transform('mean').to_numpy()
    cut = operating_threshold(y, climatology, beta=1)
    arrays['Climatology'] = climatology, climatology >= cut
    arrays['Always no heatwave'] = np.zeros(len(y)), np.zeros(len(y), dtype=bool)
    may = frame.month.eq(5).to_numpy()
    all_metrics = [dict(model=name, **measured_metrics(y, p, pred)) for name, (p, pred) in arrays.items()]
    may_metrics = [dict(model=name, **measured_metrics(y[may], p[may], pred[may])) for name, (p, pred) in arrays.items()]
    all_metrics.append(expected_coin(y)); may_metrics.append(expected_coin(y[may]))
    preprocessing = dict(model_version=VERSION, feature_names=names, mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(), reference_years=[1981, 2010], training_years=[1981, 2026],
        epochs=epochs, seeds=list(seeds), positive_weight=weight, calibration=calibration,
        threshold=threshold, calibration_rows=len(frame), threshold_rows=len(frame),
        normal_tmax_by_calendar_day={f'{m:02d}-{d:02d}': float(n)
            for m, d, n in frame[['month', 'day', 'normal_tmax']].drop_duplicates(['month', 'day']).itertuples(index=False, name=None)})
    write_json(WEIGHTS/f'{source}_preprocessing.json', preprocessing)
    write_json(WEIGHTS/f'{source}_baselines.json', baseline_models)
    traces = export_traces(source, frame, X, nets, scaler, calibration, threshold, score)
    records = [{k: r[k] for k in ['date', 'actual', 'probability', 'prediction', 'threshold', 'threshold_margin', 'tmax', 'normal_tmax']}
        | {'raw_score': r['score'], 'evaluation_kind': 'in-sample'} for r in traces]
    architecture = dict(inputs=len(names), hidden=16, output=1, activation='bipolar_sigmoid',
        loss='half_squared_error', ensemble_size=len(seeds))
    final_fit = dict(years=[1981, 2026], days=len(frame), positive_days=int(y.sum()),
        architecture=architecture, epochs=epochs, seeds=list(seeds), mode='full-dataset lab fit',
        positive_weight=weight, calibration=calibration, threshold=threshold,
        preprocessing_rows=len(frame), training_rows=len(frame), calibration_rows=len(frame),
        threshold_rows=len(frame), checking_rows=len(frame), no_partitions=True,
        reference_years=[1981, 2010],
        calibration_method='Platt map fitted to the same full-dataset ensemble scores and labels; F1 threshold selected on those same 4,186 rows.',
        in_sample_metrics=may_metrics[0], all_season_metrics=all_metrics[0],
        training_history=mean_training_history(nets),
        members=[dict(seed=n.seed, epochs=epochs, training_history=member_history(n)) for n in nets],
        evaluation_note='All scores are resubstitution on previously seen rows. May is a reporting filter, not a held-out partition.')
    print(f"{source}: all {len(frame)} rows, {epochs} epochs, May {may_metrics[0]}", flush=True)
    return dict(id=source, label='ERA5 via Open-Meteo' if source == 'era5' else 'NASA POWER / MERRA-2',
        time_standard='Asia/Kolkata' if source == 'era5' else 'UTC', architecture=architecture,
        feature_names=names, initialization=initialization_metadata(names, seeds), folds=[], final_fit=final_fit,
        metrics=may_metrics, all_season_metrics=all_metrics, records=records,
        trace_years=list(range(1981, 2027)), trace_url_template=f'/data/full_fit/{source}/{{year}}.json',
        coverage=dict(total_seasonal_days=len(frame), may_days=int(may.sum()), all_rows_used=True))


def run(epochs=EPOCHS):
    if epochs < 1:
        raise ValueError('Epochs must be positive')
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    sources = [source_fit(s, pd.read_csv(p), epochs=epochs) for s, p in FILES.items()]
    payload = dict(schema_version='3.0', model_version=VERSION,
        source='Independent ERA5 and NASA POWER full-dataset experiments',
        mode='full-dataset lab fit', evaluation_kind='in-sample', availability='available',
        issued_at=datetime.now(timezone.utc).isoformat(), target_date=None, timezone='Asia/Kolkata',
        protocol=dict(years=[1981, 2026], all_rows_used=True, no_partitions=True,
            method='The same 4,186 April–June rows per source fit preprocessing, weights, calibration and threshold, and are checked afterwards. May metrics filter the same fitted dataset to 1,426 dates.',
            calibration='Platt scaling and F1 threshold on all seasonal training scores and labels.',
            hyperparameters=dict(hidden=16, activation='bipolar_sigmoid', loss='half_squared_error',
                learning_rate=.03, momentum=.9, batch_size=128, max_epochs=epochs,
                early_stopping=False, positive_weight='full-dataset negative/positive ratio'),
            target='nagpur-grid-proxy-v1; fixed 1981–2010 climate reference, unchanged across benchmark versions.',
            limitation='Training-dataset fit; neither calibration nor classification scores are independent estimates of future prediction. Chronological evidence is a separate earlier experiment.'),
        sources=sources,
        provenance=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
            input_sha256={s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in FILES.items()},
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for path in [OUT/'full_dataset_fit.json', ROOT/'frontend/public/data/full_dataset_fit.json']:
        write_json(path, payload)
    pd.DataFrame([dict(source=s['id'], scope=scope, **m) for s in sources
        for scope, key in [('May in-sample', 'metrics'), ('April-June in-sample', 'all_season_metrics')]
        for m in s[key]]).to_csv(OUT/'full_dataset_fit_metrics.csv', index=False)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    run(parser.parse_args().epochs)
