"""Reproduce the assessed Heatline experiment; no tuning uses 2019–2026 outcomes.

Run: python -m src.experiment. Raw hourly observations are cached in data/raw.
The grid, seeds and selection policy below are fixed before evaluating test years.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import logging
from pathlib import Path
import platform
import shutil
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import yaml

from src.ann_numpy import EBPANetwork
from src.climatology import compute_wmo_climatology
from src.daily import process_hourly_to_daily
from src.features import build_and_save_features, FEATURE_COLS
from src.labels import evaluate_labels_and_audit
from src.sc_baselines import PerceptronRuleClassifier, SigmoidDeltaRuleClassifier
from src.split import save_scaler_params

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'heatline'
VERSION = 'heatline-nagpur-v1'
SEEDS = [7, 21, 42]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


def envelope(mode='replay'):
    return dict(schema_version='1.0', source='Open-Meteo ERA5 reanalysis', mode=mode,
        issued_at=datetime.now(timezone.utc).isoformat(), target_date=None,
        model_version=VERSION, availability='available', timezone='Asia/Kolkata')


def metrics(y, score, threshold):
    predicted = np.asarray(score) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(pr_auc=float(average_precision_score(y, score)),
        precision=float(precision_score(y, predicted, zero_division=0)),
        recall=float(recall_score(y, predicted, zero_division=0)),
        f1=float(f1_score(y, predicted, zero_division=0)),
        brier=float(brier_score_loss(y, np.clip(score, 0, 1))),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(threshold))


def select_threshold(y, score):
    """Max calibration-period F1; ties prefer fewer alerts (higher threshold)."""
    candidates = np.unique(np.r_[0, score, 1.000001])
    return float(max(candidates, key=lambda t: (f1_score(y, score >= t, zero_division=0), t)))


def logit(score):
    p = np.clip(score, 1e-7, 1-1e-7)
    return np.log(p / (1-p)).reshape(-1, 1)


def fit_calibrator(y, raw):
    """Regularized Platt scaling fitted only on 2015–2018, no guarantee of calibration."""
    if len(np.unique(y)) != 2:
        raise ValueError('Calibration period must contain both outcomes')
    return LogisticRegression(C=1, solver='lbfgs', random_state=42).fit(logit(raw), y)


def calibration_bins(y, p):
    return [dict(lower=float(lo), upper=float(lo+.2), count=int(mask.sum()),
        mean_score=float(p[mask].mean()) if mask.any() else None,
        observed_rate=float(y[mask].mean()) if mask.any() else None)
        for lo in np.arange(0, 1, .2)
        for mask in [(p >= lo) & (p < lo+.2 if lo < .8 else p <= 1)]]


def run(max_epochs=400):
    config = yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    process_hourly_to_daily(config)
    compute_wmo_climatology(config)
    audit = evaluate_labels_and_audit(config)
    features = build_and_save_features(config)
    daily = pd.read_csv(ROOT/'data/processed/daily.csv')
    partitions = [('train', 1981, 2010), ('selection', 2011, 2014), ('calibration', 2015, 2018), ('retrospective', 2019, 2026)]
    masks = {name: features.year.between(start, end).to_numpy() for name, start, end in partitions}
    scaler = StandardScaler().fit(features.loc[masks['train'], FEATURE_COLS])
    X = scaler.transform(features[FEATURE_COLS])
    y = features.heatwave_day.to_numpy(dtype=int)
    save_scaler_params(scaler, FEATURE_COLS, str(OUT/'scaler.json'))
    train, selection, calibration, test = [masks[k] for k in ['train','selection','calibration','retrospective']]
    runs, networks = [], {}

    def fit_run(activation, hidden, loss, eta, mu, seed, study):
        key = (activation, hidden, loss, eta, mu, seed)
        if key in networks:
            return
        net = EBPANetwork(26, hidden, activation=activation, learning_rate=eta, momentum=mu, seed=seed, loss=loss)
        result = net.train(X[train], y[train], X[selection], y[selection], max_epochs=max_epochs,
            batch_size=64, patience=60, verbose=False, pos_weight=1.0)
        score = net.predict_proba(X[selection])
        row = dict(activation=activation, hidden=hidden, loss=loss, learning_rate=eta, momentum=mu,
            seed=seed, study=study, epochs=result['epochs_trained'], best_epoch=result['best_epoch'],
            **metrics(y[selection], score, .5))
        runs.append(row); networks[key] = net
        print(f"{study}: {activation} {hidden} {loss} eta={eta} mu={mu} seed={seed}: selection AP={row['pr_auc']:.4f}", flush=True)

    for activation, hidden, loss, seed in itertools.product(['binary_sigmoid','bipolar_sigmoid'], [8,16,32], ['half_squared_error','binary_cross_entropy'], SEEDS):
        fit_run(activation, hidden, loss, .05, .9, seed, 'architecture_and_objective')
    table = pd.DataFrame(runs)
    groups = table.groupby(['activation','hidden','loss'], as_index=False).agg(mean_pr_auc=('pr_auc','mean'), std_pr_auc=('pr_auc','std'), mean_brier=('brier','mean'))
    ranked = groups.sort_values(['mean_pr_auc','mean_brier','hidden','activation','loss'], ascending=[False, True, True, True, True])
    selected = ranked.iloc[0]
    # The learning-factor study is controlled on the selected architecture. It
    # is descriptive and does not change the predeclared deployed eta/mu.
    for eta, mu, seed in itertools.product([.01,.05,.1], [0.,.9], SEEDS):
        fit_run(selected.activation, int(selected.hidden), selected.loss, eta, mu, seed, 'learning_factors')
    primary_key = (selected.activation, int(selected.hidden), selected.loss, .05, .9, 42)
    primary = networks[primary_key]
    primary.save_weights(str(OUT/'ann_weights.npz'))
    predictions, comparisons, calibration_details = {}, [], {}

    def evaluate(name, raw, calibrate=True):
        calibrator = fit_calibrator(y[calibration], raw[calibration]) if calibrate else None
        p = calibrator.predict_proba(logit(raw))[:, 1] if calibrator else raw
        threshold = select_threshold(y[calibration], p[calibration]) if name != 'Persistence' else .5
        row = dict(model=name, **metrics(y[test], p[test], threshold),
            raw_brier=float(brier_score_loss(y[test], np.clip(raw[test],0,1))),
            calibration='Platt scaling, 2015–2018' if calibrate else 'None')
        comparisons.append(row)
        predictions[name] = (raw, p, threshold)
        calibration_details[name] = dict(method=row['calibration'], threshold=threshold,
            coefficient=float(calibrator.coef_[0,0]) if calibrator else None,
            intercept=float(calibrator.intercept_[0]) if calibrator else None,
            bins=calibration_bins(y[test], p[test]))

    evaluate('NumPy ANN', primary.predict_proba(X))
    # Both mathematically consistent objectives remain independently visible.
    for objective, label in [('half_squared_error','ANN textbook squared error'), ('binary_cross_entropy','ANN cross-entropy')]:
        best = ranked[ranked.loss == objective].iloc[0]
        net = networks[(best.activation, int(best.hidden), objective, .05, .9, 42)]
        evaluate(label, net.predict_proba(X))
    evaluate('Persistence', (features.hot_streak_len.to_numpy() > 0).astype(float), False)
    train_frame = features.loc[train]
    # May-day empirical event rate in a ±3 day window, trained only on 1981–2010.
    climatology = {day: float(train_frame.loc[(train_frame.day-day).abs() <= 3, 'heatwave_day'].mean()) for day in range(1,32)}
    evaluate('Climatology', features.day.map(climatology).to_numpy(), False)
    perceptron = PerceptronRuleClassifier(eta=.01,max_epochs=100,random_state=42).fit(X[train],y[train])
    evaluate('Perceptron', perceptron.predict_proba(X)[:,1])
    delta = SigmoidDeltaRuleClassifier(eta=.01,max_epochs=200,random_state=42).fit(X[train],y[train])
    evaluate('Sigmoid delta rule', delta.predict_proba(X)[:,1])
    logistic = LogisticRegression(C=1,max_iter=2000,random_state=42).fit(X[train],y[train])
    evaluate('Logistic regression', logistic.predict_proba(X)[:,1])

    raw, p, threshold = predictions['NumPy ANN']
    joined = features.merge(daily[['date','tmax','normal_tmax']], on='date', validate='one_to_one')
    records = []
    for i, row in joined.iterrows():
        split = next(name for name, _, _ in partitions if masks[name][i])
        records.append(dict(date=row.date, score=float(raw[i]), probability=float(p[i]),
            prediction=int(p[i] >= threshold), actual=int(y[i]), tmax=float(row.tmax), normal_tmax=float(row.normal_tmax),
            threshold=threshold, split=split, target_date=row.date, mode='replay',
            input_cutoff=(pd.Timestamp(row.date)-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            availability='available', model_version=VERSION))
    history = dict(**envelope(), city='nagpur', years=list(range(1981,2027)), records=records,
        score_description='Raw sigmoid score; probability is Platt-transformed using 2015–2018 only. Neither is guaranteed calibrated.',
        issue_time_note='issued_at is artifact generation time. Historical inputs are delayed reanalysis, not archived operational forecasts.')
    training_history = [dict(epoch=int(e)+1,train_loss=float(t),val_loss=float(v)) for e,t,v in zip(primary.history['epoch'],primary.history['train_loss'],primary.history['val_loss'])]
    splits = [dict(name=name,years=f'{start}–{end}',days=int(masks[name].sum()),positives=int(y[masks[name]].sum())) for name,start,end in partitions]
    limitations = [
        'Gridded reanalysis proxy, not official station-based IMD declarations or health guidance.',
        'Only 8 positive days in 248 retrospective May days: performance estimates are unstable and adjacent days are correlated.',
        '2019–2026 was examined in the legacy project; this is retrospective evaluation, not a pristine unseen holdout.',
        'ERA5 inputs arrive with delay. Live and other-city ANN outputs are unavailable pending compatible-input validation.',
        'Platt scaling and thresholds use only 124 calibration days; sigmoid outputs are not automatically calibrated probabilities.',
        'Frozen normals use 1981–2010; calendar windows are truncated at April/May edges. Training-period displays are in-sample.',
    ]
    model = dict(**envelope(), name='Heatline · Nagpur May ANN', architecture=dict(inputs=26,hidden=int(selected.hidden),activation=selected.activation,loss=selected.loss),
        threshold=threshold, seed=42, learning_rate=.05, momentum=.9, trained_until='2010-05-31',
        metrics=comparisons, splits=splits, limitations=limitations, training_history=training_history,
        feature_names=FEATURE_COLS, label_version=config['labels']['version'],
        label_rule='Tmax ≥ 45 °C OR (Tmax ≥ 40 °C AND departure from 1981–2010 reference ≥ 4.5 °C)',
        selection_policy='Highest mean 2011–2014 average precision across seeds 7,21,42; tie: lower mean Brier, smaller hidden layer. Deploy fixed seed 42. Eta .05, mu .9 fixed.',
        calibration=calibration_details['NumPy ANN'], pr_auc_definition='Average precision (step-integrated precision–recall), not trapezoidal AUC',
        optional_extensions=dict(status='legacy benchmark only',archive='legacy/original-experiment.zip',
            note='Deep PyTorch ensemble, fuzzy and GA used different labels; old scores are not comparable to this table.'))
    experiments = dict(**envelope(), runs=runs, summary=ranked.to_dict('records'),
        evaluation_period='2011–2014 selection only', metrics_period='2019–2026 retrospective',
        test_metrics=comparisons, calibration=calibration_details,
        missed_events=[r for r in records if r['split']=='retrospective' and r['actual']==1 and r['prediction']==0],
        false_alarms=[r for r in records if r['split']=='retrospective' and r['actual']==0 and r['prediction']==1])
    from src.lab_examples import backpropagation_trace, LAB_MAPPING
    trace = backpropagation_trace()
    trace['learning_rate'] = trace['eta']
    trace['deltas'] = dict(output=trace['backward']['delta_output'], hidden=trace['backward']['delta_hidden'])
    learning = dict(**envelope('educational'), trace=trace, syllabus=LAB_MAPPING, training_history=training_history)
    for name, payload in [('model',model),('history',history),('experiments',experiments),('learning',learning)]:
        write_json(OUT/f'{name}.json', payload)
        write_json(ROOT/'frontend/public/data'/f'{name}.json', payload)
    pd.DataFrame(runs).to_csv(OUT/'runs.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'metrics.csv',index=False)
    write_json(OUT/'calibration.json', calibration_details)
    write_json(OUT/'provenance.json', dict(**envelope(), config=config, versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        raw_files=[dict(path=str(path.relative_to(ROOT)).replace('\\','/'),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted((ROOT/'data/raw').glob('*.json'))],
        max_epochs=max_epochs, seeds=SEEDS, baseline_parameters=dict(logistic_C=1,perceptron_eta=.01,perceptron_epochs=100,delta_eta=.01,delta_epochs=200),
        source_note='Existing grid_metadata declares ERA5; original cache has no per-request model provenance. Future fetches record it explicitly.'))
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    return model, experiments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-epochs',type=int,default=400)
    args=parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    run(args.max_epochs)


if __name__ == '__main__':
    main()
