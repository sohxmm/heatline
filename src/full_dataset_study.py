"""Expanding-window evidence and a separate full-data NumPy ANN refit.

Every cached season participates, but a row never trains the model that tests
it. The first fifteen years initialize the study and have no held-out score.
This is an exploratory extension, not a replacement for the v2 replay model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

from src.ann_numpy import EBPANetwork
from src.experiment import fit_calibrator, logit, write_json
from src.labels import heatwave_proxy
from src.seasonal_experiment import operating_threshold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/heatline'
WEIGHTS = ROOT / 'artifacts/full-dataset-study'
VERSION = 'heatline-walk-forward-v1'
SEEDS = (7, 21, 42)
VARIABLES = ('tmax', 'tmin', 'tmean', 'rh_mean', 'wind_mean', 'pressure_mean')
FEATURES = ([f'{name}_lag{lag}' for name in VARIABLES for lag in range(1, 8)]
            + [f'departure_lag{lag}' for lag in range(1, 8)]
            + ['previous_proxy', 'day_sin', 'day_cos'])


def fold_plan():
    """Two years each for stopping and calibration, all before evaluation."""
    return [dict(id=f'{start}-{end}', fit_years=[1981, start-5],
        selection_years=[start-4, start-3], calibration_years=[start-2, start-1],
        evaluation_years=[start, end])
        for start, end in [(1996, 2000), (2001, 2005), (2006, 2010),
                           (2011, 2015), (2016, 2020), (2021, 2026)]]


def make_features(daily: pd.DataFrame, reference_end: int):
    """Recompute labels and 52 common lag features without future normals.

    Reference years are 1981 through min(2010, fit_end). Calendar-day normals
    use a +/-7-day window and are frozen for the entire fold. Input files'
    previously computed labels and normals are deliberately discarded.
    """
    required = ['date', 'year', 'month', 'day', *VARIABLES]
    frame = daily[required].copy().sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(frame.date)
    if dates.duplicated().any() or not np.isfinite(frame[list(VARIABLES)]).all().all():
        raise ValueError('Unique dates and finite weather observations required')
    if reference_end < 1981:
        raise ValueError('Reference must include at least one training year')
    doy = pd.to_datetime('2001-' + frame.date.str[5:]).dt.dayofyear
    baseline = frame.year.between(1981, min(2010, reference_end))
    if not baseline.any():
        raise ValueError('Missing reference observations')
    normal_lookup = {int(day): round(float(frame.loc[baseline & ((doy-day).abs() <= 7), 'tmax'].mean()), 2)
                     for day in sorted(doy.unique())}
    frame['normal_tmax'] = doy.map(normal_lookup)
    frame['heatwave_day'] = heatwave_proxy(frame.tmax, frame.normal_tmax)
    parts = []
    for _, group in frame.groupby('year', sort=True):
        group = group.copy()
        if not pd.to_datetime(group.date).diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError('Missing weather history; no silent interpolation')
        for variable in VARIABLES:
            for lag in range(1, 8):
                group[f'{variable}_lag{lag}'] = group[variable].shift(lag)
        for lag in range(1, 8):
            group[f'departure_lag{lag}'] = (group.tmax-group.normal_tmax).shift(lag)
        group['previous_proxy'] = group.heatwave_day.shift(1)
        local_doy = pd.to_datetime('2001-' + group.date.str[5:]).dt.dayofyear
        group['day_sin'] = np.sin(2*np.pi*local_doy/365)
        group['day_cos'] = np.cos(2*np.pi*local_doy/365)
        parts.append(group[group.month.isin([4, 5, 6])])
    features = pd.concat(parts, ignore_index=True)
    if not np.isfinite(features[FEATURES].to_numpy()).all():
        raise ValueError('Seven complete preceding days required for every target')
    return features


def measured_metrics(y, probability, prediction):
    """Accuracy and alert precision deliberately have different field names."""
    y, p, predicted = np.asarray(y), np.asarray(probability), np.asarray(prediction)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(y, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
        precision=float(precision_score(y, predicted, zero_division=0)),
        recall=float(recall_score(y, predicted, zero_division=0)),
        f1=float(f1_score(y, predicted, zero_division=0)),
        pr_auc=float(average_precision_score(y, p)) if np.any(y) else 0.,
        brier=float(brier_score_loss(y, p)), tn=int(tn), fp=int(fp),
        fn=int(fn), tp=int(tp), days=int(len(y)), positives=int(np.sum(y)))


def expected_coin(y):
    """Analytic expected confusion counts; no lucky random seed selection."""
    positives = int(np.sum(y)); negatives = len(y)-positives
    prevalence = positives/len(y)
    return dict(model='Fair coin (expected)', accuracy=.5, balanced_accuracy=.5,
        precision=prevalence, recall=.5,
        f1=2*prevalence*.5/(prevalence+.5), pr_auc=prevalence,
        brier=.25, tn=negatives/2, fp=negatives/2, fn=positives/2,
        tp=positives/2, days=len(y), positives=positives,
        note='Analytic expected counts; PR-AUC is the no-skill prevalence reference; F1 is derived from expected counts.')


def new_network(seed):
    return EBPANetwork(52, 16, activation='bipolar_sigmoid',
        loss='half_squared_error', learning_rate=.03, momentum=.9, seed=seed)


def calibrated_scores(y, raw_calibration, raw_target):
    """Only earlier calibration labels may choose the score transform."""
    if len(np.unique(y)) < 2:
        # A one-class period cannot identify a calibration slope. Preserve the
        # model score explicitly instead of adding future calibration examples.
        return np.asarray(raw_calibration), np.asarray(raw_target), None
    calibrator = fit_calibrator(y, raw_calibration)
    payload = dict(coefficient=float(calibrator.coef_[0, 0]),
                   intercept=float(calibrator.intercept_[0]))
    return (calibrator.predict_proba(logit(raw_calibration))[:, 1],
            calibrator.predict_proba(logit(raw_target))[:, 1], payload)


def _source_study(source, daily, seeds=SEEDS, max_epochs=300):
    folds, records, all_scores, best_epochs = [], [], [], []
    for plan in fold_plan():
        frame = make_features(daily, plan['fit_years'][1])
        years = frame.year.to_numpy(); may = frame.month.eq(5).to_numpy()
        masks = {name: frame.year.between(*plan[f'{name}_years']).to_numpy()
                 & (True if name == 'fit' else may)
                 for name in ['fit', 'selection', 'calibration', 'evaluation']}
        fit, select, cal, test = [masks[k] for k in ['fit', 'selection', 'calibration', 'evaluation']]
        scaler = StandardScaler().fit(frame.loc[fit, FEATURES])
        X = scaler.transform(frame[FEATURES]); y = frame.heatwave_day.to_numpy(int)
        weight = float(np.sum(y[fit] == 0)/np.sum(y[fit] == 1))
        nets, members = [], []
        for seed in seeds:
            net = new_network(seed)
            info = net.train(X[fit], y[fit], X[select], y[select],
                max_epochs=max_epochs, patience=35, batch_size=128, pos_weight=weight)
            nets.append(net); best_epochs.append(info['best_epoch']+1)
            members.append(dict(seed=seed, epochs=info['epochs_trained'], best_epoch=info['best_epoch']+1))
        raw = np.mean([net.predict_proba(X) for net in nets], axis=0)
        cal_score, score, calibration = calibrated_scores(y[cal], raw[cal], raw[test])
        threshold = operating_threshold(y[cal], cal_score, beta=1)
        logistic = LogisticRegression(C=1, class_weight='balanced', max_iter=2000, random_state=42).fit(X[fit], y[fit])
        logistic_raw = logistic.predict_proba(X)[:, 1]
        logistic_cal, logistic_p, _ = calibrated_scores(y[cal], logistic_raw[cal], logistic_raw[test])
        logistic_threshold = operating_threshold(y[cal], logistic_cal, beta=1)
        persistence = frame.previous_proxy.to_numpy()[test]
        arrays = {'NumPy ANN': (score, score >= threshold),
            'Logistic regression': (logistic_p, logistic_p >= logistic_threshold),
            'Persistence': (persistence, persistence.astype(bool)),
            'Always no heatwave': (np.zeros(test.sum()), np.zeros(test.sum(), dtype=bool))}
        fold_metrics = [dict(model=name, **measured_metrics(y[test], p, pred))
                        for name, (p, pred) in arrays.items()]
        fold_metrics.append(expected_coin(y[test]))
        folds.append({**plan, 'reference_years': [1981, min(2010, plan['fit_years'][1])],
            'counts': {name: dict(days=int(mask.sum()), positives=int(y[mask].sum())) for name, mask in masks.items()},
            'positive_weight': weight, 'threshold': threshold, 'calibration': calibration,
            'members': members, 'metrics': fold_metrics,
            'scaler_fit_years': plan['fit_years'], 'scaler_mean': scaler.mean_.tolist()})
        for offset, index in enumerate(np.flatnonzero(test)):
            records.append(dict(date=frame.iloc[index].date, fold=plan['id'], actual=int(y[index]),
                raw_score=float(raw[index]), probability=float(score[offset]),
                prediction=int(score[offset] >= threshold), threshold=float(threshold),
                tmax=float(frame.iloc[index].tmax), normal_tmax=float(frame.iloc[index].normal_tmax),
                persistence=int(persistence[offset]), logistic_probability=float(logistic_p[offset]),
                logistic_prediction=int(logistic_p[offset] >= logistic_threshold)))
        all_scores.append(dict(y=y[test], arrays=arrays))
        print(f"{source} fold {plan['id']}: ANN accuracy={fold_metrics[0]['accuracy']:.3f} F1={fold_metrics[0]['f1']:.3f}", flush=True)
    targets = np.concatenate([r['y'] for r in all_scores])
    metrics = [dict(model=name, **measured_metrics(targets,
        np.concatenate([r['arrays'][name][0] for r in all_scores]),
        np.concatenate([r['arrays'][name][1] for r in all_scores])))
        for name in all_scores[0]['arrays']]
    metrics.append(expected_coin(targets))

    # Fit once to every seasonal row only after recording held-out evidence.
    # Epoch count comes from earlier folds, never full-data training accuracy.
    frame = make_features(daily, 2010)
    y = frame.heatwave_day.to_numpy(int)
    scaler = StandardScaler().fit(frame[FEATURES]); X = scaler.transform(frame[FEATURES])
    epochs = int(np.median(best_epochs)); weight = float(np.sum(y == 0)/np.sum(y == 1))
    final_nets = []
    for seed in seeds:
        net = new_network(seed)
        net.train(X, y, max_epochs=epochs, patience=35, batch_size=128, pos_weight=weight)
        net.save_weights(WEIGHTS/f'{source}_seed{seed}.npz'); final_nets.append(net)
    # A final calibration map uses earlier held-out raw predictions, not the
    # refit's own training scores. Its fit quality is never called a test score.
    raw_final = np.mean([net.predict_proba(X) for net in final_nets], axis=0)
    oof_raw = np.array([r['raw_score'] for r in records])
    oof_cal, final_probability, calibration = calibrated_scores(targets, oof_raw, raw_final)
    final_threshold = operating_threshold(targets, oof_cal, beta=1)
    write_json(WEIGHTS/f'{source}_preprocessing.json', dict(model_version=VERSION, feature_names=FEATURES,
        mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), reference_years=[1981, 2010],
        training_years=[1981, 2026], calibration=calibration, threshold=final_threshold,
        epochs=epochs, seeds=list(seeds), positive_weight=weight))
    may = frame.month.eq(5).to_numpy()
    coverage = dict(years=[1981, 2026], total_seasonal_days=len(frame), warmup_days=int(frame.year.le(1995).sum()),
        held_out_may_days=len(records), warmup_years=[1981, 1995], all_rows_used=True,
        participation='Every April–June row is used in the final refit; later May rows also receive one strictly forward held-out prediction. Early warmup and April/June rows do not have held-out May predictions.')
    final_fit = dict(years=[1981, 2026], days=len(frame), positive_days=int(y.sum()),
        architecture=dict(inputs=52, hidden=16, output=1, activation='bipolar_sigmoid', loss='half_squared_error', ensemble_size=len(seeds)),
        epochs=epochs, seeds=list(seeds), mode='full-data refit',
        reference_years=[1981, 2010], positive_weight=weight, calibration=calibration,
        calibration_method='Platt map and F1 threshold fitted on pooled historical out-of-fold raw scores; those labels already contribute to this final calibration and cannot be its independent test.',
        threshold=final_threshold,
        in_sample_metrics=measured_metrics(y[may], final_probability[may], final_probability[may] >= final_threshold),
        availability='saved research checkpoint; not a validated live forecast',
        evaluation_note='In-sample May diagnostics are training fit only. Use fold metrics for held-out evidence.')
    return dict(id=source, label='ERA5 via Open-Meteo' if source == 'era5' else 'NASA POWER / MERRA-2',
        time_standard='Asia/Kolkata local day' if source == 'era5' else 'UTC day',
        architecture=final_fit['architecture'], coverage=coverage, folds=folds, metrics=metrics,
        final_fit=final_fit, records=records)


def run():
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    files = {'era5': ROOT/'data/processed/seasonal_daily.csv', 'nasa_power': ROOT/'data/processed/nasa_daily.csv'}
    sources = [_source_study(source, pd.read_csv(path)) for source, path in files.items()]
    payload = dict(schema_version='1.0', model_version=VERSION, source='Source-separated ERA5 and NASA POWER studies',
        mode='walk-forward evaluation and separate full-data refit', availability='available',
        issued_at=datetime.now(timezone.utc).isoformat(), target_date=None, timezone='Asia/Kolkata',
        protocol=dict(years=[1981, 2026], warmup_years=[1981, 1995], evaluation_years=[1996, 2026],
            method='Six expanding-window, whole-year blocks; earlier stopping and calibration windows; held-out evaluation in May only.',
            calibration='Two immediately preceding May years; F1 threshold selected on calibration only. Selection uses the earlier two May years.',
            all_rows_used=True, seeds=list(SEEDS), feature_names=FEATURES,
            hyperparameters=dict(hidden=16, activation='bipolar_sigmoid', loss='half_squared_error', learning_rate=.03, momentum=.9, batch_size=128, max_epochs=300, patience=35, positive_weight='fit-fold negative/positive ratio'),
            target='nagpur-grid-proxy-v1 rule; fold-local frozen reference years 1981–min(2010, fit_end). Tmax >=45°C OR (Tmax >=40°C and departure >=4.5°C).',
            limitation='Exploratory historical study: the architecture was informed by previous experiments on some of these years; this is not a new untouched test. Warmup years cannot be independently predicted without earlier data.'),
        sources=sources,
        source_decision=dict(primary='era5',
            recommendation='Retain ERA5 as the assessed source and NASA POWER as an independent robustness check; neither is established as more accurate against Nagpur station truth.',
            rationale='ERA5 preserves the existing local-day proxy and supplies the audited assessed inputs. A larger score on NASA-defined labels does not establish better prediction of ERA5 labels or official warnings.',
            confounders=['Source-derived targets and normals differ.', 'NASA UTC and ERA5 Asia/Kolkata days differ.', 'NASA 2m and ERA5 10m wind differ even with common feature names.', 'The new common 52-input study differs from the original 26-input replay; model versions remain separate.'],
            validation_needed='Matched Nagpur station observations and official dated declarations before selecting a source for operational warning accuracy.'),
        provenance=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
            input_sha256={source: hashlib.sha256(path.read_bytes()).hexdigest() for source, path in files.items()},
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        metric_note='Accuracy=(TP+TN)/N. Precision=TP/(TP+FP). A fair coin has expected accuracy 50%; the always-negative baseline can be much higher for rare events.')
    write_json(OUT/'full_dataset_study.json', payload)
    write_json(ROOT/'frontend/public/data/full_dataset_study.json', payload)
    pd.DataFrame([dict(source=source['id'], **metric) for source in sources for metric in source['metrics']]).to_csv(OUT/'full_dataset_metrics.csv', index=False)
    return payload


if __name__ == '__main__':
    run()
