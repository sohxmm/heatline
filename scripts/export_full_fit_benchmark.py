"""Compare the v2 and v3 full-data checkpoints on identical May rows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ann_numpy import EBPANetwork
from src.experiment import logit, write_json
from src.full_dataset_fit import FILES, OUT, ROOT, fit_score_transform
from src.full_dataset_study import make_features, measured_metrics


def run():
    previous = json.loads((OUT/'full_dataset_study.json').read_text(encoding='utf-8'))
    current = json.loads((OUT/'full_dataset_fit.json').read_text(encoding='utf-8'))
    rows = []
    for source in current['sources']:
        source_id = source['id']
        old = next(s for s in previous['sources'] if s['id'] == source_id)
        frame = make_features(pd.read_csv(FILES[source_id]), 2010, source_id)
        preprocessing = json.loads((ROOT/f'artifacts/full-dataset-study/{source_id}_preprocessing.json').read_text())
        X = (frame[source['feature_names']].to_numpy()-preprocessing['mean'])/preprocessing['scale']
        scores = []
        for seed in [7, 21, 42]:
            net = EBPANetwork(len(source['feature_names']), 16)
            net.load_weights(ROOT/f'artifacts/full-dataset-study/{source_id}_seed{seed}.npz')
            scores.append(net.predict_proba(X))
        raw = np.mean(scores, axis=0)
        cal = preprocessing['calibration']
        p = 1/(1+np.exp(-(logit(raw).ravel()*cal['coefficient']+cal['intercept'])))
        y = frame.heatwave_day.to_numpy(); may = frame.month.eq(5).to_numpy()
        reproduced = measured_metrics(y[may], p[may], p[may] >= preprocessing['threshold'])
        for key, value in reproduced.items():
            if not np.isclose(value, old['final_fit']['in_sample_metrics'][key], atol=1e-12):
                raise ValueError(f'Previous checkpoint does not reproduce {source_id}/{key}')
        recalibrated, _, cut = fit_score_transform(y, raw)
        calibration_only = measured_metrics(y[may], recalibrated[may], recalibrated[may] >= cut)
        after = source['final_fit']['in_sample_metrics']
        rows.append(dict(source=source_id, same_may_rows=1426,
            before=dict(model_version=previous['model_version'], epochs=old['final_fit']['epochs'],
                threshold=old['final_fit']['threshold'], metrics=reproduced),
            calibration_only=dict(epochs=old['final_fit']['epochs'], threshold=cut, metrics=calibration_only),
            after=dict(model_version=current['model_version'], epochs=source['final_fit']['epochs'],
                threshold=source['final_fit']['threshold'], metrics=after),
            change={k: float(after[k]-reproduced[k]) for k in ['accuracy', 'precision', 'recall', 'f1', 'pr_auc', 'brier']},
            earlier_chronological_metrics=old['metrics'][0]))
    payload = dict(model_version=current['model_version'], source='Saved NumPy checkpoints and source-specific cached data',
        mode='experiment comparison', issued_at=current['issued_at'], target_date=None, availability='available',
        evaluation_kind='in-sample',
        protocol='Before and after use the identical 1,426 May 1981–2026 dates and fixed source-specific 1981–2010 target normals. Both models trained on all 4,186 seasonal rows.',
        changes='v3 fits calibration and the F1 threshold on all seasonal rows and trains for a fixed 300 epochs. v2 used pooled earlier out-of-fold calibration and median fold-selected epoch counts.',
        calibration_only_note='The intermediate control keeps the old trained weights and changes only calibration and threshold fitting to the entire seasonal dataset.',
        limitation='These are training-dataset fit comparisons. They do not establish improvement on unseen dates. Earlier chronological results retain their original protocol and are unchanged.',
        sources=rows)
    for p in [OUT/'full_dataset_benchmark.json', ROOT/'frontend/public/data/full_dataset_benchmark.json']:
        write_json(p, payload)
    pd.DataFrame([dict(source=r['source'], experiment=phase, **r[phase]['metrics'])
        for r in rows for phase in ['before', 'calibration_only', 'after']]).to_csv(OUT/'full_dataset_benchmark.csv', index=False)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == '__main__':
    run()
