"""Export inspectable per-day inference from the exact saved assessed model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.ann_numpy import EBPANetwork
from src.features import FEATURE_COLS


def export_replay_details(history, features, network, mean, scale, destinations):
    """Partition full inference traces by year so a browser loads one May at a time.

    Inputs and hidden activations describe forward computation, not causal
    feature importance. Outputs must match the published replay before export.
    """
    if features.date.duplicated().any():
        raise ValueError('Duplicate feature dates cannot identify a replay day')
    lookup = features.set_index('date')
    dates = [record['date'] for record in history['records']]
    if len(set(dates)) != len(dates) or not set(dates).issubset(lookup.index):
        raise ValueError('Replay dates must uniquely match feature observations')
    inputs = lookup.loc[dates, FEATURE_COLS].to_numpy(dtype=float)
    mean, scale = np.asarray(mean), np.asarray(scale)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Training scales must be finite and positive')
    normalized = (inputs - mean) / scale
    score = network.forward(normalized).ravel()
    expected = np.array([record['score'] for record in history['records']])
    if not np.allclose(score, expected, atol=1e-12, rtol=1e-12):
        raise ValueError('Saved model/scaler does not reproduce the published scores')
    hidden, logits = network.z.copy(), network.y_in.ravel().copy()
    years = {}
    for index, record in enumerate(history['records']):
        date = record['date']
        years.setdefault(int(date[:4]), []).append({**record,
            'inputs': dict(zip(FEATURE_COLS, inputs[index].tolist())),
            'normalized_inputs': dict(zip(FEATURE_COLS, normalized[index].tolist())),
            'hidden_activations': hidden[index].tolist(),
            'output_logit': float(logits[index]),
        })
    metadata = {key: value for key, value in history.items() if key not in ('records', 'years')}
    for year, records in years.items():
        payload = {**metadata, 'selected_year': year, 'records': records,
            'normalization': {'reference': '1981–2010 training data only', 'mean': dict(zip(FEATURE_COLS, mean.tolist())), 'scale': dict(zip(FEATURE_COLS, scale.tolist()))},
            'architecture': {'inputs': network.n_input, 'hidden': network.n_hidden, 'activation': network.activation, 'loss': network.loss},
            'calculation_note': 'Recorded historical inputs → frozen training normalization → hidden activation → sigmoid score → separately fitted Platt scaling → decision threshold. This is retrospective inference, not a live forecast.'}
        for directory in destinations:
            path = Path(directory) / f'{year}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    return years


def export_saved_replay(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    artifacts = root / 'artifacts/heatline'
    history = json.loads((artifacts/'history.json').read_text(encoding='utf-8'))
    scaler = json.loads((artifacts/'scaler.json').read_text(encoding='utf-8'))
    if scaler['feature_names'] != FEATURE_COLS:
        raise ValueError('Saved scaler feature order differs from the current model')
    network = EBPANetwork(26)
    network.load_weights(artifacts/'ann_weights.npz')
    if (artifacts/'ensemble.json').exists():
        metadata=json.loads((artifacts/'ensemble.json').read_text())
        members=[]
        for filename in metadata['members']:
            member=EBPANetwork(26);member.load_weights(artifacts/filename);members.append(member)
        return export_ensemble_replay(history,pd.read_csv(root/'data/processed/seasonal_features.csv'),members,
            scaler['mean'],scaler['scale'],[artifacts/'replay',root/'frontend/public/data/replay'])
    return export_replay_details(history, pd.read_csv(root/'data/processed/features.csv'), network,
        scaler['mean'], scaler['scale'], [artifacts/'replay', root/'frontend/public/data/replay'])


def export_ensemble_replay(history,features,members,mean,scale,destinations):
    """Export each member separately; mean logits is NOT mean sigmoid output."""
    if features.date.duplicated().any():raise ValueError('Duplicate feature dates')
    lookup=features.set_index('date');dates=[row['date'] for row in history['records']]
    if not members or len(set(dates))!=len(dates) or not set(dates).issubset(lookup.index):
        raise ValueError('Members and unique matching replay dates required')
    raw_inputs=lookup.loc[dates,FEATURE_COLS].to_numpy(dtype=float)
    mean,scale=np.asarray(mean),np.asarray(scale)
    if not np.isfinite(raw_inputs).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Finite inputs and positive training scales required')
    normalized=(raw_inputs-mean)/scale
    member_results=[]
    for member in members:
        scores=member.forward(normalized).ravel().copy()
        member_results.append(dict(seed=member.seed,scores=scores,hidden=member.z.copy(),logits=member.y_in.ravel().copy()))
    average=np.mean([r['scores'] for r in member_results],axis=0)
    np.testing.assert_allclose(average,[r['score'] for r in history['records']],atol=1e-12,rtol=1e-12)
    years={}
    for i,record in enumerate(history['records']):
        member_traces=[dict(seed=r['seed'],score=float(r['scores'][i]),output_logit=float(r['logits'][i]),hidden_activations=r['hidden'][i].tolist()) for r in member_results]
        selected=member_traces[-1]
        years.setdefault(int(record['date'][:4]),[]).append({**record,'inputs':dict(zip(FEATURE_COLS,raw_inputs[i].tolist())),
            'normalized_inputs':dict(zip(FEATURE_COLS,normalized[i].tolist())),
            'hidden_activations':selected['hidden_activations'],'output_logit':selected['output_logit'],
            'trace_member_seed':selected['seed'],'member_score':selected['score'],'members':member_traces,'aggregation':'mean of member sigmoid scores'})
    meta={k:v for k,v in history.items() if k not in ['years','records']}
    for year,records in years.items():
        payload={**meta,'selected_year':year,'records':records,'ensemble_size':len(members),
            'normalization':dict(reference='April–June 1981–2010 training data only',mean=dict(zip(FEATURE_COLS,mean.tolist())),scale=dict(zip(FEATURE_COLS,scale.tolist()))),
            'architecture':dict(inputs=26,hidden=members[0].n_hidden,activation=members[0].activation,loss=members[0].loss)}
        for directory in destinations:
            path=Path(directory)/f'{year}.json';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    return years


if __name__ == '__main__':
    result = export_saved_replay()
    print(f'Exported {sum(map(len, result.values()))} verified day traces across {len(result)} May calendars.')
