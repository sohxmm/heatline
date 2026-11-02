"""Independent NASA POWER benchmark; it never replaces ERA5 evaluation labels.

Fixed protocol: the v2 selected one-hidden-layer architecture, three fixed seeds,
training 1981–2010 April–June, May selection/calibration/evaluation periods.
NASA UTC daily features form a distinct data product. Its event counts and
scores are reported against NASA-derived targets, not official station warnings.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.ann_numpy import EBPANetwork
from src.climatology import compute_reference_normals
from src.experiment import fit_calibrator,logit,write_json
from src.labels import heatwave_proxy
from src.seasonal_experiment import operating_threshold,extra_metrics,ROOT,OUT


def prepare_nasa():
    records=[]
    for year in range(1981,2027):
        path=ROOT/f'data/raw/nasa_power/{year}.json'
        payload=json.loads(path.read_text())
        if payload['header']['time_standard']!='UTC':raise ValueError('Explicit UTC daily boundary required')
        frame=pd.DataFrame(payload['properties']['parameter']);frame['date']=pd.to_datetime(frame.index,format='%Y%m%d')
        if frame.drop(columns='date').isna().any().any() or (frame.drop(columns='date')<=-900).any().any():
            raise ValueError(f'NASA missing values in {year}; no silent substitution')
        for _,r in frame.iterrows():
            records.append(dict(date=r.date.strftime('%Y-%m-%d'),year=r.date.year,month=r.date.month,day=r.date.day,
                tmax=float(r.T2M_MAX),tmin=float(r.T2M_MIN),tmean=float(r.T2M),rh_mean=float(r.RH2M),
                wind_mean=float(r.WS2M)*3.6,pressure_mean=float(r.PS)*10,
                # These placeholder columns only satisfy the generic reference helper;
                # neither enters this benchmark's predictors or heatwave labels.
                apparent_tmax=float(r.T2M_MAX)))
    daily=pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    normals=compute_reference_normals(daily)
    daily=daily.merge(normals[['month','day','normal_tmax']],on=['month','day'],validate='many_to_one')
    daily['heatwave_day']=heatwave_proxy(daily.tmax,daily.normal_tmax)
    parts=[];cols=[]
    for _,g in daily.groupby('year',sort=True):
        g=g.sort_values('date').copy()
        if not pd.to_datetime(g.date).diff().dropna().eq(pd.Timedelta(days=1)).all():raise ValueError('NASA calendar gap')
        # Solar products are absent in early POWER responses; use the six
        # meteorological variables consistently available across all 46 seasons.
        for variable in ['tmax','tmin','tmean','rh_mean','wind_mean','pressure_mean']:
            for lag in range(1,8):
                name=f'{variable}_lag{lag}';g[name]=g[variable].shift(lag)
                if name not in cols:cols.append(name)
        for lag in range(1,8):
            name=f'departure_lag{lag}';g[name]=(g.tmax-g.normal_tmax).shift(lag)
            if name not in cols:cols.append(name)
        g['previous_proxy']=g.heatwave_day.shift(1)
        doy=pd.to_datetime('2001-'+g.date.str[5:]).dt.dayofyear
        g['day_sin']=np.sin(2*np.pi*doy/365);g['day_cos']=np.cos(2*np.pi*doy/365)
        parts.append(g[g.month.isin([4,5,6])])
    cols+=['previous_proxy','day_sin','day_cos']
    features=pd.concat(parts,ignore_index=True)
    if not np.isfinite(features[cols].to_numpy()).all():raise ValueError('Incomplete NASA lag history')
    return daily,features,cols


def run():
    daily,f,cols=prepare_nasa();model=json.loads((OUT/'model.json').read_text());arch=model['architecture']
    y=f.heatwave_day.to_numpy();may=f.month.eq(5).to_numpy();train=f.year.le(2010).to_numpy();select=f.year.between(2011,2014).to_numpy()&may
    cal=f.year.between(2015,2018).to_numpy()&may;test=f.year.between(2019,2026).to_numpy()&may
    scaler=StandardScaler().fit(f.loc[train,cols]);X=scaler.transform(f[cols]);ratio=float((train.sum()-y[train].sum())/y[train].sum())
    weights={'none':1.,'sqrt_ratio':np.sqrt(ratio),'ratio':ratio}
    chosen=json.loads((OUT/'provenance.json').read_text())['selection']['positive_weight_mode'];nets=[]
    for seed in [7,21,42]:
        net=EBPANetwork(len(cols),arch['hidden'],activation=arch['activation'],loss=arch['loss'],learning_rate=.03,momentum=.9,seed=seed)
        net.train(X[train],y[train],X[select],y[select],max_epochs=300,patience=35,batch_size=128,pos_weight=weights[chosen]);nets.append(net)
        net.save_weights(OUT/f'nasa_seed{seed}.npz')
        print(f'NASA benchmark seed {seed} trained',flush=True)
    scores=np.mean([net.predict_proba(X) for net in nets],axis=0)
    results=[];probabilities={}
    def evaluate(name,raw,calibrate=True):
        platt=fit_calibrator(y[cal],raw[cal]) if calibrate else None
        p=platt.predict_proba(logit(raw))[:,1] if platt else raw
        threshold=.5 if name=='Persistence' else operating_threshold(y[cal],p[cal],2)
        balanced=.5 if name=='Persistence' else operating_threshold(y[cal],p[cal],1)
        results.append(dict(model=name,**extra_metrics(y[test],p[test],threshold),balanced=extra_metrics(y[test],p[test],balanced)))
        probabilities[name]=(p,threshold,balanced)
    evaluate('NASA NumPy ANN',scores)
    logistic=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42).fit(X[train],y[train])
    evaluate('NASA logistic regression',logistic.predict_proba(X)[:,1]);evaluate('Persistence',f.previous_proxy.to_numpy(),False)
    p,threshold,balanced_threshold=probabilities['NASA NumPy ANN']
    records=[dict(date=f.iloc[i].date,score=float(scores[i]),probability=float(p[i]),prediction=int(p[i]>=threshold),balanced_prediction=int(p[i]>=balanced_threshold),actual=int(y[i]),tmax=float(f.iloc[i].tmax)) for i in np.flatnonzero(test)]
    payload=dict(source='NASA POWER daily API · MERRA-2/POWER meteorology',mode='independent source benchmark',
        time_standard='UTC, not Asia/Kolkata local calendar',target='IMD-inspired proxy using NASA temperatures and its own frozen 1981–2010 reference',
        selection_note='Architecture and weight-mode fixed from the ERA5 v2 selection before NASA evaluation; only training, early stopping and calibration are source-specific.',
        architecture=dict(inputs=len(cols),hidden=arch['hidden'],activation=arch['activation'],loss=arch['loss'],ensemble_size=3),positive_weight=float(weights[chosen]),
        training=dict(days=int(train.sum()),positive_days=int(y[train].sum())),evaluation=dict(years='2019–2026',days=int(test.sum()),positive_days=int(y[test].sum())),
        metrics=results,records=records,limitations=['These scores are not comparable as an improvement over ERA5 because the target data product and daily boundaries differ.',
            '52 lag features use six meteorological variables, seven lagged departures, previous proxy and calendar; early solar responses are unavailable and are excluded for every year. NASA wind is 2m, ERA5 is 10m.',
            'The benchmark is not validated against official station declarations or live warnings.'])
    write_json(OUT/'provider_benchmark.json',payload);write_json(ROOT/'frontend/public/data/provider_benchmark.json',payload)
    from src.split import save_scaler_params
    save_scaler_params(scaler,cols,str(OUT/'nasa_scaler.json'))
    experiments=json.loads((OUT/'experiments.json').read_text());experiments['provider_benchmark']={k:v for k,v in payload.items() if k!='records'}
    write_json(OUT/'experiments.json',experiments);write_json(ROOT/'frontend/public/data/experiments.json',experiments)
    daily.to_csv(ROOT/'data/processed/nasa_daily.csv',index=False)
    print(json.dumps({k:payload[k] for k in ['training','evaluation','metrics']},indent=2),flush=True)
    return payload


if __name__=='__main__':run()
