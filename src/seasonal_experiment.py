"""Seasonal, repeated-seed NumPy ANN study with a fixed May evaluation target."""
from datetime import datetime, timezone
import itertools
import hashlib
import platform
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import yaml

from src.ann_numpy import EBPANetwork
from src.features import FEATURE_COLS
from src.experiment import metrics, fit_calibrator, logit, calibration_bins, write_json
from src.sc_baselines import PerceptronRuleClassifier, SigmoidDeltaRuleClassifier
from src.seasonal_data import ROOT, prepare_era5, source_audit
from src.split import save_scaler_params

OUT=ROOT/'artifacts/heatline'
CHECKPOINTS=ROOT/'artifacts/seasonal-study'


def operating_threshold(y, score, beta=2):
    """Max F-beta on calibration data, ties: precision then higher threshold."""
    candidates=np.unique(np.r_[0,score,np.nextafter(1.,2.)])
    return float(max(candidates,key=lambda t:(fbeta_score(y,score>=t,beta=beta,zero_division=0),precision_score(y,score>=t,zero_division=0),t)))


def extra_metrics(y,score,threshold):
    return {**metrics(y,score,threshold),'f2':float(fbeta_score(y,score>=threshold,beta=2,zero_division=0)),
        'roc_auc':float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None}


def year_bootstrap(years,y,score,threshold,repeats=1000):
    """Resample whole May years, retaining event dependence within each month."""
    unique=np.unique(years);rng=np.random.RandomState(2026);values=[]
    for _ in range(repeats):
        indices=np.concatenate([np.flatnonzero(years==year) for year in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[indices]))<2:continue
        result=extra_metrics(y[indices],score[indices],threshold)
        values.append([result[k] for k in ['pr_auc','recall','precision','f1','f2','brier']])
    a=np.array(values)
    return dict(method='May-year block bootstrap; fixed fitted model and threshold',repetitions=repeats,valid_repetitions=len(values),
        intervals={key:[float(x) for x in np.quantile(a[:,i],[.025,.975])] for i,key in enumerate(['pr_auc','recall','precision','f1','f2','brier'])},
        limitation='Eight May years and few positive events; intervals do not include model-selection uncertainty.')


def run():
    cfg=yaml.safe_load((ROOT/'config_v2.yaml').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True,exist_ok=True);CHECKPOINTS.mkdir(exist_ok=True)
    daily,f,normals=prepare_era5()
    signature=hashlib.sha256(b''.join(path.read_bytes() for path in [ROOT/'config_v2.yaml',ROOT/'src/ann_numpy.py',ROOT/'data/processed/seasonal_features.csv'])).hexdigest()
    signature_path=CHECKPOINTS/'input-signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text())['sha256']!=signature:
        raise ValueError('Study checkpoints belong to different data/configuration. Use a fresh checkpoint directory.')
    write_json(signature_path,dict(sha256=signature))
    y=f.heatwave_day.to_numpy(dtype=int)
    train=f.year.between(1981,2010).to_numpy()
    may=f.month.eq(5).to_numpy()
    select=f.year.between(2011,2014).to_numpy()&may
    calibration=f.year.between(2015,2018).to_numpy()&may
    test=f.year.between(2019,2026).to_numpy()&may
    scaler=StandardScaler().fit(f.loc[train,FEATURE_COLS])
    X=scaler.transform(f[FEATURE_COLS])
    ratio=float((len(y[train])-y[train].sum())/y[train].sum())
    print(f'Seasonal training: {train.sum()} days, {y[train].sum()} positives; class ratio={ratio:.3f}',flush=True)
    studies=[];runs=[];networks={}

    def train_candidate(activation,hidden,loss,weight_mode,window='April–June',eta=None,mu=None,study='architecture_and_objective'):
        eta=cfg['learning_rate'] if eta is None else eta
        mu=cfg['momentum'] if mu is None else mu
        mask=train if window=='April–June' else train&may
        # Each controlled training-window comparison fits its own training-only scaler.
        local_scaler=scaler if window=='April–June' else StandardScaler().fit(f.loc[mask,FEATURE_COLS])
        inputs=local_scaler.transform(f[FEATURE_COLS])
        local_ratio=float((mask.sum()-y[mask].sum())/y[mask].sum())
        weight={'none':1.,'sqrt_ratio':np.sqrt(local_ratio),'ratio':local_ratio}[weight_mode]
        key=f'{activation}-{hidden}-{loss}-{weight_mode}-{window.replace("–","-")}-{eta}-{mu}'
        members=[]
        for seed in cfg['seeds']:
            checkpoint=CHECKPOINTS/f'{key}-{seed}.npz';info_path=checkpoint.with_suffix('.json')
            net=EBPANetwork(26,hidden,activation=activation,loss=loss,learning_rate=eta,momentum=mu,seed=seed)
            if checkpoint.exists() and info_path.exists():
                net.load_weights(checkpoint);info=json.loads(info_path.read_text());net.history=info['history']
            else:
                info=net.train(inputs[mask],y[mask],inputs[select],y[select],max_epochs=cfg['max_epochs'],
                    batch_size=cfg['batch_size'],patience=cfg['patience'],pos_weight=weight)
                net.save_weights(checkpoint);write_json(info_path,info)
            score=net.predict_proba(inputs[select])
            row=dict(activation=activation,hidden=hidden,loss=loss,positive_weight_mode=weight_mode,positive_weight=float(weight),training_window=window,
                learning_rate=eta,momentum=mu,seed=seed,study=study,epochs=info['epochs_trained'],best_epoch=info['best_epoch'],**extra_metrics(y[select],score,.5))
            runs.append(row);members.append(net)
        score=np.mean([net.predict_proba(inputs[select]) for net in members],axis=0)
        row=dict(key=key,activation=activation,hidden=hidden,loss=loss,positive_weight_mode=weight_mode,positive_weight=float(weight),training_window=window,
            learning_rate=eta,momentum=mu,study=study,mean_seed_pr_auc=float(np.mean([r['pr_auc'] for r in runs[-3:]])),std_seed_pr_auc=float(np.std([r['pr_auc'] for r in runs[-3:]],ddof=1)),
            selection_pr_auc=float(average_precision_score(y[select],score)),selection_brier=float(np.mean((score-y[select])**2)))
        studies.append(row);networks[key]=(members,inputs,local_scaler)
        write_json(CHECKPOINTS/'selection-progress.json',dict(runs=runs,candidates=studies))
        print(f"{study}: {hidden} {activation} {loss} weight={weight_mode} {window}: selection AP={row['selection_pr_auc']:.4f}",flush=True)
        return row

    for activation,hidden,loss,weight_mode in itertools.product(cfg['activations'],cfg['hidden_sizes'],cfg['objectives'],cfg['positive_weights']):
        train_candidate(activation,hidden,loss,weight_mode)
    # Select before evaluating calibration or retrospective records.
    ranked=sorted(studies,key=lambda r:(-r['selection_pr_auc'],r['selection_brier'],r['hidden'],r['key']))
    selected=ranked[0]
    # Descriptive ablations do not reopen selection based on later outcomes.
    for window,weight_mode in itertools.product(['May','April–June'],['none',selected['positive_weight_mode']]):
        if not any(r['training_window']==window and r['positive_weight_mode']==weight_mode and r['activation']==selected['activation'] and r['hidden']==selected['hidden'] and r['loss']==selected['loss'] for r in studies):
            train_candidate(selected['activation'],selected['hidden'],selected['loss'],weight_mode,window,study='training_window')
    for eta,mu in [(0.01,0.0),(0.01,.9),(.03,0.0),(.1,.9)]:
        train_candidate(selected['activation'],selected['hidden'],selected['loss'],selected['positive_weight_mode'],eta=eta,mu=mu,study='learning_factors')
    members,inputs,final_scaler=networks[selected['key']]
    for seed,net in zip(cfg['seeds'],members):net.save_weights(OUT/f'ann_seed{seed}.npz')
    # Seed42 remains the inspectable single-network reference; ensemble output is
    # explicitly the arithmetic mean of three independently learned sigmoid scores.
    members[-1].save_weights(OUT/'ann_weights.npz')
    save_scaler_params(final_scaler,FEATURE_COLS,str(OUT/'scaler.json'))
    raw_predictions={};results=[];details={};policies={}

    def evaluate(name,raw,calibrate=True):
        transform=fit_calibrator(y[calibration],raw[calibration]) if calibrate else None
        p=transform.predict_proba(logit(raw))[:,1] if transform else raw.copy()
        if transform is not None and transform.coef_[0,0]<=0:
            raise ValueError('Calibration inverted risk ranking; investigate before publishing')
        primary=.5 if name=='Persistence' else operating_threshold(y[calibration],p[calibration],2)
        balanced=.5 if name=='Persistence' else operating_threshold(y[calibration],p[calibration],1)
        row=dict(model=name,**extra_metrics(y[test],p[test],primary),raw_brier=float(np.mean((raw[test]-y[test])**2)),
            calibration='Platt scaling, May 2015–2018' if calibrate else 'None',decision_policy='F2 sensitivity',
            balanced=extra_metrics(y[test],p[test],balanced))
        results.append(row);raw_predictions[name]=(raw,p,primary)
        details[name]=dict(method=row['calibration'],threshold=primary,coefficient=float(transform.coef_[0,0]) if transform else None,
            intercept=float(transform.intercept_[0]) if transform else None,bins=calibration_bins(y[test],p[test]))
        policies[name]=dict(sensitive=dict(threshold=primary,calibration=extra_metrics(y[calibration],p[calibration],primary),retrospective=extra_metrics(y[test],p[test],primary)),
            balanced=dict(threshold=balanced,calibration=extra_metrics(y[calibration],p[calibration],balanced),retrospective=extra_metrics(y[test],p[test],balanced)))

    evaluate('NumPy ANN',np.mean([net.predict_proba(inputs) for net in members],axis=0))
    evaluate('Single NumPy ANN (seed 42)',members[-1].predict_proba(inputs))
    for objective,label in [('half_squared_error','ANN textbook squared error'),('binary_cross_entropy','ANN cross-entropy')]:
        best=next(r for r in ranked if r['loss']==objective)
        nets,local_inputs,_=networks[best['key']]
        evaluate(label,np.mean([net.predict_proba(local_inputs) for net in nets],axis=0))
    evaluate('Persistence',(f.hot_streak_len.to_numpy()>0).astype(float),False)
    climatology={day:float(f.loc[train&may&(f.day.sub(day).abs()<=3).to_numpy(),'heatwave_day'].mean()) for day in range(1,32)}
    evaluate('Climatology',f.day.map(climatology).fillna(float(y[train&may].mean())).to_numpy(),False)
    perceptron=PerceptronRuleClassifier(eta=.01,max_epochs=100,random_state=42).fit(X[train],y[train]);evaluate('Perceptron',perceptron.predict_proba(X)[:,1])
    delta=SigmoidDeltaRuleClassifier(eta=.01,max_epochs=200,random_state=42).fit(X[train],y[train]);evaluate('Sigmoid delta rule',delta.predict_proba(X)[:,1])
    logistic=LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=42).fit(X[train],y[train]);evaluate('Logistic regression',logistic.predict_proba(X)[:,1])
    # Simple warming extrapolation tests whether the ANN adds value beyond temperature inertia.
    extrapolated=f.tmax_lag1.to_numpy()+(f.tmax_lag1-f.tmax_lag2).to_numpy()
    margins=np.maximum(extrapolated-45,np.minimum(extrapolated-40,extrapolated-f.normal_tmax_T.to_numpy()-4.5))
    evaluate('Temperature trend',1/(1+np.exp(-margins)))
    raw,p,threshold=raw_predictions['NumPy ANN'];balanced_threshold=policies['NumPy ANN']['balanced']['threshold']
    may_indices=np.flatnonzero(may);joined=f.merge(daily[['date','tmax','normal_tmax']],on='date',validate='one_to_one')
    records=[]
    for index in may_indices:
        row=joined.iloc[index];split='train' if row.year<=2010 else 'selection' if row.year<=2014 else 'calibration' if row.year<=2018 else 'retrospective'
        records.append(dict(date=row.date,score=float(raw[index]),probability=float(p[index]),prediction=int(p[index]>=threshold),actual=int(y[index]),
            tmax=float(row.tmax),normal_tmax=float(row.normal_tmax),threshold=threshold,balanced_threshold=balanced_threshold,
            balanced_prediction=int(p[index]>=balanced_threshold),split=split,target_date=row.date,mode='replay',
            input_cutoff=(pd.Timestamp(row.date)-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),availability='available',model_version=cfg['version']))
    meta=dict(schema_version='2.0',source='Open-Meteo ERA5 reanalysis',mode='replay',issued_at=datetime.now(timezone.utc).isoformat(),target_date=None,
        model_version=cfg['version'],availability='available',timezone='Asia/Kolkata')
    history=dict(**meta,city='nagpur',years=list(range(1981,2027)),records=records,score_description='Mean of three NumPy ANN sigmoid scores, followed by separate calibration. Not guaranteed calibrated.',
        issue_time_note='Artifact generation time, not an archived operational forecast issue time.')
    epochs=min(len(net.history['epoch']) for net in members)
    training_history=[dict(epoch=i+1,train_loss=float(np.mean([net.history['train_loss'][i] for net in members])),val_loss=float(np.mean([net.history['val_loss'][i] for net in members]))) for i in range(epochs)]
    splits=[dict(name='train · April–June',years='1981–2010',days=int(train.sum()),positives=int(y[train].sum()))]+[dict(name=name,years=years,days=int(mask.sum()),positives=int(y[mask].sum())) for name,years,mask in [('selection · May','2011–2014',select),('calibration · May','2015–2018',calibration),('retrospective · May','2019–2026',test)]]
    baseline=json.loads(__import__('zipfile').ZipFile(ROOT/'legacy/heatline-v1-experiment.zip').read('artifacts/heatline/model.json'))
    model=dict(**meta,name='Heatline · seasonal NumPy ANN',architecture=dict(inputs=26,hidden=selected['hidden'],activation=selected['activation'],loss=selected['loss']),
        ensemble_size=3,seeds=cfg['seeds'],learning_rate=cfg['learning_rate'],momentum=cfg['momentum'],positive_weight=selected['positive_weight'],training_months=[4,5,6],
        threshold=threshold,decision_policy='Sensitivity: F2 maximized on May 2015–2018',balanced_threshold=balanced_threshold,trained_until='2010-06-30',
        metrics=results,splits=splits,training_history=training_history,feature_names=FEATURE_COLS,label_version='nagpur-grid-proxy-v1',
        label_rule='Tmax ≥45 °C OR (Tmax ≥40 °C AND frozen 1981–2010 reference departure ≥4.5 °C)',
        calibration=details['NumPy ANN'],selection_policy='Highest May 2011–2014 average precision of the three-seed mean; tie lower Brier then smaller hidden layer. F2 threshold and Platt scaling fitted on May 2015–2018.',
        pr_auc_definition='Average precision; not trapezoidal AUC',previous_version=dict(version=baseline['model_version'],metrics=baseline['metrics'][0]),
        limitations=['Single-grid daily proxy, not official station-based heatwave declarations.',
          'Only 8 positives in 248 retrospective May days; adjacent days are correlated.',
          '2019–2026 outcomes were already examined before this revision. Improvements are exploratory retrospective results, not an untouched-test claim.',
          'F2 emphasizes recall and can create more false alarms; balanced F1 operating point is shown separately.',
          'Three-seed averaging reduces seed dependence but does not guarantee calibration; Platt fitting uses only 124 May days.',
          'Delayed ERA5 supports historical replay only. Live/regional ANN predictions remain unvalidated and unavailable.',
          'Extra June records expand training; a May forecast still only uses preceding weather. May targets and inputs are identical to v1.'])
    yearly=[]
    for year in range(2019,2027):
        mask=test&f.year.eq(year).to_numpy();pred=p[mask]>=threshold
        yearly.append(dict(year=year,days=int(mask.sum()),positive_days=int(y[mask].sum()),detected=int(((y[mask]==1)&pred).sum()),missed=int(((y[mask]==1)&~pred).sum()),false_alarms=int(((y[mask]==0)&pred).sum())))
    # Event onset excludes continuing positives: last observed day is negative.
    onset=test&(y==1)&(f.hot_streak_len.to_numpy()==0)
    onset_result=dict(events=int(onset.sum()),detected=int((p[onset]>=threshold).sum()),missed=int((p[onset]<threshold).sum()),
        dates=[dict(date=f.iloc[i].date,prediction=int(p[i]>=threshold),probability=float(p[i])) for i in np.flatnonzero(onset)])
    bootstrap=year_bootstrap(f.loc[test,'year'].to_numpy(),y[test],p[test],threshold,cfg['bootstrap'])
    study_metrics=[]
    for row in studies:
        nets,local_inputs,_=networks[row['key']]
        scores=np.mean([net.predict_proba(local_inputs) for net in nets],axis=0)
        # Retrospective descriptive ablations; they never alter selected configuration.
        transform=fit_calibrator(y[calibration],scores[calibration]);probs=transform.predict_proba(logit(scores))[:,1]
        cut=operating_threshold(y[calibration],probs[calibration],2)
        study_metrics.append({**row,**extra_metrics(y[test],probs[test],cut)})
    audit=source_audit(daily)
    experiments=dict(**meta,runs=runs,summary=ranked,evaluation_period='May 2011–2014 selection',test_metrics=results,operating_points=policies,source_audit=audit,
        yearly_results=yearly,onset=onset_result,uncertainty=bootstrap,ablations=study_metrics,
        missed_events=[r for r in records if r['split']=='retrospective' and r['actual']==1 and r['prediction']==0],
        detected_events=[r for r in records if r['split']=='retrospective' and r['actual']==1 and r['prediction']==1],
        false_alarms=[r for r in records if r['split']=='retrospective' and r['actual']==0 and r['prediction']==1])
    from src.lab_examples import backpropagation_trace,LAB_MAPPING
    trace=backpropagation_trace();trace['learning_rate']=trace['eta'];trace['deltas']=dict(output=trace['backward']['delta_output'],hidden=trace['backward']['delta_hidden'])
    learning=dict(**{**meta,'mode':'educational'},trace=trace,syllabus=LAB_MAPPING,training_history=training_history)
    for name,payload in [('model',model),('history',history),('experiments',experiments),('learning',learning)]:
        write_json(OUT/f'{name}.json',payload);write_json(ROOT/'frontend/public/data'/f'{name}.json',payload)
    write_json(OUT/'ensemble.json',dict(version=cfg['version'],members=[f'ann_seed{seed}.npz' for seed in cfg['seeds']],aggregation='arithmetic mean of sigmoid scores'))
    write_json(OUT/'calibration.json',details);write_json(OUT/'source_audit.json',source_audit(daily))
    write_json(ROOT/'frontend/public/data/source_audit.json',source_audit(daily))
    import sklearn
    write_json(OUT/'provenance.json',dict(**meta,config=cfg,source_audit='source_audit.json',selection=selected,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        input_signature=signature,previous_version_archive='legacy/heatline-v1-experiment.zip'))
    pd.DataFrame(runs).to_csv(OUT/'runs.csv',index=False);pd.DataFrame([{k:v for k,v in row.items() if k!='balanced'} for row in results]).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(study_metrics).to_csv(OUT/'ablations.csv',index=False)
    from src.replay import export_ensemble_replay
    export_ensemble_replay(history,f,members,final_scaler.mean_,final_scaler.scale_,[OUT/'replay',ROOT/'frontend/public/data/replay'])
    print('SELECTED',selected,flush=True);print(pd.DataFrame(results).drop(columns=['balanced']).to_string(index=False),flush=True);print('YEARLY',yearly,flush=True)
    return model,experiments


if __name__=='__main__':
    logging.getLogger().setLevel(logging.WARNING)
    run()
