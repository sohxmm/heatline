"""Generate the current scientific report and figures from measured v2 artifacts."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.lab_examples import LAB_MAPPING,backpropagation_trace


def table(rows,columns):
    def fmt(v):return f'{v:.4f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(fmt(r.get(k,'')) for k in columns)+' |' for r in rows])


def export():
    artifacts=ROOT/'artifacts/heatline';docs=ROOT/'docs';docs.mkdir(exist_ok=True)
    read=lambda name:json.loads((artifacts/f'{name}.json').read_text(encoding='utf-8'))
    m,e,a=read('model'),read('experiments'),read('source_audit');benchmark=read('provider_benchmark') if (artifacts/'provider_benchmark.json').exists() else None
    figures=artifacts/'figures';figures.mkdir(exist_ok=True)
    curve=pd.DataFrame(m['training_history']);ax=curve.plot(x='epoch',y=['train_loss','val_loss'],figsize=(8,3.5));ax.set(ylabel='Weighted half-squared-error',title='Mean member training and selection objective');ax.figure.tight_layout();ax.figure.savefig(figures/'learning.png',dpi=180);plt.close(ax.figure)
    years=pd.DataFrame(e['yearly_results']);ax=years.plot(x='year',y=['detected','missed','false_alarms'],kind='bar',figsize=(9,4),color=['#5e9b83','#ca7770','#d6a45c']);ax.set(ylabel='Days',title='Every retrospective May: detections, misses and false alarms');ax.figure.tight_layout();ax.figure.savefig(figures/'yearly-events.png',dpi=180);plt.close(ax.figure)
    grid=pd.DataFrame([r for r in e['summary'] if r['study']=='architecture_and_objective']);fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,loss in zip(axes,['half_squared_error','binary_cross_entropy']):
        for activation in ['binary_sigmoid','bipolar_sigmoid']:
            data=grid[(grid.loss==loss)&(grid.activation==activation)&(grid.positive_weight_mode=='ratio')].sort_values('hidden')
            ax.plot(data.hidden,data.selection_pr_auc,marker='o',label=activation)
        ax.set(title=loss.replace('_',' '),xlabel='Hidden neurons');ax.set_xticks([8,16,32]);ax.grid(alpha=.2)
    axes[0].set_ylabel('May 2011–2014 average precision');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'activation-size.png',dpi=180);plt.close(fig)
    history=pd.DataFrame(read('history')['records']);test=history[history.split=='retrospective']
    from sklearn.metrics import precision_recall_curve
    precision,recall,_=precision_recall_curve(test.actual,test.probability)
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].step(recall,precision,where='post');axes[0].axhline(test.actual.mean(),color='gray',linestyle='--');axes[0].set(xlabel='Recall',ylabel='Precision',title='ERA5 ANN retrospective precision–recall')
    bins=[b for b in m['calibration']['bins'] if b['count']];axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].plot([b['mean_score'] for b in bins],[b['observed_rate'] for b in bins],'o-');axes[1].set(xlabel='Mean transformed score',ylabel='Observed proxy fraction',title='Reliability bins · sparse sample');fig.tight_layout();fig.savefig(figures/'precision-calibration.png',dpi=180);plt.close(fig)
    main_columns=['model','pr_auc','precision','recall','f1','f2','brier','tn','fp','fn','tp']
    balanced=[dict(model=row['model'],**row['balanced']) for row in m['metrics']]
    chosen=m['architecture'];previous=m['previous_version']['metrics'];trace=backpropagation_trace()
    nasa=''
    if benchmark:
        nasa=f'''## Independent NASA POWER benchmark

The secondary benchmark uses NASA POWER UTC daily meteorology, a separate training-only reference and the same year partitions. It has {benchmark['training']['positive_days']} positive training days and **{benchmark['evaluation']['positive_days']} positive evaluation days out of {benchmark['evaluation']['days']}**, compared with eight ERA5 positives. Different event populations and reporting windows prevent interpreting its larger scores as an improvement on the ERA5 target.

The architecture, objective and weight mode are fixed from the ERA5 selection, with no NASA test-based architecture search. Fifty-two inputs contain seven lags of six meteorological variables, seven lagged temperature departures, previous-day proxy and calendar encoding. Early solar responses are missing, so radiation is excluded from all benchmark years rather than imputed or inconsistently included. NASA's 2m wind and UTC day are not ERA5's 10m wind and local day. A 16-neuron, one-hidden-layer NumPy network is trained for each of three seeds.

F2 operating point:

{table(benchmark['metrics'],main_columns)}

F1 operating point (threshold also chosen only on calibration years):

{table([dict(model=r['model'],**r['balanced']) for r in benchmark['metrics']],main_columns)}

Persistence remains a strong classification baseline. The ANN provides better ranking than persistence here, but does not surpass it on F1. `provider_benchmark.json` contains every retrospective record; `nasa_seed*.npz` and `nasa_scaler.json` preserve the network and preprocessing.
'''
    report=f'''# Heatline: seasonal NumPy ANN for Nagpur May heatwave conditions

This report preserves the original 26-input v2 experiment. The separate full-history source-specific study (26 ERA5 inputs, 52 NASA inputs), expanding-window evaluation and all-data refit are documented in [Evaluation protocol](EVALUATION.md). Their periods, reference normals and model versions differ; do not merge the scores.

## Problem and assessed contribution

The task is next-day binary prediction of May heatwave conditions for Nagpur, using earlier climate observations. The assessed implementation explicitly computes neuron activations, gradients, bias updates and momentum in NumPy. The selected architecture is **26 → {chosen['hidden']} → 1**, with **{chosen['activation']}** hidden units and a binary sigmoid output. Three independently initialized networks are averaged for the application; the single seed-42 network remains an independently reported baseline. Fuzzy/GA/deeper PyTorch models are legacy optional theory extensions. PyTorch checks derivatives but does not perform assessed training.

## Data audit and target

The requested coordinate is 21.1458°N, 79.0882°E; the ERA5 grid is 21.25°N, 79.0°E. The expanded cache covers March 25–June 30 in each of 1981–2026, with 98 contiguous local days per year. March primes preceding history, April–June supplies training examples, and **every reported assessed evaluation is May only**. Hourly aggregation uses Asia/Kolkata. Missing hourly values, duplicate dates, gaps and missing lag history are rejected.

Original April/May cache comparison: **{a['era5_overlap_days']} overlapping days, maximum daily Tmax difference {a['max_daily_tmax_difference']}°C, {a['may_label_changes']} May label changes**. May's 26 features also match v1 exactly. The proxy remains `nagpur-grid-proxy-v1`: Tmax≥45°C OR (Tmax≥40°C and frozen-reference departure≥4.5°C), departure rounded to 0.01°C. Frozen 1981–2010 April/May reference values are retained for comparability; March/June references use only the added training years. Labels approximate a grid day and are not official IMD station declarations or warning bulletins.

The source audit uses training-era NASA samples rather than choosing a provider for more positive labels:

{table(a['nasa_samples'],['year','may_days','nasa_mean_tmax','era5_mean_tmax','mean_nasa_minus_era5','nasa_days_ge45','era5_days_ge45'])}

{a['nasa_time_standard']} A limited NOAA airport-station cross-check is also descriptive:

{table(a.get('station_samples',[]),['year','paired_days','mean_station_tmax','mean_era5_tmax','mean_era5_minus_station','mean_absolute_difference','station_days_ge45','era5_days_ge45'])}

{a.get('station_limitation','')} These comparisons establish source sensitivity, not which proxy is the official truth. Raw responses, request metadata and hashes are retained. ERA5 remains the assessed before/after target; NASA is reported separately.

## Climate inputs and chronology

The 26 inputs retain lagged temperature, humidity, dewpoint, apparent temperature, wind, pressure, integrated irradiance, soil moisture, rolling means, tendencies, preceding proxy streak, reference normal and calendar harmonics. Every prediction for T uses observed weather through T−1. Calendar/reference features for T are deterministic. June of the same year never enters the inputs of a preceding May day. Normalization fits only April–June 1981–2010, independently of validation or evaluation years.

{table(m['splits'],['name','years','days','positives'])}

Training expands from 930 May days /55 positives to **2730 seasonal days /117 positives**. The positive fraction is still small. Positive weight is fitted from training class counts only; the selected weight is {m['positive_weight']:.4f}.

## Neuron, activation and learning rule

For X, hidden Z=f(XV+v₀), output logit s=ZW+w₀, and ŷ=σ(s). Binary sigmoid is σ(a)=1/(1+exp(−a)); bipolar sigmoid is tanh(a/2), derivative (1−f(a)²)/2. Hidden bipolar activation does not change the binary output target convention.

For positive weight p, q(t)=1+(p−1)t. The textbook objective L=mean[½q(t)(ŷ−t)²] gives output-logit derivative d=q(t)(ŷ−t)ŷ(1−ŷ). Weighted BCE L=mean[−pt log ŷ−(1−t)log(1−ŷ)] gives d=q(t)ŷ−pt. Both objectives and their exact derivatives are compared; the selected revised model uses the textbook weighted squared-error objective.

Hidden derivative h=(dWᵀ)⊙f′(XV+v₀). Gradients: Zᵀd/N, mean(d), Xᵀh/N, mean(h) for W,w₀,V,v₀ respectively. All gradients use old parameters. Δθ=−η∇L+μΔθ_previous, θ←θ+Δθ. The selected experiment uses η=.03, μ=.9, batch 128, up to 300 epochs, patience 35. Early stopping evaluates the declared weighted objective on May 2011–2014. Weights use seeded Xavier initialization; biases start at zero; gradients are clipped per coordinate at 5.

Experiment 7 uses an independent scalar fixture: x=[0,1], target 1, η=α=.25, μ=0, explicitly displayed illustrative initial weights. Output {trace['forward']['output']:.12f}, output error signal {trace['backward']['delta_output']:.12f}; a full connection/bias update reduces loss {trace['loss_before']:.12f}→{trace['loss_after']:.12f}. Binary/bipolar delta examples, fixed-threshold NAND/NOR/memory, Hebbian association, perceptron AND and trained XOR remain in the executed notebook.

{table(LAB_MAPPING,['experiment','title','co','connection'])}

## Selection, repeated seeds and operating decisions

The versioned `config_v2.yaml` fixes the experiment before its retrospective evaluation: 8/16/32 hidden neurons × binary/bipolar sigmoid × half-squared-error/BCE × no/square-root-ratio/full-ratio weighting, each with seeds 7/21/42. The 36 configurations use 108 trained members. Highest **May 2011–2014 average precision of the three-member mean** chooses the primary configuration; ties use lower selection Brier then smaller hidden width. Training-window and learning-factor ablations are descriptive and do not replace this choice based on later scores. All runs are exported with their selection-period results.

Inference averages each member's sigmoid score, not logits or weights. Platt scaling fits only May 2015–2018. The sensitivity threshold maximizes F2 (recall weighted more heavily), with ties preferring precision then higher threshold. The balanced threshold independently maximizes F1 on the same calibration period. Calibration scores are fitted, not independent performance evidence. An ensemble or Platt scaling does not guarantee calibration.

## Retrospective results on unchanged ERA5 May targets

Metric interpretation: the primary ANN's 29.4% is alert precision (5 correct alerts out of 17), not accuracy. Accuracy is 233/248 = 94.0%; always-negative predictions score 240/248 = 96.8% but miss all eight positive days. Balanced accuracy is 78.8%. False alarms remain a real limitation; confusion counts, precision, recall, F1 and average precision are the main evidence.

Primary sensitivity policy:

{table(m['metrics'],main_columns)}

Balanced policy:

{table(balanced,main_columns)}

Against v1, detection increases from {previous['tp']}/8 to {m['metrics'][0]['tp']}/8 and recall from {previous['recall']:.1%} to {m['metrics'][0]['recall']:.1%}; false alarms increase from {previous['fp']} to {m['metrics'][0]['fp']}. Average precision changes from {previous['pr_auc']:.4f} to {m['metrics'][0]['pr_auc']:.4f}. This is a sensitivity trade-off, not a claim of universal superiority. Simpler baselines still win some metrics. F2 is {m['metrics'][0]['f2']:.4f} for the selected ANN under the calibration-chosen policy.

{table(e['yearly_results'],['year','days','positive_days','detected','missed','false_alarms'])}

Onset analysis: {e['onset']['detected']} detected of {e['onset']['events']} positive days that follow a negative day. Continuing hot spells are easier than onset; the application exposes both errors. Detected dates: {', '.join(r['date'] for r in e['detected_events'])}. Missed dates: {', '.join(r['date'] for r in e['missed_events'])}.

Uncertainty uses 1000 whole-May-year resamples, preserving within-month dependence. Intervals: average precision {e['uncertainty']['intervals']['pr_auc']}, recall {e['uncertainty']['intervals']['recall']}, F2 {e['uncertainty']['intervals']['f2']}. {e['uncertainty']['limitation']}

![Learning](../artifacts/heatline/figures/learning.png)
![Every retrospective May](../artifacts/heatline/figures/yearly-events.png)
![Activation and width](../artifacts/heatline/figures/activation-size.png)
![Precision–recall and calibration](../artifacts/heatline/figures/precision-calibration.png)

{nasa}

## Reproducibility, application and limitations

Run `python -m src.seasonal_experiment`, `python -m src.provider_benchmark`, `python scripts/export_seasonal_evidence.py`, and `python scripts/build_lab_notebook.py`. Raw files are cached; training makes no provider calls. To acquire missing seasons, use `python -m src.seasonal_data` and its `--provider nasa` option. Cached study checkpoints are local, ignored intermediates; saved final member weights and normalization are versioned. The original v1 is frozen in `legacy/heatline-v1-experiment.zip`.

The 2019–2026 outcomes were inspected before this revision. Consequently all apparent gains are **exploratory retrospective findings**, not proof on an untouched holdout. A later untouched May or a predeclared prospective protocol is required for a new confirmation claim. Rare events, grid/station mismatch and daily-boundary differences limit generalization. The station audit is not an official-warning reconstruction. Higher recall can mean more false alarms; the complete trade-off is visible.

The dashboard presents historical daily inputs, each member's hidden/output computation and mean score. Delayed ERA5 cannot directly serve a validated live ANN; regional live ANN outputs remain unavailable. Current six-city weather is a separately labeled provider product with cached freshness. No official warning or health-decision claim is made.

## Sources

- [Open-Meteo archive and ERA5](https://open-meteo.com/en/docs/historical-weather-api)
- [NASA POWER daily API and time standards](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
- [NASA meteorological source methodology](https://power.larc.nasa.gov/docs/methodology/data/sources_meteorology/)
- [NOAA daily synoptic summary](https://www.ncei.noaa.gov/access/search/datasets/global-summary-of-the-day/)
- [IMD criteria](https://mausam.imd.gov.in/pdfs/heatcolduser/Definition.pdf)
'''
    (docs/'ACADEMIC_REPORT.md').write_text(report,encoding='utf-8')
    print('Updated v2 report and four measured figures.')


if __name__=='__main__':export()
