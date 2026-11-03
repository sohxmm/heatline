# Evaluation protocol and source selection

Heatline separates three questions: whether the ANN detects source-defined heatwave days, whether a climate product represents Nagpur observations, and whether a model fitted on all available history is ready for future forecasting. A higher score on one question does not establish the others.

## Metric interpretation

The original seasonal ERA5 model's sensitivity policy has TP=5, FP=12, FN=3, TN=228 on 248 May days in 2019–2026. Precision is 5/17=29.4%; accuracy is 233/248=94.0%; recall is 5/8=62.5%; balanced accuracy is 78.8%. An always-negative classifier has accuracy 240/248=96.8%, but detects no positive days. Independent fair binary guesses have expected accuracy 50%. These quantities answer different questions. The original model has substantial false alarms and does not consistently outperform simpler models across metrics.

`scripts/export_metric_context.py` reproduces these calculations from saved confusion counts. It does not train or change the ANN. Average precision, precision, recall, F1, Brier score and the complete confusion matrix accompany accuracy.

## Using the complete historical dataset

Fitting, calibrating and evaluating on the same observations produces an in-sample diagnostic, not an independent estimate of future performance. The separate `heatline-walk-forward-v1` experiment uses every April–June season from 1981–2026 in a final refit and evaluates earlier versions of the model on later, unseen year blocks.

Six evaluation blocks cover May 1996–2000, 2001–2005, 2006–2010, 2011–2015, 2016–2020 and 2021–2026. Each fold uses an expanding earlier fit period, then two earlier May years for stopping and two earlier May years for calibration/threshold selection. No evaluation row updates the model that predicts it. The initial 1981–1995 period supplies warmup history and has no independent forward prediction. April and June provide training conditions; held-out evaluation remains May, matching the problem statement.

Each source has 4,186 seasonal examples and 961 held-out May predictions. Within each fold, normalization and class weights use fit rows only. Temperature normals are reconstructed from 1981 through the earlier of 2010 and the fit-end year, rather than supplying future climate observations to early historical forecasts. Short early reference periods and changing frozen references are limitations. This study's reference protocol differs from the original v2 replay and its numbers must not be merged with v2 metrics.

The architecture is fixed at 52–16–1: seven lags of six common meteorological variables (42), seven lagged temperature departures, previous-day proxy and two calendar harmonics. The same feature recipe reduces the original 26-versus-52 feature-count confound. Bipolar hidden sigmoid, binary output sigmoid, weighted half-squared error, explicit NumPy backpropagation, momentum and three fixed seeds remain the assessed neural mechanism.

After recording the fold results, a separate network ensemble is fitted on all 4,186 rows. Its epoch count comes from earlier fold stopping results. A final calibration map and F1 threshold are fitted using pooled historical out-of-fold raw scores; their labels therefore cannot independently evaluate that final calibration. Calibration transfer from earlier models to the full refit is not guaranteed. Full-data May fit statistics are explicitly in-sample. Research checkpoints are stored separately from the original replay weights.

All reported model comparisons use the same held-out dates within a source. Baselines include persistence, logistic regression, always-negative classification and analytical fair-coin expectations. The study is exploratory: the fixed architecture was informed by earlier experiments on some of these years. It is not a newly untouched test, and it does not validate live forecast inputs.

## ERA5 and NASA POWER

ERA5 remains the primary assessed source. It supplies hourly observations aggregated to Asia/Kolkata days, a finer spatial grid than NASA POWER meteorology, and the existing audited historical record. NASA POWER is a separate sensitivity study. NASA's meteorology uses UTC daily boundaries here and a different temperature distribution; its wind measurement height also differs. Source-derived labels and reference normals consequently differ even when the same heatwave rule is applied.

Selecting NASA solely because a model scores higher against NASA's own labels would change the target rather than establish better prediction of Nagpur station conditions. Neither source supplies the official warning labels used to score an operational warning system. A training-era, common-date station comparison is published separately in `station_comparison.json`; its small sample and reporting-window differences limit the conclusion.

Live six-city weather is a separately labeled provider forecast. The all-data research refit does not turn delayed ERA5 observations into timely validated ANN forecasts.

## Reproduction

```powershell
.venv/Scripts/python.exe -m src.full_dataset_study
.venv/Scripts/python.exe scripts/export_metric_context.py
.venv/Scripts/python.exe scripts/audit_station_sources.py
```

Published study JSON includes fold periods, class counts, source metadata, aggregate metrics, full-fit diagnostics and per-day held-out predictions. The CSV exports make the results independently inspectable.

## Sources

- [Scikit-learn: cross-validation and the error of testing on training observations](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Scikit-learn: data leakage and training-only preprocessing](https://scikit-learn.org/stable/common_pitfalls.html)
- [Scikit-learn: expanding-window time-series splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Open-Meteo: historical products and ERA5 resolution](https://open-meteo.com/en/docs/historical-weather-api)
- [NASA POWER: meteorological data and spatial resolution](https://power.larc.nasa.gov/docs/tutorials/)
