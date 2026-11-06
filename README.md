# Heatline

A from-scratch NumPy backpropagation ANN for **May heatwave conditions in Nagpur**, supported by reproducible experiments and a regional weather dashboard.

- **Assessed core:** neurons, activation functions, learning rules, one hidden layer, explicit backpropagation and evaluation.
- **Product extension:** a Next.js dashboard for Nagpur, Akola, Amravati, Chandrapur, Pune and Mumbai, with provider weather and historical ANN replay.
- **Optional theory:** the original deeper PyTorch ensemble, fuzzy inference and GA, retained as legacy work.

## Run locally

Use Python 3.11+ and Node 22+. From the repository root:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-api.txt
.venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open <http://localhost:3000>; API docs: <http://127.0.0.1:8000/docs>. Set `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` in `frontend/.env.local`. Packaged snapshots render while the API starts. Page requests never trigger training.

## Reproduce the evidence

The existing hourly cache is sufficient; a normal experiment run needs no weather network requests.

```powershell
.venv/Scripts/python.exe -m src.full_dataset_study
.venv/Scripts/python.exe -m src.seasonal_experiment
.venv/Scripts/python.exe -m src.provider_benchmark
.venv/Scripts/python.exe scripts/export_evidence.py
.venv/Scripts/python.exe scripts/build_lab_notebook.py
.venv/Scripts/python.exe -m pytest -q
```

`python -m src.pipeline --prepare-only` rebuilds the seasonal training table. Add `--fetch` to acquire missing ERA5 seasons. The explicit acquisition commands are `python -m src.seasonal_data` and `python -m src.seasonal_data --provider nasa`. No source is silently substituted. Refresh current weather with `python scripts/fetch_weather_snapshot.py`.

Current outputs live in `artifacts/heatline/`, with identical public JSON in `frontend/public/data/`. `provenance.json` records raw file hashes, versions, configuration and seeds. The guided notebook is already executed.

## Models and data

The primary lab version is `heatline-full-dataset-v3`. The sources remain independent:

| Pipeline | Inputs | Architecture | Daily boundary |
| --- | ---: | --- | --- |
| ERA5 via Open-Meteo | 26 lagged, rolling, trend and calendar features | 26 → 16 → 1 | Asia/Kolkata |
| NASA POWER comparison | 52 lagged and calendar features | 52 → 16 → 1 | UTC |

Each has its own scaler, weights, source-derived targets, calibration and decision thresholds. Neither source is substituted into the other network. The model-score threshold is distinct from the heatwave proxy rule: Tmax ≥45°C OR (Tmax ≥40°C and departure from the source's reference normal ≥4.5°C). These are grid-based proxy outcomes, not official IMD warning records.

The same **4,186 April–June rows from 1981–2026** per source fit normalization, class weights, the ANN, score calibration and the F1 threshold. Training uses 300 fixed epochs without reserved stopping years. All rows are checked afterwards; May results filter the fitted data to **1,426 dates**. These are **in-sample training-dataset results**, not estimates of future performance. The 1981–2010 climate normal remains the fixed target definition, not a reserved training partition. See [evaluation protocol](docs/EVALUATION.md).

| Full-dataset May fit | Accuracy | Detected / positive days | False alarms | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ERA5 ANN | 95.16% | 55 / 86 | 38 | 59.14% | 63.95% | 0.6145 |
| NASA POWER ANN | 87.94% | 468 / 499 | 141 | 76.85% | 93.79% | 0.8448 |

`full_dataset_fit_metrics.csv` includes all-season and May results for the ANN, logistic regression, perceptron, sigmoid delta rule, persistence, climatology and reference baselines. `full_dataset_benchmark.json` compares the previous and current full-data checkpoints on identical May rows, including a calibration-only control.

The additional `heatline-walk-forward-v2` experiment remains unchanged. Six chronological folds evaluate **961 May days in 1996–2026** using earlier fitting, stopping and calibration years:

| Held-out May results | Detected / positive days | False alarms | Precision | Recall | F1 | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ERA5 ANN | 33 / 50 | 43 | 43.42% | 66.00% | 0.5238 | 0.4444 |
| NASA POWER ANN | 230 / 293 | 90 | 71.88% | 78.50% | 0.7504 | 0.8062 |

Source-derived positive populations differ. These scores do not establish which source better predicts official warnings. Persistence remains a strong baseline; complete comparisons and confusion counts are published in `full_dataset_metrics.csv`. ERA5 remains primary, supported by a limited paired station audit. NASA is an independent source-sensitivity comparison.

## Application

- Dashboard: city forecast, hourly-weather entry and direct historical-date selection.
- Regional outlook: six-city forecasts, date controls, hourly profiles and city comparisons.
- May history: source-specific predictions, observed outcomes, classification margins, temperature criteria and saved neuron calculations.
- Network: selected-source architecture, initialization, real training curves and a separate 2–2–1 lab exercise.
- Results: metrics and baseline comparisons, with dataset, protocol and lab mapping in compact reference panels.

The original fixed-fit ERA5 experiment is available through the archived model-run selector. Its 26-input preprocessing and weights remain independent from current fold models. The previous common-input study is preserved in `legacy/common-input-study-v1.zip`.

Primary weights and preprocessing live in `artifacts/full-dataset-fit/`; source/year computations cover May 1981–2026 in `artifacts/heatline/full_fit/{source}/{year}.json`. The additional chronological checkpoints remain in `artifacts/full-dataset-study/`, with traces in `artifacts/heatline/study/{source}/{year}.json`. Matching public JSON records source, model version, feature order, fit periods, calibration and availability. Page requests never train a model.

Read-only interfaces include `/api/full_dataset_fit`, `/api/full_fit/{source}/{year}`, `/api/full_dataset_benchmark`, `/api/study`, `/api/study/{source}/{year}`, weather, archive and learning endpoints. See [API documentation](api/README.md). Provider weather remains separate from historical ANN predictions; live ANN prediction is unavailable until compatible timely inputs receive validation. Cached forecasts display their retrieval time and freshness.

- [Original scientific report](docs/ACADEMIC_REPORT.md)
- [Academic appendix](docs/ACADEMIC_APPENDIX.md)
- [Lab notebook](notebooks/03_heatline_lab.ipynb)

## Verification and deployment

```powershell
cd frontend
npm run typecheck
npm run build
npx playwright install chromium
npx playwright test
```

Python checks cover the scalar fixture, finite differences, independent PyTorch gradients, XOR, serialization, deterministic seeds, temporal leakage, training-only preprocessing, label boundaries, missing history and API behavior. Browser checks cover five views, accessibility, data values, forecast/replay distinctions, mobile layout, reduced motion, stale caches and API failure.

The [deployment guide](docs/DEPLOYMENT.md) covers Vercel Hobby, Render Free, CORS and refresh scheduling. GitHub Actions verifies the code and refreshes weather every eight hours after the workflow reaches the default branch. Schedules may be delayed; displayed freshness remains authoritative. Render Free sleeps when idle and has ephemeral storage, so the committed cache supplies a durable fallback.

## Legacy preservation

`legacy/original-experiment.zip` preserves the original source and results; `legacy/heatline-v1-experiment.zip` preserves the first corrected experiment before seasonal revision. Repository-root `models/`, `results/`, older notebooks and Streamlit are legacy. Do not combine their weights or metrics with current preprocessing. Current entry points are `src.full_dataset_fit`, `api.main` and `frontend/`. `src.full_dataset_study` reproduces the additional chronological experiment. The fixed-fit experiment remains available through `src.seasonal_experiment`.

Data: [Open-Meteo](https://open-meteo.com/), CC BY 4.0, with ECMWF ERA5 reanalysis for the assessed historical study. This educational prototype does not issue official heat warnings.

## Course submission package

Run `python -m scripts.package_submission --roll TEAM_LEADER_ROLL --report-dir PATH_TO_REPORT_FILES --output PATH_TO_ZIP`. The archive contains `MiniProject<roll>/Report` and `Code`, with a SHA-256 manifest. Local dependencies, credentials, caches, editor instructions, private scripts and unused archived assets are excluded. Cached data, saved model weights and the historical fixtures needed by regression tests are retained for reproduction.
