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
.venv/Scripts/python.exe -m src.seasonal_experiment
.venv/Scripts/python.exe -m src.provider_benchmark
.venv/Scripts/python.exe scripts/export_evidence.py
.venv/Scripts/python.exe scripts/build_lab_notebook.py
.venv/Scripts/python.exe -m pytest -q
```

`python -m src.pipeline --prepare-only` rebuilds the seasonal training table. Add `--fetch` to acquire missing ERA5 seasons. The explicit acquisition commands are `python -m src.seasonal_data` and `python -m src.seasonal_data --provider nasa`. No source is silently substituted. Refresh current weather with `python scripts/fetch_weather_snapshot.py`.

Current outputs live in `artifacts/heatline/`, with identical public JSON in `frontend/public/data/`. `provenance.json` records raw file hashes, versions, configuration and seeds. The guided notebook is already executed.

## Method and measured results

Training and preprocessing use **April–June 1981–2010**; selection uses **May 2011–2014**; Platt scaling and F2/F1 threshold selection use **May 2015–2018**. The **May 2019–2026** period is retrospective evaluation because these outcomes were previously examined. Revised gains are exploratory, not an untouched-test claim. Frozen May reference normals and targets remain unchanged. March initializes seasonal lags; missing or duplicate dates fail explicitly.

The versioned `nagpur-grid-proxy-v1` target is Tmax ≥45°C OR (Tmax ≥40°C and reference departure ≥4.5°C). This single-grid daily proxy is inspired by [IMD criteria](https://mausam.imd.gov.in/pdfs/heatcolduser/morning_heat_bulletin.pdf); it is not an official station-based declaration.

The fixed study compares binary/bipolar sigmoid, 8/16/32 hidden neurons, both consistent objectives, three class-weight modes and seeds 7/21/42. Selection-period average precision of the three-seed mean chooses the configuration. The primary architecture is **26→16→1**, bipolar sigmoid, weighted half-squared-error, η=.03 and μ=.9. Each member is an explicit NumPy backpropagation network; their sigmoid scores are averaged. A single-network comparison remains visible. Training-window and learning-factor ablations do not replace the selected configuration based on retrospective scores.

| ERA5 experiment | Detected / positive days | Recall | False alarms | Average precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous v1 ANN | 2 / 8 | 25.0% | 2 | 0.3636 | 0.3333 |
| Seasonal v2 ANN · F2 policy | 5 / 8 | 62.5% | 12 | 0.4097 | 0.4000 |
| Seasonal v2 ANN · F1 policy | 2 / 8 | 25.0% | 2 | 0.4097 | 0.3333 |
| Persistence | 3 / 8 | 37.5% | 4 | 0.1809 | 0.4000 |
| Seasonal logistic regression · F2 | 4 / 8 | 50.0% | 10 | 0.4627 | 0.3636 |

There are only **8 positives in 248 retrospective ERA5 days**. The F2 policy emphasizes recall at a cost in false alarms; both policies are selected only on calibration years. The source audit verifies 2,806 overlapping days, zero changed May labels, and identical May feature vectors. Full confusion counts, raw/transformed Brier scores, reliability bins, yearly outcomes, onset detections, May-year bootstrap intervals and all ablations are published. Simpler models win some metrics; neither sigmoid, averaging nor Platt scaling guarantees calibration.

The **independent NASA POWER benchmark** has a different daily product and **45 positive days** in the same evaluation years. Its ANN has average precision **0.8580**, and at the calibration-selected F1 threshold detects **41/45**, with **29 false alarms**, precision **0.5857** and F1 **0.7130**. NASA persistence has F1 **0.8222**, so the ANN does not dominate it. These are not replacements for the harder ERA5 target. NASA UTC-day boundaries, coarser meteorology, its own normals and 52 available lag features are documented separately. A limited training-era NOAA station cross-check further demonstrates source/threshold sensitivity.

- [Academic report](docs/ACADEMIC_REPORT.md): problem, inputs, neuron, rules, architecture, derivatives, experiments, results and limitations.
- [Academic appendix](docs/ACADEMIC_APPENDIX.md): supplied experiment numbering/COs and the independently calculated 2–2–1 update.
- [Executed notebook](notebooks/03_heatline_lab.ipynb): all seven supporting examples and measured results.

## Application and API

Five views: landing, regional outlook, Nagpur May explorer, learning walkthrough and evidence. The graphite/amber interface uses Manrope, Instrument Serif, Motion with reduced-motion support, shadcn controls and Bklit charts. See `frontend/THIRD_PARTY_NOTICES.md` for licenses.

Read-only endpoints: `GET /api/cities`, `/api/outlook`, `/api/history`, `/api/replay/{year}`, `/api/model`, `/api/experiments`, `/api/learning`, `/health`, `/health/ready`. Responses carry source, mode, issue time, target dates, version and availability. See [API documentation](api/README.md).

The May explorer includes a calendar and selected-day result. Expand the calculation to inspect 26 recorded inputs, standardized values, each member's hidden/output computation, their mean score and threshold decision. `python -m src.replay` verifies the saved weights/scaler reproduce published scores before exporting. Traces load one year at a time. The evidence view compares F2/F1 policies, yearly errors, source audits and NASA benchmark results.

**Provider forecasts and ANN replay are different products.** Open-Meteo weather covers six cities; apparent temperature and the temperature screen are distinct from the proxy. Live/regional ANN output remains unavailable until timely compatible inputs receive separate validation. [ERA5 arrives with delay](https://open-meteo.com/en/docs/historical-weather-api). Provider snapshot issue time means retrieval time; run initialization time is unknown. Snapshots older than 12 hours remain visibly stale.

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

`legacy/original-experiment.zip` preserves the original source and results; `legacy/heatline-v1-experiment.zip` preserves the first corrected experiment before seasonal revision. Repository-root `models/`, `results/`, older notebooks and Streamlit are legacy. Do not combine their weights or metrics with current preprocessing. Current entry points are `src.seasonal_experiment`, `src.provider_benchmark`, `api.main` and `frontend/`.

Data: [Open-Meteo](https://open-meteo.com/), CC BY 4.0, with ECMWF ERA5 reanalysis for the assessed historical study. This educational prototype does not issue official heat warnings.
