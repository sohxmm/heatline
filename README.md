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
.venv/Scripts/python.exe -m src.experiment
.venv/Scripts/python.exe scripts/export_evidence.py
.venv/Scripts/python.exe scripts/build_lab_notebook.py
.venv/Scripts/python.exe -m pytest -q
```

`python -m src.pipeline --prepare-only` rebuilds daily records, reference normals, labels and features. Add `--fetch` to fetch missing ERA5 files; another reanalysis model is never silently substituted. Refresh provider weather with `python scripts/fetch_weather_snapshot.py`.

Current outputs live in `artifacts/heatline/`, with identical public JSON in `frontend/public/data/`. `provenance.json` records raw file hashes, versions, configuration and seeds. The guided notebook is already executed.

## Method and measured results

Training and preprocessing use **1981–2010**; architecture/objective selection uses **2011–2014**; Platt scaling and F1 threshold selection use **2015–2018**. The **2019–2026** period is retrospective evaluation because the legacy project already examined it. Frozen reference normals use 1981–2010. April initializes May lags; missing or duplicate dates fail explicitly.

The versioned `nagpur-grid-proxy-v1` target is Tmax ≥45°C OR (Tmax ≥40°C and reference departure ≥4.5°C). This single-grid daily proxy is inspired by [IMD criteria](https://mausam.imd.gov.in/pdfs/heatcolduser/morning_heat_bulletin.pdf); it is not an official station-based declaration.

The study compares binary/bipolar sigmoid, 8/16/32 hidden neurons, both consistent objectives, seeds 7/21/42 and controlled learning factors. Mean selection-period average precision chooses the configuration; fixed seed 42 supplies the primary model. Retrospective scores do not guide selection.

| Model | Average precision | F1 | Brier |
| --- | ---: | ---: | ---: |
| Selected NumPy ANN (cross-entropy) | 0.3636 | 0.3333 | 0.0246 |
| Textbook NumPy ANN (half-squared-error) | 0.4375 | 0.5000 | 0.0251 |
| Persistence | 0.1809 | 0.4000 | 0.0363 |
| Sigmoid delta rule | 0.4886 | 0.3333 | 0.0221 |
| Logistic regression | 0.4991 | 0.3333 | 0.0223 |

There are only **8 positives in 248 retrospective days**. Simpler models win some metrics. Neither a sigmoid score, an ensemble nor Platt scaling guarantees calibrated probabilities. Full confusion matrices, raw/transformed Brier scores, reliability bins, missed dates, climatology/perceptron comparisons and every measured run are published.

- [Academic report](docs/ACADEMIC_REPORT.md): problem, inputs, neuron, rules, architecture, derivatives, experiments, results and limitations.
- [Academic appendix](docs/ACADEMIC_APPENDIX.md): supplied experiment numbering/COs and the independently calculated 2–2–1 update.
- [Executed notebook](notebooks/03_heatline_lab.ipynb): all seven supporting examples and measured results.

## Application and API

Five views: landing, regional outlook, Nagpur May explorer, learning walkthrough and evidence. The graphite/amber interface uses Manrope, Instrument Serif, Motion with reduced-motion support, shadcn controls and Bklit charts. See `frontend/THIRD_PARTY_NOTICES.md` for licenses.

Read-only endpoints: `GET /api/cities`, `/api/outlook`, `/api/history`, `/api/model`, `/api/experiments`, `/api/learning`, `/health`, `/health/ready`. Responses carry source, mode, issue time, target dates, version and availability. See [API documentation](api/README.md).

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

`legacy/original-experiment.zip` preserves the original source, processed data, weights and results. Repository-root `models/`, `results/`, older notebooks and Streamlit belong to that experiment. Do not combine their weights or metrics with the corrected target/scaler. Their adjusted labels, future-inclusive normals and inconsistent loss reporting are documented in the archive README. Current entry points are `src.experiment`, `api.main` and `frontend/`.

Data: [Open-Meteo](https://open-meteo.com/), CC BY 4.0, with ECMWF ERA5 reanalysis for the assessed historical study. This educational prototype does not issue official heat warnings.
