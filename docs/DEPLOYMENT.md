# Deployment and operation

## Frontend · Vercel Hobby

Import the GitHub repository, choose **frontend** as the root directory, and use
the Next.js framework preset. Build command: `npm run build`; install: `npm ci`.
Set `NEXT_PUBLIC_API_URL` to the Render HTTPS origin, with no `/api` suffix.
Without this variable, the application reads its packaged public snapshots.
Run `vercel --prod` from `frontend/` for an authenticated CLI deployment.

## API · Render Free

`render.yaml` declares a Python Free web service. Select the release branch and
apply the blueprint, or create a Free Python service with:

```text
Build: pip install -r requirements-api.txt
Start: uvicorn api.main:app --host 0.0.0.0 --port $PORT
Health check: /health
```

Set `CORS_ORIGINS` to the exact Vercel origin. Set `HEATLINE_DATA_DIR` to
`artifacts/heatline`; no database or disk is needed. `/health/ready` checks the
five required JSON resources. Requests never train a network or fetch weather.
The service includes the committed cache, so no runtime writes need to persist.

Render Free can sleep after 15 minutes of inactivity and take roughly one minute
to start. The frontend loads a saved snapshot first and bounds the API request
at eight seconds. It identifies the fallback, preserves its issue time and
marks old forecast snapshots stale. See [Render limitations](https://render.com/docs/free).

## Snapshot refresh

`.github/workflows/weather-snapshot.yml` runs every eight hours, and supports
manual dispatch. Scheduled runs execute only after the workflow reaches the
repository default branch. The workflow fetches all six cities once and commits
the public cache to `artifacts/heatline/outlook.json` and
`frontend/public/data/outlook.json`. Bot commits use `[skip ci]`; the dedicated
verification workflow also ignores cache-only changes.

Three snapshot commits per day can trigger three frontend/backend deploys per
day if Git deployment is enabled. Verify the current account limits before
enabling additional deploy hooks; no paid service is required by this design.
GitHub schedules can be delayed or disabled after prolonged repository
inactivity. The interface's 12-hour freshness state remains authoritative.

On provider failure, the fetch script retains the previous values and original
timestamps, records the failed attempt, marks the cache stale and exits nonzero.
The workflow persists that state and then reports failure. It never generates
substitute weather values. A first-run failure returns unavailable empty arrays.

## Release checks

```powershell
python -m pytest -q
cd frontend
npm ci
npm run typecheck
npm run build
npx playwright install chromium
npx playwright test
```

Confirm public landing/outlook/explorer/learn/evidence pages, source timestamps,
city/year controls, reduced motion, mobile navigation and the API readiness
response. Provider weather is distinct from ANN replay. Live/regional ANN fields
must stay unavailable until compatible timely inputs have separate validation.

Keep secrets in platform environment settings. Public snapshots contain only
weather and published experiment evidence. Neither the résumé nor local account
credentials belong in the repository or deployment bundle.
