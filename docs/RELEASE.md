# Heatline release

Public frontend: **https://heatline-six.vercel.app**

The first Vercel production release uses the packaged, timestamped weather and
experiment snapshots. The interface identifies this saved-data mode. The
FastAPI service is prepared for Render Free; its public origin must be assigned
and configured in `NEXT_PUBLIC_API_URL` before claiming hosted API integration.

The assessed experiment, guided notebook and report are complete. The selected
ANN is a Nagpur May historical study; public provider forecasts supply current
regional weather. Live/regional ANN output remains unavailable.

Local validation: 72 Python tests; six browser tests covering five views,
accessibility, mobile/reduced motion, fixture values, forecast/replay distinction,
stale data and API failure. Next.js production build and TypeScript checks pass.
GitHub CI additionally checks a fresh Linux checkout.

Submission archive: run `python scripts/package_heatline.py` to create
`dist/heatline-submission.zip` with source, cached data, evidence and SHA-256
manifest. The package excludes credentials, environment overrides, dependency
folders, build caches and temporary logs.

Scheduled refresh starts after the workflow reaches the default branch. Until
then, refresh manually and publish both cache copies. Snapshot freshness is
checked independently in the interface and API.
