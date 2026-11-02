# Heatline release

Public frontend: **https://heatline-six.vercel.app**

The first Vercel production release uses the packaged, timestamped weather and
experiment snapshots. The interface identifies this saved-data mode. The
FastAPI service is prepared for Render Free; its public origin must be assigned
and configured in `NEXT_PUBLIC_API_URL` before claiming hosted API integration.

The assessed experiment, guided notebook and report are complete. The selected
ANN is a Nagpur May historical study; public provider forecasts supply current
regional weather. Live/regional ANN output remains unavailable.

Local validation: 83 Python tests; nine browser tests covering five views,
accessibility, mobile/reduced motion, fixture values, forecast/replay distinction,
stale data and API failure. Next.js production build and TypeScript checks pass.
GitHub CI additionally checks a fresh Linux checkout.
The explorer additionally checks day selection, year changes, exact per-day
network calculations and unavailable detail snapshots. Its calendar covers May
1981–2026, with inputs and forward computations loaded one year at a time.

The seasonal v2 release expands training to 2,730 April–June days, keeps the
original May ERA5 targets unchanged, and averages three independently trained
NumPy networks. The selected configuration uses 16 bipolar hidden units and
weighted half-squared-error. Sensitivity F2 and balanced F1 policies are fitted
on calibration years only. The evidence page reports yearly misses/false alarms,
onsets, bootstrap intervals, source audits and an independent NASA POWER benchmark.
The original v1 is preserved in `legacy/heatline-v1-experiment.zip`.

Submission archive: run `python scripts/package_heatline.py` to create
`dist/heatline-submission.zip` with source, cached data, evidence and SHA-256
manifest. The package excludes credentials, environment overrides, dependency
folders, build caches and temporary logs.

Scheduled refresh starts after the workflow reaches the default branch. Until
then, refresh manually and publish both cache copies. Snapshot freshness is
checked independently in the interface and API.
