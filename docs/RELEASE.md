# Heatline release

Public frontend: https://heatline-six.vercel.app

The current source-specific experiment is `heatline-walk-forward-v2`: ERA5 26–16–1, NASA POWER 52–16–1, independent preprocessing and calibration. The calendar uses saved, held-out fold computations; final all-data research fits are separate. Current provider forecasts display cached issue times and freshness. Hosted ANN live prediction remains unavailable.

The API serves read-only artifacts. Published snapshots keep the frontend usable without a hosted backend. Local verification covers gradients, causal inputs, source-specific forward calculations, API behavior, calendar interactions, mobile layout and accessibility.

Run `python scripts/package_heatline.py` for a source-only ZIP or `python -m scripts.package_submission --roll TEAM_LEADER_ROLL --report-dir PATH --output PATH` for the course folder structure. Both exclude local state and include a SHA-256 source manifest.
