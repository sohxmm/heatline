# Heatline release

Public frontend: https://heatline-six.vercel.app

The primary experiment is `heatline-full-dataset-v3`: ERA5 26–16–1 and NASA POWER 52–16–1, with independent preprocessing and calibration. All 4,186 seasonal rows per source fit each stage; all three members train for 300 fixed epochs. The calendar defaults to saved in-sample computations for May 1981–2026. Results offer both May and complete April–June checking. The additional chronological v2 experiment remains available and unchanged. Provider forecasts display cached issue times and freshness. Hosted ANN live prediction remains unavailable.

The API serves read-only artifacts. Published snapshots keep the frontend usable without a hosted backend. Local verification covers gradients, causal inputs, source-specific forward calculations, API behavior, calendar interactions, mobile layout and accessibility.

Run `python scripts/package_heatline.py` for a source-only ZIP or `python -m scripts.package_submission --roll TEAM_LEADER_ROLL --report-dir PATH --output PATH` for the course folder structure. Both exclude local state and include a SHA-256 source manifest.
