# Heatline public API

The API serves experiment artifacts and cached provider weather. It never trains a
network, invokes a weather provider during a request, or claims that the assessed
ANN is a validated live weather forecaster.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-api.txt
.venv/Scripts/python.exe scripts/fetch_weather_snapshot.py
.venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Interactive API documentation: <http://127.0.0.1:8000/docs>.

| GET endpoint | Payload |
| --- | --- |
| `/health` | Web-process health; does not imply fresh weather |
| `/health/ready` | Checks whether all five artifact files can be read; 503 if incomplete |
| `/api/cities` | Six supported cities and assessed ANN scope |
| `/api/outlook?city=nagpur` | All city forecasts, or one city in the same `cities` array |
| `/api/history?year=2026` | Replay `records`, optionally filtered to one year |
| `/api/replay/2026` | Per-day inputs, frozen normalization and saved network computations for one May |
| `/api/model` | Architecture, preprocessing, version, metrics and limitations |
| `/api/experiments` | Measured model comparisons and controlled studies |
| `/api/learning` | Educational examples and numerical backpropagation trace |

Successful data responses include `source`, `mode`, `issued_at`, `target_date`,
`model_version`, and `availability` (`available`, `stale`, or `unavailable`).
`availability_reason` supplies context. `target_date` is null when one date does
not describe the response; individual forecast days and replay records carry dates.
Historical experiments use `replay`; learning and methodology use `educational`.

Missing or malformed artifact files return 503 with unavailable metadata. Unknown
city identifiers return 404. Invalid year queries return 422. Methods other than
GET are not supported on data endpoints. Year filters never modify the cached
source artifact. A year with no packaged predictions returns an empty array and
explicit unavailable state.

## Weather provenance and freshness

`scripts/fetch_weather_snapshot.py` requests seven days of daily and hourly
[Open-Meteo](https://open-meteo.com/en/docs) `best_match` provider forecasts for
Nagpur, Akola, Amravati, Chandrapur, Pune and Mumbai. The exact same JSON is
published to `artifacts/heatline/outlook.json` and
`frontend/public/data/outlook.json`. Temperatures are Celsius; local times are
Asia/Kolkata. Grid coordinates are recorded separately from requested coordinates.

`issued_at` is the **snapshot retrieval time**, explicitly identified by
`issue_time_basis`. The combined provider API does not expose a single model-run
initialization time, so `provider_run_issued_at` remains null. Snapshot metadata
must not be presented as a provider model issue time. The ANN field remains
unavailable for live/regional predictions.

The heatwave indicator remains unavailable without validated compatible local
climatology. The separate `temperature_indicator` is only a 40°C inland / 37°C
Mumbai temperature screening threshold. It is neither an official heatwave
declaration nor an ANN prediction. Apparent temperature is separately supplied
by the weather provider and is not used as a heatwave label.

Forecast snapshots older than 12 hours are marked stale at read time; timestamps
in the future or absent timestamps are also treated as invalid/stale. Historical
replay artifacts do not expire on this weather schedule. A failed refresh retains
previous weather values and their original issue time, marks them stale, records
the attempted refresh time, and exits with code 1. A first-run failure yields an
unavailable response with empty forecast arrays. No values are invented.

## Deployment

The repository-root `render.yaml` declares one Render **Free** Python web service
with `/health` as its health check. No database or paid disk is required. Upload
the generated artifacts with the application; this service consumes them as
read-only files. Set `CORS_ORIGINS` to the exact deployed frontend origin (multiple
origins can be comma separated). Set `NEXT_PUBLIC_API_URL` in Vercel to the Render
service URL, without `/api` appended.

| Variable | Default |
| --- | --- |
| `HEATLINE_DATA_DIR` | `artifacts/heatline` under the repository |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` |
| `WEATHER_MAX_AGE_HOURS` | `12` |

Render Free can sleep when idle. The frontend should first try the API with a
bounded timeout and then display its packaged snapshot with source and freshness
states. A cold start must not relabel a packaged snapshot as newly issued.
Refresh snapshots on a schedule and deploy/publish the updated cache; a static
snapshot is intentionally marked stale until refreshed. Render's filesystem is
ephemeral, so a local runtime write is not a durable snapshot publication.

Verification: `.venv/Scripts/python.exe -m pytest tests/test_api.py -q` uses only
temporary directories and synthetic test fixtures; it makes no network calls.
