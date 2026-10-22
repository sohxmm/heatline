"""Contract and failure tests; no network requests and no tracked-artifact writes."""
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from scripts.fetch_weather_snapshot import convert_city, failed_snapshot, fetch_snapshot, write_snapshot
from api.cities import CITIES

NOW = datetime(2026, 5, 15, 6, tzinfo=timezone.utc)


@pytest.fixture
def artifacts(tmp_path):
    meta = {"source": "Test fixture", "mode": "educational", "issued_at": NOW.isoformat(), "target_date": None, "model_version": "test-v1", "availability": "available"}
    for name in ("model", "experiments", "learning"):
        write_snapshot(tmp_path / f"{name}.json", meta)
    write_snapshot(tmp_path / "history.json", {**meta, "mode": "replay", "records": [{"date": "2019-05-01", "actual": 1}, {"date": "2026-05-01", "actual": 0}]})
    write_snapshot(tmp_path / "outlook.json", {**meta, "mode": "forecast", "cities": [{"id": city["id"], "name": city["name"], "availability": "available", "issued_at": NOW.isoformat(), "daily": [{"date": "2026-05-15", "tmax": 41}], "ann": {"availability": "unavailable", "reason": "Unvalidated live inputs"}} for city in CITIES]})
    return tmp_path


def test_contract_modes_and_read_only_api(artifacts):
    with TestClient(create_app(artifacts, clock=lambda: NOW)) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/health/ready").status_code == 200
        cities = client.get("/api/cities").json()["cities"]
        assert len(cities) == 6
        assert sum(city["assessed_core"] for city in cities) == 1
        for name, mode in (("outlook", "forecast"), ("history", "replay"), ("learning", "educational"), ("model", "educational"), ("experiments", "educational")):
            response = client.get(f"/api/{name}")
            data = response.json()
            assert response.status_code == 200
            assert data["mode"] == mode
            assert {"source", "issued_at", "target_date", "model_version", "availability"} <= data.keys()
            assert response.headers["cache-control"] == "public, max-age=60"
            assert client.post(f"/api/{name}").status_code == 405


def test_filters_do_not_mutate_cached_artifacts(artifacts):
    with TestClient(create_app(artifacts, clock=lambda: NOW)) as client:
        selected = client.get("/api/outlook?city=nagpur").json()
        assert [city["id"] for city in selected["cities"]] == ["nagpur"]
        assert len(client.get("/api/outlook").json()["cities"]) == 6
        assert selected["cities"][0]["ann"]["availability"] == "unavailable"
        assert client.get("/api/history?year=2026").json()["records"] == [{"date": "2026-05-01", "actual": 0}]
        assert len(client.get("/api/history").json()["records"]) == 2
        assert client.get("/api/history?year=2025").json()["availability"] == "unavailable"
        assert client.get("/api/history?year=2027").status_code == 422
        assert client.get("/api/outlook?city=unknown").status_code == 404


def test_replay_details_are_partitioned_and_bounded(artifacts):
    (artifacts/'replay').mkdir()
    payload={'source':'Test fixture','mode':'replay','model_version':'test-v1','availability':'available','issued_at':NOW.isoformat(),'target_date':None,'records':[{'date':'2026-05-20','inputs':{'tmax_lag1':43.2}}]}
    write_snapshot(artifacts/'replay/2026.json',payload)
    with TestClient(create_app(artifacts)) as client:
        data=client.get('/api/replay/2026').json()
        assert all(data[key]==value for key,value in payload.items())
        assert client.get('/api/replay/1980').status_code==422
        assert client.get('/api/replay/2027').status_code==422
        assert client.get('/api/replay/nonsense').status_code==422
        missing=client.get('/api/replay/2025')
        assert missing.status_code==503 and missing.json()['availability']=='unavailable'
        assert client.post('/api/replay/2026').status_code==405


def test_stale_forecast_and_replay_are_distinct(artifacts):
    with TestClient(create_app(artifacts, clock=lambda: NOW + timedelta(days=2))) as client:
        snapshot = client.get("/api/outlook").json()
        assert snapshot["availability"] == "stale"
        assert all(city["availability"] == "stale" for city in snapshot["cities"])
        assert snapshot["issued_at"] == NOW.isoformat()
        # A retrospective experiment does not become invalid every 12 hours.
        assert client.get("/api/history").json()["availability"] == "available"


def test_city_freshness_is_checked_independently(artifacts):
    snapshot = json.loads((artifacts / "outlook.json").read_text())
    snapshot["cities"][0]["issued_at"] = (NOW - timedelta(days=1)).isoformat()
    write_snapshot(artifacts / "outlook.json", snapshot)
    with TestClient(create_app(artifacts, clock=lambda: NOW)) as client:
        assert client.get("/api/outlook?city=nagpur").json()["availability"] == "stale"
        assert client.get("/api/outlook?city=mumbai").json()["availability"] == "available"


def test_missing_or_corrupt_artifact_is_explicitly_unavailable(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health/ready").status_code == 503
        missing = client.get("/api/model")
        assert missing.status_code == 503
        assert missing.json()["availability"] == "unavailable"
        (tmp_path / "model.json").write_text("not-json")
        assert client.get("/api/model").status_code == 503
        assert client.get("/api/history?year=nonsense").json()["availability"] == "unavailable"


def test_cors_is_restricted_to_configured_origins(artifacts, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://heatline.example")
    with TestClient(create_app(artifacts)) as client:
        response = client.get("/api/model", headers={"Origin": "https://heatline.example"})
        assert response.headers["access-control-allow-origin"] == "https://heatline.example"
        other = client.get("/api/model", headers={"Origin": "https://other.example"})
        assert "access-control-allow-origin" not in other.headers


def weather_fixture():
    # Synthetic values are deliberately confined to test fixtures.
    return {"latitude": 21.25, "longitude": 79.0, "elevation": 311,
            "hourly": {"time": ["2026-05-15T12:00"], "temperature_2m": [40.0], "apparent_temperature": [42.0], "relative_humidity_2m": [35], "wind_speed_10m": [10]},
            "daily": {"time": ["2026-05-15"], "temperature_2m_max": [40.0], "temperature_2m_min": [27.0], "apparent_temperature_max": [42.0], "precipitation_probability_max": [5.0], "wind_speed_10m_max": [10.0]}}


def test_provider_conversion_keeps_weather_and_ann_distinct():
    raw = weather_fixture()
    city = convert_city(CITIES[0], raw, NOW.isoformat())
    assert city["hourly"][0]["time"].endswith("+05:30")
    assert city["daily"][0]["apparent_tmax"] == 42.0
    assert city["daily"][0]["temperature_indicator"]["exceeds_threshold"] is True
    assert city["daily"][0]["heatwave_indicator"]["status"] == "unavailable"
    assert city["ann"]["availability"] == "unavailable"
    raw["hourly"]["temperature_2m"] = []
    with pytest.raises(ValueError, match="inconsistent"):
        convert_city(CITIES[0], raw, NOW.isoformat())


def test_fetch_contract_has_snapshot_issue_basis():
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [weather_fixture() for _ in CITIES]

    class Session:
        @staticmethod
        def get(url, params, timeout):
            assert url == "https://api.open-meteo.com/v1/forecast"
            assert params["timezone"] == "Asia/Kolkata"
            assert params["forecast_days"] == 7
            return Response()

    data = fetch_snapshot(session=Session, now=NOW)
    assert data["availability"] == "available"
    assert len(data["cities"]) == 6
    assert data["provider_run_issued_at"] is None
    assert "retrieval" in data["issue_time_basis"]


def test_failed_refresh_preserves_values_and_original_issue_time(artifacts):
    previous = json.loads((artifacts / "outlook.json").read_text())
    data = failed_snapshot(artifacts / "outlook.json", TimeoutError())
    assert data["availability"] == "stale"
    assert data["issued_at"] == previous["issued_at"]
    assert data["cities"][0]["daily"] == previous["cities"][0]["daily"]
    assert "last_refresh_attempt" in data


def test_first_refresh_failure_does_not_invent_forecasts(tmp_path):
    data = failed_snapshot(tmp_path / "missing.json", TimeoutError())
    assert data["availability"] == "unavailable"
    assert data["issued_at"] is None
    assert all(city["daily"] == [] for city in data["cities"])
