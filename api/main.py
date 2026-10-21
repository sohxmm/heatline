"""Serve reproducible artifacts without training or provider calls in requests.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
Forecast snapshots are refreshed separately by scripts/fetch_weather_snapshot.py.
"""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.cities import CITIES, CITY_IDS, TIMEZONE

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ARTIFACTS = ("model", "history", "experiments", "learning", "outlook")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def envelope(mode: str, *, availability: str = "unavailable", reason: str | None = None) -> dict:
    return {
        "schema_version": "1.0",
        "source": "Heatline reproducible artifacts",
        "mode": mode,
        "issued_at": None,
        "target_date": None,
        "model_version": None,
        "availability": availability,
        "availability_reason": reason,
        "timezone": TIMEZONE,
    }


class ArtifactStore:
    """mtime-aware cache; copies prevent query filters from mutating shared data."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._cache: dict[str, tuple[tuple[int, int], dict]] = {}

    def read(self, name: str) -> dict:
        if name not in REQUIRED_ARTIFACTS:
            raise ValueError("Unknown artifact")
        path = self.directory / f"{name}.json"
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._cache.get(name)
        if cached is None or cached[0] != signature:
            with path.open(encoding="utf-8-sig") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError("Artifact must be a JSON object")
            self._cache[name] = (signature, value)
        return copy.deepcopy(self._cache[name][1])


def forecast_freshness(data: dict, now: datetime, max_age_hours: float) -> dict:
    """Expired snapshots remain readable but never masquerade as current forecasts."""
    if data.get("availability") == "unavailable":
        return data
    issued = parse_time(data.get("retrieved_at") or data.get("issued_at"))
    age_hours = (now - issued).total_seconds() / 3600 if issued else None
    expired = age_hours is None or age_hours > max_age_hours or age_hours < -0.25
    if expired:
        data["availability"] = "stale"
        data["availability_reason"] = "Cached weather snapshot is older than the freshness window or its timestamp is invalid."
    data["freshness"] = {
        "checked_at": now.isoformat(),
        "max_age_hours": max_age_hours,
        "age_hours": round(age_hours, 2) if age_hours is not None else None,
    }
    for city in data.get("cities", []):
        if city.get("availability") == "unavailable":
            continue
        city_issued = parse_time(city.get("retrieved_at") or city.get("issued_at") or data.get("issued_at"))
        city_age = (now - city_issued).total_seconds() / 3600 if city_issued else None
        if city_age is None or city_age > max_age_hours or city_age < -0.25:
            city["availability"] = "stale"
            city["availability_reason"] = "Cached city forecast is outside the freshness window."
    return data


def create_app(data_dir: str | Path | None = None, clock: Callable[[], datetime] = utc_now) -> FastAPI:
    application = FastAPI(
        title="Heatline API", version="1.0.0",
        description="Read-only Nagpur-May ANN replay and separately sourced regional weather. No live ANN prediction endpoint.",
    )
    directory = Path(data_dir or os.getenv("HEATLINE_DATA_DIR", str(ROOT / "artifacts" / "heatline")))
    store = ArtifactStore(directory)
    application.state.store = store
    origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if origin.strip()]
    application.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["Accept", "Content-Type"], allow_credentials=False)
    application.add_middleware(GZipMiddleware, minimum_size=1000)
    max_age = float(os.getenv("WEATHER_MAX_AGE_HOURS", "12"))

    @application.middleware("http")
    async def cache_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "public, max-age=60" if request.url.path.startswith("/api/") and response.status_code == 200 else "no-store"
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        result = envelope("educational", reason="Invalid query parameters.")
        result["errors"] = [{"field": ".".join(map(str, error["loc"])), "message": error["msg"]} for error in exc.errors()]
        return JSONResponse(result, status_code=422)

    def read_artifact(name: str, mode: str):
        try:
            data = store.read(name)
        except (OSError, ValueError, json.JSONDecodeError):
            return JSONResponse(envelope(mode, reason=f"{name.capitalize()} artifact is unavailable. Run the documented artifact generation command."), status_code=503)
        return {**envelope(mode, availability="available"), **data}

    @application.get("/health")
    def health():
        return {**envelope("educational", availability="available"), "status": "ok", "issued_at": clock().isoformat(), "service": "heatline-api", "version": "1.0.0"}

    @application.get("/health/ready")
    def readiness():
        present, missing = [], []
        for name in REQUIRED_ARTIFACTS:
            try:
                store.read(name)
                present.append(name)
            except (OSError, ValueError):
                missing.append(name)
        result = {**envelope("educational", availability="available" if not missing else "unavailable"), "status": "ready" if not missing else "incomplete", "artifacts": present, "missing": missing}
        return JSONResponse(result, status_code=200 if not missing else 503)

    @application.get("/api/cities")
    def cities():
        return {**envelope("educational", availability="available"), "source": "Heatline declared regional coverage v1", "cities": [{**city, "timezone": TIMEZONE, "assessed_core": city["id"] == "nagpur", "ann_scope": "May historical replay only" if city["id"] == "nagpur" else "Unavailable: no independently validated regional ANN"} for city in CITIES]}

    @application.get("/api/outlook")
    def outlook(city: str | None = Query(default=None)):
        if city is not None and city not in CITY_IDS:
            return JSONResponse(envelope("forecast", reason="Unknown city. See /api/cities for supported identifiers."), status_code=404)
        data = read_artifact("outlook", "forecast")
        if isinstance(data, JSONResponse):
            return data
        data = forecast_freshness(data, clock(), max_age)
        if city:
            data["cities"] = [item for item in data.get("cities", []) if item.get("id") == city]
            if not data["cities"]:
                data.update(availability="unavailable", availability_reason="This city's weather is missing from the current snapshot.")
            elif data["cities"][0].get("availability") in ("stale", "unavailable"):
                data.update(availability=data["cities"][0]["availability"], availability_reason=data["cities"][0].get("availability_reason"))
        return data

    @application.get("/api/history")
    def history(year: int | None = Query(default=None, ge=1981, le=2026)):
        data = read_artifact("history", "replay")
        if isinstance(data, JSONResponse):
            return data
        records = data.get("records", [])
        data["available_years"] = sorted({int(str(row.get("date", ""))[:4]) for row in records if str(row.get("date", ""))[:4].isdigit()})
        if year is not None:
            data["records"] = [row for row in records if str(row.get("date", "")).startswith(f"{year}-")]
            data["selected_year"] = year
            if not data["records"]:
                data.update(availability="unavailable", availability_reason="No historical predictions are packaged for this year.")
        return data

    @application.get("/api/model")
    def model():
        return read_artifact("model", "educational")

    @application.get("/api/experiments")
    def experiments():
        return read_artifact("experiments", "educational")

    @application.get("/api/learning")
    def learning():
        return read_artifact("learning", "educational")

    return application


app = create_app()
