"""Fetch six public weather forecasts once, then atomically publish a shared cache.

python scripts/fetch_weather_snapshot.py

Both API and frontend consume the same snapshot. A failed refresh keeps the last
successful observations and their original timestamps; no weather is fabricated.
Open-Meteo best_match is provider weather, not a Heatline ANN forecast.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.cities import CITIES, TIMEZONE  # noqa: E402

ENDPOINT = "https://api.open-meteo.com/v1/forecast"
SOURCE = "Open-Meteo Forecast API (best_match weather models)"
ANN_REASON = "Live and regional ANN predictions require separate validation with timely compatible inputs. The assessed ANN supports Nagpur May retrospective replay only."
INDICATOR_REASON = "A forecast temperature alone cannot determine the declared heatwave proxy: compatible local reference normals and validation are unavailable here."


def finite(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def metadata(now: str) -> dict:
    return {
        "schema_version": "1.0",
        "source": SOURCE,
        "source_url": "https://open-meteo.com/en/docs",
        "attribution": "Weather data by Open-Meteo, CC BY 4.0",
        "mode": "forecast",
        "issued_at": now,
        "retrieved_at": now,
        "issue_time_basis": "Snapshot retrieval time; the best_match API does not expose a single provider model initialization time.",
        "provider_run_issued_at": None,
        "target_date": None,
        "model_version": "open-meteo-best-match",
        "availability": "available",
        "availability_reason": None,
        "timezone": TIMEZONE,
        "freshness_max_age_hours": 12,
        "units": {"temperature": "°C", "humidity": "%", "wind_speed": "km/h", "precipitation_probability": "%"},
        "assessed_scope": "Nagpur, May; historical ANN replay is a separate data product.",
        "regional_season": "April–June; provider weather remains available outside this season.",
    }


def unavailable_city(city: dict, reason: str) -> dict:
    return {
        **city, "timezone": TIMEZONE, "source": SOURCE, "mode": "forecast",
        "issued_at": None, "retrieved_at": None, "target_date": None,
        "model_version": "open-meteo-best-match", "availability": "unavailable",
        "availability_reason": reason, "daily": [], "hourly": [],
        "ann": {"availability": "unavailable", "reason": ANN_REASON},
    }


def convert_city(city: dict, raw: dict, now: str) -> dict:
    hourly, daily = raw["hourly"], raw["daily"]
    if not hourly.get("time") or not daily.get("time"):
        raise ValueError("Provider returned no forecast dates")
    hourly_keys = ("temperature_2m", "apparent_temperature", "relative_humidity_2m", "wind_speed_10m")
    daily_keys = ("temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "precipitation_probability_max", "wind_speed_10m_max")
    if any(len(hourly.get(key, [])) != len(hourly["time"]) for key in hourly_keys):
        raise ValueError("Provider hourly arrays have inconsistent lengths")
    if any(len(daily.get(key, [])) != len(daily["time"]) for key in daily_keys):
        raise ValueError("Provider daily arrays have inconsistent lengths")
    rows_hourly = [{
        "time": time + "+05:30",  # Requested Asia/Kolkata; no daylight-saving transitions.
        "temperature": finite(hourly["temperature_2m"][index]),
        "apparent_temperature": finite(hourly["apparent_temperature"][index]),
        "humidity": finite(hourly["relative_humidity_2m"][index]),
        "wind_speed": finite(hourly["wind_speed_10m"][index]),
    } for index, time in enumerate(hourly["time"])]
    if not any(row["temperature"] is not None for row in rows_hourly):
        raise ValueError("Provider returned no usable hourly temperature values")
    rows_daily = []
    threshold = 37.0 if city["id"] == "mumbai" else 40.0
    for index, date in enumerate(daily["time"]):
        tmax = finite(daily["temperature_2m_max"][index])
        humidities = [row["humidity"] for row in rows_hourly if row["time"].startswith(date) and row["humidity"] is not None]
        rows_daily.append({
            "date": date,
            "tmax": tmax,
            "tmin": finite(daily["temperature_2m_min"][index]),
            "apparent_tmax": finite(daily["apparent_temperature_max"][index]),
            "precipitation_probability": finite(daily["precipitation_probability_max"][index]),
            "humidity_mean": round(sum(humidities) / len(humidities), 1) if humidities else None,
            "wind_max": finite(daily["wind_speed_10m_max"][index]),
            "heatwave_indicator": {"status": "unavailable", "reason": INDICATOR_REASON, "official_declaration": False},
            "temperature_indicator": {
                "label": "High temperature screening only",
                "threshold_c": threshold,
                "exceeds_threshold": tmax >= threshold if tmax is not None else None,
                "note": "Temperature screening is separate from apparent temperature and is not an official heatwave declaration or ANN output.",
            },
        })
    return {
        **city, "timezone": TIMEZONE, "source": SOURCE, "mode": "forecast",
        "issued_at": now, "retrieved_at": now, "target_date": rows_daily[0]["date"],
        "model_version": "open-meteo-best-match", "availability": "available", "availability_reason": None,
        "grid": {"latitude": raw.get("latitude"), "longitude": raw.get("longitude"), "elevation_m": raw.get("elevation")},
        "daily": rows_daily, "hourly": rows_hourly,
        "ann": {"availability": "unavailable", "reason": ANN_REASON},
    }


def fetch_snapshot(*, session=requests, timeout: float = 25, now: datetime | None = None) -> dict:
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    params = {
        "latitude": ",".join(str(city["latitude"]) for city in CITIES),
        "longitude": ",".join(str(city["longitude"]) for city in CITIES),
        "timezone": TIMEZONE,
        "forecast_days": 7,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "models": "best_match",
        "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max,precipitation_probability_max,wind_speed_10m_max",
    }
    response = session.get(ENDPOINT, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) != len(CITIES):
        raise ValueError("Provider response must contain the six requested locations")
    result = metadata(timestamp)
    result["cities"] = []
    for city, raw in zip(CITIES, payload):
        try:
            result["cities"].append(convert_city(city, raw, timestamp))
        except (ValueError, KeyError, TypeError) as error:
            result["cities"].append(unavailable_city(city, f"Provider data could not be parsed: {type(error).__name__}."))
    available = [city for city in result["cities"] if city["availability"] == "available"]
    result["target_date"] = available[0]["target_date"] if available else None
    result["availability"] = "available" if available else "unavailable"
    if len(available) != len(CITIES):
        result["availability_reason"] = "Some city forecasts are unavailable; inspect each city's availability."
    result["regional_season_active"] = int(result["target_date"][5:7]) in (4, 5, 6) if result["target_date"] else False
    return result


def failed_snapshot(path: Path, error: Exception) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    reason = f"Weather refresh failed ({type(error).__name__}). Last successful timestamps are preserved."
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("cities"), list):
            raise ValueError("Invalid existing snapshot")
        data["availability"] = "stale" if any(city.get("daily") for city in data["cities"]) else "unavailable"
        for city in data["cities"]:
            if city.get("daily"):
                city["availability"] = "stale"
                city["availability_reason"] = reason
    except (OSError, ValueError):
        data = metadata(now)
        data.update(issued_at=None, retrieved_at=None, availability="unavailable")
        data["cities"] = [unavailable_city(city, reason) for city in CITIES]
    data["availability_reason"] = reason
    data["last_refresh_attempt"] = now
    return data


def write_snapshot(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "heatline" / "outlook.json")
    parser.add_argument("--frontend-output", type=Path, default=ROOT / "frontend" / "public" / "data" / "outlook.json")
    parser.add_argument("--no-frontend", action="store_true")
    parser.add_argument("--timeout", type=float, default=25)
    args = parser.parse_args()
    failed = False
    try:
        result = fetch_snapshot(timeout=args.timeout)
        failed = result["availability"] == "unavailable"
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        result = failed_snapshot(args.output, error)
        failed = True
    write_snapshot(args.output, result)
    if not args.no_frontend:
        write_snapshot(args.frontend_output, result)
    count = sum(city.get("availability") == "available" for city in result["cities"])
    print(f"Weather snapshot: {result['availability']}; {count}/6 cities available; issued_at={result.get('issued_at')}; output={args.output}")
    if failed:
        print(result.get("availability_reason", "Refresh failed."), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
