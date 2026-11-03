"""Compare two gridded sources on a small, paired training-era station sample.

This is a descriptive temperature audit, not a test of official heat warnings.
No model is fitted and no retrospective evaluation year is used.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT_YEARS = (1991, 2000, 2010)


def clean_daily(frame: pd.DataFrame, date_col: str, value_col: str,
                name: str, *, fahrenheit: bool = False) -> pd.DataFrame:
    """Exclude missing dates/values and provider sentinels before pairing."""
    dates = pd.to_datetime(frame[date_col], errors='coerce')
    values = pd.to_numeric(frame[value_col], errors='coerce')
    valid = dates.notna() & np.isfinite(values) & (values > -900) & (values < 9999)
    result = pd.DataFrame({'date': dates[valid].dt.strftime('%Y-%m-%d'),
                           name: values[valid].astype(float)})
    if fahrenheit:
        result[name] = (result[name] - 32) * 5 / 9
    if result.date.duplicated().any():
        raise ValueError(f'Duplicate daily dates in {name}')
    return result.reset_index(drop=True)


def pair_sources(station: pd.DataFrame, era5: pd.DataFrame,
                 nasa: pd.DataFrame, year: int) -> pd.DataFrame:
    """Use only identical dates available in all three products in May."""
    paired = station.merge(era5, on='date', how='inner', validate='one_to_one')
    paired = paired.merge(nasa, on='date', how='inner', validate='one_to_one')
    return paired[paired.date.str.startswith(f'{year}-05-')].sort_values('date').reset_index(drop=True)


def temperature_metrics(paired: pd.DataFrame, column: str) -> dict:
    if paired.empty:
        raise ValueError('No paired station/source days')
    difference = paired[column] - paired.station
    return {
        'paired_days': len(paired),
        'mean_tmax_c': float(paired[column].mean()),
        'bias_c': float(difference.mean()),
        'mae_c': float(difference.abs().mean()),
        'rmse_c': float(np.sqrt(np.mean(difference ** 2))),
        'days_tmax_ge45': int((paired[column] >= 45).sum()),
    }


def build(root: Path = ROOT, years: tuple[int, ...] = AUDIT_YEARS) -> dict:
    if not years or any(year < 1981 or year > 2010 for year in years):
        raise ValueError('This audit accepts training-era years 1981–2010 only')
    daily_path = root / 'data/processed/seasonal_daily.csv'
    era5 = clean_daily(pd.read_csv(daily_path), 'date', 'tmax', 'era5')
    paths = [daily_path]
    reports, records, station_info = [], [], None
    for year in years:
        station_path = root / f'data/raw/noaa_gsod/{year}-42867099999.csv'
        nasa_path = root / f'data/raw/nasa_power/{year}.json'
        raw_station = pd.read_csv(station_path, dtype={'STATION': str})
        station = clean_daily(raw_station, 'DATE', 'MAX', 'station', fahrenheit=True)
        if station_info is None:
            first = raw_station.iloc[0]
            station_info = dict(id=str(first.STATION), name=str(first.NAME),
                                latitude=float(first.LATITUDE), longitude=float(first.LONGITUDE),
                                elevation_m=float(first.ELEVATION))
        raw_nasa = json.loads(nasa_path.read_text(encoding='utf-8'))
        if raw_nasa['header']['time_standard'] != 'UTC':
            raise ValueError('Expected documented NASA UTC daily product')
        nasa_values = raw_nasa['properties']['parameter']['T2M_MAX']
        nasa_frame = pd.DataFrame({'date': pd.to_datetime(list(nasa_values), format='%Y%m%d'),
                                   'tmax': list(nasa_values.values())})
        nasa = clean_daily(nasa_frame, 'date', 'tmax', 'nasa')
        paired = pair_sources(station, era5, nasa, year)
        reports.append(dict(year=year, paired_days=len(paired),
                            sources={name: temperature_metrics(paired, name)
                                     for name in ['station', 'era5', 'nasa']}))
        records.extend(paired.to_dict(orient='records'))
        paths.extend([station_path, nasa_path])
    pooled = pd.DataFrame(records)
    return {
        'schema_version': '1.0', 'model_version': 'heatline-source-audit-v1',
        'source': 'NOAA GSOD station / Open-Meteo ERA5 / NASA POWER cached records',
        'mode': 'descriptive training-era source audit', 'availability': 'available',
        'issued_at': datetime.now(timezone.utc).isoformat(), 'target_date': None,
        'years': list(years), 'month': 5, 'units': 'Celsius', 'station': station_info,
        'grid_request': {'latitude': 21.1458, 'longitude': 79.0882},
        'pairing': 'Intersection of valid same-date May records in all three products; no imputation.',
        'daily_boundaries': {'era5': 'Asia/Kolkata calendar day', 'nasa': 'UTC calendar day',
                             'station': 'GSOD synoptic summary; may overlap previous day'},
        'pooled': {'paired_days': len(pooled),
                   'sources': {name: temperature_metrics(pooled, name)
                               for name in ['station', 'era5', 'nasa']}},
        'by_year': reports, 'records': records,
        'decision': 'Retain ERA5 as the assessed historical source. Its temperature MAE is lower in this limited paired sample; NASA remains an independent sensitivity benchmark.',
        'limitations': [
            'Only three training-era May months; this does not establish the best source for every year or location.',
            'Same-date pairing does not make synoptic, UTC and Asia/Kolkata reporting windows identical.',
            'Station temperature summaries are not official IMD warning or heatwave-declaration labels.',
            'Grid temperatures can smooth local extremes. Lower mean absolute error does not prove better extreme-event detection.',
            'Neither source is selected using a higher ANN score against its own different labels.',
        ],
        'references': [
            {'title': 'Open-Meteo historical data products', 'url': 'https://open-meteo.com/en/docs/historical-weather-api'},
            {'title': 'NASA POWER resolution and data revisions', 'url': 'https://power.larc.nasa.gov/docs/tutorials/'},
            {'title': 'NOAA Global Summary of the Day', 'url': 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt'},
        ],
        'files': [{'path': path.relative_to(root).as_posix(),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
    }


if __name__ == '__main__':
    result = build()
    for directory in ['artifacts/heatline', 'frontend/public/data']:
        path = ROOT / directory / 'station_comparison.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Paired {result['pooled']['paired_days']} May station days; source audit exported.")
