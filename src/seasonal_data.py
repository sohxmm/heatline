"""Source-specific seasonal data; providers are never silently mixed."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests

from src.climatology import compute_reference_normals
from src.daily import aggregate_year_hourly_to_daily
from src.features import compute_features, FEATURE_COLS
from src.labels import heatwave_proxy
from src.fetch_data import HOURLY_VARIABLES

ROOT = Path(__file__).resolve().parents[1]


def fetch_season(year, provider='era5'):
    folder = ROOT / 'data/raw' / ('seasonal_era5' if provider == 'era5' else 'nasa_power')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{year}.json'
    if path.exists():
        return path
    if provider == 'era5':
        endpoint = 'https://archive-api.open-meteo.com/v1/archive'
        params = dict(latitude=21.1458, longitude=79.0882, start_date=f'{year}-03-25',
            end_date=f'{year}-06-30', hourly=','.join(HOURLY_VARIABLES), timezone='Asia/Kolkata', models='era5')
    else:
        endpoint = 'https://power.larc.nasa.gov/api/temporal/daily/point'
        params = dict(parameters='T2M_MAX,T2M_MIN,T2M,RH2M,WS2M,PS,ALLSKY_SFC_SW_DWN,PRECTOTCORR,QV2M',
            community='AG', longitude=79.0882, latitude=21.1458, start=f'{year}0325', end=f'{year}0630',
            format='JSON', **{'time-standard':'UTC'})
    for attempt in range(4):
        try:
            response = requests.get(endpoint, params=params, timeout=70)
            response.raise_for_status()
            payload = response.json()
            payload['_heatline_provenance'] = dict(provider=provider, request_parameters=params,
                retrieved_at=datetime.now(timezone.utc).isoformat(), source_url=response.url)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload), encoding='utf-8')
            temporary.replace(path)
            return path
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def fetch_all(provider='era5', start=1981, end=2026):
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch_season, year, provider):year for year in range(start,end+1)}
        for future in as_completed(futures):
            path = future.result()
            print(f'{provider} {futures[future]} cached: {path.name}', flush=True)


def prepare_era5():
    files = [ROOT/f'data/raw/seasonal_era5/{year}.json' for year in range(1981,2027)]
    daily = pd.concat([aggregate_year_hourly_to_daily(path) for path in files],ignore_index=True)
    if len(daily) != 46 * 98:
        raise ValueError('Expected 98 complete local days per season, March 25–June 30')
    normals = compute_reference_normals(daily)
    # Freeze existing April/May target definitions for an apples-to-apples v1
    # comparison; June/March reference values use the newly acquired training data.
    frozen = pd.read_csv(ROOT/'data/processed/climatology_normals.csv')
    normals = pd.concat([normals[~normals.month.isin([4,5])], frozen], ignore_index=True).sort_values(['month','day'])
    daily = daily.merge(normals[['month','day','normal_tmax','p85_tmax','p90_tmax']],on=['month','day'],validate='many_to_one')
    daily['tmax_anomaly'] = (daily.tmax-daily.normal_tmax).round(2)
    daily['heatwave_day'] = heatwave_proxy(daily.tmax,daily.normal_tmax)
    daily['imd_strict_raw'] = daily.heatwave_day
    daily['imd_reanalysis_plains'] = ((daily.tmax>=43.5)|((daily.tmax>=40)&(daily.tmax_anomaly>=3.5))).astype(int)
    daily['climatology_p85_42c'] = (daily.tmax>=np.maximum(42,daily.p85_tmax)).astype(int)
    features = compute_features(daily,normals,target_months=(4,5,6))
    old = pd.read_csv(ROOT/'data/processed/features.csv')
    may = features[features.month==5].reset_index(drop=True)
    if not np.array_equal(may.heatwave_day,old.heatwave_day):
        raise ValueError('Expanded source changed May proxy targets; do not silently compare models')
    np.testing.assert_allclose(may[FEATURE_COLS],old[FEATURE_COLS],atol=1e-8)
    daily.to_csv(ROOT/'data/processed/seasonal_daily.csv',index=False)
    features.to_csv(ROOT/'data/processed/seasonal_features.csv',index=False)
    return daily, features, normals


def source_audit(daily):
    """Audit the original cache and a source sensitivity sample, not model selection."""
    old = pd.read_csv(ROOT/'data/processed/daily.csv')
    joined = old[['date','tmax','heatwave_day']].merge(daily[['date','tmax','heatwave_day']],on='date',suffixes=('_v1','_v2'),validate='one_to_one')
    result = dict(era5_overlap_days=len(joined),max_daily_tmax_difference=float((joined.tmax_v1-joined.tmax_v2).abs().max()),
        may_label_changes=int((joined[joined.date.str[5:7]=='05'].heatwave_day_v1 != joined[joined.date.str[5:7]=='05'].heatwave_day_v2).sum()),
        source_decision='Keep ERA5 for assessed modelling and unchanged May targets. NASA POWER is a source sensitivity audit only, not a way to increase event prevalence.',
        nasa_time_standard='UTC daily boundaries differ from ERA5 Asia/Kolkata. Provider comparison is descriptive and not a matched station validation.',
        nasa_samples=[], files=[])
    for year in [1991,2000,2010]:
        path=ROOT/f'data/raw/nasa_power/{year}.json'
        if not path.exists():continue
        data=json.loads(path.read_text())
        params=data['properties']['parameter']
        nasa=pd.DataFrame({name:values for name,values in params.items()})
        nasa.index=pd.to_datetime(nasa.index,format='%Y%m%d').strftime('%Y-%m-%d')
        subset=daily.set_index('date').join(nasa,how='inner')
        valid=subset[(subset.T2M_MAX>-900)&subset.month.eq(5)]
        result['nasa_samples'].append(dict(year=year,may_days=len(valid),nasa_mean_tmax=float(valid.T2M_MAX.mean()),era5_mean_tmax=float(valid.tmax.mean()),
            mean_nasa_minus_era5=float((valid.T2M_MAX-valid.tmax).mean()),nasa_days_ge45=int((valid.T2M_MAX>=45).sum()),era5_days_ge45=int((valid.tmax>=45).sum())))
    result['station_samples']=[]
    for path in sorted((ROOT/'data/raw/noaa_gsod').glob('*.csv')):
        station=pd.read_csv(path)
        station['date']=pd.to_datetime(station.DATE).dt.strftime('%Y-%m-%d')
        station=station[(station.MAX<9999)&pd.to_datetime(station.DATE).dt.month.eq(5)].copy()
        station['station_tmax']=(station.MAX-32)*5/9
        paired=station[['date','station_tmax']].merge(daily[['date','tmax']],on='date',validate='one_to_one')
        result['station_samples'].append(dict(year=int(path.name[:4]),station='42867099999 · Dr Baba Saheb Ambedkar International Airport',paired_days=len(paired),
            mean_station_tmax=float(paired.station_tmax.mean()),mean_era5_tmax=float(paired.tmax.mean()),mean_era5_minus_station=float((paired.tmax-paired.station_tmax).mean()),
            mean_absolute_difference=float((paired.tmax-paired.station_tmax).abs().mean()),station_days_ge45=int((paired.station_tmax>=45).sum()),era5_days_ge45=int((paired.tmax>=45).sum())))
    result['station_limitation']='NOAA GSOD maximum temperatures summarize synoptic reports; reporting windows can overlap the previous day and differ from local calendar days. Training-era descriptive audit only; not an official warning-label validation.'
    for directory in ['seasonal_era5','nasa_power','noaa_gsod']:
        for path in sorted((ROOT/'data/raw'/directory).glob('*')):
            result['files'].append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--provider',choices=['era5','nasa'],default='era5');p.add_argument('--start',type=int,default=1981);p.add_argument('--end',type=int,default=2026)
    a=p.parse_args();fetch_all(a.provider,a.start,a.end)
