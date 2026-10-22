"""Frozen 1981–2010 training-reference normals (not official station normals)."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def compute_reference_normals(df: pd.DataFrame, start=1981, end=2010, window_days=7):
    """Calendar-day ±7 day reference using training years only.

    April/May edges have truncated windows because June/March were not collected.
    Mapping to a non-leap reference year avoids leap-day offsets.
    """
    if end > 2010 or start > end:
        raise ValueError('Reference normals must stay inside the training period ending 2010')
    baseline = df[df.year.between(start, end)].copy()
    if baseline.empty:
        raise ValueError('No observations in reference period')
    calendar = pd.to_datetime(dict(year=np.full(len(baseline), 2001), month=baseline.month.to_numpy(), day=baseline.day.to_numpy()))
    doy = calendar.dt.dayofyear.to_numpy()
    records = []
    for month, day in baseline[['month', 'day']].drop_duplicates().sort_values(['month', 'day']).itertuples(index=False, name=None):
        target = pd.Timestamp(2001, int(month), int(day)).dayofyear
        sample = baseline.loc[np.abs(doy - target) <= window_days]
        records.append(dict(month=int(month), day=int(day), day_of_year=target,
            sample_count=len(sample), normal_tmax=round(sample.tmax.mean(), 2),
            std_tmax=round(sample.tmax.std(ddof=1), 2),
            p85_tmax=round(sample.tmax.quantile(.85), 2), p90_tmax=round(sample.tmax.quantile(.90), 2),
            normal_tmin=round(sample.tmin.mean(), 2), normal_apparent_tmax=round(sample.apparent_tmax.mean(), 2),
            normal_rh_mean=round(sample.rh_mean.mean(), 2)))
    return pd.DataFrame(records)


def compute_wmo_climatology(config: dict):
    """Compatibility entry point; these are training-reference gridded normals."""
    cfg = config['climatology']
    normals = compute_reference_normals(pd.read_csv(config['data']['daily_file']),
        cfg['baseline_start'], cfg['baseline_end'], cfg['window_days'])
    destination = Path(config['data']['processed_dir']) / 'climatology_normals.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    normals.to_csv(destination, index=False)
    return normals


def load_climatology_lookup(config: dict):
    path = Path(config['data']['processed_dir']) / 'climatology_normals.csv'
    if not path.exists():
        compute_wmo_climatology(config)
    return {(int(r['month']), int(r['day'])): r for r in pd.read_csv(path).to_dict('records')}


if __name__ == '__main__':
    compute_wmo_climatology(yaml.safe_load(Path('config.yaml').read_text(encoding='utf-8')))
