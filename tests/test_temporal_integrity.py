"""Pipeline invariants beyond a same-day perturbation with frozen normals."""
import numpy as np
import pandas as pd
import pytest
from src.climatology import compute_reference_normals
from src.features import compute_features, FEATURE_COLS
from src.labels import heatwave_proxy, select_primary_label


def test_reference_ignores_future_years():
    daily = pd.read_csv('data/processed/daily.csv')
    expected = compute_reference_normals(daily)
    daily.loc[daily.year > 2010, ['tmax', 'tmin', 'apparent_tmax', 'rh_mean']] += 100
    pd.testing.assert_frame_equal(expected, compute_reference_normals(daily))


def test_declared_label_boundaries():
    assert heatwave_proxy([39.99, 40, 44.99, 45, 42, 42], [30,35.5,44,44,37.51,37.5]).tolist() == [0,1,0,1,0,1]
    assert select_primary_label({'train': {}})[0] == 'imd_strict_raw'
    with pytest.raises(ValueError):
        heatwave_proxy([np.nan], [40])


@pytest.mark.parametrize('problem', ['gap', 'duplicate', 'no_april'])
def test_incomplete_history_fails(problem):
    daily = pd.read_csv('data/processed/daily.csv')
    daily = daily[daily.year == 2020]
    if problem == 'gap':
        daily = daily[daily.date != '2020-04-30']
    elif problem == 'duplicate':
        daily = pd.concat([daily, daily.iloc[[0]]])
    else:
        daily = daily[daily.month == 5]
    with pytest.raises(ValueError):
        compute_features(daily)


def test_future_weather_does_not_change_earlier_inputs():
    daily = pd.read_csv('data/processed/daily.csv')
    daily = daily[daily.year == 2020].copy()
    original = compute_features(daily)
    weather = ['tmax', 'tmin', 'tmean', 'heatwave_day']
    daily.loc[daily.date >= '2020-05-15', weather] += 50
    changed = compute_features(daily)
    earlier = original.date <= '2020-05-15'
    np.testing.assert_array_equal(original.loc[earlier, FEATURE_COLS], changed.loc[earlier, FEATURE_COLS])
