"""Validity and intersection checks for the limited temperature-source audit."""
import numpy as np
import pandas as pd
import pytest

from scripts.audit_station_sources import build, clean_daily, pair_sources, temperature_metrics


def test_provider_missing_values_and_invalid_dates_are_excluded():
    frame = pd.DataFrame({'date': ['1991-05-01', '1991-05-02', '1991-05-03',
                                   '1991-05-04', '1991-05-05', 'bad', '1991-05-07'],
                           'value': [113, 9999.9, -999, np.nan, np.inf, 100, 'missing']})
    cleaned = clean_daily(frame, 'date', 'value', 'station', fahrenheit=True)
    assert cleaned.date.tolist() == ['1991-05-01']
    assert cleaned.station.tolist() == [45]


def test_pairing_uses_common_valid_may_dates_only():
    station = pd.DataFrame({'date': ['1991-04-30', '1991-05-01', '1991-05-02'],
                            'station': [40., 45., 46.]})
    era5 = pd.DataFrame({'date': ['1991-04-30', '1991-05-01', '1991-05-03'],
                         'era5': [39., 44., 47.]})
    nasa = pd.DataFrame({'date': ['1991-04-30', '1991-05-01', '1991-05-02'],
                         'nasa': [41., 47., 48.]})
    pairs = pair_sources(station, era5, nasa, 1991)
    assert pairs.date.tolist() == ['1991-05-01']
    assert temperature_metrics(pairs, 'era5')['mae_c'] == 1
    assert temperature_metrics(pairs, 'nasa')['bias_c'] == 2


def test_duplicate_daily_values_fail_instead_of_multiplying_pairs():
    frame = pd.DataFrame({'date': ['1991-05-01'] * 2, 'value': [40., 41.]})
    with pytest.raises(ValueError, match='Duplicate'):
        clean_daily(frame, 'date', 'value', 'era5')


def test_audit_rejects_retrospective_years():
    with pytest.raises(ValueError, match='training-era'):
        build(years=(2026,))


def test_actual_cached_audit_is_paired_and_descriptive():
    result = build()
    assert result['pooled']['paired_days'] == 92
    assert result['years'] == [1991, 2000, 2010]
    assert len({row['date'] for row in result['records']}) == 92
    assert result['station']['id'] == '42867099999'
    assert result['pooled']['sources']['era5']['mae_c'] == pytest.approx(.956, abs=.0005)
    assert result['pooled']['sources']['nasa']['mae_c'] == pytest.approx(1.826, abs=.0005)
    assert result['daily_boundaries']['nasa'] == 'UTC calendar day'
    assert any('not official' in note for note in result['limitations'])
