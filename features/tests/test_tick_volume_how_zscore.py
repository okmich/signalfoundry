import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.volume import tick_volume_how_zscore, tick_volume_phase_zscore


def test_requires_datetime_index():
    volume = pd.Series([100.0, 120.0, 140.0], index=[0, 1, 2])

    with pytest.raises(ValueError, match="DatetimeIndex"):
        tick_volume_how_zscore(volume, min_periods=2)


def test_min_periods_applies_within_minute_of_week_bucket():
    index = pd.date_range("2024-01-01 10:00", periods=5, freq="7D")
    volume = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0], index=index)

    result = tick_volume_how_zscore(volume, min_periods=3)

    assert result.name == "tick_vol_how_zscore"
    assert result.iloc[:2].isna().all()
    assert np.isfinite(result.iloc[2])
    assert result.iloc[2] == pytest.approx(1.0, rel=1e-9)


def test_min_periods_is_tracked_per_bucket():
    index = pd.to_datetime(
        [
            "2024-01-01 10:00",
            "2024-01-01 11:00",
            "2024-01-08 10:00",
            "2024-01-08 11:00",
        ]
    )
    volume = pd.Series([100.0, 1000.0, 120.0, 900.0], index=index)

    result = tick_volume_how_zscore(volume, min_periods=2)

    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert np.isfinite(result.iloc[2])
    assert np.isfinite(result.iloc[3])


def test_daily_phase_warmup_by_day():
    index = pd.date_range("2024-01-01 10:00", periods=5, freq="1D")
    volume = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0], index=index)

    result = tick_volume_phase_zscore(volume, min_periods=3, phase="daily")

    assert result.name == "tick_vol_phase_zscore"
    assert result.iloc[:2].isna().all()
    assert np.isfinite(result.iloc[2])
    assert result.iloc[2] == pytest.approx(1.0, rel=1e-9)


def test_daily_and_weekly_modes_differ_on_mixed_weekdays():
    index = pd.to_datetime(
        [
            "2024-01-01 10:00",
            "2024-01-02 10:00",
            "2024-01-03 10:00",
        ]
    )
    volume = pd.Series([100.0, 120.0, 140.0], index=index)

    daily = tick_volume_phase_zscore(volume, min_periods=2, phase="daily")
    weekly = tick_volume_phase_zscore(volume, min_periods=2, phase="weekly")

    assert np.isnan(daily.iloc[0])
    assert np.isfinite(daily.iloc[1])
    assert np.isnan(weekly.iloc[1])
    assert np.isnan(weekly.iloc[2])


def test_strict_phase_match_does_not_mix_5min_slots_within_hour():
    index = pd.to_datetime(
        [
            "2024-01-01 10:00",
            "2024-01-01 10:05",
            "2024-01-08 10:00",
            "2024-01-08 10:05",
        ]
    )
    volume = pd.Series([10.0, 1000.0, 20.0, 1200.0], index=index)

    result = tick_volume_how_zscore(volume, min_periods=2)

    assert result.iloc[2] == pytest.approx(0.7071067812, rel=1e-8)
    assert result.iloc[3] == pytest.approx(0.7071067812, rel=1e-8)


def test_how_wrapper_matches_weekly_phase_values():
    index = pd.to_datetime(
        [
            "2024-01-01 10:00",
            "2024-01-01 10:05",
            "2024-01-08 10:00",
            "2024-01-08 10:05",
        ]
    )
    volume = pd.Series([10.0, 1000.0, 20.0, 1200.0], index=index)

    wrapped = tick_volume_how_zscore(volume, min_periods=2)
    weekly = tick_volume_phase_zscore(volume, min_periods=2, phase="weekly")

    assert wrapped.name == "tick_vol_how_zscore"
    assert weekly.name == "tick_vol_phase_zscore"
    np.testing.assert_allclose(wrapped.to_numpy(), weekly.to_numpy(), equal_nan=True)


def test_requires_monotonic_datetime_index_for_causality():
    index = pd.to_datetime(["2024-01-08 10:00", "2024-01-01 10:00"])
    volume = pd.Series([100.0, 120.0], index=index)

    with pytest.raises(ValueError, match="monotonic"):
        tick_volume_how_zscore(volume, min_periods=2)


def test_requires_positive_min_periods():
    index = pd.date_range("2024-01-01 10:00", periods=3, freq="7D")
    volume = pd.Series([100.0, 110.0, 120.0], index=index)

    with pytest.raises(ValueError, match="min_periods"):
        tick_volume_how_zscore(volume, min_periods=0)


def test_requires_valid_phase_mode():
    index = pd.date_range("2024-01-01 10:00", periods=3, freq="7D")
    volume = pd.Series([100.0, 110.0, 120.0], index=index)

    with pytest.raises(ValueError, match="phase"):
        tick_volume_phase_zscore(volume, min_periods=2, phase="monthly")


def test_constant_bucket_volume_produces_finite_values():
    index = pd.date_range("2024-01-01 10:00", periods=6, freq="7D")
    volume = pd.Series(np.full(6, 100.0), index=index)

    result = tick_volume_how_zscore(volume, min_periods=3)

    valid = result.dropna()
    assert len(valid) == 4
    assert np.isfinite(valid.to_numpy()).all()
    assert np.allclose(valid.to_numpy(), 0.0)
