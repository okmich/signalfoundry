"""Regressions for the 2026-11-24 review: aligned market data, leg/bar alignment, and the event-rate definition."""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm.hierarchical import ZigzagDirection
from okmich_quant_ml.hmm.hierarchical.inputs import (LegSizeScore, LegThreshold, MarketData, Zigzag,
                                                     ZigzagLegBuilder, events_per_hour, observed_hours, volume_rate)


def _close(n: int = 3000, seed: int = 0, start: str = "2026-01-05") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n))), index=idx)


def test_market_data_rejects_series_not_indexed_like_close():
    close = _close(100)
    late_volume = pd.Series(np.ones(100), index=close.index + pd.Timedelta("1D"))
    with pytest.raises(ValueError, match="volume is not indexed like close"):
        MarketData(close=close, volume=late_volume)
    with pytest.raises(ValueError, match="strictly increasing"):
        MarketData(close=close.iloc[::-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        MarketData(close=pd.concat([close, close.iloc[-1:]]))
    MarketData(close=close, volume=pd.Series(np.ones(100), index=close.index))        # aligned: fine


def test_features_reject_legs_built_on_another_series():
    close = _close()
    legs = ZigzagLegBuilder(LegThreshold.legs_per_hour(1.0)).fit_transform(close)
    shifted = close.copy()
    shifted.index = shifted.index + pd.Timedelta("1D")
    md_other = MarketData(close=shifted, volume=pd.Series(np.ones(len(close)), index=shifted.index))
    with pytest.raises(ValueError, match="do not match the market data's timestamps"):
        volume_rate(legs, md_other)
    with pytest.raises(ValueError, match="do not match"):
        LegSizeScore().scores(legs, md_other)
    with pytest.raises(ValueError, match="market data has"):
        LegSizeScore().scores(legs, MarketData(close=close.iloc[:100]))


def test_legs_must_not_read_past_their_confirmation():
    close = _close(100)
    md = MarketData(close=close, volume=pd.Series(np.ones(100), index=close.index))
    t = close.index
    bad = Zigzag(0, ZigzagDirection.UP, 0, 60, 1.0, 1.1, 0.1, 2, t[0], t[2])     # extremum after confirmation
    with pytest.raises(ValueError, match="start_bar <= end_bar <= confirm_bar"):
        volume_rate([bad], md)


def test_empty_and_one_bar_feeds_have_no_legs():
    builder = ZigzagLegBuilder(LegThreshold.fixed(0.001)).fit(_close(500))
    assert builder.transform(_close(500).iloc[:0]) == []
    assert builder.transform(_close(500).iloc[:1]) == []


def test_a_rejected_builder_fit_keeps_the_previous_threshold():
    builder = ZigzagLegBuilder(LegThreshold.k_sigma(2.0)).fit(_close())
    theta = builder.theta_
    flat = pd.Series(100.0, index=_close(50).index)                         # zero volatility: no threshold
    with pytest.raises(ValueError):
        builder.fit(flat)
    assert builder.theta_ == theta


def _leg(k: int, t: pd.Timestamp) -> Zigzag:
    return Zigzag(k, ZigzagDirection.UP, 0, 1, 1.0, 1.1, 0.1, 1, t, t)


def test_event_rate_definitions():
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    legs = [_leg(0, t0), _leg(1, t0 + pd.Timedelta("1h"))]
    assert events_per_hour(legs) == pytest.approx(1.0)                     # interarrival: (n - 1) / span
    assert events_per_hour(legs, exposure_hours=4.0) == pytest.approx(0.5)  # over a defined window
    assert events_per_hour(legs, exposure_hours=0.0) == 0.0
    assert events_per_hour(legs[:1]) == 0.0


def test_observed_hours_ignore_gaps():
    friday = pd.date_range("2026-01-09 20:00", periods=12, freq="5min", tz="UTC")
    monday = pd.date_range("2026-01-12 00:00", periods=12, freq="5min", tz="UTC")
    assert observed_hours(friday) == pytest.approx(1.0)
    assert observed_hours(friday.append(monday)) == pytest.approx(2.0)     # the weekend adds nothing


def test_calibrated_rate_is_per_observed_hour():
    """A series with a long gap: the rate counts legs per hour of bars, not per calendar hour spanned."""
    a, b = _close(3000, seed=1), _close(3000, seed=2, start="2026-03-02")
    b = b * (a.iloc[-1] / b.iloc[0])
    prices = pd.concat([a, b])
    builder = ZigzagLegBuilder(LegThreshold.legs_per_hour(1.0)).fit(prices)
    legs = builder.transform(prices)
    assert 0.8 <= builder.rate_ <= 1.2
    assert builder.rate_ == pytest.approx(len(legs) / observed_hours(prices.index))
    calendar = (prices.index[-1] - prices.index[0]) / pd.Timedelta("1h")
    assert len(legs) / calendar < 0.6                                        # the old definition would be diluted
