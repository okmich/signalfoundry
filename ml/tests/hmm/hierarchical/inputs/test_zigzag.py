"""Zigzag legs: configurable leg size, a threshold frozen at fit time, causality, calibration warnings, alignment."""
import warnings

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm.hierarchical.inputs import (LegThreshold, ThresholdMode, ZigzagLegBuilder, aggregate_zigzags,
                                                     calibrate_k, events_per_hour, legs_confirmed_from, legs_to_bars,
                                                     realized_vol)


def _prices(n: int = 6000, seed: int = 3, vol: float = 0.0008, start: str = "2026-01-05") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, vol, n))), index=idx, name="close")


def _same_legs(a, b) -> bool:
    key = lambda legs: [(z.start_bar, z.end_bar, z.confirm_bar, z.direction) for z in legs]
    return key(a) == key(b)


def test_leg_threshold_validation_and_band():
    assert LegThreshold.legs_per_hour(1.0).band == pytest.approx((0.8, 1.2))
    assert LegThreshold.legs_per_hour((0.5, 2.0)).band == (0.5, 2.0)
    with pytest.raises(ValueError):
        LegThreshold.legs_per_hour((2.0, 1.0))
    with pytest.raises(ValueError):
        LegThreshold.k_sigma(-1.0)
    with pytest.raises(ValueError):
        LegThreshold(ThresholdMode.FIXED, (0.1, 0.2))
    with pytest.raises(TypeError):
        ZigzagLegBuilder("1/hour")
    with pytest.raises(RuntimeError, match="fit"):
        ZigzagLegBuilder(LegThreshold.fixed(0.001)).transform(_prices(100))


@pytest.mark.parametrize("threshold", [LegThreshold.fixed(0.002), LegThreshold.k_sigma(4.0),
                                       LegThreshold.legs_per_hour(1.0)], ids=["fixed", "k_sigma", "rate"])
def test_builder_freezes_theta_and_uses_it(threshold):
    train = _prices()
    builder = ZigzagLegBuilder(threshold).fit(train)
    if threshold.mode is ThresholdMode.TARGET_RATE:
        lo, hi = threshold.band
        assert lo <= builder.rate_ <= hi
    if threshold.mode is ThresholdMode.FIXED:
        assert builder.theta_ == 0.002
    loud = _prices(seed=9, vol=0.003, start="2026-02-05")              # a far more volatile later block
    assert _same_legs(builder.transform(loud), aggregate_zigzags(loud, builder.theta_))


def test_legs_are_prefix_invariant():
    prices = _prices(8000, seed=5)
    builder = ZigzagLegBuilder(LegThreshold.legs_per_hour(1.0)).fit(prices.iloc[:4000])
    full = builder.transform(prices)
    for cut in (3000, 5500, 7999):
        part = builder.transform(prices.iloc[:cut + 1])
        assert _same_legs(part, [z for z in full if z.confirm_bar <= cut])


def test_calibration_warns_when_the_target_is_unreachable():
    prices = _prices(3000)
    sigma = realized_vol(prices, pd.Timedelta("24h"))
    with pytest.warns(RuntimeWarning, match="not reachable"):
        k = calibrate_k(prices, sigma, (500.0, 600.0))
    assert k == pytest.approx(0.1)
    with pytest.warns(RuntimeWarning, match="not reachable"):
        calibrate_k(prices, sigma, (1e-5, 2e-5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        k = calibrate_k(prices, sigma, (0.8, 1.2))
    rate = events_per_hour(aggregate_zigzags(prices, k * float(np.nanmedian(sigma))))
    assert 0.8 <= rate <= 1.2


def test_legs_to_bars_and_confirmed_from():
    prices = _prices(2000)
    legs = ZigzagLegBuilder(LegThreshold.legs_per_hour(1.0)).fit_transform(prices)
    values = np.arange(len(legs), dtype=float)
    bars = legs_to_bars(values, legs, len(prices))
    assert np.isnan(bars[:legs[0].confirm_bar]).all()
    for k, z in enumerate(legs[:-1]):
        assert (bars[z.confirm_bar:legs[k + 1].confirm_bar] == k).all()
    delayed = legs_to_bars(values, legs, len(prices), delay=1)
    assert np.isnan(delayed[legs[0].confirm_bar]) and delayed[legs[0].confirm_bar + 1] == 0
    later = legs_confirmed_from(legs, prices.index[1000])
    assert all(z.end_time >= prices.index[1000] for z in later)
    assert len(later) == sum(z.confirm_bar >= 1000 for z in legs)
    with pytest.raises(ValueError):
        legs_to_bars(values[:-1], legs, len(prices))
