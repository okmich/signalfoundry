"""Per-leg features: Tayal's structure (f1) and volume (f2) on hand-built legs, the fixed BOCPD break score,
degenerate bucketing, the symbol encoder, and causality."""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm.hierarchical import ZigzagDirection
from okmich_quant_ml.hmm.hierarchical.inputs import (BOCPDBreakScore, LegDirection, LegSizeScore, LegThreshold,
                                                     MarketData, QuantileLegFeature, SymbolEncoder, TayalTrend,
                                                     TayalVolume, VolumeRateUnit, Zigzag, ZigzagLegBuilder, bucketize,
                                                     leg_feature_frame, quantile_boundaries, tayal_structure,
                                                     tayal_volume_signal)

T0 = pd.Timestamp("2026-01-05", tz="UTC")


def _legs_from_pivots(pivots, bars_per_leg: int = 4) -> list[Zigzag]:
    legs = []
    for n in range(1, len(pivots)):
        up = pivots[n] > pivots[n - 1]
        s, e = (n - 1) * bars_per_leg, n * bars_per_leg
        legs.append(Zigzag(n - 1, ZigzagDirection.UP if up else ZigzagDirection.DOWN, s, e, float(pivots[n - 1]),
                           float(pivots[n]), abs(pivots[n] / pivots[n - 1] - 1), e + 1,
                           T0 + pd.Timedelta(minutes=5 * s), T0 + pd.Timedelta(minutes=5 * (e + 1))))
    return legs


def test_tayal_structure_reads_higher_highs_and_lows():
    up = _legs_from_pivots([1, 3, 2, 4, 3, 5, 4])
    assert list(tayal_structure(up)) == [0, 0, 0, 1, 1, 1]
    down = _legs_from_pivots([9, 7, 8, 6, 7, 5])
    assert list(tayal_structure(down)) == [0, 0, 0, -1, -1]
    mixed = _legs_from_pivots([1, 3, 2, 4, 1, 5])          # the last low breaks the higher-low run
    assert list(tayal_structure(mixed)) == [0, 0, 0, 0, 0]
    assert list(TayalTrend().transform(up, None)) == [1, 1, 1, 2, 2, 2]


def _volume_market(legs, leg_volumes) -> MarketData:
    """Volume only on each leg's interior bars (pivot bars, shared by neighbouring legs, carry none), scaled so
    phi(leg) per bar equals the given value exactly."""
    n_bars = legs[-1].confirm_bar + 2
    vol = np.zeros(n_bars)
    for z, v in zip(legs, leg_volumes):
        span = z.end_bar - z.start_bar + 1
        vol[z.start_bar + 1: z.end_bar] = v * span / (z.end_bar - z.start_bar - 1)
    idx = pd.date_range(T0, periods=n_bars, freq="5min")
    close = pd.Series(np.linspace(1, 2, n_bars), index=idx)
    return MarketData(close=close, high=close, low=close, volume=pd.Series(vol, index=idx))


@pytest.mark.parametrize("volumes, expected", [((10, 10, 20), 1), ((10, 10, 5), -1), ((10, 20, 40), 0),
                                               ((10, 10, 12.5), 0), ((10, 10, 12.6), 1)])
def test_tayal_volume_groups_the_three_ratios(volumes, expected):
    legs = _legs_from_pivots([1, 2, 1.5, 2.5])
    md = _volume_market(legs, volumes)
    signal = tayal_volume_signal(legs, md, alpha=0.25, unit=VolumeRateUnit.BAR)
    assert list(signal[:2]) == [0, 0]
    assert signal[2] == expected


def test_tayal_volume_per_second_uses_the_clock_span():
    legs = _legs_from_pivots([1, 2, 1.5, 2.5])
    md = _volume_market(legs, (10, 10, 20))
    assert tayal_volume_signal(legs, md, unit=VolumeRateUnit.SECOND)[2] == 1
    with pytest.raises(ValueError, match="volume"):
        TayalVolume().transform(legs, MarketData(close=md.close))


def test_degenerate_scores_go_to_the_middle_bin():
    noisy_constant = 0.004 + np.random.default_rng(0).normal(0, 1e-18, 200)
    bounds = quantile_boundaries(noisy_constant, 3)
    assert set(bucketize(noisy_constant, bounds)) == {1}
    assert np.all(np.diff(bounds) > 0)
    assert set(bucketize(np.zeros(5), quantile_boundaries(np.array([]), 3))) == {1}


def _market(n: int = 8000, seed: int = 1) -> tuple[pd.Series, MarketData]:
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(n) < n // 2, 0.0006, 0.002)          # a volatility break half-way
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol))), index=idx)
    volume = pd.Series(rng.poisson(50, n).astype(float), index=idx)
    return close, MarketData(close=close, high=close * 1.0002, low=close * 0.9998, volume=volume)


def test_bocpd_break_score_is_informative_not_constant():
    close, md = _market()
    cp = BOCPDBreakScore().changepoint_prob(close)
    assert cp.shape == (len(close),) and np.all((cp >= 0) & (cp <= 1))
    n = len(close) // 2
    assert cp[n: n + 30].max() > 10 * np.median(cp)                 # the break registers
    legs = ZigzagLegBuilder(LegThreshold.legs_per_hour(1.0)).fit_transform(close)
    scores = BOCPDBreakScore().scores(legs, md)
    assert np.ptp(scores) > 1e-3
    flows = bucketize(scores, quantile_boundaries(scores, 3))
    assert np.bincount(flows, minlength=3).min() > 0.2 * len(legs)   # all three buckets used


def test_symbol_encoder_matches_the_legacy_layout_and_tayal_alphabets():
    enc = SymbolEncoder([LegDirection(), TayalTrend(), TayalVolume()])
    assert enc.n_symbols == 18
    for d in (0, 1):
        for s in range(3):
            for f in range(3):
                assert enc.encode([d, s, f]) == d * 9 + s * 3 + f            # the legacy 18-symbol layout
                assert enc.decode(enc.encode([d, s, f])) == (d, s, f)
    np.testing.assert_array_equal(enc.symbols_with("direction", 1), np.arange(9, 18))
    six = SymbolEncoder([LegDirection(), TayalTrend()])
    assert six.n_symbols == 6 and six.symbol_names()[5] == "up/up_trend"
    with pytest.raises(ValueError, match="unique"):
        SymbolEncoder([LegDirection(), LegDirection()])


def test_quantile_feature_freezes_boundaries():
    close, md = _market()
    builder = ZigzagLegBuilder(LegThreshold.legs_per_hour(1.0)).fit(close.iloc[:4000])
    legs = builder.transform(close)
    feat = QuantileLegFeature(LegSizeScore(), 3)
    with pytest.raises(RuntimeError):
        feat.transform(legs, md)
    feat.fit([z for z in legs if z.confirm_bar < 4000], md)
    frozen = feat.boundaries_.copy()
    feat.transform(legs, md)
    np.testing.assert_array_equal(feat.boundaries_, frozen)


def test_leg_features_are_prefix_invariant():
    close, md = _market(6000, seed=4)
    builder = ZigzagLegBuilder(LegThreshold.legs_per_hour(1.0)).fit(close)
    full = builder.transform(close)
    full_frame = leg_feature_frame(full, md)
    full_cp = BOCPDBreakScore().scores(full, md)
    cut = 4500
    part_md = MarketData(close=close.iloc[:cut + 1], high=md.high.iloc[:cut + 1], low=md.low.iloc[:cut + 1],
                         volume=md.volume.iloc[:cut + 1])
    part = builder.transform(part_md.close)
    pd.testing.assert_frame_equal(leg_feature_frame(part, part_md), full_frame.iloc[:len(part)])
    np.testing.assert_allclose(BOCPDBreakScore().scores(part, part_md), full_cp[:len(part)], atol=1e-12)


def test_leg_feature_frame_columns():
    close, md = _market(3000)
    legs = ZigzagLegBuilder(LegThreshold.legs_per_hour(1.0)).fit_transform(close)
    frame = leg_feature_frame(legs, md)
    assert list(frame.columns) == ["direction", "signed_size_sigma", "log_duration_bars", "confirm_lag_bars",
                                   "structure", "log_volume_rate", "volume_signal"]
    assert np.all(np.sign(frame["signed_size_sigma"]) == frame["direction"])
    assert "log_volume_rate" not in leg_feature_frame(legs, MarketData(close=close)).columns
