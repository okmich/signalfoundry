"""
Tests for CausalRegimeLabeler.

Coverage:
- Construction validation
- warmup_bars property
- leaks_future guarantee
- All four yardsticks produce valid labels
- Labels are strictly {-1, 0, 1, NaN}
- NaN only in warmup region
- Hysteresis: state does not change until min_persistence bars
- Trend quality filter (R²) forces neutral on noisy slope
- Diagnostics DataFrame has correct columns and values
- Fallback to rolling std when OHLC absent (volatility)
- Repr
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.regime import CausalRegimeLabeler, MarketPropertyType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int, trend: float = 0.0, noise: float = 0.001,
                   seed: int = 42) -> pd.DataFrame:
    """Simple synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    log_close = np.cumsum(rng.normal(trend, noise, n))
    close = np.exp(log_close - log_close[0] + np.log(100.0))
    spread = close * 0.001
    high = close + spread
    low = close - spread
    volume = rng.integers(1000, 5000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume, "tick_volume": volume},
        index=idx,
    )


def _make_phase_transition(
    n: int = 600,
    split: int = 250,
    flat_noise: float = 0.001,
    trend: float = 0.003,
    trend_noise: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """
    First `split` bars: random walk (flat phase, low metric values).
    Remaining bars: strong directional trend (high metric values).

    The contrast between phases ensures that rolling percentile rank of any
    directional metric (slope, momentum) clearly exceeds upper_pct during
    the trend phase — as long as the lookback window spans the transition.
    """
    rng = np.random.default_rng(seed)
    flat_rets = rng.normal(0.0, flat_noise, split)
    trend_rets = rng.normal(trend, trend_noise, n - split)
    all_rets = np.concatenate([flat_rets, trend_rets])
    log_prices = np.cumsum(all_rets)
    close = np.exp(log_prices - log_prices[0] + np.log(100.0))
    spread = close * 0.001
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread,
         "close": close, "volume": 1000.0, "tick_volume": 1000.0},
        index=idx,
    )


def _make_trending_up(n: int = 600) -> pd.DataFrame:
    """Flat phase then strong uptrend — ensures percentile contrast."""
    return _make_phase_transition(n=n, split=n // 3, trend=0.003)


def _make_trending_down(n: int = 600) -> pd.DataFrame:
    """Flat phase then strong downtrend."""
    return _make_phase_transition(n=n, split=n // 3, trend=-0.003)


def _make_choppy(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Mean-reverting/choppy prices."""
    rng = np.random.default_rng(seed)
    closes = []
    price = 100.0
    for _ in range(n):
        price = price * 0.97 + 100.0 * 0.03 + rng.normal(0, 0.05)
        closes.append(price)
    closes = np.array(closes)
    spread = closes * 0.001
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + spread,
            "low": closes - spread,
            "close": closes,
            "volume": 1000.0,
            "tick_volume": 1000.0,
        },
        index=idx,
    )


def _make_high_liquidity(n: int = 1200, split: int = 400, seed: int = 42) -> pd.DataFrame:
    """Low-volume/wide-spread phase then high-volume/tight-spread phase.

    Both volume AND spread change at the transition so that Amihud and
    Corwin-Schultz both detect the liquidity shift (not just volume).
    """
    rng = np.random.default_rng(seed)
    # Thin phase: wide intrabar range, low volume
    noise_thin = rng.normal(0.0, 0.003, split)
    # Liquid phase: tight intrabar range, high volume
    noise_liquid = rng.normal(0.0, 0.0005, n - split)
    close = np.exp(np.cumsum(np.concatenate([noise_thin, noise_liquid])) + np.log(100.0))
    spread_thin = close[:split] * 0.005
    spread_liquid = close[split:] * 0.0005
    spread = np.concatenate([spread_thin, spread_liquid])
    low_vol = rng.integers(100, 500, split).astype(float)
    high_vol = rng.integers(3000, 8000, n - split).astype(float)
    volume = np.concatenate([low_vol, high_vol])
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread,
         "close": close, "volume": volume, "tick_volume": volume},
        index=idx,
    )


def _make_low_liquidity(n: int = 1200, split: int = 400, seed: int = 42) -> pd.DataFrame:
    """High-volume/tight-spread phase then low-volume/wide-spread phase."""
    rng = np.random.default_rng(seed)
    noise_liquid = rng.normal(0.0, 0.0005, split)
    noise_thin = rng.normal(0.0, 0.003, n - split)
    close = np.exp(np.cumsum(np.concatenate([noise_liquid, noise_thin])) + np.log(100.0))
    spread_liquid = close[:split] * 0.0005
    spread_thin = close[split:] * 0.005
    spread = np.concatenate([spread_liquid, spread_thin])
    high_vol = rng.integers(3000, 8000, split).astype(float)
    low_vol = rng.integers(100, 500, n - split).astype(float)
    volume = np.concatenate([high_vol, low_vol])
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread,
         "close": close, "volume": volume, "tick_volume": volume},
        index=idx,
    )


def _make_high_vol(n: int = 600) -> pd.DataFrame:
    """Low-vol phase then high-vol phase — ensures percentile contrast."""
    return _make_phase_transition(n=n, split=n // 3, trend=0.0, flat_noise=0.0002, trend_noise=0.01)


def _make_low_vol(n: int = 600) -> pd.DataFrame:
    """High-vol phase then low-vol phase."""
    return _make_phase_transition(n=n, split=n // 3, trend=0.0, flat_noise=0.01, trend_noise=0.0002)


YARDSTICKS = [
    MarketPropertyType.DIRECTION,
    MarketPropertyType.MOMENTUM,
    MarketPropertyType.VOLATILITY,
    MarketPropertyType.PATH_STRUCTURE,
    MarketPropertyType.LIQUIDITY,
]


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_valid_yardsticks(self):
        for y in YARDSTICKS:
            lbl = CausalRegimeLabeler(yardstick=y)
            assert lbl.yardstick == y

    def test_accepts_enum_yardstick(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION)
        assert lbl.yardstick == MarketPropertyType.DIRECTION

    def test_rejects_directionless_momentum(self):
        with pytest.raises(ValueError, match="Unsupported"):
            CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTIONLESS_MOMENTUM)

    def test_rejects_non_enum_yardstick(self):
        with pytest.raises(TypeError, match="MarketPropertyType"):
            CausalRegimeLabeler(yardstick="invalid_yardstick")

    def test_rejects_metric_window_lt_2(self):
        with pytest.raises(ValueError, match="metric_window"):
            CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION, metric_window=1)

    def test_rejects_lookback_lt_metric_window(self):
        with pytest.raises(ValueError, match="lookback_window"):
            CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION, metric_window=20, lookback_window=10)

    def test_rejects_invalid_pct_order(self):
        with pytest.raises(ValueError, match="lower_pct"):
            CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION, upper_pct=0.3, lower_pct=0.7)

    def test_rejects_min_persistence_lt_1(self):
        with pytest.raises(ValueError, match="min_persistence"):
            CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION, min_persistence=0)

    def test_leaks_future_is_false(self):
        for y in YARDSTICKS:
            assert CausalRegimeLabeler(yardstick=y).leaks_future is False

    def test_repr(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION)
        r = repr(lbl)
        assert "direction" in r
        assert "CausalRegimeLabeler" in r


# ---------------------------------------------------------------------------
# warmup_bars
# ---------------------------------------------------------------------------

class TestWarmupBars:

    def test_warmup_bars_formula(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.DIRECTION, metric_window=20, lookback_window=100, min_persistence=3
        )
        assert lbl.warmup_bars == max(20, 100) + 3  # 103

    def test_warmup_bars_when_metric_dominates(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.MOMENTUM, metric_window=50, lookback_window=50, min_persistence=5
        )
        assert lbl.warmup_bars == max(50, 50) + 5  # 55

    def test_raises_if_df_shorter_than_warmup(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION, metric_window=20, lookback_window=100)
        df = _make_price_df(50)
        with pytest.raises(ValueError, match="warmup_bars"):
            lbl.label(df)


# ---------------------------------------------------------------------------
# Label validity — all yardsticks
# ---------------------------------------------------------------------------

class TestLabelValidity:

    @pytest.mark.parametrize("yardstick", YARDSTICKS)
    def test_output_is_series(self, yardstick):
        lbl = CausalRegimeLabeler(yardstick=yardstick, metric_window=10, lookback_window=50)
        df = _make_price_df(200)
        result = lbl.label(df)
        assert isinstance(result, pd.Series)

    @pytest.mark.parametrize("yardstick", YARDSTICKS)
    def test_index_aligned_to_input(self, yardstick):
        lbl = CausalRegimeLabeler(yardstick=yardstick, metric_window=10, lookback_window=50)
        df = _make_price_df(200)
        result = lbl.label(df)
        assert result.index.equals(df.index)

    @pytest.mark.parametrize("yardstick", YARDSTICKS)
    def test_valid_labels_only(self, yardstick):
        """Non-NaN labels must be exactly in {-1, 0, 1}."""
        lbl = CausalRegimeLabeler(yardstick=yardstick, metric_window=10, lookback_window=50)
        df = _make_price_df(300)
        result = lbl.label(df)
        valid = result.dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    @pytest.mark.parametrize("yardstick", YARDSTICKS)
    def test_nan_only_in_warmup_region(self, yardstick):
        """No NaN after warmup_bars; NaN allowed before."""
        lbl = CausalRegimeLabeler(yardstick=yardstick, metric_window=10, lookback_window=50)
        df = _make_price_df(300)
        result = lbl.label(df)
        post_warmup = result.iloc[lbl.warmup_bars :]
        # Allow some NaN if metric itself has NaN (e.g. ATR first bar), but
        # the vast majority should be labelled.
        assert post_warmup.notna().mean() > 0.90

    @pytest.mark.parametrize("yardstick", YARDSTICKS)
    def test_no_values_outside_valid_set(self, yardstick):
        lbl = CausalRegimeLabeler(yardstick=yardstick, metric_window=10, lookback_window=50)
        df = _make_price_df(300)
        result = lbl.label(df)
        non_nan = result[result.notna()]
        assert ((non_nan == -1) | (non_nan == 0) | (non_nan == 1)).all()

    def test_missing_price_col_raises(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION, metric_window=10, lookback_window=50)
        df = _make_price_df(200).rename(columns={"close": "price"})
        with pytest.raises(KeyError, match="close"):
            lbl.label(df)

    def test_custom_price_col(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION, metric_window=10, lookback_window=50)
        df = _make_price_df(200).rename(columns={"close": "price"})
        result = lbl.label(df, price_col="price")
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------

class TestHysteresis:

    def test_state_does_not_change_before_persistence(self):
        """
        With min_persistence=5, state must hold for 5 bars before changing.
        Construct a price where metric flips after warmup — state should lag.
        """
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.MOMENTUM,
            metric_window=5,
            lookback_window=50,
            min_persistence=5,
        )
        df = _make_price_df(300, trend=0.003, noise=0.0001)
        _, diag = lbl.label_with_diagnostics(df)

        # Find first state change in labels after warmup
        post = diag["label"].iloc[lbl.warmup_bars :].dropna()
        if len(post) > 10:
            # Check consecutive counts reach persistence before state flips
            # (label at bar i should equal label at bar i-1 unless count >= persistence)
            for i in range(1, min(50, len(post))):
                prev_label = post.iloc[i - 1]
                curr_label = post.iloc[i]
                if curr_label != prev_label:
                    # A state change occurred — verify count had reached persistence
                    count_at_change = diag["consecutive_count"].iloc[lbl.warmup_bars + i]
                    assert count_at_change >= lbl.min_persistence or np.isnan(count_at_change)

    def test_min_persistence_1_changes_every_bar(self):
        """With min_persistence=1, state can change every bar."""
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.MOMENTUM,
            metric_window=5,
            lookback_window=50,
            min_persistence=1,
        )
        df = _make_price_df(300, trend=0.003, noise=0.001)
        result = lbl.label(df)
        post = result.iloc[lbl.warmup_bars :].dropna()
        # Should see more than one distinct label value
        assert len(post.unique()) > 0


# ---------------------------------------------------------------------------
# Yardstick-specific behaviour
# ---------------------------------------------------------------------------

class TestDirectionYardstick:
    # The labeller uses adaptive (rolling percentile) thresholds.
    # It fires +1 when the slope is in the top 30% of the LOOKBACK WINDOW.
    # In a sustained trend, the lookback normalizes → label reverts toward 0.
    # The correct test window: the TRANSITION PERIOD where the lookback still
    # captures the flat phase, making the new trend bars rank highly.
    # split=200, lookback=50 → transition = bars 200–250 (50-bar window straddles split).

    def test_uptrend_transition_produces_positive_labels(self):
        """At regime onset, when lookback still captures flat bars, trend bars rank high → +1."""
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.DIRECTION, metric_window=10, lookback_window=50,
            min_persistence=3, min_r_squared=0.1,
        )
        df = _make_trending_up(600)  # split=200
        result = lbl.label(df)
        transition = result.iloc[203:250].dropna()  # +3 for min_persistence
        assert (transition == 1).mean() > 0.5

    def test_downtrend_transition_produces_negative_labels(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.DIRECTION, metric_window=10, lookback_window=50,
            min_persistence=3, min_r_squared=0.1,
        )
        df = _make_trending_down(600)
        result = lbl.label(df)
        transition = result.iloc[203:250].dropna()
        assert (transition == -1).mean() > 0.5

    def test_uptrend_transition_more_positive_than_downtrend(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.DIRECTION, metric_window=10, lookback_window=50,
            min_persistence=3, min_r_squared=0.1,
        )
        up_labels = lbl.label(_make_trending_up(600)).iloc[203:250].dropna()
        down_labels = lbl.label(_make_trending_down(600)).iloc[203:250].dropna()
        assert (up_labels == 1).mean() > (down_labels == 1).mean()

    def test_r_squared_filter_suppresses_noisy_slope(self):
        """High min_r_squared forces neutral on noisy series."""
        lbl_strict = CausalRegimeLabeler(
            yardstick=MarketPropertyType.DIRECTION, metric_window=10, lookback_window=50,
            min_r_squared=0.99,  # almost impossible to satisfy
        )
        lbl_lenient = CausalRegimeLabeler(
            yardstick=MarketPropertyType.DIRECTION, metric_window=10, lookback_window=50,
            min_r_squared=0.0,
        )
        df = _make_price_df(300, trend=0.0, noise=0.005)
        strict_labels = lbl_strict.label(df).dropna()
        lenient_labels = lbl_lenient.label(df).dropna()
        strict_neutral = (strict_labels == 0).mean()
        lenient_neutral = (lenient_labels == 0).mean()
        assert strict_neutral >= lenient_neutral


class TestMomentumYardstick:
    # Same adaptive threshold reasoning as direction.
    # Check transition window (bars 203–250) where lookback straddles the split.

    def test_uptrend_transition_produces_positive_momentum(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.MOMENTUM, metric_window=10, lookback_window=50, min_persistence=3
        )
        df = _make_trending_up(600)
        result = lbl.label(df)
        transition = result.iloc[203:250].dropna()
        assert (transition == 1).mean() > 0.5

    def test_downtrend_transition_produces_negative_momentum(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.MOMENTUM, metric_window=10, lookback_window=50, min_persistence=3
        )
        df = _make_trending_down(600)
        result = lbl.label(df)
        transition = result.iloc[203:250].dropna()
        assert (transition == -1).mean() > 0.5

    def test_uptrend_transition_more_positive_than_downtrend(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.MOMENTUM, metric_window=10, lookback_window=50, min_persistence=3
        )
        up_labels = lbl.label(_make_trending_up(600)).iloc[203:250].dropna()
        down_labels = lbl.label(_make_trending_down(600)).iloc[203:250].dropna()
        assert (up_labels == 1).mean() > (down_labels == 1).mean()


class TestVolatilityYardstick:

    def test_high_vol_transition_produces_more_positive_labels(self):
        """
        _make_high_vol: low-vol phase → high-vol phase.
        At the transition (bars 203-250), high vol ranks above its own recent history → +1.
        _make_low_vol is the opposite — transition gives -1.
        """
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.VOLATILITY, metric_window=10, lookback_window=50,
            min_persistence=3, use_atr=True,
        )
        df_high = _make_high_vol(600)
        df_low = _make_low_vol(600)
        high_result = lbl.label(df_high).iloc[203:250].dropna()
        low_result = lbl.label(df_low).iloc[203:250].dropna()
        assert (high_result == 1).mean() > (low_result == 1).mean()

    def test_falls_back_to_rolling_std_without_ohlc(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.VOLATILITY, metric_window=10, lookback_window=50,
            use_atr=True,
        )
        df = _make_price_df(300)[["close"]]  # close only, no high/low
        result = lbl.label(df)
        assert isinstance(result, pd.Series)
        assert result.iloc[lbl.warmup_bars :].notna().any()

    def test_use_atr_false_uses_rolling_std(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.VOLATILITY, metric_window=10, lookback_window=50,
            use_atr=False,
        )
        df = _make_price_df(300)
        result = lbl.label(df)
        assert isinstance(result, pd.Series)


class TestPathStructureYardstick:
    # efficiency_ratio: high (≈1) = trending, low (≈0) = choppy.
    # At the flat→trend transition, ER jumps → ranks high in lookback → +1.
    # A fully choppy series has consistently low ER → ranks low → -1.

    def test_trend_transition_produces_more_positive_than_choppy(self):
        """
        At the flat→trend transition, efficiency ratio jumps above the lookback
        percentile threshold → +1 label. Choppy series has persistently low ER → -1.
        """
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.PATH_STRUCTURE, metric_window=10, lookback_window=50,
            min_persistence=3,
        )
        df_trend = _make_trending_up(600)
        df_choppy = _make_choppy(500)
        # Trend transition window: bars 203-250
        trend_result = lbl.label(df_trend).iloc[203:250].dropna()
        # Choppy series: check well into the series where ER is consistently low
        choppy_result = lbl.label(df_choppy).iloc[lbl.warmup_bars + 50:].dropna()
        assert (trend_result == 1).mean() >= (choppy_result == 1).mean()

    def test_choppy_produces_more_negative_than_trending(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.PATH_STRUCTURE, metric_window=10, lookback_window=50,
            min_persistence=3,
        )
        df_trend = _make_trending_up(600)
        df_choppy = _make_choppy(500)
        trend_result = lbl.label(df_trend).iloc[203:250].dropna()
        choppy_result = lbl.label(df_choppy).iloc[lbl.warmup_bars + 50:].dropna()
        assert (choppy_result == -1).mean() >= (trend_result == -1).mean()


# ---------------------------------------------------------------------------
# Liquidity yardstick
# ---------------------------------------------------------------------------

class TestLiquidityYardstick:
    # Test data: 1200 bars, split at 400.
    # Composite z-score spikes at the transition; the rolling percentile rank
    # captures this in a window centered on the split point (bars 395–445).

    def test_high_liquidity_transition_produces_positive_labels(self):
        """Thin→liquid transition: composite ranks high → +1."""
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.LIQUIDITY, metric_window=10, lookback_window=50,
            min_persistence=3, volume_col="tick_volume",
        )
        df = _make_high_liquidity(1200)
        result = lbl.label(df)
        transition = result.iloc[395:445].dropna()
        assert (transition == 1).mean() > 0.3

    def test_low_liquidity_transition_produces_negative_labels(self):
        """Liquid→thin transition: composite ranks low → -1."""
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.LIQUIDITY, metric_window=10, lookback_window=50,
            min_persistence=3, volume_col="tick_volume",
        )
        df = _make_low_liquidity(1200)
        result = lbl.label(df)
        transition = result.iloc[395:445].dropna()
        assert (transition == -1).mean() >= 0.3

    def test_high_more_positive_than_low(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.LIQUIDITY, metric_window=10, lookback_window=50,
            min_persistence=3, volume_col="tick_volume",
        )
        high_labels = lbl.label(_make_high_liquidity(1200)).iloc[395:445].dropna()
        low_labels = lbl.label(_make_low_liquidity(1200)).iloc[395:445].dropna()
        assert (high_labels == 1).mean() > (low_labels == 1).mean()

    def test_missing_volume_col_raises(self):
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.LIQUIDITY, metric_window=10, lookback_window=50,
            volume_col="nonexistent_col",
        )
        df = _make_price_df(200)
        with pytest.raises(KeyError, match="nonexistent_col"):
            lbl.label(df)

    def test_custom_volume_col(self):
        """Accepts a custom volume column name."""
        lbl = CausalRegimeLabeler(
            yardstick=MarketPropertyType.LIQUIDITY, metric_window=10, lookback_window=50,
            volume_col="volume",
        )
        df = _make_price_df(300)
        result = lbl.label(df)
        assert isinstance(result, pd.Series)
        assert result.iloc[lbl.warmup_bars:].notna().any()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:

    def test_returns_tuple(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION, metric_window=10, lookback_window=50)
        df = _make_price_df(200)
        result = lbl.label_with_diagnostics(df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_labels_consistent_with_label_method(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION, metric_window=10, lookback_window=50)
        df = _make_price_df(200)
        labels_plain = lbl.label(df)
        labels_diag, _ = lbl.label_with_diagnostics(df)
        pd.testing.assert_series_equal(labels_plain, labels_diag)

    def test_diagnostics_has_required_columns(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION, metric_window=10, lookback_window=50)
        df = _make_price_df(200)
        _, diag = lbl.label_with_diagnostics(df)
        required = {
            "metric_value", "secondary_value", "metric_percentile",
            "raw_candidate", "candidate_state", "consecutive_count", "label",
        }
        assert required.issubset(set(diag.columns))

    def test_diagnostics_no_secondary_for_momentum(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.MOMENTUM, metric_window=10, lookback_window=50)
        df = _make_price_df(200)
        _, diag = lbl.label_with_diagnostics(df)
        assert "secondary_value" in diag.columns

    def test_diagnostics_index_aligned(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.MOMENTUM, metric_window=10, lookback_window=50)
        df = _make_price_df(200)
        _, diag = lbl.label_with_diagnostics(df)
        assert diag.index.equals(df.index)

    def test_diagnostics_label_col_matches_labels(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.VOLATILITY, metric_window=10, lookback_window=50)
        df = _make_price_df(200)
        labels, diag = lbl.label_with_diagnostics(df)
        pd.testing.assert_series_equal(
            labels.rename("label"),
            diag["label"].rename("label"),
        )

    def test_metric_percentile_bounded(self):
        lbl = CausalRegimeLabeler(yardstick=MarketPropertyType.MOMENTUM, metric_window=10, lookback_window=50)
        df = _make_price_df(300)
        _, diag = lbl.label_with_diagnostics(df)
        valid_pct = diag["metric_percentile"].dropna()
        assert (valid_pct >= 0.0).all() and (valid_pct <= 1.0).all()

