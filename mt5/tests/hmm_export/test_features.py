"""The feature definitions are a contract with HmmFeatures.mqh - pin them hard."""

import numpy as np
import pytest

from okmich_quant_mt5.hmm_export import FeatureSpec, Mql5Feature, build_features, ema, macd, true_range, wilder_atr

SPEC = FeatureSpec(names=(Mql5Feature.MACD_ATR, Mql5Feature.ATR_CLOSE), macd_fast=18, macd_slow=40, macd_signal=11,
                   atr_period=14)


@pytest.fixture
def bars():
    rng = np.random.default_rng(11)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0008, 900))
    spread = np.abs(rng.normal(0.0, 0.0004, 900)) + 0.0002
    return close + spread, close - spread, close


class TestEma:
    def test_sma_seed_and_recursion(self):
        x = np.arange(1.0, 11.0)
        out = ema(x, 3)

        assert np.all(np.isnan(out[:2])), "EMA must be undefined before the seed window closes"
        assert out[2] == pytest.approx(2.0)  # mean(1, 2, 3)
        assert out[3] == pytest.approx(3.0)  # 0.5*4 + 0.5*2
        assert out[4] == pytest.approx(4.0)
        assert out[5] == pytest.approx(5.0)

    def test_shorter_than_period_is_all_nan(self):
        assert np.all(np.isnan(ema(np.arange(3.0), 5)))

    def test_period_one_is_the_series(self):
        x = np.array([3.0, 1.0, 4.0, 1.0])
        np.testing.assert_allclose(ema(x, 1), x)


class TestTrueRange:
    def test_first_bar_undefined(self):
        high, low, close = np.array([2.0, 3.0]), np.array([1.0, 2.0]), np.array([1.5, 2.5])
        tr = true_range(high, low, close)
        assert np.isnan(tr[0]), "bar 0 has no previous close, so TR is undefined"

    def test_takes_the_widest_of_three(self):
        # Gap up: |high - prev_close| dominates the intrabar range.
        high = np.array([10.0, 20.0])
        low = np.array([9.0, 19.0])
        close = np.array([9.5, 19.5])
        assert true_range(high, low, close)[1] == pytest.approx(10.5)


class TestWilderAtr:
    def test_first_valid_index_is_the_period(self):
        rng = np.random.default_rng(3)
        close = 100 + np.cumsum(rng.normal(0, 1, 60))
        high, low = close + 1.0, close - 1.0

        atr = wilder_atr(high, low, close, 14)
        assert np.all(np.isnan(atr[:14])), "seed averages TR[1..14], so index 14 is the first output"
        assert np.isfinite(atr[14])

    def test_seed_is_the_mean_of_the_first_period_true_ranges(self):
        rng = np.random.default_rng(5)
        close = 100 + np.cumsum(rng.normal(0, 1, 40))
        high, low = close + 0.7, close - 0.7

        tr = true_range(high, low, close)
        atr = wilder_atr(high, low, close, 10)
        assert atr[10] == pytest.approx(tr[1:11].mean())

    def test_wilder_recursion(self):
        rng = np.random.default_rng(7)
        close = 50 + np.cumsum(rng.normal(0, 0.5, 40))
        high, low = close + 0.3, close - 0.2

        tr = true_range(high, low, close)
        atr = wilder_atr(high, low, close, 10)
        assert atr[11] == pytest.approx((atr[10] * 9 + tr[11]) / 10)

    def test_matches_talib(self):
        """TA-Lib is the convention the rest of the repo fits against."""
        talib = pytest.importorskip("talib")

        rng = np.random.default_rng(19)
        close = 100 + np.cumsum(rng.normal(0, 0.8, 500))
        high, low = close + np.abs(rng.normal(0, 0.4, 500)), close - np.abs(rng.normal(0, 0.4, 500))

        ours = wilder_atr(high, low, close, 14)
        theirs = talib.ATR(high, low, close, timeperiod=14)

        both = np.isfinite(ours) & np.isfinite(theirs)
        assert both.sum() > 400
        np.testing.assert_allclose(ours[both], theirs[both], rtol=1e-12, atol=1e-12)


class TestMacd:
    def test_valid_indices(self):
        rng = np.random.default_rng(13)
        close = 1.1 + np.cumsum(rng.normal(0, 0.001, 300))

        line, signal, hist = macd(close, 18, 40, 11)

        assert np.all(np.isnan(line[:39])) and np.isfinite(line[39]), "MACD line valid from slow-1"
        assert np.all(np.isnan(signal[:49])) and np.isfinite(signal[49]), "signal valid from slow+signal-2"
        assert np.all(np.isnan(hist[:49])) and np.isfinite(hist[49])

    def test_signal_is_sma_seeded_on_the_macd_subsequence(self):
        rng = np.random.default_rng(17)
        close = 1.1 + np.cumsum(rng.normal(0, 0.001, 200))

        line, signal, _ = macd(close, 18, 40, 11)
        assert signal[49] == pytest.approx(line[39:50].mean())

    def test_line_is_fast_minus_slow(self):
        rng = np.random.default_rng(23)
        close = 1.1 + np.cumsum(rng.normal(0, 0.001, 200))

        line, _, _ = macd(close, 18, 40, 11)
        expected = ema(close, 18) - ema(close, 40)
        np.testing.assert_allclose(line[40:], expected[40:], rtol=1e-14)


class TestBuildFeatures:
    def test_shape_and_warmup(self, bars):
        high, low, close = bars
        out = build_features(high, low, close, SPEC)

        assert out.shape == (900, 2)
        assert np.all(np.isnan(out[: SPEC.warmup_bars]))
        assert np.isfinite(out[SPEC.warmup_bars]).all(), "every column must be finite at the declared warm-up"

    def test_warmup_is_the_binding_constraint(self):
        assert SPEC.warmup_bars == 49  # max(40 + 11 - 2, 14)
        assert FeatureSpec(names=(Mql5Feature.ATR,), atr_period=200).warmup_bars == 200

    def test_column_order_follows_the_spec(self, bars):
        high, low, close = bars
        forward = build_features(high, low, close, SPEC)
        reversed_spec = FeatureSpec(names=SPEC.names[::-1], macd_fast=18, macd_slow=40, macd_signal=11, atr_period=14)
        backward = build_features(high, low, close, reversed_spec)

        np.testing.assert_allclose(forward[60:, 0], backward[60:, 1])
        np.testing.assert_allclose(forward[60:, 1], backward[60:, 0])

    def test_rows_are_masked_wholesale(self, bars):
        """A row is either entirely usable or entirely NaN - the MQL5 side skips whole bars."""
        high, low, close = bars
        out = build_features(high, low, close, SPEC)
        finite_per_row = np.isfinite(out).sum(axis=1)
        assert set(np.unique(finite_per_row)).issubset({0, out.shape[1]})

    def test_dimensionless_features_are_scale_free(self, bars):
        """The reason to prefer macd_atr / atr_close: a price rescale must not move them."""
        high, low, close = bars
        base = build_features(high, low, close, SPEC)
        scaled = build_features(high * 7.0, low * 7.0, close * 7.0, SPEC)

        # MACD is a difference of two near-equal EMAs, so on a ~1.1 price it loses ~10
        # digits to cancellation. That floor is why parity uses an absolute tolerance.
        np.testing.assert_allclose(base[60:], scaled[60:], rtol=1e-8)

    def test_price_scaled_features_are_not(self, bars):
        high, low, close = bars
        spec = FeatureSpec(names=(Mql5Feature.MACD, Mql5Feature.ATR), macd_fast=18, macd_slow=40, macd_signal=11,
                           atr_period=14)

        base = build_features(high, low, close, spec)
        scaled = build_features(high * 7.0, low * 7.0, close * 7.0, spec)
        assert not np.allclose(base[60:], scaled[60:])
        assert spec.price_scaled == (Mql5Feature.MACD, Mql5Feature.ATR)


class TestFeatureSpecValidation:
    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            FeatureSpec(names=())

    def test_rejects_duplicates(self):
        with pytest.raises(ValueError, match="duplicates"):
            FeatureSpec(names=(Mql5Feature.ATR, Mql5Feature.ATR))

    def test_rejects_slow_not_exceeding_fast(self):
        with pytest.raises(ValueError, match="must exceed"):
            FeatureSpec(names=(Mql5Feature.MACD,), macd_fast=40, macd_slow=18)

    def test_rejects_non_positive_period(self):
        with pytest.raises(ValueError, match="atr_period"):
            FeatureSpec(names=(Mql5Feature.ATR,), atr_period=0)

    def test_accepts_plain_strings(self):
        spec = FeatureSpec(names=("macd_atr", "atr_close"))
        assert spec.names == (Mql5Feature.MACD_ATR, Mql5Feature.ATR_CLOSE)
