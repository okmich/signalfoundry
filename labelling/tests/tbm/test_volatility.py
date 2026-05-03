"""Tests for labelling.tbm.volatility."""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.tbm.volatility import (
    get_atr_vol,
    get_daily_vol,
    get_garman_klass_vol,
    get_parkinson_vol,
    get_std_vol,
)


class TestGetDailyVol:
    def test_matches_batch_ewm_log_ret_std(self, gbm_close):
        vol = get_daily_vol(gbm_close, span=50)
        log_ret = np.log(gbm_close / gbm_close.shift(1))
        expected = log_ret.ewm(span=50, adjust=False).std()
        pd.testing.assert_series_equal(vol, expected, check_names=False)

    def test_leading_nans_preserved(self, gbm_close):
        vol = get_daily_vol(gbm_close, span=20)
        assert np.isnan(vol.iloc[0])
        assert vol.index.equals(gbm_close.index)

    def test_invalid_span(self, gbm_close):
        with pytest.raises(ValueError):
            get_daily_vol(gbm_close, span=1)

    def test_annualize(self, gbm_close):
        v = get_daily_vol(gbm_close, span=50, annualize=False)
        v_ann = get_daily_vol(gbm_close, span=50, annualize=True)
        ratio = (v_ann / v).dropna()
        np.testing.assert_allclose(ratio.values, np.sqrt(252.0), rtol=1e-6)

    def test_annualize_with_custom_factor(self, gbm_close):
        v = get_daily_vol(gbm_close, span=50, annualize=False)
        v_ann = get_daily_vol(gbm_close, span=50, annualize=True, annualization_factor=365.0)
        ratio = (v_ann / v).dropna()
        np.testing.assert_allclose(ratio.values, np.sqrt(365.0), rtol=1e-6)

    def test_rejects_non_positive_close(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1h")
        bad = pd.Series([100.0, 101.0, 0.0, 102.0, 103.0], index=idx)
        with pytest.raises(ValueError, match="strictly positive"):
            get_daily_vol(bad, span=3)

    def test_rejects_non_finite_close(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1h")
        bad = pd.Series([100.0, 101.0, np.inf, 102.0, 103.0], index=idx)
        with pytest.raises(ValueError, match="non-finite"):
            get_daily_vol(bad, span=3)


class TestAlternativeEstimators:
    def test_atr_returns_aligned_series(self, gbm_ohlc):
        v = get_atr_vol(gbm_ohlc, window=14)
        assert isinstance(v, pd.Series)
        assert v.index.equals(gbm_ohlc.index)

    def test_parkinson_returns_aligned_series(self, gbm_ohlc):
        v = get_parkinson_vol(gbm_ohlc, window=20)
        assert v.index.equals(gbm_ohlc.index)

    def test_garman_klass_returns_aligned_series(self, gbm_ohlc):
        v = get_garman_klass_vol(gbm_ohlc, window=20)
        assert v.index.equals(gbm_ohlc.index)

    def test_std_returns_aligned_series(self, gbm_ohlc):
        v = get_std_vol(gbm_ohlc["close"], window=20)
        assert v.index.equals(gbm_ohlc.index)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError):
            get_atr_vol(df)


class TestVolKindTagging:
    def test_get_daily_vol_tagged_return(self, gbm_close):
        v = get_daily_vol(gbm_close, span=50, annualize=False)
        assert v.attrs["vol_kind"] == "return"

    def test_get_daily_vol_annualize_tagged_annualized(self, gbm_close):
        v = get_daily_vol(gbm_close, span=50, annualize=True)
        assert v.attrs["vol_kind"] == "annualized"

    def test_get_atr_vol_tagged_price(self, gbm_ohlc):
        v = get_atr_vol(gbm_ohlc, window=14)
        assert v.attrs["vol_kind"] == "price"

    def test_get_parkinson_vol_tagged_annualized(self, gbm_ohlc):
        v = get_parkinson_vol(gbm_ohlc, window=20)
        assert v.attrs["vol_kind"] == "annualized"

    def test_get_garman_klass_vol_tagged_annualized(self, gbm_ohlc):
        v = get_garman_klass_vol(gbm_ohlc, window=20)
        assert v.attrs["vol_kind"] == "annualized"

    def test_get_std_vol_tagged_price(self, gbm_ohlc):
        v = get_std_vol(gbm_ohlc["close"], window=20)
        assert v.attrs["vol_kind"] == "price"
